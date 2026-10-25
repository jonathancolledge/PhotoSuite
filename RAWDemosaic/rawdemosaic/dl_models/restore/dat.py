# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""DAT (Dual Aggregation Transformer, zhengchen1999/DAT), reproduced from the
official basicsr/archs/dat_arch.py -- verified tensor-shape-for-tensor-shape
against both 4x-UltraSharpV2.pth and 4xRealWebPhoto_v4_dat2.pth (see
restore/registry.py), and cross-checked against the official repo's
options/Test/test_DAT_2_x4.yml, which confirms both checkpoints are the
stock "DAT-2" config (embed_dim=180, depth=[6]*6, num_heads=[6]*6,
split_size=[8,32], expansion_factor=2, resi_connection='1conv', upscale=4)
unchanged -- community fine-tunes of this architecture apparently always
start from the official training config rather than inventing their own.

Every hyperparameter below is still derived generically from the
checkpoint's own tensor shapes rather than hardcoded to that one config,
following this project's usual "auto-detect, don't hardcode" approach --
including split_size, which isn't given directly by any weight shape but
*is* recoverable from the DynamicPosBias buffers' shapes (see
_infer_split_size), the same way Swin2SR's window_size was recovered from
relative_coords_table.

Two deliberate simplifications relative to the reference source, neither
of which changes forward-pass output:
  * No timm/einops dependency: DropPath is unconditionally identity (this
    project always runs eval-mode inference, where DropPath is a no-op
    identity anyway), trunc_normal_-based init is skipped entirely (every
    call here immediately does a strict load_state_dict, which overwrites
    whatever init produced), and einops' Rearrange/rearrange calls are
    replaced with the equivalent plain view/transpose/permute.
  * attn_mask_0/attn_mask_1 buffers (cached shift-window masks, valid only
    when the input is exactly `img_size` x `img_size`, i.e. the training
    patch size -- never true for a real photo) are not registered at all;
    the mask is instead always computed dynamically in forward(), which is
    exactly what the reference code itself falls back to whenever the
    input resolution doesn't match training resolution (see the
    `self.patches_resolution != _H or ...` branch in the reference
    Adaptive_Spatial_Attention.forward()). build_dat() strips these keys
    out of the incoming state_dict before the strict load since this
    module never expects them.

DAT alternates two block types within each of its Residual Groups, by
block-index parity (even -> DSTB, odd -> DCTB):
  * Adaptive_Spatial_Attention (DSTB): the input's channels are split in
    half, and each half is run through a differently-oriented rectangular
    windowed attention (Spatial_Attention, with orientations (H_sp, W_sp)
    and (W_sp, H_sp) so the two halves together cover both window shapes).
    Alternating blocks additionally shift the windows (exact condition
    reproduced from the reference: rg_idx even -> only b_idx==2 shifts;
    rg_idx odd -> b_idx in {0, 4} shift), verified against these two
    checkpoints' own attn_mask key placement (9 shifted blocks each).
  * Adaptive_Channel_Attention (DCTB): XCiT-style channel attention
    (normalize Q/K, attend over channels rather than tokens).
Both block types also run a parallel depthwise-conv branch and cross-gate
it against the attention output via an "Adaptive Interaction Module"
(channel_interaction / spatial_interaction, each an
AdaptiveAvgPool2d/Conv-BN-GELU-Conv squeeze-excite-style gate feeding a
sigmoid). Like Restormer/NAFNet/MPRNet/DeblurGANv2's globally-pooled
modules, this per-tile global pooling means DAT needs registry.py's
_NO_TILE (whole-image, no tiling) treatment to avoid visible per-tile
seams.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def img2windows(img: torch.Tensor, H_sp: int, W_sp: int) -> torch.Tensor:
    """(B, C, H, W) -> (B', H_sp*W_sp, C)"""
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    return img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)


def windows2img(img_splits_hw: torch.Tensor, H_sp: int, W_sp: int, H: int, W: int) -> torch.Tensor:
    """(B', H_sp*W_sp, C) -> (B, H, W, C)"""
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    return img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class SpatialGate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W))
        x2 = x2.flatten(2).transpose(-1, -2).contiguous()
        return x1 * x2


class SGFN(nn.Module):
    """Spatial-Gate Feed-Forward Network."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.sg = SpatialGate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, in_features)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.sg(x, H, W)
        return self.fc2(x)


class DynamicPosBias(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.pos_dim))
        self.pos2 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.pos_dim))
        self.pos3 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, num_heads))

    def forward(self, biases: torch.Tensor) -> torch.Tensor:
        return self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))


class Spatial_Attention(nn.Module):
    """Rectangular-window self-attention with dynamic relative position bias."""

    def __init__(self, dim: int, idx: int, split_size: list[int], num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx

        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        if idx == 0:
            H_sp, W_sp = split_size[0], split_size[1]
        else:
            W_sp, H_sp = split_size[0], split_size[1]
        self.H_sp = H_sp
        self.W_sp = W_sp

        self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
        position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
        biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w], indexing="ij"))
        biases = biases.flatten(1).transpose(0, 1).contiguous().float()
        self.register_buffer("rpe_biases", biases)

        coords_h = torch.arange(self.H_sp)
        coords_w = torch.arange(self.W_sp)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.H_sp - 1
        relative_coords[:, :, 1] += self.W_sp - 1
        relative_coords[:, :, 0] *= 2 * self.W_sp - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

    def im2win(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        return x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

    def forward(self, qkv: torch.Tensor, H: int, W: int, mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = qkv[0], qkv[1], qkv[2]
        B, L, C = q.shape

        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        pos = self.pos(self.rpe_biases)
        relative_position_bias = pos[self.relative_position_index.view(-1)].view(
            self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)
        return windows2img(x, self.H_sp, self.W_sp, H, W)


def _aim_modules(dim: int) -> tuple[nn.Sequential, nn.Sequential, nn.Sequential]:
    """The dwconv / channel_interaction / spatial_interaction triple shared
    identically by Adaptive_Spatial_Attention and Adaptive_Channel_Attention."""
    dwconv = nn.Sequential(
        nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
        nn.BatchNorm2d(dim),
        nn.GELU(),
    )
    channel_interaction = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim, dim // 8, kernel_size=1),
        nn.BatchNorm2d(dim // 8),
        nn.GELU(),
        nn.Conv2d(dim // 8, dim, kernel_size=1),
    )
    spatial_interaction = nn.Sequential(
        nn.Conv2d(dim, dim // 16, kernel_size=1),
        nn.BatchNorm2d(dim // 16),
        nn.GELU(),
        nn.Conv2d(dim // 16, 1, kernel_size=1),
    )
    return dwconv, channel_interaction, spatial_interaction


class Adaptive_Spatial_Attention(nn.Module):
    """DSTB attention: dual-orientation windowed attention (optionally
    shifted) blended with a parallel depthwise-conv branch via AIM gating."""

    def __init__(self, dim: int, num_heads: int, split_size: list[int], shift_size: list[int], rg_idx: int, b_idx: int) -> None:
        super().__init__()
        self.dim = dim
        self.split_size = split_size
        self.shift_size = shift_size
        self.rg_idx = rg_idx
        self.b_idx = b_idx
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.attns = nn.ModuleList([
            Spatial_Attention(dim // 2, idx=i, split_size=split_size, num_heads=num_heads // 2)
            for i in range(2)
        ])

        self.do_shift = (rg_idx % 2 == 0 and b_idx > 0 and (b_idx - 2) % 4 == 0) or (rg_idx % 2 != 0 and b_idx % 4 == 0)

        self.dwconv, self.channel_interaction, self.spatial_interaction = _aim_modules(dim)

    def calculate_mask(self, H: int, W: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        img_mask_0 = torch.zeros((1, H, W, 1), device=device)
        img_mask_1 = torch.zeros((1, H, W, 1), device=device)
        sp0, sp1 = self.split_size[0], self.split_size[1]
        sh0, sh1 = self.shift_size[0], self.shift_size[1]

        h_slices_0 = (slice(0, -sp0), slice(-sp0, -sh0), slice(-sh0, None))
        w_slices_0 = (slice(0, -sp1), slice(-sp1, -sh1), slice(-sh1, None))
        h_slices_1 = (slice(0, -sp1), slice(-sp1, -sh1), slice(-sh1, None))
        w_slices_1 = (slice(0, -sp0), slice(-sp0, -sh0), slice(-sh0, None))

        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        img_mask_0 = img_mask_0.view(1, H // sp0, sp0, W // sp1, sp1, 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, sp0, sp1, 1)
        mask_windows_0 = img_mask_0.view(-1, sp0 * sp1)
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        img_mask_1 = img_mask_1.view(1, H // sp1, sp1, W // sp0, sp0, 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, sp1, sp0, 1)
        mask_windows_1 = img_mask_1.view(-1, sp1 * sp0)
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))

        return attn_mask_0, attn_mask_1

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        v = qkv[2].transpose(-2, -1).contiguous().view(B, C, H, W)

        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv = qkv.reshape(3 * B, H, W, C).permute(0, 3, 1, 2)
        qkv = F.pad(qkv, (0, pad_r, 0, pad_b)).reshape(3, B, C, -1).transpose(-2, -1)
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W

        if self.do_shift:
            qkv = qkv.view(3, B, _H, _W, C)
            qkv_0 = torch.roll(qkv[:, :, :, :, : C // 2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            qkv_0 = qkv_0.view(3, B, _L, C // 2)
            qkv_1 = torch.roll(qkv[:, :, :, :, C // 2 :], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_1 = qkv_1.view(3, B, _L, C // 2)

            mask_0, mask_1 = self.calculate_mask(_H, _W, x.device)
            x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_0)
            x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_1)

            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
            x1 = x1[:, :H, :W, :].reshape(B, L, C // 2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C // 2)
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            x1 = self.attns[0](qkv[:, :, :, : C // 2], _H, _W)[:, :H, :W, :].reshape(B, L, C // 2)
            x2 = self.attns[1](qkv[:, :, :, C // 2 :], _H, _W)[:, :H, :W, :].reshape(B, L, C // 2)
            attened_x = torch.cat([x1, x2], dim=2)

        conv_x = self.dwconv(v)

        channel_map = self.channel_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, 1, C)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        spatial_map = self.spatial_interaction(attention_reshape)

        attened_x = attened_x * torch.sigmoid(channel_map)
        conv_x = torch.sigmoid(spatial_map) * conv_x
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + conv_x
        return self.proj(x)


class Adaptive_Channel_Attention(nn.Module):
    """DCTB attention: XCiT-style normalized channel attention blended with
    a parallel depthwise-conv branch via AIM gating."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dwconv, self.channel_interaction, self.spatial_interaction = _aim_modules(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        v_ = v.reshape(B, C, N).contiguous().view(B, C, H, W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        conv_x = self.dwconv(v_)

        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        channel_map = self.channel_interaction(attention_reshape)
        spatial_map = self.spatial_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, N, 1)

        attened_x = attened_x * torch.sigmoid(spatial_map)
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        x = attened_x + conv_x
        return self.proj(x)


class DATB(nn.Module):
    def __init__(self, dim: int, num_heads: int, split_size: list[int], shift_size: list[int], expansion_factor: float, rg_idx: int, b_idx: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if b_idx % 2 == 0:
            self.attn: nn.Module = Adaptive_Spatial_Attention(dim, num_heads, split_size, shift_size, rg_idx, b_idx)
        else:
            self.attn = Adaptive_Channel_Attention(dim, num_heads)
        self.ffn = SGFN(dim, int(dim * expansion_factor))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class ResidualGroup(nn.Module):
    def __init__(self, dim: int, num_heads: int, split_size: list[int], expansion_factor: float, depth: int, rg_idx: int) -> None:
        super().__init__()
        shift_size = [split_size[0] // 2, split_size[1] // 2]
        self.blocks = nn.ModuleList([
            DATB(dim, num_heads, split_size, shift_size, expansion_factor, rg_idx, b_idx=i)
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        res = x
        for blk in self.blocks:
            x = blk(x, H, W)
        B, _, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return res + x


class Upsample(nn.Sequential):
    def __init__(self, scale: int, num_feat: int) -> None:
        m = []
        if (scale & (scale - 1)) == 0:  # 2^n
            for _ in range(int(math.log(scale, 2))):
                m += [nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2)]
        elif scale == 3:
            m += [nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1), nn.PixelShuffle(3)]
        else:
            raise ValueError(f"scale {scale} is not supported")
        super().__init__(*m)


class DAT(nn.Module):
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        split_size: list[int],
        depth: list[int],
        num_heads: list[int],
        expansion_factor: float,
        upscale: int,
        num_feat: int = 64,
        img_range: float = 1.0,
    ) -> None:
        super().__init__()
        self.img_range = img_range
        if in_chans == 3:
            self.mean = torch.Tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        self.before_RG_norm = nn.LayerNorm(embed_dim)

        self.layers = nn.ModuleList([
            ResidualGroup(embed_dim, num_heads[i], split_size, expansion_factor, depth[i], rg_idx=i)
            for i in range(len(depth))
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, in_chans, 3, 1, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.before_RG_norm(x)
        for layer in self.layers:
            x = layer(x, H, W)
        x = self.norm(x)
        return x.transpose(1, 2).contiguous().view(-1, x.shape[-1], H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 36 blocks (6 groups x 6) at constant channel width feed a single
        # unnormalized residual stream (pre-norm design: only each block's
        # *input* to attn/ffn is normalized, not the accumulating residual
        # itself) -- unlike a U-Net, there's no downsampling to keep
        # magnitudes bounded, and under fp16 autocast this residual stream
        # was measured overflowing fp16's ~65504 max well before the final
        # norm, producing -inf -> NaN. DAT's attention is windowed (local,
        # not full quadratic), so it's cheap enough in memory that always
        # running it in fp32 -- opting out of any ambient autocast -- is a
        # fine tradeoff for correctness.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
            x = x / self.img_range + self.mean
            return x


def _infer_split_size(state_dict: dict, prefix: str) -> list[int]:
    """Recovers [H_sp, W_sp] for attns.0 from the shapes of its
    DynamicPosBias buffers -- neither weight is otherwise resolution- or
    hyperparameter-labelled, but rpe_biases has (2*H_sp-1)*(2*W_sp-1) rows
    and relative_position_index is (H_sp*W_sp) square, which together pin
    down the unique (unordered-then-oriented) integer pair."""
    total = state_dict[f"{prefix}.relative_position_index"].shape[0]
    rpe_n = state_dict[f"{prefix}.rpe_biases"].shape[0]
    for h_sp in range(1, total + 1):
        if total % h_sp:
            continue
        w_sp = total // h_sp
        if (2 * h_sp - 1) * (2 * w_sp - 1) == rpe_n:
            return [h_sp, w_sp]
    raise ValueError(f"Could not infer split_size from {prefix} (total={total}, rpe_n={rpe_n})")


def is_dat_state_dict(state_dict: dict) -> bool:
    return any(k.endswith("attn.attns.0.rpe_biases") for k in state_dict)


def build_dat(state_dict: dict) -> tuple[nn.Module, dict]:
    state_dict = {k: v for k, v in state_dict.items() if not re.search(r"\.attn_mask_[01]$", k)}

    embed_dim = state_dict["conv_first.weight"].shape[0]
    in_chans = state_dict["conv_first.weight"].shape[1]

    num_layers = 0
    while f"layers.{num_layers}.blocks.0.norm1.weight" in state_dict:
        num_layers += 1
    if num_layers == 0:
        raise ValueError("Not a recognized DAT state_dict")

    depth = []
    num_heads = []
    for i in range(num_layers):
        d = 0
        while f"layers.{i}.blocks.{d}.norm1.weight" in state_dict:
            d += 1
        depth.append(d)
        num_heads.append(state_dict[f"layers.{i}.blocks.0.attn.attns.0.pos.pos3.2.weight"].shape[0] * 2)

    split_size = _infer_split_size(state_dict, "layers.0.blocks.0.attn.attns.0")

    ffn_hidden = state_dict["layers.0.blocks.0.ffn.fc1.weight"].shape[0]
    expansion_factor = ffn_hidden / embed_dim

    num_feat = state_dict["conv_before_upsample.0.weight"].shape[0]
    out_chans = state_dict["conv_last.weight"].shape[0]

    up_stage_out = state_dict["upsample.0.weight"].shape[0]
    if up_stage_out == 9 * num_feat:
        upscale = 3
    else:
        stages = 0
        while f"upsample.{2 * stages}.weight" in state_dict:
            stages += 1
        upscale = 2**stages

    model = DAT(
        in_chans=in_chans, embed_dim=embed_dim, split_size=split_size, depth=depth,
        num_heads=num_heads, expansion_factor=expansion_factor, upscale=upscale, num_feat=num_feat,
    )
    # This project's DAT.before_RG_norm/DAT.norm stand in for the reference
    # Sequential(Rearrange, LayerNorm)'s [1] entry and the top-level norm
    # respectively; remap so strict loading lines up with the checkpoint's
    # own "before_RG.1.*" naming.
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("before_RG.1."):
            k = "before_RG_norm." + k[len("before_RG.1.") :]
        remapped[k] = v

    model.load_state_dict(remapped, strict=True)
    meta = {
        "architecture": "DAT", "scale": upscale, "in_channels": in_chans, "out_channels": out_chans,
        "size_multiple": 1,  # DAT pads internally to a multiple of max(split_size) itself
    }
    return model, meta
