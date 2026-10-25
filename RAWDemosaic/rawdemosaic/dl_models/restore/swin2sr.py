# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Swin2SR (mv-lab/swin2sr), reproduced from the official
models/network_swin2sr.py (verified tensor-shape-for-tensor-shape against
the actual checkpoints -- see restore/registry.py). Every hyperparameter
(embed_dim, depths, num_heads, window_size, mlp_ratio, resi_connection,
upsampler variant, upscale) is derived directly from the checkpoint's own
tensor shapes rather than a per-file lookup table, so any future Swin2SR
checkpoint should work without code changes.

The reference implementation depends on timm for three inference-irrelevant
utilities: DropPath (a training-only stochastic-depth regularizer -- a no-op
at eval time, so nn.Identity() here), trunc_normal_ init (overwritten
immediately by load_state_dict, so unused), and to_2tuple (trivial, inlined).

Each SwinTransformerBlock registers a persistent `attn_mask` buffer sized
for whatever resolution the checkpoint happened to be *trained* at; at
*inference* on a different-sized image the model recomputes this mask
on-the-fly per SwinTransformerBlock.forward's own resolution check, so the
stored buffer's shape is never actually load-bearing here. load_state_dict
is therefore called non-strict, with an explicit check that the only
tolerated mismatches are exactly these attn_mask buffers (anything else
still raises).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2tuple(x) -> tuple:
    return x if isinstance(x, (tuple, list)) else (x, x)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: tuple, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False))

        wh, ww = window_size
        relative_coords_h = torch.arange(-(wh - 1), wh, dtype=torch.float32)
        relative_coords_w = torch.arange(-(ww - 1), ww, dtype=torch.float32)
        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
            .permute(1, 2, 0).contiguous().unsqueeze(0)
        )
        relative_coords_table[:, :, :, 0] /= wh - 1
        relative_coords_table[:, :, :, 1] /= ww - 1
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0
        ) / math.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table)

        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += wh - 1
        relative_coords[:, :, 1] += ww - 1
        relative_coords[:, :, 0] *= 2 * ww - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(b_, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale

        rel_pos_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        wh, ww = self.window_size
        relative_position_bias = rel_pos_bias_table[self.relative_position_index.view(-1)].view(wh * ww, wh * ww, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size, mlp_ratio, qkv_bias) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, _to_2tuple(self.window_size), num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

        attn_mask = self.calculate_mask(input_resolution) if self.shift_size > 0 else None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size) -> torch.Tensor:
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    def forward(self, x: torch.Tensor, x_size) -> torch.Tensor:
        h, w = x_size
        b, _l, c = x.shape
        shortcut = x
        x = x.view(b, h, w, c)

        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x
        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, c)

        if self.input_resolution == x_size:
            mask = self.attn_mask
        elif self.shift_size > 0:
            mask = self.calculate_mask(x_size).to(x.device)
        else:
            mask = None
        attn_windows = self.attn(x_windows, mask=mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x
        x = x.view(b, h * w, c)

        x = shortcut + self.norm1(x)
        x = x + self.norm2(self.mlp(x))
        return x


class PatchEmbed(nn.Module):
    """patch_size is always 1 in every Swin2SR config, but proj is still a
    real learned 1x1 conv (not an identity reshape) -- it must run."""

    def __init__(self, embed_dim: int, norm: bool) -> None:
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, embed_dim, 1, 1, 0)
        self.norm = nn.LayerNorm(embed_dim) if norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x) if self.norm is not None else x


class PatchUnEmbed(nn.Module):
    def forward(self, x: torch.Tensor, x_size, embed_dim: int) -> torch.Tensor:
        b = x.shape[0]
        return x.transpose(1, 2).view(b, embed_dim, x_size[0], x_size[1])


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio, qkv_bias) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim, input_resolution, num_heads, window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, x_size) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, x_size)
        return x


class RSTB(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio, qkv_bias, resi_connection) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.embed_dim = dim
        self.residual_group = BasicLayer(dim, input_resolution, depth, num_heads, window_size, mlp_ratio, qkv_bias)

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        self.patch_embed = PatchEmbed(dim, norm=False)
        self.patch_unembed = PatchUnEmbed()

    def forward(self, x: torch.Tensor, x_size) -> torch.Tensor:
        out = self.residual_group(x, x_size)
        out = self.patch_unembed(out, x_size, self.embed_dim)
        out = self.conv(out)
        out = self.patch_embed(out)
        return out + x


class Upsample(nn.Sequential):
    def __init__(self, scale: int, num_feat: int) -> None:
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m += [nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2)]
        elif scale == 3:
            m += [nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1), nn.PixelShuffle(3)]
        else:
            raise ValueError(f"scale {scale} not supported")
        super().__init__(*m)


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale: int, num_feat: int, num_out_ch: int) -> None:
        super().__init__(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1), nn.PixelShuffle(scale))


class Swin2SR(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: list[int] | None = None,
        num_heads: list[int] | None = None,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        img_range: float = 1.0,
        upscale: int = 4,
        upsampler: str = "",
        resi_connection: str = "1conv",
        input_resolution: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        depths = depths or [6, 6, 6, 6]
        num_heads = num_heads or [6, 6, 6, 6]
        num_feat = 64
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.embed_dim = embed_dim
        if in_chans == 3:
            self.mean = torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(embed_dim, norm=True)
        self.patch_unembed = PatchUnEmbed()

        self.layers = nn.ModuleList([
            RSTB(embed_dim, input_resolution, depths[i], num_heads[i], window_size, mlp_ratio, qkv_bias, resi_connection)
            for i in range(len(depths))
        ])
        self.norm = nn.LayerNorm(embed_dim)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        else:
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        if upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, in_chans, 3, 1, 1)
        elif upsampler == "pixelshuffle_aux":
            self.conv_bicubic = nn.Conv2d(in_chans, num_feat, 3, 1, 1)
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_aux = nn.Conv2d(num_feat, in_chans, 3, 1, 1)
            self.conv_after_aux = nn.Sequential(nn.Conv2d(3, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, in_chans, 3, 1, 1)
        elif upsampler == "pixelshuffledirect":
            self.upsample = UpsampleOneStep(upscale, embed_dim, in_chans)
        elif upsampler == "nearest+conv":
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, in_chans, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        return self.patch_unembed(x, x_size, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffle_aux":
            bicubic = F.interpolate(x, size=(h * self.upscale, w * self.upscale), mode="bicubic", align_corners=False)
            bicubic = self.conv_bicubic(bicubic)
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            aux = self.conv_aux(x)
            x = self.conv_after_aux(aux)
            x = self.upsample(x)[:, :, : h * self.upscale, : w * self.upscale] + bicubic[:, :, : h * self.upscale, : w * self.upscale]
            x = self.conv_last(x)
        elif self.upsampler == "pixelshuffledirect":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == "nearest+conv":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode="nearest")))
            x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode="nearest")))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        return x[:, :, : h * self.upscale, : w * self.upscale]


def is_swin2sr_state_dict(state_dict: dict) -> bool:
    return "conv_first.weight" in state_dict and any(
        ".residual_group.blocks.0.attn.logit_scale" in k for k in state_dict
    )


def build_swin2sr(state_dict: dict) -> tuple[nn.Module, dict]:
    embed_dim = state_dict["conv_first.weight"].shape[0]
    in_chans = state_dict["conv_first.weight"].shape[1]

    num_layers = 0
    while f"layers.{num_layers}.residual_group.blocks.0.norm1.weight" in state_dict:
        num_layers += 1
    if num_layers == 0:
        raise ValueError("Not a recognized Swin2SR state_dict")

    depths, num_heads = [], []
    for i in range(num_layers):
        d = 0
        while f"layers.{i}.residual_group.blocks.{d}.norm1.weight" in state_dict:
            d += 1
        depths.append(d)
        num_heads.append(state_dict[f"layers.{i}.residual_group.blocks.0.attn.logit_scale"].shape[0])

    coords_table = state_dict["layers.0.residual_group.blocks.0.attn.relative_coords_table"]
    window_size = (coords_table.shape[1] + 1) // 2
    mlp_hidden = state_dict["layers.0.residual_group.blocks.0.mlp.fc1.weight"].shape[0]
    mlp_ratio = mlp_hidden / embed_dim
    qkv_bias = "layers.0.residual_group.blocks.0.attn.q_bias" in state_dict
    resi_connection = "1conv" if "layers.0.conv.weight" in state_dict else "3conv"

    if "conv_up1.weight" in state_dict:
        upsampler, upscale = "nearest+conv", 4
    elif "conv_bicubic.weight" in state_dict:
        num_feat = state_dict["conv_before_upsample.0.weight"].shape[0]
        n_steps = sum(1 for k in state_dict if k.startswith("upsample.") and k.endswith(".weight") and state_dict[k].ndim == 4)
        upsampler, upscale = "pixelshuffle_aux", 2**n_steps
    elif "upsample.0.weight" in state_dict and "conv_before_upsample.0.weight" in state_dict:
        n_steps = sum(1 for k in state_dict if k.startswith("upsample.") and k.endswith(".weight") and state_dict[k].ndim == 4)
        upsampler, upscale = "pixelshuffle", 2**n_steps
    elif "upsample.0.weight" in state_dict:
        out_ch_scale2 = state_dict["upsample.0.weight"].shape[0]
        upscale = round((out_ch_scale2 / in_chans) ** 0.5)
        upsampler = "pixelshuffledirect"
    else:
        upsampler, upscale = "", 1

    img_range = 1.0

    model = Swin2SR(
        in_chans=in_chans, embed_dim=embed_dim, depths=depths, num_heads=num_heads,
        window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, img_range=img_range,
        upscale=upscale, upsampler=upsampler, resi_connection=resi_connection,
        # Only used to size the (later discarded/recomputed, see module
        # docstring) construction-time attn_mask buffers -- must be a valid
        # multiple of window_size AND strictly greater than it, or
        # SwinTransformerBlock's own "window >= resolution" guard forces
        # shift_size=0 permanently on every block (silently disabling SW-MSA
        # for real inference too, not just at this throwaway construction).
        input_resolution=(window_size * 2, window_size * 2),
    )

    filtered = {k: v for k, v in state_dict.items() if not k.endswith(".attn_mask")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    bad_missing = [k for k in missing if not k.endswith(".attn_mask")]
    if bad_missing or unexpected:
        raise RuntimeError(f"Swin2SR state_dict mismatch -- missing={bad_missing[:10]} unexpected={list(unexpected)[:10]}")

    meta = {
        "architecture": "Swin2SR", "embed_dim": embed_dim, "depths": depths, "num_heads": num_heads,
        "window_size": window_size, "upsampler": upsampler, "scale": upscale, "resi_connection": resi_connection,
        "in_channels": in_chans, "out_channels": in_chans,
        "size_multiple": 1,  # self-pads/crops internally (check_image_size)
    }
    return model, meta
