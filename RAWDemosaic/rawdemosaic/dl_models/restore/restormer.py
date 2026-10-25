# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Restormer (swz30/Restormer), reproduced from the official
basicsr/models/archs/restormer_arch.py (verified tensor-shape-for-tensor-
shape against the actual checkpoints -- see restore/registry.py).

The reference implementation uses einops.rearrange for a handful of pure
reshape/permute operations; reimplemented here with plain torch view/permute
to avoid adding a dependency for that alone.
"""

from __future__ import annotations

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    # b c h w -> b (h w) c
    b, c, h, w = x.shape
    return x.flatten(2).transpose(1, 2)


def _to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # b (h w) c -> b c h w
    b, _, c = x.shape
    return x.transpose(1, 2).view(b, c, h, w)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))
        self.bias = nn.Parameter(torch.zeros(torch.Size(normalized_shape)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str) -> None:
        super().__init__()
        self.body = BiasFreeLayerNorm(dim) if layer_norm_type == "BiasFree" else WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, 3, 1, 1, groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # b (head c) h w -> b head c (h w)
            return t.view(b, self.num_heads, c // self.num_heads, h * w)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v  # b head c (h w)
        out = out.view(b, c, h, w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float, bias: bool, layer_norm_type: str) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c: int = 3, embed_dim: int = 48, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False), nn.PixelUnshuffle(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False), nn.PixelShuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Restormer(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: list[int] | None = None,
        num_refinement_blocks: int = 4,
        heads: list[int] | None = None,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
        dual_pixel_task: bool = False,
    ) -> None:
        super().__init__()
        num_blocks = num_blocks or [4, 6, 6, 8]
        heads = heads or [1, 2, 4, 8]

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        def stage(d: int, h: int, n: int) -> nn.Sequential:
            return nn.Sequential(*[TransformerBlock(d, h, ffn_expansion_factor, bias, layer_norm_type) for _ in range(n)])

        self.encoder_level1 = stage(dim, heads[0], num_blocks[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = stage(dim * 2, heads[1], num_blocks[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = stage(dim * 4, heads[2], num_blocks[2])
        self.down3_4 = Downsample(dim * 4)
        self.latent = stage(dim * 8, heads[3], num_blocks[3])

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = stage(dim * 4, heads[2], num_blocks[2])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = stage(dim * 2, heads[1], num_blocks[1])

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = stage(dim * 2, heads[0], num_blocks[0])

        self.refinement = stage(dim * 2, heads[0], num_refinement_blocks)

        self.dual_pixel_task = dual_pixel_task
        if dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim * 2, 1, bias=bias)

        self.output = nn.Conv2d(dim * 2, out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            return self.output(out_dec_level1)
        return self.output(out_dec_level1) + inp_img


def is_restormer_state_dict(state_dict: dict) -> bool:
    return "patch_embed.proj.weight" in state_dict and any(
        k.startswith("encoder_level1.") and ".attn.temperature" in k for k in state_dict
    )


def build_restormer(state_dict: dict) -> tuple[nn.Module, dict]:
    dim = state_dict["patch_embed.proj.weight"].shape[0]
    inp_channels = state_dict["patch_embed.proj.weight"].shape[1]
    out_channels = state_dict["output.weight"].shape[0]
    bias = "patch_embed.proj.bias" in state_dict
    layer_norm_type = "WithBias" if "encoder_level1.0.norm1.body.bias" in state_dict else "BiasFree"
    dual_pixel_task = "skip_conv.weight" in state_dict

    def count_blocks(prefix: str) -> int:
        indices = {int(k.split(".")[1]) for k in state_dict if k.startswith(prefix + ".") and ".attn.temperature" in k}
        return (max(indices) + 1) if indices else 0

    num_blocks = [count_blocks(f"encoder_level{i}") for i in (1, 2, 3)] + [count_blocks("latent")]
    num_refinement_blocks = count_blocks("refinement")
    heads = [state_dict[f"encoder_level{i}.0.attn.temperature"].shape[0] for i in (1, 2, 3)] + [
        state_dict["latent.0.attn.temperature"].shape[0]
    ]
    # ffn_expansion_factor is a fixed architectural constant (2.66) in every
    # published Restormer checkpoint, not something that varies per file --
    # and it can't be cleanly back-derived from one stage's hidden_features
    # (int(dim * factor) truncates differently at each stage's dim).
    ffn_expansion_factor = 2.66

    model = Restormer(
        inp_channels=inp_channels, out_channels=out_channels, dim=dim,
        num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
        heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
        layer_norm_type=layer_norm_type, dual_pixel_task=dual_pixel_task,
    )
    model.load_state_dict(state_dict, strict=True)
    meta = {
        "architecture": "Restormer", "dim": dim, "num_blocks": num_blocks,
        "heads": heads, "dual_pixel_task": dual_pixel_task,
        "in_channels": inp_channels, "out_channels": out_channels,
        "size_multiple": 8,  # 3x PixelUnshuffle(2) downsampling, no internal padding
    }
    return model, meta
