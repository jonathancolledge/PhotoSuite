# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""NAFSSR (megvii-research), reproduced from the official
basicsr/models/archs/NAFSSR_arch.py (verified tensor-shape-for-tensor-shape
against NAFSSR-L_2x.pth -- see restore/registry.py). Reuses NAFBlock and
LayerNorm2d from nafnet.py unchanged (NAFSSR's own arch file imports them
from NAFNet_arch.py directly, rather than redefining them).

NAFSSR is a STEREO (dual-image) super-resolution network: it takes a
concatenated 6-channel (left-RGB + right-RGB) input and produces a
concatenated 6-channel (super-resolved left + right) output, using a
"Stereo Cross Attention Module" (SCAM) between the two views in every
block. There's no real second view in this single-image tool, so
registry.py feeds the same image as both "views" (duplicated, not
degrading NAFBlock's own per-view processing at all) and averages the two
resulting views back into one image afterward -- SCAM's cross-attention
between two identical views degrades gracefully to attending over a
mirror of itself, rather than crashing or requiring a different code path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nafnet import LayerNorm2d, NAFBlock


class SCAM(nn.Module):
    """Stereo Cross Attention Module."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.scale = c**-0.5
        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, 1, 1, 0)
        self.r_proj1 = nn.Conv2d(c, c, 1, 1, 0)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.l_proj2 = nn.Conv2d(c, c, 1, 1, 0)
        self.r_proj2 = nn.Conv2d(c, c, 1, 1, 0)

    def forward(self, x_l: torch.Tensor, x_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)
        q_r_t = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3)
        v_l = self.l_proj2(x_l).permute(0, 2, 3, 1)
        v_r = self.r_proj2(x_r).permute(0, 2, 3, 1)

        attention = torch.matmul(q_l, q_r_t) * self.scale
        f_r2l = torch.matmul(torch.softmax(attention, dim=-1), v_r)
        f_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), v_l)

        f_r2l = f_r2l.permute(0, 3, 1, 2) * self.beta
        f_l2r = f_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + f_r2l, x_r + f_l2r


class NAFBlockSR(nn.Module):
    def __init__(self, c: int, fusion: bool = False) -> None:
        super().__init__()
        self.blk = NAFBlock(c)
        self.fusion = SCAM(c) if fusion else None

    def forward(self, *feats: torch.Tensor) -> tuple[torch.Tensor, ...]:
        feats = tuple(self.blk(x) for x in feats)
        if self.fusion is not None:
            feats = self.fusion(*feats)
        return feats


class NAFSSR(nn.Module):
    def __init__(
        self,
        up_scale: int = 4,
        width: int = 48,
        num_blks: int = 16,
        img_channel: int = 3,
        fusion_from: int = -1,
        fusion_to: int = -1,
    ) -> None:
        super().__init__()
        self.up_scale = up_scale
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, groups=1, bias=True)
        self.body = nn.ModuleList([
            NAFBlockSR(width, fusion=(fusion_from <= i <= fusion_to)) for i in range(num_blks)
        ])
        self.up = nn.Sequential(
            nn.Conv2d(width, img_channel * up_scale**2, 3, 1, 1, groups=1, bias=True),
            nn.PixelShuffle(up_scale),
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode="bilinear")
        views = inp.chunk(2, dim=1)
        feats = tuple(self.intro(x) for x in views)
        for block in self.body:
            feats = block(*feats)
        out = torch.cat([self.up(x) for x in feats], dim=1)
        return out + inp_hr


def is_nafssr_state_dict(state_dict: dict) -> bool:
    return "body.0.module.blk.beta" in state_dict or "body.0.blk.beta" in state_dict


def build_nafssr(state_dict: dict) -> tuple[nn.Module, dict]:
    # The reference DropPath wrapper (a no-op at inference) adds a ".module."
    # path segment; some re-exported checkpoints omit it. Normalize either way.
    if any(".module.blk." in k for k in state_dict):
        state_dict = {k.replace(".module.", "."): v for k, v in state_dict.items()}

    num_blks = 0
    while f"body.{num_blks}.blk.beta" in state_dict:
        num_blks += 1
    if num_blks == 0:
        raise ValueError("Not a recognized NAFSSR state_dict")

    width = state_dict["intro.weight"].shape[0]
    img_channel = state_dict["intro.weight"].shape[1]
    up_out = state_dict["up.0.weight"].shape[0]
    up_scale = round((up_out / img_channel) ** 0.5)

    fusion_indices = sorted(
        {int(k.split(".")[1]) for k in state_dict if k.split(".")[0] == "body" and ".fusion." in k}
    )
    fusion_from = fusion_indices[0] if fusion_indices else -1
    fusion_to = fusion_indices[-1] if fusion_indices else -1

    model = NAFSSR(
        up_scale=up_scale, width=width, num_blks=num_blks, img_channel=img_channel,
        fusion_from=fusion_from, fusion_to=fusion_to,
    )
    model.load_state_dict(state_dict, strict=True)
    meta = {
        "architecture": "NAFSSR", "scale": up_scale, "width": width, "num_blks": num_blks,
        "in_channels": img_channel * 2, "out_channels": img_channel * 2,
        "size_multiple": 1,  # NAFBlock's own padder logic isn't used here (no encoder/decoder downsampling)
        "stereo_duplicate": True,  # registry.py: feed the single image as both "views" and average the outputs
    }
    return model, meta
