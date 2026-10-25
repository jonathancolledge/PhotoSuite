# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""NAFNet (megvii-research), reproduced from the official basicsr
NAFNet_arch.py/arch_util.py (verified tensor-shape-for-tensor-shape against
the actual checkpoints -- see restore/registry.py). LayerNorm2d's forward is
reimplemented as a plain functional op (the original's custom
torch.autograd.Function only exists to make its backward pass cheaper during
training; inference-only here needs no custom backward).

NAFSSR (stereo/dual-image super-resolution) checkpoints are NOT supported --
they take a left+right image pair, not a fit for this single-image tool.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def _layer_norm_2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = x.mean(1, keepdim=True)
    var = (x - mu).pow(2).mean(1, keepdim=True)
    y = (x - mu) / (var + eps).sqrt()
    c = weight.shape[0]
    return weight.view(1, c, 1, 1) * y + bias.view(1, c, 1, 1)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _layer_norm_2d(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2) -> None:
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, groups=1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, groups=1, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self,
        img_channel: int = 3,
        width: int = 16,
        middle_blk_num: int = 1,
        enc_blk_nums: list[int] | None = None,
        dec_blk_nums: list[int] | None = None,
    ) -> None:
        super().__init__()
        enc_blk_nums = enc_blk_nums or []
        dec_blk_nums = dec_blk_nums or []

        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, groups=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, h, w = inp.shape
        inp = self._check_image_size(inp)

        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x) + inp
        return x[:, :, :h, :w]

    def _check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))


def is_nafnet_state_dict(state_dict: dict) -> bool:
    return "intro.weight" in state_dict and "ending.weight" in state_dict and any(
        k.startswith("encoders.") for k in state_dict
    )


def build_nafnet(state_dict: dict) -> tuple[nn.Module, dict]:
    def block_count(prefix: str, stage: int) -> int:
        indices = set()
        pat = re.compile(rf"^{re.escape(prefix)}\.{stage}\.(\d+)\.beta$")
        for k in state_dict:
            m = pat.match(k)
            if m:
                indices.add(int(m.group(1)))
        return (max(indices) + 1) if indices else 0

    num_stages = 0
    while f"downs.{num_stages}.weight" in state_dict:
        num_stages += 1

    enc_blk_nums = [block_count("encoders", i) for i in range(num_stages)]
    dec_blk_nums = [block_count("decoders", i) for i in range(num_stages)]
    middle_indices = {int(m.group(1)) for k in state_dict if (m := re.match(r"^middle_blks\.(\d+)\.beta$", k))}
    middle_blk_num = (max(middle_indices) + 1) if middle_indices else 0
    width = state_dict["intro.weight"].shape[0]
    img_channel = state_dict["intro.weight"].shape[1]

    model = NAFNet(
        img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums,
    )
    model.load_state_dict(state_dict, strict=True)
    meta = {
        "architecture": "NAFNet", "width": width,
        "enc_blk_nums": enc_blk_nums, "middle_blk_num": middle_blk_num, "dec_blk_nums": dec_blk_nums,
        "in_channels": img_channel, "out_channels": img_channel,
        "size_multiple": 1,  # self-pads/crops internally (_check_image_size)
    }
    return model, meta
