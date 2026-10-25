# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""RRDBNet (ESRGAN / Real-ESRGAN / RealSR) and SRVGGNetCompact
(realesr-general-x4v3 / realesr-animevideov3), reproduced from Real-ESRGAN's
basicsr rrdbnet_arch.py / srvgg_arch.py (BSD/Apache-licensed, widely
reused) -- verified tensor-shape-for-tensor-shape against the actual
checkpoints in PostProcessingModels before use (see restore/registry.py).

Two on-disk state_dict key conventions exist for what is architecturally
the same RRDBNet:
  * "clean" (Real-ESRGAN's own releases): body.N.rdb1/rdb2/rdb3.convX,
    conv_body, conv_hr.
  * "old ESRGAN" (the original ESRGAN repo and RealSR's DF2K/DF2K_JPEG/DPED
    checkpoints, which reuse ESRGAN's own generator unmodified): RRDB_trunk.
    N.RDB1/RDB2/RDB3.convX, trunk_conv, upconv1/upconv2, HRconv.
_remap_old_esrgan_keys() below normalizes the second convention onto the
first so a single RRDBNet class handles both.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        num_upconv: int = 2,
    ) -> None:
        """num_upconv: how many 2x-nearest+conv upsample stages actually
        exist as real layers (0, 1, or 2) -- Real-ESRGAN/RealSR checkpoints
        always have 2 (using pixel_unshuffle on the input instead to get a
        net scale below 4x), but some community "old-style" ESRGAN exports
        (e.g. a pure 1x deblur/refocus model) genuinely omit the upsample
        stage(s) as distinct layers rather than relying on pixel_unshuffle;
        loading one of those into a fixed-4-tail-conv class would leave
        conv_up1/conv_up2 with no matching checkpoint weights at all."""
        super().__init__()
        self.scale = scale
        self.num_upconv = num_upconv
        # pixel_unshuffle is only ever a Real-ESRGAN-style "shrink the input
        # instead" trick, which always keeps both upconv stages (net scale
        # is 4x internally, then unshuffle-then-4x nets out at 1x/2x); a
        # network missing an upconv stage as a real layer (num_upconv < 2)
        # is a genuinely different-scale network, not this trick.
        unshuffle_in_ch = num_in_ch
        if num_upconv == 2:
            if scale == 2:
                unshuffle_in_ch = num_in_ch * 4
            elif scale == 1:
                unshuffle_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(unshuffle_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1) if num_upconv >= 1 else None
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1) if num_upconv >= 2 else None
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_upconv == 2 and self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.num_upconv == 2 and self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        if self.conv_up1 is not None:
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.conv_up2 is not None:
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def _remap_old_esrgan_keys(state_dict: dict) -> dict:
    remapped = {}
    for k, v in state_dict.items():
        k = re.sub(r"^RRDB_trunk\.(\d+)\.RDB1\.", r"body.\1.rdb1.", k)
        k = re.sub(r"^RRDB_trunk\.(\d+)\.RDB2\.", r"body.\1.rdb2.", k)
        k = re.sub(r"^RRDB_trunk\.(\d+)\.RDB3\.", r"body.\1.rdb3.", k)
        k = k.replace("trunk_conv.", "conv_body.")
        k = k.replace("upconv1.", "conv_up1.")
        k = k.replace("upconv2.", "conv_up2.")
        k = k.replace("HRconv.", "conv_hr.")
        remapped[k] = v
    return remapped


def _remap_sequential_esrgan_keys(state_dict: dict) -> dict:
    """The "chaiNNer"/old-traiNNer-BasicSR export format: the whole network
    is one flat nn.Sequential named "model", so every layer is addressed by
    a bare integer index instead of a name. Verified against traiNNer's own
    RRDBNet_arch.py: model.0=conv_first (bias, no activation so it's never
    bundled+flattened into extra sibling indices); model.1=a ShortcutBlock
    whose own .sub is [RRDB]*nb + a bare trunk_conv (mirroring RRDB_trunk/
    trunk_conv above, so the same RDB1/2/3 remap applies inside .sub); then
    whichever of conv_up1/conv_up2/HRconv/conv_last actually exist as
    distinct flat indices (2 remaining = a true 1x model with neither
    upconv stage; 3 = one upconv stage; 4 = both, i.e. RealESRGAN's usual
    4x tail) -- the exact index numbers vary (each conv_block with an
    activation contributes an extra sibling index for that activation,
    which has no state_dict entries of its own and is simply skipped over),
    so what matters is only the ascending ORDER they appear in.
    """
    num_block = 0
    while f"model.1.sub.{num_block}.RDB1.conv1.0.weight" in state_dict:
        num_block += 1

    remapped = {}
    tail_indices = sorted(
        {
            int(m.group(1))
            for k in state_dict
            if (m := re.match(r"^model\.(\d+)\.weight$", k)) and int(m.group(1)) > 1
        }
    )
    tail_names = {2: ["conv_hr", "conv_last"], 3: ["conv_up1", "conv_hr", "conv_last"], 4: ["conv_up1", "conv_up2", "conv_hr", "conv_last"]}
    tail_map = dict(zip(tail_indices, tail_names.get(len(tail_indices), [])))

    for k, v in state_dict.items():
        if k in ("model.0.weight", "model.0.bias"):
            remapped[k.replace("model.0.", "conv_first.")] = v
            continue
        m = re.match(rf"^model\.1\.sub\.{num_block}\.(weight|bias)$", k)
        if m:
            remapped[f"conv_body.{m.group(1)}"] = v
            continue
        k2 = re.sub(r"^model\.1\.sub\.(\d+)\.RDB1\.(\w+)\.0\.", r"body.\1.rdb1.\2.", k)
        k2 = re.sub(r"^model\.1\.sub\.(\d+)\.RDB2\.(\w+)\.0\.", r"body.\1.rdb2.\2.", k2)
        k2 = re.sub(r"^model\.1\.sub\.(\d+)\.RDB3\.(\w+)\.0\.", r"body.\1.rdb3.\2.", k2)
        if k2 != k:
            remapped[k2] = v
            continue
        m = re.match(r"^model\.(\d+)\.(weight|bias)$", k)
        if m and int(m.group(1)) in tail_map:
            remapped[f"{tail_map[int(m.group(1))]}.{m.group(2)}"] = v
    return remapped


def is_rrdbnet_state_dict(state_dict: dict) -> bool:
    keys = state_dict.keys()
    return (
        any(k.startswith("body.0.rdb1.") for k in keys)
        or any(k.startswith("RRDB_trunk.0.RDB1.") for k in keys)
        or "model.1.sub.0.RDB1.conv1.0.weight" in keys
    )


def build_rrdbnet(state_dict: dict) -> tuple[nn.Module, dict]:
    """Infers num_feat/num_block/num_grow_ch/scale/channels directly from the
    checkpoint's own tensor shapes -- no per-file metadata needed."""
    if "model.0.weight" in state_dict and "model.1.sub.0.RDB1.conv1.0.weight" in state_dict:
        state_dict = _remap_sequential_esrgan_keys(state_dict)
    elif any(k.startswith("RRDB_trunk.") for k in state_dict):
        state_dict = _remap_old_esrgan_keys(state_dict)

    num_block = 0
    while f"body.{num_block}.rdb1.conv1.weight" in state_dict:
        num_block += 1
    if num_block == 0:
        raise ValueError("Not a recognized RRDBNet state_dict (no body.N.rdb1 blocks found)")

    num_feat = state_dict["conv_first.weight"].shape[0]
    num_grow_ch = state_dict["body.0.rdb1.conv1.weight"].shape[0]
    num_out_ch = state_dict["conv_last.weight"].shape[0]
    unshuffle_in_ch = state_dict["conv_first.weight"].shape[1]
    num_upconv = int("conv_up1.weight" in state_dict) + int("conv_up2.weight" in state_dict)

    if num_upconv < 2:
        # No pixel_unshuffle games here (see RRDBNet.__init__'s own note):
        # a checkpoint missing an upconv stage as a real layer has num_upconv
        # real 2x upsample stages and nothing else -- 0 (true same-
        # resolution, e.g. a 1x deblur/refocus model) or 1 (a plain 2x, no
        # unshuffle trick).
        scale, num_in_ch = (1 if num_upconv == 0 else 2), unshuffle_in_ch
        size_multiple = 1
    else:
        # unshuffle_in_ch = num_in_ch * scale^2 for scale in {1, 2}, else == num_in_ch (scale=4).
        if unshuffle_in_ch == num_out_ch:
            scale, num_in_ch = 4, unshuffle_in_ch
        elif unshuffle_in_ch == num_out_ch * 4:
            scale, num_in_ch = 2, num_out_ch
        elif unshuffle_in_ch == num_out_ch * 16:
            scale, num_in_ch = 1, num_out_ch
        else:
            scale, num_in_ch = 4, unshuffle_in_ch
        # pixel_unshuffle(scale=2) needs H,W divisible by 2; scale=1 by 4;
        # scale=4 skips unshuffle entirely (fully convolutional, no constraint).
        size_multiple = {4: 1, 2: 2, 1: 4}[scale]

    model = RRDBNet(
        num_in_ch=num_in_ch, num_out_ch=num_out_ch, scale=scale,
        num_feat=num_feat, num_block=num_block, num_grow_ch=num_grow_ch, num_upconv=num_upconv,
    )
    model.load_state_dict(state_dict, strict=True)
    meta = {
        "architecture": "RRDBNet", "scale": scale, "num_block": num_block, "num_feat": num_feat,
        "in_channels": num_in_ch, "out_channels": num_out_ch, "size_multiple": size_multiple,
    }
    return model, meta


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network with a PixelShuffle upsampler, used by
    Real-ESRGAN's fast "general"/"animevideo" models."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
    ) -> None:
        super().__init__()
        self.num_conv = num_conv
        self.upscale = upscale

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


def is_srvgg_state_dict(state_dict: dict) -> bool:
    return "body.0.weight" in state_dict and "body.1.weight" in state_dict and state_dict["body.1.weight"].ndim == 1


def build_srvgg(state_dict: dict) -> tuple[nn.Module, dict]:
    num_feat = state_dict["body.0.weight"].shape[0]
    num_in_ch = state_dict["body.0.weight"].shape[1]
    # body indices: 0=first conv,1=act, [2=conv,3=act]*num_conv, last conv at 2+2*num_conv
    last_idx = max(int(k.split(".")[1]) for k in state_dict if k.startswith("body."))
    num_conv = (last_idx - 2) // 2
    last_conv_out = state_dict[f"body.{last_idx}.weight"].shape[0]
    # last_conv_out = num_out_ch * upscale^2; Real-ESRGAN's compact models are all RGB-in/out.
    num_out_ch = num_in_ch
    upscale_sq = last_conv_out // num_out_ch
    upscale = round(upscale_sq**0.5)

    model = SRVGGNetCompact(
        num_in_ch=num_in_ch, num_out_ch=num_out_ch, num_feat=num_feat, num_conv=num_conv, upscale=upscale,
    )
    model.load_state_dict(state_dict, strict=True)
    meta = {
        "architecture": "SRVGGNetCompact", "scale": upscale, "num_conv": num_conv, "num_feat": num_feat,
        "in_channels": num_in_ch, "out_channels": num_out_ch, "size_multiple": 1,
    }
    return model, meta
