# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Single entry point for the post-processing model picker: point it at any
.pth checkpoint from PostProcessingModels (or elsewhere) and it identifies
which of the seven supported architectures it is from the checkpoint's own
tensor names/shapes (no per-file metadata needed), builds and loads it, and
runs it tiled over an arbitrary-sized image.

Detection order matters only in that each is_XXX_state_dict() check must be
specific enough not to false-positive on another architecture -- verified
against every checkpoint in PostProcessingModels during development (see
each module's own docstring for what was checked).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import dat, deblurganv2, nafnet, nafssr, restormer, rrdbnet, swin2sr
try:  # MPRNet's architecture is under its own Academic Public License, so it isn't part of the GPL release
    from . import mprnet
except ImportError:
    mprnet = None
from ..inference import get_device, tiled_inference_multi_pass, tiled_inference_pad_crop

_sessions: dict[str, tuple[torch.nn.Module, dict]] = {}


class UnknownArchitectureError(RuntimeError):
    pass


def _unwrap_state_dict(ckpt) -> dict:
    if not isinstance(ckpt, dict):
        raise UnknownArchitectureError("Checkpoint is not a state_dict-shaped file")
    for key in ("params_ema", "params", "state_dict", "model", "net", "netG", "g", "model_state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    # Already a flat state_dict (an OrderedDict of tensors)?
    if all(hasattr(v, "shape") for v in ckpt.values()):
        return ckpt
    raise UnknownArchitectureError("Could not find a state_dict inside this checkpoint")


# tile_size/tile_pad tuned per architecture now that tiling uses pad-and-crop
# (see inference.tiled_inference_pad_crop_np), not feather-blending: no
# averaging-blur cost to worry about, so tiles are sized generously (bounded
# mainly by VRAM) to make tiling itself rare. Restormer/MPRNet/NAFNet/
# DeblurGANv2 default to _NO_TILE (bigger than any realistic single photo's
# longest side, so ceil(dim/tile_size) == 1 and the "tiler" runs the whole
# image in one pass) because each has a module whose statistics are pooled
# over the WHOLE tile rather than being purely per-pixel/local, so any tile
# grid at all means visible per-tile inconsistency ("patches") that padding
# context alone doesn't fully hide:
#   * Restormer's MDTA attention and MPRNet's CALayer both explicitly pool
#     over the full spatial extent (channel-attention / global-avg-pool).
#   * NAFNet's "Simplified Channel Attention" is also a global average pool
#     (nn.AdaptiveAvgPool2d(1)) inside every NAFBlock.
#   * DeblurGANv2's FPN head uses InstanceNorm2d, which by definition
#     normalizes using the CURRENT input's own full-spatial-extent
#     statistics -- and it runs in train mode here (see deblurganv2.py), so
#     it actually computes them per call rather than using frozen stats.
#   * DAT's Adaptive_Spatial_Attention/Adaptive_Channel_Attention blocks
#     both have an AIM channel_interaction gate built on
#     nn.AdaptiveAvgPool2d(1), same per-tile-global-pooling issue as NAFNet.
# Avoiding tiling entirely is the real fix, matching Restormer's own
# reference scripts, which never tile at all even on 1280x720 images.
# run_model() below catches CUDA OOM and halves the tile size until it
# fits (and uses autocast to raise how much fits), so this is safe for
# genuinely huge images too -- RRDBNet/SRVGGNetCompact/Swin2SR don't need
# it: they're fully convolutional / local-window-only, with no operation
# anywhere that aggregates over more than a fixed local neighbourhood.
# "seam_prone": True marks exactly this set, so run_model() knows that if
# even the whole-image attempt doesn't fit, it must switch to the
# multi-pass offset-blend tiler (see inference.tiled_inference_multi_pass)
# rather than naively shrinking to a single small-tile grid, which would
# reintroduce the very seams _NO_TILE exists to avoid.
_NO_TILE = 8192  # exceeds the long side of any realistic single-frame photo

# "use_amp": False marks architectures whose tiles are already small enough
# (512/256px) that autocast's memory savings are never needed -- there's no
# reason to pay any of fp16's precision cost for them, so they always run
# fp32. DAT is here too, but for a different reason: it forces fp32
# internally regardless (see dat.py's DAT.forward -- fp16 caused a real
# overflow-to-NaN bug there), so marking it here just avoids entering a
# pointless outer autocast context that DAT immediately overrides anyway.
_DETECTORS = [
    (rrdbnet.is_rrdbnet_state_dict, rrdbnet.build_rrdbnet, {"value_range": "0_1", "tile_size": 512, "tile_pad": 16, "use_amp": False}),
    (rrdbnet.is_srvgg_state_dict, rrdbnet.build_srvgg, {"value_range": "0_1", "tile_size": 512, "tile_pad": 16, "use_amp": False}),
    # Checked before plain NAFNet: NAFSSR's body.N.module.blk./.fusion. keys
    # never overlap with NAFNet's encoders/decoders/middle_blks structure,
    # but grouping them together here keeps the two NAFBlock-based entries
    # next to each other in the list.
    (nafssr.is_nafssr_state_dict, nafssr.build_nafssr, {"value_range": "0_1", "tile_size": _NO_TILE, "tile_pad": 16, "seam_prone": True}),
    (nafnet.is_nafnet_state_dict, nafnet.build_nafnet, {"value_range": "0_1", "tile_size": _NO_TILE, "tile_pad": 16, "seam_prone": True}),
    (restormer.is_restormer_state_dict, restormer.build_restormer, {"value_range": "0_1", "tile_size": _NO_TILE, "tile_pad": 32, "seam_prone": True}),
    *([(mprnet.is_mprnet_state_dict, mprnet.build_mprnet, {"value_range": "0_1", "tile_size": _NO_TILE, "tile_pad": 32, "seam_prone": True})] if mprnet else []),
    (swin2sr.is_swin2sr_state_dict, swin2sr.build_swin2sr, {"value_range": "0_1", "tile_size": 256, "tile_pad": 16, "use_amp": False}),
    (deblurganv2.is_deblurganv2_state_dict, deblurganv2.build_deblurganv2, {"value_range": "-1_1", "tile_size": _NO_TILE, "tile_pad": 32, "seam_prone": True}),
    (dat.is_dat_state_dict, dat.build_dat, {"value_range": "0_1", "tile_size": _NO_TILE, "tile_pad": 32, "seam_prone": True, "use_amp": False}),
]


def load_model(path: str | Path) -> tuple[torch.nn.Module, dict]:
    """Returns (model, meta). meta always has: architecture, value_range
    ("0_1" or "-1_1"), tile_size, tile_pad, size_multiple, in_channels,
    out_channels, scale, inference_mode ("eval" or "train")."""
    path = str(path)
    if path in _sessions:
        return _sessions[path]

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_state_dict(ckpt)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

    for is_match, builder, defaults in _DETECTORS:
        if is_match(state_dict):
            model, arch_meta = builder(state_dict)
            if arch_meta.get("in_channels") not in (1, 3, 6):
                raise UnknownArchitectureError(
                    f"'{Path(path).name}' ({arch_meta['architecture']}) needs "
                    f"{arch_meta.get('in_channels')}-channel input, which this single-image tool can't provide."
                )
            device = get_device()
            model = model.to(device)
            # Some GANs (DeblurGANv2) need train-mode norm-layer behaviour
            # even at inference -- see deblurganv2.py's own comment on this.
            model.train(arch_meta.get("inference_mode") == "train")
            meta = {**defaults, **arch_meta}
            meta.setdefault("scale", 1)
            meta.setdefault("size_multiple", 1)
            _sessions[path] = (model, meta)
            return model, meta

    raise UnknownArchitectureError(
        f"'{Path(path).name}' doesn't match any supported architecture "
        "(RealESRGAN/ESRGAN RRDBNet, SRVGGNetCompact, NAFNet, Restormer, MPRNet, Swin2SR, DeblurGANv2-FPNInception, DAT)."
    )


def run_model(
    model: torch.nn.Module,
    meta: dict,
    image: np.ndarray,
    *,
    tile_size: int | None = None,
    tile_pad: int | None = None,
    avoid_mixed_precision: bool = False,
) -> np.ndarray:
    """image: HxWx3 float32 in [0, 1]. Returns HxWx3 float32 in [0, 1],
    upscaled by meta['scale'] if the model is a super-resolution model.

    avoid_mixed_precision: skip the fp16 autocast attempt entirely (see
    run_tile below) even for architectures that would otherwise use it as a
    memory-saving fallback on OOM. Doesn't affect architectures that never
    use it anyway (meta["use_amp"] is False)."""
    tile_size = tile_size or meta["tile_size"]
    tile_pad = tile_pad if tile_pad is not None else meta["tile_pad"]
    in_channels = meta["in_channels"]
    out_channels = meta["out_channels"]
    scale = meta.get("scale", 1)
    size_multiple = meta.get("size_multiple", 1)
    min_tile_size = meta.get("min_tile_size", 0)
    seam_prone = meta.get("seam_prone", False)

    work = image
    if in_channels == 1 and image.shape[-1] == 3:
        # e.g. Swin2SR's grayscale JPEG-artifact-removal variant.
        work = (image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114)[..., None]
    elif in_channels == 6 and image.shape[-1] == 3:
        # A stereo (NAFSSR) or dual-pixel (Restormer's dual_pixel_task)
        # model with only a single real image available: feed it as both
        # "views" concatenated on the channel axis. For NAFSSR specifically
        # this makes its cross-view attention degrade to attending over a
        # mirror of itself rather than exploiting real parallax -- not as
        # good as genuine stereo input, but a reasonable, crash-free way to
        # run a stereo-only checkpoint on one image.
        work = np.concatenate([image, image], axis=-1)

    if meta["value_range"] == "-1_1":
        work = work * 2.0 - 1.0

    device = next(model.parameters()).device
    use_amp = device.type == "cuda" and meta.get("use_amp", True) and not avoid_mixed_precision

    def run_tile(tensor: torch.Tensor) -> torch.Tensor:
        # fp32 first, always: it's both more accurate and (per the DAT
        # overflow bug this project hit) sometimes the only numerically
        # safe option, and for the small-tile architectures above
        # (use_amp False) it never even gets close to needing fp16's
        # memory savings. Only escalate to autocast fp16 -- which roughly
        # halves the memory footprint of the big dense/conv/feed-forward
        # activations (the actual OOM culprit: e.g. Restormer's GDFN
        # feed-forward expansion at full resolution needs ~10GB for one
        # intermediate tensor on a 10MP image) -- if the fp32 attempt at
        # this exact tile size actually doesn't fit.
        if not use_amp:
            return model(tensor)
        try:
            return model(tensor)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return model(tensor).float()

    def run_tile_fp32_only(tensor: torch.Tensor) -> torch.Tensor:
        return model(tensor)

    # Defaults bias toward "whole image, one pass, zero tiling seams" (see
    # _NO_TILE above) -- clamp to the image's own longest side first so we
    # don't waste an attempt (and its OOM) at a size that's already
    # equivalent to a bigger one already tried. For a genuinely huge image
    # that still doesn't fit in VRAM at that size (fp32, then fp16 above),
    # halve and retry rather than crash, down to whatever floor the
    # architecture actually needs.
    h, w = work.shape[:2]
    attempt_tile_size = min(tile_size, max(h, w))
    floor = max(min_tile_size, 256)
    multi_pass = False  # set once a seam-prone whole-image attempt OOMs
    while True:
        try:
            if multi_pass:
                out = tiled_inference_multi_pass(
                    run_tile_fp32_only, work.astype(np.float32),
                    tile_size=attempt_tile_size, tile_pad=tile_pad, size_multiple=size_multiple,
                    min_tile_size=min_tile_size, out_channels=out_channels, out_scale=scale,
                )
            else:
                out = tiled_inference_pad_crop(
                    run_tile, work.astype(np.float32),
                    tile_size=attempt_tile_size, tile_pad=tile_pad, size_multiple=size_multiple,
                    min_tile_size=min_tile_size, out_channels=out_channels, out_scale=scale,
                )
            break
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if seam_prone and not multi_pass:
                # The whole-image attempt (fp32, then fp16 if allowed)
                # didn't fit at all. Naively shrinking to a single small-
                # tile grid from here would reintroduce the per-tile seams
                # _NO_TILE exists to avoid -- switch to the multi-pass
                # offset-blend tiler instead, which hides seams by
                # combining several small-tile (always fp32; there's no
                # fp16 memory saving left worth trading away once we're
                # deliberately using small tiles) passes at shifted grid
                # positions. Try it at a generously-sized tile first (the
                # per-tile memory footprint is what matters now, not the
                # whole image's), falling through to the shrink-and-retry
                # below if even that doesn't fit.
                multi_pass = True
                attempt_tile_size = min(attempt_tile_size, 1024)
                continue
            if attempt_tile_size <= floor:
                raise
            attempt_tile_size = max(floor, attempt_tile_size // 2)

    if meta["value_range"] == "-1_1":
        out = (out + 1.0) / 2.0

    if out.shape[-1] == 1 and image.shape[-1] == 3:
        out = np.repeat(out, 3, axis=-1)
    elif out.shape[-1] == 6:
        # NAFSSR-style: average the two (left, right) super-resolved views
        # back into one image -- with an identical duplicated input, both
        # views are already nearly identical, so this is just a mild,
        # sensible way to combine them rather than arbitrarily picking one.
        out = (out[..., :3] + out[..., 3:]) / 2.0

    return np.clip(out, 0.0, 1.0).astype(np.float32)
