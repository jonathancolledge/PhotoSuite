# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Shared device selection + overlap-tiled inference for the DL denoisers.

Tiling keeps VRAM use bounded regardless of image size (a 45+MP raw would
otherwise blow past 24GB on some architectures at full resolution).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

_device: torch.device | None = None


def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _feather_window(size: int, overlap: int) -> np.ndarray:
    w = np.ones(size, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return w


def tiled_inference_np(
    run_tile: Callable[[np.ndarray], np.ndarray],
    image: np.ndarray,
    *,
    tile_size: int = 512,
    overlap: int = 32,
    out_channels: int | None = None,
    out_scale: int = 1,
) -> np.ndarray:
    """Backend-agnostic overlap-tiled inference: image is HxWxC float32,
    run_tile takes/returns a 1xCxhxw numpy array (any backend can wrap this --
    onnxruntime consumes/produces numpy directly; the torch-specific
    `tiled_inference` below is a thin wrapper over this for torch models).

    out_scale > 1 supports joint demosaic+upsample models (e.g. a packed
    half-resolution Bayer input producing a full-resolution RGB output):
    the output plane is `out_scale`x the input tile's spatial size.
    """
    h, w, c = image.shape
    out_c = out_channels or c
    step = tile_size - overlap

    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    n_tiles_h = max(1, -(-(h - overlap) // step)) if h > tile_size else 1
    n_tiles_w = max(1, -(-(w - overlap) // step)) if w > tile_size else 1
    needed_h = (n_tiles_h - 1) * step + tile_size if h > tile_size else tile_size
    needed_w = (n_tiles_w - 1) * step + tile_size if w > tile_size else tile_size
    pad_h = max(pad_h, needed_h - h)
    pad_w = max(pad_w, needed_w - w)

    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    ph, pw = padded.shape[:2]
    out_tile = tile_size * out_scale
    out_step = step * out_scale

    acc = np.zeros((ph * out_scale, pw * out_scale, out_c), dtype=np.float32)
    weight = np.zeros((ph * out_scale, pw * out_scale, 1), dtype=np.float32)

    win_h = _feather_window(out_tile, overlap * out_scale)
    win_w = _feather_window(out_tile, overlap * out_scale)
    win2d = (win_h[:, None] * win_w[None, :])[:, :, None]

    for ty in range(0, ph - tile_size + 1, step) if ph > tile_size else [0]:
        for tx in range(0, pw - tile_size + 1, step) if pw > tile_size else [0]:
            tile = padded[ty : ty + tile_size, tx : tx + tile_size, :]
            tensor = tile.transpose(2, 0, 1)[None].astype(np.float32)
            result = run_tile(tensor)
            result_np = result[0].transpose(1, 2, 0)
            oty, otx = ty * out_scale, tx * out_scale
            acc[oty : oty + out_tile, otx : otx + out_tile, :] += result_np * win2d
            weight[oty : oty + out_tile, otx : otx + out_tile, :] += win2d

    weight = np.maximum(weight, 1e-8)
    out = acc / weight
    return out[: h * out_scale, : w * out_scale, :]


@torch.no_grad()
def tiled_inference(
    run_tile: Callable[[torch.Tensor], torch.Tensor],
    image: np.ndarray,
    *,
    tile_size: int = 512,
    overlap: int = 32,
    out_channels: int | None = None,
    out_scale: int = 1,
) -> np.ndarray:
    """image: HxWxC float32 array. run_tile takes/returns a 1xCxhxw tensor
    already on the target device. Returns HxWxC' float32 array (C' defaults
    to C, override via out_channels for models that change channel count).
    out_scale > 1 for super-resolution models whose output is a multiple of
    the input tile's spatial size (e.g. 4x upscaling).
    """
    device = get_device()

    def run_tile_np(tile_np: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(tile_np).to(device)
        result = run_tile(tensor)
        return result.cpu().numpy()

    return tiled_inference_np(
        run_tile_np, image, tile_size=tile_size, overlap=overlap, out_channels=out_channels, out_scale=out_scale
    )


def tiled_inference_pad_crop_np(
    run_tile: Callable[[np.ndarray], np.ndarray],
    image: np.ndarray,
    *,
    tile_size: int = 256,
    tile_pad: int = 16,
    size_multiple: int = 1,
    min_tile_size: int = 0,
    out_channels: int | None = None,
    out_scale: int = 1,
) -> np.ndarray:
    """Real-ESRGAN's own tiling strategy (realesrgan/utils.py RealESRGANer.
    tile_process), used by dl_models/restore/registry.py instead of the
    feather-blend tiled_inference_np above: each tile gets `tile_pad` extra
    pixels of REAL surrounding image content on each side (not synthetic
    padding), the model runs on that padded tile, and only the tile's own
    exclusive, non-overlapping region is kept in the output -- no blending
    between adjacent tiles at all.

    This matters because averaging two independently-computed restoration/
    super-resolution outputs at a feathered seam (the other function's
    approach) blurs wherever their hallucinated fine detail doesn't line up
    pixel-for-pixel, which reads as an overall "soft" image once seams are
    frequent enough -- confirmed against Real-ESRGAN's reference tiler
    while diagnosing exactly that softness.

    size_multiple: some architectures need every individual forward-pass
    input divisible by a fixed factor (e.g. Restormer/MPRNet need a
    multiple of 8 -- they have no internal padding of their own -- and
    DeblurGANv2's Inception backbone needs a multiple of 32, per its own
    predict.py). An edge tile smaller than that (after adding tile_pad
    context) is reflect-padded up to it here and the extra cropped back off
    afterward. Architectures that already self-pad internally (NAFNet,
    Swin2SR) should just pass 1.

    min_tile_size: an absolute floor on top of size_multiple, for
    architectures where being a mere multiple of size_multiple still isn't
    enough -- e.g. DeblurGANv2's Inception stem uses many unpadded ("valid")
    convolutions that crop several pixels at almost every layer, so a small
    but technically-32-aligned tile (a pathological edge tile can end up
    tiny) can still shrink to a non-positive size partway through and crash.
    """
    h, w, c = image.shape
    out_c = out_channels or c
    output = np.zeros((h * out_scale, w * out_scale, out_c), dtype=np.float32)

    tiles_x = max(1, math.ceil(w / tile_size))
    tiles_y = max(1, math.ceil(h / tile_size))

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            input_start_x = tx * tile_size
            input_end_x = min(input_start_x + tile_size, w)
            input_start_y = ty * tile_size
            input_end_y = min(input_start_y + tile_size, h)

            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, w)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, h)

            tile = image[input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad, :]
            tile_h, tile_w = tile.shape[:2]

            target_h = max(tile_h, min_tile_size)
            target_h += (-target_h) % size_multiple
            target_w = max(tile_w, min_tile_size)
            target_w += (-target_w) % size_multiple
            extra_h, extra_w = target_h - tile_h, target_w - tile_w
            if extra_h or extra_w:
                # "edge" (not "reflect"): a pathologically small edge tile
                # can need padding larger than its own size, which reflect
                # mode can't do in one step; these pixels are just scaffold
                # to satisfy the model's minimum size and get cropped back
                # off below, so their exact content doesn't matter.
                tile = np.pad(tile, ((0, extra_h), (0, extra_w), (0, 0)), mode="edge")

            tensor = tile.transpose(2, 0, 1)[None].astype(np.float32)
            result = run_tile(tensor)
            result_np = result[0].transpose(1, 2, 0)
            # Drop the size_multiple padding (in output-scale units) before
            # extracting this tile's own region below.
            result_np = result_np[: tile_h * out_scale, : tile_w * out_scale, :]

            out_start_x_tile = (input_start_x - input_start_x_pad) * out_scale
            out_end_x_tile = out_start_x_tile + (input_end_x - input_start_x) * out_scale
            out_start_y_tile = (input_start_y - input_start_y_pad) * out_scale
            out_end_y_tile = out_start_y_tile + (input_end_y - input_start_y) * out_scale

            output[
                input_start_y * out_scale : input_end_y * out_scale,
                input_start_x * out_scale : input_end_x * out_scale,
                :,
            ] = result_np[out_start_y_tile:out_end_y_tile, out_start_x_tile:out_end_x_tile, :]

    return output


def _axis_edge_weight(size: int, tile_size: int, seam_offset: int, margin: int) -> np.ndarray:
    """1.0 in the interior of every tile_size-spaced grid cell along one
    axis, ramping down to 0.0 within `margin` pixels of a seam. A pass built
    by prepending `seam_offset` pixels of padding before tiling (then
    cropping that padding back off) has its seams at positions where
    (coord + seam_offset) % tile_size == 0 in the ORIGINAL, unpadded frame."""
    coord = np.arange(size, dtype=np.float32)
    pos_in_tile = (coord + seam_offset) % tile_size
    dist_to_seam = np.minimum(pos_in_tile, tile_size - pos_in_tile)
    return np.clip(dist_to_seam / max(margin, 1), 0.0, 1.0)


def tiled_inference_multi_pass_np(
    run_tile: Callable[[np.ndarray], np.ndarray],
    image: np.ndarray,
    *,
    tile_size: int = 512,
    tile_pad: int = 16,
    size_multiple: int = 1,
    min_tile_size: int = 0,
    out_channels: int | None = None,
    out_scale: int = 1,
    grid_shifts: int = 2,
    blend_margin: int | None = None,
) -> np.ndarray:
    """Runs tiled_inference_pad_crop_np across a grid_shifts x grid_shifts
    grid of tile-grid offsets (via reflect-padding the input by each
    (offset_y, offset_x) before tiling, then cropping that padding back off
    so every pass returns output in the same, original coordinate frame),
    and combines the passes with a per-pass, per-pixel weight that favours
    whichever pass placed that pixel furthest from one of ITS OWN tile
    seams -- both its horizontal seam row AND its vertical seam column.

    For architectures with per-tile global pooling (see registry.py's
    _NO_TILE comment), no single pass's pad-crop tiling is seam-free: each
    pass's seams form a full crosshatch of bad rows/columns, not isolated
    points. Shifting only ONE pass diagonally against another still leaves
    "blind spots" wherever one pass's bad ROW crosses the other pass's bad
    COLUMN (both simultaneously bad there, since horizontal and vertical
    lines always cross somewhere regardless of shift) -- confirmed by a
    synthetic seam-prone test that hit exactly this and produced a much
    worse discontinuity than no blending at all. Shifting y and x
    INDEPENDENTLY across a 2D grid_shifts x grid_shifts grid fixes this: for
    any point, at most one of the grid_shifts y-offsets can have it near a
    seam row (they're tile_size/grid_shifts apart, wider than the blend
    margin), and likewise for x-offsets independently, so at least one of
    the grid_shifts^2 (y-offset, x-offset) combinations is always good in
    BOTH axes at once. grid_shifts=1 degenerates to a single plain pad-crop
    pass (no blending, since there's nothing to blend against).

    This is the small-memory-footprint, fp32-only alternative to processing
    the whole image in one autocast fp16 pass, for images too big to fit
    either way: it costs grid_shifts^2 x the compute of a single pass, in
    exchange for every individual forward call only ever handling one
    small, fixed-size tile.
    """
    h, w, c = image.shape
    out_c = out_channels or c
    margin = tile_pad if blend_margin is None else blend_margin
    offsets = [round(k * tile_size / grid_shifts) for k in range(grid_shifts)]

    acc = np.zeros((h * out_scale, w * out_scale, out_c), dtype=np.float32)
    weight_sum = np.zeros((h * out_scale, w * out_scale, 1), dtype=np.float32)

    for offset_y in offsets:
        for offset_x in offsets:
            if offset_y == 0 and offset_x == 0:
                shifted = image
            else:
                shifted = np.pad(image, ((offset_y, 0), (offset_x, 0), (0, 0)), mode="reflect")
            pass_out = tiled_inference_pad_crop_np(
                run_tile, shifted, tile_size=tile_size, tile_pad=tile_pad, size_multiple=size_multiple,
                min_tile_size=min_tile_size, out_channels=out_channels, out_scale=out_scale,
            )
            oy, ox = offset_y * out_scale, offset_x * out_scale
            pass_out = pass_out[oy : oy + h * out_scale, ox : ox + w * out_scale, :]

            wh = _axis_edge_weight(h * out_scale, tile_size * out_scale, oy, margin * out_scale)
            ww = _axis_edge_weight(w * out_scale, tile_size * out_scale, ox, margin * out_scale)
            w_map = (wh[:, None] * ww[None, :])[:, :, None]
            acc += pass_out * w_map
            weight_sum += w_map

    weight_sum = np.maximum(weight_sum, 1e-8)
    return acc / weight_sum


@torch.no_grad()
def tiled_inference_multi_pass(
    run_tile: Callable[[torch.Tensor], torch.Tensor],
    image: np.ndarray,
    *,
    tile_size: int = 512,
    tile_pad: int = 16,
    size_multiple: int = 1,
    min_tile_size: int = 0,
    out_channels: int | None = None,
    out_scale: int = 1,
    grid_shifts: int = 2,
    blend_margin: int | None = None,
) -> np.ndarray:
    device = get_device()

    def run_tile_np(tile_np: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(tile_np).to(device)
        result = run_tile(tensor)
        return result.cpu().numpy()

    return tiled_inference_multi_pass_np(
        run_tile_np, image, tile_size=tile_size, tile_pad=tile_pad, size_multiple=size_multiple,
        min_tile_size=min_tile_size, out_channels=out_channels, out_scale=out_scale,
        grid_shifts=grid_shifts, blend_margin=blend_margin,
    )


@torch.no_grad()
def tiled_inference_pad_crop(
    run_tile: Callable[[torch.Tensor], torch.Tensor],
    image: np.ndarray,
    *,
    tile_size: int = 256,
    tile_pad: int = 16,
    size_multiple: int = 1,
    min_tile_size: int = 0,
    out_channels: int | None = None,
    out_scale: int = 1,
) -> np.ndarray:
    device = get_device()

    def run_tile_np(tile_np: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(tile_np).to(device)
        result = run_tile(tensor)
        return result.cpu().numpy()

    return tiled_inference_pad_crop_np(
        run_tile_np, image, tile_size=tile_size, tile_pad=tile_pad, size_multiple=size_multiple,
        min_tile_size=min_tile_size, out_channels=out_channels, out_scale=out_scale,
    )
