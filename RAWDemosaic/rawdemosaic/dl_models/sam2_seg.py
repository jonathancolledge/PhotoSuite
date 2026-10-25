# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""SAM2 (Segment Anything 2) point/box-prompted image segmentation, for the
interactive mask editor in "Blend multiple files" mode (see
gui/mask_editor_dialog.py).

Uses the HuggingFace-hosted checkpoint via SAM2ImagePredictor.from_pretrained
(confirmed working: verified against the real facebook/sam2.1-hiera-large
repo, loads on CUDA, and a single positive point on a real test photo
produced a clean, accurate mask) rather than the git-clone + manual
checkpoint/config-file route the project README leads with -- avoids
managing checkpoint downloads and yaml config paths ourselves, and sidesteps
that README's "strongly recommended to use WSL" Windows caveat entirely: the
PyPI `sam2` package's own build doesn't attempt the optional custom CUDA
kernel (used for video multi-object post-processing, irrelevant to our
single-image use here) that caveat is actually about.

Two-phase usage matches SAM2's own designed interaction model: encoding an
image (ImageSession.__init__) is the expensive part (one full model forward
pass), so it happens once per image; each subsequent predict() call is the
cheap decoder-only step, fast enough to re-run on every point/box edit for
live interactive feedback.
"""

from __future__ import annotations

import numpy as np
import torch

from .inference import get_device

DEFAULT_CHECKPOINT = "facebook/sam2.1-hiera-large"

_predictor_sessions: dict[str, object] = {}


def _autocast_dtype(device: torch.device) -> torch.dtype:
    # bfloat16 is what SAM2's own README/examples use on CUDA; CPU has no
    # bfloat16 autocast kernel support to rely on here, so fall back to
    # plain fp32 (CPU is a correctness fallback for this feature, not the
    # expected path -- SAM2 is only practically interactive on GPU).
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def load_predictor(checkpoint: str = DEFAULT_CHECKPOINT):
    """Cached by checkpoint name/path, matching
    dl_models/restore/registry.py's own _sessions pattern."""
    if checkpoint in _predictor_sessions:
        return _predictor_sessions[checkpoint]

    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # device must go IN to from_pretrained, not be applied after: it forwards
    # **kwargs down to sam2's own build_sam2(..., device="cuda", ...), which
    # unconditionally does model.to(device) internally using that hardcoded
    # "cuda" default *before* from_pretrained ever returns -- confirmed via
    # a real crash on a GPU-less machine (RuntimeError: Found no NVIDIA
    # driver), moving the model again afterwards would be too late.
    predictor = SAM2ImagePredictor.from_pretrained(checkpoint, device=get_device())
    _predictor_sessions[checkpoint] = predictor
    return predictor


class ImageSession:
    """One image's encoded SAM2 state. Create once per target image (the
    encoder pass), then call predict() as many times as needed as the user
    adds/removes points or a box."""

    def __init__(self, image: np.ndarray, checkpoint: str = DEFAULT_CHECKPOINT) -> None:
        """image: HxWx3 uint8 RGB."""
        self._predictor = load_predictor(checkpoint)
        device = get_device()
        with torch.inference_mode(), torch.autocast(device.type, dtype=_autocast_dtype(device)):
            self._predictor.set_image(image)

    def predict(
        self, points: list[tuple[int, int, bool]], box: tuple[int, int, int, int] | None
    ) -> np.ndarray:
        """points: (x, y, is_positive) in image pixel coordinates. box:
        (x0, y0, x1, y1) or None. Returns the best-scoring binary mask
        (HxW bool) for this prompt set -- at least one point or a box is
        required (an empty prompt set has no defined SAM2 prediction; the
        caller is expected to treat a genuinely empty mask instance as an
        all-False mask without calling this at all, not to call it with
        nothing to prompt with).
        """
        if not points and box is None:
            raise ValueError("predict() needs at least one point or a box")

        point_coords = np.array([[p[0], p[1]] for p in points], dtype=np.float32) if points else None
        point_labels = np.array([1 if p[2] else 0 for p in points], dtype=np.int32) if points else None
        box_arr = np.array(box, dtype=np.float32) if box is not None else None

        # A single point is genuinely ambiguous (could mean "just this
        # part" vs. "the whole object" at several scales) -- SAM2's own
        # multimask_output=True returns its 3 best candidates at different
        # granularities for exactly that case, and the highest-scoring one
        # is picked here. With 2+ prompts (multiple points and/or a box)
        # the ambiguity is mostly resolved already, so a single combined
        # mask (multimask_output=False) is what SAM2 is designed to
        # produce well in that regime.
        multimask = box is None and len(points) <= 1

        device = get_device()
        with torch.inference_mode(), torch.autocast(device.type, dtype=_autocast_dtype(device)):
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords, point_labels=point_labels, box=box_arr,
                multimask_output=multimask,
            )
        best = masks[int(np.argmax(scores))] if multimask else masks[0]
        return best.astype(bool)
