# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Cascading percentage-weight bookkeeping + the blend-mode implementations.

Reused for both the final 4-column blend and each dual-demosaic column's
2-way weight (same widget/logic, just with 2 or 4 slots).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CascadingWeights:
    """N slots, each either "auto" (shares the leftover pool equally with
    other auto slots) or "explicit" (a value the user set directly).

    Matches the user's worked example: set slot 0 to 90% -> the other,
    still-auto slots split the remaining 10% evenly; then set slot 1 to a
    value -> slot 0 stays at its explicit 90%, remaining auto slots
    re-split whatever's left. (Simplification: "still auto" is determined
    by explicit/auto state, not left-to-right position -- setting slot 2
    while slot 1 is still auto redistributes the remainder across *all*
    other auto slots, not just ones to the right. Every example in the
    spec happens to only ever leave slots to the right un-set, so this
    produces identical results while being simpler and order-independent.)
    """

    n: int
    explicit: dict[int, float] = field(default_factory=dict)

    def set(self, index: int, value_percent: float) -> None:
        value_percent = max(0.0, min(100.0, value_percent))
        self.explicit[index] = value_percent
        total_explicit = sum(self.explicit.values())
        if total_explicit > 100.0 and len(self.explicit) > 1:
            # Scale down the *other* explicit slots proportionally so the
            # just-set one keeps its requested value and the total is 100.
            other_total = total_explicit - value_percent
            room = max(0.0, 100.0 - value_percent)
            if other_total > 0:
                scale = room / other_total
                for k in list(self.explicit.keys()):
                    if k != index:
                        self.explicit[k] *= scale

    def clear(self, index: int) -> None:
        self.explicit.pop(index, None)

    def reset(self) -> None:
        self.explicit.clear()

    def weights(self) -> list[float]:
        """Returns N fractions (0..1) summing to 1.0 (given at least one
        active slot; caller is responsible for excluding inactive columns
        by not including them in `n`/reads)."""
        auto_slots = [i for i in range(self.n) if i not in self.explicit]
        explicit_total = sum(self.explicit.get(i, 0.0) for i in range(self.n))
        remainder = max(0.0, 100.0 - explicit_total)
        share = remainder / len(auto_slots) if auto_slots else 0.0

        out = []
        for i in range(self.n):
            pct = self.explicit[i] if i in self.explicit else share
            out.append(pct)
        total = sum(out) or 1.0
        return [p / total for p in out]


def weighted_mean(images: list[np.ndarray], weights: list[float]) -> np.ndarray:
    stack = np.stack(images, axis=0).astype(np.float32)
    w = np.asarray(weights, dtype=np.float32).reshape(-1, *([1] * (stack.ndim - 1)))
    return np.sum(stack * w, axis=0)


def weighted_median(images: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Per-pixel weighted median: the value at which cumulative weight
    (values sorted ascending) first reaches 50%."""
    stack = np.stack(images, axis=0).astype(np.float32)  # (N, H, W, C)
    w = np.asarray(weights, dtype=np.float32)
    order = np.argsort(stack, axis=0)
    sorted_vals = np.take_along_axis(stack, order, axis=0)
    w_broadcast = w.reshape(-1, *([1] * (stack.ndim - 1)))
    sorted_weights = np.take_along_axis(
        np.broadcast_to(w_broadcast, stack.shape), order, axis=0
    )
    cum_weights = np.cumsum(sorted_weights, axis=0)
    half = cum_weights[-1] * 0.5
    mask = cum_weights >= half
    idx = np.argmax(mask, axis=0)
    return np.take_along_axis(sorted_vals, idx[None, ...], axis=0)[0]


def _local_detail_energy(image: np.ndarray, window: int = 5) -> np.ndarray:
    """Small-window Laplacian-energy map (per-pixel local high-frequency
    strength), used by the detail-adaptive blend mode."""
    gray = image.mean(axis=-1) if image.ndim == 3 else image
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from scipy.ndimage import convolve, uniform_filter

    lap = convolve(gray, kernel, mode="reflect")
    energy = uniform_filter(lap**2, size=window, mode="reflect")
    return energy


def detail_adaptive(images: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Blend where each image's per-pixel contribution is its global weight
    scaled by local detail (high-frequency) energy relative to the other
    images, renormalized to sum to 1 at every pixel. Favors whichever
    column resolved the most local detail in high-contrast regions, while
    the user's global percentages still set the overall balance.
    """
    energies = [_local_detail_energy(img) for img in images]
    eps = 1e-6
    stacked_energy = np.stack(energies, axis=0) + eps  # (N, H, W)
    w = np.asarray(weights, dtype=np.float32).reshape(-1, 1, 1)
    local_weights = stacked_energy * w
    local_weights /= local_weights.sum(axis=0, keepdims=True)

    stack = np.stack(images, axis=0).astype(np.float32)  # (N, H, W, C)
    lw = local_weights[..., None]  # (N, H, W, 1)
    return np.sum(stack * lw, axis=0)


BLEND_MODES = {
    "Weighted Mean": weighted_mean,
    "Weighted Median": weighted_median,
    "Detail-adaptive": detail_adaptive,
}
