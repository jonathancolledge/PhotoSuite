# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Exports RawForge's original float32 PyTorch (TorchScript) models to float32 ONNX for the browser
version (models/rawforge/X.pt -> models/rawforge/X_full.onnx).

RawForge's own ONNX release is float16. The _fp32 copies made by convert_fp32_onnx only change the
number format (needed on WebGPU), so their weights keep float16 precision. These exports carry the full
float32 weights the desktop app uses. Inputs and output are named as in RawForge's ONNX files
(input [N,3,H,W], cond [N,1] -> output; Restormer has no cond), with dynamic sizes. Each export is checked against PyTorch on
the CPU before it is kept.

    python -m rawdemosaic.dl_models.export_rawforge_fp32

The .pt files are read from models-desktop/rawforge (filled by desktop/setup_models.py), else from
RawForge's own download cache.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

from ..paths import APP_ROOT, MODELS_DIR

NAMES = ["ShadowWeightedL1_super_light", "ShadowWeightedL1_light", "ShadowWeightedL1", "ShadowWeightedL1_24_deep_500",
         "realblur_gamma_140", "Deblur_deep_24", "xtrans_fixed_exposure_no_conditioning_400", "restormer"]


def _sources() -> list[Path]:
    dirs = [APP_ROOT / "models-desktop" / "rawforge"]
    if os.name == "nt" and "LOCALAPPDATA" in os.environ:
        dirs.append(Path(os.environ["LOCALAPPDATA"]) / "RawForge" / "RawForge")
    dirs.append(Path.home() / ".local" / "share" / "RawForge")
    return dirs


def export(src: Path, dst: Path) -> float:
    import onnxruntime as ort
    import torch

    model = torch.jit.load(str(src), map_location="cpu").float().eval()
    x = torch.rand(1, 3, 256, 256) * 0.5
    cond = torch.tensor([[0.25]])
    tmp = dst.with_suffix(".part")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, (x, cond), str(tmp), input_names=["input", "cond"], output_names=["output"], opset_version=18,
                          dynamic_axes={"input": {0: "batch_size", 2: "height", 3: "width"}, "cond": {0: "batch_size"},
                                        "output": {0: "batch_size", 2: "height", 3: "width"}}, dynamo=False)
    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}   # an unused cond (Restormer) is dropped, as in RawForge's own ONNX
    worst = 0.0
    for h, w in ((256, 256), (192, 320)):
        xi = torch.rand(1, 3, h, w) * 0.5
        with torch.no_grad():
            ref = model(xi, cond=cond).numpy()
        feeds = {"input": xi.numpy(), "cond": cond.numpy()}
        out = sess.run(None, {k: v for k, v in feeds.items() if k in names})[0]
        worst = max(worst, float(np.abs(out - ref).max()))
    del sess
    if worst > 1e-3:
        tmp.unlink()
        raise RuntimeError(f"export differs from PyTorch by {worst:.2e}")
    os.replace(tmp, dst)
    return worst


def main() -> None:
    out_dir = MODELS_DIR / "rawforge"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in NAMES:
        dst = out_dir / f"{name}_full.onnx"
        if dst.exists():
            print(f"  have {dst.name}")
            continue
        src = next((d / f"{name}.pt" for d in _sources() if (d / f"{name}.pt").is_file()), None)
        if src is None:
            print(f"  missing {name}.pt (run desktop/setup_models.py first)")
            ok = False
            continue
        try:
            print(f"  {dst.name}: matches PyTorch to {export(src, dst):.1e}")
        except Exception as ex:
            ok = False
            print(f"  could not export {name}: {ex}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
