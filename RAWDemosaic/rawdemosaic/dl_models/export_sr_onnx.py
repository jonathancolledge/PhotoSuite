# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Exports the RealSR checkpoints in PostProcessingModels/RealSR to ONNX in
models/sr, for the browser version (rawdemosaic.html), which can only run
ONNX models. RealSR (jixiaozhong/RealSR) publishes .pth files only.

The model is built by the desktop app's own loader (restore/registry.py), so
the exported network is exactly the one the desktop runs; each export is
checked against PyTorch before being kept. Already-exported files are skipped.

    python -m rawdemosaic.dl_models.export_sr_onnx
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from ..paths import MODELS_DIR, POST_PROCESSING_MODELS_DIR
from .restore import registry

CHECKPOINTS = {
    "DF2K.pth": "realsr_df2k_x4.onnx",
    "DF2K_JPEG.pth": "realsr_df2k_jpeg_x4.onnx",
    "DPED.pth": "realsr_dped_x4.onnx",
}


def export(src, dst) -> float:
    model, _ = registry.load_model(src)
    model = model.float().cpu().eval()
    tmp = dst.with_suffix(".onnx.part")
    torch.onnx.export(
        model, torch.rand(1, 3, 64, 64), str(tmp), opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "H", 3: "W"}}, dynamo=False,
    )
    import onnxruntime as ort

    yy, xx = np.mgrid[0:96, 0:120] / 96.0
    x = np.stack([0.2 + 0.6 * xx, 0.3 + 0.4 * yy, 0.5 + 0.3 * np.sin(9 * xx) * np.cos(7 * yy)])[None].astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    got = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    err = float(np.abs(ref - got).max())
    if err > 1e-3:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ONNX export of {src.name} does not match PyTorch (max difference {err:.2g})")
    tmp.replace(dst)
    return err


def main() -> int:
    out_dir = MODELS_DIR / "sr"
    out_dir.mkdir(parents=True, exist_ok=True)
    for pth, onnx_name in CHECKPOINTS.items():
        src, dst = POST_PROCESSING_MODELS_DIR / "RealSR" / pth, out_dir / onnx_name
        if dst.is_file():
            print(f"{onnx_name}: already exported")
        elif not src.is_file():
            print(f"{onnx_name}: skipped ({src} not found)")
        else:
            print(f"{onnx_name}: exported (max difference from PyTorch {export(src, dst):.1e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
