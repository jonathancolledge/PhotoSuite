# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Makes float32 copies of RawForge's float16 ONNX models (models/rawforge/X.onnx ->
models/rawforge/X_fp32.onnx) for the browser version: in ONNX Runtime Web's
WebGPU backend these float16 models return all zeros (seen on an RTX 3090 and
in headless Chromium), so the page fell back to the CPU, while float32 models
run correctly on the GPU. The page uses an _fp32 copy when it finds one.

Weights, constants, casts and the graph's inputs/outputs are converted; each
copy is checked against the original on the CPU before it is kept.

    python -m rawdemosaic.dl_models.convert_fp32_onnx
"""

from __future__ import annotations

import sys

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from ..paths import MODELS_DIR

F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT


def _to_f32_tensor(t: onnx.TensorProto) -> onnx.TensorProto:
    if t.data_type != F16:
        return t
    return numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name)


def _convert_graph(g: onnx.GraphProto) -> None:
    for i, t in enumerate(g.initializer):
        if t.data_type == F16:
            g.initializer[i].CopyFrom(_to_f32_tensor(t))
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        tt = vi.type.tensor_type
        if tt.elem_type == F16:
            tt.elem_type = F32
    for n in g.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type == F16:
                a.t.CopyFrom(_to_f32_tensor(a.t))
            elif a.type == onnx.AttributeProto.TENSORS:
                for k, t in enumerate(a.tensors):
                    if t.data_type == F16:
                        a.tensors[k].CopyFrom(_to_f32_tensor(t))
            elif a.type == onnx.AttributeProto.GRAPH:
                _convert_graph(a.g)
            elif a.type == onnx.AttributeProto.GRAPHS:
                for sub in a.graphs:
                    _convert_graph(sub)
            if n.op_type == "Cast" and a.name == "to" and a.i == F16:
                a.i = F32
            if n.op_type in ("ConstantOfShape",) and a.name == "value" and a.t.data_type == F16:
                a.t.CopyFrom(_to_f32_tensor(a.t))


def convert(src, dst) -> float:
    import onnxruntime as ort

    m = onnx.load(str(src))
    _convert_graph(m.graph)
    tmp = dst.with_suffix(".onnx.part")
    onnx.save(m, str(tmp))

    a = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    n = 256
    yy, xx = np.mgrid[0:n, 0:n] / n
    x = np.stack([0.2 + 0.6 * xx, 0.3 + 0.4 * yy, 0.5 + 0.3 * np.sin(19 * xx) * np.cos(17 * yy)])[None]
    names = {i.name for i in a.get_inputs()}
    feeds16 = {"input": x.astype(np.float16)}
    feeds32 = {"input": x.astype(np.float32)}
    if "cond" in names:
        feeds16["cond"] = np.array([[0.25]], np.float16)
        feeds32["cond"] = np.array([[0.25]], np.float32)
    ref = a.run(None, feeds16)[0].astype(np.float32)
    got = b.run(None, feeds32)[0]
    err = float(np.abs(ref - got).max())
    if not np.isfinite(got).all() or err > 0.02:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"float32 copy of {src.name} does not match the original (max difference {err:.3g})")
    tmp.replace(dst)
    return err


def main() -> int:
    folder = MODELS_DIR / "rawforge"
    for src in sorted(folder.glob("*.onnx")):
        if src.stem.endswith("_fp32"):
            continue
        dst = src.with_name(src.stem + "_fp32.onnx")
        if dst.is_file():
            print(f"{dst.name}: already converted")
            continue
        print(f"{dst.name}: converted (max difference from the float16 original {convert(src, dst):.1e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
