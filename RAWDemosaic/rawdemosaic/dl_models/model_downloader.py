# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Fetches and caches the official NIND-denoise ONNX weights bundled by the
darktable-ai project (https://github.com/darktable-org/darktable-ai) so this
app can use real trained weights without the user hunting for checkpoints.

`.dtmodel` files are plain zip archives (verified by inspection) containing
a config.json plus one or more .onnx files -- no darktable installation is
needed to use them, they're loaded directly via onnxruntime.
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

from ..paths import MODELS_DIR as CACHE_DIR

_RELEASE = "https://github.com/darktable-org/darktable-ai/releases/download/release-5.6.0"

MODEL_SOURCES = {
    "denoise-nind": f"{_RELEASE}/denoise-nind.dtmodel",
    "rawdenoise-nind": f"{_RELEASE}/rawdenoise-nind.dtmodel",
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def ensure_model(model_id: str, progress_callback=None) -> Path:
    """Downloads + extracts the named model on first use. Returns the
    extracted directory (containing config.json and the .onnx file(s)).
    """
    if model_id not in MODEL_SOURCES:
        raise KeyError(f"Unknown model id: {model_id}")

    extract_dir = CACHE_DIR / model_id
    config_path = extract_dir / "config.json"
    if config_path.is_file():
        return extract_dir

    archive_path = CACHE_DIR / f"{model_id}.dtmodel"
    if not archive_path.is_file():
        if progress_callback:
            progress_callback(f"Downloading {model_id} (~55MB, one-time)...")
        _download(MODEL_SOURCES[model_id], archive_path)

    with zipfile.ZipFile(archive_path) as z:
        z.extractall(CACHE_DIR)

    if not config_path.is_file():
        raise RuntimeError(f"Extracted {model_id} but config.json is missing at {config_path}")
    return extract_dir


def load_config(model_dir: Path) -> dict:
    return json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
