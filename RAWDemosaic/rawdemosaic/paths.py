# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Install-location-relative paths, so the app works correctly wherever it's
actually installed (not just in the folder it was developed in) -- e.g.
for install.ps1/install.sh (see scripts/) setting up a fresh install
elsewhere, on Windows or Linux, at whatever path the user chose.

APP_ROOT is derived from this file's own location rather than hardcoded, so
it's correct regardless of platform or install path: this file lives at
<app_root>/rawdemosaic/paths.py, so its grandparent directory is <app_root>.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = APP_ROOT / "models"  # NIND-Unet ONNX models (see dl_models/model_downloader.py)
if (APP_ROOT.parent / "suite.json").is_file():  # installed with the other desktop apps: their shared models folder
    MODELS_DIR = APP_ROOT.parent / "models" / "RAWDemosaic"
POST_PROCESSING_MODELS_DIR = APP_ROOT / "PostProcessingModels"  # restoration/super-res checkpoints
HF_CACHE_DIR = MODELS_DIR / "huggingface"  # SAM2 checkpoint cache (see dl_models/sam2_seg.py)

# Redirect huggingface_hub's cache into the app folder instead of the user's
# home directory, so SAM2's auto-downloaded checkpoint ends up under
# RAWDemosaic/models like every other model this app uses. Must happen before
# any huggingface_hub/sam2 import anywhere (both are always imported lazily),
# and setdefault() so an operator's own HF_HOME still wins if already set.
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
