#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
# Photo Suite — Linux installer for Photo Library, RAW Demosaic and PANOSTITCH.
#
# Creates ONE Python environment (.venv-linux, in this folder) for all three apps, checks the system bits it
# can't install itself (venv, tkinter), adds launchers and applications-menu entries. Re-running it is safe:
# it updates the environment in place. Photos, Photo Library's catalog, settings and models are never touched.
#
#   bash install_linux.sh                 GPU (NVIDIA/CUDA) if one is found, else CPU
#   bash install_linux.sh --cpu           CPU only (no CUDA packages)
#   bash install_linux.sh --models        also download the models (Photo Library's AI, RAW Demosaic's, PANOSTITCH's)
#   bash install_linux.sh --yes           accept BM3D's and the face models' non-commercial licences without asking
#   bash install_linux.sh --no-system     never offer to install system packages (just report them)
#   bash install_linux.sh --no-menu       no applications-menu entries
#   bash install_linux.sh --uninstall     remove the environment and the menu entries
#
# Needs: Python 3.11+ with venv and tkinter, and internet access for pip. For the GPU: an NVIDIA driver new
# enough for CUDA 13 (580 or later). The folder can be shared with Windows (e.g. WSL on /mnt/c): Windows uses
# .venv, Linux .venv-linux.
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
VENV="$HERE/.venv-linux"
APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
MODE=auto; SYSTEM=ask; MENU=yes; MODELS=no; YES=no
# name | launcher | comment
APPS=("Photo Library|photolibrary.sh|Photo catalog" "RAW Demosaic|rawdemosaic.sh|RAW processing" "PANOSTITCH|panostitch.sh|Panorama stitcher")

for a in "$@"; do
  case "$a" in
    --cpu) MODE=cpu ;;
    --models) MODELS=yes ;;
    --yes) YES=yes ;;
    --no-system) SYSTEM=no ;;
    --no-menu) MENU=no ;;
    --uninstall)
      rm -rf "$VENV" "$APPS_DIR"/photosuite-*.desktop
      echo "Removed the environment and menu entries. Photos, the catalog, settings and models are untouched."
      exit 0 ;;
    -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $a (try --help)"; exit 2 ;;
  esac
done

say()  { printf '\n\033[1;33m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '    \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mError:\033[0m %s\n' "$*" >&2; exit 1; }

# ── system packages we can't pip-install: how this distro names them
pkg_cmd() {   # $1 = what: venv | tk
  if command -v apt-get >/dev/null; then
    [ "$1" = venv ] && echo "sudo apt-get install -y python3-venv python3-dev" || echo "sudo apt-get install -y python3-tk"
  elif command -v dnf >/dev/null; then
    [ "$1" = venv ] && echo "sudo dnf install -y python3-devel" || echo "sudo dnf install -y python3-tkinter"
  elif command -v pacman >/dev/null; then
    [ "$1" = venv ] && echo "" || echo "sudo pacman -S --needed tk"
  elif command -v zypper >/dev/null; then
    [ "$1" = venv ] && echo "sudo zypper install -y python3-virtualenv python3-devel" || echo "sudo zypper install -y python3-tk"
  else
    echo ""
  fi
}
offer() {     # $1 = description, $2 = command
  local cmd="$2"
  if [ -z "$cmd" ]; then die "$1 is missing — please install it with your distribution's package manager and re-run."; fi
  if [ "$SYSTEM" = no ] || [ ! -t 0 ]; then
    die "$1 is missing. Install it with:  $cmd   — then re-run this script."
  fi
  read -r -p "    $1 is missing. Run '$cmd' now? [Y/n] " ans
  case "${ans:-y}" in [Yy]*) eval "$cmd" ;; *) die "$1 is needed. Install it with:  $cmd" ;; esac
}

say "Checking Python"
PY="$(command -v python3 || true)"
[ -n "$PY" ] || die "python3 not found — install Python 3.11 or newer."
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
  || die "Python 3.11 or newer is needed (found $("$PY" --version 2>&1))."
ok "$("$PY" --version)"
if ! "$PY" -c 'import venv, ensurepip' 2>/dev/null; then offer "Python's venv module" "$(pkg_cmd venv)"; fi
ok "venv"
if ! "$PY" -c 'import tkinter' 2>/dev/null; then offer "Python's tkinter (the window toolkit)" "$(pkg_cmd tk)"; fi
"$PY" -c 'import tkinter' 2>/dev/null || die "tkinter still can't be imported."
ok "tkinter $("$PY" -c 'import tkinter; print(tkinter.TkVersion)')"

say "Looking for an NVIDIA GPU"
GPU=no
if [ "$MODE" != cpu ] && command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  GPU=yes
  nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader | while IFS= read -r l; do ok "GPU $l"; done
  DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
  if [ "${DRV:-0}" -lt 580 ]; then
    warn "driver $DRV is older than 580 (CUDA 13): the GPU packages may not load — update the driver, or use --cpu"
  fi
else
  [ "$MODE" = cpu ] && ok "CPU only (--cpu)" || warn "no NVIDIA GPU/driver found — installing the CPU version (models will be slow)"
fi

say "Creating the Python environment in $VENV"
if [ -x "$VENV/bin/python" ]; then ok "exists — updating it"; else "$PY" -m venv "$VENV"; ok "created"; fi
VPY="$VENV/bin/python"
"$VPY" -m pip install --upgrade --quiet pip setuptools wheel

say "Installing PyTorch (the first time this downloads ~3 GB with the GPU libraries)"
if [ "$GPU" = yes ]; then
  "$VPY" -m pip install --upgrade --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu130
else
  "$VPY" -m pip install --upgrade --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
ok "$("$VPY" -c "import torch; print('PyTorch', torch.__version__, '- CUDA' if torch.cuda.is_available() else '- CPU')")"

say "Installing the other packages"
"$VPY" -m pip install --upgrade --quiet -r "$HERE/requirements.txt"
# SAM2 is a source package whose build imports torch; its optional CUDA extension needs a compiler — skipped
SAM2_BUILD_CUDA=0 "$VPY" -m pip install --upgrade --quiet --no-build-isolation sam2
ok "rawpy, Numba, OpenCV, transformers, RawForge, SAM2, PyMaxflow and the rest"
echo "    BM3D (RAW Demosaic's BM3D denoise option) is free for non-commercial use only — it is not open source."
ans=n
if [ "$YES" = yes ]; then ans=y; elif [ -t 0 ]; then read -r -p "    Install BM3D for non-commercial use? [Y/n] " ans; ans="${ans:-y}"; fi
case "$ans" in
  [Yy]*) "$VPY" -m pip install --upgrade --quiet bm3d; ok "BM3D" ;;
  *) warn "BM3D not installed: RAW Demosaic's BM3D option will not work (everything else does)" ;;
esac

# Does this ONNX Runtime's CUDA support actually load here? Run in a separate process: on some systems (seen
# under WSL2, Ubuntu 24.04) the CUDA 13 builds crash outright while loading their CUDA provider.
probe_gpu() {
  ( "$VPY" - "$HERE" <<'EOF' >/dev/null 2>&1; rc=$?; exit $rc ) 2>/dev/null
import os, sys, glob, ctypes
import torch
import onnxruntime as o
o.preload_dlls()
cap = os.path.join(os.path.dirname(o.__file__), 'capi')
ctypes.CDLL(os.path.join(cap, 'libonnxruntime_providers_shared.so'), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(cap, 'libonnxruntime_providers_cuda.so'))
models = sorted(glob.glob(os.path.join(sys.argv[1], 'models', 'panoramamaker', 'raco_aliked_lightglue_pipeline_k*.onnx')))
if models:
    import numpy as np
    s = o.InferenceSession(models[0], providers=['CUDAExecutionProvider'])
    assert s.get_providers()[0] == 'CUDAExecutionProvider'
    s.run(None, {'images': np.random.rand(2, 3, 256, 384).astype(np.float32)})
EOF
}
FALLBACK_ORT=1.26.0      # last release built for CUDA 12 — works where the CUDA 13 builds crash
if [ "$GPU" = yes ]; then
  "$VPY" -m pip uninstall --quiet -y onnxruntime >/dev/null 2>&1 || true      # the CPU build would shadow the GPU one
  "$VPY" -m pip install --upgrade --quiet "onnxruntime-gpu[cuda,cudnn]"
  if probe_gpu; then
    ok "ONNX Runtime $("$VPY" -c 'import onnxruntime as o; print(o.__version__)') (GPU)"
  else
    warn "ONNX Runtime $("$VPY" -c 'import onnxruntime as o; print(o.__version__)')'s CUDA support fails on this system — trying $FALLBACK_ORT (CUDA 12)"
    "$VPY" -m pip uninstall --quiet -y onnxruntime-gpu >/dev/null 2>&1 || true
    "$VPY" -m pip install --quiet "onnxruntime-gpu[cuda,cudnn]==$FALLBACK_ORT"
    if probe_gpu; then
      ok "ONNX Runtime $FALLBACK_ORT (GPU, CUDA 12) works here"
    else
      warn "no GPU build of ONNX Runtime works here — installing the CPU version (PyTorch models still use the GPU)"
      "$VPY" -m pip uninstall --quiet -y onnxruntime-gpu >/dev/null 2>&1 || true
      "$VPY" -m pip install --upgrade --quiet onnxruntime
    fi
  fi
else
  "$VPY" -m pip uninstall --quiet -y onnxruntime-gpu >/dev/null 2>&1 || true
  "$VPY" -m pip install --upgrade --quiet onnxruntime
  ok "ONNX Runtime (CPU)"
fi

if [ "$MODELS" = yes ]; then
  say "Downloading Photo Library's AI models (~6 GB)"
  PL_ARGS=(); [ "$YES" = yes ] && PL_ARGS=(--yes)      # --yes also accepts the face models' non-commercial licence
  "$VPY" "$HERE/PhotoLibrary/desktop/photolibrary_setup_models.py" "${PL_ARGS[@]}" || warn "the download failed — try again from Photo Library: Settings & AI → Download missing models"
  say "Downloading PANOSTITCH's ALIKED + LightGlue models (~450 MB)"
  "$VPY" "$HERE/panoramamaker/desktop/panostitch_setup_models.py" || warn "the download failed — PANOSTITCH matches with SIFT until the models are there"
  say "Fetching RAW Demosaic's models"
  "$VPY" "$HERE/RAWDemosaic/desktop/setup_models.py" || warn "RAW Demosaic's model download failed — run it again later"
fi

say "Checking the installation"
"$VPY" "$HERE/check_install.py" || die "the final check failed — see the message above"

say "Launchers"
for entry in "${APPS[@]}"; do
  IFS='|' read -r name launcher comment <<< "$entry"
  chmod +x "$HERE/$launcher" 2>/dev/null || true
  ok "$HERE/$launcher"
  if [ "$MENU" = yes ]; then
    mkdir -p "$APPS_DIR"
    f="$APPS_DIR/photosuite-${launcher%.sh}.desktop"
    cat > "$f" <<EOF
[Desktop Entry]
Type=Application
Name=$name
Comment=$comment
Exec=bash "$HERE/$launcher"
Path=$HERE
Icon=applications-graphics
Terminal=false
Categories=Graphics;Photography;
EOF
    ok "menu entry $f"
  fi
done

say "Done"
echo "    Start the apps from the applications menu, or with:  bash \"$HERE/photolibrary.sh\"  (rawdemosaic.sh, panostitch.sh)"
echo "    PANOSTITCH as Hugin's control point detector:  Program  $VPY"
echo "                                                   Arguments  \"$HERE/panoramamaker/desktop/panostitch_cpfind.py\" -o %o %s"
echo "    Uninstall:  bash \"$0\" --uninstall"
