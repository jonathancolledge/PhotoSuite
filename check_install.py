# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Final check run by install_windows.ps1 and install_linux.sh: every app's modules import, which devices the
models can use, and which models are in the shared models folder. Exit code 1 if something can't be imported."""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, 'models')
ok = '    OK ' if os.name == 'nt' else '    ✓ '
warn = '    !  '
failed = False


def app(folder, modules):
    """Import an app's modules the way it runs: from its own folder."""
    global failed
    d = os.path.join(HERE, folder)
    sys.path.insert(0, d)
    try:
        for m in modules: importlib.import_module(m)
        print(f'{ok}{folder}: {", ".join(modules)}')
    except Exception as ex:
        failed = True
        print(f'{warn}{folder}: {type(ex).__name__}: {ex}')
    finally:
        sys.path.remove(d)


import torch  # noqa: E402 — first, so ONNX Runtime finds its CUDA libraries
import onnxruntime as ort  # noqa: E402
try:
    ort.preload_dlls()
except Exception:
    pass
import transformers  # noqa: E402
print(f'{ok}PyTorch {torch.__version__} ({"CUDA: " + ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) if torch.cuda.is_available() else "CPU"}), '
      f'ONNX Runtime {ort.__version__} ({"GPU" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPU"}), transformers {transformers.__version__}')

app(os.path.join('PhotoLibrary', 'desktop'), ['photolibrary_catalog', 'photolibrary_ai', 'photolibrary_setup_models'])
app(os.path.join('RAWDemosaic', 'desktop'), ['rawdemosaic_engine', 'rawdemosaic_develop'])
app(os.path.join('panoramamaker', 'desktop'), ['panostitch_engine', 'panostitch_hugin'])
for m in ('rawpy', 'numba', 'cv2', 'maxflow', 'sam2', 'RawForge', 'exifread', 'tkinter'):
    try:
        importlib.import_module(m)
    except Exception as ex:
        failed = True
        print(f'{warn}{m}: {ex}')
try:
    import bm3d  # noqa: F401 — optional (non-commercial licence)
except ImportError:
    print(f'{warn}BM3D not installed (optional): RAW Demosaic\'s BM3D option is unavailable')

# models
sys.path.insert(0, os.path.join(HERE, 'PhotoLibrary', 'desktop'))
import photolibrary_setup_models as SM  # noqa: E402
have = SM.present(os.path.join(MODELS, 'PhotoLibrary'))
missing = [k for k, v in have.items() if not v]
print((warn + 'Photo Library AI models missing: ' + ', '.join(missing) + ' (Settings & AI -> Download missing models)') if missing
      else ok + 'Photo Library AI models')


def count(d, ext):
    n = 0
    for _, _, fs in os.walk(d):
        n += sum(f.endswith(ext) for f in fs)
    return n


rd = os.path.join(MODELS, 'RAWDemosaic')
n = count(rd, ('.onnx', '.pt', '.pth'))
print(f'{ok}RAW Demosaic models: {n} files' if n else f'{warn}no RAW Demosaic models in {rd} (run RAWDemosaic/desktop/setup_models.py)')
pm = os.path.join(MODELS, 'panoramamaker')
n = count(pm, '.onnx')
print(f'{ok}PANOSTITCH ALIKED models: {n}' if n else f'{warn}no ALIKED models in {pm} — PANOSTITCH falls back to SIFT')
sys.exit(1 if failed else 0)
