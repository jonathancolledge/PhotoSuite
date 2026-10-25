# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Download the models the browser versions use: RAWDemosaic/rawdemosaic.html and panoramamaker/panoramamaker.html.

    Windows:  .venv\\Scripts\\python.exe download_web_models.py [--mst] [--rawdemosaic | --panostitch]
    Linux:    .venv-linux/bin/python download_web_models.py [--mst] [--rawdemosaic | --panostitch]

Run it with the suite's environment (after install_windows.bat / install_linux.sh): a few of the models are
converted or exported here, which needs PyTorch and ONNX. Only what is missing is downloaded, so it can be run
again at any time. The desktop apps don't need any of this: their own models come from setup_models.py,
photolibrary_setup_models.py and panostitch_setup_models.py (or the installers' -Models / --models).

Where the files go, and what to choose as "models folder" in each page:

  models/RAWDemosaic     rawdemosaic.html    rawforge/  RawForge ONNX, their float32 copies (…_fp32.onnx) and, when
                                                        the desktop's PyTorch weights are there, float32 exports
                                                        (…_full.onnx)
                                             denoise-nind/, rawdenoise-nind/   NIND-Unet
                                             sr/        Real-ESRGAN ONNX; RealSR exported from its .pth checkpoints
                                             sam2/      SAM2.1 tiny, base-plus and large (ONNX)
                                             inpaint/   LaMa, MI-GAN; with --mst also MST and its HAWP line detector
                                             lensfun/   the Lensfun lens database (the page can also fetch it itself)
  models/panoramamaker   panoramamaker.html  RaCo-ALIKED + LightGlue+ ONNX (the same files the desktop app uses)

  --mst   also fetch the MST inpainting and HAWP models. They are only published inside large archives (MST's is
          5.7 GB), so the two files are streamed out of the download, which takes a long time.

Each model keeps its own licence; see THIRD_PARTY.md. All of these are open-source (MIT, BSD, Apache-2.0,
GPL-3.0, CC BY-SA 3.0); none is non-commercial.
"""
import json
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
RD_MODELS = os.path.join(HERE, 'models', 'RAWDemosaic')
PS_MODELS = os.path.join(HERE, 'models', 'panoramamaker')
HF = 'https://huggingface.co'

RAWFORGE_ONNX = 'https://github.com/rymuelle/RawForge/releases/download/onnx_v1.0.0/'
RAWFORGE = ['ShadowWeightedL1_super_light', 'ShadowWeightedL1_light', 'ShadowWeightedL1', 'ShadowWeightedL1_24_deep_500',
            'realblur_gamma_140', 'Deblur_deep_24', 'xtrans_fixed_exposure_no_conditioning_400', 'restormer']
SR = {'RealESR_Gx4_fp16.onnx': f'{HF}/svjack/AI-onnx/resolve/main/RealESR_Gx4_fp16.onnx',
      'real_esrgan_x2.onnx': f'{HF}/SceneWorks/real-esrgan-onnx/resolve/main/real_esrgan_x2.onnx',
      'real_esrgan_x4.onnx': f'{HF}/SceneWorks/real-esrgan-onnx/resolve/main/real_esrgan_x4.onnx'}
# RealSR publishes only .pth checkpoints: exported here. Names as RAW Demosaic's setup_models.py saves them.
REALSR = {'realsr_df2k_x4.onnx': 'RealSR_DF2K_x4.pth', 'realsr_df2k_jpeg_x4.onnx': 'RealSR_DF2K_JPEG_x4.pth',
          'realsr_dped_x4.onnx': 'RealSR_DPED_x4.pth'}
SAM2 = ['tiny', 'base-plus', 'large']
SAM2_FILES = ['vision_encoder_fp16.onnx', 'vision_encoder_fp16.onnx_data', 'prompt_encoder_mask_decoder.onnx',
              'prompt_encoder_mask_decoder.onnx_data']
INPAINT = {'lama_fp32.onnx': f'{HF}/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx',
           'migan_pipeline.onnx': f'{HF}/anyisalin/migan-onnx/resolve/main/onnx/migan_pipeline.onnx'}
PINTO = 'https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo'
MST = {'108_HAWP': ['hawp_512x512_float32_opt.onnx'], '163_MST_inpainting': ['mst_p2m_encoder_512x512.onnx', 'mst_p2m_decoder_512x512.onnx']}


def say(msg):
    print(msg, flush=True)


def fetch(url, dst):
    """Download url to dst unless it is already there (written to .part first, so a broken download is redone)."""
    if os.path.isfile(dst):
        say(f'  have {os.path.relpath(dst, HERE)}')
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    say(f'  downloading {os.path.relpath(dst, HERE)}')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'PhotoSuite'})
        with urllib.request.urlopen(req) as r, open(dst + '.part', 'wb') as f:
            shutil.copyfileobj(r, f, 1 << 20)
        os.replace(dst + '.part', dst)
        return True
    except Exception as ex:
        say(f'  ! could not download {os.path.basename(dst)}: {ex}')
        if os.path.exists(dst + '.part'): os.remove(dst + '.part')
        return False


def rawdemosaic_package():
    sys.path.insert(0, os.path.join(HERE, 'RAWDemosaic'))   # the rawdemosaic package; its paths follow suite.json


def nind():
    say('NIND-Unet (darktable-ai, GPL-3.0)')
    rawdemosaic_package()
    from rawdemosaic.dl_models import model_downloader as M
    for m in ('denoise-nind', 'rawdenoise-nind'):
        try:
            say(f'  {m}: {M.ensure_model(m)}')
        except Exception as ex:
            say(f'  ! {m}: {ex}')


def rawforge():
    say('RawForge ONNX (MIT)')
    d = os.path.join(RD_MODELS, 'rawforge')
    for n in RAWFORGE:
        fetch(RAWFORGE_ONNX + n + '.onnx', os.path.join(d, n + '.onnx'))
    rawdemosaic_package()
    from rawdemosaic.dl_models import convert_fp32_onnx as C, export_rawforge_fp32 as X
    say('  float32 copies (the float16 originals return zeros on WebGPU)')
    for n in RAWFORGE:
        src, dst = os.path.join(d, n + '.onnx'), os.path.join(d, n + '_fp32.onnx')
        if os.path.isfile(src) and not os.path.isfile(dst):
            say(f'  {os.path.basename(dst)}: max difference {C.convert(Path(src), Path(dst)):.1e}')
    say("  float32 exports of the original PyTorch weights (the desktop's, from setup_models.py)")
    cands = [d] + [str(p) for p in X._sources()]
    for n in RAWFORGE:
        dst = os.path.join(d, n + '_full.onnx')
        if os.path.isfile(dst): continue
        src = next((os.path.join(c, n + '.pt') for c in cands if os.path.isfile(os.path.join(c, n + '.pt'))), None)
        if not src:
            say(f'  {n}_full.onnx skipped: no {n}.pt (run RAWDemosaic/desktop/setup_models.py first); the page then uses the _fp32 copy')
            continue
        try:
            say(f'  {os.path.basename(dst)}: matches PyTorch to {X.export(Path(src), Path(dst)):.1e}')
        except Exception as ex:
            say(f'  ! {n}: {ex}')


def super_resolution():
    say('Super-resolution: Real-ESRGAN (BSD-3-Clause), RealSR (Apache-2.0)')
    d = os.path.join(RD_MODELS, 'sr')
    for name, url in SR.items():
        fetch(url, os.path.join(d, name))
    rawdemosaic_package()
    from rawdemosaic.dl_models import export_sr_onnx as X
    for onnx_name, pth in REALSR.items():
        dst = os.path.join(d, onnx_name)
        if os.path.isfile(dst): continue
        legacy = next(k for k, v in X.CHECKPOINTS.items() if v == onnx_name)
        src = next((p for p in (os.path.join(d, pth), os.path.join(str(X.POST_PROCESSING_MODELS_DIR), 'RealSR', legacy)) if os.path.isfile(p)), None)
        if not src:
            say(f'  {onnx_name} skipped: no {pth} (RAWDemosaic/desktop/setup_models.py fetches the DF2K one)')
            continue
        say(f'  {onnx_name}: exported (max difference from PyTorch {X.export(Path(src), Path(dst)):.1e})')


def sam2():
    say('SAM2.1 ONNX (Apache-2.0)')
    for v in SAM2:
        for f in SAM2_FILES:
            fetch(f'{HF}/onnx-community/sam2.1-hiera-{v}-ONNX/resolve/main/onnx/{f}', os.path.join(RD_MODELS, 'sam2', f'sam2.1-hiera-{v}', f))


def inpaint(mst):
    say('Inpainting: LaMa (Apache-2.0), MI-GAN (MIT)' + (', MST and HAWP (MIT, via PINTO model zoo)' if mst else ''))
    d = os.path.join(RD_MODELS, 'inpaint')
    os.makedirs(d, exist_ok=True)
    for name, url in INPAINT.items():
        fetch(url, os.path.join(d, name))
    if not mst:
        say('  MST / HAWP skipped (add --mst to stream them out of their large archives)')
        return
    for archive, names in MST.items():
        want = [n for n in names if not os.path.isfile(os.path.join(d, n))]
        if not want:
            say(f'  have {", ".join(names)}'); continue
        say(f'  streaming {", ".join(want)} out of {archive} (this can take a long time)')
        try:
            with urllib.request.urlopen(f'{PINTO}/{archive}/resources.tar.gz') as r, tarfile.open(fileobj=r, mode='r|gz') as tar:
                for member in tar:
                    base = os.path.basename(member.name)
                    if member.isfile() and base in want:
                        with tar.extractfile(member) as src, open(os.path.join(d, base + '.part'), 'wb') as dst:
                            shutil.copyfileobj(src, dst, 1 << 20)
                        os.replace(os.path.join(d, base + '.part'), os.path.join(d, base))
                        want.remove(base)
                        say(f'  got {base}')
                        if not want: break
        except Exception as ex:
            say(f'  ! {archive}: {ex}')
        for n in want: say(f'  ! {n} was not found in {archive}')


def lensfun():
    say('Lensfun lens database (CC BY-SA 3.0)')
    try:
        with urllib.request.urlopen('https://data.jsdelivr.com/v1/packages/gh/lensfun/lensfun@master?structure=flat') as r:
            names = [os.path.basename(f['name']) for f in json.load(r)['files'] if f['name'].startswith('/data/db/') and f['name'].endswith('.xml')]
    except Exception as ex:
        say(f'  ! could not list the database ({ex}); the page fetches it itself when online')
        return
    got = sum(fetch(f'https://raw.githubusercontent.com/lensfun/lensfun/master/data/db/{n}', os.path.join(RD_MODELS, 'lensfun', n))
              for n in names)
    say(f'  {got} of {len(names)} files')


def panostitch():
    say('PANOSTITCH: RaCo-ALIKED + LightGlue+ ONNX (Apache-2.0; ALIKED BSD-3-Clause)')
    sys.path.insert(0, os.path.join(HERE, 'panoramamaker', 'desktop'))
    import panostitch_setup_models as P
    P.download(PS_MODELS, log=lambda level, msg: say('  ' + msg))


def main():
    args = sys.argv[1:]
    if '-h' in args or '--help' in args:
        print(__doc__); return
    only_rd, only_ps = '--rawdemosaic' in args, '--panostitch' in args
    if not only_ps:
        say(f'RAW Demosaic web models -> {RD_MODELS}')
        for step in (nind, rawforge, super_resolution, sam2, lambda: inpaint('--mst' in args), lensfun):
            step()
    if not only_rd:
        say(f'PANOSTITCH web models -> {PS_MODELS}')
        panostitch()
    say('Done. In each page, choose its folder above as the models folder.')


if __name__ == '__main__':
    main()
