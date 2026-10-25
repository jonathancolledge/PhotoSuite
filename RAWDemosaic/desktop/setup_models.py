# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Fills models-desktop with the full-precision models RAW Demosaic desktop uses.

Files that already exist elsewhere in the RAWDemosaic folder (models\\, PostProcessingModels\\, the
Hugging Face cache) or in RawForge's own cache are hard-linked (no extra disk space; copied if a link is
not possible); anything still missing is downloaded. Safe to re-run.

    .venv\\Scripts\\python.exe desktop\\setup_models.py [--no-download]

models-desktop\\
    rawdenoise-nind\\model_bayer.onnx, model_linear.onnx   RAW NIND-Unet (float32)
    denoise-nind\\model.onnx                               NIND-Unet (float32)
    rawforge\\*.pt                                          RawForge, original float32 TorchScript weights
    sr\\*.pth                                               super-resolution (Real-ESRGAN, RealSR, Swin2SR, ...)
    restore\\*.pth                                          same-size restoration (MPRNet, NAFNet, Restormer, ...)
    sam2\\sam2.1_hiera_large.pt                             SAM2.1 (float32 checkpoint)
"""
import glob
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEST = os.path.join(ROOT, 'models-desktop')
if os.path.isfile(os.path.join(ROOT, '..', 'suite.json')):   # installed with the other desktop apps: the shared models folder
    DEST = os.path.normpath(os.path.join(ROOT, '..', 'models', 'RAWDemosaic'))
PPM = os.path.join(ROOT, 'PostProcessingModels')
# RawForge's own cache (platformdirs user_data_dir('RawForge')): %LOCALAPPDATA%\RawForge\RawForge, ~/.local/share/RawForge
RF_CACHE = (os.path.join(os.environ['LOCALAPPDATA'], 'RawForge', 'RawForge') if os.name == 'nt' and 'LOCALAPPDATA' in os.environ
            else os.path.expanduser('~/.local/share/RawForge'))

RF_URL = 'https://github.com/rymuelle/RawRefinery/releases/download/v1.2.1-alpha/'
RF_XT_URL = 'https://github.com/rymuelle/RawForge/releases/download/xtrans_v1.0.0/'
RAWFORGE = {f: RF_URL + f for f in ('ShadowWeightedL1_super_light.pt', 'ShadowWeightedL1_light.pt', 'ShadowWeightedL1.pt',
                                    'ShadowWeightedL1_24_deep_500.pt', 'realblur_gamma_140.pt', 'Deblur_deep_24.pt')}
RAWFORGE.update({f: RF_XT_URL + f for f in ('xtrans_fixed_exposure_no_conditioning_400.pt', 'restormer.pt')})

# super-resolution checkpoints: (file name in models-desktop\sr, names to look for, download URL or None)
SR = [
    ('RealESRGAN_x2plus.pth', ['RealESRGAN_x2plus.pth'], 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'),
    ('RealESRGAN_x4plus.pth', ['RealESRGAN_x4plus.pth'], 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'),
    ('realesr-general-x4v3.pth', ['realesr-general-x4v3.pth'], 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'),
    ('RealESRNet_x4plus.pth', ['RealESRNet_x4plus.pth'], 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'),
    ('RealSR_DF2K_x4.pth', ['RealSR/DF2K.pth'], 'https://drive.google.com/uc?export=download&id=1pWGfSw-UxOkrtbh14GeLQgYnMLdLguOF'),
    ('RealSR_DF2K_JPEG_x4.pth', ['RealSR/DF2K_JPEG.pth'], None),
    ('RealSR_DPED_x4.pth', ['RealSR/DPED.pth'], None),
    ('Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth', ['Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth'], 'https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth'),
    ('Swin2SR_ClassicalSR_X2_64.pth', ['Swin2SR_ClassicalSR_X2_64.pth'], None),
    ('Swin2SR_ClassicalSR_X4_64.pth', ['Swin2SR_ClassicalSR_X4_64.pth'], None),
    ('Swin2SR_CompressedSR_X4_48.pth', ['Swin2SR_CompressedSR_X4_48.pth'], None),
    ('Swin2SR_Lightweight_X2_64.pth', ['Swin2SR_Lightweight_X2_64.pth'], None),
    ('ESRGAN_SRx4_DF2KOST_official.pth', ['ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'], None),
    ('4x-UltraSharpV2.pth', ['4x-UltraSharpV2.pth'], None),
    ('4xUltrasharp_4xUltrasharpV10.pth', ['4xUltrasharp_4xUltrasharpV10.pth'], None),
    ('4x_foolhardy_Remacri.pth', ['4x_foolhardy_Remacri.pth'], None),
    ('4xRealWebPhoto_v4_dat2.pth', ['4xRealWebPhoto_v4_dat2.pth'], None),
]
RESTORE = [
    ('MPRNet_deblurring.pth', ['MPRNet/model_deblurring.pth']),
    ('MPRNet_denoising.pth', ['MPRNet/model_denoising.pth']),
    ('MPRNet_deraining.pth', ['MPRNet/model_deraining.pth']),
    ('NAFNet-GoPro-width64.pth', ['NAFNet-GoPro-width64.pth']),
    ('NAFNet-REDS-width64.pth', ['NAFNet-REDS-width64.pth']),
    ('NAFNet-SIDD-width64.pth', ['NAFNet-SIDD-width64.pth']),
    ('NAFSSR-L_2x.pth', ['NAFSSR-L_2x.pth']),
    ('Restormer_single_image_defocus_deblurring.pth', ['Restormer/single_image_defocus_deblurring.pth']),
    ('Restormer_dual_pixel_defocus_deblurring.pth', ['Restormer/dual_pixel_defocus_deblurring.pth']),
    ('Swin2SR_Jpeg_dynamic.pth', ['Swin2SR_Jpeg_dynamic.pth']),
    ('1x_ReFocus_V3_140000_G.pth', ['1x_ReFocus_V3_140000_G.pth']),
]
SAM2 = ('sam2.1_hiera_large.pt', 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt')


def say(msg):
    print(msg, flush=True)


def place(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return 'have'
    try:
        os.link(src, dst)
        return 'linked'
    except OSError:
        shutil.copy2(src, dst)
        return 'copied'


def download(url, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + '.part'
    say(f'  downloading {os.path.basename(dst)} ...')
    req = urllib.request.Request(url, headers={'User-Agent': 'RAWDemosaic-setup'})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, 'wb') as f:
        shutil.copyfileobj(r, f, 1 << 20)
    with open(tmp, 'rb') as f:
        head = f.read(64)
    if head.lstrip().startswith(b'<'):
        os.remove(tmp)
        raise RuntimeError('the server returned a web page instead of the file')
    os.replace(tmp, dst)


def find(names, roots):
    for n in names:
        for r in roots:
            p = os.path.join(r, n)
            if os.path.isfile(p): return p
            hits = glob.glob(os.path.join(r, '**', n), recursive=True)
            if hits: return hits[0]
    return None


def get(dst, names, roots, url, allow_download):
    if os.path.exists(dst):
        say(f'  have {os.path.relpath(dst, DEST)}')
        return True
    src = find(names, roots)
    if src:
        say(f'  {place(src, dst)} {os.path.relpath(dst, DEST)}  <-  {os.path.relpath(src, ROOT) if src.startswith(ROOT) else src}')
        return True
    if url and allow_download:
        try:
            download(url, dst)
            return True
        except Exception as ex:
            say(f'  could not download {os.path.basename(dst)}: {ex}')
    else:
        say(f'  missing {os.path.relpath(dst, DEST)} (not found locally{"" if url else ", no download source"})')
    return False


def main():
    allow = '--no-download' not in sys.argv
    say(f'models-desktop: {DEST}')
    say('These models are not part of RAW Demosaic and keep their own licences (see THIRD_PARTY.md):\n'
        '  NIND-Unet (darktable-ai) GPL-3.0 · RawForge / RawRefinery MIT · SAM2.1 Apache-2.0 · Real-ESRGAN BSD-3-Clause ·\n'
        '  RealSR, Swin2SR Apache-2.0 · NAFNet MIT · Restormer MIT. Checkpoints found locally (e.g. community upscalers)\n'
        '  are linked as they are: check each one\'s own licence.')
    say('NIND-Unet')
    ok = True
    for sub, f in (('rawdenoise-nind', 'model_bayer.onnx'), ('rawdenoise-nind', 'model_linear.onnx'), ('denoise-nind', 'model.onnx'), ('denoise-nind', 'config.json'), ('rawdenoise-nind', 'config.json')):
        src = os.path.join(ROOT, 'models', sub, f)
        dst = os.path.join(DEST, sub, f)
        if not os.path.exists(src) and allow and f != 'config.json':
            try:
                sys.path.insert(0, ROOT)
                from rawdemosaic.dl_models import model_downloader
                model_downloader.ensure_model(sub)
            except Exception as ex:
                say(f'  could not fetch {sub}: {ex}')
        if os.path.exists(src):
            say(f'  {place(src, dst)} {sub}/{f}')
        elif f != 'config.json':
            ok = False; say(f'  missing {sub}/{f}')
    say('RawForge (float32 TorchScript)')
    for f, url in RAWFORGE.items():
        for name, u in ((f, url), (f + '.sig', url + '.sig')):
            ok &= get(os.path.join(DEST, 'rawforge', name), [name], [RF_CACHE], u, allow) or name.endswith('.sig')
    say('Super-resolution')
    for dst, names, url in SR:
        get(os.path.join(DEST, 'sr', dst), names, [PPM], url, allow)
    say('Restoration')
    for dst, names in RESTORE:
        get(os.path.join(DEST, 'restore', dst), names, [PPM], None, allow)
    say('Inpainting (LaMa, MI-GAN, MST + HAWP; float32 ONNX) and the Lensfun database')
    for f in ('lama_fp32.onnx', 'migan_pipeline.onnx', 'mst_p2m_encoder_512x512.onnx', 'mst_p2m_decoder_512x512.onnx', 'hawp_512x512_float32_opt.onnx'):
        src = os.path.join(ROOT, 'models', 'inpaint', f)
        if os.path.exists(src): say(f'  {place(src, os.path.join(DEST, "inpaint", f))} inpaint/{f}')
        else: say(f'  missing inpaint/{f} (the install script downloads it into models/inpaint)')
    lf = os.path.join(ROOT, 'models', 'lensfun')
    n = 0
    for f in sorted(glob.glob(os.path.join(lf, '*.xml'))):
        place(f, os.path.join(DEST, 'lensfun', os.path.basename(f))); n += 1
    say(f'  lensfun: {n} files' if n else '  missing lensfun (the install script downloads it into models/lensfun)')
    say('SAM2.1 (float32)')
    get(os.path.join(DEST, 'sam2', SAM2[0]), [SAM2[0]], [os.path.join(ROOT, 'models', 'huggingface')], SAM2[1], allow)
    say('Done.' if ok else 'Done, with some models missing (see above).')


if __name__ == '__main__':
    main()
