# Third-party code, libraries, models and data

Photo Suite is licensed under the **GNU General Public License v3.0 or later** (see [LICENSE](LICENSE)). It has to
be: RAW Demosaic's AMaZE and RCD demosaics are ports of RawTherapee code, which is GPL-3.0-or-later, and Photo
Library uses RAW Demosaic's engine. Everything below is credited with its own licence. Each item is one of:

- **Ported**: code translated from another project into this repository, keeping its licence and copyright.
- **Library**: installed by the installers from PyPI (or the PyTorch index) and imported; not part of this repository.
- **Web**: loaded from a CDN by the browser versions (`rawdemosaic.html`, `panoramamaker.html`); not part of this repository.
- **Model / data**: downloaded by a setup script, `download_web_models.py`, the installers' `-Models` / `--models`
  option, the apps themselves, or the browser versions; never part of this repository.

Licences were checked against each project's own LICENSE file or model card in October 2026.

## Code ported into this repository

| What | Where in this repository | Upstream | Licence |
|---|---|---|---|
| AMaZE demosaic. Copyright (c) 2008-2010 Emil Martinec, optimised by Ingo Weyrich | `RAWDemosaic/desktop/rawdemosaic_demosaic.py`, `RAWDemosaic/rawdemosaic.html` | [RawTherapee](https://github.com/RawTherapee/RawTherapee) `rtengine/amaze_demosaic_RT.cc` | GPL-3.0-or-later |
| RCD demosaic. Copyright (c) 2017-2020 Luis Sanz Rodríguez and Ingo Weyrich | same files | RawTherapee `rtengine/rcd_demosaic.cc` | GPL-3.0-or-later |
| RawTherapee's `border_interpolate` | `RAWDemosaic/desktop/rawdemosaic_demosaic.py` | RawTherapee | GPL-3.0-or-later |
| SIFT keypoints and descriptors (David Lowe). Copyright (c) 2006-2010 Rob Hess; (C) 2009 Willow Garage; (C) 2020 Intel | `panoramamaker/panoramamaker.html` (notice reproduced above the code) | [OpenCV](https://github.com/opencv/opencv) `modules/features2d/src/sift.dispatch.cpp`, based on Rob Hess's SIFT | Apache-2.0 (OpenCV); BSD-style (Rob Hess). SIFT's patent, US 6,711,293, expired in March 2020. |
| Face alignment template and SCRFD output decoding | `PhotoLibrary/desktop/photolibrary_ai.py` | [InsightFace](https://github.com/deepinsight/insightface) (code) | MIT |
| DAT architecture | `RAWDemosaic/rawdemosaic/dl_models/restore/dat.py` | [zhengchen1999/DAT](https://github.com/zhengchen1999/DAT) | Apache-2.0 |
| DeblurGAN-v2 FPN-Inception architecture | `…/restore/deblurganv2.py` | [VITA-Group/DeblurGANv2](https://github.com/VITA-Group/DeblurGANv2) | BSD-3-Clause |
| InceptionResNetV2 backbone (used by DeblurGAN-v2) | `…/restore/deblurganv2.py` | [Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) | BSD-3-Clause |
| NAFNet and NAFSSR architectures | `…/restore/nafnet.py`, `…/restore/nafssr.py` | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) | MIT |
| Restormer architecture | `…/restore/restormer.py` | [swz30/Restormer](https://github.com/swz30/Restormer) | MIT |
| RRDBNet and SRVGGNetCompact architectures | `…/restore/rrdbnet.py` | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | BSD-3-Clause |
| Swin2SR architecture | `…/restore/swin2sr.py` | [mv-lab/swin2sr](https://github.com/mv-lab/swin2sr) | Apache-2.0 |

**Not included:** MPRNet's architecture (`mprnet.py`). [swz30/MPRNet](https://github.com/swz30/MPRNet) is under an
*Academic Public License* (non-commercial only, and copies must keep that licence), which can't be combined with the
GPL. RAW Demosaic treats it as optional: without the file, MPRNet checkpoints are simply not recognised.

The SIFT in the panorama web app started from two tutorial repositories, *SIFT-SURF* and *Computer-Vision*
(feature detection with SIFT, SURF and ORB). Neither has a licence and neither is included: their code only calls
OpenCV, and the algorithm here follows OpenCV's own implementation, credited above. No SURF code is included.

## Libraries (installed, not included)

| Library | Used by | Licence |
|---|---|---|
| [PyTorch](https://pytorch.org) and torchvision | all three apps | BSD-3-Clause |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) (`onnxruntime-gpu`) | all three apps | MIT |
| [NumPy](https://numpy.org), [SciPy](https://scipy.org) | all three apps | BSD-3-Clause |
| [Numba](https://github.com/numba/numba) | RAW Demosaic's demosaics | BSD-2-Clause |
| [Pillow](https://github.com/python-pillow/Pillow) | all three apps | MIT-CMU (HPND) |
| [OpenCV](https://github.com/opencv/opencv) (`opencv-python-headless`) | PANOSTITCH, Photo Library | Apache-2.0 |
| [rawpy](https://github.com/letmaik/rawpy), which bundles [LibRaw](https://github.com/LibRaw/LibRaw) | RAW reading | MIT (rawpy); LGPL-2.1 or CDDL-1.0 (LibRaw) |
| [tifffile](https://github.com/cgohlke/tifffile) | TIFF reading and writing | BSD-3-Clause |
| [ExifRead](https://github.com/ianare/exif-py) | Photo Library's EXIF | BSD-3-Clause |
| [Hugging Face transformers](https://github.com/huggingface/transformers), huggingface_hub, tokenizers, safetensors, accelerate | Photo Library's AI | Apache-2.0 |
| sentencepiece, protobuf, einops, timm | transformers models | Apache-2.0 / BSD-3-Clause / MIT |
| [SAM2](https://github.com/facebookresearch/sam2), with hydra-core and iopath | RAW Demosaic's masks | Apache-2.0 (SAM2), MIT (hydra-core, iopath) |
| [RawForge](https://github.com/rymuelle/RawForge) | RAW Demosaic's RAW denoising | MIT |
| [colour-science](https://github.com/colour-science/colour) | RAW Demosaic | BSD-3-Clause |
| [ONNX](https://github.com/onnx/onnx) | RAW Demosaic's model export tools | Apache-2.0 |
| [PyMaxflow](https://github.com/pmneila/PyMaxflow) | PANOSTITCH's seam cutting | GPL-3.0 |
| [psutil](https://github.com/giampaolo/psutil) | PANOSTITCH | BSD-3-Clause |
| [bm3d](https://pypi.org/project/bm3d/) (Tampere University) | RAW Demosaic's optional BM3D denoise | **Free for non-commercial use only, not open source.** Installed only if you agree; the rest works without it. |
| Tk / tkinter | every window | Python Software Foundation / Tcl-Tk licence |

## Loaded by the browser versions (not included)

| Library | Licence |
|---|---|
| [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) | MIT |
| [libraw-wasm](https://github.com/ybouane/LibRaw-Wasm) (LibRaw compiled to WebAssembly) | ISC (wrapper); LGPL-2.1 or CDDL-1.0 (LibRaw) |
| [JSZip](https://github.com/Stuk/jszip) | MIT or GPL-3.0 (dual) |
| [W3.CSS](https://www.w3schools.com/w3css/): an optional `w3.css` beside the page (not included; the pages carry their own fallback styles) | free to use, no licence required |

## Models and data (downloaded, never included)

| Model / data | Used by | Source | Licence |
|---|---|---|---|
| InsightFace **buffalo_l** (SCRFD-10G detector, ArcFace R50) | Photo Library: faces | [deepinsight/insightface](https://github.com/deepinsight/insightface) release v0.7 | **Non-commercial research use only.** Downloaded only after you accept that. |
| SigLIP2 so400m-patch14-384 | Photo Library: search by meaning | [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) | Apache-2.0 |
| Florence-2 large | Photo Library: captions and tags | [florence-community/Florence-2-large](https://huggingface.co/florence-community/Florence-2-large) (port of microsoft/Florence-2-large) | MIT |
| NIND-Unet denoisers (RAW and RGB) | RAW Demosaic | [darktable-org/darktable-ai](https://github.com/darktable-org/darktable-ai), from [trougnouf/nind-denoise](https://github.com/trougnouf/nind-denoise) | GPL-3.0 |
| RawForge / RawRefinery denoise, deblur and sharpen models | RAW Demosaic | [rymuelle/RawForge](https://github.com/rymuelle/RawForge), [rymuelle/RawRefinery](https://github.com/rymuelle/RawRefinery) | MIT |
| SAM2.1 hiera-large (and the tiny, base-plus and large ONNX copies the web version uses) | RAW Demosaic: masks | [facebook/sam2.1-hiera-large](https://huggingface.co/facebook/sam2.1-hiera-large), [onnx-community](https://huggingface.co/onnx-community) | Apache-2.0 |
| Real-ESRGAN, RealESRNet, realesr-general | RAW Demosaic: super-resolution | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) releases; ONNX copies [SceneWorks/real-esrgan-onnx](https://huggingface.co/SceneWorks/real-esrgan-onnx), [svjack/AI-onnx](https://huggingface.co/svjack/AI-onnx) | BSD-3-Clause |
| RealSR | RAW Demosaic: super-resolution | [jixiaozhong/RealSR](https://github.com/jixiaozhong/RealSR) | Apache-2.0 |
| Swin2SR | RAW Demosaic: super-resolution | [mv-lab/swin2sr](https://github.com/mv-lab/swin2sr) releases | Apache-2.0 |
| NAFNet, NAFSSR, Restormer checkpoints (if you supply them) | RAW Demosaic: restoration | the projects above | MIT |
| Other upscalers you add yourself (e.g. community ESRGAN/DAT models) | RAW Demosaic | their authors | their own; check each |
| LaMa (ONNX) | RAW Demosaic web: inpainting | [advimman/lama](https://github.com/advimman/lama), [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) | Apache-2.0 |
| MI-GAN (ONNX) | RAW Demosaic web: inpainting | [Picsart-AI-Research/MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN), [anyisalin/migan-onnx](https://huggingface.co/anyisalin/migan-onnx) | MIT |
| MST inpainting and the HAWP line detector (ONNX; optional, `download_web_models.py --mst`) | RAW Demosaic web: inpainting along straight lines | [ewrfcas/MST_inpainting](https://github.com/ewrfcas/MST_inpainting), [cherubicXN/hawp](https://github.com/cherubicXN/hawp), converted by [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) | MIT |
| RaCo-ALIKED + LightGlue+ fused pipeline (ONNX) | PANOSTITCH, desktop and web | [fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX) release v3.0, of [ALIKED](https://github.com/Shiaoming/ALIKED) and [LightGlue](https://github.com/cvg/LightGlue) | Apache-2.0 (exports and LightGlue), BSD-3-Clause (ALIKED) |
| Lensfun lens database | RAW Demosaic: lens corrections | [lensfun/lensfun](https://github.com/lensfun/lensfun) `data/db` | CC BY-SA 3.0 |
| Camera colour matrices | RAW Demosaic | LibRaw's camera database, via rawpy | as LibRaw |

## External tools

PANOSTITCH can export [Hugin](https://hugin.sourceforge.io) projects and act as Hugin's control point detector; Hugin
is not included or required. The demosaic algorithms credited above come from RawTherapee, but RawTherapee itself is
not used or needed.
