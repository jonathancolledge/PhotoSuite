# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Download Photo Library's AI models into the models folder (default: the suite's models\\PhotoLibrary, else ..\\models).

    .venv\\Scripts\\python.exe photolibrary_setup_models.py [models_dir] [faces] [siglip] [florence] [--yes]

  buffalo_l/   InsightFace's buffalo_l pack: det_10g.onnx (SCRFD-10G face detector, the successor of
               RetinaFace) and w600k_r50.onnx (ArcFace R50 face embeddings). Non-commercial research licence.
  siglip2/     google/siglip2-so400m-patch14-384: image and text embeddings for search by meaning. Apache-2.0.
  florence2/   Florence-2-large: captions, from which tags are taken. MIT.

The models are not part of Photo Library and keep their own licences (see THIRD_PARTY.md). The face models
are for non-commercial research use only, so they are downloaded only once that is accepted: --yes on the
command line, or the question the app asks. The app calls download() itself from Settings & AI → Download
models; this file also runs on its own.
"""
import os
import shutil
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(os.path.dirname(HERE))   # installed with the other desktop apps: the shared models folder
DEFAULT_MODELS = (os.path.join(SUITE, 'models', 'PhotoLibrary') if os.path.isfile(os.path.join(SUITE, 'suite.json'))
                  else os.path.join(os.path.dirname(HERE), 'models'))

BUFFALO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
BUFFALO_FILES = ('det_10g.onnx', 'w600k_r50.onnx')
SIGLIP_REPO = 'google/siglip2-so400m-patch14-384'
# transformers' own Florence-2 port (no remote code); the original microsoft/ repo needs trust_remote_code,
# which recent transformers versions no longer run
FLORENCE_REPOS = ('florence-community/Florence-2-large', 'microsoft/Florence-2-large')

PARTS = {'faces': 'buffalo_l', 'siglip': 'siglip2', 'florence': 'florence2'}
LICENSES = {
    'faces': 'InsightFace buffalo_l (github.com/deepinsight/insightface): for NON-COMMERCIAL RESEARCH use only',
    'siglip': 'SigLIP2 so400m (huggingface.co/google/siglip2-so400m-patch14-384): Apache-2.0',
    'florence': 'Florence-2 large (huggingface.co/florence-community/Florence-2-large): MIT',
}
FACES_NOTICE = ('The face models (InsightFace buffalo_l: SCRFD detector + ArcFace) are licensed for non-commercial research '
                'use only. They are not part of Photo Library and are downloaded from InsightFace\'s own release.')


def present(models_dir):
    """Which models are already in the folder -> {'faces': bool, 'siglip': bool, 'florence': bool}."""
    def hf(name):   # config and weights both there (a download in progress has no .safetensors yet)
        p = os.path.join(models_dir, name)
        return os.path.isfile(os.path.join(p, 'config.json')) and any(f.endswith('.safetensors') for f in os.listdir(p))
    return dict(faces=all(os.path.isfile(os.path.join(models_dir, 'buffalo_l', f)) for f in BUFFALO_FILES),
                siglip=hf('siglip2'), florence=hf('florence2'))


def _faces(models_dir, log):
    out = os.path.join(models_dir, 'buffalo_l')
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(models_dir, 'buffalo_l.zip.part')
    log('info', f'downloading {BUFFALO_URL} (about 280 MB)')
    with urllib.request.urlopen(BUFFALO_URL) as r, open(tmp, 'wb') as f:
        shutil.copyfileobj(r, f, 1 << 20)
    with zipfile.ZipFile(tmp) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base in BUFFALO_FILES:
                with z.open(name) as src, open(os.path.join(out, base), 'wb') as dst:
                    shutil.copyfileobj(src, dst)
    os.remove(tmp)
    log('ok', f'face models -> {out}')


def _hf(repo, out, log, patterns=None):
    from huggingface_hub import snapshot_download
    log('info', f'downloading {repo} from Hugging Face')
    snapshot_download(repo, local_dir=out, allow_patterns=patterns)
    log('ok', f'{repo} -> {out}')


def download(models_dir=DEFAULT_MODELS, which=('faces', 'siglip', 'florence'), log=lambda level, msg: print(msg),
             accept_noncommercial=False):
    """accept_noncommercial: the face models' non-commercial licence was accepted (else they are skipped)."""
    os.makedirs(models_dir, exist_ok=True)
    have = present(models_dir)
    for part in which:
        if have.get(part):
            log('info', f'{PARTS[part]} is already there'); continue
        if part == 'faces' and not accept_noncommercial:
            log('warn', 'face models skipped: their non-commercial licence was not accepted'); continue
        log('info', 'licence: ' + LICENSES[part])
        if part == 'faces':
            _faces(models_dir, log)
        elif part == 'siglip':
            _hf(SIGLIP_REPO, os.path.join(models_dir, 'siglip2'), log,
                ['*.json', '*.safetensors', '*.model', '*.txt'])
        elif part == 'florence':
            err = None
            for repo in FLORENCE_REPOS:
                try:
                    _hf(repo, os.path.join(models_dir, 'florence2'), log, ['*.json', '*.safetensors', '*.py', '*.txt', '*.model'])
                    err = None
                    break
                except Exception as ex:
                    err = ex
                    log('warn', f'{repo}: {ex}')
            if err: raise err


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--yes']
    d = next((a for a in args if a not in PARTS), DEFAULT_MODELS)
    which = [a for a in args if a in PARTS] or list(PARTS)
    ok = '--yes' in sys.argv
    if 'faces' in which and not ok and not present(d)['faces'] and sys.stdin.isatty():
        print(FACES_NOTICE)
        ok = input('Download them for non-commercial use? [y/N] ').strip().lower().startswith('y')
    download(d, which, accept_noncommercial=ok)
