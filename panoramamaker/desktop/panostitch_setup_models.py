# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Download PANOSTITCH's RaCo-ALIKED + LightGlue+ matcher models (ONNX) into its models folder.

    python panostitch_setup_models.py [models_dir] [k512 k1024 …]

They are Fabio Sim's fused extractor-matcher exports (github.com/fabio-sim/LightGlue-ONNX, release v3.0,
Apache-2.0) of ALIKED (BSD-3-Clause) and LightGlue (Apache-2.0). Without them PANOSTITCH matches with SIFT.
"""
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.normpath(os.path.join(HERE, '..', '..'))   # installed with the other desktop apps: shared models
DEFAULT_MODELS = (os.path.join(SUITE, 'models', 'panoramamaker') if os.path.isfile(os.path.join(SUITE, 'suite.json'))
                  else os.path.normpath(os.path.join(HERE, '..', 'models')))
RELEASE = 'https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v3.0/'
SIZES = ('k512', 'k1024', 'k1536', 'k2048', 'k2560', 'k3072', 'k3584')
LICENSE = 'RaCo-ALIKED + LightGlue+ ONNX (github.com/fabio-sim/LightGlue-ONNX): Apache-2.0; ALIKED BSD-3-Clause, LightGlue Apache-2.0'


def download(models_dir=DEFAULT_MODELS, sizes=SIZES, log=lambda level, msg: print(msg)):
    os.makedirs(models_dir, exist_ok=True)
    log('info', 'licence: ' + LICENSE)
    for k in sizes:
        name = f'raco_aliked_lightglue_pipeline_{k}.onnx'
        dst = os.path.join(models_dir, name)
        if os.path.isfile(dst):
            log('info', f'{name} is already there'); continue
        log('info', f'downloading {name}')
        with urllib.request.urlopen(RELEASE + name) as r, open(dst + '.part', 'wb') as f:
            shutil.copyfileobj(r, f, 1 << 20)
        os.replace(dst + '.part', dst)
    log('ok', f'ALIKED + LightGlue models -> {models_dir}')


if __name__ == '__main__':
    args = sys.argv[1:]
    download(next((a for a in args if a not in SIZES), DEFAULT_MODELS), [a for a in args if a in SIZES] or SIZES)
