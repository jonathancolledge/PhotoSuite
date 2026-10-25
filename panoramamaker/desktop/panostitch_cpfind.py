# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""PANOSTITCH control point detector for Hugin (ALIKED + LightGlue on the GPU, SIFT fallback).

Hugin → Preferences → Control Points → Add…
    Program:    <PhotoSuite>\\.venv\\Scripts\\python.exe   (Linux: <PhotoSuite>/.venv-linux/bin/python)
    Arguments:  "<PhotoSuite>\\panoramamaker\\desktop\\panostitch_cpfind.py" -o %o %s
    Type:       All images at once
Command line:  python panostitch_cpfind.py [options] -o output.pto input.pto
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SUITE = os.path.normpath(os.path.join(HERE, '..', '..'))   # installed with the other desktop apps: shared models
DEFAULT_MODELS = (os.path.join(SUITE, 'models', 'panoramamaker') if os.path.isfile(os.path.join(SUITE, 'suite.json'))
                  else os.path.normpath(os.path.join(HERE, '..', 'models')))

import panostitch_engine as E
import panostitch_hugin as PH


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try: stream.reconfigure(errors='replace')   # Hugin's log window / a cp1252 console
        except Exception: pass
    ap = argparse.ArgumentParser(description='Adds ALIKED+LightGlue control points to a Hugin project.')
    ap.add_argument('input', help='Hugin project (.pto) whose images to match')
    ap.add_argument('-o', '--output', required=True, help='project to write (input + control points)')
    ap.add_argument('--matcher', choices=['auto', 'aliked', 'sift'], default='auto',
                    help='auto: ALIKED+LightGlue, SIFT where it finds too little (default)')
    ap.add_argument('--model', default='k2048', help='keypoint budget, e.g. k1024, k2048, k3584 (default k2048)')
    ap.add_argument('--models-dir', default=DEFAULT_MODELS)
    ap.add_argument('--device', default=None, help='cuda:0, cuda:1 or cpu (default: first GPU)')
    ap.add_argument('--res', default='full', help='matching resolution: "full" or a longest side in px (default full)')
    ap.add_argument('--points', type=int, default=PH.DEFAULT_POINTS,
                    help='max control points per pair; 0 = all good matches (default)')
    ap.add_argument('--window', type=int, default=0, help='only match images up to N apart in the list (0 = all pairs)')
    ap.add_argument('--no-prune', action='store_true',
                    help='keep matches that disagree with a single viewpoint (parallax); by default they are dropped')
    a = ap.parse_args(argv)
    strip = lambda p: p.strip().strip('"')
    pin, pout = strip(a.input), strip(a.output)

    def log(level, msg): print(f'[{level}] {msg}', flush=True)
    last = [0.0]
    def progress(frac, label=None):
        if label and time.time() - last[0] > 0.5:
            last[0] = time.time(); print(f'  {frac * 100:3.0f}%  {label}', flush=True)
    job = E.Job(log=log, progress=progress)

    models = E.find_models(a.models_dir)
    model_path = models.get(a.model.lower()) or (list(models.values())[-1] if models else None)
    if a.matcher != 'sift' and not model_path:
        log('warn', f'no ALIKED models in {a.models_dir} — using SIFT')
        a.matcher = 'sift'
    device = a.device or E.list_devices()[0]
    res = None if a.res.lower() == 'full' else int(a.res)
    log('info', f'PANOSTITCH cpfind: {a.matcher}, {os.path.basename(model_path or "")} on {device}, '
                f'matching at {"full resolution" if res is None else f"{res}px"}, '
                f'{"all good matches" if a.points <= 0 else f"at most {a.points} points per pair"}')
    t0 = time.time()
    pairs, points = PH.add_control_points(pin, pout, job, matcher=a.matcher, model_path=model_path, device=device,
                                          match_res=res, max_points=a.points, window=a.window, prune=not a.no_prune)
    log('ok', f'{points} control points on {pairs} image pairs written to {pout} in {time.time() - t0:.1f}s')
    return 0 if pairs else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as ex:
        print(f'[err] {ex}', file=sys.stderr, flush=True)
        sys.exit(2)
