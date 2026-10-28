# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""
Colour management for the desktop app, with LittleCMS (through Pillow's ImageCms):

  - the monitor profile: every view is converted from its colour space to the screen's;
  - soft proofing: the developed image through a printer (or any output) profile, shown as it will print,
    with paper and ink simulation and a gamut warning;
  - converting an export to an ICC profile (e.g. a lab's printer profile), which is then embedded.

Views and soft proofs are 8-bit (as the screen). Exports are converted through an 86 x 86 x 86 table built
with LittleCMS from the output space's own profile, interpolated in floating point, so 16-bit TIFFs stay smooth.
"""
import io
import os
import sys
import threading

import numpy as np
from PIL import Image, ImageCms

import rawdemosaic_engine as E

try:
    from numba import njit, prange
except ImportError:   # pragma: no cover - numba is a requirement; this keeps imports working without it
    njit = lambda *a, **k: (lambda f: f)
    prange = range

INTENTS = [('perceptual', 'Perceptual', ImageCms.Intent.PERCEPTUAL),
           ('relative', 'Relative colorimetric', ImageCms.Intent.RELATIVE_COLORIMETRIC),
           ('saturation', 'Saturation', ImageCms.Intent.SATURATION),
           ('absolute', 'Absolute colorimetric', ImageCms.Intent.ABSOLUTE_COLORIMETRIC)]
INTENT = {k: v for k, _, v in INTENTS}
WARN = (255, 0, 255)   # gamut warning: magenta, rare in photographs

# views of linear Rec. 2020 are sent to LittleCMS as 8-bit Rec. 2020 with a 2.2 gamma: wide enough for any screen
WIDE_GAMMA = 563 / 256
WIDE_ICC = E.icc_profile('rec2020', gamma=WIDE_GAMMA, name='Rec. 2020 gamma 2.2 (RAW Demosaic view)')

_lock = threading.Lock()   # Pillow's transforms keep a cache: one at a time
_tf = {}
_profiles = {}
_monitor = None            # the profile file used for the screen, or None (the screen is taken as sRGB)


# ───────────────────────────────────────────── profiles
def profile_dirs():
    """Where profiles are looked for: the app's profiles folder, then the system's."""
    d = [os.path.join(E.APP_ROOT, 'profiles')]
    if os.name == 'nt':
        d.append(os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'System32', 'spool', 'drivers', 'color'))
    elif sys.platform == 'darwin':
        d += ['/Library/ColorSync/Profiles', os.path.expanduser('~/Library/ColorSync/Profiles')]
    else:
        d += [os.path.expanduser('~/.local/share/icc'), os.path.expanduser('~/.color/icc'), '/usr/share/color/icc', '/usr/local/share/color/icc']
    return d


def info(path):
    """dict(path, desc, cls ('mntr', 'prtr', ...), space ('RGB ', 'CMYK', ...)), or None if it is not a profile."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    if key in _profiles: return _profiles[key]
    r = None
    try:
        with open(path, 'rb') as f: head = f.read(128)
        if len(head) == 128 and head[36:40] == b'acsp':
            try: desc = ImageCms.getProfileDescription(ImageCms.getOpenProfile(path)).strip()
            except Exception: desc = ''
            r = dict(path=path, desc=desc or os.path.splitext(os.path.basename(path))[0],
                     cls=head[12:16].decode('latin-1'), space=head[16:20].decode('latin-1'))
    except OSError:
        pass
    _profiles[key] = r
    return r


def list_profiles(classes=None, space='RGB '):
    """Profiles found in profile_dirs(), optionally of some classes only ('mntr', 'prtr', 'spac', ...)."""
    out, seen = [], set()
    for d in profile_dirs():
        try: names = sorted(os.listdir(d), key=str.lower)
        except OSError: continue
        for n in names:
            if os.path.splitext(n)[1].lower() not in ('.icc', '.icm'): continue
            p = os.path.join(d, n)
            i = info(p)
            if not i or (classes and i['cls'] not in classes) or (space and i['space'] != space): continue
            if os.path.normcase(p) in seen: continue
            seen.add(os.path.normcase(p)); out.append(i)
    return out


def labelled(items):
    """[(label, path)] for a combobox: the profile's description, with its folder when two share one."""
    out, used = [], set()
    for i in items:
        l = i['desc']
        if l in used: l = f"{l}  ({os.path.basename(os.path.dirname(i['path']))})"
        used.add(l); out.append((l, i['path']))
    return out


def system_monitor():
    """The profile the system has for the screen (Windows), or None."""
    try:
        p = ImageCms.get_display_profile()
        return getattr(p, 'filename', None) if p is not None else None
    except Exception:
        return None


def _open(p):
    if isinstance(p, bytes): return ImageCms.ImageCmsProfile(io.BytesIO(p))
    if p == 'srgb': return ImageCms.createProfile('sRGB')
    return ImageCms.getOpenProfile(p)


def _transform(key, build):
    t = _tf.get(key)
    if t is None:
        t = _tf[key] = build()
    return t


# ───────────────────────────────────────────── the screen
def set_monitor(choice):
    """choice: 'system' (the profile the system has for the screen), '' (none: the screen is taken as sRGB) or a file."""
    global _monitor
    p = system_monitor() if choice == 'system' else choice or None
    if p and info(p) is None: p = None
    if p and 'srgb' in (info(p)['desc'] or '').lower().replace(' ', '') and choice == 'system':
        p = None   # the stock sRGB profile: nothing to convert
    _monitor = p
    _tf.clear()
    return p


def monitor():
    return _monitor


def screen_srgb(img):
    """An sRGB PIL image (results, previews) -> the screen's colours."""
    if not _monitor: return img
    if img.mode != 'RGB': img = img.convert('RGB')
    t = _transform(('s', _monitor), lambda: ImageCms.buildTransform(_open('srgb'), _open(_monitor), 'RGB', 'RGB', INTENT['relative']))
    with _lock:
        return ImageCms.applyTransform(img, t)


def wide8(px):
    """Linear Rec. 2020 (float, H x W x 3) -> 8-bit Rec. 2020 with a 2.2 gamma (WIDE_ICC)."""
    return (np.power(np.clip(px, 0, 1), 1 / WIDE_GAMMA) * 255 + 0.5).astype(np.uint8)


def screen_linear(px, proof=None):
    """Linear Rec. 2020 (the Develop view) -> uint8 for the screen, through `proof` if given
    (dict(path, intent, bpc, paper, gamut)). None when no conversion is needed (no monitor profile, no proof)."""
    if not _monitor and not proof: return None
    img = Image.fromarray(wide8(px))
    mon = _monitor or 'srgb'
    if not proof:
        t = _transform(('w', mon), lambda: ImageCms.buildTransform(_open(WIDE_ICC), _open(mon), 'RGB', 'RGB', INTENT['relative']))
        with _lock:
            return np.asarray(ImageCms.applyTransform(img, t))
    flags = ImageCms.Flags.SOFTPROOFING | (ImageCms.Flags.BLACKPOINTCOMPENSATION if proof.get('bpc') else ImageCms.Flags.NONE)
    pint = INTENT['absolute'] if proof.get('paper') else INTENT['relative']   # simulate the paper's white and the ink's black, or not

    def build(f):
        return lambda: ImageCms.buildProofTransform(_open(WIDE_ICC), _open(mon), _open(proof['path']), 'RGB', 'RGB',
                                                    renderingIntent=INTENT[proof.get('intent', 'relative')], proofRenderingIntent=pint, flags=f)
    key = ('p', mon, proof['path'], proof.get('intent'), bool(proof.get('bpc')), bool(proof.get('paper')))
    t = _transform(key, build(flags))
    with _lock:
        out = np.array(ImageCms.applyTransform(img, t))
    if proof.get('gamut'):   # LittleCMS marks colours outside the printer's gamut: compare with and without the check
        tg = _transform(key + ('g',), build(flags | ImageCms.Flags.GAMUTCHECK))
        with _lock:
            g = np.asarray(ImageCms.applyTransform(img, tg))
        out[np.any(g != out, axis=2)] = WARN
    return out


# ───────────────────────────────────────────── converting exports
GRID = 86   # table nodes at 0, 3, ..., 255: exact 8-bit inputs for LittleCMS


def check_output_profile(path):
    i = info(path)
    if i is None: raise ValueError(f'{os.path.basename(path)} is not an ICC profile')
    if i['space'] != 'RGB ': raise ValueError(f'{i["desc"]} is a {i["space"].strip()} profile: only RGB profiles can be converted to '
                                               '(most labs and inkjet drivers take RGB)')
    return i


def export_table(space, conv):
    """space (an output space, e.g. 'prophoto') -> conv['path'], with conv['intent'] and conv['bpc']: a GRID^3 x 3 float32 table."""
    check_output_profile(conv['path'])
    v = np.arange(0, 256, 255 // (GRID - 1), dtype=np.uint8)
    r, g, b = np.meshgrid(v, v, v, indexing='ij')
    grid = np.stack([r, g, b], -1).reshape(GRID * GRID, GRID, 3)
    flags = ImageCms.Flags.NOOPTIMIZE | (ImageCms.Flags.BLACKPOINTCOMPENSATION if conv.get('bpc') else ImageCms.Flags.NONE)   # each node exact
    t = ImageCms.buildTransform(_open(E.icc_profile(space)), _open(conv['path']), 'RGB', 'RGB', INTENT[conv.get('intent', 'relative')], flags)
    with _lock:
        out = np.asarray(ImageCms.applyTransform(Image.fromarray(grid), t))
    return np.ascontiguousarray(out.reshape(GRID, GRID, GRID, 3).astype(np.float32) / 255)


@njit(parallel=True, cache=True)
def _apply_table(enc, lut, out):
    n = lut.shape[0] - 1
    for y in prange(enc.shape[0]):
        for x in range(enc.shape[1]):
            f = np.empty(3)
            i = np.empty(3, np.int64)
            for c in range(3):
                v = min(max(enc[y, x, c], 0.0), 1.0) * n
                k = min(int(v), n - 1)
                i[c] = k; f[c] = v - k
            for o in range(3):   # trilinear
                a = 0.0
                for dr in range(2):
                    wr = f[0] if dr else 1 - f[0]
                    for dg in range(2):
                        wg = f[1] if dg else 1 - f[1]
                        for db in range(2):
                            wb = f[2] if db else 1 - f[2]
                            a += wr * wg * wb * lut[i[0] + dr, i[1] + dg, i[2] + db, o]
                out[y, x, o] = a


def convert(enc, lut):
    """Encoded output-space values in [0,1] (H x W x 3) -> the profile's, in [0,1] (float32)."""
    out = np.empty(enc.shape, np.float32)
    _apply_table(np.ascontiguousarray(enc, np.float32), lut, out)
    return out


def icc_bytes(path):
    with open(path, 'rb') as f:
        return f.read()
