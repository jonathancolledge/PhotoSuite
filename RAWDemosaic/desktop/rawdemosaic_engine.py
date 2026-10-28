# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""RAW Demosaic desktop engine: the web app's (rawdemosaic.html) processing pipeline, ported to numpy,
Numba, PyTorch and ONNX Runtime, plus the desktop-only extras (BM3D, median and detail-adaptive blends,
restoration models, SAM2-masked blending of finished files).

Everything runs in full precision: float32 weights everywhere (RawForge's original TorchScript models
rather than its float16 ONNX exports), no autocast, and TF32 switched off on the GPU.

Colour pipeline (identical to the web app): black-subtracted mosaic normalised to (white - black) ->
optional detail boost / RAW denoise -> white balance on the mosaic -> clip to 1 -> demosaic -> camera RGB
-> linear Rec.2020 (D65) working space via the camera matrix -> optional post-demosaic denoise -> column
blend -> output space (sRGB, Adobe RGB, ProPhoto) with its transfer curve and an embedded ICC profile.
"""
import json
import math
import os
import struct
import threading
import time

import numpy as np

import rawdemosaic_demosaic as DM

HERE = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(HERE)
# installed with the other desktop apps (a release folder holding suite.json): the shared models folder
SUITE = os.path.dirname(APP_ROOT)
IN_SUITE = os.path.isfile(os.path.join(SUITE, 'suite.json'))
DEFAULT_MODELS = os.path.join(SUITE, 'models', 'RAWDemosaic') if IN_SUITE else os.path.join(APP_ROOT, 'models-desktop')

RAW_EXTS = ('.nef', '.nrw', '.cr2', '.cr3', '.crw', '.arw', '.srf', '.sr2', '.raf', '.dng', '.orf', '.rw2', '.pef',
            '.srw', '.x3f', '.3fr', '.fff', '.iiq', '.mos', '.mef', '.mrw', '.erf', '.kdc', '.dcr', '.raw', '.rwl', '.gpr')
IMAGE_EXTS = ('.jpg', '.jpeg', '.jfif', '.png', '.tif', '.tiff', '.bmp', '.webp')


class Cancelled(Exception):
    pass


class Job:
    """Progress/log sink handed to long operations. All methods are safe to call from a worker thread."""
    def __init__(self, log=None, progress=None):
        self._log, self._progress = log, progress
        self.cancel_event = threading.Event()

    def log(self, level, msg):
        if self._log: self._log(level, msg)

    def progress(self, frac, label=None):
        if self._progress: self._progress(frac, label)

    def check(self):
        if self.cancel_event.is_set(): raise Cancelled()


def make_eta():
    t0 = time.time()

    def eta(done, total):
        if done < 2 or done >= total: return ''
        left = (time.time() - t0) / done * (total - done)
        return f' — about {math.ceil(left / 5) * 5} s left' if left < 90 else f' — about {round(left / 60)} min left'
    return eta


# ───────────────────────────────────────────── colour
def _xyz_of_xy(x, y):
    return np.array([x / y, 1.0, (1 - x - y) / y])


def rgb_to_xyz(prim, white):
    P = np.stack([_xyz_of_xy(*p) for p in prim], axis=1)
    S = np.linalg.solve(P, _xyz_of_xy(*white))
    return P * S[None, :]


D65, D50 = (0.3127, 0.3290), (0.3457, 0.3585)
XYZ_D65, XYZ_D50 = np.array([0.95047, 1.0, 1.08883]), np.array([0.96422, 1.0, 0.82521])
SPACES = {
    'rec2020': dict(prim=[(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)], white=D65, name='Rec. 2020 linear (RAW Demosaic)', label='Rec. 2020 (linear)'),
    'srgb': dict(prim=[(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)], white=D65, name='sRGB IEC61966-2.1 (RAW Demosaic)', label='sRGB'),
    'adobe': dict(prim=[(0.64, 0.33), (0.21, 0.71), (0.15, 0.06)], white=D65, name='Adobe RGB (1998) compatible (RAW Demosaic)', label='Adobe RGB (1998)'),
    'prophoto': dict(prim=[(0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001)], white=D50, name='ProPhoto RGB (RAW Demosaic)', label='ProPhoto RGB'),
}
for _s in SPACES.values():
    _s['toXYZ'] = rgb_to_xyz(_s['prim'], _s['white'])
    _s['fromXYZ'] = np.linalg.inv(_s['toXYZ'])
OUTPUT_SPACES = ['srgb', 'adobe', 'prophoto']
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def adapt(src_white, dst_white):
    s, d = BRADFORD @ src_white, BRADFORD @ dst_white
    return np.linalg.inv(BRADFORD) @ np.diag(d / s) @ BRADFORD


D65_TO_D50 = adapt(XYZ_D65, XYZ_D50)
D50_TO_D65 = adapt(XYZ_D50, XYZ_D65)
REC2020_Y = SPACES['rec2020']['toXYZ'][1]


def cam_to_working(xyz_to_cam):
    """White-balanced camera RGB -> linear Rec.2020. xyz_to_cam: XYZ(D65) -> camera; its columns are
    scaled so that a white-balanced neutral maps to D65 white."""
    xyz_to_cam = np.asarray(xyz_to_cam, float)
    c = xyz_to_cam @ XYZ_D65
    cam_to_xyz = np.linalg.inv(xyz_to_cam) @ np.diag([c[0] / c[1], 1, c[2] / c[1]])
    return SPACES['rec2020']['fromXYZ'] @ cam_to_xyz


def working_to(space):
    m = SPACES['rec2020']['toXYZ']
    if SPACES[space]['white'] == D50: m = D65_TO_D50 @ m
    return SPACES[space]['fromXYZ'] @ m


def space_to_working(space):
    return np.linalg.inv(working_to(space))


def encode(space, v):
    """Linear [0,1] -> the space's transfer curve (v is clipped to [0,1] first)."""
    v = np.clip(v, 0.0, 1.0)
    if space == 'srgb':
        return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1 / 2.4) - 0.055)
    if space == 'adobe':
        return np.power(v, 256 / 563)
    if space == 'prophoto':
        return np.where(v < 1 / 512, 16 * v, np.power(v, 1 / 1.8))
    return v


def decode(space, v):
    v = np.clip(v, 0.0, 1.0)
    if space == 'srgb':
        return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))
    if space == 'adobe':
        return np.power(v, 563 / 256)
    if space == 'prophoto':
        return np.where(v < 16 / 512, v / 16, np.power(v, 1.8))
    return v


def apply_matrix(img, M, out=None):
    """HxWx3 float32 times a 3x3 matrix, in row strips to keep memory bounded."""
    M = np.asarray(M, np.float32)
    out = np.empty_like(img, np.float32) if out is None else out
    step = max(1, 4_000_000 // max(1, img.shape[1]))
    for y in range(0, img.shape[0], step):
        out[y:y + step] = img[y:y + step] @ M.T
    return out


def to_output(work, space, dtype=np.float32):
    """Linear working RGB -> encoded output-space values in [0,1] (float32), in strips."""
    M = working_to(space).astype(np.float32)
    out = np.empty(work.shape, np.float32)
    step = max(1, 2_000_000 // max(1, work.shape[1]))
    for y in range(0, work.shape[0], step):
        out[y:y + step] = encode(space, work[y:y + step] @ M.T)
    return out


def to_uint16(enc):
    return np.round(np.clip(enc, 0, 1) * 65535).astype(np.uint16)


def to_srgb8(work):
    """Linear working RGB -> sRGB uint8 for display / JPEG."""
    M = working_to('srgb').astype(np.float32)
    out = np.empty(work.shape, np.uint8)
    step = max(1, 2_000_000 // max(1, work.shape[1]))
    for y in range(0, work.shape[0], step):
        out[y:y + step] = np.round(encode('srgb', work[y:y + step] @ M.T) * 255).astype(np.uint8)
    return out


def display_image(work, max_side=0):
    """Linear working RGB -> sRGB uint8, box-downscaled so its long side is at most max_side."""
    h, w = work.shape[:2]
    f = max(1, math.ceil(max(w, h) / max_side)) if max_side else 1
    if f > 1:
        sh, sw = h // f, w // f
        work = work[:sh * f, :sw * f].reshape(sh, f, sw, f, 3).mean(axis=(1, 3))
    return to_srgb8(work)


def orient(img, flip):
    """dcraw/LibRaw flip (bit 2: transpose, bit 1: vertical, bit 0: horizontal), as the web app's orientSrc."""
    if not flip: return img
    if flip & 2: img = img[::-1]
    if flip & 1: img = img[:, ::-1]
    if flip & 4: img = np.swapaxes(img, 0, 1)
    return np.ascontiguousarray(img)


# ───────────────────────────────────────────── ICC profiles, TIFF and JPEG output
def icc_profile(space, gamma=None, name=None):
    """ICC v2 matrix/TRC display profile for an output space (byte-identical in structure to the web app's).
    gamma / name: the space's primaries with a plain gamma curve instead (e.g. for views)."""
    sp = SPACES[space]
    M = sp['toXYZ']
    if sp['white'] == D65: M = D65_TO_D50 @ M   # ICC colorants are relative to the D50 PCS
    name = (name or sp['name']).encode('ascii') + b'\0'
    desc = bytearray(12 + len(name) + 4 + 4 + 2 + 1 + 67)
    desc[0:4] = b'desc'; struct.pack_into('>I', desc, 8, len(name)); desc[12:12 + len(name)] = name
    text = b'text\0\0\0\0' + b'No copyright, use freely\0'

    def xyz(v):
        return b'XYZ \0\0\0\0' + b''.join(struct.pack('>i', round(x * 65536)) for x in v)
    if gamma:
        curv = b'curv\0\0\0\0' + struct.pack('>IH', 1, round(gamma * 256))
    elif space == 'adobe':
        curv = b'curv\0\0\0\0' + struct.pack('>IH', 1, 563)
    elif space == 'rec2020':
        curv = b'curv\0\0\0\0' + struct.pack('>I', 0)
    else:
        n = 1024
        x = np.arange(n) / (n - 1)
        vals = np.round(np.clip(decode(space, x), 0, 1) * 65535).astype('>u2')
        curv = b'curv\0\0\0\0' + struct.pack('>I', n) + vals.tobytes()
    tags = [(b'desc', bytes(desc)), (b'cprt', text), (b'wtpt', xyz(XYZ_D50)),
            (b'rXYZ', xyz(M[:, 0])), (b'gXYZ', xyz(M[:, 1])), (b'bXYZ', xyz(M[:, 2])),
            (b'rTRC', curv), (b'gTRC', curv), (b'bTRC', curv)]
    off = 128 + 4 + 12 * len(tags)
    offsets, table, blobs = {}, [], []
    for sig, data in tags:
        if data not in offsets:
            offsets[data] = off; blobs.append((off, data)); off = (off + len(data) + 3) & ~3
        table.append((sig, offsets[data], len(data)))
    b = bytearray(off)
    struct.pack_into('>I', b, 0, off); struct.pack_into('>I', b, 8, 0x02100000)
    b[12:16] = b'mntr'; b[16:20] = b'RGB '; b[20:24] = b'XYZ '
    t = time.gmtime()
    struct.pack_into('>6H', b, 24, t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
    b[36:40] = b'acsp'
    struct.pack_into('>3i', b, 68, *[round(v * 65536) for v in XYZ_D50])
    struct.pack_into('>I', b, 128, len(tags))
    for i, (sig, o, ln) in enumerate(table):
        b[132 + 12 * i:136 + 12 * i] = sig; struct.pack_into('>II', b, 136 + 12 * i, o, ln)
    for o, data in blobs:
        b[o:o + len(data)] = data
    return bytes(b)


def write_tiff16(path, img16, space, software='RAW Demosaic (desktop)', icc=None):
    import tifffile
    tifffile.imwrite(path, img16, photometric='rgb', planarconfig='contig', iccprofile=icc or icc_profile(space),
                     resolution=(300, 300), resolutionunit='INCH', software=software, metadata=None)


def write_jpeg(path, srgb8, quality=95, icc=None):
    from PIL import Image
    Image.fromarray(srgb8).save(path, 'JPEG', quality=int(quality), subsampling=0 if quality >= 90 else 2,
                                icc_profile=icc or icc_profile('srgb'), optimize=True)


# ───────────────────────────────────────────── ICC parsing (for JPEG/TIFF/PNG input and re-blending)
XYZ_TO_R2020 = SPACES['rec2020']['fromXYZ']


class Profile:
    """A matrix/TRC (or grey) ICC profile: to_rec maps linear profile RGB to linear Rec.2020."""
    def __init__(self, name, to_rec, trc, gray=False, space=None):
        self.name, self.to_rec, self.trc, self.gray, self.space = name, np.asarray(to_rec, float), trc, gray, space

    def linearize(self, v, c):
        return self.trc[c](v)


def _srgb_profile(name='sRGB (assumed)'):
    f = lambda v: decode('srgb', v)
    return Profile(name, space_to_working('srgb'), [f, f, f], space='srgb')


def _linear_srgb_profile():
    f = lambda v: v
    return Profile('linear sRGB (assumed)', space_to_working('srgb'), [f, f, f])


def parse_icc(data):
    """Matrix/TRC and grey ICC profiles (what working spaces and cameras use). Our own profiles are
    recognised by name and decoded with their exact maths. Returns a Profile, or None."""
    try:
        u8 = bytes(data)
        n = struct.unpack_from('>I', u8, 128)[0]
        tags = {}
        for i in range(n):
            o = 132 + i * 12
            tags[u8[o:o + 4].decode('latin-1')] = struct.unpack_from('>II', u8, o + 4)
        s15 = lambda o: struct.unpack_from('>i', u8, o)[0] / 65536
        name = ''
        if 'desc' in tags:
            o = tags['desc'][0]; typ = u8[o:o + 4]
            if typ == b'desc':
                ln = struct.unpack_from('>I', u8, o + 8)[0]; name = u8[o + 12:o + 12 + ln].split(b'\0')[0].decode('latin-1')
            elif typ == b'mluc':
                so, sl = struct.unpack_from('>II', u8, o + 20); name = u8[o + so:o + so + sl].decode('utf-16-be', 'replace')
        for key, sp in SPACES.items():
            if name == sp['name']:
                f = (lambda k: (lambda v: decode(k, v)))(key)
                return Profile(sp['label'] + ' (RAW Demosaic)', space_to_working(key), [f, f, f], space=key)

        def curve(t):
            o, ln = tags[t]; typ = u8[o:o + 4]
            if typ == b'curv':
                cnt = struct.unpack_from('>I', u8, o + 8)[0]
                if cnt == 0: return lambda x: x
                if cnt == 1:
                    g = struct.unpack_from('>H', u8, o + 12)[0] / 256
                    return lambda x: np.power(np.maximum(0, x), g)
                tab = np.frombuffer(u8, '>u2', cnt, o + 12).astype(np.float64) / 65535
                xs = np.linspace(0, 1, cnt)
                return lambda x: np.interp(np.clip(x, 0, 1), xs, tab)
            if typ == b'para':
                ft = struct.unpack_from('>H', u8, o + 8)[0]
                P = [s15(o + 12 + 4 * k) if o + 12 + 4 * k < o + ln else 0 for k in range(7)]
                g, a, b, c, d, e, f = P
                if ft == 0: return lambda x: np.power(np.maximum(0, x), g)
                if ft == 1: return lambda x: np.where(x >= -b / a, np.power(np.maximum(0, a * x + b), g), 0)
                if ft == 2: return lambda x: np.where(x >= -b / a, np.power(np.maximum(0, a * x + b), g) + c, c)
                if ft == 3: return lambda x: np.where(x >= d, np.power(np.maximum(0, a * x + b), g), c * x)
                if ft == 4: return lambda x: np.where(x >= d, np.power(np.maximum(0, a * x + b), g) + e, c * x + f)
            return None
        if all(k in tags for k in ('rXYZ', 'gXYZ', 'bXYZ', 'rTRC')):
            col = lambda t: [s15(tags[t][0] + 8), s15(tags[t][0] + 12), s15(tags[t][0] + 16)]
            to_xyz50 = np.array([col('rXYZ'), col('gXYZ'), col('bXYZ')]).T
            trc = [curve(t) for t in ('rTRC', 'gTRC', 'bTRC')]
            if any(f is None for f in trc): return None
            return Profile(name or 'embedded profile', XYZ_TO_R2020 @ D50_TO_D65 @ to_xyz50, trc)
        if 'kTRC' in tags:
            f = curve('kTRC')
            if f: return Profile(name or 'grey profile', np.eye(3), [f, f, f], gray=True)
    except Exception:
        pass
    return None


def load_image_linear(path):
    """A JPEG/PNG/TIFF/BMP/WebP -> (linear Rec.2020 float32 HxWx3, info dict, alpha or None)."""
    ext = os.path.splitext(path)[1].lower()
    icc, alpha, orientation, note = None, None, 1, ''
    exif = {}
    if ext in ('.tif', '.tiff'):
        import tifffile
        with tifffile.TiffFile(path) as tf:
            page = next((p for p in tf.pages if not (getattr(p, 'subfiletype', 0) & 1)), tf.pages[0])
            a = page.asarray()
            tg = page.tags
            if 34675 in tg: icc = bytes(tg[34675].value)
            if 274 in tg: orientation = int(tg[274].value)
            photometric = int(tg[262].value) if 262 in tg else 2
            for k, t in ((271, 'make'), (272, 'model')):
                if k in tg: exif[t] = str(tg[k].value).strip()
            if photometric not in (0, 1, 2):
                raise ValueError('Only RGB and greyscale TIFFs are supported (not CMYK, Lab or palette).')
        if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[2] not in (3, 4):
            a = np.moveaxis(a, 0, -1)
        if a.ndim == 2: a = a[..., None]
        if photometric == 0: a = (np.iinfo(a.dtype).max - a) if a.dtype.kind in 'ui' else 1 - a
        kind = f'{a.dtype.itemsize * 8}-bit{" float" if a.dtype.kind == "f" else ""} TIFF'
    else:
        from PIL import Image, ImageOps
        im = Image.open(path)
        icc = im.info.get('icc_profile')
        try: orientation = int(im.getexif().get(0x112, 1))
        except Exception: pass
        a = None
        with open(path, 'rb') as f:
            head = f.read(32)
        if head[:8] == b'\x89PNG\r\n\x1a\n' and head[24] == 16 and head[25] in (2, 6):   # 16-bit RGB(A) PNG: Pillow reads 8 bits
            try:
                import cv2
                b = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if b is not None and b.dtype == np.uint16 and b.ndim == 3:
                    a = b[..., [2, 1, 0, 3]] if b.shape[2] == 4 else b[..., ::-1]
            except Exception:
                note = 'This 16-bit PNG is read with 8 bits per channel (install opencv-python-headless for 16).'
        if a is None:
            if im.mode in ('I;16', 'I;16B', 'I;16L', 'I'):
                a = np.asarray(im).astype(np.uint16)
            else:
                im = im.convert('RGBA' if ('A' in im.mode or 'transparency' in im.info) else 'RGB')
                a = np.asarray(im)
        if a.ndim == 2: a = a[..., None]
        kind = f'{a.dtype.itemsize * 8}-bit {ext[1:].upper().replace("JPG", "JPEG")}'
    chans = a.shape[2]
    if chans in (2, 4):
        al = a[..., -1]
        if al.min() < (np.iinfo(al.dtype).max if al.dtype.kind in 'ui' else 1):
            alpha = (al.astype(np.float32) / (np.iinfo(al.dtype).max if al.dtype.kind in 'ui' else 1))
        a = a[..., :-1]
    prof = parse_icc(icc) if icc else None
    if prof is None:
        prof = _linear_srgb_profile() if a.dtype.kind == 'f' else _srgb_profile()
    mx = float(np.iinfo(a.dtype).max) if a.dtype.kind in 'ui' else 1.0
    gray = a.shape[2] == 1
    rgb = np.empty(a.shape[:2] + (3,), np.float32)
    if mx > 1 and mx <= 65535:
        lut = [prof.linearize(np.arange(int(mx) + 1) / mx, c).astype(np.float32) for c in range(3)]
        for c in range(3):
            rgb[..., c] = lut[c][a[..., 0 if gray else c]]
    else:
        for c in range(3):
            rgb[..., c] = prof.linearize(a[..., 0 if gray else c].astype(np.float64), c)
    if not prof.gray:
        apply_matrix(rgb, prof.to_rec, out=rgb)
    np.maximum(rgb, 0, out=rgb)
    rgb, alpha = _exif_orient(rgb, orientation), (_exif_orient(alpha, orientation) if alpha is not None else None)
    info = dict(kind=kind, profile=prof.name, W=rgb.shape[1], H=rgb.shape[0], make=exif.get('make', ''),
                model=exif.get('model', ''), note=note, is_rgb=True)
    return rgb, info, alpha


def _exif_orient(img, o):
    if o == 2: return img[:, ::-1].copy()
    if o == 3: return img[::-1, ::-1].copy()
    if o == 4: return img[::-1].copy()
    if o == 5: return np.swapaxes(img, 0, 1).copy()
    if o == 6: return np.swapaxes(img, 0, 1)[:, ::-1].copy()
    if o == 7: return np.swapaxes(img, 0, 1)[::-1, ::-1].copy()
    if o == 8: return np.swapaxes(img, 0, 1)[::-1].copy()
    return img


# ───────────────────────────────────────────── RAW files
def _tiff_ifds(buf):
    """Every IFD (with SubIFDs) of a TIFF-structured file: list of {tag: (type, count, value-offset, le)}."""
    if len(buf) < 8 or buf[:2] not in (b'II', b'MM'): return [], True
    le = buf[:2] == b'II'
    e = '<' if le else '>'
    SZ = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8}
    ifds, seen = [], set()

    def walk(off, depth=0):
        k = 0
        while off and off not in seen and off + 2 <= len(buf) and k < 64 and depth < 4:
            seen.add(off); k += 1
            n = struct.unpack_from(e + 'H', buf, off)[0]
            tags = {}
            for i in range(n):
                p = off + 2 + 12 * i
                if p + 12 > len(buf): break
                tag, typ, cnt = struct.unpack_from(e + 'HHI', buf, p)
                size = SZ.get(typ, 1) * cnt
                vo = p + 8 if size <= 4 else struct.unpack_from(e + 'I', buf, p + 8)[0]
                tags[tag] = (typ, cnt, vo)
            ifds.append(tags)
            if 330 in tags:
                for sub in _tag_values(buf, tags[330], le): walk(int(sub), depth + 1)
            nxt = off + 2 + 12 * n
            off = struct.unpack_from(e + 'I', buf, nxt)[0] if nxt + 4 <= len(buf) else 0
    walk(struct.unpack_from(e + 'I', buf, 4)[0])
    return ifds, le


def _tag_values(buf, t, le):
    typ, cnt, vo = t
    e = '<' if le else '>'
    fmt = {1: 'B', 3: 'H', 4: 'I', 6: 'b', 8: 'h', 9: 'i', 11: 'f', 12: 'd', 7: 'B', 2: 'B'}
    if typ in (5, 10):
        raw = struct.unpack_from(e + ('I' if typ == 5 else 'i') * (2 * cnt), buf, vo)
        return [raw[2 * k] / (raw[2 * k + 1] or 1) for k in range(cnt)]
    return list(struct.unpack_from(e + fmt.get(typ, 'B') * cnt, buf, vo))


def _exif_lens(buf):
    """LensModel, LensMake and FocalLengthIn35mmFilm from the EXIF IFD of a TIFF-structured file."""
    ifds, le = _tiff_ifds(buf)
    out = {}
    try:
        if not ifds or 0x8769 not in ifds[0]: return out
        e = '<' if le else '>'
        off = int(_tag_values(buf, ifds[0][0x8769], le)[0])
        n = struct.unpack_from(e + 'H', buf, off)[0]
        SZ = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1}
        for i in range(n):
            q = off + 2 + 12 * i
            tag, typ, cnt = struct.unpack_from(e + 'HHI', buf, q)
            vo = q + 8 if SZ.get(typ, 1) * cnt <= 4 else struct.unpack_from(e + 'I', buf, q + 8)[0]
            if tag in (0xa434, 0xa433) and typ == 2:
                out['lens' if tag == 0xa434 else 'lens_make'] = buf[vo:vo + cnt].split(b'\0')[0].decode('latin-1').strip()
            elif tag == 0xa405 and typ == 3:
                out['f35'] = struct.unpack_from(e + 'H', buf, vo)[0]
    except Exception:
        pass
    return out


def _dng_info(buf):
    ifds, le = _tiff_ifds(buf)
    if not ifds or not any(50706 in t for t in ifds): return None
    raw = next((t for t in ifds if (_tag_values(buf, t[254], le)[0] if 254 in t else 0) == 0 and 262 in t
                and _tag_values(buf, t[262], le)[0] in (32803, 34892)), ifds[0])
    ifd0 = ifds[0]
    v = lambda t, k: _tag_values(buf, t[k], le) if k in t else None
    out = dict(black=v(raw, 50714), repeat=v(raw, 50713) or [1, 1], white=(v(raw, 50717) or [None])[0])
    colors = []
    for cm, cc, il in ((50721, 50723, 50778), (50722, 50724, 50779)):
        m = v(ifd0, cm)
        if m and len(m) >= 9:
            colors.append(dict(colormatrix=np.array(m[:9]).reshape(3, 3), calibration=(np.array(v(ifd0, cc)[:9]).reshape(3, 3) if v(ifd0, cc) and len(v(ifd0, cc)) >= 9 else None),
                               illuminant=(v(ifd0, il) or [21])[0]))
    out['colors'] = colors
    out['analog'] = v(ifd0, 50727)
    for k, name in ((271, 'make'), (272, 'model')):
        if k in ifd0: out[name] = bytes(_tag_values(buf, ifd0[k], le)).split(b'\0')[0].decode('latin-1').strip()
    return out


def _raf_info(buf):
    if buf[:15] != b'FUJIFILMCCD-RAW': return None
    out = {'model': buf[28:60].split(b'\0')[0].decode('latin-1').strip(), 'make': 'FUJIFILM'}
    hdr_off, cfa_off, cfa_len = struct.unpack_from('>I', buf, 92)[0], struct.unpack_from('>I', buf, 100)[0], struct.unpack_from('>I', buf, 104)[0]
    try:
        n = struct.unpack_from('>I', buf, hdr_off)[0]; p = hdr_off + 4
        for _ in range(n):
            if p + 4 > len(buf): break
            tag, size = struct.unpack_from('>HH', buf, p)
            if tag == 0x131 and size == 36: out['xtrans_layout'] = list(buf[p + 4:p + 40])
            p += 4 + size
    except Exception:
        pass
    if cfa_off and cfa_off + 8 < len(buf) and buf[cfa_off:cfa_off + 2] in (b'II', b'MM'):
        blk = buf[cfa_off:cfa_off + cfa_len]
        e = '<' if blk[:2] == b'II' else '>'

        def ifd(off, depth):
            cnt = struct.unpack_from(e + 'H', blk, off)[0]
            for i in range(cnt):
                q = off + 2 + 12 * i
                tag, typ, num = struct.unpack_from(e + 'HHI', blk, q)
                if tag == 0xf000 and depth < 2: ifd(struct.unpack_from(e + 'I', blk, q + 8)[0], depth + 1)
                if tag == 0xf00a and typ in (3, 4):
                    sz = 4 if typ == 4 else 2
                    base = q + 8 if sz * num <= 4 else struct.unpack_from(e + 'I', blk, q + 8)[0]
                    out['black'] = list(struct.unpack_from(e + ('I' if typ == 4 else 'H') * num, blk, base))
        try: ifd(struct.unpack_from(e + 'I', blk, 4)[0], 0)
        except Exception: pass
    return out


ILLUMINANT_CCT = {1: 5500, 2: 4200, 3: 2850, 4: 5500, 9: 5500, 10: 6500, 11: 7500, 12: 6430, 13: 5000, 14: 4150, 15: 3500,
                  17: 2856, 18: 4874, 19: 6774, 20: 5503, 21: 6504, 22: 7504, 23: 5003, 24: 3200}
SRGB_XYZ_TO_RGB = [[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]]


class RawFile:
    """A RAW file's mosaic and metadata, read with LibRaw (rawpy) the way the web app reads it with libraw-wasm."""
    def __init__(self, path):
        import rawpy
        self.path = path
        with open(path, 'rb') as f:
            buf = f.read()
        self.warnings = []
        with rawpy.imread(path) as r:
            pat = r.raw_pattern
            if pat is None or r.num_colors != 3:
                raise ValueError('This sensor type is not supported (only Bayer and Fuji X-Trans).')
            pat = np.array(pat)
            self.is_xtrans = pat.shape == (6, 6)
            if pat.shape not in ((2, 2), (6, 6)):
                raise ValueError('Unsupported colour filter pattern for this camera.')
            self.raw = np.ascontiguousarray(r.raw_image_visible).copy()
            self.H, self.W = self.raw.shape
            desc = r.color_desc.decode('ascii', 'replace')
            ids = np.vectorize(lambda v: {'R': 0, 'G': 1, 'B': 2}.get(desc[v], 1))(pat)
            self.P = pat.shape[0]
            self.pattern = ids.astype(np.int64).flatten()
            if not self.is_xtrans and ''.join(map(str, sorted(self.pattern))) != '0112':
                raise ValueError('Unsupported colour filter pattern for this camera.')
            bl = np.array(r.black_level_per_channel, float)
            per = [[], [], []]
            for i, cid in enumerate(desc[:4]):
                if cid in 'RGB': per['RGB'.index(cid)].append(bl[i])
            self.black = [float(np.mean(p)) if p else float(bl[0]) for p in per]
            self.black_source = 'LibRaw'
            self.white = float(r.white_level or 65535)
            cm = np.array(r.camera_whitebalance[:3], float); pm = np.array(r.daylight_whitebalance[:3], float)
            wb = cm if (cm[0] > 0 and cm[1] > 0) else pm
            self.wb = (wb / wb[1]).tolist() if wb[1] > 0 else [1.0, 1.0, 1.0]
            self.flip = int(r.sizes.flip)
            try: self.iso = float(r.other.iso_speed)
            except Exception: self.iso = 0.0
            try:
                o = r.other
                self.shutter, self.aperture, self.focal = float(o.shutter_speed), float(o.aperture), float(o.focal_length)
            except Exception:
                self.shutter = self.aperture = self.focal = 0.0
            xyz = np.array(r.rgb_xyz_matrix[:3], float)
            self.xyz_to_cam, self.matrix_source = None, ''
            if np.any(xyz):
                self.xyz_to_cam, self.matrix_source = xyz, 'LibRaw camera database'
            try:
                ln = r.lens
                self.lens, self.lens_make = (ln.model or '').strip(), (ln.make or '').strip()
                lo, hi = float(ln.min_focal or 0), float(ln.max_focal or 0)
            except Exception:
                self.lens, self.lens_make, lo, hi = '', '', 0, 0
            try:
                t = r.extract_thumb()
                self.thumb = t
            except Exception:
                self.thumb = None
        self.make = self.model = ''
        dng = _dng_info(buf) if buf[:2] in (b'II', b'MM') else None
        raf = _raf_info(buf)
        if dng:
            self.make, self.model = dng.get('make', ''), dng.get('model', '')
        elif raf:
            self.make, self.model = raf['make'], raf['model']
        else:
            ifds, le = _tiff_ifds(buf)
            if ifds:
                for k, name in ((271, 'make'), (272, 'model')):
                    if k in ifds[0]:
                        setattr(self, name, bytes(_tag_values(buf, ifds[0][k], le)).split(b'\0')[0].decode('latin-1').strip())
        self.f35 = 0
        if buf[:2] in (b'II', b'MM'):
            ex = _exif_lens(buf)
            if not self.lens and ex.get('lens') and not set(ex['lens']) <= set('-'): self.lens = ex['lens']
            if not self.lens_make: self.lens_make = ex.get('lens_make', '')
            self.f35 = ex.get('f35', 0)
        if not self.lens and lo:
            self.lens = f'{round(lo)}{"-" + str(round(hi)) if hi and hi != lo else ""}mm'
        if self.xyz_to_cam is None and dng and dng['colors']:
            c = min(dng['colors'], key=lambda d: abs(ILLUMINANT_CCT.get(d['illuminant'], 6504) - 6504))
            m = c['colormatrix']
            if c['calibration'] is not None and np.any(c['calibration']): m = c['calibration'] @ m
            ab = dng.get('analog')
            if ab and ab[0]: m = np.diag(ab[:3]) @ m
            self.xyz_to_cam, self.matrix_source = m, 'DNG embedded matrix'
        if self.xyz_to_cam is None:
            self.xyz_to_cam, self.matrix_source = np.array(SRGB_XYZ_TO_RGB), 'none (generic sRGB matrix)'
            self.warnings.append('No colour matrix for this camera: colours will be approximate.')
        if self.is_xtrans:
            if raf and raf.get('black'):
                m = float(np.mean(raf['black'])); self.black = [m, m, m]; self.black_source = 'Fuji RAF tag'
        elif dng and dng.get('black'):
            blv, (rr, rc) = dng['black'], dng['repeat'][:2]
            per = [[], [], []]
            if len(blv) >= rr * rc and rr == 2 and rc == 2:
                for k in range(4): per[int(self.pattern[k])].append(blv[k])
            else:
                for k in range(3): per[k].append(float(np.mean(blv)))
            self.black = [float(np.mean(p)) for p in per]
            self.black_source = 'DNG BlackLevel tag'
        if dng and dng.get('white'): self.white = float(dng['white'])
        self._xt = {}

    def summary(self):
        exp = ''
        if self.shutter: exp = f'{self.shutter:.1f} s' if self.shutter >= 1 else f'1/{round(1 / self.shutter)} s'
        bits = [f'ISO {self.iso:g}' if self.iso else '', exp, f'f/{self.aperture:.1f}' if self.aperture else '', f'{round(self.focal)} mm' if self.focal else '']
        return [('Camera', f'{self.make} {self.model}'.strip()),
                ('Sensor', f'{"X-Trans" if self.is_xtrans else "Bayer"}, {self.W} × {self.H} ({self.W * self.H / 1e6:.1f} MP)'),
                ('Lens', self.lens),
                ('Exposure', ', '.join(b for b in bits if b)),
                ('Black / white', ' / '.join(str(round(v)) for v in self.black) + f' / {self.white:g} ({self.black_source})'),
                ('White balance', ', '.join(f'{v:.3f}' for v in self.wb)),
                ('Colour matrix', self.matrix_source)]

    def xtrans_rgb(self, passes):
        """LibRaw's own Markesteijn demosaic in linear camera space (no WB, black subtracted, unscaled)."""
        if passes in self._xt: return self._xt[passes]
        import rawpy
        with rawpy.imread(self.path) as r:
            im = r.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD if passes == 3 else rawpy.DemosaicAlgorithm.PPG,
                               output_color=rawpy.ColorSpace.raw, use_camera_wb=False, use_auto_wb=False, user_wb=[1, 1, 1, 1],
                               no_auto_bright=True, no_auto_scale=True, gamma=(1, 1), output_bps=16, user_flip=0,
                               highlight_mode=rawpy.HighlightMode.Clip, half_size=False)
        if im.shape[:2] != (self.H, self.W):
            raise ValueError(f'Unexpected X-Trans decode size {im.shape[1]}×{im.shape[0]}')
        self._xt[passes] = im
        return im

    def thumbnail(self):
        """The embedded preview as an RGB uint8 array (or None)."""
        t = self.thumb
        if t is None: return None
        try:
            import rawpy
            from PIL import Image
            import io
            if t.format == rawpy.ThumbFormat.JPEG:
                return np.asarray(Image.open(io.BytesIO(t.data)).convert('RGB'))
            return np.asarray(t.data)[..., :3]
        except Exception:
            return None


def open_input(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return ImageInput(path)
    return RawFile(path)


class ImageInput:
    """An ordinary image: linear Rec.2020, for the after-demosaic stages."""
    is_rgb = True

    def __init__(self, path):
        self.path = path
        self.rgb, self.info, self.alpha = load_image_linear(path)
        self.H, self.W = self.rgb.shape[:2]
        self.warnings = [self.info['note']] if self.info.get('note') else []
        self.flip = 0

    def summary(self):
        return [('Image', f'{self.info["kind"]}, {self.W} × {self.H} ({self.W * self.H / 1e6:.1f} MP)'),
                ('Colour profile', self.info['profile']),
                ('Camera', f'{self.info.get("make", "")} {self.info.get("model", "")}'.strip()),
                ('Processing', 'Not a RAW file: no demosaicing or RAW denoising; post-demosaic denoise, detail boost, BM3D and output stages apply.')]


RawFile.is_rgb = False


# ───────────────────────────────────────────── mosaic helpers
def normalize_mosaic(raw, pattern, P, black, white):
    H, W = raw.shape
    pat = np.asarray(pattern).reshape(P, P)
    blk = np.asarray(black, np.float32)[pat]
    blk = np.tile(blk, (H // P + 1, W // P + 1))[:H, :W]
    scale = np.float32(1 / (white - np.mean(black)))
    out = raw.astype(np.float32) - blk
    np.maximum(out, 0, out=out)
    out *= scale
    return out


def rggb_phase(pattern):
    p = list(pattern)
    for dr in range(2):
        for dc in range(2):
            if p[dr * 2 + dc] == 0 and p[((dr + 1) % 2) * 2 + (dc + 1) % 2] == 2: return dr, dc
    raise ValueError('Unsupported Bayer pattern')


def cfa_colour_map(pattern, P, H, W):
    pat = np.asarray(pattern).reshape(P, P)
    return np.tile(pat, (H // P + 1, W // P + 1))[:H, :W]


def gaussian(plane, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(plane, sigma, mode='reflect', truncate=4.0)


def boost_mosaic(mosaic, P, gains, base_sigma):
    """Multi-scale detail boost (port of raw_detail_boost.py / the web app): per CFA position at the tile
    stride, non-decimated Gaussian-difference bands, sigma doubling per level, finest first."""
    last = max([i for i, g in enumerate(gains) if g != 1], default=-1)
    if last < 0: return mosaic
    out = mosaic.copy()
    for oy in range(P):
        for ox in range(P):
            prev = mosaic[oy::P, ox::P].astype(np.float32)
            res = np.zeros_like(prev); sigma = base_sigma
            for L in range(last + 1):
                b = gaussian(prev, sigma)
                res += (prev - b) * np.float32(gains[L])
                prev = b; sigma *= 2
            res += prev
            out[oy::P, ox::P] = res
    return out


def boost_rgb(work, gains):
    out = np.empty_like(work)
    for c in range(3):
        out[..., c] = np.maximum(0, boost_mosaic(work[..., c], 1, gains, 1.0))
    return out


def _bm3d():
    """The bm3d package: free for non-commercial use only, so it is an optional install."""
    try:
        import bm3d
    except ImportError:
        raise RuntimeError('BM3D is not installed. It is free for non-commercial use only, so it is optional: '
                           'install it with  pip install bm3d  (or answer yes when the installer asks), or set BM3D to None.') from None
    return bm3d


def bm3d_mosaic(mosaic, P, sigma, job=None):
    """BM3D per CFA position (each same-colour sub-lattice denoised as its own image)."""
    bm3d = _bm3d().bm3d
    out = mosaic.copy()
    n = P * P
    for k, (oy, ox) in enumerate((a, b) for a in range(P) for b in range(P)):
        if job: job.check(); job.progress(k / n, f'BM3D (RAW): plane {k + 1}/{n}')
        out[oy::P, ox::P] = bm3d(mosaic[oy::P, ox::P].astype(np.float64), sigma).astype(np.float32)
    return out


def bm3d_rgb(work, sigma):
    _b = _bm3d().bm3d_rgb
    return np.maximum(0, _b(work.astype(np.float64), sigma)).astype(np.float32)


# ───────────────────────────────────────────── models
def list_devices():
    """['cuda:0', 'cuda:1', ..., 'cpu'] with GPU names for display."""
    devs = []
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            devs.append((f'cuda:{i}', f'cuda:{i} · {torch.cuda.get_device_name(i)}'))
    except Exception:
        pass
    return devs + [('cpu', 'cpu')]


def _full_precision_torch():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    return torch


RAWFORGE = {
    'TreeNetDenoiseSuperLight': dict(file='ShadowWeightedL1_super_light.pt', maxIso=65535, cond=True),
    'TreeNetDenoiseLight': dict(file='ShadowWeightedL1_light.pt', maxIso=65535, cond=True),
    'TreeNetDenoise': dict(file='ShadowWeightedL1.pt', maxIso=65535, cond=True),
    'TreeNetDenoiseHeavy': dict(file='ShadowWeightedL1_24_deep_500.pt', maxIso=65535, cond=True),
    'Deblur': dict(file='realblur_gamma_140.pt', cond=True),
    'DeepSharpen': dict(file='Deblur_deep_24.pt', cond=True),
    'TreeNetDenoiseXTrans': dict(file='xtrans_fixed_exposure_no_conditioning_400.pt', cond=True, condScale=0, xtrans=True),
    'RestormerXTrans': dict(file='restormer.pt', cond=True, xtrans=True),
}
BAYER_RF = ['TreeNetDenoiseSuperLight', 'TreeNetDenoiseLight', 'TreeNetDenoise', 'TreeNetDenoiseHeavy', 'Deblur', 'DeepSharpen']
XTRANS_RF = ['TreeNetDenoiseXTrans', 'RestormerXTrans']
NIND = {
    'nind_bayer': ('RAW NIND-Unet (joint, Bayer)', os.path.join('rawdenoise-nind', 'model_bayer.onnx')),
    'nind_linear': ('NIND-Unet linear (X-Trans / post-demosaic)', os.path.join('rawdenoise-nind', 'model_linear.onnx')),
    'nind_rgb': ('NIND-Unet (post-demosaic)', os.path.join('denoise-nind', 'model.onnx')),
}
XYZ_TO_REC2020_RF = np.array([[1.71666343, -0.35567332, -0.25336809], [-0.66667384, 1.61645574, 0.0157683], [0.01764248, -0.04277698, 0.94224328]])


def find_pth_models(root):
    """Restoration / super-resolution checkpoints under models-desktop/sr and models-desktop/restore:
    [(label 'sr/RealESRGAN_x4plus', path)]."""
    out = []
    for sub in ('sr', 'restore'):
        d = os.path.join(root, sub)
        if not os.path.isdir(d): continue
        for base, _, files in os.walk(d):
            for f in sorted(files):
                if f.lower().endswith(('.pth', '.pt', '.safetensors')):
                    p = os.path.join(base, f)
                    out.append((os.path.relpath(p, root).replace('\\', '/').rsplit('.', 1)[0], p))
    return out


class Models:
    """Loads and caches every model on one device, in full precision."""
    def __init__(self, root, device='cuda:0'):
        self.root, self.device = root, device
        self._ort, self._rf, self._sam = {}, {}, {}

    def set(self, root, device):
        if root != self.root or device != self.device:
            self.release()
            self.root, self.device = root, device

    def release(self):
        self._ort.clear(); self._rf.clear(); self._sam.clear()
        try:
            from rawdemosaic.dl_models.restore import registry
            registry._sessions.clear()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception:
            pass

    def path(self, rel):
        p = os.path.join(self.root, rel)
        if not os.path.isfile(p):
            raise FileNotFoundError(f'Model not found: {p} — run desktop\\setup_models.py (or the install script) to fill models-desktop.')
        return p

    def torch_device(self):
        torch = _full_precision_torch()
        return torch.device(self.device if self.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')

    # ONNX Runtime (NIND-Unet, float32 models)
    def ort(self, key):
        if key in self._ort: return self._ort[key]
        import onnxruntime as ort
        try: ort.preload_dlls()
        except Exception: pass
        ort.set_default_logger_severity(3)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = ['CPUExecutionProvider']
        if self.device.startswith('cuda') and 'CUDAExecutionProvider' in ort.get_available_providers():
            prov = [('CUDAExecutionProvider', {'device_id': int(self.device.split(':')[1]), 'use_tf32': 0}), 'CPUExecutionProvider']
        s = ort.InferenceSession(self.path(NIND[key][1]), so, providers=prov)
        self._ort[key] = (s, s.get_inputs()[0].name)
        return self._ort[key]

    def run_ort(self, key, x):
        s, name = self.ort(key)
        return s.run(None, {name: x[None].astype(np.float32)})[0][0]

    # RawForge (original float32 TorchScript weights)
    def rawforge(self, name):
        if name in self._rf: return self._rf[name]
        torch = _full_precision_torch()
        m = torch.jit.load(self.path(os.path.join('rawforge', RAWFORGE[name]['file'])), map_location='cpu')
        m = m.float().eval().to(self.torch_device())
        self._rf[name] = m
        return m

    # restoration / super-resolution .pth (the old desktop app's architecture registry)
    def restore(self, path):
        _full_precision_torch()
        import sys
        if APP_ROOT not in sys.path: sys.path.insert(0, APP_ROOT)
        from rawdemosaic.dl_models import inference
        from rawdemosaic.dl_models.restore import registry
        dev = self.torch_device()
        if inference._device != dev:
            registry._sessions.clear()
            inference._device = dev
        return registry.load_model(path)

    def sam2(self, name='large'):
        if name in self._sam: return self._sam[name]
        _full_precision_torch()
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        cfg, ck = {'large': ('configs/sam2.1/sam2.1_hiera_l.yaml', 'sam2.1_hiera_large.pt'),
                   'base_plus': ('configs/sam2.1/sam2.1_hiera_b+.yaml', 'sam2.1_hiera_base_plus.pt')}[name]
        model = build_sam2(cfg, self.path(os.path.join('sam2', ck)), device=str(self.torch_device()))
        self._sam[name] = SAM2ImagePredictor(model)
        return self._sam[name]


# ───────────────────────────────────────────── tiled inference (as the web app's tiledInfer)
def reflect_idx(i, n):
    """numpy-pad 'reflect' indexing (d c b | a b c d), any distance out."""
    i = np.asarray(i)
    if n == 1: return np.zeros_like(i)
    p = 2 * n - 2
    i = np.abs(i) % p
    return np.where(i >= n, p - i, i)


def _starts(n, tile, step):
    if n <= tile: return [0]
    out, y = [], 0
    while True:
        if y + tile >= n:
            out.append(n - tile); break
        out.append(y); y += step
    return out


def tiled_infer(img, tile, overlap, out_c, run_tile, job, label, scale=1):
    """Overlap-tiled inference with linear feathering between neighbouring tiles (none at the image edges).
    img: HxWxC float32. run_tile(chw) -> (out_c, tile*scale, tile*scale)."""
    h, w, c = img.shape
    step = tile - overlap
    ys, xs = _starts(h, tile, step), _starts(w, tile, step)
    H2, W2, T2 = h * scale, w * scale, tile * scale
    acc = np.zeros((H2, W2, out_c), np.float32)
    wsum = np.zeros((H2, W2), np.float32)
    total, done, eta = len(ys) * len(xs), 0, make_eta()

    def axis_weights(starts, k):
        wts = np.ones(T2, np.float32)
        i = np.arange(T2)
        prev = (starts[k - 1] + tile - starts[k]) * scale if k > 0 else 0
        nxt = (starts[k] + tile - starts[k + 1]) * scale if k < len(starts) - 1 else 0
        if prev > 0: wts = np.where(i < prev, np.minimum(wts, (i + 0.5) / prev), wts)
        if nxt > 0: wts = np.where(i >= T2 - nxt, np.minimum(wts, (T2 - i - 0.5) / nxt), wts)
        return wts.astype(np.float32)
    for ti, ty in enumerate(ys):
        ry = reflect_idx(ty + np.arange(tile), h)
        wy = axis_weights(ys, ti)
        for tj, tx in enumerate(xs):
            job.check()
            rx = reflect_idx(tx + np.arange(tile), w)
            chw = np.ascontiguousarray(img[ry[:, None], rx[None, :]].transpose(2, 0, 1), np.float32)   # gathers only the tile
            res = run_tile(chw)
            wx = axis_weights(xs, tj)
            oy, ox = ty * scale, tx * scale
            hh, ww = min(T2, H2 - oy), min(T2, W2 - ox)
            wgt = wy[:hh, None] * wx[None, :ww]
            wsum[oy:oy + hh, ox:ox + ww] += wgt
            acc[oy:oy + hh, ox:ox + ww] += res[:, :hh, :ww].transpose(1, 2, 0) * wgt[..., None]
            done += 1
            job.progress(done / total, f'{label}: tile {done}/{total}{eta(done, total)}')
    s = np.where(wsum > 1e-8, 1 / np.maximum(wsum, 1e-8), 0).astype(np.float32)
    acc *= s[..., None]
    return acc


def _gain_match(inp, out):
    mi, mo = float(inp.mean()), float(out.mean())
    if abs(mo) > 1e-12: out = out * np.float32(mi / mo)
    return out


# ───────────────────────────────────────────── RAW NIND-Unet (joint demosaic + denoise)
def nind_joint(mosaic, pattern, models, job):
    """Roll to RGGB, pack 2x2 -> 4 channels, 512 tiles (32 overlap) at half resolution -> full-resolution
    RGB, per-tile gain matched to the input. Output: linear camera RGB, no WB."""
    H, W = mosaic.shape
    dr, dc = rggb_phase(pattern)
    He, We = H + H % 2, W + W % 2
    yy = np.arange(He); xx = np.arange(We)
    yy = np.where(yy >= H, reflect_idx(yy, H), yy); xx = np.where(xx >= W, reflect_idx(xx, W), xx)
    a = mosaic[((yy + dr) % H)[:, None], ((xx + dc) % W)[None, :]]
    packed = np.stack([a[0::2, 0::2], a[0::2, 1::2], a[1::2, 0::2], a[1::2, 1::2]], -1)
    rgb2 = tiled_infer(packed, 512, 32, 3, lambda chw: _gain_match(chw, models.run_ort('nind_bayer', chw)), job, 'RAW NIND-Unet', scale=2)
    sy = (np.arange(H) - dr) % H; sx = (np.arange(W) - dc) % W
    return np.ascontiguousarray(rgb2[sy][:, sx])


# ───────────────────────────────────────────── RawForge
def rawforge_cond(iso, params):
    if not params.get('cond'): return 0.0
    c = iso or 0
    if params.get('maxIso'): c = min(c, params['maxIso'])
    c /= 6400
    if 'condScale' in params: c *= c * params['condScale']
    return c


def rawforge_chain(img, names, iso, models, job, label):
    torch = _full_precision_torch()
    dev = models.torch_device()
    for name in names:
        m = models.rawforge(name)
        cond = torch.tensor([[rawforge_cond(iso, RAWFORGE[name])]], dtype=torch.float32, device=dev)

        def run(chw, m=m, cond=cond):
            with torch.no_grad():
                return m(torch.from_numpy(chw[None]).to(dev), cond=cond)[0].float().cpu().numpy()
        img = tiled_infer(img, 256, 64, 3, run, job, f'{label}: {name}')
    return img


def rawforge_blend(orig, den, lumi, chroma):
    """RawForge's postprocess(): optional luminance/chroma blend back towards the input."""
    if not lumi > 0: return den
    k = (orig * den).sum(-1, keepdims=True) / ((den * den).sum(-1, keepdims=True) + 1e-6)
    L = k * den
    return (1 - lumi) * den + L * lumi + chroma * (orig - L)


def rawforge_bayer(mosaic, pattern, col, raw, models, job):
    """Replicates RawHandler.as_rgb (RGGB crop, camera -> linear Rec.2020 per 2x2 block, Malvar, clip) and
    ImageSaver.to_raw (back to camera space, re-mosaic). Returns a new full-size mosaic (no WB)."""
    H, W = mosaic.shape
    dr, dc = rggb_phase(pattern)
    h, w = (H - dr) & ~1, (W - dc) & ~1
    T = XYZ_TO_REC2020_RF @ np.linalg.inv(raw.xyz_to_cam)
    M4 = np.array([[T[0, 0], T[0, 1] / 2, T[0, 1] / 2, T[0, 2]], [T[1, 0], T[1, 1], 0, T[1, 2]],
                   [T[1, 0], 0, T[1, 1], T[1, 2]], [T[2, 0], T[2, 1] / 2, T[2, 1] / 2, T[2, 2]]], np.float32)
    sub = mosaic[dr:dr + h, dc:dc + w]
    v = np.stack([sub[0::2, 0::2], sub[0::2, 1::2], sub[1::2, 0::2], sub[1::2, 1::2]], -1)
    q = v @ M4.T
    bay = np.empty((h, w), np.float32)
    bay[0::2, 0::2], bay[0::2, 1::2], bay[1::2, 0::2], bay[1::2, 1::2] = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    img = DM.demosaic_malvar(bay, [0, 1, 1, 2], True)
    out = rawforge_chain(img, [m for m in (col['rf1'], col.get('rf2')) if m], raw.iso, models, job, 'RawForge')
    out = rawforge_blend(img, out, col.get('lumi', 0), col.get('chroma', 0))
    Ti = np.linalg.inv(T).astype(np.float32)
    res = mosaic.copy()
    chmap = np.array([[0, 1], [1, 2]])
    for py in range(2):
        for px in range(2):
            ch = chmap[py, px]
            res[dr + py:dr + h:2, dc + px:dc + w:2] = np.clip(out[py::2, px::2] @ Ti[ch], 0, 1)
    return res


def rawforge_xtrans(cam, col, raw, models, job):
    """RawForge feeds LibRaw's demosaiced camera RGB (no WB, black subtracted) / white level and keeps the
    model's RGB. Input/output here: camera RGB normalised to (white - black)."""
    k = np.float32((raw.white - np.mean(raw.black)) / raw.white)
    img = cam * k
    out = rawforge_chain(img, [m for m in (col['rf1'], col.get('rf2')) if m], raw.iso, models, job, 'RawForge X-Trans')
    out = rawforge_blend(img, out, col.get('lumi', 0), col.get('chroma', 0))
    return np.maximum(0, out / k)


# ───────────────────────────────────────────── post-demosaic NIND-Unet
def post_denoise(work, kind, models, job):
    if kind == 'nind_linear':   # trained on linear Rec.2020 (our working space); output scale is learned -> match gain
        out = tiled_infer(work, 512, 48, 3, lambda chw: _gain_match(chw, models.run_ort('nind_linear', chw)), job, 'NIND-Unet (linear)')
        return np.clip(out, 0, 1)
    g = encode('srgb', work).astype(np.float32)   # general RGB denoiser: run on gamma-encoded values
    out = tiled_infer(g, 768, 64, 3, lambda chw: models.run_ort('nind_rgb', chw), job, 'NIND-Unet')
    return decode('srgb', out).astype(np.float32)


# ───────────────────────────────────────────── columns
DEFAULT_GAINS = [2, 2, 2, 1, 1, 1, 1, 1, 1, 1]
DEMOSAIC_NAMES = {'amaze': 'AMaZE', 'rcd': 'RCD', 'mhc': 'Malvar (MHC)', 'bilinear': 'Bilinear',
                  'xt3': 'X-Trans Markesteijn 3-pass', 'xt1': 'X-Trans Markesteijn 1-pass'}


def new_column(**over):
    c = dict(name='Column', enabled=True, weight=1.0, rawDenoise='none', rf1='TreeNetDenoiseHeavy', rf2='', lumi=0.0, chroma=0.0,
             demosaic='amaze', postDenoise='none', boost=dict(enabled=False, stage='after', gains=list(DEFAULT_GAINS)),
             bm3d=dict(mode='none', sigma=0.02))
    for k, v in over.items():
        c[k] = (dict(c[k], **v) if isinstance(c.get(k), dict) and isinstance(v, dict) else v)
    return c


def default_columns():
    return [new_column(name='AMaZE', demosaic='amaze'),
            new_column(name='RAW NIND-Unet', weight=0.5, rawDenoise='nind', boost=dict(enabled=True, stage='before', gains=[3, 3, 3, 1, 1, 1, 1, 1, 1, 1])),
            new_column(name='RawForge', weight=2.0, rawDenoise='rawforge', rf1='TreeNetDenoiseHeavy', rf2='DeepSharpen', demosaic='amaze')]


def sanitize_column(c, is_xtrans):
    c = new_column(**c)
    if is_xtrans:
        if c['rawDenoise'] == 'nind':
            c['rawDenoise'] = 'none'
            if c['postDenoise'] == 'none': c['postDenoise'] = 'nind_linear'
        if c['demosaic'] not in ('xt3', 'xt1'): c['demosaic'] = 'xt3'
        if c['rf1'] not in XTRANS_RF: c['rf1'] = 'TreeNetDenoiseXTrans'
        if c['rf2'] and c['rf2'] not in XTRANS_RF: c['rf2'] = ''
        if c['bm3d']['mode'] == 'raw': c['bm3d']['mode'] = 'rgb'
    else:
        if c['demosaic'] not in ('amaze', 'rcd', 'mhc', 'bilinear'): c['demosaic'] = 'amaze'
        if c['rf1'] not in BAYER_RF: c['rf1'] = 'TreeNetDenoiseHeavy'
        if c['rf2'] and c['rf2'] not in BAYER_RF: c['rf2'] = ''
    return c


def run_column(col, src, mosaic, cam_to_work, models, job):
    """One column -> linear working RGB (H x W x 3), as the web app's runColumn, plus BM3D."""
    boost, bm = col['boost'], col['bm3d']
    if src.is_rgb:   # already linear working RGB; only the after-demosaic stages apply
        work = src.rgb.copy()
        if boost['enabled']:
            job.progress(0, 'Detail boost'); work = boost_rgb(work, boost['gains'])
        if col['postDenoise'] != 'none': work = post_denoise(work, col['postDenoise'], models, job)
        if bm['mode'] != 'none':
            job.progress(0, 'BM3D'); work = bm3d_rgb(work, bm['sigma'])
        return work
    wb = np.asarray(src.wb, np.float32)
    balanced = False
    if not src.is_xtrans:
        m, pattern = mosaic, src.pattern
        if boost['enabled'] and (boost['stage'] == 'before' or col['rawDenoise'] == 'nind'):
            job.progress(0, 'Detail boost'); m = boost_mosaic(m, src.P, boost['gains'], 0.5)
        if col['rawDenoise'] == 'nind':
            if bm['mode'] == 'raw': m = bm3d_mosaic(m, src.P, bm['sigma'], job)
            cam = nind_joint(m, pattern, models, job)
        else:
            if col['rawDenoise'] == 'rawforge': m = rawforge_bayer(m, pattern, col, src, models, job)
            if bm['mode'] == 'raw': m = bm3d_mosaic(m, src.P, bm['sigma'], job)
            if boost['enabled'] and boost['stage'] == 'after':
                job.progress(0, 'Detail boost'); m = boost_mosaic(m, src.P, boost['gains'], 0.5)
            # white balance on the mosaic before demosaicing (as RawTherapee does), then clip to [0,1]
            wm = np.minimum(1, m * wb[cfa_colour_map(pattern, 2, *m.shape)]).astype(np.float32)
            job.check(); job.progress(0, f'Demosaic ({DEMOSAIC_NAMES[col["demosaic"]]})')
            cam = DM.DEMOSAICS[col['demosaic']](wm, pattern)
            balanced = True
    else:
        src_rgb = src.xtrans_rgb(1 if col['demosaic'] == 'xt1' else 3)
        cam = src_rgb.astype(np.float32) * np.float32(1 / (src.white - np.mean(src.black)))
        if col['rawDenoise'] == 'rawforge': cam = rawforge_xtrans(cam, col, src, models, job)
    if not balanced: cam = cam * wb
    np.minimum(cam, 1, out=cam)
    apply_matrix(cam, cam_to_work, out=cam)
    np.maximum(cam, 0, out=cam)
    if col['postDenoise'] != 'none': cam = post_denoise(cam, col['postDenoise'], models, job)
    if bm['mode'] == 'rgb':
        job.progress(0, 'BM3D (RGB)'); cam = bm3d_rgb(cam, bm['sigma'])
    return cam


# ───────────────────────────────────────────── blending
BLEND_MODES = {'mean': 'Weighted mean', 'median': 'Weighted median', 'detail': 'Detail-adaptive'}


def blend_images(images, weights, mode='mean'):
    w = np.asarray(weights, np.float64); w = w / w.sum()
    if mode == 'mean' or len(images) == 1:
        acc = np.zeros_like(images[0], np.float32)
        for im, wi in zip(images, w): acc += im * np.float32(wi)
        return acc
    import sys
    if APP_ROOT not in sys.path: sys.path.insert(0, APP_ROOT)
    from rawdemosaic import blend as B
    out = np.empty_like(images[0], np.float32)
    step = max(1, 1_500_000 // images[0].shape[1])   # strips: the median sorts a stack of every image
    f = B.weighted_median if mode == 'median' else B.detail_adaptive
    pad = 0 if mode == 'median' else 8
    for y in range(0, images[0].shape[0], step):
        y0, y1 = max(0, y - pad), min(images[0].shape[0], y + step + pad)
        r = f([im[y0:y1] for im in images], list(w))
        out[y:y + step] = r[y - y0:y - y0 + min(step, images[0].shape[0] - y)]
    return out


def masked_blend(images, weights, mask_index, mask, exclude=False):
    """The masked file takes the mask itself as its weight (1 = only it, 0 = not at all); the other files
    share what is left in proportion to their own weights."""
    eff = (1 - mask) if exclude else mask
    others = [i for i in range(len(images)) if i != mask_index]
    tot = sum(weights[i] for i in others)
    out = images[mask_index] * eff[..., None]
    for i in others:
        share = weights[i] / tot if tot > 0 else 1 / len(others)
        out += images[i] * ((1 - eff) * share)[..., None]
    return out


# ───────────────────────────────────────────── super-resolution / restoration
def _down_kernel(d, kind):
    if d <= 1: return [0], [1.0]
    c = (d - 1) / 2
    if kind == 'box': return list(range(d)), [1 / d] * d
    sinc = lambda x: 1.0 if x == 0 else math.sin(math.pi * x) / (math.pi * x)
    off, wt = [], []
    for k in range(math.ceil(c - 3 * d), math.floor(c + 3 * d) + 1):
        x = (k - c) / d
        if abs(x) < 3: off.append(k); wt.append(sinc(x) * sinc(x / 3))
    s = sum(wt)
    return off, [v / s for v in wt]


_KERNELS = {}


def _downscale_t(x, d, off, wt, r0, c0, oh, ow):
    """x: torch (3, R, C). Output pixel (j, i) = sum_k wt[k] * x[r0 + j*d + off[k], c0 + i*d + off[k]] (separable)."""
    import torch
    import torch.nn.functional as F
    if d == 1 and off == [0]:
        return x[:, r0:r0 + oh, c0:c0 + ow]
    kmin, kmax = off[0], off[-1]
    K = kmax - kmin + 1
    key = (tuple(off), tuple(wt), x.dtype, str(x.device))
    k = _KERNELS.get(key)
    if k is None:
        kk = np.zeros(K, np.float32); kk[np.array(off) - kmin] = wt
        k = _KERNELS[key] = torch.from_numpy(kk).to(dtype=x.dtype, device=x.device)
    rs, cs = r0 + kmin, c0 + kmin
    sub = x[:, rs:rs + (oh - 1) * d + K, cs:cs + (ow - 1) * d + K][:, None]
    sub = F.conv2d(sub, k.view(1, 1, 1, K), stride=(1, d))
    sub = F.conv2d(sub, k.view(1, 1, K, 1), stride=(d, 1))
    return sub[:, 0]


def run_restore_chain(enc, chain, models, job):
    """Restoration models applied in turn to the encoded image (any architecture the registry knows)."""
    import sys
    if APP_ROOT not in sys.path: sys.path.insert(0, APP_ROOT)
    from rawdemosaic.dl_models.restore import registry
    for k, p in enumerate(chain):
        job.check(); job.progress(k / max(1, len(chain)), f'Restoration: {os.path.basename(p)}')
        model, meta = models.restore(p)
        enc = registry.run_model(model, meta, enc, avoid_mixed_precision=True)
        job.log('info', f'restoration {os.path.basename(p)} ({meta["architecture"]}) → {enc.shape[1]}×{enc.shape[0]}')
    return enc


def super_resolve(enc, sr, models, job, each=None):
    """Super-resolution blend, as the web app's superResolve: every model's output is scaled to the target
    (Lanczos-3 or box, an integer factor) and the weighted results are summed. Local architectures run in
    fixed tiles with a 16 px margin that is discarded; seam-prone ones (DAT, Restormer, ...) on the whole
    image through the registry. enc: HxWx3 float32 encoded in the output space. Returns float32.
    sr: dict(enabled, target (1, 2 or 4), downscale ('lanczos' | 'box'), models=[dict(path, weight)]).
    each: a list that receives (model file name, its own result scaled to the target) for every model."""
    import sys
    if APP_ROOT not in sys.path: sys.path.insert(0, APP_ROOT)
    from rawdemosaic.dl_models.restore import registry
    torch = _full_precision_torch()
    dev = models.torch_device()
    h, w = enc.shape[:2]
    t = int(sr['target'])
    entries = [m for m in sr['models'] if m.get('path') and m.get('weight', 0) > 0]
    wsum = sum(m['weight'] for m in entries)
    acc = np.zeros((h * t, w * t, 3), np.float32)
    loaded = []
    for m in entries:
        model, meta = models.restore(m['path'])
        s = int(meta.get('scale', 1))
        if s % t or s < t: raise ValueError(f'{os.path.basename(m["path"])} (×{s}) cannot make a ×{t} image')
        d = s // t
        off, wt = _down_kernel(d, sr.get('downscale', 'lanczos'))
        local = not meta.get('seam_prone') and meta['in_channels'] == 3 and meta['out_channels'] == 3 and meta['value_range'] == '0_1'
        loaded.append((m, model, meta, s, d, off, wt, local, m['weight'] / wsum))
    own = {id(m): np.zeros((h * t, w * t, 3), np.float32) for m, *_ in loaded} if each is not None else {}
    core, pad = 512, 32   # the web app uses 256 + 16; a 32 px margin removes most of the remaining seam difference
    T = core + 2 * pad
    # whole-image models first
    for m, model, meta, s, d, off, wt, local, wgt in loaded:
        if local: continue
        job.progress(0, f'Super-resolution ({os.path.basename(m["path"])}, whole image)')
        out = registry.run_model(model, meta, enc, avoid_mixed_precision=True)
        P = max(-off[0], off[-1], 0)
        step = 512
        for y0 in range(0, h * t, step):
            job.check()
            oh = min(step, h * t - y0)
            r_lo, r_hi = y0 * d + off[0], (y0 + oh - 1) * d + off[-1] + 1
            rows = reflect_idx(np.arange(r_lo - P, r_hi + P), out.shape[0])
            cols = reflect_idx(np.arange(-P, out.shape[1] + P), out.shape[1])
            x = torch.from_numpy(np.ascontiguousarray(out[rows][:, cols].transpose(2, 0, 1))).to(dev)
            # x row P is source row r_lo = y0*d + off[0]; x column P is source column 0
            res = _downscale_t(x, d, off, wt, P - off[0], P, oh, w * t)
            r = res.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            acc[y0:y0 + oh] += np.float32(wgt) * r
            if own: own[id(m)][y0:y0 + oh] = r
        del out
    tiles = [(y0, x0) for y0 in range(0, h, core) for x0 in range(0, w, core)]
    locals_ = [e for e in loaded if e[7]]
    total, done, eta = len(tiles) * len(locals_), 0, make_eta()
    for y0, x0 in tiles if locals_ else []:
        ry = reflect_idx(y0 - pad + np.arange(T), h); rx = reflect_idx(x0 - pad + np.arange(T), w)
        chw = torch.from_numpy(np.ascontiguousarray(enc[ry[:, None], rx[None, :]].transpose(2, 0, 1))).to(dev)[None]
        ch_, cw_ = min(core, h - y0), min(core, w - x0)
        oh_, ow_ = ch_ * t, cw_ * t
        tile_acc = torch.zeros((3, oh_, ow_), dtype=torch.float32, device=dev)
        for m, model, meta, s, d, off, wt, local, wgt in locals_:
            job.check()
            with torch.no_grad():
                res = model(chw)[0].float()
            r = _downscale_t(res, d, off, wt, pad * s, pad * s, oh_, ow_).clamp(0, 1)
            tile_acc += wgt * r
            if own: own[id(m)][y0 * t:y0 * t + oh_, x0 * t:x0 * t + ow_] = r.permute(1, 2, 0).cpu().numpy()
            done += 1
            job.progress(done / total, f'Super-resolution ({os.path.basename(m["path"])}): tile {math.ceil(done / len(locals_))}/{len(tiles)}{eta(done, total)}')
        acc[y0 * t:y0 * t + oh_, x0 * t:x0 * t + ow_] += tile_acc.permute(1, 2, 0).cpu().numpy()
    np.clip(acc, 0, 1, out=acc)
    if each is not None:
        for m, *_ in loaded: each.append((os.path.splitext(os.path.basename(m['path']))[0], own[id(m)]))
    return acc


def finish_image(work, space, out, models, job, each=None):
    """Linear working (display orientation) -> encoded output (float32 [0,1]) after the optional
    restoration chain and super-resolution blend. Returns (enc, is_processed); each (a list) receives the
    super-resolution models' individual results."""
    restore, sr = out.get('restore') or [], out.get('sr') or {}
    if not restore and not (sr.get('enabled') and any(m.get('path') and m.get('weight', 0) > 0 for m in sr.get('models', []))):
        return to_output(work, space), False
    enc = to_output(work, space)
    if restore: enc = run_restore_chain(enc, restore, models, job)
    if sr.get('enabled') and any(m.get('path') and m.get('weight', 0) > 0 for m in sr.get('models', [])):
        enc = super_resolve(enc, sr, models, job, each)
    return enc, True


def encoded_to_srgb8(enc, space):
    """Output-space encoded [0,1] -> sRGB uint8 (for the JPEG and display after super-resolution)."""
    if space == 'srgb':
        return np.round(np.clip(enc, 0, 1) * 255).astype(np.uint8)
    M = (working_to('srgb') @ space_to_working(space)).astype(np.float32)
    out = np.empty(enc.shape, np.uint8)
    step = max(1, 2_000_000 // max(1, enc.shape[1]))
    for y in range(0, enc.shape[0], step):
        lin = decode(space, enc[y:y + step]).astype(np.float32) @ M.T
        out[y:y + step] = np.round(encode('srgb', lin) * 255).astype(np.uint8)
    return out


# ───────────────────────────────────────────── whole files
def out_paths(src_path, out_dir, suffix=''):
    stem = os.path.splitext(os.path.basename(src_path))[0] + suffix
    d = out_dir or os.path.dirname(src_path)
    return os.path.join(d, stem)


def unique_stem(stem, suffixes):
    """stem, or stem_2, stem_3, ... so that no stem + suffix (e.g. '.tif', '_col1.tif') names an existing file."""
    cand, n = stem, 1
    while any(os.path.exists(cand + sfx) for sfx in suffixes):
        n += 1; cand = f'{stem}_{n}'
    return cand


def output_suffixes(opts, n_columns=0):
    """The files a run writes for one stem (for unique_stem)."""
    sfx = ['.tif', '.jpg'] + [f'_col{i + 1}.tif' for i in range(n_columns)]
    sr = opts.get('sr') or {}
    if sr.get('save_each'):
        sfx += [f'_sr_{os.path.splitext(os.path.basename(m.get("path", "")))[0]}{e}' for m in sr.get('models', []) for e in ('.tif', '.jpg')]
    return sfx


def quick_preview(src, max_side=1600):
    """A small view of a RAW with no usable embedded preview, as the web app makes it: the mosaic is box-averaged
    by a multiple of the pattern size (each output pixel keeps its CFA colour), then RCD-demosaiced (Bayer) or
    each colour averaged over the block (X-Trans). Returns sRGB uint8 in display orientation."""
    raw, P = src.raw, src.P
    H, W = raw.shape
    f = P * max(1, int(math.floor(max(W, H) / (max_side * P) + 0.5)))
    w, h = W // f, H // f
    if w < 16 or h < 16: raise ValueError('this image is too small for a quick preview')
    block = raw[:h * f, :w * f].astype(np.float32).reshape(h, f, w, f)
    pat = np.asarray(src.pattern).reshape(P, P)
    wb = np.asarray(src.wb, np.float32)
    if P == 2:
        small = np.empty((h, w), np.float32)
        for py in range(2):
            for px in range(2):   # one sample of the output pixel's own colour from each block (f even keeps the phase)
                small[py::2, px::2] = block[py::2, py::2][:, :, px::2, px::2].mean(axis=(1, 3))
        mos = normalize_mosaic(small, src.pattern, 2, src.black, src.white)
        wm = np.minimum(1, mos * wb[cfa_colour_map(src.pattern, 2, h, w)]).astype(np.float32)
        cam = np.minimum(DM.demosaic_rcd(wm, src.pattern), 1)
    else:
        cfa = np.tile(pat, (f // P, f // P))
        scale = 1 / (src.white - np.mean(src.black))
        cam = np.empty((h, w, 3), np.float32)
        for c in range(3):
            m = (cfa == c).astype(np.float32)
            cam[..., c] = np.maximum(0, np.einsum('yixj,ij->yx', block, m) / m.sum() - src.black[c]) * scale
        cam = np.minimum(1, cam * wb)
    apply_matrix(cam, cam_to_working(src.xyz_to_cam), out=cam)
    np.maximum(cam, 0, out=cam)
    return display_image(orient(cam, src.flip), 0)


def process(src, columns, opts, models, job, quick=False, on_column=None):
    """Run every enabled column on a RawFile / ImageInput and blend them.
    opts: blend_mode, out_space, save_columns, out_dir, jpeg, jpeg_quality, tiff, restore, sr, unique.
    on_column(index, column, display uint8, files) is called as each column finishes.
    Returns dict(blend=linear working (display orientation), columns=[(col, display uint8)], files=[...])."""
    t0 = time.time()
    is_xt = (not src.is_rgb) and src.is_xtrans
    cols = [sanitize_column(c, is_xt) for c in columns if c.get('enabled', True) and c.get('weight', 0) > 0]
    if not cols: raise ValueError('Enable at least one column with a weight above 0.')
    if src.is_rgb:
        for c in cols: c['rawDenoise'] = 'none'
    region = None
    if quick:   # centre crop, aligned to the CFA period
        W, H = src.W, src.H
        tw, th = min(W, 1536), min(H, 1020)
        x0, y0 = (W - tw) // 2 // 6 * 6, (H - th) // 2 // 6 * 6
        region = (y0, x0, th, tw)
    src_c = _CropView(src, region) if region else src
    mosaic, cam_to_work = None, None
    if not src.is_rgb:
        cam_to_work = cam_to_working(src.xyz_to_cam)
        if not src.is_xtrans:
            mosaic = normalize_mosaic(src_c.raw, src_c.pattern, src.P, src.black, src.white)
    images, out_cols, files = [], [], []
    stem = out_paths(src.path, opts.get('out_dir'), '_preview' if quick else '')
    if not quick and opts.get('unique', True):   # never replace earlier results: name_2, name_3, ...
        stem = unique_stem(stem, output_suffixes(opts, len(cols)))
    space = opts.get('out_space', 'srgb')
    keep = opts.get('blend_mode', 'mean') != 'mean'
    acc = None
    wts = [c['weight'] for c in cols]
    for ci, col in enumerate(cols):
        job.check()
        job.log('info', f'column {ci + 1}: {col["name"]} ({describe_column(col, is_xt, src.is_rgb)})')
        tc = time.time()
        work = run_column(col, src_c, mosaic, cam_to_work, models, job)
        job.log('ok', f'column {ci + 1} done in {time.time() - tc:.1f} s')
        flip = 0 if src.is_rgb else src.flip
        oriented = orient(work, flip)
        out_cols.append((col, display_image(oriented, 0)))   # full size, like the blend
        col_files = []
        if opts.get('save_columns') and not quick:
            p = f'{stem}_col{ci + 1}.tif'
            write_tiff16(p, to_uint16(to_output(oriented, space)), space)
            files.append(p); col_files.append(p); job.log('info', f'saved {os.path.basename(p)}')
        if on_column: on_column(ci, col, out_cols[-1][1], col_files)
        if keep:
            images.append(oriented)
        else:
            acc = oriented * np.float32(wts[ci] / sum(wts)) if acc is None else acc + oriented * np.float32(wts[ci] / sum(wts))
        del work
    blend = blend_images(images, wts, opts['blend_mode']) if keep else acc
    images = None
    result = dict(blend=blend, columns=out_cols, files=files, seconds=time.time() - t0, quick=quick, stem=stem,
                  region=region or (0, 0, src.H, src.W), src_hw=(src.H, src.W), flip=0 if src.is_rgb else src.flip)
    if not quick:
        files += save_outputs(blend, stem, opts, models, job)
    return result


def save_outputs(work, stem, opts, models, job):
    """Linear working -> TIFF (+ JPEG), after restoration / super-resolution.
    opts['convert'] = dict(path, intent, bpc): convert both to that ICC profile (e.g. a printer's) and embed it."""
    space = opts.get('out_space', 'srgb')
    conv = opts.get('convert')
    if conv:   # from ProPhoto, so that nothing the profile can reproduce is clipped on the way
        import rawdemosaic_colour as CM
        space = 'prophoto'
        lut, icc, label = CM.export_table(space, conv), CM.icc_bytes(conv['path']), CM.info(conv['path'])['desc']
        to8 = lambda e: np.round(np.clip(e, 0, 1) * 255).astype(np.uint8)
    sr = opts.get('sr') or {}
    each = [] if sr.get('save_each') and sr.get('enabled') and sum(1 for m in sr.get('models', []) if m.get('weight', 0) > 0) > 1 else None
    enc, processed = finish_image(work, space, opts, models, job, each)
    if conv:
        job.progress(0.97, f'Converting to {label}')
        each = [(n, CM.convert(img, lut)) for n, img in each or []] or None
        enc = CM.convert(enc, lut)
    files = []
    for name, img in each or []:   # each super-resolution model's own result, at the same size as the blend
        safe = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in name)
        if opts.get('tiff', True):
            p = f'{stem}_sr_{safe}.tif'
            write_tiff16(p, to_uint16(img), space, icc=icc if conv else None); files.append(p); job.log('ok', f'saved {os.path.basename(p)}')
        if opts.get('jpeg'):
            p = f'{stem}_sr_{safe}.jpg'
            write_jpeg(p, to8(img) if conv else encoded_to_srgb8(img, space), opts.get('jpeg_quality', 95), icc=icc if conv else None)
            files.append(p); job.log('ok', f'saved {os.path.basename(p)}')
    each = None
    if opts.get('tiff', True):
        p = stem + '.tif'
        job.progress(0.99, 'Writing the TIFF')
        write_tiff16(p, to_uint16(enc), space, icc=icc if conv else None)
        files.append(p); job.log('ok', f'saved {os.path.basename(p)} ({enc.shape[1]}×{enc.shape[0]}, {label if conv else SPACES[space]["label"]})')
    if opts.get('jpeg'):
        p = stem + '.jpg'
        if conv:
            write_jpeg(p, to8(enc), opts.get('jpeg_quality', 95), icc=icc)
        else:
            write_jpeg(p, encoded_to_srgb8(enc, space) if processed else to_srgb8(work), opts.get('jpeg_quality', 95))
        files.append(p); job.log('ok', f'saved {os.path.basename(p)} ({label if conv else "sRGB"}, quality {opts.get("jpeg_quality", 95)})')
    return files


class _CropView:
    """A centre-cropped view of a RawFile/ImageInput for quick previews."""
    def __init__(self, src, region):
        y0, x0, h, w = region
        self.__dict__.update({k: v for k, v in src.__dict__.items() if not k.startswith('_')})
        self.is_rgb = src.is_rgb
        self.H, self.W = h, w
        if src.is_rgb:
            self.rgb = src.rgb[y0:y0 + h, x0:x0 + w].copy()
        else:
            self.is_xtrans = src.is_xtrans
            self.raw = src.raw[y0:y0 + h, x0:x0 + w]
            P = src.P
            pat = np.asarray(src.pattern).reshape(P, P)
            self.pattern = np.array([pat[(y + y0) % P, (x + x0) % P] for y in range(P) for x in range(P)], np.int64)
            self._src, self._region = src, region

    def xtrans_rgb(self, passes):
        y0, x0, h, w = self._region
        return self._src.xtrans_rgb(passes)[y0:y0 + h, x0:x0 + w]


def describe_column(c, is_xt, is_rgb):
    parts = []
    if not is_rgb:
        if c['rawDenoise'] == 'nind': parts.append('RAW NIND-Unet')
        else:
            if c['rawDenoise'] == 'rawforge': parts.append('RawForge ' + ' → '.join(m for m in (c['rf1'], c['rf2']) if m))
            parts.append(DEMOSAIC_NAMES.get(c['demosaic'], c['demosaic']))
    if c['boost']['enabled']: parts.append(f'boost {",".join(f"{g:g}" for g in c["boost"]["gains"][:4])}…' + ('' if is_rgb else f' ({c["boost"]["stage"]})'))
    if c['bm3d']['mode'] != 'none': parts.append(f'BM3D {c["bm3d"]["mode"]} σ={c["bm3d"]["sigma"]:g}')
    if c['postDenoise'] != 'none': parts.append({'nind_rgb': 'NIND-Unet', 'nind_linear': 'NIND-Unet linear'}[c['postDenoise']])
    return ', '.join(parts) or 'no stages'


# ───────────────────────────────────────────── blending finished files
def load_for_blend(path):
    """A finished image -> linear Rec.2020 (its ICC profile honoured; this app's own column TIFFs exactly)."""
    rgb, info, _ = load_image_linear(path)
    return rgb, info


def blend_files(paths, weights, mode, opts, models, job, mask=None, mask_index=None, exclude=False, out_stem=None):
    images = []
    for k, p in enumerate(paths):
        job.check(); job.progress(k / len(paths), f'Reading {os.path.basename(p)}')
        rgb, info = load_for_blend(p)
        job.log('info', f'{os.path.basename(p)}: {rgb.shape[1]}×{rgb.shape[0]}, {info["profile"]}')
        images.append(rgb)
    th, tw = max(i.shape[0] for i in images), max(i.shape[1] for i in images)
    for k, im in enumerate(images):
        if im.shape[:2] != (th, tw):
            from PIL import Image
            job.log('warn', f'{os.path.basename(paths[k])} resized to {tw}×{th} (Lanczos) to match the largest')
            images[k] = np.dstack([np.asarray(Image.fromarray(im[..., c]).resize((tw, th), Image.LANCZOS)) for c in range(3)]).astype(np.float32)
            np.maximum(images[k], 0, out=images[k])
    if mask is not None:
        if mask.shape != (th, tw):
            from PIL import Image
            mask = np.asarray(Image.fromarray(mask.astype(np.float32)).resize((tw, th), Image.BILINEAR))
        out = masked_blend(images, weights, mask_index, mask.astype(np.float32), exclude)
    else:
        out = blend_images(images, weights, mode)
    images = None
    files = []
    if out_stem:
        files = save_outputs(out, out_stem, opts, models, job)
    return out, files


def expand_feather(mask, grow_px, feather_px):
    """Grow (positive) or shrink (negative) a binary mask by a Euclidean distance, then feather its edge."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    m = np.asarray(mask, bool)
    if grow_px > 0: m = distance_transform_edt(~m) <= grow_px
    elif grow_px < 0: m = distance_transform_edt(m) > -grow_px
    out = m.astype(np.float32)
    if feather_px > 0: out = gaussian_filter(out, feather_px / 2.0)
    return np.clip(out, 0, 1)


# ───────────────────────────────────────────── settings
def load_json(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default
