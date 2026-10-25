# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""PANOSTITCH desktop engine: native port of panoramamaker.html's stitching pipeline.

No browser limits: ALIKED+LightGlue runs on ONNX Runtime with CUDA, SIFT is OpenCV's, and the
panorama is rendered in horizontal strips straight into RAM, so its size is bounded only by
memory (TIFF output switches to BigTIFF past 4GB). Geometry, camera model, exposure and blending
follow the web app; the one deliberate difference is orientation — the web app's renderer comes
out upside-down and relies on "flip vertically" being on, this one renders the right way up, so
flip defaults to off here.

GUI-free so it can be scripted and tested; the tkinter app drives it through Job callbacks.
"""
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import tifffile

MATCH_RES = 1280          # matching resolution (longest side), same as the web app
# Largest ALIKED+LightGlue input that runs in a 24GB card's own memory (RTX 3090, measured): 2048px
# takes ~0.2s a pair; 2304-2560px still run but Windows spills to system RAM (~1.1s); ~2750px fails.
ALIKED_MAX_RES = 2048
_ALIKED_FIT = {}          # device -> resolution that last worked after running out of GPU memory
_SESSIONS = {}            # (model path, device) -> AlikedMatcher, reused across stitches
# nvidia-smi without flashing a console window on Windows (the flag is Windows-only: an error elsewhere)
_NO_WINDOW = {'creationflags': 0x08000000} if os.name == 'nt' else {}
FEATHER = 0.12            # blend feather, as a fraction of each photo's width/height
MIN_WEIGHT = 0.02
SIFT_CONTRAST = 0.02      # OpenCV's default 0.04 found too few features on real low-contrast photos
MODEL_RE = r'^raco_aliked_lightglue_pipeline_(k\d+)\.onnx$'
IMAGE_EXTS = ('.jpg', '.jpeg', '.jfif', '.png', '.webp', '.tif', '.tiff', '.bmp')


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


# ───────────────────────────────────────────── image I/O
def exif_info(path):
    """EXIF fields that matter here: orientation (1-8), focal length mm, 35mm-equivalent focal
    length, and exposure value (Hugin's Eev). Missing fields are None; unreadable files give {}."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            sub = ex.get_ifd(0x8769)
    except Exception:
        return {}
    def num(v):
        try: return float(v)
        except Exception: return None
    o = ex.get(0x0112)
    info = {'orientation': int(o) if o in range(1, 9) else 1,
            'focal': num(sub.get(0x920A)), 'focal35': num(sub.get(0xA405)) or None}
    fn, t, iso = num(sub.get(0x829D)), num(sub.get(0x829A)), num(sub.get(0x8827))
    if fn and t and fn > 0 and t > 0:
        info['eev'] = math.log2(fn * fn / t) - (math.log2(iso / 100) if iso and iso > 0 else 0)
    return info


def orient(a, o):
    """Applies EXIF orientation o to stored pixels a, giving the upright image a viewer shows."""
    if a is None or o == 1: return a
    a = {2: lambda: a[:, ::-1], 3: lambda: a[::-1, ::-1], 4: lambda: a[::-1], 5: lambda: a.swapaxes(0, 1),
         6: lambda: a[::-1].swapaxes(0, 1), 7: lambda: a[::-1, ::-1].swapaxes(0, 1),
         8: lambda: a[:, ::-1].swapaxes(0, 1)}[o]()
    return np.ascontiguousarray(a)


def upright_to_stored(pts, o, W, H):
    """Maps Nx2 pixel coordinates in the upright image back to the stored pixel grid (W x H, as
    stored), which is what Hugin's control points refer to. Inverse of orient()."""
    u, v = pts[:, 0], pts[:, 1]
    x, y = {1: (u, v), 2: (W - 1 - u, v), 3: (W - 1 - u, H - 1 - v), 4: (u, H - 1 - v), 5: (v, u),
            6: (v, H - 1 - u), 7: (W - 1 - v, H - 1 - u), 8: (W - 1 - v, u)}[o]
    return np.stack([x, y], 1)


def load_image(path, with_orientation=False):
    """Returns (rgb, alpha): rgb is HxWx3 uint8 or uint16, alpha is HxW uint8 or None — upright,
    i.e. with EXIF orientation applied. with_orientation=True adds the orientation that was applied."""
    rgb, alpha = _load_stored(path)
    o = exif_info(path).get('orientation', 1)
    rgb, alpha = orient(rgb, o), orient(alpha, o)
    return (rgb, alpha, o) if with_orientation else (rgb, alpha)


def _load_stored(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        a = np.asarray(tifffile.imread(path))
        if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[2] not in (3, 4):
            a = np.moveaxis(a, 0, -1)
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        alpha = None
        if a.shape[2] == 4:
            al = a[..., 3]
            alpha = (al > 0).astype(np.uint8) * 255
        a = a[..., :3]
        if a.dtype.kind == 'f':
            a = (np.clip(a, 0, 1) * 65535 + 0.5).astype(np.uint16)
        elif a.dtype not in (np.uint8, np.uint16):
            a = np.clip(a, 0, 65535).astype(np.uint16)
        return np.ascontiguousarray(a), alpha
    data = np.fromfile(path, np.uint8)            # fromfile + imdecode copes with non-ASCII paths
    if ext == '.png':
        a = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)  # keeps 16-bit and alpha
    else:   # orientation is applied by load_image, so the stored→upright mapping is known exactly
        a = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if a is None:
        raise ValueError(f'could not decode {os.path.basename(path)}')
    alpha = None
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    elif a.shape[2] == 4:
        alpha = (a[..., 3] > 0).astype(np.uint8) * 255
        a = a[..., :3]
    return np.ascontiguousarray(a[..., ::-1]), alpha


def to_uint8(rgb):
    if rgb.dtype == np.uint8: return rgb
    return (rgb.astype(np.uint32) * 255 // 65535).astype(np.uint8)


def to_float(rgb):
    return rgb.astype(np.float32) / (65535.0 if rgb.dtype == np.uint16 else 255.0)


def save_image(path, rgb, alpha=None, bit16=None, quality=92):
    """rgb: HxWx3 uint8/uint16. alpha: HxW uint8 (0/255) or None. TIFF is always 16-bit;
    PNG is 16-bit when bit16 (default: when rgb is uint16); JPEG is 8-bit and has no alpha."""
    ext = os.path.splitext(path)[1].lower()
    if bit16 is None: bit16 = rgb.dtype == np.uint16
    if ext in ('.tif', '.tiff'):
        c = rgb if rgb.dtype == np.uint16 else rgb.astype(np.uint16) * 257
        extras = ()
        if alpha is not None:
            c = np.concatenate([c, (alpha.astype(np.uint16) * 257)[..., None]], -1)
            extras = ('unassalpha',)
        tifffile.imwrite(path, c, photometric='rgb', extrasamples=extras, bigtiff=c.nbytes > 3_900_000_000,
                         software='PANOSTITCH desktop')
        return
    if ext == '.png':
        c = (rgb if rgb.dtype == np.uint16 else rgb.astype(np.uint16) * 257) if bit16 else to_uint8(rgb)
        c = c[..., ::-1]
        if alpha is not None:
            a = alpha.astype(c.dtype) * (257 if c.dtype == np.uint16 else 1)
            c = np.concatenate([c, a[..., None]], -1)
        ok, buf = cv2.imencode('.png', np.ascontiguousarray(c), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    elif ext in ('.jpg', '.jpeg'):
        if max(rgb.shape[:2]) > 65500:
            raise ValueError(f'JPEG is limited to 65500px per side and this is {rgb.shape[1]}×{rgb.shape[0]} — save as TIFF or PNG')
        ok, buf = cv2.imencode('.jpg', np.ascontiguousarray(to_uint8(rgb)[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        raise ValueError(f'unsupported output format: {ext}')
    if not ok: raise ValueError(f'could not encode {os.path.basename(path)}')
    buf.tofile(path)


class Shot:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.img, _, self.orientation = load_image(path, with_orientation=True)
        self.h, self.w = self.img.shape[:2]
        # the stored (pre-orientation) size, which Hugin uses
        self.raw_w, self.raw_h = (self.h, self.w) if self.orientation >= 5 else (self.w, self.h)
        self._views = {}
        self.exif = exif_info(path)

    @property
    def exif_focal_px(self):
        """Focal length in full-resolution pixels from the 35mm-equivalent EXIF focal (36mm = the long
        side), or None."""
        f35 = self.exif.get('focal35')
        return max(self.w, self.h) * f35 / 36.0 if f35 else None

    def view(self, res=None):
        """(rgb uint8, gray, scale) for matching: longest side res (None = full resolution).
        scale converts view pixels to full-resolution pixels."""
        key = res or 0
        v = self._views.get(key)
        if v is None:
            s = 1.0 if not res else min(1.0, res / max(self.w, self.h))
            rgb = to_uint8(self.img)
            if s < 0.999:
                rgb = cv2.resize(rgb, (max(2, round(self.w * s)), max(2, round(self.h * s))), interpolation=cv2.INTER_AREA)
            v = (rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), self.w / rgb.shape[1])
            self._views[key] = v
        return v

    def sharpness_map(self):
        """Local sharpness measured on the full-resolution photo (mean |Laplacian| over ~31px), stored
        at 1/8 size. Camera shake and missed focus show here even when invisible at seam resolution.
        Returns (map, factor): map pixel = full-resolution pixel * factor."""
        if getattr(self, '_sharp', None) is None:
            g = self.view()[1].astype(np.float32)
            lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
            lap = cv2.boxFilter(lap, -1, (31, 31))
            sm = cv2.resize(lap, (max(2, self.w // 8), max(2, self.h // 8)), interpolation=cv2.INTER_AREA)
            self._sharp = (sm, sm.shape[1] / self.w)
        return self._sharp

    def stored_points(self, pts):
        """Full-resolution upright pixel coordinates -> stored pixel grid (for Hugin)."""
        return upright_to_stored(np.asarray(pts, np.float64), self.orientation, self.raw_w, self.raw_h)


# ───────────────────────────────────────────── matchers
def find_models(folder):
    import re
    out = {}
    if folder and os.path.isdir(folder):
        for f in os.listdir(folder):
            m = re.match(MODEL_RE, f, re.I)
            if m: out[m.group(1).lower()] = os.path.join(folder, f)
    return dict(sorted(out.items(), key=lambda kv: int(kv[0][1:])))


def resolve_device(device):
    """'auto' -> the least busy CUDA GPU right now (by nvidia-smi utilisation), else cpu."""
    if device != 'auto': return device
    devs = [d for d in list_devices() if d.startswith('cuda')]   # list_devices: ['auto', 'cuda:0', …, 'cpu']
    if not devs: return 'cpu'
    try:
        import subprocess
        r = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=10, **_NO_WINDOW)
        util ={f'cuda:{int(a)}': float(b) for a, b in (l.split(',') for l in r.stdout.strip().splitlines())}
        return min(devs, key=lambda d: util.get(d, 100.0))
    except Exception:
        return devs[0]


def list_devices():
    """['cuda:0', 'cuda:1', ..., 'cpu'] — the CUDA entries only if ONNX Runtime can actually use CUDA."""
    devs = []
    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            try:
                import subprocess
                r = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                   capture_output=True, text=True, timeout=10, **_NO_WINDOW)
                names = [n.strip() for n in r.stdout.splitlines() if n.strip()]
            except Exception:
                names = ['GPU']
            devs = [f'cuda:{i}' for i in range(max(1, len(names)))]
    except Exception:
        pass
    return (['auto'] if devs else []) + devs + ['cpu']


_CUDART = None


def _cuda_set_device(i):
    """Makes GPU i current on the calling thread. CUDA's current device is per thread (new threads
    start on GPU 0), and an ONNX Runtime session made for another GPU fails with "invalid resource
    handle" if run while a different one is current — e.g. on GPU 1 from the app's next task thread."""
    global _CUDART
    if _CUDART is False: return
    try:
        if _CUDART is None:
            import ctypes, glob
            # the CUDA runtime ONNX Runtime already loaded (from the venv's nvidia-* packages, or the system)
            names = ('cudart64_13.dll', 'cudart64_12.dll') if os.name == 'nt' else ('libcudart.so.13', 'libcudart.so.12', 'libcudart.so')
            pattern = 'cudart64_*.dll' if os.name == 'nt' else 'libcudart.so*'
            load = ctypes.WinDLL if os.name == 'nt' else ctypes.CDLL
            if os.name != 'nt':
                # the exact copy ONNX Runtime loaded — a system CUDA (e.g. 12 under WSL) may sit beside the
                # venv's, and setting the device in a different runtime copy would do nothing
                try:
                    with open('/proc/self/maps') as fh:
                        loaded = [l.split()[-1] for l in fh if 'libcudart.so' in l]
                    if loaded: names = (loaded[0],) + names
                except OSError:
                    pass
            for nm in names:
                try:
                    _CUDART = load(nm); break
                except OSError:
                    continue
            if _CUDART is None:
                found = glob.glob(os.path.join(os.path.dirname(os.path.dirname(cv2.__file__)), 'nvidia', '**', pattern), recursive=True)
                _CUDART = load(found[0]) if found else False
        if _CUDART: _CUDART.cudaSetDevice(int(i))
    except Exception:
        _CUDART = False


class AlikedMatcher:
    def __init__(self, model_path, device, job):
        import onnxruntime as ort
        try: ort.preload_dlls()
        except Exception: pass
        ort.set_default_logger_severity(3)   # its warnings (constant folding, memcpy nodes) are noise here
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.device_id = int(device.split(':')[1]) if device.startswith('cuda') else None
        if device.startswith('cuda'):
            _cuda_set_device(self.device_id)
            providers = [('CUDAExecutionProvider', {'device_id': int(device.split(':')[1]),
                                                    'arena_extend_strategy': 'kSameAsRequested'}), 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(model_path, so, providers=providers)
        self.using = self.sess.get_providers()[0]
        if device.startswith('cuda') and self.using != 'CUDAExecutionProvider':
            job.log('warn', 'CUDA was requested but ONNX Runtime fell back to the CPU')

    def match(self, a, b):
        """a, b: RGB uint8 match images. Returns p0 (Nx2), p1 (Nx2), scores (N)."""
        h = -(-max(a.shape[0], b.shape[0]) // 32) * 32
        w = -(-max(a.shape[1], b.shape[1]) // 32) * 32  # ALIKED needs sides divisible by 32
        x = np.zeros((2, 3, h, w), np.float32)
        x[0, :, :a.shape[0], :a.shape[1]] = a.transpose(2, 0, 1) / 255.0
        x[1, :, :b.shape[0], :b.shape[1]] = b.transpose(2, 0, 1) / 255.0
        if self.device_id is not None:
            _cuda_set_device(self.device_id)
        kp, mt, sc = self.sess.run(None, {'images': x})
        sel = mt[:, 0] == 0
        i, j = mt[sel, 1].astype(np.int64), mt[sel, 2].astype(np.int64)
        return kp[0, i].astype(np.float32), kp[1, j].astype(np.float32), sc[sel].astype(np.float32)


class SiftMatcher:
    def __init__(self, nfeatures=0):
        self.sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=SIFT_CONTRAST)
        self.cache = {}

    def features(self, key, gray):
        f = self.cache.get(key)
        if f is None:
            kp, des = self.sift.detectAndCompute(gray, None)
            f = (np.float32([k.pt for k in kp]).reshape(-1, 2), des)
            self.cache[key] = f
        return f

    @staticmethod
    def match_desc(fa, fb, ratio=0.75):
        (pa, da), (pb, db) = fa, fb
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
        flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 64})
        good = [m for m, n in (p for p in flann.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < ratio * n.distance]
        if not good:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
        p0 = pa[[m.queryIdx for m in good]]
        p1 = pb[[m.trainIdx for m in good]]
        sc = np.float32([1.0 / (1.0 + m.distance) for m in good])
        return p0, p1, sc


def ransac_h(p0, p1, thr):
    if len(p0) < 8: return None
    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, thr, maxIters=3000, confidence=0.999)
    if H is None or abs(H[2, 2]) < 1e-12: return None
    m = mask.ravel().astype(bool)
    if m.sum() < 8: return None
    return H / H[2, 2], m


def best_homography(p0, p1, tol_scale=1.0):
    res = None
    for thr in (2.0, 3.0, 4.0):
        r = ransac_h(p0, p1, thr * tol_scale)
        if r is not None and (res is None or r[1].sum() > res[1].sum()): res = r
        if res is not None and res[1].sum() >= 12: break
    return res


def refine_matches(ga, gb, q0, q1, H, scale, radius=12, min_ncc=0.75):
    """Sub-pixel full-resolution positions for matches found on downscaled photos. For each pair,
    a patch around q0 in A is warped by the local (affine) part of H into B's frame and
    cross-correlated over a window around q1 a little wider than the downscale factor; the peak
    is fitted with a parabola. ga, gb: full-resolution uint8 grey. Returns (q1 refined,
    number refined, mean shift); points that are near an edge, featureless or ambiguous keep q1."""
    out = q1.copy()
    s = int(math.ceil(scale)) + 2
    r = radius
    hb, wb = gb.shape
    ha, wa = ga.shape
    n_ok, shift = 0, 0.0
    for k in range(len(q0)):
        x, y = q0[k]
        u, v = q1[k]
        if not (2 * r < x < wa - 2 * r and 2 * r < y < ha - 2 * r): continue   # patch would leave photo A
        d = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        if abs(d) < 1e-9: continue
        px, py = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / d, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / d
        J = np.array([[(H[0, 0] - px * H[2, 0]) / d, (H[0, 1] - px * H[2, 1]) / d],
                      [(H[1, 0] - py * H[2, 0]) / d, (H[1, 1] - py * H[2, 1]) / d]])
        M = np.c_[J, np.array([r, r]) - J @ np.array([x, y])]           # A coords -> template coords
        t = cv2.warpAffine(ga, M, (2 * r + 1, 2 * r + 1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if t.std() < 3: continue                                           # sky, haze: nothing to lock on to
        cx, cy = int(round(u)), int(round(v))
        x0, y0 = cx - r - s, cy - r - s
        if x0 < 0 or y0 < 0 or x0 + 2 * (r + s) + 1 > wb or y0 + 2 * (r + s) + 1 > hb: continue
        reg = gb[y0:y0 + 2 * (r + s) + 1, x0:x0 + 2 * (r + s) + 1]
        c = cv2.matchTemplate(reg, t, cv2.TM_CCOEFF_NORMED)
        _, peak, _, (bx, by) = cv2.minMaxLoc(c)
        if peak < min_ncc or not (0 < bx < c.shape[1] - 1 and 0 < by < c.shape[0] - 1): continue
        dx = (c[by, bx - 1] - c[by, bx + 1]) / (2 * (c[by, bx - 1] - 2 * peak + c[by, bx + 1]) or 1e-9)
        dy = (c[by - 1, bx] - c[by + 1, bx]) / (2 * (c[by - 1, bx] - 2 * peak + c[by + 1, bx]) or 1e-9)
        nu, nv = x0 + bx + r + max(-0.5, min(0.5, dx)), y0 + by + r + max(-0.5, min(0.5, dy))
        shift += math.hypot(nu - u, nv - v)
        out[k] = (nu, nv)
        n_ok += 1
    return out, n_ok, shift / max(1, n_ok)


def mean_luma(gray, x, y, r=6):
    h, w = gray.shape
    x0, y0 = max(0, round(x - r)), max(0, round(y - r))
    x1, y1 = min(w, round(x + r)), min(h, round(y + r))
    if x1 <= x0 or y1 <= y0: return None
    return float(gray[y0:y1, x0:x1].mean())


# ───────────────────────────────────────────── geometry helpers
def apply_h(H, x, y):
    v = H @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


def centering(w, h):
    return np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]], float)


def focals_from_homography(h):
    h = (h / h[2, 2]).ravel()
    out = []
    d1, d2 = h[6] * h[7], (h[7] - h[6]) * (h[7] + h[6])
    v1 = -(h[0] * h[1] + h[3] * h[4]) / d1 if d1 else -1
    v2 = (h[0] ** 2 + h[3] ** 2 - h[1] ** 2 - h[4] ** 2) / d2 if d2 else -1
    if v1 < v2: v1, v2 = v2, v1
    f1 = None
    if v1 > 0 and v2 > 0: f1 = math.sqrt(v1 if abs(d1) > abs(d2) else v2)
    elif v1 > 0: f1 = math.sqrt(v1)
    elif v2 > 0: f1 = math.sqrt(v2)
    d1, d2 = h[0] * h[3] + h[1] * h[4], h[0] ** 2 + h[1] ** 2 - h[3] ** 2 - h[4] ** 2
    v1 = -h[2] * h[5] / d1 if d1 else -1
    v2 = (h[5] ** 2 - h[2] ** 2) / d2 if d2 else -1
    if v1 < v2: v1, v2 = v2, v1
    f0 = None
    if v1 > 0 and v2 > 0: f0 = math.sqrt(v1 if abs(d1) > abs(d2) else v2)
    elif v1 > 0: f0 = math.sqrt(v1)
    elif v2 > 0: f0 = math.sqrt(v2)
    return [f for f in (f1, f0) if f is not None and math.isfinite(f) and f > 0]


def orthonormalize(Rraw):
    c0, c1 = Rraw[:, 0].copy(), Rraw[:, 1].copy()
    n0 = np.linalg.norm(c0)
    if not n0 > 1e-9: return None
    c0 /= n0
    c1 -= c0.dot(c1) * c0
    n1 = np.linalg.norm(c1)
    if not n1 > 1e-9: return None
    c1 /= n1
    return np.stack([c0, c1, np.cross(c0, c1)], 1)


def rays(shot, pts, f):
    """Unit camera rays for pixel positions (same centring as optimize_camera_model)."""
    v = np.c_[(pts[:, 0] - shot.w / 2) / f, (pts[:, 1] - shot.h / 2) / f, np.ones(len(pts))]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def kabsch(A, B):
    """Rotation R minimising |R a - b| over rows of A, B (unit rays)."""
    U, _, Vt = np.linalg.svd(B.T @ A)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def pair_rotation(ra, rb, f, thr, iters=150, rng=None):
    """RANSAC rotation with rb ≈ R ra (a pure camera rotation). Returns (R, inlier mask, pixel residuals)."""
    n = len(ra)
    if n < 3: return None, np.zeros(n, bool), np.full(n, np.inf)
    rng = rng or np.random.default_rng(1)
    best = None
    for _ in range(iters):
        s = rng.choice(n, 3, replace=False)
        R = kabsch(ra[s], rb[s])
        cnt = int((np.linalg.norm(ra @ R.T - rb, axis=1) * f < thr).sum())
        if best is None or cnt > best[0]: best = (cnt, R)
    R = best[1]
    for _ in range(2):                      # polish on the inliers
        m = np.linalg.norm(ra @ R.T - rb, axis=1) * f < thr
        if m.sum() >= 3: R = kabsch(ra[m], rb[m])
    res = np.linalg.norm(ra @ R.T - rb, axis=1) * f
    return R, res < thr, res


def rodrigues(v):
    th = float(np.linalg.norm(v))
    if th < 1e-12: return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def forward_project(mode, ray, outF):
    r = ray / np.linalg.norm(ray)
    if mode == 'rectilinear':
        if r[2] <= 1e-4: return None
        return outF * r[0] / r[2], outF * r[1] / r[2]
    if mode == 'cylindrical':
        d = math.hypot(r[0], r[2])
        if d < 1e-6: return None
        return outF * math.atan2(r[0], r[2]), outF * r[1] / d
    return outF * math.atan2(r[0], r[2]), outF * math.asin(max(-1.0, min(1.0, r[1])))


# ───────────────────────────────────────────── graph
def connected_components(n, edges):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for e in edges:
        a, b = find(e['i']), find(e['j'])
        if a != b: parent[b] = a
    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def max_spanning_tree(nodes, edges):
    parent = {n: n for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    tree = []
    for e in sorted(edges, key=lambda e: -e['inliers']):
        a, b = find(e['i']), find(e['j'])
        if a != b:
            parent[b] = a; tree.append(e)
    return tree


# ───────────────────────────────────────────── bundle adjustment / camera model
def bundle_adjust(cumulative, edges, ref, job):
    """Re-optimise every placement jointly against all edges' matches (forward-transfer error)."""
    from scipy.optimize import least_squares
    nodes = [k for k in cumulative if k != ref]
    usable = [e for e in edges if len(e['pts']) and e['i'] in cumulative and e['j'] in cumulative]
    if not nodes or not usable: return
    idx = {n: i for i, n in enumerate(nodes)}
    x0 = np.concatenate([(cumulative[n] / cumulative[n][2, 2]).ravel()[:8] for n in nodes])

    def getH(p, n):
        if n == ref: return np.eye(3)
        k = idx[n] * 8
        return np.append(p[k:k + 8], 1.0).reshape(3, 3)

    def residuals(p):
        out = []
        for e in usable:
            try:
                rel = np.linalg.inv(getH(p, e['j'])) @ getH(p, e['i'])
            except np.linalg.LinAlgError:
                out.append(np.full(len(e['pts']) * 2, 1e4)); continue
            P = e['pts']
            v = np.c_[P[:, 0], P[:, 1], np.ones(len(P))] @ rel.T
            d = v[:, 2]
            bad = ~(np.abs(d) > 1e-12)
            d = np.where(bad, 1.0, d)
            r = np.c_[v[:, 0] / d - P[:, 2], v[:, 1] / d - P[:, 3]]
            r[bad] = 1e4
            out.append(r.ravel())
        return np.concatenate(out)

    r0 = residuals(x0)
    before = float(r0 @ r0)
    if not np.isfinite(before) or before == 0: return
    method = 'lm' if len(r0) >= len(x0) else 'trf'
    sol = least_squares(residuals, x0, method=method, x_scale='jac', max_nfev=200 * len(x0))
    after = float(sol.fun @ sol.fun)
    if after < before:
        for n in nodes: cumulative[n] = getH(sol.x, n)
        job.log('ok', f'bundle adjustment: match residual {math.sqrt(before / len(r0)):.2f}px -> {math.sqrt(after / len(r0)):.2f}px RMS')


def optimize_camera_model(rotations, edges, shots, ref, f, job, fix_f=False):
    """Jointly refine per-photo rotations and the shared focal length (unless fix_f, e.g. from EXIF)
    against the matches, in the same pinhole model the renderer uses. Robust (soft-L1) loss: matches
    on near foreground that moved with the photographer (parallax) can't be fitted by any rotation
    and would otherwise drag the solution — notably the focal length. Returns the focal."""
    from scipy.optimize import least_squares
    nodes = [k for k in rotations if k != ref]
    usable = [e for e in edges if len(e['pts']) and e['i'] in rotations and e['j'] in rotations]
    if not nodes or not usable or not f > 0: return f
    idx = {n: i for i, n in enumerate(nodes)}
    seed = dict(rotations)
    P = len(nodes) * 3 + (0 if fix_f else 1)
    fs = 0.002 * max(max(shots[n].w, shots[n].h) for n in rotations)   # ~12px on a 26MP photo

    def robust_cost(r):
        return 0.5 * fs * fs * float(np.sum(2 * (np.sqrt(1 + (r / fs) ** 2) - 1)))

    def rot(p, n):
        if n == ref: return np.eye(3)
        k = idx[n] * 3
        return rodrigues(p[k:k + 3]) @ seed[n]

    def residuals(p):
        fo = f if fix_f else math.exp(p[-1])
        out = []
        for e in usable:
            Ri, Rj = rot(p, e['i']), rot(p, e['j'])
            si, sj = shots[e['i']], shots[e['j']]
            Q = e['pts']
            r0 = np.c_[(Q[:, 0] - si.w / 2) / fo, (Q[:, 1] - si.h / 2) / fo, np.ones(len(Q))]
            jv = r0 @ (Rj.T @ Ri).T
            z = jv[:, 2]
            bad = ~(z > 1e-6)
            z = np.where(bad, 1.0, z)
            r = np.c_[fo * jv[:, 0] / z + sj.w / 2 - Q[:, 2], fo * jv[:, 1] / z + sj.h / 2 - Q[:, 3]]
            r[bad] = 1e4
            out.append(r.ravel())
        return np.concatenate(out)

    seed0 = math.log(f)
    p0 = np.zeros(P)
    if not fix_f: p0[-1] = seed0
    r0 = residuals(p0)
    start = robust_cost(r0)
    best = None
    offsets = (0,) if fix_f else (0, math.log(2), -math.log(2), 2 * math.log(2), -2 * math.log(2))
    for off in offsets:
        p = np.zeros(P)
        if not fix_f: p[-1] = seed0 + off
        try:
            sol = least_squares(residuals, p, method='trf', loss='soft_l1', f_scale=fs, max_nfev=100 * P)
        except Exception:
            continue
        cost = robust_cost(sol.fun)
        if best is None or cost < best[1]: best = (sol.x, cost, sol.fun)
    if best is None or not best[1] < start: return f
    p, cost, r1 = best
    fNew = f if fix_f else math.exp(p[-1])
    for n in nodes: rotations[n] = rot(p, n)
    a0, a1 = np.abs(r0[np.abs(r0) < 1e3]), np.abs(r1[np.abs(r1) < 1e3])
    job.log('ok', f'camera model: f {round(fNew)}px ' + ('(EXIF)' if fix_f else f'(from {round(f)}px)') +
                  f', match residual median {np.median(a0):.2f}px -> {np.median(a1):.2f}px')
    return fNew


def compute_gains(nodes, edges, ref):
    idx = {n: i for i, n in enumerate(nodes)}
    rows, rhs = [], []
    for e in edges:
        r = e.get('ratio')
        if r is None or not math.isfinite(r) or r <= 0: continue
        row = np.zeros(len(nodes)); row[idx[e['j']]] = 1; row[idx[e['i']]] = -1
        rows.append(row); rhs.append(math.log(r))
    if not rows: return {n: 1.0 for n in nodes}
    anchor = np.zeros(len(nodes)); anchor[idx[ref]] = 1000.0
    A = np.vstack(rows + [anchor]); b = np.array(rhs + [0.0])
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    return {n: math.exp(max(-1.5, min(1.5, sol[idx[n]]))) for n in nodes}


# ───────────────────────────────────────────── stitching
class Stitcher:
    def __init__(self, shots, job, matcher='auto', model_path=None, device='auto', projection='auto',
                 exposure=False, flip=False, scale=1.0, match_res=None, warp=False, warp_stiffness=1.0,
                 seams=False, seam_blend=None, multiband=False):
        """match_res: longest side photos are matched at (None = full resolution). warp: bend each
        photo locally onto its already-placed neighbours where near ground shows parallax
        (Layout.fit_warp); warp_stiffness: higher = gentler, more rigid."""
        self.warp, self.warp_stiffness = warp, warp_stiffness
        self.seams, self.seam_blend = seams, seam_blend   # seam cutting (Layout.find_seams); blend in seam-map px
        self.multiband = multiband                        # multi-band blending across the seams (Layout._multiband)
        self.shots, self.job = shots, job
        self.matcher_mode, self.model_path, self.device = matcher, model_path, device
        self.projection, self.exposure, self.flip, self.scale = projection, exposure, flip, scale
        self.match_res = match_res
        # ALIKED's dense feature maps grow with every pixel: 2560px fits a 24GB card, ~2750px doesn't.
        # "Full resolution" therefore means: match at the largest size that fits, then refine every
        # point on the full-resolution photos (refine_matches). Lowered further if the GPU runs out.
        self.aliked_res = min(match_res or ALIKED_MAX_RES, _ALIKED_FIT.get(device, ALIKED_MAX_RES))
        self.refine = match_res is None
        self.aliked = None
        big = match_res is None or match_res > 2048
        self.sift = SiftMatcher(nfeatures=30000 if big else 0)   # uncapped SIFT on 26MP finds 100k+ points
        self.edges = []

    # ── matching
    def _ensure_aliked(self):
        if self.aliked is None:
            if not self.model_path or not os.path.isfile(self.model_path):
                raise RuntimeError('no ALIKED model file — choose a models folder, or use SIFT')
            if self.device == 'auto':
                self.device = resolve_device('auto')
                self.aliked_res = min(self.aliked_res, _ALIKED_FIT.get(self.device, ALIKED_MAX_RES))
                self.job.log('info', f'auto GPU choice: {self.device} (least busy)')
            key = (os.path.normcase(os.path.abspath(self.model_path)), self.device)
            self.aliked = _SESSIONS.get(key)
            if self.aliked is None:
                import time
                t0 = time.time()
                # one session per model+GPU, kept: a second one would hold its own GPU memory
                for k in [k for k in _SESSIONS if k[1] == self.device]: del _SESSIONS[k]
                self.aliked = _SESSIONS[key] = AlikedMatcher(self.model_path, self.device, self.job)
                self.job.log('ok', f'{os.path.basename(self.model_path)} loaded on {self.aliked.using} in {time.time() - t0:.1f}s')
        return self.aliked

    @staticmethod
    def _to_full(p, s):
        """View pixel coordinates -> full resolution (pixel centres at integers)."""
        return p if s == 1.0 else (p + 0.5) * s - 0.5

    def _sift_pair(self, i, j):
        (_, ga, sa), (_, gb, sb) = self.shots[i].view(self.match_res), self.shots[j].view(self.match_res)
        p0, p1, sc = SiftMatcher.match_desc(self.sift.features(i, ga), self.sift.features(j, gb))
        return self._to_full(p0, sa), self._to_full(p1, sb), sc, 'sift'

    def _aliked_pair(self, i, j):
        while True:
            (ra, _, sa), (rb, _, sb) = self.shots[i].view(self.aliked_res), self.shots[j].view(self.aliked_res)
            try:
                p0, p1, sc = self._ensure_aliked().match(ra, rb)
                return self._to_full(p0, sa), self._to_full(p1, sb), sc
            except Exception as e:
                if not ('memory' in str(e).lower() or 'alloc' in str(e).lower()): raise
                cur = max(ra.shape[:2] + rb.shape[:2])
                if cur <= 1280: raise
                self.aliked_res = max(1280, int(cur * 0.8))
                _ALIKED_FIT[self.device] = self.aliked_res     # remembered for later runs in this session
                self.job.log('warn', f'not enough GPU memory to match at {cur}px — using {self.aliked_res}px from now on')

    def match_pair(self, i, j):
        mode = self.matcher_mode
        if mode == 'sift': return self._sift_pair(i, j)
        try:
            p0, p1, sc = self._aliked_pair(i, j)
            if mode == 'aliked' or len(p0) >= 12: return p0, p1, sc, 'aliked'
            self.job.log('warn', f'ALIKED found only {len(p0)} matches, trying SIFT…')
        except Exception as e:
            if mode == 'aliked': raise
            self.job.log('warn', f'ALIKED failed ({e}), falling back to SIFT')
        return self._sift_pair(i, j)

    def match_all(self, window=0):
        """window > 0 only tries pairs at most that many places apart in the list (for long sequences)."""
        n = len(self.shots)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if not window or j - i <= window]
        self.job.log('info', f'matching {n} photos ({len(pairs)} candidate pairs)…')
        edges = []
        for k, (i, j) in enumerate(pairs):
            self.job.check()
            a, b = self.shots[i], self.shots[j]
            label = f'{a.name} × {b.name}'
            self.job.progress(0.5 * k / max(1, len(pairs)), f'matching {k + 1}/{len(pairs)}: {label}')
            try:
                p0, p1, sc, method = self.match_pair(i, j)
            except Cancelled:
                raise
            except Exception as e:
                self.job.log('warn', f'{label}: {e}'); continue
            # points are in full-resolution pixels; the RANSAC tolerance keeps its old meaning (2-4px at 1280)
            res = best_homography(p0, p1, tol_scale=max(1.0, max(a.w, a.h, b.w, b.h) / 1280))
            if res is None or res[1].sum() < 12 or res[1].sum() / len(p0) < 0.25:
                self.job.log('info', f'{label}: {method} {len(p0)} matches, '
                                     f'{"no consistent fit" if res is None else f"{res[1].sum()} inliers"} — no overlap assumed')
                continue
            H, m = res
            inl = int(m.sum())
            # matches for the parallax warp: the homography rejects near-ground points that moved with
            # the photographer, an epipolar (fundamental matrix) test keeps them and still drops mismatches
            wm = m.copy()
            if len(p0) >= 15:
                try:
                    F, fm = cv2.findFundamentalMat(p0.astype(np.float32), p1.astype(np.float32), cv2.FM_RANSAC,
                                                   max(1.0, max(a.w, a.h, b.w, b.h) / 1280), 0.999)
                    if F is not None and fm is not None: wm |= fm.ravel().astype(bool)
                except cv2.error:
                    pass
            w0, w1 = p0[wm].astype(np.float64), p1[wm].astype(np.float64)
            note = ''
            if self.refine and method == 'aliked' and self.aliked_res < max(a.w, a.h):
                w1, ok, moved = refine_matches(a.view()[1], b.view()[1], w0, w1, H, max(a.w, a.h) / self.aliked_res)
                note = f', {ok}/{len(w0)} refined at full resolution (mean shift {moved:.2f}px)'
            hsel = m[wm]                                   # the homography inliers among them
            q0, q1 = w0[hsel], w1[hsel]
            if wm.sum() > m.sum(): note += f', +{int(wm.sum() - m.sum())} parallax matches'
            ratio = None
            ls = [(mean_luma(a.view(1280)[1], *q0[t] * a.view(1280)[1].shape[1] / a.w),
                   mean_luma(b.view(1280)[1], *q1[t] * b.view(1280)[1].shape[1] / b.w)) for t in range(min(12, len(q0)))]
            ls = sorted(lb / la for la, lb in ls if la and lb and la > 2 and lb > 2)
            if ls: ratio = ls[len(ls) // 2]
            step = max(1, math.ceil(len(q0) / 400))
            pts = np.c_[q0[::step], q1[::step]]
            edges.append({'i': i, 'j': j, 'H': H, 'inliers': inl, 'total': len(p0), 'ratio': ratio, 'pts': pts,
                          'cp': (q0, q1, np.asarray(sc, np.float64)[m]), 'warp': (w0, w1), 'method': method})
            self.job.log('ok', f'{label}: {method} {inl}/{len(p0)} inliers{note}')
        self.edges = edges
        return edges

    # ── layout
    def _camera_model(self, group, edges, ref, cumulative, quiet=False):
        """Focal length (full-resolution pixels) and a rotation per photo, from the homographies,
        refined jointly. quiet: problems are logged at info level (rectilinear output doesn't need it)."""
        shots, job = self.shots, self.job
        warn = 'info' if quiet else 'warn'
        focals = []
        for e in edges:
            si, sj = shots[e['i']], shots[e['j']]
            Hc = centering(sj.w, sj.h) @ e['H'] @ np.linalg.inv(centering(si.w, si.h))
            focals += [x for x in focals_from_homography(Hc) if x > 10]
        # EXIF first, as Hugin does: on scenes with near foreground (parallax) the matches pull a free
        # focal length far off (11000px vs EXIF 7100px on the test set). Only trusted when every photo
        # reports the same lens; a zoom used at several focal lengths falls back to estimating.
        exif = [shots[n].exif_focal_px for n in group if n in cumulative]
        fix_f = bool(exif) and all(exif) and max(exif) / min(exif) < 1.02
        if fix_f: f = float(np.median(exif))
        elif focals: f = float(np.median(focals))
        else:
            f = 0.85 * max(shots[ref].w, shots[ref].h)
            job.log(warn, 'could not estimate focal length from matches — using a default lens assumption')
        # starting rotations: fit a rotation to each pair's matches directly (RANSAC) and chain the
        # strongest pairs out from the reference. Orthonormalising chained homographies (the fallback
        # below) can start a photo tens of pixels off, and the optimiser then stalls short of it.
        pairR = {}
        thr = max(3.0, 0.0008 * max(max(shots[n].w, shots[n].h) for n in group))
        for e in edges:
            P = e['pts']
            if len(P) < 8: continue
            R, m, _ = pair_rotation(rays(shots[e['i']], P[:, :2], f), rays(shots[e['j']], P[:, 2:], f), f, thr)
            if R is not None and m.sum() >= 8: pairR[id(e)] = (R, int(m.sum()))
        chained = {ref: np.eye(3)}
        strong = sorted((e for e in edges if id(e) in pairR), key=lambda e: -pairR[id(e)][1])
        for _ in range(len(group)):
            for e in strong:
                Rij = pairR[id(e)][0]                    # ray_j = Rij ray_i  =>  R_j = R_i Rij^T
                if e['i'] in chained and e['j'] not in chained: chained[e['j']] = chained[e['i']] @ Rij.T
                elif e['j'] in chained and e['i'] not in chained: chained[e['i']] = chained[e['j']] @ Rij
        rotations = {ref: np.eye(3)}
        for n in group:
            if n == ref or n not in cumulative: continue
            if n in chained:
                rotations[n] = chained[n]; continue
            Hc = centering(shots[ref].w, shots[ref].h) @ cumulative[n] @ np.linalg.inv(centering(shots[n].w, shots[n].h))
            Rraw = np.array([[Hc[0, 0], Hc[0, 1], Hc[0, 2] / f], [Hc[1, 0], Hc[1, 1], Hc[1, 2] / f],
                             [Hc[2, 0] * f, Hc[2, 1] * f, Hc[2, 2]]])
            R = orthonormalize(Rraw)
            if R is None:
                job.log(warn, f'{shots[n].name}: degenerate rotation, skipped for curved projection'); continue
            rotations[n] = R
        try:
            f = optimize_camera_model(rotations, edges, shots, ref, f, job, fix_f=fix_f)
        except Exception as ex:
            job.log(warn, f'camera model refinement skipped: {ex}')
        return rotations, f

    def layout_group(self, group, all_edges):
        shots, job = self.shots, self.job
        edges = [e for e in all_edges if e['i'] in group and e['j'] in group]
        tree = max_spanning_tree(group, edges)
        adj = {g: [] for g in group}
        for e in tree:
            adj[e['i']].append((e['j'], e, True)); adj[e['j']].append((e['i'], e, False))
        degree = {g: 0 for g in group}
        for e in edges:
            degree[e['i']] += e['inliers']; degree[e['j']] += e['inliers']
        ref = max(group, key=lambda g: degree[g])
        cumulative = {ref: np.eye(3)}
        queue = [ref]
        while queue:
            cur = queue.pop(0)
            for to, e, fwd in adj[cur]:
                if to in cumulative: continue
                i = cur if fwd else to                     # e['H'] maps e.i -> e.j
                try:
                    m = np.linalg.inv(e['H']) if cur == i else e['H']
                except np.linalg.LinAlgError:
                    continue
                cumulative[to] = cumulative[cur] @ m
                queue.append(to)
        try:
            bundle_adjust(cumulative, edges, ref, job)
        except Exception as ex:
            job.log('warn', f'bundle adjustment skipped: {ex}')
        placed = [g for g in group if g in cumulative]
        if len(placed) < 2: raise RuntimeError('not enough photos could be geometrically placed')
        if self.exposure:
            gains = compute_gains(placed, [e for e in edges if e['i'] in placed and e['j'] in placed], ref)
            job.log('info', 'exposure gains: ' + ', '.join(f'{shots[n].name} ×{gains[n]:.2f}' for n in placed))
        else:
            gains = {n: 1.0 for n in placed}

        GRID = 8
        pts = []
        for n in placed:
            s = shots[n]
            for gx in range(GRID + 1):
                for gy in range(GRID + 1):
                    pts.append(apply_h(cumulative[n], gx / GRID * s.w, gy / GRID * s.h))
        pts = np.array(pts)
        planar = (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())
        avgW = np.mean([shots[n].w for n in placed]); avgH = np.mean([shots[n].h for n in placed])
        planar_wide = (planar[1] - planar[0]) > 2.6 * avgW or (planar[3] - planar[2]) > 2.6 * avgH

        mode = ('cylindrical' if planar_wide else 'rectilinear') if self.projection == 'auto' else self.projection
        # the camera model drives curved projections, and also gives Hugin its starting lens and positions
        quiet = mode == 'rectilinear'
        try:
            rotations, f = self._camera_model(group, edges, ref, cumulative, quiet)
        except Exception as ex:
            if not quiet: raise
            rotations, f = None, None
            job.log('info', f'camera model not solved ({ex}) — Hugin export will start from EXIF values')
        if mode != 'rectilinear':
            if self.projection == 'auto':
                maxPitch = max(abs(math.asin(max(-1, min(1, rotations[n][1, 2])))) for n in placed if n in rotations)
                if maxPitch > 0.6: mode = 'equirectangular'
        render_nodes = placed if mode == 'rectilinear' else [n for n in placed if n in rotations]
        if len(render_nodes) < 2: raise RuntimeError('not enough photos could be placed on the chosen projection')

        if mode == 'rectilinear':
            minX, maxX, minY, maxY = planar
        else:
            ps = []
            for n in render_nodes:
                s, R = shots[n], rotations[n]
                for gx in range(GRID + 1):
                    for gy in range(GRID + 1):
                        ray = R @ np.array([(gx / GRID * s.w - s.w / 2) / f, (gy / GRID * s.h - s.h / 2) / f, 1.0])
                        p = forward_project(mode, ray, f)
                        if p is not None: ps.append(p)
            ps = np.array(ps)
            minX, maxX, minY, maxY = ps[:, 0].min(), ps[:, 0].max(), ps[:, 1].min(), ps[:, 1].max()
        sc = self.scale
        W, H = max(8, round(math.ceil(maxX - minX) * sc)), max(8, round(math.ceil(maxY - minY) * sc))
        inl = sum(e['inliers'] for e in edges if e['i'] in placed and e['j'] in placed)
        L = Layout(self.shots, render_nodes, mode, cumulative, rotations, f, gains, minX, minY, sc, W, H, self.flip, inl)
        L.edges, L.ref = edges, ref       # kept for Hugin export and the parallax warp
        if self.warp:
            job.check()
            job.progress(0.5, 'fitting the parallax warp')
            L.warp_stats = L.fit_warp(self.warp_stiffness, job)
        if self.seams or self.seam_blend is not None:
            # seam cutting chooses WHERE photos meet (graph cut); without it, at the nearest-centre line.
            # seam_blend None = the classic wide feather, no seam masks at all
            job.check()
            L.find_seams(job, blend_px=self.seam_blend if self.seam_blend is not None else 2.0,
                         method='graphcut' if self.seams else 'nearest')
            L.multiband = self.multiband
        return L

    def run(self):
        """Matches everything and lays out each connected group. Returns (layouts, unmatched_names)."""
        edges = self.match_all()
        if not edges: raise RuntimeError('no two photos could be matched to each other — nothing to stitch')
        groups = connected_components(len(self.shots), edges)
        matched = {e['i'] for e in edges} | {e['j'] for e in edges}
        unmatched = [self.shots[i].name for i in range(len(self.shots)) if i not in matched]
        if unmatched: self.job.log('warn', f"didn't overlap with anything, skipped: {', '.join(unmatched)}")
        layouts = []
        for g in groups:
            if len(g) < 2: continue
            self.job.check()
            try:
                layouts.append(self.layout_group(g, edges))
            except Cancelled:
                raise
            except Exception as ex:
                self.job.log('err', f'a group of {len(g)} photos could not be laid out: {ex}')
        return layouts, unmatched


# ───────────────────────────────────────────── local parallax warp
class WarpField:
    """A smooth displacement over one photo (source-pixel offsets on a coarse grid, bilinear between
    nodes). The renderer samples the photo at s + D(s) instead of s, s from the global model."""
    def __init__(self, w, h, cell):
        self.w, self.h, self.cell = w, h, cell
        self.nx, self.ny = max(1, math.ceil(w / cell)), max(1, math.ceil(h / cell))
        self.dx = np.zeros((self.ny + 1, self.nx + 1))
        self.dy = np.zeros((self.ny + 1, self.nx + 1))

    def _bilinear(self, sx, sy):
        gx = np.clip(np.asarray(sx, np.float64) / self.cell, 0, self.nx)
        gy = np.clip(np.asarray(sy, np.float64) / self.cell, 0, self.ny)
        i, j = np.minimum(gx.astype(np.int64), self.nx - 1), np.minimum(gy.astype(np.int64), self.ny - 1)
        fx, fy = gx - i, gy - j
        return i, j, fx, fy

    def sample(self, sx, sy):
        i, j, fx, fy = self._bilinear(sx, sy)
        def interp(g):
            return ((g[j, i] * (1 - fx) + g[j, i + 1] * fx) * (1 - fy) +
                    (g[j + 1, i] * (1 - fx) + g[j + 1, i + 1] * fx) * fy)
        return interp(self.dx), interp(self.dy)

    def invert(self, q, iters=4):
        """Source position s with s + D(s) = q (the field is smooth and small, so fixed-point works)."""
        s = q.copy()
        for _ in range(iters):
            dx, dy = self.sample(s[:, 0], s[:, 1])
            s = q - np.c_[dx, dy]
        return s

    def fit(self, pts, disp, stiffness=1.0, robust_px=3.0, iters=4):
        """Least squares: D(pts) ≈ disp, plus a bending penalty (second differences: smooth, zoom- or
        shear-like corrections — what a step forward between shots produces — cost nothing, only
        wiggles do), a little membrane against ripples, and a weak pull to zero so the correction
        fades out away from the matches. Each grid cell's data counts as at most one observation, so
        dense clusters don't overpower smoothness; reweighted to shrug off the odd bad match.
        Returns per-point residuals of the final fit."""
        import scipy.sparse as sp
        from scipy.sparse.linalg import spsolve
        nN = (self.nx + 1) * (self.ny + 1)
        node = lambda i, j: j * (self.nx + 1) + i
        i, j, fx, fy = self._bilinear(pts[:, 0], pts[:, 1])
        k = len(pts)
        cols = np.stack([node(i, j), node(i + 1, j), node(i, j + 1), node(i + 1, j + 1)], 1)
        vals = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1)
        B = sp.csr_matrix((vals.ravel(), (np.repeat(np.arange(k), 4), cols.ravel())), shape=(k, nN))
        cell_id = j * self.nx + i
        per_cell = np.bincount(cell_id, minlength=self.nx * self.ny)[cell_id]
        base_w = 1.0 / per_cell
        ii, jj = np.meshgrid(np.arange(self.nx + 1), np.arange(self.ny + 1))

        def diff_rows(stencil_nodes, coefs):
            n = stencil_nodes[0].size
            r = np.concatenate([np.arange(n)] * len(coefs))
            c = np.concatenate([s.ravel() for s in stencil_nodes])
            v = np.concatenate([np.full(n, k, float) for k in coefs])
            return sp.csr_matrix((v, (r, c)), shape=(n, nN))
        first = sp.vstack([diff_rows((node(ii[:, :-1], jj[:, :-1]), node(ii[:, 1:], jj[:, 1:])), (1, -1)),
                           diff_rows((node(ii[:-1, :], jj[:-1, :]), node(ii[1:, :], jj[1:, :])), (1, -1))])
        bend = [diff_rows((node(ii[:, :-2], jj[:, :-2]), node(ii[:, 1:-1], jj[:, 1:-1]), node(ii[:, 2:], jj[:, 2:])), (1, -2, 1)),
                diff_rows((node(ii[:-2, :], jj[:-2, :]), node(ii[1:-1, :], jj[1:-1, :]), node(ii[2:, :], jj[2:, :])), (1, -2, 1))]
        if self.nx >= 1 and self.ny >= 1:   # cross term d²/dxdy keeps shears smooth too
            bend.append(diff_rows((node(ii[:-1, :-1], jj[:-1, :-1]), node(ii[:-1, 1:], jj[:-1, 1:]),
                                   node(ii[1:, :-1], jj[1:, :-1]), node(ii[1:, 1:], jj[1:, 1:])), (1, -1, -1, 1)))
        bend = sp.vstack([b for b in bend if b.shape[0]])
        lam = stiffness
        reg = lam * (bend.T @ bend) + 0.01 * lam * (first.T @ first) + 0.001 * lam * sp.identity(nN)
        w = base_w.copy()
        for _ in range(iters):
            W = sp.diags(w)
            A = (B.T @ W @ B + reg).tocsc()
            x = spsolve(A, B.T @ (w * disp[:, 0]))
            y = spsolve(A, B.T @ (w * disp[:, 1]))
            res = np.hypot(B @ x - disp[:, 0], B @ y - disp[:, 1])
            sigma = max(robust_px, 2.5 * float(np.median(res)))   # adapt: big smooth parallax isn't "outlier"
            w = base_w / (1 + (res / sigma) ** 2)
        self.dx, self.dy = x.reshape(self.ny + 1, self.nx + 1), y.reshape(self.ny + 1, self.nx + 1)
        return res

    def max_shift(self):
        return float(np.hypot(self.dx, self.dy).max())


class Layout:
    """A solved panorama: everything needed to render it, or any single photo of it, at its size."""
    def __init__(self, shots, nodes, mode, cumulative, rotations, f, gains, minX, minY, scale, W, H, flip, inliers):
        self.shots, self.nodes, self.mode = shots, nodes, mode
        self.cumulative, self.rotations, self.f, self.gains = cumulative, rotations, f, gains
        self.scale, self.W, self.H, self.flip, self.inliers = scale, W, H, flip, inliers
        self.cx, self.cy = -minX * scale, -minY * scale    # output position of the projection origin
        self.outF = (f or 0) * scale
        self.unmap = np.array([[1 / scale, 0, -self.cx / scale], [0, 1 / scale, -self.cy / scale], [0, 0, 1]])
        self.fields = {}                                  # node -> WarpField (parallax warp), if fitted
        self.seams = None                                 # seam masks from find_seams, if computed
        self.H0 = H                                       # height before any horizon correction
        self.hz = None                                    # horizon correction: {'d': per-column shift, 'pad'}
        self.bbox = {n: self._node_bbox(n) for n in nodes}
        self.bit16 = any(shots[n].img.dtype == np.uint16 for n in nodes)

    # ── horizon correction: output pixel (x, y) shows what was at (x, y − pad + d(x))
    def _pre(self, xs, ys):
        """Output coordinates -> the coordinates the panorama was solved in (undoes the horizon
        correction). Everything below render level (seams, warp, projection) uses the latter."""
        if self.hz is None: return xs, ys
        d = self.hz['d']
        return xs, ys - self.hz['pad'] + np.interp(xs, np.arange(len(d), dtype=np.float64), d)

    def _post_y(self, x, y):
        """The inverse of _pre for single points (solved coordinates -> output)."""
        if self.hz is None: return y
        d = self.hz['d']
        return y + self.hz['pad'] - float(np.interp(x, np.arange(len(d)), d))

    def _box(self, n):
        """Photo n's render box in output coordinates (the solved box, moved by the horizon shift)."""
        x0, y0, x1, y1 = self.bbox[n]
        if self.hz is None or x1 <= x0: return (x0, y0, x1, y1)
        seg = self.hz['d'][x0:x1]
        pad = self.hz['pad']
        return (x0, max(0, int(math.floor(y0 + pad - seg.max())) - 1), x1, min(self.H, int(math.ceil(y1 + pad - seg.min())) + 1))

    def leveled(self, xs_pts, ys_pts, smooth_frac=0.003):
        """A copy of this layout with the horizon corrected: the line through (xs_pts, ys_pts) — in
        this layout's output pixels, not flipped — is made straight (each column moved up or down by
        the line's offset from its median height; the canvas grows to keep everything). Re-rendering
        the copy re-blends from the photos. Works on an already corrected layout too."""
        import copy
        xs_pts, ys_pts = np.asarray(xs_pts, np.float64), np.asarray(ys_pts, np.float64)
        _, ys_solved = self._pre(xs_pts, ys_pts)                 # into the solved frame
        L = copy.copy(self)
        L.hz, L.H = None, self.H0
        d = horizon_shift(xs_pts, ys_solved, self.W, smooth_frac)
        pad, below = int(math.ceil(max(0.0, d.max()))), int(math.ceil(max(0.0, -d.min())))
        L.hz = {'d': d, 'pad': pad}
        L.H = self.H0 + pad + below
        return L

    # ── levelling by rotation (spheres, tilted cameras)
    def rays_at(self, xs, ys):
        """World view directions (unit, x right, y down, z forward) of output pixels in the solved
        frame (before any horizon column shift, not flipped)."""
        X, Y = np.asarray(xs, np.float64) + 0.5, np.asarray(ys, np.float64) + 0.5
        if self.mode == 'rectilinear':
            if not self.f: raise RuntimeError('this stitch has no camera model — use "Straighten the line" instead')
            P = np.stack([X, Y, np.ones_like(X)], -1) @ self.unmap.T
            px, py = P[..., 0] / P[..., 2], P[..., 1] / P[..., 2]
            r = self.shots[self.ref]
            v = np.stack([(px - r.w / 2) / self.f, (py - r.h / 2) / self.f, np.ones_like(px)], -1)
        else:
            dx, dy = (X - self.cx) / self.outF, (Y - self.cy) / self.outF
            if self.mode == 'cylindrical':
                v = np.stack([np.sin(dx), dy, np.cos(dx)], -1)
            else:
                v = np.stack([np.cos(dy) * np.sin(dx), np.sin(dy), np.cos(dy) * np.cos(dx)], -1)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def level_rotation(self, xs, ys):
        """The rotation that makes the traced horizon (output pixels, not flipped) level: fits the
        circle on the view sphere the horizon lies on and turns its axis upright — the smallest
        rotation that does it, so the view direction is kept. Returns (R_fix, info)."""
        xs, ys = self._pre(np.asarray(xs, np.float64), np.asarray(ys, np.float64))
        return level_rotation_from_rays(self.rays_at(xs, ys), self.outF or self.f or 1.0)

    def rotated(self, R_fix, job=None):
        """A new layout with the whole scene turned by R_fix (world rays r -> R_fix r), re-framed, with
        the parallax warp kept and the seams found again (they live in output pixels, which moved)."""
        return self._turned({n: R_fix for n in self.nodes}, job)

    def _turned(self, Rw, job=None):
        """A new layout with each photo n turned by the world-frame rotation Rw[n] (its rays r -> Rw[n] r),
        re-framed, the parallax warp kept and the seams found again."""
        rot = {n: Rw.get(n, np.eye(3)) @ R for n, R in (self.rotations or {}).items()}
        cum = self.cumulative
        if self.mode == 'rectilinear':
            r = self.shots[self.ref]
            K = np.array([[self.f, 0, r.w / 2], [0, self.f, r.h / 2], [0, 0, 1.0]])
            Ki = np.linalg.inv(K)
            cum = {n: (lambda M: M / M[2, 2])(K @ Rw.get(n, np.eye(3)) @ Ki @ C) for n, C in self.cumulative.items()}
        GRID = 8
        ps = []
        for n in self.nodes:
            s = self.shots[n]
            for gx in range(GRID + 1):
                for gy in range(GRID + 1):
                    x, y = gx / GRID * s.w, gy / GRID * s.h
                    if self.mode == 'rectilinear':
                        v = cum[n] @ np.array([x, y, 1.0])
                        if not v[2] > 1e-9:
                            raise RuntimeError('levelling tilts part of this flat (rectilinear) panorama behind the camera — '
                                               'stitch it cylindrical or equirectangular, or use "Straighten the line"')
                        ps.append((v[0] / v[2], v[1] / v[2]))
                    else:
                        p = forward_project(self.mode, rot[n] @ np.array([(x - s.w / 2) / self.f, (y - s.h / 2) / self.f, 1.0]), self.f)
                        if p is not None: ps.append(p)
        ps = np.array(ps)
        minX, maxX, minY, maxY = ps[:, 0].min(), ps[:, 0].max(), ps[:, 1].min(), ps[:, 1].max()
        sc = self.scale
        W, H = max(8, round(math.ceil(maxX - minX) * sc)), max(8, round(math.ceil(maxY - minY) * sc))
        if W * H > 4 * self.W * self.H:
            raise RuntimeError(f'levelling would make the canvas {W}×{H} — the traced line probably isn\'t the horizon')
        L = Layout(self.shots, self.nodes, self.mode, cum, rot or None, self.f, self.gains, minX, minY, sc, W, H, self.flip, self.inliers)
        L.edges, L.ref = getattr(self, 'edges', []), self.ref
        L.fields = dict(self.fields)                       # source-space: unaffected by turning the world
        if self.fields:                                    # widen boxes for the warp, as fit_warp does
            for n, fld in L.fields.items():
                m = int(math.ceil(fld.max_shift() * L.scale)) + 2
                x0, y0, x1, y1 = L.bbox[n]
                L.bbox[n] = (max(0, x0 - m), max(0, y0 - m), min(L.W, x1 + m), min(L.H, y1 + m))
        L.multiband = getattr(self, 'multiband', False)
        if self.seams:
            L.find_seams(job, blend_px=self.seams['blend'], method=self.seams.get('method', 'graphcut'))
        return L

    # ── horizon points per photo (fix steps; Hugin export)
    def horizon_samples(self, xs, ys, per_photo=60):
        """Traced horizon points (output pixels, not flipped) -> [(photo n, sx, sy, x)] — sx, sy in that
        photo's full-resolution upright pixels, x the output column: each point is assigned to the photo
        the panorama shows there (seam map if there is one, else the photo whose centre is nearest), at
        most per_photo evenly spread points per photo (None: all)."""
        x_out = np.asarray(xs, np.float64)
        xs, ys = self._pre(np.asarray(xs, np.float64), np.asarray(ys, np.float64))
        best = np.full(len(xs), -1); score = np.full(len(xs), -1.0)
        coords = {}
        for n in self.nodes:
            sx, sy, ok = self._source_coords(n, xs, ys)
            s = self.shots[n]
            if self.seams is not None and getattr(self, 'seam_labels', None) is not None:
                k = self.seams['k']
                Hs, Ws = self.seam_labels.shape
                lx = np.clip(((xs + 0.5) * k).astype(int), 0, Ws - 1); ly = np.clip(((ys + 0.5) * k).astype(int), 0, Hs - 1)
                sc = np.where(ok & (self.seam_labels[ly, lx] == n), 2.0, 0.0)
                sc = np.where(ok & (sc == 0), 1 - np.hypot(sx / s.w - 0.5, sy / s.h - 0.5), sc)   # fallback inside the photo
            else:
                sc = np.where(ok, 1 - np.hypot(sx / s.w - 0.5, sy / s.h - 0.5), -1.0)
            better = sc > score
            best[better], score[better] = n, sc[better]
            coords[n] = (sx, sy)
        out = []
        for n in self.nodes:
            idx = np.nonzero(best == n)[0]
            if not len(idx): continue
            if per_photo and len(idx) > per_photo: idx = idx[np.linspace(0, len(idx) - 1, per_photo).round().astype(int)]
            out += [(n, float(coords[n][0][i]) - 0.5, float(coords[n][1][i]) - 0.5, float(x_out[i])) for i in idx]   # pixel centres
        return out

    def horizon_steps(self, xs, ys):
        """The traced horizon cut into per-photo stretches, tolerant of hand tracing near a step:
        points within a guard band (1% of the width) either side of each photo boundary are ignored
        (lines that overshoot or stop short of the step), each side's height is measured from a window
        just beyond the band, and points that don't follow their side (a longer overshoot) are dropped.
        Returns (samples [(n, sx, sy, x)], bounds [(x_boundary, side-A indices, side-B indices)], kept
        indices) — samples sorted by x, indices into them."""
        smp = sorted((t for t in self.horizon_samples(xs, ys, per_photo=None) if not self.rotations or t[0] in self.rotations),
                     key=lambda t: t[3])
        if len(smp) < 6: return smp, [], np.arange(len(smp))
        own = np.array([t[0] for t in smp]); x = np.array([t[3] for t in smp])
        y_src = np.array([t[2] for t in smp])        # height in its own photo: comparable within one side
        guard, win = 0.01 * self.W, 0.04 * self.W
        # runs of one photo along x; a run narrower than the guard band is tracing noise near a seam
        runs = []
        for i in range(len(smp)):
            if runs and runs[-1][0] == own[i]: runs[-1][2] = i + 1
            else: runs.append([own[i], i, i + 1])
        changed = True
        while changed and len(runs) > 1:
            changed = False
            for k, (n, i0, i1) in enumerate(runs):
                if x[i1 - 1] - x[i0] < guard and len(runs) > 1:
                    if k > 0: runs[k - 1][2] = i1
                    else: runs[1][1] = i0
                    del runs[k]
                    merged = [runs[0]]
                    for r in runs[1:]:
                        if r[0] == merged[-1][0]: merged[-1][2] = r[2]
                        else: merged.append(r)
                    runs, changed = merged, True
                    break
        keep = np.ones(len(smp), bool)
        bounds = []
        for (na, a0, a1), (nb, b0, b1) in zip(runs[:-1], runs[1:]):
            xb = (x[a1 - 1] + x[b0]) / 2
            keep &= ~((np.abs(x - xb) < guard))
            A = np.array([i for i in range(a0, a1) if own[i] == na and xb - guard - win <= x[i] <= xb - guard])
            B = np.array([i for i in range(b0, b1) if own[i] == nb and xb + guard <= x[i] <= xb + guard + win])
            sides = []
            for S in (A, B):
                if len(S) >= 3:                              # drop points off their side's line
                    c = np.polyfit(x[S], y_src[S], 1)
                    r = np.abs(np.polyval(c, x[S]) - y_src[S])
                    mad = np.median(r) * 1.4826 + 1e-9
                    S = S[r <= max(3 * mad, 3.0)]
                sides.append(S)
            if len(sides[0]) >= 3 and len(sides[1]) >= 3:
                bounds.append((xb, sides[0], sides[1]))
        return smp, bounds, np.nonzero(keep)[0]

    def level_photos(self, xs, ys, job=None):
        """Fixes steps in the horizon. Wherever the traced line passes from one photo to the next, the
        horizon just before and just after the boundary must be at the same height (continuity — so
        hills and slopes within a photo are left alone); each photo but the reference is turned by the
        small rotation that achieves it, while the matches between photos still have to fit (robustly:
        near-ground parallax can't drag it). Returns (new layout — re-framed, warp kept, seams found
        again — and info: the steps before and after, in output pixels)."""
        from scipy.optimize import least_squares
        if not self.rotations or not self.f:
            raise RuntimeError('this stitch has no camera model — "Fix steps" needs one (try "Straighten the line")')
        samples, bounds, _ = self.horizon_steps(xs, ys)
        if len(samples) < 6: raise RuntimeError('trace the horizon across the panorama first')
        if not bounds:
            raise RuntimeError('the traced horizon doesn\'t cross from one photo into another with enough line on '
                               'both sides — trace a few percent of the width either side of the step')
        f = self.f
        owner = np.array([t[0] for t in samples])
        xpos = np.array([t[3] for t in samples])
        rays = []
        for n, sx, sy, _ in samples:
            s = self.shots[n]
            d = np.array([(sx - s.w / 2) / f, (sy - s.h / 2) / f, 1.0])
            rays.append(d / np.linalg.norm(d))
        rays = np.array(rays)
        free = [n for n in self.rotations if n != self.ref]
        idx = {n: k for k, n in enumerate(free)}
        edges = [e for e in getattr(self, 'edges', []) if e['i'] in self.rotations and e['j'] in self.rotations and len(e['pts'])]
        fs = 0.002 * max(max(self.shots[n].w, self.shots[n].h) for n in self.rotations)

        def Rw(p, n):
            return np.eye(3) if n not in idx else rodrigues(p[idx[n] * 3: idx[n] * 3 + 3])

        def elev(p, sel):
            out = np.empty(len(sel))
            for n in np.unique(owner[sel]):
                m = owner[sel] == n
                v = rays[sel][m] @ (Rw(p, n) @ self.rotations[n]).T
                out[m] = np.arcsin(np.clip(v[:, 1], -1, 1))
            return out

        def at_boundary(p, S, xb):
            """the side's horizon height, extended in a straight line to the boundary"""
            e = elev(p, S)
            c = np.polyfit(xpos[S] - xb, e, 1)
            return c[1]

        def steps(p):
            return np.array([at_boundary(p, a, xb) - at_boundary(p, b, xb) for xb, a, b in bounds])
        # A traced step is what to fix, not an outlier: steps are plain (quadratic) terms, each weighted
        # against the matches of the two photos it joins (w² = 25 × their count), so it wins over a few
        # weak or misleading matches but can't drag a well-matched pair by more than a pixel or so.
        # The matches get their robustness by reweighting instead (parallax / mismatches count less).
        def pair_count(a, b):
            return sum(len(e['pts']) for e in edges if {e['i'], e['j']} & {a, b})
        w_b = np.array([5.0 * math.sqrt(max(1, pair_count(owner[a[0]], owner[b[0]]))) for _, a, b in bounds])
        w_m = [np.ones(2 * len(e['pts'])) for e in edges]

        def resid(p, match_only=False):
            out = []
            for e in edges:
                Ri, Rj = Rw(p, e['i']) @ self.rotations[e['i']], Rw(p, e['j']) @ self.rotations[e['j']]
                si, sj = self.shots[e['i']], self.shots[e['j']]
                Q = e['pts']
                r0 = np.c_[(Q[:, 0] - si.w / 2) / f, (Q[:, 1] - si.h / 2) / f, np.ones(len(Q))]
                jv = r0 @ (Rj.T @ Ri).T
                z = np.where(jv[:, 2] > 1e-6, jv[:, 2], 1.0)
                out.append(np.c_[f * jv[:, 0] / z + sj.w / 2 - Q[:, 2], f * jv[:, 1] / z + sj.h / 2 - Q[:, 3]].ravel())
            if match_only: return out
            out = [r * np.sqrt(w) for r, w in zip(out, w_m)]
            out.append(w_b * f * steps(p))
            # a gentle preference for the smallest turn: one constraint per boundary can be met by many
            # rotations, and without this the solver may wander to a large one (yaw/roll it can't see)
            out.append(f * p)
            return np.concatenate(out)

        p = np.zeros(3 * len(free))
        before = np.abs(steps(p)) * f * self.scale
        if job: job.progress(0.05, 'fixing steps: solving each photo\'s tilt')
        for _ in range(3):                                   # reweight the matches around the current fit
            w_m = [1.0 / np.sqrt(1.0 + (r / fs) ** 2) for r in resid(p, match_only=True)]
            p = least_squares(resid, p, method='trf', max_nfev=200).x
        sol = type('S', (), {'x': p})
        after = np.abs(steps(sol.x)) * f * self.scale
        turns = {self.shots[n].name: math.degrees(np.linalg.norm(sol.x[idx[n] * 3: idx[n] * 3 + 3])) for n in free}
        worst = max(turns.items(), key=lambda kv: kv[1]) if turns else ('', 0.0)
        if worst[1] > 2.0:
            # real stitching steps are small misplacements; a big turn means the traced line itself jumps
            # (e.g. ridge top on one side of a photo boundary, far horizon on the other)
            xs_b = ', '.join(f'{xb / self.W * 100:.0f}%' for xb, _, _ in bounds)
            raise RuntimeError(f'closing the step would turn {worst[0]} by {worst[1]:.1f}° — far more than a stitching error. '
                               f'The traced line probably follows different features either side of a photo boundary '
                               f'(at {xs_b} across the panorama): check it follows the same horizon there.')
        info = {'steps_before_px': before.round(1).tolist(), 'steps_after_px': after.round(1).tolist(),
                'turns_deg': turns, 'boundaries': len(bounds)}
        if job: job.progress(0.1, 'fixing steps: re-framing' + (' and finding seams again' if self.seams else ''))
        return self._turned({n: Rw(sol.x, n) for n in free}, job), info

    # ── parallax warp
    def transfer(self, a, b, pa):
        """Source positions in photo a -> where the global model puts the same scene point in photo b
        (source-pixel, texel-centre convention as _source_coords). Returns (pb, ok)."""
        sa, sb = self.shots[a], self.shots[b]
        if self.mode == 'rectilinear':
            M = np.linalg.inv(self.cumulative[b]) @ self.cumulative[a]
            v = np.c_[pa, np.ones(len(pa))] @ M.T
            ok = np.abs(v[:, 2]) > 1e-9
            v[~ok, 2] = 1
            return v[:, :2] / v[:, 2:3], ok
        ray = np.c_[(pa[:, 0] - sa.w / 2) / self.f, (pa[:, 1] - sa.h / 2) / self.f, np.ones(len(pa))]
        c = ray @ (self.rotations[b].T @ self.rotations[a]).T
        ok = c[:, 2] > 1e-6
        c[~ok, 2] = 1
        return np.c_[c[:, 0] / c[:, 2] * self.f + sb.w / 2, c[:, 1] / c[:, 2] * self.f + sb.h / 2], ok

    def warp_order(self):
        """Reference first, then by link distance from it: in each overlap the photo later in this
        order is the one that bends (asymmetric), its partner stays as already placed."""
        adj = {n: set() for n in self.nodes}
        for e in getattr(self, 'edges', []):
            if e['i'] in adj and e['j'] in adj: adj[e['i']].add(e['j']); adj[e['j']].add(e['i'])
        depth, queue = {self.ref: 0}, [self.ref]
        while queue:
            c = queue.pop(0)
            for t in sorted(adj[c]):
                if t not in depth: depth[t] = depth[c] + 1; queue.append(t)
        return sorted(self.nodes, key=lambda n: (depth.get(n, 99), n))

    def warp_data(self, b, order, exclude=None):
        """(positions in b, displacements) that would bring b's matches onto its already-placed
        partners. exclude: optional {(edge id): boolean mask} of points to leave out (held-out tests)."""
        rank = {n: k for k, n in enumerate(order)}
        P, Dsp = [], []
        for e in getattr(self, 'edges', []):
            if 'warp' not in e or b not in (e['i'], e['j']): continue
            a = e['j'] if e['i'] == b else e['i']
            if a not in rank or rank[a] >= rank[b]: continue
            w0, w1 = e['warp']
            qa, qb = (w1, w0) if e['i'] == b else (w0, w1)
            keep = np.ones(len(qa), bool) if exclude is None else ~exclude.get(id(e), np.zeros(len(qa), bool))
            qa, qb = qa[keep] + 0.5, qb[keep] + 0.5               # match coords -> texel-centre coords
            sa = self.fields[a].invert(qa) if a in self.fields else qa   # where a is actually drawn from
            tb, ok = self.transfer(a, b, sa)
            sb = self.shots[b]
            ok &= (tb[:, 0] >= 0) & (tb[:, 1] >= 0) & (tb[:, 0] < sb.w) & (tb[:, 1] < sb.h)
            P.append(tb[ok]); Dsp.append(qb[ok] - tb[ok])
        if not P: return np.zeros((0, 2)), np.zeros((0, 2))
        return np.concatenate(P), np.concatenate(Dsp)

    def fit_warp(self, stiffness=1.0, job=None, exclude=None):
        """Fits a WarpField for every photo but the reference, in warp_order. Returns per-photo stats."""
        self.fields = {}
        order = self.warp_order()
        stats = {}
        for b in order[1:]:
            pts, disp = self.warp_data(b, order, exclude)
            if len(pts) < 20: continue
            s = self.shots[b]
            fld = WarpField(s.w, s.h, max(s.w, s.h) / 48)
            res = fld.fit(pts, disp, stiffness, robust_px=max(3.0, 0.0005 * max(s.w, s.h)))
            self.fields[b] = fld
            mag = np.hypot(disp[:, 0], disp[:, 1])
            stats[s.name] = {'points': len(pts), 'before_median': float(np.median(mag)), 'before_p90': float(np.percentile(mag, 90)),
                             'after_median': float(np.median(res)), 'after_p90': float(np.percentile(res, 90)), 'max_shift': fld.max_shift()}
            if job: job.log('info', f'parallax warp {s.name}: {len(pts)} matches, misalignment median {stats[s.name]["before_median"]:.1f}px '
                                    f'(90% {stats[s.name]["before_p90"]:.1f}) -> {stats[s.name]["after_median"]:.1f}px '
                                    f'(90% {stats[s.name]["after_p90"]:.1f}); largest shift {fld.max_shift():.0f}px')
        # content can move by up to the largest shift: widen each photo's render box to match
        for n, fld in self.fields.items():
            m = int(math.ceil(fld.max_shift() * self.scale)) + 2
            x0, y0, x1, y1 = self.bbox[n]
            self.bbox[n] = (max(0, x0 - m), max(0, y0 - m), min(self.W, x1 + m), min(self.H, y1 + m))
        return stats

    @property
    def names(self):
        return [self.shots[n].name for n in self.nodes]

    def _node_bbox(self, n):
        s, G = self.shots[n], 16
        xs, ys = [], []
        for gx in range(G + 1):
            for gy in range(G + 1):
                x, y = gx / G * s.w, gy / G * s.h
                if self.mode == 'rectilinear':
                    px, py = apply_h(self.cumulative[n], x, y)
                    ox, oy = px * self.scale + self.cx, py * self.scale + self.cy
                else:
                    ray = self.rotations[n] @ np.array([(x - s.w / 2) / self.f, (y - s.h / 2) / self.f, 1.0])
                    p = forward_project(self.mode, ray, self.outF)
                    if p is None: continue
                    ox, oy = p[0] + self.cx, p[1] + self.cy
                xs.append(ox); ys.append(oy)
        if not xs: return (0, 0, self.W, self.H)
        x0, x1 = max(0, int(math.floor(min(xs))) - 2), min(self.W, int(math.ceil(max(xs))) + 2)
        y0, y1 = max(0, int(math.floor(min(ys))) - 2), min(self.H, int(math.ceil(max(ys))) + 2)
        if self.mode != 'rectilinear':
            # a curved projection can wrap past the bbox the sample grid suggests; widen generously
            x0, x1 = max(0, x0 - self.W // 20), min(self.W, x1 + self.W // 20)
            y0, y1 = max(0, y0 - self.H // 20), min(self.H, y1 + self.H // 20)
        return (x0, y0, x1, y1)

    def photo_spans(self):
        """Where each photo sits in the panorama, for labelling: [(name, left, centre_x, right, centre_y)]
        in output pixels (left/right along the photo's middle row), sorted left to right."""
        out = []
        for n in self.nodes:
            s = self.shots[n]
            pts = []
            for fx in (0.0, 0.5, 1.0):
                x, y = fx * s.w, s.h / 2
                if self.mode == 'rectilinear':
                    px, py = apply_h(self.cumulative[n], x, y)
                    pts.append((px * self.scale + self.cx, py * self.scale + self.cy))
                else:
                    ray = self.rotations[n] @ np.array([(x - s.w / 2) / self.f, 0.0, 1.0])
                    p = forward_project(self.mode, ray, self.outF)
                    pts.append(None if p is None else (p[0] + self.cx, p[1] + self.cy))
            if pts[1] is None: continue
            (c, cy), l, r = pts[1], pts[0] and pts[0][0], pts[2] and pts[2][0]
            if l is None or r is None or not l < c < r:   # wrapped round a 360° seam: centre only
                l = r = c
            cy = self._post_y(c, cy)                     # horizon correction, if any
            if self.flip: cy = self.H - cy
            clampx = lambda v: min(float(self.W), max(0.0, v))
            out.append((s.name, clampx(l), clampx(c), clampx(r), min(float(self.H), max(0.0, cy))))
        return sorted(out, key=lambda t: t[2])

    # ── seam cutting
    def _seam_layer(self, n, k):
        """Photo n on the seam canvas (output scaled by k), cropped to its box: colour (with gain),
        a detail (high-pass) image, sharpness, a border ramp, and coverage."""
        s = self.shots[n]
        bx0, by0, bx1, by1 = self.bbox[n]
        x0, y0 = int(math.floor(bx0 * k)), int(math.floor(by0 * k))
        x1, y1 = min(self.seam_shape[1], int(math.ceil(bx1 * k))), min(self.seam_shape[0], int(math.ceil(by1 * k)))
        if x1 <= x0 or y1 <= y0: return None
        jj, ii = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        sx, sy, ok = self._source_coords(n, (jj + 0.5) / k - 0.5, (ii + 0.5) / k - 0.5)
        if not ok.any(): return None
        rgb, _, vs = s.view(max(64, int(max(s.w, s.h) * k * 1.5)))
        mx, my = (sx / vs - 0.5).astype(np.float32), (sy / vs - 0.5).astype(np.float32)
        col = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255
        col *= self.gains.get(n, 1.0)
        g = col.mean(-1)
        detail = g - cv2.GaussianBlur(g, (0, 0), 2)
        sm, sf = s.sharpness_map()
        sharp = cv2.remap(sm, (sx * sf - 0.5).astype(np.float32), (sy * sf - 0.5).astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
        u, v = sx / s.w, sy / s.h
        edge_raw = np.minimum(np.minimum(u, 1 - u), np.minimum(v, 1 - v)).astype(np.float32)
        edge = np.clip(edge_raw / FEATHER, 0, 1)
        return {'box': (x0, y0, x1, y1), 'col': col, 'detail': detail, 'sharp': np.where(ok, sharp, 0),
                'edge': edge, 'edge_raw': edge_raw, 'ok': ok}

    def find_seams(self, job=None, seam_px=2000, blend_px=2.0, prefer_sharp=True, method='graphcut'):
        """Chooses which photo each part of the panorama comes from, on a canvas ~seam_px across.
        method 'graphcut' (seam cutting): graph cut, alpha-expansion — seams run where the overlapping
        photos agree; each area prefers the sharper photo and stays away from photo borders.
        method 'nearest' (no seam cutting): each spot goes to the photo whose centre is nearest (the
        classic layout multi-band blending is usually done on). The labels become per-photo masks
        softened by blend_px seam pixels, which render() then uses in place of the wide feather."""
        import maxflow
        k = min(1.0, seam_px / max(self.W, self.H))
        Hs, Ws = max(2, round(self.H * k)), max(2, round(self.W * k))
        self.seam_shape = (Hs, Ws)
        layers = {}
        for t, n in enumerate(self.nodes):
            if job: job.check(); job.progress(0.1 + 0.3 * t / len(self.nodes), f'seams: preparing {self.shots[n].name}')
            lay = self._seam_layer(n, k)
            if lay is not None: layers[n] = lay
        if len(layers) < 2: return
        if method == 'nearest':
            labels = np.full((Hs, Ws), -1, np.int32)
            best = np.full((Hs, Ws), -1.0, np.float32)
            for n, lay in layers.items():
                x0, y0, x1, y1 = lay['box']
                # distance in from the photo's border, as a fraction (not clipped: deeper in = more central)
                e = np.where(lay['ok'], lay['edge_raw'], -1.0)
                sub = best[y0:y1, x0:x1]
                better = e > sub
                sub[better] = e[better]
                labels[y0:y1, x0:x1][better] = n
            self._set_seams(labels, k, layers, blend_px, 'nearest', job)
            return
        INF = 1e6
        # SEAM0: cost of seam length itself — keeps near-identical photos from breaking into islands
        A_SHARP, A_EDGE, SEAM0 = (0.25 if prefer_sharp else 0.0), 0.05, 0.03

        def canvas(fill, dtype=np.float32, ch=None):
            return np.full((Hs, Ws) if ch is None else (Hs, Ws, ch), fill, dtype)
        smax = canvas(0)
        for lay in layers.values():
            x0, y0, x1, y1 = lay['box']
            np.maximum(smax[y0:y1, x0:x1], lay['sharp'], out=smax[y0:y1, x0:x1])

        def data_cost(n, x0, y0, x1, y1):
            """Cost of giving photo n the pixels of box (x0..x1, y0..y1); INF where it doesn't cover."""
            out = np.full((y1 - y0, x1 - x0), INF, np.float32)
            lay = layers[n]
            lx0, ly0, lx1, ly1 = lay['box']
            ix0, iy0, ix1, iy1 = max(x0, lx0), max(y0, ly0), min(x1, lx1), min(y1, ly1)
            if ix0 >= ix1 or iy0 >= iy1: return out
            sl = (slice(iy0 - ly0, iy1 - ly0), slice(ix0 - lx0, ix1 - lx0))
            ok = lay['ok'][sl]
            rel = lay['sharp'][sl] / np.maximum(smax[iy0:iy1, ix0:ix1], 1e-6)
            c = A_SHARP * np.clip(1 - rel - 0.1, 0, None) + A_EDGE * (1 - lay['edge'][sl])
            out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = np.where(ok, c, INF)
            return out

        def stack(names, x0, y0, x1, y1):
            """Colour+detail of the given photos over a box, and their coverage."""
            h, w = y1 - y0, x1 - x0
            C = np.zeros((len(names), h, w, 4), np.float32)
            K = np.zeros((len(names), h, w), bool)
            for t, n in enumerate(names):
                lay = layers[n]
                lx0, ly0, lx1, ly1 = lay['box']
                ix0, iy0, ix1, iy1 = max(x0, lx0), max(y0, ly0), min(x1, lx1), min(y1, ly1)
                if ix0 >= ix1 or iy0 >= iy1: continue
                sl = (slice(iy0 - ly0, iy1 - ly0), slice(ix0 - lx0, ix1 - lx0))
                dst = (slice(iy0 - y0, iy1 - y0), slice(ix0 - x0, ix1 - x0))
                C[t][dst + (slice(0, 3),)] = lay['col'][sl]
                C[t][dst + (3,)] = lay['detail'][sl] * 2
                K[t][dst] = lay['ok'][sl]
            return C, K

        # initial labels: cheapest photo per pixel
        labels = np.full((Hs, Ws), -1, np.int32)
        best = canvas(INF)
        for n in layers:
            c = data_cost(n, 0, 0, Ws, Hs)
            better = c < best
            labels[better], best[better] = n, c[better]

        def diff(Ca, Ka, Cb, Kb):
            """Per-pixel disagreement between two photos (1 where either doesn't cover)."""
            d = np.abs(Ca - Cb).mean(-1) * 1.5
            return np.where(Ka & Kb, np.minimum(d, 1.0), 1.0)

        order = list(layers)
        for sweep in range(2):
            changed = 0
            for a in order:
                if job: job.check(); job.progress(0.4 + 0.25 * (sweep + order.index(a) / len(order)), f'seams: graph cut, pass {sweep + 1}')
                lx0, ly0, lx1, ly1 = layers[a]['box']
                x0, y0, x1, y1 = max(0, lx0 - 1), max(0, ly0 - 1), min(Ws, lx1 + 1), min(Hs, ly1 + 1)
                L = labels[y0:y1, x0:x1]
                present = [n for n in layers if n != a and (L == n).any()]
                names = [a] + present
                C, K = stack(names, x0, y0, x1, y1)
                idx = np.full(L.shape, -1, np.int32)
                for t, n in enumerate(names): idx[L == n] = t
                yy, xx = np.indices(L.shape)
                lab_ok = idx >= 0
                safe = np.where(lab_ok, idx, 0)
                Cl, Kl = C[safe, yy, xx], K[safe, yy, xx] & lab_ok          # current photo's colour at each pixel
                Ca, Ka = C[0], K[0]                                         # alpha's colour
                d_cur_a = diff(Cl, Kl, Ca, Ka)                              # d(l_p, a) at p
                e1 = data_cost(a, x0, y0, x1, y1)
                e0 = np.zeros(L.shape, np.float32)
                for t, n in enumerate(names):
                    if t == 0: continue
                    m = idx == t
                    if m.any(): e0[m] = data_cost(n, x0, y0, x1, y1)[m]
                e0[~lab_ok] = 0                                             # uncovered stays uncovered
                e1 = np.where(lab_ok, e1, INF)
                e1[idx == 0] = e0[idx == 0] = np.minimum(e0, e1)[idx == 0]  # already a: indifferent
                unary1, unary0 = e1.astype(np.float64), e0.astype(np.float64)
                rights, downs = [], []
                for axis, store in ((1, rights), (0, downs)):
                    sp_ = (slice(None), slice(None, -1)) if axis == 1 else (slice(None, -1), slice(None))
                    sq_ = (slice(None), slice(1, None)) if axis == 1 else (slice(1, None), slice(None))
                    lp, lq = idx[sp_], idx[sq_]
                    both = (lp >= 0) & (lq >= 0)
                    # colour of q's label at p and of p's label at q
                    Cq_at_p = C[np.where(lq >= 0, lq, 0), yy[sp_], xx[sp_]]; Kq_at_p = K[np.where(lq >= 0, lq, 0), yy[sp_], xx[sp_]]
                    Cp_at_q = C[np.where(lp >= 0, lp, 0), yy[sq_], xx[sq_]]; Kp_at_q = K[np.where(lp >= 0, lp, 0), yy[sq_], xx[sq_]]
                    Cl_p, Kl_p, Cl_q, Kl_q = Cl[sp_], Kl[sp_], Cl[sq_], Kl[sq_]
                    V = lambda d_p, d_q, same: np.where(same, 0.0, d_p + d_q + SEAM0)
                    E00 = V(diff(Cl_p, Kl_p, Cq_at_p, Kq_at_p), diff(Cp_at_q, Kp_at_q, Cl_q, Kl_q), lp == lq)
                    E01 = V(d_cur_a[sp_], diff(Cp_at_q, Kp_at_q, Ca[sq_], Ka[sq_]), lp == 0)   # p keeps l_p, q -> a
                    E10 = V(diff(Ca[sp_], Ka[sp_], Cq_at_p, Kq_at_p), d_cur_a[sq_], lq == 0)  # p -> a, q keeps l_q
                    E00, E01, E10 = [np.where(both, E, 0.0) for E in (E00, E01, E10)]
                    # E = E00 + (E10-E00) xp + (0-E10) xq + (E01+E10-E00)(1-xp) xq
                    cp, cq = E10 - E00, -E10
                    for c, sl in ((cp, sp_), (cq, sq_)):
                        pos = np.clip(c, 0, None); neg = np.clip(-c, 0, None)
                        unary1[sl] += pos; unary0[sl] += neg
                    store.append(np.clip(E01 + E10 - E00, 0, None))
                g = maxflow.Graph[float]()
                nodes = g.add_grid_nodes(L.shape)
                wr = np.zeros(L.shape); wr[:, :-1] = rights[0]
                wd = np.zeros(L.shape); wd[:-1, :] = downs[0]
                g.add_grid_edges(nodes, weights=wr, structure=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]), symmetric=False)
                g.add_grid_edges(nodes, weights=wd, structure=np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]), symmetric=False)
                # x=1 (switch to a) <=> sink segment: its cost goes on the source edge
                g.add_grid_tedges(nodes, unary1, unary0)
                g.maxflow()
                to_a = g.get_grid_segments(nodes) & Ka & lab_ok & (idx != 0)
                if to_a.any():
                    L[to_a] = a
                    changed += int(to_a.sum())
            if changed == 0: break
        self._set_seams(labels, k, layers, blend_px, 'graphcut', job)

    def _set_seams(self, labels, k, layers, blend_px, method, job=None):
        self.seam_labels = labels
        self.seams = {'k': k, 'boxes': {n: lay['box'] for n, lay in layers.items()}, 'blend': blend_px, 'method': method}
        self.seams['masks'] = self._seam_masks(blend_px)
        if job:
            Hs, Ws = labels.shape
            share = {self.shots[n].name: f'{100 * (labels == n).sum() / max(1, (labels >= 0).sum()):.0f}%' for n in layers}
            how = 'seam map' if method == 'graphcut' else 'nearest-centre seams (no seam cutting)'
            job.log('info', f'seams: {Ws}×{Hs} {how}, area per photo {share}')

    def _seam_masks(self, sigma):
        """Each photo's seam region softened by a Gaussian of sigma seam-map pixels, cropped to its box
        plus the blur's reach: {n: ((x0, y0), mask, k)} where mask pixel = output pixel * k - (x0, y0).
        Wide blurs run on a downsampled label map (same result, far faster)."""
        labels, k = self.seam_labels, self.seams['k']
        Hs, Ws = labels.shape
        d = max(1, int(sigma // 6))                   # keep the blur ~6 px wide on whatever grid it runs
        pad = int(math.ceil(sigma * 3)) + d + 1
        masks = {}
        for n, (x0, y0, x1, y1) in self.seams['boxes'].items():
            x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(Ws, x1 + pad), min(Hs, y1 + pad)
            m = (labels[y0:y1, x0:x1] == n).astype(np.float32)
            if d > 1:
                m = cv2.resize(m, (max(1, round(m.shape[1] / d)), max(1, round(m.shape[0] / d))), interpolation=cv2.INTER_AREA)
            if sigma > 0: m = cv2.GaussianBlur(m, (0, 0), sigma / d)
            sx_, sy_ = m.shape[1] / (x1 - x0), m.shape[0] / (y1 - y0)
            masks[n] = ((x0 * sx_, y0 * sy_), m, (k * sx_, k * sy_))
        return masks

    def _seam_weight(self, n, xs, ys, masks=None):
        """Soft seam mask of photo n at output pixels (masks: from _seam_masks; default the seams')."""
        (mx0, my0), m, (kx, ky) = (masks or self.seams['masks'])[n]
        mx = ((xs + 0.5) * kx - 0.5 - mx0).astype(np.float32)
        my = ((ys + 0.5) * ky - 0.5 - my0).astype(np.float32)
        return cv2.remap(m, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _source_coords(self, n, xs, ys):
        """Output pixel centres -> (sx, sy) source-pixel coordinates (texel-centre convention: pixel
        i spans [i, i+1)) and a validity mask — the same maths as the web app's FS_WARP shader."""
        s = self.shots[n]
        X, Y = xs + 0.5, ys + 0.5
        if self.mode == 'rectilinear':
            Hs = np.linalg.inv(self.cumulative[n]) @ self.unmap
            u = Hs[0, 0] * X + Hs[0, 1] * Y + Hs[0, 2]
            v = Hs[1, 0] * X + Hs[1, 1] * Y + Hs[1, 2]
            w = Hs[2, 0] * X + Hs[2, 1] * Y + Hs[2, 2]
            ok = np.abs(w) >= 1e-6
            w = np.where(ok, w, 1.0)
            sx, sy = u / w, v / w
        else:
            dx, dy = (X - self.cx) / self.outF, (Y - self.cy) / self.outF
            if self.mode == 'cylindrical':
                rx, ry, rz = np.sin(dx), dy, np.cos(dx)
            else:
                rx, ry, rz = np.cos(dy) * np.sin(dx), np.sin(dy), np.cos(dy) * np.cos(dx)
            Rt = self.rotations[n].T
            cx_ = Rt[0, 0] * rx + Rt[0, 1] * ry + Rt[0, 2] * rz
            cy_ = Rt[1, 0] * rx + Rt[1, 1] * ry + Rt[1, 2] * rz
            cz_ = Rt[2, 0] * rx + Rt[2, 1] * ry + Rt[2, 2] * rz
            ok = cz_ > 0.0005
            cz_ = np.where(ok, cz_, 1.0)
            sx, sy = cx_ / cz_ * self.f + s.w / 2, cy_ / cz_ * self.f + s.h / 2
        fld = self.fields.get(n)
        if fld is not None:                                # local parallax correction on top
            dx, dy = fld.sample(sx, sy)
            sx, sy = sx + dx, sy + dy
        ok &= (sx >= 0) & (sy >= 0) & (sx < s.w) & (sy < s.h)
        return sx, sy, ok

    def _weights(self, n, sx, sy, ok):
        s = self.shots[n]
        u, v = sx / s.w, sy / s.h
        w = np.minimum(np.minimum(u, 1 - u) / FEATHER, np.minimum(v, 1 - v) / FEATHER)
        w = np.clip(w, 0, 1)
        return np.where(ok, np.maximum(w, MIN_WEIGHT), 0).astype(np.float32)

    def _sources(self, k=None):
        """Per-photo source images at roughly output resolution (downsampled once if the output is
        smaller, so a reduced-size render doesn't alias), with the factor to scale coordinates by."""
        k = min(1.0, self.scale) if k is None else k
        out = {}
        for n in self.nodes:
            img = self.shots[n].img
            if k < 0.999:
                img = cv2.resize(img, (max(2, round(img.shape[1] * k)), max(2, round(img.shape[0] * k))), interpolation=cv2.INTER_AREA)
            out[n] = (img, img.shape[1] / self.shots[n].w, img.shape[0] / self.shots[n].h)
        return out

    def _sample(self, src, sx, sy, kx, ky):
        mapx = (sx * kx - 0.5).astype(np.float32)
        mapy = (sy * ky - 0.5).astype(np.float32)
        maxv = 65535.0 if src.dtype == np.uint16 else 255.0
        return cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / maxv

    def _strips(self, rows_target=4_000_000):
        rows = max(16, min(self.H, rows_target // max(1, self.W)))
        return [(y0, min(self.H, y0 + rows)) for y0 in range(0, self.H, rows)]

    def _render_grid(self, job, q=1.0, masks='seams', frac0=0.5, frac1=1.0, label=None, as_float=False):
        """Renders the panorama on a grid q times the output size: uint16 when q=1 (the output itself),
        float32 0..1 when q<1 or as_float (multi-band takes differences — never of unsigned ints).
        masks: 'seams' (the seam masks, if any), a dict from _seam_masks, or None (the feathered
        blend). Not flipped. Returns (rgb, alpha)."""
        if isinstance(masks, str): masks = self.seams['masks'] if self.seams else None
        Wq, Hq = (self.W, self.H) if q == 1 else (max(1, math.ceil(self.W * q)), max(1, math.ceil(self.H * q)))
        full = q == 1 and not as_float
        out = np.zeros((Hq, Wq, 3), np.uint16 if full else np.float32)
        alpha = np.zeros((Hq, Wq), np.uint8)
        srcs = self._sources(min(1.0, self.scale * q))
        rows = max(16, min(Hq, 4_000_000 // max(1, Wq)))
        strips = [(y0, min(Hq, y0 + rows)) for y0 in range(0, Hq, rows)]
        boxes = {n: (int(math.floor(b[0] * q)), int(math.floor(b[1] * q)), min(Wq, int(math.ceil(b[2] * q))), min(Hq, int(math.ceil(b[3] * q))))
                 for n, b in ((n, self._box(n)) for n in self.nodes)}
        done = [0]
        lock = threading.Lock()

        def do_strip(y0, y1):
            job.check()
            h = y1 - y0
            acc = np.zeros((h, Wq, 3), np.float32)
            ws = np.zeros((h, Wq), np.float32)
            for n in self.nodes:
                bx0, by0, bx1, by1 = boxes[n]
                r0, r1 = max(y0, by0), min(y1, by1)
                if r0 >= r1 or bx0 >= bx1: continue
                gy, gx = np.mgrid[r0:r1, bx0:bx1].astype(np.float64)
                xs, ys = (gx, gy) if q == 1 else ((gx + 0.5) / q - 0.5, (gy + 0.5) / q - 0.5)
                xs, ys = self._pre(xs, ys)
                sx, sy, ok = self._source_coords(n, xs, ys)
                if not ok.any(): continue
                if masks is not None and n in masks:
                    s_ = self.shots[n]
                    u, v = sx / s_.w, sy / s_.h
                    ramp = np.clip(np.minimum(np.minimum(u, 1 - u), np.minimum(v, 1 - v)) / (FEATHER / 6), MIN_WEIGHT, 1)
                    w = np.where(ok, self._seam_weight(n, xs, ys, masks) * ramp, 0).astype(np.float32)
                    # never leave a covered pixel empty: at photo edges the coarse seam map can miss
                    # coverage; this floor sits above the 0.0005 "covered" threshold below
                    w = np.where(ok, np.maximum(w, 0.001), 0).astype(np.float32)
                else:
                    w = self._weights(n, sx, sy, ok)
                img, kx, ky = srcs[n]
                c = self._sample(img, sx, sy, kx, ky) * (self.gains.get(n, 1.0) * w[..., None])
                acc[r0 - y0:r1 - y0, bx0:bx1] += c
                ws[r0 - y0:r1 - y0, bx0:bx1] += w
            cov = ws > 0.0005
            res = np.where(cov[..., None], acc / np.where(cov, ws, 1.0)[..., None], 0.0)
            out[y0:y1] = (np.clip(res, 0, 1) * 65535 + 0.5).astype(np.uint16) if full else res
            alpha[y0:y1] = cov.astype(np.uint8) * 255
            with lock:
                done[0] += 1
                job.progress(frac0 + (frac1 - frac0) * done[0] / len(strips),
                             label or f'rendering {Wq}×{Hq}: strip {done[0]}/{len(strips)}')

        with ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4)))) as ex:
            for fut in [ex.submit(do_strip, a, b) for a, b in strips]:
                fut.result()
        return out, alpha

    def render(self, job, frac0=0.5, frac1=1.0):
        """Returns (rgb uint16 HxWx3, alpha uint8 HxW) — alpha 0 where no photo covers."""
        mb = getattr(self, 'multiband', False) if self.seams else False
        if mb == 'full':
            out, alpha = self._render_pyramid(job, frac0, frac1)
        else:
            out, alpha = self._render_grid(job, 1.0, 'seams', frac0, frac0 + (frac1 - frac0) * (0.75 if mb else 1.0))
            if mb:
                self._multiband(out, alpha, job, frac0 + (frac1 - frac0) * 0.75, frac1)
        if self.flip:
            out, alpha = out[::-1], alpha[::-1]
        return np.ascontiguousarray(out), np.ascontiguousarray(alpha)

    # ── full (textbook) multi-band: Laplacian pyramids in RAM
    def _warp_crop(self, n, x0, y0, x1, y1, masks):
        """Photo n warped onto output region [x0,x1)×[y0,y1) at full size: colour (float32, with gain),
        coverage, and its seam weight (with the short border ramp)."""
        s = self.shots[n]
        h, w = y1 - y0, x1 - x0
        I = np.zeros((h, w, 3), np.float32)
        ok_all = np.zeros((h, w), bool)
        wt = np.zeros((h, w), np.float32)
        srcs = self._sources()
        img, kx, ky = srcs[n]
        rows = max(16, 2_000_000 // max(1, w))

        def strip(r0, r1):
            gy, gx = np.mgrid[y0 + r0:y0 + r1, x0:x1].astype(np.float64)
            xs, ys = self._pre(gx, gy)
            sx, sy, ok = self._source_coords(n, xs, ys)
            if not ok.any(): return
            u, v = sx / s.w, sy / s.h
            ramp = np.clip(np.minimum(np.minimum(u, 1 - u), np.minimum(v, 1 - v)) / (FEATHER / 6), MIN_WEIGHT, 1)
            I[r0:r1] = self._sample(img, sx, sy, kx, ky) * self.gains.get(n, 1.0)
            ok_all[r0:r1] = ok
            wt[r0:r1] = np.where(ok, np.maximum(self._seam_weight(n, xs, ys, masks) * ramp, 0.001), 0)
        with ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4)))) as ex:
            for fut in [ex.submit(strip, r, min(h, r + rows)) for r in range(0, h, rows)]:
                fut.result()
        return I, ok_all, wt

    @staticmethod
    def _fill_outside(I, ok):
        """Fills pixels outside a photo's coverage smoothly from its edge (push-pull), so the pyramid
        doesn't see a step to black there — its coarse bands would otherwise darken the blend."""
        okf = ok.astype(np.float32)
        G, Wt = [I * okf[..., None]], [okf]
        while min(G[-1].shape[:2]) > 8:
            G.append(cv2.pyrDown(G[-1])); Wt.append(cv2.pyrDown(Wt[-1]))
        F = G[-1] / np.maximum(Wt[-1], 1e-6)[..., None]
        if not (Wt[-1] > 1e-6).all():
            mean = (G[0].reshape(-1, 3).sum(0) / max(1.0, okf.sum())).astype(np.float32)
            F = np.where((Wt[-1] > 1e-6)[..., None], F, mean)
        for l in range(len(G) - 2, -1, -1):
            up = cv2.pyrUp(F, dstsize=(G[l].shape[1], G[l].shape[0]))
            wl = np.clip(Wt[l], 0, 1)[..., None]
            F = G[l] + (1 - wl) * up if l else np.where(ok[..., None], I, up)
        return F

    def _render_pyramid(self, job, frac0, frac1):
        """Burt–Adelson multi-band blending, the textbook way: each photo (at full size, gaps filled
        smoothly) is split into a Laplacian pyramid — octave bands from fine texture down to overall
        brightness — and each band is blended with that photo's seam mask blurred to the band's scale
        (its Gaussian pyramid); the blended pyramid is then collapsed. Holds the whole panorama's
        pyramid in RAM (~22 bytes per output pixel) plus one photo's at a time."""
        W, H = self.W, self.H
        L = int(np.clip(math.floor(math.log2(max(W, H) / 32)), 1, 12))
        S = 2 ** L
        Wp, Hp = -(-W // S) * S, -(-H // S) * S
        acc = [np.zeros((Hp >> l, Wp >> l, 3), np.float32) for l in range(L + 1)]
        wsum = [np.zeros((Hp >> l, Wp >> l), np.float32) for l in range(L + 1)]
        alpha = np.zeros((H, W), np.uint8)
        masks = self.seams['masks']
        nodes = [n for n in self.nodes if n in masks]
        for t, n in enumerate(nodes):
            job.check()
            job.progress(frac0 + (frac1 - frac0) * 0.85 * t / len(nodes), f'multi-band (full pyramid): {self.shots[n].name}')
            bx0, by0, bx1, by1 = self._box(n)
            if bx1 <= bx0 or by1 <= by0: continue
            x0, y0 = max(0, (bx0 // S - 2) * S), max(0, (by0 // S - 2) * S)
            x1, y1 = min(Wp, (-(-bx1 // S) + 2) * S), min(Hp, (-(-by1 // S) + 2) * S)
            I, ok, wt = self._warp_crop(n, x0, y0, x1, y1, masks)
            if not ok.any(): continue
            ya, xa = min(y1, H), min(x1, W)
            alpha[y0:ya, x0:xa] |= ok[:ya - y0, :xa - x0].astype(np.uint8) * 255
            G = self._fill_outside(I, ok)
            del I
            M = wt
            for l in range(L + 1):
                job.check()
                if l < L:
                    Gn = cv2.pyrDown(G)
                    band = G - cv2.pyrUp(Gn, dstsize=(G.shape[1], G.shape[0]))
                else:
                    Gn, band = None, G
                sl = (slice(y0 >> l, (y0 >> l) + band.shape[0]), slice(x0 >> l, (x0 >> l) + band.shape[1]))
                acc[l][sl] += band * M[..., None]
                wsum[l][sl] += M
                if l < L:
                    G, M = Gn, cv2.pyrDown(M)
            del G, M
        job.progress(frac0 + (frac1 - frac0) * 0.9, 'multi-band (full pyramid): collapsing')
        img = acc[L] / np.maximum(wsum[L], 1e-6)[..., None]
        for l in range(L - 1, -1, -1):
            img = cv2.pyrUp(img, dstsize=(Wp >> l, Hp >> l)) + acc[l] / np.maximum(wsum[l], 1e-6)[..., None]
            acc[l] = wsum[l] = None
        img = img[:H, :W]
        out = np.where((alpha > 0)[..., None], (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16), 0).astype(np.uint16)
        return out, alpha

    def _multiband_level(self, job, q, lp, masks_fine, masks_coarse, ramp_fine, ramp_coarse):
        """One band's correction on a grid q × output: Σ_n (M_n^coarse − M_n^fine) · lowpass(I_n), with
        M the normalised seam weights and lowpass a Gaussian of lp grid pixels applied to each photo
        on its own (normalised by its coverage, so its border doesn't darken it). ramp_*: how far in
        from its border (fraction of the photo) a photo's weight fades in — wider for coarser bands,
        so brightness also fades gradually where a photo simply ends inside another."""
        Wq, Hq = max(1, math.ceil(self.W * q)), max(1, math.ceil(self.H * q))
        srcs = self._sources(min(1.0, self.scale * q))
        per = []
        Wf, Wc = np.zeros((Hq, Wq), np.float32), np.zeros((Hq, Wq), np.float32)
        for n in self.nodes:
            job.check()
            if n not in masks_fine: continue
            b = self._box(n)
            x0, y0 = int(math.floor(b[0] * q)), int(math.floor(b[1] * q))
            x1, y1 = min(Wq, int(math.ceil(b[2] * q))), min(Hq, int(math.ceil(b[3] * q)))
            if x1 <= x0 or y1 <= y0: continue
            gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
            xs, ys = self._pre((gx + 0.5) / q - 0.5, (gy + 0.5) / q - 0.5)
            sx, sy, ok = self._source_coords(n, xs, ys)
            if not ok.any(): continue
            s_ = self.shots[n]
            u, v = sx / s_.w, sy / s_.h
            edge = np.minimum(np.minimum(u, 1 - u), np.minimum(v, 1 - v))
            rf, rc = (np.clip(edge / r, MIN_WEIGHT, 1) for r in (ramp_fine, ramp_coarse))
            wf = np.where(ok, np.maximum(self._seam_weight(n, xs, ys, masks_fine) * rf, 0.001), 0).astype(np.float32)
            wc = np.where(ok, np.maximum(self._seam_weight(n, xs, ys, masks_coarse) * rc, 0.001), 0).astype(np.float32)
            img, kx, ky = srcs[n]
            I = self._sample(img, sx, sy, kx, ky) * self.gains.get(n, 1.0)
            okf = ok.astype(np.float32)
            num = cv2.GaussianBlur(I * okf[..., None], (0, 0), lp)
            den = cv2.GaussianBlur(okf, (0, 0), lp)
            low = num / np.maximum(den, 1e-4)[..., None]
            per.append(((x0, y0, x1, y1), wf, wc, low))
            Wf[y0:y1, x0:x1] += wf
            Wc[y0:y1, x0:x1] += wc
        corr = np.zeros((Hq, Wq, 3), np.float32)
        for (x0, y0, x1, y1), wf, wc, low in per:
            dm = wc / np.maximum(Wc[y0:y1, x0:x1], 1e-6) - wf / np.maximum(Wf[y0:y1, x0:x1], 1e-6)
            corr[y0:y1, x0:x1] += dm[..., None] * low
        return corr

    def _multiband(self, out, alpha, job, frac0, frac1):
        """Multi-band blending on top of a sharp-seam render (in place): each photo is split into
        frequency bands and each band is blended over a width matching its scale — fine texture
        switches photo within a few pixels (no ghosting), brightness and colour fade over up to a
        quarter of the panorama (no visible seam). Since the sharp-seam render already is the finest
        band's blend, band k adds Σ_n (M_n^k − M_n^{k−1}) · lowpass_k(I_n) (_multiband_level): both
        factors are smooth, so each band is computed on a grid matched to its scale."""
        k = self.seams['k']
        s0 = self.seams['blend']
        Hs, Ws = self.seam_labels.shape
        levels = []
        sig = s0
        while sig < max(Hs, Ws) / 4 and len(levels) < 9:
            levels.append((sig, sig * 2)); sig *= 2
        corrections = []
        cov_full = alpha > 0
        for t, (sp, sc) in enumerate(levels):
            job.check()
            lp_out = sp / k                               # band boundary in output pixels
            q = min(1.0, 2.0 / lp_out)                    # the lowpass spans 2 grid pixels: smooth enough
            job.progress(frac0 + (frac1 - frac0) * 0.9 * t / len(levels), f'multi-band blending: band {t + 1}/{len(levels)}')
            # border ramp doubles per band like the masks, from the sharp render's FEATHER/6 up to half the photo
            r_f, r_c = min(0.5, FEATHER / 6 * 2 ** t), min(0.5, FEATHER / 6 * 2 ** (t + 1))
            corrections.append((q, self._multiband_level(job, q, lp_out * q, self._seam_masks(sp), self._seam_masks(sc), r_f, r_c)))
        # add the corrections to the full-size render, strip by strip
        rows = max(16, min(self.H, 4_000_000 // max(1, self.W)))
        xs = np.arange(self.W, dtype=np.float32)

        def apply(y0, y1):
            job.check()
            res = out[y0:y1].astype(np.float32) / 65535
            ys = np.arange(y0, y1, dtype=np.float32)
            for q, corr in corrections:
                mx = np.broadcast_to(((xs + 0.5) * q - 0.5)[None, :], (y1 - y0, self.W)).astype(np.float32)
                my = np.broadcast_to(((ys + 0.5) * q - 0.5)[:, None], (y1 - y0, self.W)).astype(np.float32)
                res += cv2.remap(corr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            c = cov_full[y0:y1, :, None]
            out[y0:y1] = np.where(c, (np.clip(res, 0, 1) * 65535 + 0.5).astype(np.uint16), out[y0:y1])
        strips = [(y0, min(self.H, y0 + rows)) for y0 in range(0, self.H, rows)]
        with ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4)))) as ex:
            for fut in [ex.submit(apply, a, b) for a, b in strips]:
                fut.result()
        job.progress(frac1, 'multi-band blending done')

    def seam_mask_layer(self, n, job=None):
        """Where the automatic seams give photo n the picture: an 8-bit mask the panorama's size (soft at
        the seams, 0 outside the photo) — a starting point for compositing the layers by hand."""
        mask = np.zeros((self.H, self.W), np.uint8)
        if not self.seams or n not in self.seams['masks']: return mask
        bx0, by0, bx1, by1 = self._box(n)
        for y0, y1 in self._strips():
            if job: job.check()
            r0, r1 = max(y0, by0), min(y1, by1)
            if r0 >= r1 or bx0 >= bx1: continue
            ys, xs = np.mgrid[r0:r1, bx0:bx1].astype(np.float64)
            xs, ys = self._pre(xs, ys)
            _, _, ok = self._source_coords(n, xs, ys)
            m = self._seam_weight(n, xs, ys)
            mask[r0:r1, bx0:bx1] = np.where(ok, np.clip(m * 255 + 0.5, 0, 255), 0).astype(np.uint8)
        return np.ascontiguousarray(mask[::-1]) if self.flip else mask

    def render_layer(self, n, job=None):
        """One photo alone, the panorama's full size, opaque inside its footprint (with this
        panorama's exposure gain) and transparent elsewhere — stacks exactly on render()."""
        out = np.zeros((self.H, self.W, 3), np.uint16)
        alpha = np.zeros((self.H, self.W), np.uint8)
        s = self.shots[n]
        k = min(1.0, self.scale)
        img = s.img if k >= 0.999 else cv2.resize(s.img, (max(2, round(s.w * k)), max(2, round(s.h * k))), interpolation=cv2.INTER_AREA)
        kx, ky = img.shape[1] / s.w, img.shape[0] / s.h
        bx0, by0, bx1, by1 = self._box(n)
        for y0, y1 in self._strips():
            if job: job.check()
            r0, r1 = max(y0, by0), min(y1, by1)
            if r0 >= r1 or bx0 >= bx1: continue
            ys, xs = np.mgrid[r0:r1, bx0:bx1].astype(np.float64)
            xs, ys = self._pre(xs, ys)
            sx, sy, ok = self._source_coords(n, xs, ys)
            if not ok.any(): continue
            c = np.clip(self._sample(img, sx, sy, kx, ky) * self.gains.get(n, 1.0), 0, 1)
            out[r0:r1, bx0:bx1] = np.where(ok[..., None], (c * 65535 + 0.5).astype(np.uint16), 0)
            alpha[r0:r1, bx0:bx1] = ok.astype(np.uint8) * 255
        if self.flip:
            out, alpha = out[::-1], alpha[::-1]
        return np.ascontiguousarray(out), np.ascontiguousarray(alpha)


# ───────────────────────────────────────────── projects: a solved layout as plain data
def layout_state(L):
    """Everything needed to rebuild layout L without matching again: (meta — JSON-able, arrays —
    name -> ndarray). The photos themselves are referenced by path."""
    meta = {'shot_paths': [s.path for s in L.shots], 'nodes': [int(n) for n in L.nodes], 'mode': L.mode,
            'f': None if L.f is None else float(L.f), 'gains': {str(n): float(g) for n, g in L.gains.items()},
            'minX': float(-L.cx / L.scale), 'minY': float(-L.cy / L.scale), 'scale': float(L.scale),
            'W': int(L.W), 'H0': int(L.H0), 'H': int(L.H), 'flip': bool(L.flip), 'inliers': int(L.inliers),
            'ref': int(getattr(L, 'ref', L.nodes[0])), 'multiband': getattr(L, 'multiband', False),
            'cum': [int(n) for n in (L.cumulative or {})], 'rot': [int(n) for n in (L.rotations or {})],
            'edges': [], 'fields': {}, 'seams': None, 'hz_pad': None}
    a = {}
    for n, M in (L.cumulative or {}).items(): a[f'cum_{n}'] = np.asarray(M, np.float64)
    for n, R in (L.rotations or {}).items(): a[f'rot_{n}'] = np.asarray(R, np.float64)
    for k, e in enumerate(getattr(L, 'edges', [])):
        meta['edges'].append({'i': int(e['i']), 'j': int(e['j']), 'inliers': int(e['inliers']), 'total': int(e['total']),
                              'ratio': None if e.get('ratio') is None else float(e['ratio']), 'method': e.get('method', '')})
        a[f'e{k}_H'], a[f'e{k}_pts'] = e['H'], e['pts']
        a[f'e{k}_cp0'], a[f'e{k}_cp1'], a[f'e{k}_cpsc'] = e['cp']
        if 'warp' in e: a[f'e{k}_w0'], a[f'e{k}_w1'] = e['warp']
    for n, fld in L.fields.items():
        meta['fields'][str(n)] = {'w': fld.w, 'h': fld.h, 'cell': fld.cell}
        a[f'fld_{n}_dx'], a[f'fld_{n}_dy'] = fld.dx, fld.dy
    if L.seams:
        meta['seams'] = {'k': float(L.seams['k']), 'blend': float(L.seams['blend']), 'method': L.seams.get('method', 'graphcut'),
                         'boxes': {str(n): [int(v) for v in b] for n, b in L.seams['boxes'].items()}}
        a['seam_labels'] = L.seam_labels
    if L.hz is not None:
        meta['hz_pad'] = int(L.hz['pad'])
        a['hz_d'] = L.hz['d']
    return meta, a


def layout_from_state(meta, a, shots):
    """Rebuilds a layout saved by layout_state; shots: the Shot list, in meta['shot_paths'] order."""
    cum = {n: a[f'cum_{n}'] for n in meta['cum']}
    rot = {n: a[f'rot_{n}'] for n in meta['rot']} or None
    gains = {int(n): g for n, g in meta['gains'].items()}
    L = Layout(shots, list(meta['nodes']), meta['mode'], cum, rot, meta['f'], gains, meta['minX'], meta['minY'],
               meta['scale'], meta['W'], meta['H0'], meta['flip'], meta['inliers'])
    L.ref, L.multiband = meta['ref'], meta['multiband']
    L.edges = []
    for k, em in enumerate(meta['edges']):
        e = dict(em)
        e['H'], e['pts'] = a[f'e{k}_H'], a[f'e{k}_pts']
        e['cp'] = (a[f'e{k}_cp0'], a[f'e{k}_cp1'], a[f'e{k}_cpsc'])
        if f'e{k}_w0' in a: e['warp'] = (a[f'e{k}_w0'], a[f'e{k}_w1'])
        L.edges.append(e)
    for ns, fm in meta['fields'].items():
        fld = WarpField(fm['w'], fm['h'], fm['cell'])
        fld.dx, fld.dy = a[f'fld_{ns}_dx'], a[f'fld_{ns}_dy']
        L.fields[int(ns)] = fld
    for n, fld in L.fields.items():                      # widen render boxes for the warp, as fit_warp does
        m = int(math.ceil(fld.max_shift() * L.scale)) + 2
        x0, y0, x1, y1 = L.bbox[n]
        L.bbox[n] = (max(0, x0 - m), max(0, y0 - m), min(L.W, x1 + m), min(L.H, y1 + m))
    if meta['seams']:
        L.seam_labels = a['seam_labels']
        L.seam_shape = L.seam_labels.shape
        sm = meta['seams']
        L.seams = {'k': sm['k'], 'blend': sm['blend'], 'method': sm.get('method', 'graphcut'),
                   'boxes': {int(n): tuple(b) for n, b in sm['boxes'].items()}}
        L.seams['masks'] = L._seam_masks(sm['blend'])
    if meta['hz_pad'] is not None:
        L.hz = {'d': a['hz_d'], 'pad': meta['hz_pad']}
        L.H = meta['H']
    return L


# ───────────────────────────────────────────── horizon correction helpers (GUI-free)
def horizon_shift(xs_pts, ys_pts, W, smooth_frac=0.003):
    """Per-column vertical shift d(x) (length W) that makes the line through the points straight:
    d = line height − its median height, interpolated across columns (held constant beyond the
    line's ends) and lightly smoothed against hand jitter."""
    order = np.argsort(xs_pts)
    xs, ys = np.asarray(xs_pts, np.float64)[order], np.asarray(ys_pts, np.float64)[order]
    keep = np.r_[True, np.diff(xs) > 1e-9]
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 2: raise ValueError('draw a line across the horizon first')
    line = np.interp(np.arange(W, dtype=np.float64), xs, ys)
    sig = max(1.0, smooth_frac * W)
    line = cv2.GaussianBlur(line.reshape(1, -1).astype(np.float64), (0, 0), sigmaX=sig, borderType=cv2.BORDER_REPLICATE).ravel()
    inside = (np.arange(W) >= xs[0]) & (np.arange(W) <= xs[-1])
    return (line - np.median(line[inside])).astype(np.float64)


def level_image(rgb, alpha, xs_pts, ys_pts, smooth_frac=0.003):
    """Horizon correction for an image without a layout (e.g. opened from a file): the same column
    shifts as Layout.leveled, applied to the pixels (bilinear). Returns (rgb, alpha)."""
    H, W = rgb.shape[:2]
    d = horizon_shift(xs_pts, ys_pts, W, smooth_frac)
    pad, below = int(math.ceil(max(0.0, d.max()))), int(math.ceil(max(0.0, -d.min())))
    Hn = H + pad + below
    out = np.zeros((Hn, W, 3), rgb.dtype)
    a_out = np.zeros((Hn, W), np.uint8)
    a_in = alpha if alpha is not None else np.full((H, W), 255, np.uint8)
    mx = np.tile(np.arange(W, dtype=np.float32), (1, 1))
    rows = max(16, 4_000_000 // max(1, W))
    for y0 in range(0, Hn, rows):
        y1 = min(Hn, y0 + rows)
        my = (np.arange(y0, y1, dtype=np.float64)[:, None] - pad + d[None, :]).astype(np.float32)
        mxs = np.broadcast_to(mx, my.shape).astype(np.float32)
        out[y0:y1] = cv2.remap(rgb, mxs, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a_out[y0:y1] = cv2.remap(a_in, mxs, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, a_out


def level_rotation_from_rays(r, f_px=1.0):
    """Rotation making the horizon through rays r (Nx3, unit) level. Traces spanning less than ~120°
    of yaw fit a great circle (plane through the eye — a short arc can't pin down more); wider ones
    fit any circle, so a horizon raised a little by distant hills all round still counts as level.
    Returns (R_fix, info) — info: tilt corrected (°), spread of the line about the fit (° and px)."""
    yaw = np.unwrap(np.arctan2(r[:, 0], r[:, 2])[np.argsort(np.arctan2(r[:, 0], r[:, 2]))])
    span = math.degrees(yaw.max() - yaw.min()) if len(yaw) > 1 else 0.0
    if span < 120:
        n = np.linalg.svd(r, full_matrices=False)[2][-1]
    else:
        n = np.linalg.svd(r - r.mean(0), full_matrices=False)[2][-1]
    if n[1] < 0: n = -n                                    # point "down" (y down), like the world axis
    ey = np.array([0.0, 1.0, 0.0])
    axis = np.cross(n, ey)
    s, c = np.linalg.norm(axis), float(n @ ey)
    R = np.eye(3) if s < 1e-12 else rodrigues(axis / s * math.atan2(s, c))
    elev = np.arcsin(np.clip((r @ R.T)[:, 1], -1, 1))
    spread = float(np.std(elev))
    return R, {'tilt_deg': math.degrees(math.atan2(s, c)), 'spread_deg': math.degrees(spread),
               'spread_px': spread * f_px, 'span_deg': span, 'fit': 'great circle' if span < 120 else 'circle'}


def level_equirect_image(rgb, alpha, xs_pts, ys_pts):
    """Levels a full equirectangular sphere image (2:1, 360°×180°, e.g. from other software) by
    rotating the sphere so the traced horizon (pixels) is level. Returns (rgb, alpha, info)."""
    H, W = rgb.shape[:2]
    if abs(W / H - 2) > 0.02:
        raise ValueError(f'this image is {W}×{H}, not a 2:1 equirectangular sphere — use "Straighten the line"')
    def rays(x, y):
        lon, lat = ((x + 0.5) / W - 0.5) * 2 * math.pi, ((y + 0.5) / H - 0.5) * math.pi
        return np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
    R, info = level_rotation_from_rays(rays(np.asarray(xs_pts, np.float64), np.asarray(ys_pts, np.float64)), W / (2 * math.pi))
    out = np.zeros_like(rgb)
    a_out = np.zeros((H, W), np.uint8)
    a_in = alpha if alpha is not None else np.full((H, W), 255, np.uint8)
    rows = max(16, 4_000_000 // W)
    for y0 in range(0, H, rows):
        y1 = min(H, y0 + rows)
        yy, xx = np.mgrid[y0:y1, 0:W].astype(np.float64)
        src = rays(xx, yy) @ R                              # output ray -> source ray (R^T r)
        lon = np.arctan2(src[..., 0], src[..., 2]); lat = np.arcsin(np.clip(src[..., 1], -1, 1))
        mx = ((lon / (2 * math.pi) + 0.5) * W - 0.5).astype(np.float32)
        my = ((lat / math.pi + 0.5) * H - 0.5).astype(np.float32)
        out[y0:y1] = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        a_out[y0:y1] = cv2.remap(a_in, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    return out, a_out, info


def edge_strength(rgb_small):
    """0..1 map of horizontal edges (strong vertical brightness change), for the magnetic tool."""
    g = cv2.cvtColor(to_uint8(rgb_small), cv2.COLOR_RGB2GRAY).astype(np.float32)
    e = np.abs(cv2.Sobel(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.CV_32F, 0, 1, ksize=3))
    e = cv2.GaussianBlur(e, (0, 0), 1.0)
    return e / max(1e-6, float(np.percentile(e, 99.5)))


def magnetic_path(edge, xa, ya, xb, yb, reach=2, snap=6):
    """Path from (xa, ya) to near (xb, yb) along the strongest horizontal edge, one point per column
    (a horizon is a function of x), moving at most `reach` rows per column; ends on the best edge
    within `snap` rows of yb. Dynamic programming — instant at preview size. Returns (xs, ys)."""
    H, W = edge.shape
    xa, xb = int(np.clip(xa, 0, W - 1)), int(np.clip(xb, 0, W - 1))
    ya, yb = int(np.clip(ya, 0, H - 1)), int(np.clip(yb, 0, H - 1))
    if xa == xb: return np.array([xa]), np.array([ya])
    step = 1 if xb > xa else -1
    cols = np.arange(xa, xb + step, step)
    cost_col = 1.0 - np.clip(edge, 0, 1)
    INF = 1e9
    cost = np.full(H, INF); cost[ya] = 0.0
    back = np.zeros((len(cols), H), np.int8)
    for t in range(1, len(cols)):
        best = np.full(H, INF); arg = np.zeros(H, np.int8)
        for dy in range(-reach, reach + 1):
            sh = np.full(H, INF)
            if dy >= 0: sh[dy:] = cost[:H - dy] if dy else cost
            else: sh[:dy] = cost[-dy:]
            sh = sh + 0.05 * abs(dy)
            better = sh < best
            best[better], arg[better] = sh[better], dy
        cost = best + cost_col[:, cols[t]]
        back[t] = arg
    lo, hi = max(0, yb - snap), min(H, yb + snap + 1)
    y = lo + int(np.argmin(cost[lo:hi]))
    ys = np.empty(len(cols), np.int64)
    for t in range(len(cols) - 1, -1, -1):
        ys[t] = y
        y -= int(back[t, y])          # int(): an int8 scalar would turn y itself into int8 and overflow
    return cols, ys


# ───────────────────────────────────────────── layers for manual compositing
def harmonise_layer(rgb, alpha, pano_rgb, pano_alpha, sigma, shown=None):
    """A layer (one photo, full panorama size) with its brightness and colour brought to those of the
    finished, blended panorama. The difference between the two is measured only where this photo is
    the one the panorama shows (shown: its seam mask, 0-255) — there it holds just the blend's
    brightness/colour adjustments, never another photo's picture — then extended smoothly over the
    rest of the layer and added. Measuring it where OTHER photos are shown would carry their
    (slightly misaligned) shapes into the layer as faint ghosts. Without a seam mask the difference is
    taken everywhere but smoothed three times as much. The layer's own detail is untouched."""
    if shown is None:
        return _harmonise_everywhere(rgb, alpha, pano_rgb, pano_alpha, 3 * sigma)
    ys_, xs_ = np.nonzero(alpha)
    if not len(ys_): return rgb
    y0, y1, x0, x1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
    maxv = 65535.0 if rgb.dtype == np.uint16 else 255.0
    pmax = 65535.0 if pano_rgb.dtype == np.uint16 else 255.0
    inside = alpha[y0:y1, x0:x1] > 0
    meas = inside & (shown[y0:y1, x0:x1] >= 128)
    if pano_alpha is not None: meas &= pano_alpha[y0:y1, x0:x1] > 0
    if meas.sum() < 200:                                      # hardly shown anywhere: fall back
        return _harmonise_everywhere(rgb, alpha, pano_rgb, pano_alpha, 3 * sigma)
    I = rgb[y0:y1, x0:x1].astype(np.float32) / maxv
    R = pano_rgb[y0:y1, x0:x1].astype(np.float32) / pmax
    d = max(1, int(sigma / 4))
    h, w = y1 - y0, x1 - x0
    sz = (max(1, w // d), max(1, h // d))
    m = meas.astype(np.float32)
    Ds = cv2.resize((R - I) * m[..., None], sz, interpolation=cv2.INTER_AREA)
    Ws = cv2.resize(m, sz, interpolation=cv2.INTER_AREA)
    ok = Ws > 0.3
    Dn = np.where(ok[..., None], Ds / np.maximum(Ws, 1e-6)[..., None], 0).astype(np.float32)
    Df = Layout._fill_outside(Dn, ok)                         # carry it smoothly into the rest of the layer
    corr = cv2.GaussianBlur(Df, (0, 0), sigma / d)
    corr = cv2.resize(corr, (w, h), interpolation=cv2.INTER_LINEAR)
    out = rgb.copy()
    out[y0:y1, x0:x1] = np.where(inside[..., None], (np.clip(I + corr, 0, 1) * maxv + 0.5).astype(rgb.dtype), out[y0:y1, x0:x1])
    return out


def _harmonise_everywhere(rgb, alpha, pano_rgb, pano_alpha, sigma):
    """harmonise_layer without a seam mask: the smooth difference measured wherever both have picture."""
    ys_, xs_ = np.nonzero(alpha)
    if not len(ys_): return rgb
    y0, y1, x0, x1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
    maxv = 65535.0 if rgb.dtype == np.uint16 else 255.0
    pmax = 65535.0 if pano_rgb.dtype == np.uint16 else 255.0
    I = rgb[y0:y1, x0:x1].astype(np.float32) / maxv
    R = pano_rgb[y0:y1, x0:x1].astype(np.float32) / pmax
    both = ((alpha[y0:y1, x0:x1] > 0) & (pano_alpha[y0:y1, x0:x1] > 0) if pano_alpha is not None
            else alpha[y0:y1, x0:x1] > 0).astype(np.float32)
    d = max(1, int(sigma / 4))                                  # smooth: work on a reduced grid
    h, w = y1 - y0, x1 - x0
    sz = (max(1, w // d), max(1, h // d))
    D = cv2.resize((R - I) * both[..., None], sz, interpolation=cv2.INTER_AREA)
    Wt = cv2.resize(both, sz, interpolation=cv2.INTER_AREA)
    num = cv2.GaussianBlur(D, (0, 0), sigma / d)
    den = cv2.GaussianBlur(Wt, (0, 0), sigma / d)
    corr = num / np.maximum(den, 1e-4)[..., None]
    corr = cv2.resize(corr, (w, h), interpolation=cv2.INTER_LINEAR)
    out = rgb.copy()
    inside = alpha[y0:y1, x0:x1] > 0
    out[y0:y1, x0:x1] = np.where(inside[..., None], (np.clip(I + corr, 0, 1) * maxv + 0.5).astype(rgb.dtype), out[y0:y1, x0:x1])
    return out


# ───────────────────────────────────────────── add a photo to a finished panorama
def add_photo(pano_rgb, pano_alpha, photo_rgb, mode, job, photo_name='photo', matcher='sift', model_path=None,
              device='auto', blend='feather'):
    """Places photo into pano with one homography and blends it in, growing the canvas as needed.
    matcher: 'sift' or 'aliked' (ALIKED+LightGlue: found at several scales, then refined on a
    matched-scale crop and at full resolution). mode: 'top' (photo preferred), 'fill' (only where
    the panorama is empty) or 'sharp' (whichever is sharper; seam blend only). blend: 'feather'
    (photo faded in over its border) or 'multiband' (seam cut between photo and panorama, photo's
    brightness matched in the overlap, full-pyramid multi-band blend).
    Returns (rgb, alpha, info). Exact for rectilinear panoramas; a local approximation on curved ones."""
    ph, pw = photo_rgb.shape[:2]
    H0, W0 = pano_rgb.shape[:2]
    if matcher == 'aliked':
        Hf, inl, nmatch = _locate_aliked(pano_rgb, photo_rgb, model_path, device, job, photo_name)
    else:
        Hf, inl, nmatch = _locate_sift(pano_rgb, photo_rgb, job, photo_name)
    return _place_photo(pano_rgb, pano_alpha, photo_rgb, Hf, mode, job, photo_name, blend, inl, nmatch)


def _aliked_session(model_path, device, job):
    """The shared ALIKED+LightGlue session for this model and GPU (same cache as the stitcher)."""
    if not model_path or not os.path.isfile(model_path):
        raise RuntimeError('no ALIKED model file — choose the models folder on the Stitch tab, or use SIFT')
    device = resolve_device(device)
    key = (os.path.normcase(os.path.abspath(model_path)), device)
    m = _SESSIONS.get(key)
    if m is None:
        for k in [k for k in _SESSIONS if k[1] == device]: del _SESSIONS[k]
        m = _SESSIONS[key] = AlikedMatcher(model_path, device, job)
    return m


def _locate_aliked(pano_rgb, photo_rgb, model_path, device, job, photo_name):
    """Photo -> panorama homography with ALIKED+LightGlue. The photo's scale in the panorama is
    unknown, and LightGlue copes with modest scale changes only, so: (1) match the photo at several
    sizes against the panorama at 2048px and keep the best; (2) crop that part of the panorama at
    full resolution, scale it to the photo's scale, match again; (3) refine every match at full
    resolution. Returns (H photo->pano full-res, inliers, matches)."""
    ph, pw = photo_rgb.shape[:2]
    H0, W0 = pano_rgb.shape[:2]
    m = _aliked_session(model_path, device, job)
    pano8, photo8 = to_uint8(pano_rgb), to_uint8(photo_rgb)
    rs = lambda im, k: im if k >= 0.999 else cv2.resize(im, (max(2, round(im.shape[1] * k)), max(2, round(im.shape[0] * k))), interpolation=cv2.INTER_AREA)
    sp = min(1.0, ALIKED_MAX_RES / max(W0, H0))
    pano_s = rs(pano8, sp)
    best = None
    for t, side in enumerate((384, 640, 1024, 1536, 2048)):
        job.check()
        job.progress(0.05 + 0.25 * t / 5, f'finding {photo_name} in the panorama (ALIKED, photo at {side}px)')
        k = min(1.0, side / max(pw, ph))
        p0, p1, _ = m.match(rs(photo8, k), pano_s)
        res = best_homography(p0, p1)
        n = 0 if res is None else int(res[1].sum())
        if n >= 12 and n / max(1, len(p0)) >= 0.25 and (best is None or n > best[0]):
            best = (n, len(p0), np.diag([1 / sp, 1 / sp, 1.0]) @ res[0] @ np.diag([k, k, 1.0]))
        if k >= 1.0: break
    if best is None:
        raise RuntimeError(f"couldn't place {photo_name} in the panorama with ALIKED — not enough consistent matches "
                           '(does it overlap it? SIFT may cope with a large difference in scale)')
    inl, nmatch, H1 = best
    job.log('info', f'add photo: ALIKED found it ({inl} inliers at the best scale) — refining on a full-resolution crop')
    # (2) matched-scale crop of the panorama around where the photo lands
    corners = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float64)
    v = np.c_[corners, np.ones(4)] @ H1.T
    if not (v[:, 2] > 1e-9).all(): return H1, inl, nmatch
    c = v[:, :2] / v[:, 2:3]
    mx, my = 0.1 * (c[:, 0].max() - c[:, 0].min()), 0.1 * (c[:, 1].max() - c[:, 1].min())
    cx0, cy0 = int(max(0, c[:, 0].min() - mx)), int(max(0, c[:, 1].min() - my))
    cx1, cy1 = int(min(W0, c[:, 0].max() + mx)), int(min(H0, c[:, 1].max() + my))
    if cx1 - cx0 < 32 or cy1 - cy0 < 32: return H1, inl, nmatch
    d = H1[2] @ np.array([pw / 2, ph / 2, 1.0])
    J = (H1[:2, :2] - np.outer((H1 @ np.array([pw / 2, ph / 2, 1.0]))[:2] / d, H1[2, :2])) / d
    sc = math.sqrt(abs(np.linalg.det(J))) or 1.0                     # panorama px per photo px
    kp = min(1.0, ALIKED_MAX_RES / max(pw, ph))
    kc = min(2.0, kp / sc)
    big = max((cx1 - cx0) * kc, (cy1 - cy0) * kc)
    if big > ALIKED_MAX_RES:
        f = ALIKED_MAX_RES / big; kp *= f; kc *= f
    job.progress(0.35, f'matching {photo_name} against the matching part of the panorama')
    photo2 = rs(photo8, kp)
    crop = pano8[cy0:cy1, cx0:cx1]
    crop2 = crop if abs(kc - 1) < 1e-3 else cv2.resize(crop, (max(2, round(crop.shape[1] * kc)), max(2, round(crop.shape[0] * kc))),
                                                      interpolation=cv2.INTER_AREA if kc < 1 else cv2.INTER_CUBIC)
    p0, p1, _ = m.match(photo2, crop2)
    q0 = (p0.astype(np.float64) + 0.5) / kp - 0.5
    q1 = (p1.astype(np.float64) + 0.5) / kc - 0.5 + np.array([cx0, cy0])
    res = best_homography(q0, q1, tol_scale=max(1.0, 1 / kp))
    if res is None or res[1].sum() < max(12, 0.5 * inl):
        job.log('info', 'add photo: the refining match was weaker — keeping the first placement'); return H1, inl, nmatch
    H2, mk = res
    q0, q1 = q0[mk], q1[mk]
    # (3) sub-pixel at full resolution
    job.progress(0.45, 'refining the matches at full resolution')
    gp, gq = cv2.cvtColor(photo8, cv2.COLOR_RGB2GRAY), cv2.cvtColor(pano8, cv2.COLOR_RGB2GRAY)
    q1r, n_ok, moved = refine_matches(gp, gq, q0, q1, H2, max(1.0 / kp, 1.0 / max(kc, 1e-6)))
    H3, m3 = cv2.findHomography(q0.astype(np.float32), q1r.astype(np.float32), cv2.RANSAC, 3.0, maxIters=3000, confidence=0.999)
    if H3 is None or m3.sum() < 0.5 * len(q0):
        H3, m3 = H2, mk[mk]
    job.log('info', f'add photo: {len(p0)} ALIKED matches on the crop, {int(m3.sum())} inliers after {n_ok} were refined '
                    f'at full resolution (mean shift {moved:.2f}px)')
    return H3 / H3[2, 2], int(m3.sum()), len(p0)


def _locate_sift(pano_rgb, photo_rgb, job, photo_name):
    """Photo -> panorama homography with SIFT (scale-invariant, so one pass at reduced sizes)."""
    ph, pw = photo_rgb.shape[:2]
    H0, W0 = pano_rgb.shape[:2]
    sift = cv2.SIFT_create(contrastThreshold=SIFT_CONTRAST)
    sn = min(1.0, MATCH_RES / max(pw, ph))
    sp = min(1.0, 4096 / max(W0, H0))
    pm = cv2.resize(cv2.cvtColor(to_uint8(photo_rgb), cv2.COLOR_RGB2GRAY), (max(2, round(pw * sn)), max(2, round(ph * sn))), interpolation=cv2.INTER_AREA)
    qm = cv2.resize(cv2.cvtColor(to_uint8(pano_rgb), cv2.COLOR_RGB2GRAY), (max(2, round(W0 * sp)), max(2, round(H0 * sp))), interpolation=cv2.INTER_AREA)
    job.progress(0.05, f'finding features in {photo_name}')
    kp, dp = sift.detectAndCompute(pm, None)
    job.check()
    job.progress(0.2, 'finding features in the panorama')
    kq, dq = sift.detectAndCompute(qm, None)
    job.check()
    job.progress(0.45, f'matching {len(kp)} photo features against {len(kq)} panorama features')
    p0, p1, _ = SiftMatcher.match_desc((np.float32([k.pt for k in kp]).reshape(-1, 2), dp),
                                       (np.float32([k.pt for k in kq]).reshape(-1, 2), dq))
    res = best_homography(p0, p1)
    inl = 0 if res is None else int(res[1].sum())
    job.log('info', f'add photo: {len(p0)} SIFT matches, RANSAC {inl}/{len(p0)} inliers')
    if res is None or inl < 12 or inl / max(1, len(p0)) < 0.25:
        raise RuntimeError(f"couldn't place {photo_name} in the panorama reliably — not enough consistent matches (does it overlap it?)")
    Hm = res[0]
    Hf = np.diag([W0 / qm.shape[1], H0 / qm.shape[0], 1.0]) @ Hm @ np.diag([pm.shape[1] / pw, pm.shape[0] / ph, 1.0])
    return Hf, inl, len(p0)


def _place_photo(pano_rgb, pano_alpha, photo_rgb, Hf, mode, job, photo_name, blend, inl, nmatch):
    """Grows the canvas for the placed photo and blends it in (see add_photo)."""
    ph, pw = photo_rgb.shape[:2]
    H0, W0 = pano_rgb.shape[:2]
    xs, ys = [], []
    for t in np.linspace(0, 1, 17):
        for x, y in ((t * pw, 0), (t * pw, ph), (0, t * ph), (pw, t * ph)):
            v = Hf @ np.array([x, y, 1.0])
            if not v[2] > 1e-9: raise RuntimeError('the matched position folds the photo behind the camera — not trusting it')
            xs.append(v[0] / v[2]); ys.append(v[1] / v[2])
    area = (max(xs) - min(xs)) * (max(ys) - min(ys)) / (pw * ph)
    if not 0.02 < area < 50: raise RuntimeError('the match would shrink or stretch the photo implausibly — not trusting it')
    ox, oy = -math.floor(min(0, min(xs))), -math.floor(min(0, min(ys)))
    W, H = math.ceil(max(W0, max(xs))) + ox, math.ceil(max(H0, max(ys))) + oy
    bx0, by0 = max(0, math.floor(min(xs) + ox)), max(0, math.floor(min(ys) + oy))
    bx1, by1 = min(W, math.ceil(max(xs) + ox)), min(H, math.ceil(max(ys) + oy))
    dtype = np.uint16 if (pano_rgb.dtype == np.uint16 or photo_rgb.dtype == np.uint16) else np.uint8
    maxv = 65535.0 if dtype == np.uint16 else 255.0
    out = np.zeros((H, W, 3), dtype)
    alpha = np.zeros((H, W), np.uint8)
    pr = pano_rgb if pano_rgb.dtype == dtype else (pano_rgb.astype(np.uint16) * 257)
    out[oy:oy + H0, ox:ox + W0] = pr
    if pano_alpha is not None:
        alpha[oy:oy + H0, ox:ox + W0] = pano_alpha
    else:  # no alpha: treat (near-)black as empty, the way stitched panoramas fill their gaps
        alpha[oy:oy + H0, ox:ox + W0] = (to_uint8(pano_rgb).astype(np.uint16).sum(-1) > 8).astype(np.uint8) * 255
    Hinv = np.linalg.inv(Hf)
    if blend == 'multiband':
        _blend_photo_multiband(out, alpha, photo_rgb, Hinv, ox, oy, (bx0, by0, bx1, by1), mode, job, photo_name, maxv, dtype)
        return out, alpha, {'inliers': inl, 'matches': nmatch, 'size': (W, H)}
    job.progress(0.6, f'warping {photo_name} into the panorama')
    rows = max(16, 4_000_000 // max(1, bx1 - bx0))
    for y0 in range(by0, by1, rows):
        job.check()
        y1 = min(by1, y0 + rows)
        yy, xx = np.mgrid[y0:y1, bx0:bx1].astype(np.float64)
        X, Y = xx + 0.5 - ox, yy + 0.5 - oy
        u = Hinv[0, 0] * X + Hinv[0, 1] * Y + Hinv[0, 2]
        v = Hinv[1, 0] * X + Hinv[1, 1] * Y + Hinv[1, 2]
        w = Hinv[2, 0] * X + Hinv[2, 1] * Y + Hinv[2, 2]
        ok = np.abs(w) >= 1e-9
        w = np.where(ok, w, 1.0)
        sx, sy = u / w, v / w
        ok &= (sx >= 0) & (sy >= 0) & (sx < pw) & (sy < ph)
        if not ok.any(): continue
        fu, fv = sx / pw, sy / ph
        wt = np.clip(np.minimum(np.minimum(fu, 1 - fu), np.minimum(fv, 1 - fv)) / FEATHER, 0, 1)
        wt = np.where(ok, np.maximum(wt, MIN_WEIGHT), 0.0)
        c = cv2.remap(photo_rgb, (sx - 0.5).astype(np.float32), (sy - 0.5).astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / (65535.0 if photo_rgb.dtype == np.uint16 else 255.0)
        reg = out[y0:y1, bx0:bx1].astype(np.float32) / maxv
        cov = alpha[y0:y1, bx0:bx1] > 0
        k = np.where(cov, wt, 1.0) if mode != 'fill' else np.where(cov, 0.0, 1.0)
        k = np.where(ok, k, 0.0)[..., None]
        out[y0:y1, bx0:bx1] = (np.clip(reg * (1 - k) + c * k, 0, 1) * maxv + 0.5).astype(dtype)
        alpha[y0:y1, bx0:bx1] = np.where(ok | cov, 255, 0).astype(np.uint8)
        job.progress(0.6 + 0.4 * (y1 - by0) / max(1, by1 - by0), f'warping {photo_name} into the panorama')
    return out, alpha, {'inliers': inl, 'matches': nmatch, 'size': (W, H)}


def pyramid_blend(imgs, oks, weights, job=None):
    """Burt–Adelson multi-band blend of a few same-size images: each (filled smoothly outside its
    coverage `ok`) is split into a Laplacian pyramid and each band is blended with its weight's
    Gaussian pyramid. Returns float32 HxWx3."""
    h, w = imgs[0].shape[:2]
    L = int(np.clip(math.floor(math.log2(max(h, w) / 32)), 1, 10))
    S = 2 ** L
    hp, wp = -(-h // S) * S, -(-w // S) * S
    acc = [np.zeros((hp >> l, wp >> l, 3), np.float32) for l in range(L + 1)]
    ws = [np.zeros((hp >> l, wp >> l), np.float32) for l in range(L + 1)]
    for I, ok, wt in zip(imgs, oks, weights):
        if job: job.check()
        G = np.zeros((hp, wp, 3), np.float32); okp = np.zeros((hp, wp), bool); M = np.zeros((hp, wp), np.float32)
        G[:h, :w], okp[:h, :w], M[:h, :w] = I, ok, wt
        G = Layout._fill_outside(G, okp)
        for l in range(L + 1):
            if l < L:
                Gn = cv2.pyrDown(G)
                band = G - cv2.pyrUp(Gn, dstsize=(G.shape[1], G.shape[0]))
            else:
                Gn, band = None, G
            acc[l] += band * M[..., None]
            ws[l] += M
            if l < L: G, M = Gn, cv2.pyrDown(M)
    img = acc[L] / np.maximum(ws[L], 1e-6)[..., None]
    for l in range(L - 1, -1, -1):
        img = cv2.pyrUp(img, dstsize=(wp >> l, hp >> l)) + acc[l] / np.maximum(ws[l], 1e-6)[..., None]
    return img[:h, :w]


def _blend_photo_multiband(out, alpha, photo_rgb, Hinv, ox, oy, box, mode, job, photo_name, maxv, dtype):
    """Blends a placed photo into the canvas (in place) with a seam and multi-band blending: a two-way
    graph cut picks the seam between the photo and what's already there (where they agree; preferring
    the photo for 'top', the panorama for 'fill', the sharper of the two for 'sharp'), the photo's
    brightness is matched to the panorama's in the overlap, and the two are pyramid-blended."""
    import maxflow
    Hc, Wc = out.shape[:2]
    ph, pw = photo_rgb.shape[:2]
    bx0, by0, bx1, by1 = box
    pad = 48
    x0, y0, x1, y1 = max(0, bx0 - pad), max(0, by0 - pad), min(Wc, bx1 + pad), min(Hc, by1 + pad)
    h, w = y1 - y0, x1 - x0
    job.progress(0.55, f'warping {photo_name} into the panorama')
    B = np.zeros((h, w, 3), np.float32); okB = np.zeros((h, w), bool); ramp = np.zeros((h, w), np.float32)
    pmax = 65535.0 if photo_rgb.dtype == np.uint16 else 255.0
    rows = max(16, 4_000_000 // max(1, w))
    for r0 in range(0, h, rows):
        job.check()
        r1 = min(h, r0 + rows)
        yy, xx = np.mgrid[y0 + r0:y0 + r1, x0:x1].astype(np.float64)
        X, Y = xx + 0.5 - ox, yy + 0.5 - oy
        u = Hinv[0, 0] * X + Hinv[0, 1] * Y + Hinv[0, 2]
        v = Hinv[1, 0] * X + Hinv[1, 1] * Y + Hinv[1, 2]
        wv = Hinv[2, 0] * X + Hinv[2, 1] * Y + Hinv[2, 2]
        ok = np.abs(wv) >= 1e-9
        wv = np.where(ok, wv, 1.0)
        sx, sy = u / wv, v / wv
        ok &= (sx >= 0) & (sy >= 0) & (sx < pw) & (sy < ph)
        B[r0:r1] = cv2.remap(photo_rgb, (sx - 0.5).astype(np.float32), (sy - 0.5).astype(np.float32), cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / pmax
        okB[r0:r1] = ok
        fu, fv = sx / pw, sy / ph
        ramp[r0:r1] = np.where(ok, np.clip(np.minimum(np.minimum(fu, 1 - fu), np.minimum(fv, 1 - fv)) / FEATHER, 0, 1), 0)
    A = out[y0:y1, x0:x1].astype(np.float32) / maxv
    okA = alpha[y0:y1, x0:x1] > 0
    both = okA & okB
    if both.sum() > 500:                                     # match the photo's brightness to the panorama
        la, lb = A[both].mean(-1), B[both].mean(-1)
        g = float(np.clip(np.median(la) / max(1e-4, np.median(lb)), 0.5, 2.0))
        B *= g
        job.log('info', f'add photo: brightness matched to the panorama (×{g:.3f})')
    # seam: two-way graph cut on a reduced grid
    job.progress(0.65, 'finding the seam between the photo and the panorama')
    k = min(1.0, 1500 / max(h, w))
    sz = (max(2, round(w * k)), max(2, round(h * k)))
    As, Bs = cv2.resize(A, sz, interpolation=cv2.INTER_AREA), cv2.resize(B, sz, interpolation=cv2.INTER_AREA)
    oAs = cv2.resize(okA.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0
    oBs = cv2.resize(okB.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0
    rBs = cv2.resize(ramp, sz, interpolation=cv2.INTER_AREA)
    bs = oAs & oBs
    INF = 1e6
    cA, cB = np.zeros(oAs.shape), 0.3 * (1 - rBs)             # seams keep off the photo's own border
    if mode == 'top':
        cA[bs] += 0.15
    elif mode == 'fill':
        cB[bs] += 0.15
    else:                                                     # sharpest wins
        def sharp(img):
            g = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            return cv2.resize(cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F)), -1, (31, 31)), sz, interpolation=cv2.INTER_AREA)
        sA, sB = sharp(A), sharp(B)
        top = np.maximum(np.maximum(sA, sB), 1e-6)
        cA += 0.25 * np.clip(1 - sA / top - 0.1, 0, None); cB += 0.25 * np.clip(1 - sB / top - 0.1, 0, None)
    cA[~oAs] = INF; cB[~oBs] = INF
    neither = ~oAs & ~oBs
    cA[neither] = cB[neither] = 0
    d = np.where(bs, np.minimum(np.abs(As - Bs).mean(-1) * 1.5, 1.0), 1.0)
    wr = np.zeros(d.shape); wr[:, :-1] = d[:, :-1] + d[:, 1:] + 0.03
    wd = np.zeros(d.shape); wd[:-1, :] = d[:-1, :] + d[1:, :] + 0.03
    gph = maxflow.Graph[float]()
    nodes = gph.add_grid_nodes(d.shape)
    gph.add_grid_edges(nodes, weights=wr, structure=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]), symmetric=True)
    gph.add_grid_edges(nodes, weights=wd, structure=np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]), symmetric=True)
    gph.add_grid_tedges(nodes, cB, cA)                       # sink side = photo: pays its cost on the source edge
    gph.maxflow()
    labB = gph.get_grid_segments(nodes).astype(np.float32)
    mB = cv2.GaussianBlur(cv2.resize(labB, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), max(1.0, 1.0 / k))
    wB = np.where(okB, np.maximum(mB, 1e-3), 0).astype(np.float32)
    wA = np.where(okA, np.maximum(1 - mB, 1e-3), 0).astype(np.float32)
    job.progress(0.75, 'multi-band blending the photo in')
    res = pyramid_blend([A, B], [okA, okB], [wA, wB], job)
    cov = okA | okB
    out[y0:y1, x0:x1] = np.where(cov[..., None], (np.clip(res, 0, 1) * maxv + 0.5).astype(dtype), out[y0:y1, x0:x1])
    alpha[y0:y1, x0:x1] = cov.astype(np.uint8) * 255
    job.log('info', f'add photo: seam chosen — the photo supplies {100 * float(labB[bs].mean()) if bs.any() else 0:.0f}% of the overlap')


# ───────────────────────────────────────────── zenith / nadir patches
def extract_pole(eq_rgb, zenith, fov_deg=110, size=2048):
    """Gnomonic patch centred on the zenith (top of the equirectangular image) or nadir (bottom)."""
    H, W = eq_rgb.shape[:2]
    f = (size / 2) / math.tan(math.radians(fov_deg) / 2)
    sign = 1.0 if zenith else -1.0
    cy, cx = np.mgrid[0:size, 0:size].astype(np.float64) - size / 2
    dx, dy, dz = cx / f, np.full_like(cx, sign), cy / f
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    lon, lat = np.arctan2(dx / n, dz / n), np.arcsin(np.clip(dy / n, -1, 1))
    sx = (lon / (2 * math.pi) + 0.5) * W - 0.5
    sy = (0.5 - lat / math.pi) * H - 0.5
    return cv2.remap(eq_rgb, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def composite_poles(eq_rgb, zen, nad, fov_deg=110, feather_deg=8):
    """Projects processed zenith/nadir patches back onto the equirectangular image, feathered at the cone edge."""
    out = eq_rgb.copy()
    H, W = eq_rgb.shape[:2]
    maxv = 65535.0 if eq_rgb.dtype == np.uint16 else 255.0
    outer = math.radians(fov_deg) / 2
    inner = max(0.0, outer - math.radians(feather_deg))
    for patch, sign in ((zen, 1.0), (nad, -1.0)):
        if patch is None: continue
        ph, pw = patch.shape[:2]
        pf = (pw / 2) / math.tan(outer)
        lat_edge = math.pi / 2 - outer
        if sign > 0: y0, y1 = 0, min(H, math.ceil((0.5 - lat_edge / math.pi) * H) + 2)
        else: y0, y1 = max(0, math.floor((0.5 + lat_edge / math.pi) * H) - 2), H
        yy, xx = np.mgrid[y0:y1, 0:W].astype(np.float64)
        lat = (0.5 - (yy + 0.5) / H) * math.pi
        lon = ((xx + 0.5) / W - 0.5) * 2 * math.pi
        colat = math.pi / 2 - lat if sign > 0 else math.pi / 2 + lat
        dx, dy, dz = np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat)
        ok = (sign * dy > 1e-6) & (colat <= outer)
        dy = np.where(ok, dy, 1.0)
        px = pf * sign * dx / dy + pw / 2
        py = pf * sign * dz / dy + ph / 2
        ok &= (px >= 0) & (py >= 0) & (px < pw - 1) & (py < ph - 1)
        wt = np.where(colat <= inner, 1.0, (outer - colat) / max(1e-6, outer - inner))
        wt = np.where(ok, np.clip(wt, 0, 1), 0.0)[..., None]
        pp = patch if patch.dtype == eq_rgb.dtype else (patch.astype(np.uint16) * 257 if eq_rgb.dtype == np.uint16 else to_uint8(patch))
        c = cv2.remap(pp, px.astype(np.float32), py.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        reg = out[y0:y1].astype(np.float32)
        out[y0:y1] = np.clip(reg * (1 - wt) + c * wt, 0, maxv).astype(eq_rgb.dtype)
    return out
