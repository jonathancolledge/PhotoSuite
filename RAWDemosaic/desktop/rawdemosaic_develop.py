# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Develop for RAW Demosaic desktop: the web app's Develop panel (rawdemosaic.html, develop-src and geom-src)
ported to numpy and Numba. Real-time, Lightroom-style adjustments of a linear Rec.2020 image.

Order (as the web app): retouching (spots, AI fills) -> geometry (rotation, perspective, Lensfun lens model,
lateral CA) -> outpainted areas -> white balance & exposure -> dehaze -> shadows/midtones/highlights (local)
-> tone (whites/blacks/brightness/contrast) -> microcontrast/clarity/local contrast -> defringe -> colour
(vibrance/saturation/mixer/equalizer/grading, in OkLab) -> curves -> unsharp mask; local adjustments are
blended in by their masks.

Parameter dicts use the web app's names ("worker params"), so settings files are shared between the two.
"""
import math

import numpy as np
from numba import njit, prange

_C = True   # Numba on-disk cache


def jround(x):
    """JavaScript's Math.round (halves go up; Python's round() goes to even)."""
    return int(math.floor(x + 0.5))


# ───────────────────────────────────────────── colour maths
def _xy(x, y):
    return np.array([x / y, 1.0, (1 - x - y) / y])


def rgb_to_xyz(prim, w):
    P = np.stack([_xy(*p) for p in prim], axis=1)
    return P * np.linalg.solve(P, _xy(*w))[None, :]


D65 = (0.3127, 0.3290)
R2020_TO_XYZ = rgb_to_xyz([(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)], D65)
XYZ_TO_R2020 = np.linalg.inv(R2020_TO_XYZ)
SRGB_TO_XYZ = rgb_to_xyz([(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)], D65)
R2020_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ) @ R2020_TO_XYZ
SRGB_TO_R2020 = np.linalg.inv(R2020_TO_SRGB)
KY = R2020_TO_XYZ[1].copy()
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
OK_M1 = np.array([[0.8189330101, 0.3618667424, -0.1288597137], [0.0329845436, 0.9293118715, 0.0361456387], [0.0482003018, 0.2643662691, 0.6338517070]])
OK_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099], [0.0259040371, 0.7827717662, -0.8086757660]])
R2020_TO_LMS = OK_M1 @ R2020_TO_XYZ
LMS_TO_R2020 = np.linalg.inv(R2020_TO_LMS)
OK_M2I = np.linalg.inv(OK_M2)


def _encS(v):
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(np.maximum(v, 0), 1 / 2.4) - 0.055)


def decS(e):
    e = np.asarray(e, np.float64)
    return np.where(e <= 0.04045, e / 12.92, np.power((e + 0.055) / 1.055, 2.4))


ENC_N = 4095
ENC = _encS(np.arange(ENC_N + 1) / ENC_N).astype(np.float64)   # the web app's 4096-entry sRGB encode table


@njit(inline='always')
def _enc(v, ENC):
    if v < 0.0: v = 0.0
    elif v > 1.0: v = 1.0
    return ENC[int(v * 4095 + 0.5)]


def planck_xy(T):
    T = min(25000, max(1667, T))
    x = (-0.2661239e9 / T ** 3 - 0.2343589e6 / T ** 2 + 0.8776956e3 / T + 0.179910) if T <= 4000 else \
        (-3.0258469e9 / T ** 3 + 2.1070379e6 / T ** 2 + 0.2226347e3 / T + 0.240390)
    if T <= 2222: y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
    elif T <= 4000: y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
    else: y = 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483
    return x, y


def wb_matrix(temp, tint):
    """White balance: the assumed scene white moves along the Planckian locus (temperature, mireds) and
    across it (tint), and is chromatically adapted (Bradford) to D65. None when neutral."""
    if not temp and not tint: return None
    ref = 1e6 / 6504
    T = 1e6 / max(40, ref - temp * 1.0)
    uv = lambda p: (4 * p[0] / (-2 * p[0] + 12 * p[1] + 3), 6 * p[1] / (-2 * p[0] + 12 * p[1] + 3))
    xy = lambda q: (3 * q[0] / (2 * q[0] - 8 * q[1] + 4), 2 * q[1] / (2 * q[0] - 8 * q[1] + 4))
    u0, v0 = uv(planck_xy(1e6 / ref)); u1, v1 = uv(planck_xy(T)); u2, v2 = uv(planck_xy(T * 1.01))
    du, dv = u2 - u1, v2 - v1
    ln = math.hypot(du, dv) or 1
    nu, nv = -dv / ln, du / ln
    ud, vd = uv(D65)
    duv = -tint * 0.00025
    src = xy((ud + (u1 - u0) + nu * duv, vd + (v1 - v0) + nv * duv))
    s, d = BRADFORD @ _xy(*src), BRADFORD @ _xy(*D65)
    A = np.linalg.inv(BRADFORD) @ np.diag(d / s) @ BRADFORD
    return XYZ_TO_R2020 @ A @ R2020_TO_XYZ


def oklab_hex(L, C, h):
    a, b = C * math.cos(math.radians(h)), C * math.sin(math.radians(h))
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    e = lambda v: jround(255 * max(0, min(1, 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055)))
    return '#%02x%02x%02x' % (e(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s), e(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
                              e(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s))


# ───────────────────────────────────────────── blurs (exact Gaussian when small, three boxes when large)
def gauss_spec(sigma):
    r = int(math.floor(4 * sigma + 0.5))
    if r <= 12:
        k = np.exp(-0.5 * np.arange(-r, r + 1) ** 2 / (sigma * sigma))
        return ('k', (k / k.sum()).astype(np.float64))
    n = 3
    wl = int(math.floor(math.sqrt(12 * sigma * sigma / n + 1)))
    if wl % 2 == 0: wl -= 1
    wu = wl + 2
    m = jround((12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4))
    return ('b', [wl if i < m else wu for i in range(3)])


@njit(inline='always')
def _rhs(i, n):   # half-sample symmetric reflection (c b a | a b c), any distance out
    while i < 0 or i >= n:
        if i < 0: i = -i - 1
        if i >= n: i = 2 * n - 1 - i
    return i


@njit(parallel=True, nogil=True, cache=_C)
def _conv_rows(src, ker, out):
    h, w = src.shape
    r = (ker.shape[0] - 1) // 2
    for y in prange(h):
        for x in range(w):
            s = 0.0
            if x >= r and x < w - r:
                for j in range(-r, r + 1): s += ker[j + r] * src[y, x + j]
            else:
                for j in range(-r, r + 1): s += ker[j + r] * src[y, _rhs(x + j, w)]
            out[y, x] = s


@njit(parallel=True, nogil=True, cache=_C)
def _conv_cols(src, ker, out):
    h, w = src.shape
    r = (ker.shape[0] - 1) // 2
    for y in prange(h):
        acc = np.zeros(w)
        for j in range(-r, r + 1):
            yy = _rhs(y + j, h); kk = ker[j + r]
            for x in range(w): acc[x] += kk * src[yy, x]
        for x in range(w): out[y, x] = acc[x]


@njit(parallel=True, nogil=True, cache=_C)
def _box_rows(src, sizes, out):
    h, w = src.shape
    for y in prange(h):
        a = np.empty(w); b = np.empty(w)
        for x in range(w): a[x] = src[y, x]
        for size in sizes:
            r = (size - 1) // 2; inv = 1.0 / size
            s = 0.0
            for j in range(-r, r + 1): s += a[_rhs(j, w)]
            for x in range(w):
                b[x] = s * inv
                s += a[_rhs(x + r + 1, w)] - a[_rhs(x - r, w)]
            t = a; a = b; b = t
        for x in range(w): out[y, x] = a[x]


@njit(parallel=True, nogil=True, cache=_C)
def _box_cols(src, sizes, out):
    h, w = src.shape
    nb = (w + 63) // 64
    for bi in prange(nb):
        x0 = bi * 64; x1 = min(w, x0 + 64); m = x1 - x0
        a = np.empty((h, m)); b = np.empty((h, m)); s = np.empty(m)
        for y in range(h):
            for x in range(m): a[y, x] = src[y, x0 + x]
        for size in sizes:
            r = (size - 1) // 2; inv = 1.0 / size
            for x in range(m): s[x] = 0.0
            for j in range(-r, r + 1):
                yy = _rhs(j, h)
                for x in range(m): s[x] += a[yy, x]
            for y in range(h):
                yp = _rhs(y + r + 1, h); ym = _rhs(y - r, h)
                for x in range(m):
                    b[y, x] = s[x] * inv
                    s[x] += a[yp, x] - a[ym, x]
            t = a; a = b; b = t
        for y in range(h):
            for x in range(m): out[y, x0 + x] = a[y, x]


def blur(a, sigma):
    """2-D float32 blur with half-sample-symmetric edges, as the web app's: an exact Gaussian kernel when its
    radius is at most 12, else three box passes."""
    kind, spec = gauss_spec(sigma)
    src = np.ascontiguousarray(a, np.float32)
    tmp = np.empty_like(src); out = np.empty_like(src)
    if kind == 'k':
        _conv_rows(src, spec, tmp); _conv_cols(tmp, spec, out)
    else:
        sizes = np.array(spec, np.int64)
        _box_rows(src, sizes, tmp); _box_cols(tmp, sizes, out)
    return out


# ───────────────────────────────────────────── geometry (geom-src)
def canvas_dims(w, h, g):
    return (h, w) if g['rot90'] % 2 else (w, h)


# lens model packed for Numba: [dist model (0 none, 1 ptlens, 2 poly3, 3 poly5), a, b, c, dA,
#  tca mode (0 none, 1 linear, 2 poly3, 3 measured ca), t0..t5, vig on, k1, k2, k3, vA, uD, uV, uT]
def pack_lens(lens, sc):
    P = np.zeros(22)
    P[19] = P[20] = P[21] = 1.0
    if not lens: return P
    dA = lens.get('distAmt', 1) if lens.get('distAmt') is not None else 1
    vA = lens.get('vigAmt', 1) if lens.get('vigAmt') is not None else 1
    d = lens.get('dist')
    if d and dA:
        P[0] = {'ptlens': 1, 'poly3': 2, 'poly5': 3}.get(d['model'], 0)
        P[1:4] = (list(d['terms']) + [0, 0, 0])[:3]
        P[4] = dA
    if lens.get('ca'):
        c = lens['ca']; P[5] = 3; P[6:10] = [c['vr'], c['br'], c['vb'], c['bb']]
    elif lens.get('tca'):
        t = lens['tca']
        if t['model'] == 'linear': P[5] = 1; P[6:8] = t['terms'][:2]
        else: P[5] = 2; P[6:12] = t['terms'][:6]
    v = lens.get('vig')
    if v and vA:
        P[12] = 1; P[13:16] = v['terms'][:3]; P[16] = vA
    uD = lens.get('unitD', 1) * sc; uV = lens.get('unitV', 1) * sc
    uT = lens['caUnit'] * sc if lens.get('ca') else uD
    P[19], P[20], P[21] = uD, uV, uT
    return P


@njit(inline='always')
def _dist(P, r):
    m = P[0]
    a, b, c = P[1], P[2], P[3]
    if m == 1:
        d = 1 - a - b - c; q = r / d
        base = (a * q * q * q + b * q * q + c * q + d) / d
    elif m == 2:
        d = 1 - a; k = a / (d * d * d); base = 1 + k * r * r
    elif m == 3:
        base = 1 + a * r * r + b * r * r * r * r
    else:
        return 1.0
    return base if P[4] == 1 else 1 + (base - 1) * P[4]


@njit(inline='always')
def _to_src(x, y, k, CW, CH, sw, sh):
    rx = x + CW / 2; ry = y + CH / 2
    if k == 0: return rx, ry
    if k == 1: return ry, sh - rx
    if k == 2: return sw - rx, sh - ry
    return sw - ry, rx


@njit(inline='always')
def _map(px, py, k, co, si, S, Hi, hasH, P, CW, CH, sw, sh, out):
    """Output-canvas position (relative to its centre, source pixels) -> [gx, gy, rx, ry, bx, by, gain]."""
    x = co * px - si * py; y = si * px + co * py
    if hasH:
        u = x / S; v = y / S; w = Hi[6] * u + Hi[7] * v + Hi[8]
        if w <= 1e-6: return False
        x = (Hi[0] * u + Hi[1] * v + Hi[2]) / w * S; y = (Hi[3] * u + Hi[4] * v + Hi[5]) / w * S
    if P[0] != 0:
        f = _dist(P, math.hypot(x, y) / P[19]); x *= f; y *= f
    out[0], out[1] = _to_src(x, y, k, CW, CH, sw, sh)
    if P[5] != 0:
        r = math.hypot(x, y) / P[21]
        if P[5] == 3: fr = P[6] + P[7] * r * r; fb = P[8] + P[9] * r * r
        elif P[5] == 1: fr = P[6]; fb = P[7]
        else: fr = P[10] * r * r + P[8] * r + P[6]; fb = P[11] * r * r + P[9] * r + P[7]
        out[2], out[3] = _to_src(x * fr, y * fr, k, CW, CH, sw, sh)
        out[4], out[5] = _to_src(x * fb, y * fb, k, CW, CH, sw, sh)
    else:
        out[2] = out[4] = out[0]; out[3] = out[5] = out[1]
    if P[12] != 0:
        r = math.hypot(x, y) / P[20]; r2 = r * r
        v = 1 + P[13] * r2 + P[14] * r2 * r2 + P[15] * r2 * r2 * r2
        out[6] = 1 + (1 / max(0.05, v) - 1) * P[16]
    else:
        out[6] = 1.0
    return True


@njit(inline='always')
def _at(data, sw, sh, x, y, ch):
    x -= 0.5; y -= 0.5
    if x < -0.5 or y < -0.5 or x > sw - 0.5 or y > sh - 0.5: return 0.0
    xi = min(max(int(math.floor(x)), 0), sw - 2); yi = min(max(int(math.floor(y)), 0), sh - 2)
    fx = min(max(x - xi, 0.0), 1.0); fy = min(max(y - yi, 0.0), 1.0)
    return (data[yi, xi, ch] * (1 - fx) + data[yi, xi + 1, ch] * fx) * (1 - fy) + (data[yi + 1, xi, ch] * (1 - fx) + data[yi + 1, xi + 1, ch] * fx) * fy


@njit(parallel=True, nogil=True, cache=_C)
def _apply_geometry(data, ow, oh, k, co, si, S, Hi, hasH, P, CW, CH, x0, y0, sc, tca, out, valid, want_valid):
    sh, sw = data.shape[0], data.shape[1]
    for oy in prange(oh):
        Q = np.empty(7)
        for ox in range(ow):
            ok = _map(x0 + (ox + 0.5) / sc - CW / 2, y0 + (oy + 0.5) / sc - CH / 2, k, co, si, S, Hi, hasH, P, CW, CH, sw, sh, Q)
            if not ok:
                out[oy, ox, 0] = 0; out[oy, ox, 1] = 0; out[oy, ox, 2] = 0
                continue
            if want_valid and Q[0] >= 1 and Q[1] >= 1 and Q[0] <= sw - 1 and Q[1] <= sh - 1: valid[oy, ox] = 1
            gn = Q[6]
            if tca:
                out[oy, ox, 0] = _at(data, sw, sh, Q[2], Q[3], 0) * gn
                out[oy, ox, 1] = _at(data, sw, sh, Q[0], Q[1], 1) * gn
                out[oy, ox, 2] = _at(data, sw, sh, Q[4], Q[5], 2) * gn
            else:
                out[oy, ox, 0] = _at(data, sw, sh, Q[0], Q[1], 0) * gn
                out[oy, ox, 1] = _at(data, sw, sh, Q[0], Q[1], 1) * gn
                out[oy, ox, 2] = _at(data, sw, sh, Q[0], Q[1], 2) * gn


def _geo_args(g, lens, sw, sh, sc_src):
    CW, CH = canvas_dims(sw, sh, g)
    t = -(g.get('angle') or 0) * math.pi / 180
    Hi = np.asarray(g['Hinv'], np.float64) if g.get('Hinv') else np.zeros(9)
    return CW, CH, g['rot90'] & 3, math.cos(t), math.sin(t), max(CW, CH) / 2, Hi, bool(g.get('Hinv')), pack_lens(lens, sc_src)


def lens_on(lens):
    return bool(lens and (lens.get('dist') or lens.get('tca') or lens.get('vig') or lens.get('ca')))


def apply_geometry(src, g, crop, max_side, lens, full_w, want_valid=False):
    """src: HxWx3 float32 (source orientation). Returns dict(data, w, h, valid)."""
    sh, sw = src.shape[:2]
    CW, CH = canvas_dims(sw, sh, g)
    lon = lens_on(lens)
    identity = not g['rot90'] and not g.get('angle') and not g.get('Hinv') and not lon and crop['x'] == 0 and crop['y'] == 0 and crop['w'] == 1 and crop['h'] == 1
    ow, oh, sc = max(1, jround(crop['w'] * CW)), max(1, jround(crop['h'] * CH)), 1.0
    if max_side and max(ow, oh) > max_side:
        sc = max_side / max(ow, oh); ow, oh = max(1, jround(ow * sc)), max(1, jround(oh * sc))
    if identity and sc == 1:
        return dict(data=src, w=sw, h=sh, valid=None)
    CW, CH, k, co, si, S, Hi, hasH, P = _geo_args(g, lens if lon else None, sw, sh, sw / full_w)
    out = np.empty((oh, ow, 3), np.float32)
    valid = np.zeros((oh, ow), np.uint8) if want_valid else np.zeros((1, 1), np.uint8)
    tca = bool(lon and (lens.get('tca') or lens.get('ca')))
    _apply_geometry(src, ow, oh, k, co, si, S, Hi, hasH, P, CW, CH, crop['x'] * CW, crop['y'] * CH, sc, tca, out, valid, want_valid)
    return dict(data=out, w=ow, h=oh, valid=valid if want_valid else None)


def map_points(g, lens, sw, sh, sc_src, pts):
    """Map output-canvas points (relative to the canvas centre) to source positions; None where invalid."""
    CW, CH, k, co, si, S, Hi, hasH, P = _geo_args(g, lens, sw, sh, sc_src)
    return _map_points(np.asarray(pts, np.float64), k, co, si, S, Hi, hasH, P, CW, CH, sw, sh)


@njit(cache=_C)
def _map_points(pts, k, co, si, S, Hi, hasH, P, CW, CH, sw, sh):
    out = np.full((pts.shape[0], 2), np.nan)
    Q = np.empty(7)
    for i in range(pts.shape[0]):
        if _map(pts[i, 0], pts[i, 1], k, co, si, S, Hi, hasH, P, CW, CH, sw, sh, Q):
            out[i, 0] = Q[0]; out[i, 1] = Q[1]
    return out


@njit(parallel=True, nogil=True, cache=_C)
def _mask_alpha(r, k, co, si, S, Hi, hasH, P, CW, CH, cx, cy, cw, ch, TW, TH, rx0, ry0, w, h, out):
    rh, rw = r.shape
    for oy in prange(h):
        Q = np.empty(7)
        v = cy + ch * (ry0 + oy + 0.5) / TH
        for ox in range(w):
            u = cx + cw * (rx0 + ox + 0.5) / TW
            if not _map((u - 0.5) * CW, (v - 0.5) * CH, k, co, si, S, Hi, hasH, P, CW, CH, rw, rh, Q): continue
            if Q[0] < 0 or Q[1] < 0 or Q[0] > rw or Q[1] > rh: continue
            x = max(0.0, min(rw - 1.001, Q[0] - 0.5)); y = max(0.0, min(rh - 1.001, Q[1] - 0.5))
            xi = int(x); yi = int(y); fx = x - xi; fy = y - yi
            val = (r[yi, xi] * (1 - fx) + r[yi, xi + 1] * fx) * (1 - fy) + (r[yi + 1, xi] * (1 - fx) + r[yi + 1, xi + 1] * fx) * fy
            out[oy, ox] = np.uint8(int(val))


def mask_alpha(raster, g, lens, crop, TW, TH, rx0, ry0, w, h, full_w):
    """A mask raster (source orientation, uint8) mapped through the geometry onto output pixels."""
    ln = dict(lens, tca=None, vig=None, ca=None) if lens else None
    rh, rw = raster.shape
    CW, CH, k, co, si, S, Hi, hasH, P = _geo_args(g, ln, rw, rh, rw / full_w)
    out = np.zeros((h, w), np.uint8)
    _mask_alpha(raster, k, co, si, S, Hi, hasH, P, CW, CH, crop['x'], crop['y'], crop['w'], crop['h'], TW, TH, rx0, ry0, w, h, out)
    return out


# ───────────────────────────────────────────── the develop pipeline
@njit(parallel=True, nogil=True, cache=_C)
def _core(src, M, Dk, dh, A, gain, Bsh, lg, shA, hiA, miA, tone_on, TONE, tone1, Ldet, Bmc, Bcl, Blc, mcA, clA, lcA, KY, ENC, out):
    h, w = src.shape[0], src.shape[1]
    TN = TONE.shape[0] - 1
    hasD, hasS, hasL = Dk.shape[0] > 0, Bsh.shape[0] > 0, Ldet.shape[0] > 0
    hasMc, hasCl, hasLc = Bmc.shape[0] > 0, Bcl.shape[0] > 0, Blc.shape[0] > 0
    for y in prange(h):
        for x in range(w):
            s0 = src[y, x, 0]; s1 = src[y, x, 1]; s2 = src[y, x, 2]
            r = M[0, 0] * s0 + M[0, 1] * s1 + M[0, 2] * s2
            g = M[1, 0] * s0 + M[1, 1] * s1 + M[1, 2] * s2
            b = M[2, 0] * s0 + M[2, 1] * s1 + M[2, 2] * s2
            if hasD:
                if dh > 0:
                    tr = max(0.15, 1 - 0.95 * dh * min(1.0, Dk[y, x] * gain / A))
                    r = (r - A) / tr + A; g = (g - A) / tr + A; b = (b - A) / tr + A
                    if r < 0: r = 0.0
                    if g < 0: g = 0.0
                    if b < 0: b = 0.0
                else:
                    kk = -dh * 0.5
                    r += (A - r) * kk; g += (A - g) * kk; b += (A - b) * kk
            Y = KY[0] * r + KY[1] * g + KY[2] * b
            if hasS:
                eB = _enc(2.0 ** (Bsh[y, x] + lg), ENC)
                wS = max(0.0, 1 - eB / 0.6); wH = max(0.0, (eB - 0.4) / 0.6); wM0 = max(0.0, 1 - ((eB - 0.5) / 0.4) ** 2)
                f = 2.0 ** (shA * wS * wS + hiA * wH * wH + miA * wM0 * wM0)
                r *= f; g *= f; b *= f; Y *= f
            if tone_on and Y > 1e-7:
                if Y >= 1: Y2 = tone1 * Y
                else: Y2 = TONE[int(_enc(Y, ENC) * TN + 0.5)]
                kk = Y2 / Y; r *= kk; g *= kk; b *= kk; Y = Y2
            if hasL:
                ex = 0.0
                if hasMc: ex += mcA * (Ldet[y, x] - Bmc[y, x])
                if hasCl:
                    e = _enc(Y, ENC); ex += clA * (Ldet[y, x] - Bcl[y, x]) * 4 * e * (1 - e)
                if hasLc: ex += lcA * (Ldet[y, x] - Blc[y, x])
                if ex != 0:
                    f = 2.0 ** max(-2.0, min(2.0, ex)); r *= f; g *= f; b *= f
            out[y, x, 0] = r; out[y, x, 1] = g; out[y, x, 2] = b


BANDS = np.array([29, 55, 110, 142, 195, 264, 300, 328], np.float64)   # red orange yellow green aqua blue purple magenta (OkLab hue)


@njit(parallel=True, nogil=True, cache=_C)
def _colour(px, sat, vib, any_hsl, hue_a, sat_a, lum_a, eq_on, eq_hue, eq_sat, eq_bri, any_gr, gOff, bal, M1, Mi, M2, M2I, BANDS):
    h, w = px.shape[0], px.shape[1]
    nb = BANDS.shape[0]
    for y in prange(h):
        for x in range(w):
            r = px[y, x, 0]; g = px[y, x, 1]; b = px[y, x, 2]
            l = np.cbrt(M1[0, 0] * r + M1[0, 1] * g + M1[0, 2] * b)
            m = np.cbrt(M1[1, 0] * r + M1[1, 1] * g + M1[1, 2] * b)
            s = np.cbrt(M1[2, 0] * r + M1[2, 1] * g + M1[2, 2] * b)
            L = M2[0, 0] * l + M2[0, 1] * m + M2[0, 2] * s
            A = M2[1, 0] * l + M2[1, 1] * m + M2[1, 2] * s
            B = M2[2, 0] * l + M2[2, 1] * m + M2[2, 2] * s
            C = math.hypot(A, B); hh = math.atan2(B, A) * 180 / math.pi
            if hh < 0: hh += 360
            f = sat
            if vib != 0: f *= 1 + vib * max(0.0, 1 - C / 0.2)
            if any_hsl and C > 1e-4:
                i0 = nb - 1
                for kk in range(nb):
                    if hh >= BANDS[kk]: i0 = kk
                i1 = (i0 + 1) % nb
                span = BANDS[i1] - BANDS[i0]
                if span <= 0: span += 360
                d = hh - BANDS[i0]
                if d < 0: d += 360
                u = d / span; u = u * u * (3 - 2 * u)
                w0 = 1 - u; w1 = u; cw = min(1.0, C / 0.06)
                hh += (w0 * hue_a[i0] + w1 * hue_a[i1]) / 100 * 30 * cw
                f *= 1 + (w0 * sat_a[i0] + w1 * sat_a[i1]) / 100
                L *= 1 + (w0 * lum_a[i0] + w1 * lum_a[i1]) / 100 * 0.3 * cw
            if eq_on and C > 1e-4:
                kq = ((hh % 360) + 360) % 360
                j0 = int(kq); j1 = (j0 + 1) % 360; u = kq - j0; cw = min(1.0, C / 0.06)
                if eq_hue.shape[0] > 0: hh += (eq_hue[j0] * (1 - u) + eq_hue[j1] * u) * cw
                if eq_sat.shape[0] > 0: f *= eq_sat[j0] * (1 - u) + eq_sat[j1] * u
                if eq_bri.shape[0] > 0: L *= 1 + (eq_bri[j0] * (1 - u) + eq_bri[j1] * u) * cw
            C *= max(0.0, f)
            hr = hh * math.pi / 180
            A = C * math.cos(hr); B = C * math.sin(hr)
            if any_gr:
                lt = min(1.0, max(0.0, L)) - bal * 0.25
                ws = max(0.0, 1 - lt / 0.5) ** 2; wh = max(0.0, (lt - 0.5) / 0.5) ** 2; wm = max(0.0, 1 - ws - wh)
                A += ws * gOff[0, 0] + wm * gOff[1, 0] + wh * gOff[2, 0]
                B += ws * gOff[0, 1] + wm * gOff[1, 1] + wh * gOff[2, 1]
            l2 = M2I[0, 0] * L + M2I[0, 1] * A + M2I[0, 2] * B
            m2 = M2I[1, 0] * L + M2I[1, 1] * A + M2I[1, 2] * B
            s2 = M2I[2, 0] * L + M2I[2, 1] * A + M2I[2, 2] * B
            l3 = l2 * l2 * l2; m3 = m2 * m2 * m2; s3 = s2 * s2 * s2
            px[y, x, 0] = max(0.0, Mi[0, 0] * l3 + Mi[0, 1] * m3 + Mi[0, 2] * s3)
            px[y, x, 1] = max(0.0, Mi[1, 0] * l3 + Mi[1, 1] * m3 + Mi[1, 2] * s3)
            px[y, x, 2] = max(0.0, Mi[2, 0] * l3 + Mi[2, 1] * m3 + Mi[2, 2] * s3)


@njit(parallel=True, nogil=True, cache=_C)
def _defringe(px, E, bands, M1, Mi, M2, M2I):
    h, w = px.shape[0], px.shape[1]
    for y in prange(h):
        for x in range(w):
            e = min(1.0, max(0.0, (E[y, x] - 0.03) / 0.2))
            if e <= 0: continue
            r = px[y, x, 0]; g = px[y, x, 1]; b = px[y, x, 2]
            l = np.cbrt(max(0.0, M1[0, 0] * r + M1[0, 1] * g + M1[0, 2] * b))
            m = np.cbrt(max(0.0, M1[1, 0] * r + M1[1, 1] * g + M1[1, 2] * b))
            s = np.cbrt(max(0.0, M1[2, 0] * r + M1[2, 1] * g + M1[2, 2] * b))
            Lk = M2[0, 0] * l + M2[0, 1] * m + M2[0, 2] * s
            A = M2[1, 0] * l + M2[1, 1] * m + M2[1, 2] * s
            Bk = M2[2, 0] * l + M2[2, 1] * m + M2[2, 2] * s
            C = math.hypot(A, Bk)
            if C < 0.01: continue
            hh = math.atan2(Bk, A) * 180 / math.pi
            if hh < 0: hh += 360
            kk = 0.0
            for j in range(bands.shape[0]):
                dh = abs(hh - bands[j, 1])
                if dh > 180: dh = 360 - dh
                if dh >= bands[j, 2]: continue
                wgt = 0.5 + 0.5 * math.cos(math.pi * dh / bands[j, 2])
                kk = max(kk, min(1.0, bands[j, 0] * e * wgt * 1.5))
            if kk == 0: continue
            f = 1 - kk; a2 = A * f; b2 = Bk * f
            l2 = M2I[0, 0] * Lk + M2I[0, 1] * a2 + M2I[0, 2] * b2
            m2 = M2I[1, 0] * Lk + M2I[1, 1] * a2 + M2I[1, 2] * b2
            s2 = M2I[2, 0] * Lk + M2I[2, 1] * a2 + M2I[2, 2] * b2
            l3 = l2 * l2 * l2; m3 = m2 * m2 * m2; s3 = s2 * s2 * s2
            px[y, x, 0] = max(0.0, Mi[0, 0] * l3 + Mi[0, 1] * m3 + Mi[0, 2] * s3)
            px[y, x, 1] = max(0.0, Mi[1, 0] * l3 + Mi[1, 1] * m3 + Mi[1, 2] * s3)
            px[y, x, 2] = max(0.0, Mi[2, 0] * l3 + Mi[2, 1] * m3 + Mi[2, 2] * s3)


@njit(parallel=True, nogil=True, cache=_C)
def _curves(px, tab, ENC):
    h, w = px.shape[0], px.shape[1]
    N = tab.shape[1] - 1
    for y in prange(h):
        for x in range(w):
            for k in range(3):
                v = px[y, x, k]
                if v >= 1: px[y, x, k] = tab[k, N] * v
                else: px[y, x, k] = tab[k, int(_enc(v, ENC) * N + 0.5)]


@njit(parallel=True, nogil=True, cache=_C)
def _usm(px, Y, Yb, amt, thr):
    h, w = px.shape[0], px.shape[1]
    for y in prange(h):
        for x in range(w):
            yy = Y[y, x]
            if yy <= 1e-6: continue
            e0 = min(yy, 1.0); e1 = min(Yb[y, x], 1.0)
            e = 12.92 * e0 if e0 <= 0.0031308 else 1.055 * e0 ** (1 / 2.4) - 0.055
            eb = 12.92 * e1 if e1 <= 0.0031308 else 1.055 * max(e1, 0.0) ** (1 / 2.4) - 0.055
            d0 = e - eb
            d = (1.0 if d0 > 0 else -1.0 if d0 < 0 else 0.0) * max(0.0, abs(d0) - thr)
            if d == 0: continue
            q = max(0.0, e + amt * d)
            lin = q / 12.92 if q <= 0.04045 else ((q + 0.055) / 1.055) ** 2.4
            kk = lin / e0
            px[y, x, 0] *= kk; px[y, x, 1] *= kk; px[y, x, 2] *= kk


@njit(parallel=True, nogil=True, cache=_C)
def _to_srgb8(px, M, ENC, out):
    h, w = px.shape[0], px.shape[1]
    for y in prange(h):
        for x in range(w):
            r = px[y, x, 0]; g = px[y, x, 1]; b = px[y, x, 2]
            for c in range(3):
                out[y, x, c] = np.uint8(round(_enc(M[c, 0] * r + M[c, 1] * g + M[c, 2] * b, ENC) * 255))   # rounds, as Uint8ClampedArray


def to_srgb8(px):
    out = np.empty(px.shape, np.uint8)
    _to_srgb8(px, R2020_TO_SRGB, ENC, out)
    return out


class Target:
    """An image the pipeline runs on, with its blur caches."""
    def __init__(self, data):
        self.data = np.ascontiguousarray(data, np.float32)
        self.h, self.w = self.data.shape[:2]
        self.cache = {}
        self.clones = []   # cloned spots on this image: (x0, y0, alpha, qx, qy), see Develop.clone_fields


# ───────────────────────────────────────────── cloned spots after Develop
# A clone copies the source's pixels into the picture before Develop, but adjustments that follow the local
# brightness (shadows, highlights, midtones, clarity, local contrast, texture, dehaze, sharpening) then treat
# the copy according to the target's surroundings, so it can come out darker or lighter than its source. So
# after developing, each cloned spot also copies the developed pixels from its source: the clone looks exactly
# as its source does. Positions are followed through the geometry (rotation, perspective, lens distortion).
@njit(nogil=True, cache=_C)
def _src_pt(ox, oy, x0c, y0c, sc, k, co, si, S, Hi, hasH, P, CW, CH, sw, sh, Q):
    """Output pixel (array coordinates, pixel centres) -> source pixel-centre position (full resolution)."""
    ok = _map(x0c + (ox + 0.5) / sc - CW / 2, y0c + (oy + 0.5) / sc - CH / 2, k, co, si, S, Hi, hasH, P, CW, CH, sw, sh, Q)
    return ok, Q[0] - 0.5, Q[1] - 0.5


@njit(nogil=True, cache=_C)
def _locate(tx, ty, ow, oh, stride, x0c, y0c, sc, k, co, si, S, Hi, hasH, P, CW, CH, sw, sh):
    """Where source position (tx, ty) lands in the output: a coarse search, then Newton steps.
    Returns (qx, qy, output pixels per source pixel), qx = nan when it cannot be found."""
    Q = np.empty(7)
    best = 1e30; bx = -1.0; by = -1.0
    for oy in range(-stride, oh + stride, stride):
        for ox in range(-stride, ow + stride, stride):
            ok, sx, sy = _src_pt(ox, oy, x0c, y0c, sc, k, co, si, S, Hi, hasH, P, CW, CH, sw, sh, Q)
            if ok:
                d = (sx - tx) ** 2 + (sy - ty) ** 2
                if d < best: best = d; bx = ox; by = oy
    if bx < -stride + 0.5 and by < -stride + 0.5 and best >= 1e30: return np.nan, np.nan, 1.0
    qx = float(bx); qy = float(by); det = 1.0
    for _ in range(12):
        ok, sx, sy = _src_pt(qx, qy, x0c, y0c, sc, k, co, si, S, Hi, hasH, P, CW, CH, sw, sh, Q)
        ok1, ax, ay = _src_pt(qx + 1, qy, x0c, y0c, sc, k, co, si, S, Hi, hasH, P, CW, CH, sw, sh, Q)
        ok2, cx, cy = _src_pt(qx, qy + 1, x0c, y0c, sc, k, co, si, S, Hi, hasH, P, CW, CH, sw, sh, Q)
        if not (ok and ok1 and ok2): break
        j00 = ax - sx; j10 = ay - sy; j01 = cx - sx; j11 = cy - sy
        det = j00 * j11 - j01 * j10
        if abs(det) < 1e-12: break
        ex = tx - sx; ey = ty - sy
        stx = (j11 * ex - j01 * ey) / det; sty = (-j10 * ex + j00 * ey) / det
        qx += stx; qy += sty
        if abs(stx) + abs(sty) < 1e-3: break
    return qx, qy, 1.0 / math.sqrt(max(abs(det), 1e-12))


@njit(parallel=True, nogil=True, cache=_C)
def _clone_field(bx0, by0, bw, bh, offx, offy, x0c, y0c, sc, k, co, si, S, Hi, hasH, P, CW, CH, sw, sh,
                 tx, ty, R, soft, op, dx, dy, alpha, qx, qy):
    for j in prange(bh):
        Q = np.empty(7)
        for i in range(bw):
            ox = bx0 + i + offx; oy = by0 + j + offy
            ok, sx, sy = _src_pt(ox, oy, x0c, y0c, sc, k, co, si, S, Hi, hasH, P, CW, CH, sw, sh, Q)
            if not ok: continue
            r = math.hypot(sx - tx, sy - ty)
            if r >= R: continue
            a = min(1.0, (R - r) / soft); a = a * a * (3 - 2 * a) * op
            if a <= 0: continue
            ok1, ax, ay = _src_pt(ox + 1, oy, x0c, y0c, sc, k, co, si, S, Hi, hasH, P, CW, CH, sw, sh, Q)
            ok2, cx, cy = _src_pt(ox, oy + 1, x0c, y0c, sc, k, co, si, S, Hi, hasH, P, CW, CH, sw, sh, Q)
            if not (ok1 and ok2): continue
            j00 = ax - sx; j10 = ay - sy; j01 = cx - sx; j11 = cy - sy
            det = j00 * j11 - j01 * j10
            if abs(det) < 1e-12: continue
            alpha[j, i] = a
            qx[j, i] = bx0 + i + (j11 * dx - j01 * dy) / det
            qy[j, i] = by0 + j + (-j10 * dx + j00 * dy) / det


@njit(parallel=True, nogil=True, cache=_C)
def _clone_apply(out, pre, bx0, by0, alpha, qx, qy):
    h, w = pre.shape[0], pre.shape[1]
    for j in prange(alpha.shape[0]):
        for i in range(alpha.shape[1]):
            a = alpha[j, i]
            if a <= 0: continue
            x = qx[j, i]; y = qy[j, i]
            if x < 0 or y < 0 or x > w - 1 or y > h - 1: continue
            xi = min(int(x), w - 2); yi = min(int(y), h - 2); fx = x - xi; fy = y - yi
            for c in range(3):
                v = (pre[yi, xi, c] * (1 - fx) + pre[yi, xi + 1, c] * fx) * (1 - fy) + (pre[yi + 1, xi, c] * (1 - fx) + pre[yi + 1, xi + 1, c] * fx) * fy
                out[by0 + j, bx0 + i, c] += (v - out[by0 + j, bx0 + i, c]) * a


def apply_clones(out, clones):
    if not clones: return out
    pre = out.copy()   # every clone copies from the picture as it was before any of them
    for bx0, by0, alpha, qx, qy in clones:
        _clone_apply(out, pre, bx0, by0, alpha, qx, qy)
    return out


@njit(parallel=True, nogil=True, cache=_C)
def _lum(d, KY, log, out):
    h, w = d.shape[0], d.shape[1]
    for y in prange(h):
        for x in range(w):
            v = KY[0] * d[y, x, 0] + KY[1] * d[y, x, 1] + KY[2] * d[y, x, 2]
            out[y, x] = math.log2(v + 1e-5) if log else v


def luminance(d, log=False):
    out = np.empty(d.shape[:2], np.float32)
    _lum(d, KY, log, out)
    return out


def _L0(t):
    if 'L0' not in t.cache:
        t.cache['L0'] = luminance(t.data, True)
    return t.cache['L0']


def _DK(t):
    if 'DK' not in t.cache:
        t.cache['DK'] = t.data.min(axis=2)
    return t.cache['DK']


def develop_core(t, p, scale, D, airlight, ext=None):
    c = t.cache
    sig = lambda f: f * D * scale
    g0 = lambda k: p.get(k) or 0
    want = dict(sh=sig(0.015) if (g0('shadows') or g0('highlights') or g0('midtones')) else 0,
                cl=sig(0.006) if g0('clarity') else 0, lc=sig(0.03) if g0('localContrast') else 0,
                mc=1.5 * scale if g0('micro') and 1.5 * scale >= 0.35 else 0, dh=sig(0.01) if g0('dehaze') else 0)

    def blurC(name, get, sigma):
        k = f'{name}:{sigma:.3f}'
        if k not in c:
            c[k] = ext(name, sigma) if (ext and sigma >= 6) else blur(get(t), sigma)
        return c[k]
    E0 = np.zeros((0, 0), np.float32)
    Bsh = blurC('L', _L0, want['sh']) if want['sh'] else E0
    Bcl = blurC('L', _L0, want['cl']) if want['cl'] else E0
    Blc = blurC('L', _L0, want['lc']) if want['lc'] else E0
    Bmc = blurC('L', _L0, want['mc']) if want['mc'] else E0
    Ldet = _L0(t) if (want['cl'] or want['lc'] or want['mc']) else E0
    Dk = blurC('D', _DK, want['dh']) if want['dh'] else E0
    gain = 2.0 ** g0('exposure'); lg = math.log2(gain)
    W = wb_matrix(g0('temp'), g0('tint'))
    M = (W * gain) if W is not None else np.eye(3) * gain
    bp = -g0('blacks') / 100 * 0.1; wp = 1 - g0('whites') / 100 * 0.25
    bg = 2.0 ** (-g0('brightness') / 100); ca = max(-0.95, min(0.95, g0('contrast') / 100 * 0.9))
    TN = 2048
    e = np.arange(TN + 1) / TN
    e = (e - bp) / (wp - bp)
    mid = (e > 0) & (e < 1)
    em = np.power(np.where(mid, e, 0.5), bg)
    em = em - ca * np.sin(2 * np.pi * em) / (2 * np.pi)
    e = np.where(mid, em, e)
    TONE = np.where(e <= 0, 0.0, decS(np.maximum(e, 0)))
    tone_on = bool(bp or wp != 1 or bg != 1 or ca)
    out = np.empty_like(t.data)
    _core(t.data, M, Dk, g0('dehaze') / 100, airlight * gain, gain, Bsh, lg, g0('shadows') / 100 * 1.6, g0('highlights') / 100 * 1.3,
          g0('midtones') / 100 * 1.0, tone_on, TONE, float(TONE[TN]), Ldet, Bmc, Bcl, Blc, g0('micro') / 100 * 1.2,
          g0('clarity') / 100 * 0.9, g0('localContrast') / 100 * 0.6, KY, ENC, out)
    if p.get('defringe'): defringe_pass(out, t, p['defringe'], scale, blurC)
    colour_pass(out, p)
    if p.get('curveLut'): curve_pass(out, p['curveLut'])
    if (p.get('usmAmount') or 0) > 0: unsharp(out, p, scale)
    return apply_clones(out, t.clones)


def defringe_pass(px, t, d, scale, blurC):
    s1 = max(0.4, scale); key = f'fe:{s1:.3f}'
    E = t.cache.get(key)
    if E is None:
        L = _L0(t); B = blurC('L', _L0, s1)
        E = t.cache[key] = blur(np.abs(L - B), max(0.6, 2 * scale))
    bands = []
    if d.get('purple'): bands.append([d['purple'] / 100, d.get('purpleHue', 315) or 315, 40])
    if d.get('green'): bands.append([d['green'] / 100, d.get('greenHue', 145) or 145, 35])
    if not bands: return
    _defringe(px, E, np.array(bands, np.float64), R2020_TO_LMS, LMS_TO_R2020, OK_M2, OK_M2I)


def colour_pass(px, p):
    hsl = p.get('hsl') or {}
    arr = lambda a: np.array([(a[i] if a and i < len(a) and a[i] else 0) for i in range(8)], np.float64)
    hue_a, sat_a, lum_a = arr(hsl.get('hue')), arr(hsl.get('sat')), arr(hsl.get('lum'))
    any_hsl = bool(hue_a.any() or sat_a.any() or lum_a.any())
    gr = p.get('grading') or {}
    any_gr = any(gr.get(k) and gr[k].get('sat') for k in ('shadows', 'midtones', 'highlights'))
    sat = 1 + (p.get('saturation') or 0) / 100; vib = (p.get('vibrance') or 0) / 100
    eq = p.get('eq')
    if not any_hsl and not any_gr and sat == 1 and not vib and not eq: return
    gOff = np.zeros((3, 2))
    for i, k in enumerate(('shadows', 'midtones', 'highlights')):
        v = gr.get(k) or {'hue': 0, 'sat': 0}
        a = (v.get('hue') or 0) * math.pi / 180; s = (v.get('sat') or 0) / 100 * 0.06
        gOff[i] = (s * math.cos(a), s * math.sin(a))
    E = np.zeros(0)
    eqa = lambda k: np.asarray(eq[k], np.float64) if eq and eq.get(k) is not None else E
    _colour(px, sat, vib, any_hsl, hue_a, sat_a, lum_a, bool(eq), eqa('hue'), eqa('sat'), eqa('bri'), bool(any_gr), gOff,
            (gr.get('balance') or 0) / 100, R2020_TO_LMS, LMS_TO_R2020, OK_M2, OK_M2I, BANDS)


def curve_pass(px, lut):
    N = lut['n'] - 1
    tabs = np.empty((3, N + 1))
    base = np.arange(N + 1) / N
    for k, ch in enumerate(('r', 'g', 'b')):
        e = base.copy()
        if lut.get('m') is not None: e = np.asarray(lut['m'], np.float64)
        if lut.get(ch) is not None: e = np.asarray(lut[ch], np.float64)[np.round(np.clip(e, 0, 1) * N).astype(np.int64)]
        tabs[k] = decS(np.clip(e, 0, 1))
    _curves(px, tabs, ENC)


def unsharp(px, p, scale):
    sigma = (p.get('usmRadius') or 1) * scale
    if sigma < 0.3: return
    Y = luminance(px)
    Yb = blur(Y, sigma)
    _usm(px, Y, Yb, p['usmAmount'] / 100, (p.get('usmThreshold') or 0) / 255)


LOCAL_KEYS = ['exposure', 'contrast', 'highlights', 'shadows', 'midtones', 'whites', 'blacks', 'brightness', 'temp', 'tint',
              'saturation', 'vibrance', 'clarity', 'dehaze', 'micro', 'localContrast']


def local_params(p, adj):
    q = dict(p)
    for k in LOCAL_KEYS:
        if adj.get(k): q[k] = (q.get(k) or 0) + adj[k]
    if adj.get('sharpness'): q['usmAmount'] = max(0, (q.get('usmAmount') or 0) + adj['sharpness'])
    return q


def develop(t, p, scale, D, airlight, ext=None, locals_=()):
    out = develop_core(t, p, scale, D, airlight, ext)
    for a, adj in locals_:
        if not a.any(): continue
        o2 = develop_core(t, local_params(p, adj), scale, D, airlight, ext)
        wgt = (a.astype(np.float32) / 255)[..., None]
        out += (o2 - out) * wgt
    return out


# ───────────────────────────────────────────── retouching (spots, AI fills), applied to the base images in place
def _sample3(b, x, y):
    """Bilinear, pixel-centre coordinates, clamped (vectorised)."""
    H, W = b.shape[:2]
    x = np.clip(x, 0, W - 1.001); y = np.clip(y, 0, H - 1.001)
    xi = x.astype(np.int64); yi = y.astype(np.int64); fx = (x - xi)[..., None]; fy = (y - yi)[..., None]
    return (b[yi, xi] * (1 - fx) + b[yi, xi + 1] * fx) * (1 - fy) + (b[yi + 1, xi] * (1 - fx) + b[yi + 1, xi + 1] * fx) * fy


def _save_patch(b, x0, y0, x1, y1, saved):
    H, W = b.shape[:2]
    x0 = max(0, int(math.floor(x0))); y0 = max(0, int(math.floor(y0))); x1 = min(W, int(math.ceil(x1))); y1 = min(H, int(math.ceil(y1)))
    if x1 <= x0 or y1 <= y0: return None
    q = dict(x0=x0, y0=y0, w=x1 - x0, h=y1 - y0, data=b[y0:y1, x0:x1].copy())
    saved.append(q)
    return q


def apply_spot(b, it, saved):
    """Clone copies the source disc; heal also carries the (log) colour difference measured round the edge."""
    H, W = b.shape[:2]
    R = max(1, it['r'] * math.hypot(W, H)); tx = it['x'] * W - 0.5; ty = it['y'] * H - 0.5
    dx = (it['sx'] - it['x']) * W; dy = (it['sy'] - it['y']) * H
    q = _save_patch(b, tx - R - 1, ty - R - 1, tx + R + 2, ty + R + 2, saved)
    if not q: return
    heal = it.get('mode') != 'clone'
    N = 48
    if heal:
        a = 2 * np.pi * np.arange(N) / N
        bx = tx + np.cos(a) * R * 1.02; by = ty + np.sin(a) * R * 1.02
        T = np.zeros((N, 3)); S = np.zeros((N, 3))
        for ox, oy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            T += _sample3(b, bx + ox, by + oy); S += _sample3(b, bx + ox + dx, by + oy + dy)
        dl = np.log((T + 5e-4) / (S + 5e-4))
    yy, xx = np.mgrid[q['y0']:q['y0'] + q['h'], q['x0']:q['x0'] + q['w']].astype(np.float64)
    r = np.hypot(xx - tx, yy - ty)
    soft = max(0.5, R * min(1, max(0.02, it.get('feather', 0.5) if it.get('feather') is not None else 0.5)))
    op = it.get('opacity', 1) if it.get('opacity') is not None else 1
    a = np.where(r >= R, 0.0, np.minimum(1, (R - r) / soft))
    a = a * a * (3 - 2 * a) * op
    A = _sample3(b, xx + dx, yy + dy)
    if heal:
        wgt = 1 / ((xx[..., None] - bx) ** 2 + (yy[..., None] - by) ** 2 + 0.25)   # h x w x N
        off = (wgt @ dl) / wgt.sum(-1, keepdims=True)
        A = np.maximum(0, (A + 5e-4) * np.exp(off) - 5e-4)
    cur = b[q['y0']:q['y0'] + q['h'], q['x0']:q['x0'] + q['w']]
    a3 = a[..., None]
    b[q['y0']:q['y0'] + q['h'], q['x0']:q['x0'] + q['w']] = np.where(a3 > 0, cur + (A - cur) * a3, cur).astype(np.float32)


def apply_fill(b, f, saved):
    """An AI fill patch: rect (normalised source), w, h, data (linear, h x w x 3), alpha (h x w)."""
    H, W = b.shape[:2]
    x0, y0 = f['rect']['x'] * W, f['rect']['y'] * H
    x1, y1 = (f['rect']['x'] + f['rect']['w']) * W, (f['rect']['y'] + f['rect']['h']) * H
    q = _save_patch(b, x0, y0, x1, y1, saved)
    if not q: return
    kx, ky = f['w'] / (x1 - x0), f['h'] / (y1 - y0)
    yy, xx = np.mgrid[0:q['h'], 0:q['w']].astype(np.float64)
    fx = np.clip((xx + q['x0'] + 0.5 - x0) * kx - 0.5, 0, f['w'] - 1.001); fy = np.clip((yy + q['y0'] + 0.5 - y0) * ky - 0.5, 0, f['h'] - 1.001)
    al = f['alpha'][..., None].astype(np.float64)
    a = _sample3(al, fx, fy)[..., 0]
    v = _sample3(f['data'].astype(np.float64), fx, fy)
    cur = b[q['y0']:q['y0'] + q['h'], q['x0']:q['x0'] + q['w']]
    a3 = np.where(a > 0.002, a, 0)[..., None]
    b[q['y0']:q['y0'] + q['h'], q['x0']:q['x0'] + q['w']] = (cur + (v - cur) * a3).astype(np.float32)


def spot_source(b, x, y, r):
    """A source for a new spot: the nearby disc whose surroundings match the target's best."""
    H, W = b.shape[:2]
    R = max(2, r * math.hypot(W, H)); tx = x * W; ty = y * H

    def ring(cx, cy):
        a = 2 * np.pi * np.arange(24) / 24
        pts = [(cx + np.cos(a) * R * f, cy + np.sin(a) * R * f) for f in (1.1, 1.4)]
        return np.stack([_sample3(b, px, py) for px, py in pts], 1).reshape(-1, 3)   # matches the web's order loosely
    tr = ring(tx, ty)
    best, be = None, float('inf')
    for dist in (2.2, 3, 4):
        for k in range(16):
            a = 2 * math.pi * k / 16
            cx, cy = tx + math.cos(a) * R * dist, ty + math.sin(a) * R * dist
            if cx < R or cy < R or cx > W - R or cy > H - R: continue
            sr = ring(cx, cy)
            e = float(np.sum(np.log((sr + 1e-3) / (tr + 1e-3)) ** 2))
            Bc = _sample3(b, np.array([cx]), np.array([cy]))[0]
            e += float(np.sum(np.log((Bc + 1e-3) / (tr[0] + 1e-3)) ** 2)) * 4 + dist * 0.02
            if e < be: be, best = e, (cx / W, cy / H)
    return best or (min(0.98, x + 3 * R / W), y)


# ───────────────────────────────────────────── the engine (the web's develop worker)
class Develop:
    PREVIEW_MAX = 2560

    def __init__(self, full):
        """full: linear Rec.2020 float32 H x W x 3 (display orientation)."""
        self.orig_full = np.ascontiguousarray(full, np.float32)
        self.full = self.orig_full.copy()
        H, W = full.shape[:2]
        self.W, self.H = W, H
        f = max(1, math.ceil(max(W, H) / self.PREVIEW_MAX))
        w, h = W // f, H // f
        self.orig_prev = np.ascontiguousarray(full[:h * f, :w * f].reshape(h, f, w, f, 3).mean(axis=(1, 3)), np.float32)
        self.prev = self.orig_prev.copy()
        dk = self.prev.min(axis=2)
        bl = blur(dk, max(1, 0.01 * math.hypot(w, h))).ravel()
        s = np.sort(bl[::7])
        self.airlight = max(0.05, min(1, float(s[int(len(s) * 0.998)]) if len(s) else 1))
        self.D = math.hypot(W, H)
        self.fills = {}           # id -> fill patch (normalised source rect)
        self.op_layers = {}       # id -> dict(gkey, patches)
        self.mask_rasters = {}    # key -> uint8 raster
        self._retouch_key = '[]'
        self._geo = {}            # cache of geometry-applied bases
        self._low = None

    # retouching
    def sync_retouch(self, lst):
        import json
        key = json.dumps(lst or [], sort_keys=True)
        if key == self._retouch_key: return
        self.full = self.orig_full.copy(); self.prev = self.orig_prev.copy()
        for b in (self.prev, self.full):
            saved = []
            for it in lst or []:
                if it['kind'] == 'spot': apply_spot(b, it, saved)
                elif it['kind'] == 'fill' and it['id'] in self.fills: apply_fill(b, self.fills[it['id']], saved)
        self._retouch_key = key
        self._geo.clear(); self._low = None
        # cloned spots in full-resolution source pixels: target centre, radius, softness, opacity, offset to the source
        W, H = self.W, self.H
        self._clones = []
        for it in lst or []:
            if it['kind'] == 'spot' and it.get('mode') == 'clone':
                Rr = max(1, it['r'] * math.hypot(W, H))
                fe = it.get('feather', 0.5) if it.get('feather') is not None else 0.5
                self._clones.append((it['x'] * W - 0.5, it['y'] * H - 0.5, Rr, max(0.5, Rr * min(1, max(0.02, fe))),
                                     it.get('opacity', 1) if it.get('opacity') is not None else 1, (it['sx'] - it['x']) * W, (it['sy'] - it['y']) * H))

    def clone_fields(self, g, lens, crop, sc, w, h, offx=0, offy=0):
        """For each cloned spot, where its pixels lie in an output image (w x h, offset (offx, offy) within the
        whole output at sc output pixels per full-resolution pixel) and where their source pixels lie."""
        out = []
        if not getattr(self, '_clones', None): return out
        ln = dict(lens, vig=None) if lens else None   # positions only
        CW, CH, k, co, si, S, Hi, hasH, P = _geo_args(g, ln, self.W, self.H, 1.0)
        x0c, y0c = crop['x'] * CW, crop['y'] * CH
        TW, TH = max(1, jround(crop['w'] * CW * sc)), max(1, jround(crop['h'] * CH * sc))
        for tx, ty, Rr, soft, op, dx, dy in self._clones:
            qx, qy, k_out = _locate(tx, ty, TW, TH, max(4, int(min(TW, TH) / 200)), x0c, y0c, sc, k, co, si, S, Hi, hasH, P, CW, CH, self.W, self.H)
            if not np.isfinite(qx): continue
            rad = Rr * k_out * 1.25 + 3
            bx0, by0 = max(0, int(math.floor(qx - offx - rad))), max(0, int(math.floor(qy - offy - rad)))
            bx1, by1 = min(w, int(math.ceil(qx - offx + rad)) + 1), min(h, int(math.ceil(qy - offy + rad)) + 1)
            if bx1 <= bx0 or by1 <= by0: continue
            bw, bh = bx1 - bx0, by1 - by0
            alpha = np.zeros((bh, bw), np.float32); fx = np.zeros((bh, bw), np.float64); fy = np.zeros((bh, bw), np.float64)
            _clone_field(bx0, by0, bw, bh, offx, offy, x0c, y0c, sc, k, co, si, S, Hi, hasH, P, CW, CH, self.W, self.H,
                         tx, ty, Rr, soft, op, dx, dy, alpha, fx, fy)
            if alpha.any(): out.append((bx0, by0, alpha, fx, fy))
        return out

    @staticmethod
    def geom_key(g, crop, lens):
        import json
        return json.dumps([g['rot90'], g.get('angle'), crop, g.get('Hinv'), lens], sort_keys=True, default=float)

    @staticmethod
    def canvas_key(g, lens):
        import json
        return json.dumps([g['rot90'], g.get('angle'), g.get('Hinv'), lens], sort_keys=True, default=float)

    def out_layer(self, p):
        o = p.get('outpaint')
        if not o: return None
        L = self.op_layers.get(o['id'])
        return L if L and L['gkey'] == self.canvas_key(p['geometry'], p.get('lens')) else None

    def fill_empty(self, t, valid, crop, L, TW, TH, rx0, ry0):
        if valid is None: return
        ys, xs = np.nonzero(valid == 0)
        if not len(ys): return
        v = crop['y'] + crop['h'] * (ry0 + ys + 0.5) / TH
        u = crop['x'] + crop['w'] * (rx0 + xs + 0.5) / TW
        done = np.zeros(len(ys), bool)
        for q in L['patches']:
            sel = ~done & (u >= q['u0']) & (u <= q['u1']) & (v >= q['v0']) & (v <= q['v1'])
            if not sel.any(): continue
            x = np.clip((u[sel] - q['u0']) / (q['u1'] - q['u0']) * q['w'] - 0.5, 0, q['w'] - 1.001)
            y = np.clip((v[sel] - q['v0']) / (q['v1'] - q['v0']) * q['h'] - 0.5, 0, q['h'] - 1.001)
            t[ys[sel], xs[sel]] = _sample3(q['data'], x, y)
            done |= sel

    def masks_for(self, t, p, crop, TW, TH, rx0, ry0):
        out = []
        for l in p.get('locals') or []:
            r = self.mask_rasters.get(l['key'])
            if r is None: continue
            k = 'ma:' + l['key']
            if k not in t.cache:
                t.cache[k] = mask_alpha(r, p['geometry'], p.get('lens'), crop, TW, TH, rx0, ry0, t.w, t.h, self.W)
            out.append((t.cache[k], l['adj']))
        return out

    def preview_target(self, p, max_side, uncropped=False):
        g = p['geometry']; crop = {'x': 0, 'y': 0, 'w': 1, 'h': 1} if uncropped else g['crop']
        self.sync_retouch(p.get('retouch'))
        L = self.out_layer(p)
        key = ('prev', self.geom_key(g, crop, p.get('lens')), max_side, p['outpaint']['id'] if L else '')
        if key not in self._geo:
            A = apply_geometry(self.prev, g, crop, max_side, p.get('lens'), self.W, want_valid=bool(L))
            data = A['data'] if A['data'] is not self.prev else self.prev
            if L:
                data = data.copy(); self.fill_empty(data, A['valid'], crop, L, A['w'], A['h'], 0, 0)
            t = Target(data)
            CW, _ = canvas_dims(self.W, self.H, g)
            t.scale = t.w / (crop['w'] * CW)
            t.clones = self.clone_fields(g, p.get('lens'), crop, t.scale, t.w, t.h)
            for k in [k for k in self._geo if k[0] == 'prev']: del self._geo[k]
            self._geo[key] = t
        return self._geo[key]

    def full_geometry(self, p):
        g = p['geometry']
        self.sync_retouch(p.get('retouch'))
        L = self.out_layer(p)
        key = ('full', self.geom_key(g, g['crop'], p.get('lens')), p['outpaint']['id'] if L else '')
        if key not in self._geo:
            A = apply_geometry(self.full, g, g['crop'], 0, p.get('lens'), self.W, want_valid=bool(L))
            data = A['data']
            if L:
                data = data.copy(); self.fill_empty(data, A['valid'], g['crop'], L, A['w'], A['h'], 0, 0)
            for k in [k for k in self._geo if k[0] == 'full']: del self._geo[k]
            self._geo[key] = data
            self._low = None
        return self._geo[key], key

    def render_preview(self, p, max_side, uncropped=False, neutral=False):
        t = self.preview_target(p, max_side, uncropped)
        crop = {'x': 0, 'y': 0, 'w': 1, 'h': 1} if uncropped else p['geometry']['crop']
        if neutral: return t.data
        return develop(t, p, t.scale, self.D, self.airlight, None, self.masks_for(t, p, crop, t.w, t.h, 0, 0))

    def render_full(self, p):
        F, _ = self.full_geometry(p)
        t = Target(F)
        t.clones = self.clone_fields(p['geometry'], p.get('lens'), p['geometry']['crop'], 1.0, t.w, t.h)
        return develop(t, p, 1, self.D, self.airlight, None, self.masks_for(t, p, p['geometry']['crop'], t.w, t.h, 0, 0))

    def _low_maps(self, F, key):
        if self._low and self._low['key'] == key: return self._low
        H, W = F.shape[:2]
        f = max(1, jround(max(W, H) / 2000)); w, h = -(-W // f), -(-H // f)
        Lg = np.log2(KY[0] * F[..., 0] + KY[1] * F[..., 1] + KY[2] * F[..., 2] + 1e-5)
        K = F.min(axis=2)
        pad = lambda a: np.pad(a, ((0, h * f - H), (0, w * f - W)), constant_values=np.nan)
        Lm = np.nanmean(pad(Lg).reshape(h, f, w, f), axis=(1, 3)).astype(np.float32)
        Km = np.nanmean(pad(K).reshape(h, f, w, f), axis=(1, 3)).astype(np.float32)
        self._low = dict(key=key, f=f, w=w, h=h, L=Lm, K=Km, cache={})
        return self._low

    def render_region(self, p, region, neutral=False):
        """A full-resolution region (x, y, w, h of the geometry-applied image), with blurs that reach further
        than its margin sampled from a low-resolution copy of the whole picture."""
        F, fk = self.full_geometry(p)
        Hh, Ww = F.shape[:2]
        rx, ry, rw, rh = region
        mg = int(math.ceil(3 * max(p.get('usmRadius') or 1, 3)))
        x0, y0, x1, y1 = max(0, rx - mg), max(0, ry - mg), min(Ww, rx + rw + mg), min(Hh, ry + rh + mg)
        g = p['geometry']
        cl = self.clone_fields(g, p.get('lens'), g['crop'], 1.0, x1 - x0, y1 - y0, x0, y0)
        if cl:   # clone sources outside the region: develop a larger area so that they can be copied
            ax0 = min([x0] + [x0 + int(math.floor(np.nanmin(np.where(a > 0, qx, np.nan)))) - mg for _, _, a, qx, qy in cl])
            ay0 = min([y0] + [y0 + int(math.floor(np.nanmin(np.where(a > 0, qy, np.nan)))) - mg for _, _, a, qx, qy in cl])
            ax1 = max([x1] + [x0 + int(math.ceil(np.nanmax(np.where(a > 0, qx, np.nan)))) + mg + 2 for _, _, a, qx, qy in cl])
            ay1 = max([y1] + [y0 + int(math.ceil(np.nanmax(np.where(a > 0, qy, np.nan)))) + mg + 2 for _, _, a, qx, qy in cl])
            ax0, ay0, ax1, ay1 = max(0, ax0), max(0, ay0), min(Ww, ax1), min(Hh, ay1)
            if (ax0, ay0, ax1, ay1) != (x0, y0, x1, y1):
                x0, y0, x1, y1 = ax0, ay0, ax1, ay1
                cl = self.clone_fields(g, p.get('lens'), g['crop'], 1.0, x1 - x0, y1 - y0, x0, y0)
        t = Target(F[y0:y1, x0:x1])
        t.clones = cl
        if neutral: px = t.data
        else:
            low = self._low_maps(F, fk)

            def ext(name, sigma):
                ks = f'{name}:{sigma:.3f}'
                if ks not in low['cache']: low['cache'][ks] = blur(low['L'] if name == 'L' else low['K'], sigma / low['f'])
                B = low['cache'][ks]; f = low['f']
                fy = np.clip((y0 + np.arange(t.h) + 0.5) / f - 0.5, 0, low['h'] - 1.001)
                fx = np.clip((x0 + np.arange(t.w) + 0.5) / f - 0.5, 0, low['w'] - 1.001)
                yi, xi = fy.astype(np.int64), fx.astype(np.int64)
                wy, wx = (fy - yi)[:, None], (fx - xi)[None, :]
                return ((B[yi][:, xi] * (1 - wx) + B[yi][:, xi + 1] * wx) * (1 - wy) + (B[yi + 1][:, xi] * (1 - wx) + B[yi + 1][:, xi + 1] * wx) * wy).astype(np.float32)
            px = develop(t, p, 1, self.D, self.airlight, ext, self.masks_for(t, p, p['geometry']['crop'], Ww, Hh, x0, y0))
        rw, rh = min(rw, Ww - rx), min(rh, Hh - ry)
        return px[ry - y0:ry - y0 + rh, rx - x0:rx - x0 + rw]

    # ----- analysis
    def analysis_base(self, p):
        g = p['geometry']
        self.sync_retouch(p.get('retouch'))
        A = apply_geometry(self.prev, g, g['crop'], 700, p.get('lens'), self.W)
        t = Target(A['data'])
        CW, _ = canvas_dims(self.W, self.H, g)
        t.scale = t.w / (g['crop']['w'] * CW)
        return t

    def auto_tone(self, params):
        """Like Lightroom's Auto: each tone slider is searched (bisection, on a small copy) until the developed
        result meets a target."""
        t = self.analysis_base(params)
        n = t.w * t.h
        p = dict(params, exposure=0, contrast=0, highlights=0, shadows=0, midtones=0, whites=0, blacks=0, brightness=0, vibrance=0, usmAmount=0, micro=0)
        p.pop('curveLut', None)
        st = {}

        def measure():
            px = develop(t, p, t.scale, self.D, self.airlight)
            f = px.reshape(-1, 3).astype(np.float64)
            Y = f @ KY
            encv = lambda v: ENC[(np.clip(v, 0, 1) * 4095 + 0.5).astype(np.int64)]
            st['Y'] = np.bincount(np.minimum(1023, (encv(Y) * 1023).astype(np.int64)), minlength=1024)
            st['M'] = np.bincount(np.minimum(1023, (encv(f.max(1)) * 1023).astype(np.int64)), minlength=1024)
            sub = f[::16]
            lms = np.cbrt(np.maximum(0, sub @ R2020_TO_LMS.T)) @ OK_M2.T
            st['chroma'] = float(np.mean(np.hypot(lms[:, 1], lms[:, 2]))) if len(sub) else 0

        def pct(bins, q):
            c = np.cumsum(bins); i = int(np.searchsorted(c, q * n))
            return min(i, 1023) / 1023 if i < 1024 else 1

        def solve(key, lo, hi, f, target, iters=11):
            for _ in range(iters):
                p[key] = (lo + hi) / 2; measure()
                if f() < target: lo = p[key]
                else: hi = p[key]
            p[key] = (lo + hi) / 2
            return p[key]
        measure()
        med0, hi0 = pct(st['Y'], 0.5), pct(st['M'], 0.99)
        target = 0.34 if med0 < 0.08 and hi0 > 0.6 else 0.44
        solve('exposure', -3, 3.5, lambda: pct(st['Y'], 0.5), target)
        measure()
        p97, p10 = pct(st['M'], 0.97), pct(st['Y'], 0.10)
        p['highlights'] = -jround(max(0, min(80, (p97 - 0.78) * 320)))
        p['shadows'] = jround(max(0, min(65, (0.22 - p10) * 280)))
        solve('exposure', p['exposure'] - 1.5, p['exposure'] + 1.5, lambda: pct(st['Y'], 0.5), target)
        measure()
        iqr = pct(st['Y'], 0.75) - pct(st['Y'], 0.25)
        p['contrast'] = jround(max(-20, min(45, (0.30 - iqr) * 220)))
        solve('whites', -70, 70, lambda: pct(st['M'], 0.998), 0.985)
        solve('blacks', -60, 40, lambda: pct(st['Y'], 0.002), 0.012)
        solve('exposure', p['exposure'] - 1, p['exposure'] + 1, lambda: pct(st['Y'], 0.5), target, 9)
        measure()
        p['vibrance'] = jround(max(0, min(40, (0.07 - st['chroma']) / 0.07 * 60)))
        return dict(exposure=jround(p['exposure'] * 100) / 100, contrast=p['contrast'], highlights=p['highlights'], shadows=p['shadows'],
                    whites=jround(p['whites']), blacks=jround(p['blacks']), vibrance=p['vibrance'])

    @staticmethod
    def solve_wb(avg):
        def err(T, tn):
            W = wb_matrix(T, tn)
            v = W @ avg if W is not None else np.asarray(avg)
            if min(v) <= 0: return float('inf')
            lab = OK_M2 @ np.cbrt(np.maximum(0, R2020_TO_LMS @ v))
            return math.hypot(lab[1], lab[2]) / max(1e-6, lab[0])
        best, be = [0.0, 0.0], err(0, 0)

        def search(c0, c1, span, step):
            nonlocal best, be
            for T in np.arange(c0 - span, c0 + span + 1e-9, step):
                for tn in np.arange(c1 - span, c1 + span + 1e-9, step):
                    if abs(T) > 200 or abs(tn) > 150: continue
                    e = err(T, tn)
                    if e < be: be, best = e, [float(T), float(tn)]
        search(0, 0, 200, 8); search(best[0], best[1], 8, 1); search(best[0], best[1], 1, 0.1)
        return best

    def auto_wb(self, params):
        t = self.analysis_base(params)
        d = t.data.reshape(-1, 3).astype(np.float64)
        Y = d @ KY
        srt = np.sort(Y); n = len(Y)
        q = lambda f: srt[min(n - 1, int(f * n))]
        lo, hi, mid0, mid1 = q(0.90), q(0.995), q(0.2), q(0.95)
        ok = (Y > 1e-6) & (d.max(1) <= 0.98)
        wpm = ok & (Y >= lo) & (Y <= hi); gwm = ok & (Y >= mid0) & (Y <= mid1)
        if not wpm.any() or not gwm.any(): return None
        wp = (d[wpm] / Y[wpm, None]).mean(0)
        gw = np.exp(np.log(np.maximum(1e-6, d[gwm] / Y[gwm, None])).mean(0))
        best = self.solve_wb(np.sqrt(wp * gw))
        return jround(best[0] * 0.8), jround(best[1] * 0.8)

    def pick_wb(self, params, u, v):
        g = params['geometry']; c = g['crop']
        CW, CH = canvas_dims(self.prev.shape[1], self.prev.shape[0], g)
        d = 4.5
        box = dict(x=c['x'] + u * c['w'] - d / CW, y=c['y'] + v * c['h'] - d / CH, w=2 * d / CW, h=2 * d / CH)
        self.sync_retouch(params.get('retouch'))
        A = apply_geometry(self.prev, g, box, 0, params.get('lens'), self.W)
        px = A['data'].reshape(-1, 3).astype(np.float64)
        px = px[px.sum(1) > 0]
        if not len(px) or px.sum() <= 0: return None
        best = self.solve_wb(px.sum(0))
        return jround(best[0] * 10) / 10, jround(best[1] * 10) / 10

    def estimate_ca(self, retouch):
        """Lateral CA: radial shifts of red and blue against green at edges across the picture (NCC,
        sub-pixel), fitted with shift(rho) = a1 rho + a3 rho^3, robustly."""
        self.sync_retouch(retouch)
        b = self.full; H, W = b.shape[:2]; cx, cy = W / 2, H / 2; Hd = math.hypot(W, H) / 2
        T, SMAX, STEP = 40, 5, 0.25
        from scipy.ndimage import map_coordinates
        samples = []
        nx, ny = max(8, jround(W / 120)), max(6, jround(H / 120))
        jj, ii = np.mgrid[0:T, 0:T].astype(np.float64)
        shifts = np.arange(-SMAX, SMAX + 1e-9, STEP)

        def at(x, y, c):   # bilinear, pixel-edge coordinates
            return map_coordinates(b[..., c], [np.clip(y - 0.5, 0, H - 1), np.clip(x - 0.5, 0, W - 1)], order=1, mode='nearest')
        for ty in range(ny):
            for tx in range(nx):
                px, py = (tx + 0.5) * W / nx, (ty + 0.5) * H / ny
                dx, dy = px - cx, py - cy; dist = math.hypot(dx, dy); rho = dist / Hd
                if rho < 0.25 or px - T / 2 - SMAX - 2 < 0 or py - T / 2 - SMAX - 2 < 0 or px + T / 2 + SMAX + 2 > W or py + T / 2 + SMAX + 2 > H: continue
                ux, uy = dx / dist, dy / dist; x0, y0 = px - T / 2, py - T / 2
                g = np.log(at(x0 + ii + 0.5, y0 + jj + 0.5, 1) + 1e-3)
                gx = g[1:-1, 2:] - g[1:-1, :-2]; gy = g[2:, 1:-1] - g[:-2, 1:-1]
                energy = float(np.mean(np.abs(gx * ux + gy * uy)))
                if energy < 0.04: continue
                g = g - g.mean(); gs = float((g * g).sum())

                def shift(c):
                    sc = []
                    for s in shifts:
                        q = np.log(at(x0 + ii + 0.5 + s * ux, y0 + jj + 0.5 + s * uy, c) + 1e-3); q = q - q.mean()
                        sc.append(float((g * q).sum() / math.sqrt(gs * float((q * q).sum()) + 1e-12)))
                    bs = int(np.argmax(sc))
                    if sc[bs] < 0.8 or bs == 0 or bs == len(sc) - 1: return None
                    a, c0, e = sc[bs - 1], sc[bs], sc[bs + 1]; den = a - 2 * c0 + e
                    return -SMAX + (bs + (0.5 * (a - e) / den if den < 0 else 0)) * STEP
                sR = shift(0)
                if sR is None: continue
                sB = shift(2)
                if sB is None: continue
                samples.append((rho, sR, sB, energy, tx, ty))
        if len(samples) < 8: return dict(ok=False, n=len(samples))
        S = np.array(samples)
        self._ca_samples = samples

        def fit(col):
            use = np.ones(len(S), bool); a1 = a3 = 0.0
            for _ in range(3):
                r, y, w = S[use, 0], S[use, col], S[use, 3]
                r3 = r ** 3
                s11, s13, s33 = (w * r * r).sum(), (w * r * r3).sum(), (w * r3 * r3).sum()
                y1, y3 = (w * r * y).sum(), (w * r3 * y).sum()
                det = s11 * s33 - s13 * s13
                if abs(det) < 1e-12: break
                a1, a3 = (y1 * s33 - y3 * s13) / det, (s11 * y3 - s13 * y1) / det
                res = np.abs(a1 * S[:, 0] + a3 * S[:, 0] ** 3 - S[:, col])
                med = float(np.median(res[use])) if use.any() else 0
                nu = use & (res <= max(0.15, 3 * 1.4826 * med))
                if nu.sum() < 6: break
                use = nu
            return a1, a3
        R, B = fit(1), fit(2)
        return dict(ok=True, n=len(samples), ca=dict(vr=1 + R[0] / Hd, br=R[1] / Hd, vb=1 + B[0] / Hd, bb=B[1] / Hd),
                    cornerR=R[0] + R[1], cornerB=B[0] + B[1])

    def source_region(self, retouch, rect, max_side):
        """A region of the retouched full-resolution picture (normalised source rect), box-downscaled."""
        self.sync_retouch(retouch)
        b = self.full; H, W = b.shape[:2]
        x0, y0 = max(0, int(math.floor(rect['x'] * W))), max(0, int(math.floor(rect['y'] * H)))
        x1, y1 = min(W, int(math.ceil((rect['x'] + rect['w']) * W))), min(H, int(math.ceil((rect['y'] + rect['h']) * H)))
        f = max(1, math.ceil(max(x1 - x0, y1 - y0) / max_side)); w, h = (x1 - x0) // f, (y1 - y0) // f
        d = b[y0:y0 + h * f, x0:x0 + w * f].reshape(h, f, w, f, 3).mean(axis=(1, 3)).astype(np.float32)
        return dict(rect=dict(x=x0 / W, y=y0 / H, w=w * f / W, h=h * f / H), w=w, h=h, data=d)

    def outpaint_windows(self, p):
        """Windows round the empty areas of the whole canvas, with the picture's pixels at up to 1024 px."""
        g = p['geometry']; allc = dict(x=0, y=0, w=1, h=1)
        self.sync_retouch(p.get('retouch'))
        A = apply_geometry(self.prev, g, allc, 0, p.get('lens'), self.W, want_valid=True)
        wins = []
        if A['valid'] is not None:
            from scipy.ndimage import label, find_objects
            Bk = 8
            gh, gw = -(-A['h'] // Bk), -(-A['w'] // Bk)
            occ = np.zeros((gh, gw), bool)
            ys, xs = np.nonzero(A['valid'] == 0)
            occ[ys // Bk, xs // Bk] = True
            lab, n = label(occ)
            rects = []
            for sl in find_objects(lab):
                a, c = sl[1].start / gw, sl[1].stop / gw; b2, d = sl[0].start / gh, sl[0].stop / gh
                m2 = max(0.08, 0.6 * max(c - a, d - b2))
                rects.append([max(0, a - m2), max(0, b2 - m2), min(1, c + m2), min(1, d + m2)])
            merged = True
            while merged:
                merged = False
                for i in range(len(rects)):
                    for j in range(i + 1, len(rects)):
                        r, q = rects[i], rects[j]
                        if r[0] < q[2] and q[0] < r[2] and r[1] < q[3] and q[1] < r[3]:
                            rects[i] = [min(r[0], q[0]), min(r[1], q[1]), max(r[2], q[2]), max(r[3], q[3])]; del rects[j]; merged = True; break
                    if merged: break
            for u0, v0, u1, v1 in rects:
                Wn = apply_geometry(self.full, g, dict(x=u0, y=v0, w=u1 - u0, h=v1 - v0), 1024, p.get('lens'), self.W, want_valid=True)
                wins.append(dict(u0=u0, v0=v0, u1=u1, v1=v1, w=Wn['w'], h=Wn['h'], data=Wn['data'], valid=Wn['valid']))
        return self.canvas_key(g, p.get('lens')), wins


# ───────────────────────────────────────────── the page side: curves, lens, perspective, crop, masks
def monotone_curve(pts):
    """Fritsch-Carlson monotone cubic through sorted points; returns f(x array)."""
    xs = np.array([q[0] for q in pts], float); ys = np.array([q[1] for q in pts], float); n = len(xs)
    d = np.diff(ys) / np.maximum(1e-9, np.diff(xs))
    m = np.empty(n); m[0] = d[0]; m[-1] = d[-1]
    for i in range(1, n - 1): m[i] = 0 if d[i - 1] * d[i] <= 0 else (d[i - 1] + d[i]) / 2
    for i in range(n - 1):
        if d[i] == 0: m[i] = m[i + 1] = 0; continue
        a, b = m[i] / d[i], m[i + 1] / d[i]; s = a * a + b * b
        if s > 9: t = 3 / math.sqrt(s); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i]

    def f(x):
        x = np.asarray(x, float)
        i = np.clip(np.searchsorted(xs, x, side='left') - 1, 0, n - 2)
        h = xs[i + 1] - xs[i]; t = (x - xs[i]) / h; t2, t3 = t * t, t * t * t
        y = (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * m[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * m[i + 1]
        return np.where(x <= xs[0], ys[0], np.where(x >= xs[-1], ys[-1], y))
    return f


BAND_HUES = [29, 55, 110, 142, 195, 264, 300, 328]
BAND_NAMES = ['Red', 'Orange', 'Yellow', 'Green', 'Aqua', 'Blue', 'Purple', 'Magenta']


def periodic_curve(vals):
    n = len(BAND_HUES)
    w = lambda k: k % n
    X = lambda k: BAND_HUES[w(k)] + 360 * (k // n)
    V = lambda k: vals[w(k)]
    slope = lambda k: ((V(k + 1) - V(k)) / (X(k + 1) - X(k)) + (V(k) - V(k - 1)) / (X(k) - X(k - 1))) / 2

    def f(h):
        h = ((h % 360) + 360) % 360
        k = -1
        while k < n - 1 and h >= X(k + 1): k += 1
        x0 = X(k); dx = X(k + 1) - x0; t = (h - x0) / dx; t2, t3 = t * t, t * t * t
        return (2 * t3 - 3 * t2 + 1) * V(k) + (t3 - 2 * t2 + t) * dx * slope(k) + (-2 * t3 + 3 * t2) * V(k + 1) + (t3 - t2) * dx * slope(k + 1)
    return f


def is_identity(c):
    return len(c) == 2 and c[0][0] == 0 and c[0][1] == 0 and c[1][0] == 1 and c[1][1] == 1


def curve_lut(curves):
    N = 1024
    tab = lambda c: None if is_identity(c) else np.clip(monotone_curve(c)(np.arange(N) / (N - 1)), 0, 1)
    lut = dict(n=N, m=tab(curves['m']), r=tab(curves['r']), g=tab(curves['g']), b=tab(curves['b']))
    return lut if any(lut[k] is not None for k in 'mrgb') else None


def eq_tables(eq):
    def tab(vals, mp):
        if not any(vals): return None
        f = periodic_curve(vals)
        return np.array([mp(f(h)) for h in range(360)], np.float64)
    e = dict(hue=tab(eq['hue'], lambda v: v * 45), sat=tab(eq['sat'], lambda v: max(0, 1 + v)), bri=tab(eq['bri'], lambda v: v * 0.5))
    return e if any(v is not None for v in e.values()) else None


# ---- perspective: guided upright and keystone
def _cross(a, b):
    return np.cross(np.asarray(a, float), np.asarray(b, float))


def _vanishing(lines):
    if len(lines) == 2: return _cross(lines[0], lines[1])
    M = np.zeros((3, 3))
    for l in lines:
        u = np.asarray(l, float) / (math.hypot(l[0], l[1]) or 1); M += np.outer(u, u)
    tr = np.trace(M); x = np.array([0.3, 0.2, 1.0])
    for _ in range(200):
        y = tr * x - M @ x; x = y / (np.linalg.norm(y) or 1)
    return x


def upright_from_guides(guides, CW, CH):
    S = max(CW, CH) / 2
    fam = {'v': [], 'h': []}
    for x1, y1, x2, y2 in guides:
        a = [(x1 - 0.5) * CW / S, (y1 - 0.5) * CH / S, 1]; b = [(x2 - 0.5) * CW / S, (y2 - 0.5) * CH / S, 1]
        vert = abs((y2 - y1) * CH) > abs((x2 - x1) * CW)
        fam['v' if vert else 'h'].append(dict(l=_cross(a, b), d=np.array([b[0] - a[0], b[1] - a[1], 0])))
    vp = lambda f: None if not f else f[0]['d'] if len(f) == 1 else _vanishing([x['l'] for x in f])
    finite = lambda v: abs(v[2]) > 1e-9 and math.hypot(v[0] / v[2], v[1] / v[2]) < 1e4
    vV, vH = vp(fam['v']), vp(fam['h'])
    I3 = np.eye(3)
    if vV is None and vH is None: return I3

    def keystone(v):
        if not finite(v): return I3, np.array([v[0], v[1]])
        p = np.array([v[0] / v[2], v[1] / v[2]]); r2 = p @ p
        if r2 < 1.4 * 1.4: raise ValueError('these lines meet inside the picture')
        return np.array([[1, 0, 0], [0, 1, 0], [-p[0] / r2, -p[1] / r2, 1]]), p
    rot = lambda a: np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    if vV is not None and vH is None:
        P, d = keystone(vV)
        if d[1] < 0: d = -d
        return rot(math.atan2(d[0], d[1])) @ P
    if vH is not None and vV is None:
        P, d = keystone(vH)
        if d[0] < 0: d = -d
        return rot(math.atan2(-d[1], d[0])) @ P
    L = _cross(vV, vH)
    P = I3
    if abs(L[2]) > 1e-12:
        if abs(L[2]) / math.hypot(L[0], L[1]) < 1.4: raise ValueError('the vertical and horizontal lines do not fit together')
        P = np.array([[1, 0, 0], [0, 1, 0], [L[0] / L[2], L[1] / L[2], 1]])
    elif math.hypot(L[0], L[1]) > 1e-12: raise ValueError('the vertical and horizontal lines do not fit together')

    def dirn(v):
        q = P @ v; n = math.hypot(q[0], q[1]) or 1
        return np.array([q[0] / n, q[1] / n])
    dV, dH = dirn(vV), dirn(vH)
    if dV[1] < 0: dV = -dV
    if dH[0] < 0: dH = -dH
    det = dH[0] * dV[1] - dV[0] * dH[1]
    if abs(det) < 0.2: raise ValueError('the vertical and horizontal lines are too close in direction')
    A = np.array([[dV[1] / det, -dV[0] / det], [-dH[1] / det, dH[0] / det]])
    A = A / math.sqrt(abs(np.linalg.det(A)))
    G = np.eye(3); G[:2, :2] = A
    return G @ P


def hinv_for(g, CW, CH):
    """Output -> source homography (row-major 9 floats) for the perspective settings, or (None, error)."""
    pp = g.get('persp') or {}
    if not pp.get('guides') and not pp.get('vertical') and not pp.get('horizontal') and not pp.get('aspect'): return None, ''
    err = ''
    try: G = upright_from_guides(pp.get('guides') or [], CW, CH)
    except ValueError as ex: err, G = str(ex), np.eye(3)
    a = (pp.get('horizontal') or 0) / 100 * 0.5; b = (pp.get('vertical') or 0) / 100 * 0.5; s = 2 ** ((pp.get('aspect') or 0) / 100 * 0.5)
    F = np.diag([1 / math.sqrt(s), math.sqrt(s), 1]) @ np.array([[1, 0, 0], [0, 1, 0], [a, b, 1]]) @ G
    Hi = np.linalg.inv(F).ravel()
    return (list(map(float, Hi)) if np.all(np.isfinite(Hi)) else None), err


# ---- masks: rasterised in the developed image's orientation, at up to 2048 px
def grow_shrink(a, r):
    """Grow (r > 0) or shrink (r < 0) a uint8 mask by r pixels, with a one-pixel soft edge (as the web's EDT)."""
    from scipy.ndimage import distance_transform_edt
    inside = r > 0; R = abs(r)
    m = a >= 128
    d = distance_transform_edt(~m) if inside else distance_transform_edt(m)
    t = np.clip(R + 0.5 - d, 0, 1)
    if inside: return np.round(255 * np.maximum(m.astype(float), t)).astype(np.uint8)
    return np.where(m, np.round(255 * (1 - t)), 0).astype(np.uint8)


def css_blur(a, px):
    """A canvas blur(px) filter (Gaussian, standard deviation px) on a float alpha plane."""
    if px <= 0: return a
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(a, px, mode='constant')


def draw_comp(c, W, H, frame, frame_diag, img_w, bitmaps):
    """One mask part as float alpha 0..1 (H x W) in image orientation; returns (alpha, brush radius)."""
    import cv2
    F = frame
    px = lambda x, y: ((x - F['x']) / F['w'] * W, (y - F['y']) / F['h'] * H)
    A = np.zeros((H, W), np.uint8)
    SH = 4   # sub-pixel bits for OpenCV
    if c['type'] == 'brush':
        r = max(0.5, c['size'] * frame_diag * W / img_w)
        pts = np.array([px(*q) for q in c['pts']]) * (1 << SH)
        if len(pts) == 1:
            cv2.circle(A, tuple(np.round(pts[0]).astype(int)), int(round(r * (1 << SH))), 255, -1, cv2.LINE_AA, SH)
        else:
            cv2.polylines(A, [np.round(pts).astype(np.int32)], False, 255, max(1, int(round(2 * r))), cv2.LINE_AA, SH)
            for q in (pts[0], pts[-1]):
                cv2.circle(A, tuple(np.round(q).astype(int)), int(round(r * (1 << SH))), 255, -1, cv2.LINE_AA, SH)
        return A.astype(np.float32) / 255, r
    if c['type'] == 'poly':
        if len(c['pts']) < 3: return A.astype(np.float32), 0
        pts = np.array([px(*q) for q in c['pts']]) * (1 << SH)
        cv2.fillPoly(A, [np.round(pts).astype(np.int32)], 255, cv2.LINE_AA, SH)
        return A.astype(np.float32) / 255, 0
    if c['type'] in ('sam', 'fill'):
        bm = bitmaps.get(c['ref'])
        if bm is None: return A.astype(np.float32), 0
        fr = bm['frame']
        x0, y0 = px(fr['x'], fr['y']); x1, y1 = px(fr['x'] + fr['w'], fr['y'] + fr['h'])
        if x1 - x0 < 1 or y1 - y0 < 1: return A.astype(np.float32), 0
        # place the bitmap (bilinear) into the raster
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        u = (xx + 0.5 - x0) / (x1 - x0) * bm['w'] - 0.5; v = (yy + 0.5 - y0) / (y1 - y0) * bm['h'] - 0.5
        inside = (u > -0.5) & (v > -0.5) & (u < bm['w'] - 0.5) & (v < bm['h'] - 0.5)
        val = cv2.remap(bm['data'].astype(np.float32), u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return np.where(inside, val / 255, 0).astype(np.float32), 0
    return A.astype(np.float32), 0


def rasterize_mask(m, W, H, frame, frame_diag, img_w, bitmaps):
    a = np.zeros((H, W), np.float32)
    for c in m['comps']:
        layer, r = draw_comp(c, W, H, frame, frame_diag, img_w, bitmaps)
        if c['type'] == 'brush' and c.get('feather', 0) > 0: layer = css_blur(layer, r * c['feather'] * 0.5)
        if c['type'] == 'poly' and (c.get('featherPx') or 0) > 0: layer = css_blur(layer, c['featherPx'] * W / img_w)
        layer = layer * (c.get('flow', 1) if c.get('flow') is not None else 1)
        a = a * (1 - layer) if c.get('erase') else layer + a * (1 - layer)
    a8 = np.round(np.clip(a, 0, 1) * 255).astype(np.uint8)
    k = W / img_w
    if m.get('grow'): a8 = grow_shrink(a8, m['grow'] * k)
    if (m.get('feather') or 0) > 0:
        a8 = np.round(np.clip(css_blur(a8.astype(np.float32) / 255, max(0.3, m['feather'] * k / 2)), 0, 1) * 255).astype(np.uint8)
    if m.get('invert'): a8 = 255 - a8
    return a8


# ---- Lensfun
class Lensfun:
    def __init__(self):
        self.lenses, self.by_key, self.cameras, self.compat, self.source = [], {}, [], {}, ''

    def load(self, folders):
        import glob
        import os
        for d in folders:
            files = sorted(glob.glob(os.path.join(d, '*.xml')))
            if files:
                for f in files:
                    try: self._parse(f)
                    except Exception: pass
                self.source = d
                break
        return bool(self.lenses)

    def _parse(self, path):
        import xml.etree.ElementTree as ET
        root = ET.parse(path).getroot()
        kids = lambda el, tag: [e for e in el if e.tag == tag]

        def txt(el, tag):
            cs = kids(el, tag)
            de = next((e for e in cs if not e.get('lang')), None); en = next((e for e in cs if e.get('lang') == 'en'), None)
            return ((de if de is not None else en).text or '').strip() if (de is not None or en is not None) else '', \
                   ((en if en is not None else de).text or '').strip() if (de is not None or en is not None) else ''
        num = lambda e, a, d=0.0: float(e.get(a)) if e.get(a) is not None else d
        for m in kids(root, 'mount'):
            self.compat[txt(m, 'name')[0]] = [(e.text or '').strip() for e in kids(m, 'compat')]
        for c in kids(root, 'camera'):
            mk, md = txt(c, 'maker'), txt(c, 'model')
            try: crop = float(txt(c, 'cropfactor')[0] or 1)
            except ValueError: crop = 1
            self.cameras.append(dict(maker=mk[0], makerEn=mk[1], model=md[0], modelEn=md[1], mount=txt(c, 'mount')[0], crop=crop))
        import re
        for l in kids(root, 'lens'):
            mk, md, ar = txt(l, 'maker'), txt(l, 'model'), txt(l, 'aspect-ratio')[0]
            try: crop = float(txt(l, 'cropfactor')[0] or 1)
            except ValueError: crop = 1
            asp = (float(ar.split(':')[0]) / float(ar.split(':')[1]) if ':' in ar else float(ar)) if ar else 1.5
            L = dict(maker=mk[1], model=md[0], modelEn=md[1], mounts=[(e.text or '').strip() for e in kids(l, 'mount')], crop=crop, aspect=asp,
                     dist=[], tca=[], vig=[], minF=0.0, maxF=0.0)
            for cal in kids(l, 'calibration'):
                for d in kids(cal, 'distortion'):
                    mdl = d.get('model')
                    terms = [num(d, 'a'), num(d, 'b'), num(d, 'c')] if mdl == 'ptlens' else [num(d, 'k1'), 0, 0] if mdl == 'poly3' else [num(d, 'k1'), num(d, 'k2'), 0] if mdl == 'poly5' else None
                    if terms: L['dist'].append(dict(focal=num(d, 'focal'), model=mdl, terms=terms))
                for d in kids(cal, 'tca'):
                    mdl = d.get('model')
                    if mdl == 'linear': L['tca'].append(dict(focal=num(d, 'focal'), model=mdl, terms=[num(d, 'kr', 1), num(d, 'kb', 1)]))
                    elif mdl == 'poly3': L['tca'].append(dict(focal=num(d, 'focal'), model=mdl, terms=[num(d, 'vr', 1), num(d, 'vb', 1), num(d, 'cr'), num(d, 'cb'), num(d, 'br'), num(d, 'bb')]))
                for d in kids(cal, 'vignetting'):
                    if d.get('model') == 'pa':
                        L['vig'].append(dict(focal=num(d, 'focal'), aperture=num(d, 'aperture', 8), distance=num(d, 'distance', 1000), terms=[num(d, 'k1'), num(d, 'k2'), num(d, 'k3')]))
            fe = kids(l, 'focal')
            if fe: L['minF'] = num(fe[0], 'min', num(fe[0], 'value')); L['maxF'] = num(fe[0], 'max', L['minF'])
            if not L['minF']:
                fs = [c['focal'] for c in L['dist'] + L['tca'] + L['vig'] if c['focal']]
                mm = re.search(r'(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*mm', L['modelEn'] or L['model'], re.I)
                if mm: L['minF'] = float(mm[1]); L['maxF'] = float(mm[2] or mm[1])
                elif fs: L['minF'], L['maxF'] = min(fs), max(fs)
            L['dist'].sort(key=lambda c: c['focal']); L['tca'].sort(key=lambda c: c['focal'])
            if not L['dist'] and not L['tca'] and not L['vig']: continue
            L['key'] = f"{L['maker']}|{L['model']}|{L['mounts'][0] if L['mounts'] else ''}|{L['crop']:g}"
            if L['key'] in self.by_key: continue
            self.by_key[L['key']] = L; self.lenses.append(L)

    @staticmethod
    def norm(s):
        import re
        s = (s or '').lower()
        s = re.sub(r'([0-9])([a-z])', r'\1 \2', s); s = re.sub(r'([a-z])([0-9])', r'\1 \2', s)
        return [w for w in re.split(r'[^a-z0-9.]+', s) if w and w not in ('mm', 'f', 'lens', 'zoom', '.')]

    def find_camera(self, make, model):
        if not make and not model: return None
        mk = (self.norm(make) or [''])[0]; md = ' '.join(w for w in self.norm(model) if w != mk)
        best, bs = None, 0
        for c in self.cameras:
            cm = self.norm(c['makerEn'] + ' ' + c['maker'])
            if mk and mk not in cm: continue
            for name in (c['model'], c['modelEn']):
                n = ' '.join(w for w in self.norm(name) if w not in cm)
                s = 3 if n == md else 1 if (md and n.endswith(md)) or (n and md.endswith(n)) else 0
                if s > bs: bs, best = s, c
        return best

    def score(self, q, L):
        if not q: return 0
        T = set(self.norm(f"{L['maker']} {L['model']} {L['modelEn']}"))
        hit = sum(1 for w in q if w in T)
        return 2 * hit / len(q) - 0.5 * (len(T) - hit) / max(1, len(T))

    def candidates(self, lens_make, lens, focal, cam):
        mounts = set([cam['mount']] + self.compat.get(cam['mount'], [])) if cam and cam.get('mount') else None
        q = self.norm(f'{lens_make or ""} {lens or ""}')
        out = []
        for L in self.lenses:
            if mounts and not any(m in mounts for m in L['mounts']): continue
            if focal and L['minF'] and (focal < L['minF'] * 0.97 or focal > L['maxF'] * 1.03): continue
            out.append((self.score(q, L), L))
        out.sort(key=lambda x: (-x[0], x[1]['modelEn']))
        return out

    @staticmethod
    def at(L, focal, aperture, distance=1000):
        def lin(lst):
            if not lst: return None
            f = focal or lst[0]['focal']
            if f <= lst[0]['focal']: return lst[0]
            if f >= lst[-1]['focal']: return lst[-1]
            i = 0
            while lst[i + 1]['focal'] < f: i += 1
            a, b = lst[i], lst[i + 1]
            if a['model'] != b['model'] or b['focal'] == a['focal']: return a if f - a['focal'] < b['focal'] - f else b
            t = (f - a['focal']) / (b['focal'] - a['focal'])
            return dict(model=a['model'], terms=[v + (b['terms'][k] - v) * t for k, v in enumerate(a['terms'])])
        vig = None
        if L['vig']:
            f = focal or L['vig'][0]['focal']; ap = aperture or 8; df = (L['maxF'] - L['minF']) or 1
            ws, acc = 0.0, [0.0, 0.0, 0.0]
            for v in L['vig']:
                d = math.hypot(math.hypot((v['focal'] - f) / df, 4 / v['aperture'] - 4 / ap), 0.1 / v['distance'] - 0.1 / distance)
                if d < 1e-4: vig = dict(terms=list(v['terms'])); break
                w = 1 / d ** 3.5; ws += w; acc = [a + w * t for a, t in zip(acc, v['terms'])]
            if vig is None: vig = dict(terms=[a / ws for a in acc])
        return dict(dist=lin(L['dist']), tca=lin(L['tca']), vig=vig)

    @staticmethod
    def label(L):
        has = ', '.join(x for x in (L['dist'] and 'distortion', L['tca'] and 'CA', L['vig'] and 'vignetting') if x)
        name = L['modelEn'] or L['model']
        crop = f" (crop {L['crop']:g})" if L['crop'] != 1 else ''
        return f"{'' if name.lower().startswith((L['maker'] or '').lower()) else L['maker'] + ' '}{name}{crop} — {has}"


# ───────────────────────────────────────────── AI: inpainting (LaMa, MI-GAN, MST + HAWP), as the web's proc-src
def resize_rgb(src, W, H):
    """Bilinear (area-averaging first when shrinking a lot)."""
    s = src
    while s.shape[1] >= 2 * W and s.shape[0] >= 2 * H:
        nh, nw = s.shape[0] >> 1, s.shape[1] >> 1
        s = s[:2 * nh, :2 * nw].reshape(nh, 2, nw, 2, 3).mean(axis=(1, 3))
    sh, sw = s.shape[:2]
    fy = np.clip((np.arange(H) + 0.5) * sh / H - 0.5, 0, sh - 1.001); fx = np.clip((np.arange(W) + 0.5) * sw / W - 0.5, 0, sw - 1.001)
    yi, xi = fy.astype(np.int64), fx.astype(np.int64)
    wy, wx = (fy - yi)[:, None, None], (fx - xi)[None, :, None]
    return ((s[yi][:, xi] * (1 - wx) + s[yi][:, xi + 1] * wx) * (1 - wy) + (s[yi + 1][:, xi] * (1 - wx) + s[yi + 1][:, xi + 1] * wx) * wy).astype(np.float32)


def resize_mask(src, W, H):
    """Any hole pixel in the footprint counts."""
    h, w = src.shape
    out = np.zeros((H, W), np.uint8)
    for y in range(H):
        y0 = int(math.floor(y * h / H)); y1 = max(y0 + 1, int(math.ceil((y + 1) * h / H)))
        row = src[y0:min(h, y1)].any(axis=0)
        for x in range(W):
            x0 = int(math.floor(x * w / W)); x1 = max(x0 + 1, int(math.ceil((x + 1) * w / W)))
            out[y, x] = 1 if row[x0:min(w, x1)].any() else 0
    return out


def canny_edges(img):
    """scikit-image's canny(grey, sigma=2) with thresholds 0.1 / 0.2, as MST was trained."""
    from scipy.ndimage import correlate1d
    S = img.shape[0]
    g = (0.2125 * img[..., 0] + 0.7154 * img[..., 1] + 0.0721 * img[..., 2]).astype(np.float64)
    k = np.exp(-np.arange(-8, 9) ** 2 / 8); k /= k.sum()
    b = correlate1d(correlate1d(g, k, axis=1, mode='reflect'), k, axis=0, mode='reflect')
    gx = np.zeros_like(b); gy = np.zeros_like(b)
    gx[1:-1, 1:-1] = (-b[:-2, :-2] - 2 * b[1:-1, :-2] - b[2:, :-2] + b[:-2, 2:] + 2 * b[1:-1, 2:] + b[2:, 2:])
    gy[1:-1, 1:-1] = (-b[:-2, :-2] - 2 * b[:-2, 1:-1] - b[:-2, 2:] + b[2:, :-2] + 2 * b[2:, 1:-1] + b[2:, 2:])
    mag = np.hypot(gx, gy)
    nms = np.zeros_like(mag)
    ys, xs = np.nonzero(mag[1:-1, 1:-1] > 0); ys += 1; xs += 1
    a = mag[ys, xs]; ux, uy = gx[ys, xs] / a, gy[ys, xs] / a

    def at(px, py):
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int); fx, fy = px - x0, py - y0
        x0 = np.clip(x0, 0, S - 2); y0 = np.clip(y0, 0, S - 2)
        return (mag[y0, x0] * (1 - fx) + mag[y0, x0 + 1] * fx) * (1 - fy) + (mag[y0 + 1, x0] * (1 - fx) + mag[y0 + 1, x0 + 1] * fx) * fy
    keep = (a >= at(xs + ux, ys + uy)) & (a >= at(xs - ux, ys - uy))
    nms[ys[keep], xs[keep]] = a[keep]
    from scipy.ndimage import label
    strong = nms > 0.2; weak = nms > 0.1
    lab, _ = label(weak, structure=np.ones((3, 3)))
    good = np.unique(lab[strong]); good = good[good > 0]
    return np.isin(lab, good).astype(np.float32)


class Inpainter:
    """LaMa (512, float), MI-GAN (any size, uint8) and MST (with HAWP lines and Canny edges), via ONNX Runtime."""
    def __init__(self, models):
        self.models = models
        self._s = {}

    def session(self, rel):
        import os
        if rel in self._s: return self._s[rel]
        import onnxruntime as ort
        try: ort.preload_dlls()
        except Exception: pass
        ort.set_default_logger_severity(3)
        dev = self.models.device
        prov = ['CPUExecutionProvider']
        if dev.startswith('cuda') and 'CUDAExecutionProvider' in ort.get_available_providers():
            prov = [('CUDAExecutionProvider', {'device_id': int(dev.split(':')[1]), 'use_tf32': 0}), 'CPUExecutionProvider']
        s = ort.InferenceSession(self.models.path(os.path.join('inpaint', rel)), providers=prov)
        self._s[rel] = s
        return s

    def run(self, rel, feeds):
        s = self.session(rel)
        names = {i.name for i in s.get_inputs()}
        outs = s.run(None, {k: v for k, v in feeds.items() if k in names})
        return {o.name: v for o, v in zip(s.get_outputs(), outs)}

    def inpaint(self, model, image, hole):
        """image: h x w x 3 display-encoded sRGB in [0,1]; hole: h x w uint8 (1 = fill)."""
        h, w = hole.shape
        if model == 'migan':
            img = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8).transpose(2, 0, 1)[None]
            mk = np.where(hole > 0, 0, 255).astype(np.uint8)[None, None]
            r = self.run('migan_pipeline.onnx', {'image': img, 'mask': mk})
            out = r['result'][0].transpose(1, 2, 0).astype(np.float32) / 255
        else:
            S = 512
            img = resize_rgb(image, S, S); mk = resize_mask(hole, S, S)
            if model == 'mst': res = self.mst(img, mk, S)
            else:
                r = self.run('lama_fp32.onnx', {'image': img.transpose(2, 0, 1)[None].astype(np.float32), 'mask': mk[None, None].astype(np.float32)})
                res = r['output'][0].transpose(1, 2, 0).astype(np.float32) / 255
            out = resize_rgb(res, w, h)
        return np.clip(out, 0, 1)

    def mst(self, img, hole, S):
        m = hole.astype(np.float32)
        lines = self.hawp_lines(img, hole, S); edges = canny_edges(img)
        x = np.empty((6, S, S), np.float32)
        x[:3] = (img * (1 - m[..., None]) + m[..., None]).transpose(2, 0, 1)
        x[3] = lines; x[4] = edges * (1 - m) + m; x[5] = m
        enc = self.run('mst_p2m_encoder_512x512.onnx', {'input.1': x[None]})
        full1 = sorted([(k, v) for k, v in enc.items() if v.shape[1] == 1 and v.shape[2] == S], key=lambda kv: float(kv[0]) if kv[0].replace('.', '').isdigit() else 0)
        if len(full1) < 2: raise RuntimeError('unexpected MST encoder outputs')
        edge, line = full1[0][1][0, 0], full1[1][1][0, 0]
        y = np.empty((6, S, S), np.float32)
        y[:3] = x[:3]; y[3] = np.clip(lines + line, 0, 1); y[4] = edge * m + edges * (1 - m); y[5] = np.clip(line + edge, 0, 1)
        dec = list(self.run('mst_p2m_decoder_512x512.onnx', {'input.1': y[None]}).values())
        return dec[0][0].transpose(1, 2, 0).astype(np.float32)

    def hawp_lines(self, img, hole, S):
        mean = np.array([109.730, 103.832, 98.681]); std = np.array([22.275, 22.124, 23.229])
        mp = np.zeros((S, S), np.float32)

        def run(masked):
            v = img * 255
            if masked: v = np.where(hole[..., None] > 0, 127.5, v)
            x = ((v - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
            r = list(self.run('hawp_512x512_float32_opt.onnx', {'input.1': x}).values())
            L = r[0] if r[0].ndim == 2 else r[1]; sc = r[1] if L is r[0] else r[0]
            return L.reshape(-1, 4), sc.ravel()
        in_hole = lambda x, y: hole[min(S - 1, max(0, jround(y))), min(S - 1, max(0, jround(x)))]

        def draw(x0, y0, x1, y1):
            steps = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) * 2))
            for t in range(steps + 1):
                px, py = x0 + (x1 - x0) * t / steps, y0 + (y1 - y0) * t / steps
                xi, yi = math.floor(px), math.floor(py); fx, fy = px - xi, py - yi
                for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                    X, Y = xi + dx, yi + dy
                    if 0 <= X < S and 0 <= Y < S: mp[Y, X] = max(mp[Y, X], min(1, w * 1.6))
        L, sc = run(False)
        for k in range(len(sc)):
            x0, y0, x1, y1 = L[k]
            if sc[k] > 0.85 and (not in_hole(x0, y0) or not in_hole(x1, y1)): draw(x0, y0, x1, y1)
        L, sc = run(True)
        for k in range(len(sc)):
            x0, y0, x1, y1 = L[k]
            if sc[k] > 0.8 and in_hole(x0, y0) and in_hole(x1, y1): draw(x0, y0, x1, y1)
        return mp


def lin_to_display(lin):
    v = np.asarray(lin, np.float64) @ R2020_TO_SRGB.T
    return np.where(v <= 0.0031308, 12.92 * np.maximum(v, 0), 1.055 * np.power(np.clip(v, 0, 1), 1 / 2.4) - 0.055).astype(np.float32)


def display_to_lin(enc):
    return np.maximum(0, decS(enc) @ SRGB_TO_R2020.T).astype(np.float32)


def dilate(m, r):
    from scipy.ndimage import binary_dilation
    return binary_dilation(m > 0, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool), iterations=r).astype(np.uint8) if r > 0 else m


def feather_alpha(hole, r):
    """Box-blurred hole, so a fill fades in over a few pixels."""
    from scipy.ndimage import uniform_filter1d
    a = hole.astype(np.float64)
    one = np.ones_like(a)
    n1, n0 = uniform_filter1d(one, 2 * r + 1, axis=1, mode='constant'), uniform_filter1d(one, 2 * r + 1, axis=0, mode='constant')
    for _ in range(2):   # means over the pixels inside the picture
        a = uniform_filter1d(a, 2 * r + 1, axis=1, mode='constant') / n1
        a = uniform_filter1d(a, 2 * r + 1, axis=0, mode='constant') / n0
    return np.where(hole > 0, 1, np.minimum(1, a * 2)).astype(np.float32)


# ───────────────────────────────────────────── magnetic lasso: edge costs and live wire
def edge_cost(rgb8):
    """Cost map of a display image: low on strong edges (Sobel of the grey), as the web app's."""
    g = rgb8[..., 0] * 0.299 + rgb8[..., 1] * 0.587 + rgb8[..., 2] * 0.114
    mag = np.zeros(g.shape, np.float64)
    gx = -g[:-2, :-2] - 2 * g[1:-1, :-2] - g[2:, :-2] + g[:-2, 2:] + 2 * g[1:-1, 2:] + g[2:, 2:]
    gy = -g[:-2, :-2] - 2 * g[:-2, 1:-1] - g[:-2, 2:] + g[2:, :-2] + 2 * g[2:, 1:-1] + g[2:, 2:]
    mag[1:-1, 1:-1] = np.hypot(gx, gy)
    s = np.sort(mag.ravel()[::13])
    mx = s[int(len(s) * 0.98)] if len(s) else 1
    return (0.08 + np.maximum(0, 1 - mag / (mx or 1))).astype(np.float32)


@njit(nogil=True, cache=_C)
def _live_wire(cost, ax, ay, bx, by, x0, y0, ww, hh):
    W = cost.shape[1]
    n = ww * hh
    dist = np.full(n, np.inf); prev = np.full(n, -1, np.int64); done = np.zeros(n, np.uint8)
    cap = 1 << 16
    hk = np.empty(cap); hi = np.empty(cap, np.int64); hn = 0
    s0 = (ay - y0) * ww + (ax - x0); t = (by - y0) * ww + (bx - x0)
    dist[s0] = 0.0
    hk[0] = 0.0; hi[0] = s0; hn = 1
    while hn > 0:
        i = hi[0]; lk = hk[hn - 1]; li = hi[hn - 1]; hn -= 1
        c = 0
        while True:   # sift down
            l = 2 * c + 1
            if l >= hn: break
            r = l + 1
            m = r if (r < hn and hk[r] < hk[l]) else l
            if hk[m] >= lk: break
            hk[c] = hk[m]; hi[c] = hi[m]; c = m
        hk[c] = lk; hi[c] = li
        if done[i]: continue
        done[i] = 1
        if i == t: break
        x = i % ww; y = i // ww; d = dist[i]
        for dy in range(-1, 2):
            Y = y + dy
            if Y < 0 or Y >= hh: continue
            for dx in range(-1, 2):
                if dx == 0 and dy == 0: continue
                X = x + dx
                if X < 0 or X >= ww: continue
                j = Y * ww + X
                if done[j]: continue
                nd = d + cost[Y + y0, X + x0] * (1.4142 if (dx != 0 and dy != 0) else 1.0)
                if nd < dist[j]:
                    dist[j] = nd; prev[j] = i
                    if hn == cap:
                        cap *= 2
                        k2 = np.empty(cap); i2 = np.empty(cap, np.int64)
                        k2[:hn] = hk[:hn]; i2[:hn] = hi[:hn]; hk = k2; hi = i2
                    c = hn; hn += 1
                    while c > 0:   # sift up
                        pc = (c - 1) >> 1
                        if hk[pc] <= nd: break
                        hk[c] = hk[pc]; hi[c] = hi[pc]; c = pc
                    hk[c] = nd; hi[c] = j
    path = []
    i = t
    while i >= 0 and i != s0:
        path.append(i); i = prev[i]
    return path


def live_wire(cost, a, b):
    """Cheapest 8-connected path between two view points (0..1) through the edge-cost map, searched in a band
    round the straight line. Returns a list of (u, v)."""
    H, W = cost.shape
    ax, ay = round(a[0] * (W - 1)), round(a[1] * (H - 1)); bx, by = round(b[0] * (W - 1)), round(b[1] * (H - 1))
    mg = min(120, 15 + 0.15 * math.hypot(bx - ax, by - ay))
    x0, y0 = max(0, int(math.floor(min(ax, bx) - mg))), max(0, int(math.floor(min(ay, by) - mg)))
    x1, y1 = min(W - 1, int(math.ceil(max(ax, bx) + mg))), min(H - 1, int(math.ceil(max(ay, by) + mg)))
    ww, hh = x1 - x0 + 1, y1 - y0 + 1
    if ww * hh > 6e6: return [tuple(b)]
    path = _live_wire(cost, ax, ay, bx, by, x0, y0, ww, hh)
    out = [((i % ww + x0) / (W - 1), (i // ww + y0) / (H - 1)) for i in reversed(path)]
    return out or [tuple(b)]
