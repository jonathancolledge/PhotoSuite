# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Hugin interchange for PANOSTITCH: control points from ALIKED+LightGlue (or SIFT) written as a
Hugin .pto project, either a complete new project from a finished stitch (write_pto) or added to
a project Hugin hands over when this is set up as its control point detector (add_control_points).

Hugin's control points are in the image's *stored* pixel grid (before EXIF rotation) — matching
is done on upright images and mapped back with Shot.stored_points.
"""
import math
import os
import re

import cv2
import numpy as np

import panostitch_engine as E

PTO_PROJ = {'rectilinear': 0, 'cylindrical': 1, 'equirectangular': 2}
DEFAULT_POINTS = 0          # 0 = every good match (Hugin copes with thousands; a cap only thins the spread)


# ───────────────────────────────────────────── control point choice
def select_points(q0, q1, sc, w, h, max_points=DEFAULT_POINTS, grid=8):
    """Best-scoring matches spread over the overlap: one per grid cell (in the first image) in
    score order, then a second per cell, … until max_points (0 or None: all). Returns indices."""
    if not max_points or len(q0) <= max_points: return np.arange(len(q0))
    cell = (np.clip((q0[:, 0] / w * grid).astype(int), 0, grid - 1) * grid +
            np.clip((q0[:, 1] / h * grid).astype(int), 0, grid - 1))
    order = np.argsort(-sc, kind='stable')
    rank = np.empty(len(order), int)
    seen = {}
    for k in order:                       # rank within its cell: 0 for the cell's best, 1 next, …
        c = cell[k]
        rank[k] = seen.get(c, 0)
        seen[c] = rank[k] + 1
    pick = sorted(order, key=lambda k: (rank[k], -sc[k]))[:max_points]
    return np.array(pick, int)


# ───────────────────────────────────────────── rotation-only consistency (what Hugin's optimiser models)
rays, kabsch, pair_rotation = E.rays, E.kabsch, E.pair_rotation


def _spread(pts):
    """Area of the convex hull of points (how much of the photo they cover)."""
    if len(pts) < 3: return 0.0
    return float(cv2.contourArea(cv2.convexHull(np.asarray(pts, np.float32))))


def consistent_points(edges, shots, focal, thr_frac=0.0008, min_keep=0.4, min_spread=0.5):
    """For each edge, the matches consistent with one shared camera centre — the model Hugin optimises
    — at focal lengths focal[shot]. Adaptive per pair: the strict tolerance (~5px on 26MP) is loosened
    step by step (up to ~2% of the photo) until enough matches pass (min_keep of them) and they cover a
    good part (min_spread) of the area all the matches span. A strict tolerance alone keeps only a small
    cluster wherever lens distortion, a slightly wrong field of view or hand-held parallax bend the
    picture — and Hugin can't place photos from a cluster (it solves distortion and field of view itself
    if it gets well-spread points). If no tolerance qualifies (strong parallax), the pair keeps all its
    matches (they already passed a geometric check). Returns {edge id: (R_ij, mask, tolerance px or
    None)} and the total kept."""
    out, total = {}, 0
    for e in edges:
        q0, q1, _ = e['cp']
        a, b = shots[e['i']], shots[e['j']]
        f = (focal[e['i']] + focal[e['j']]) / 2
        base = max(3.0, thr_frac * max(a.w, a.h))
        ra, rb = rays(a, q0, focal[e['i']]), rays(b, q1, focal[e['j']])
        full = _spread(q0)
        choice = None
        for mult in (1, 2, 4, 8, 16, 25):
            thr = min(base * mult, 0.02 * max(a.w, a.h))
            R, m, _ = pair_rotation(ra, rb, f, thr)
            if R is not None and m.sum() >= max(12, min_keep * len(q0)) and _spread(q0[m]) >= min_spread * full:
                choice = (R, m, thr); break
        if choice is None:
            R = R if R is not None else np.eye(3)
            choice = (R, np.ones(len(q0), bool), None)
        out[id(e)] = choice
        total += int(choice[1].sum())
    return out, total


def best_focal_scale(edges, shots, focal):
    """The common scale on the given focal lengths that the most matches agree with."""
    def score(k):
        return consistent_points(edges, shots, {n: f * k for n, f in focal.items()})[1]
    grid = np.exp(np.linspace(math.log(0.6), math.log(1.6), 21))
    scores = [score(k) for k in grid]
    k0 = grid[int(np.argmax(scores))]
    fine = k0 * np.exp(np.linspace(-0.05, 0.05, 11))
    fs = [score(k) for k in fine]
    return float(fine[int(np.argmax(fs))]), max(fs)


def solve_rotations(edges, shots, nodes, ref, focal, cons):
    """Camera-to-world rotations (ref = identity) at fixed focal lengths: chained along the
    strongest consistent pairs, then refined jointly on all consistent points (robust loss)."""
    from scipy.optimize import least_squares
    good = [e for e in edges if id(e) in cons and cons[id(e)][1].sum() >= 8 and e['i'] in nodes and e['j'] in nodes]
    R = {ref: np.eye(3)}
    for e in sorted(good, key=lambda e: -cons[id(e)][1].sum()) * len(nodes):
        Rij = cons[id(e)][0]                 # ray_j = Rij ray_i  =>  R_j = R_i Rij^T
        if e['i'] in R and e['j'] not in R: R[e['j']] = R[e['i']] @ Rij.T
        elif e['j'] in R and e['i'] not in R: R[e['i']] = R[e['j']] @ Rij
    free = [n for n in R if n != ref]
    if not free: return R
    data = []
    for e in good:
        if e['i'] in R and e['j'] in R:
            q0, q1, _ = e['cp']
            m = cons[id(e)][1]
            data.append((e['i'], e['j'], rays(shots[e['i']], q0[m], focal[e['i']]), rays(shots[e['j']], q1[m], focal[e['j']]),
                         (focal[e['i']] + focal[e['j']]) / 2))
    seed = dict(R)
    idx = {n: k for k, n in enumerate(free)}
    def rot(p, n):
        return seed[n] if n == ref else E.rodrigues(p[idx[n] * 3: idx[n] * 3 + 3]) @ seed[n]
    def resid(p):
        return np.concatenate([((ri @ rot(p, i).T) - (rj @ rot(p, j).T)).ravel() * f for i, j, ri, rj, f in data])
    sol = least_squares(resid, np.zeros(3 * len(free)), loss='huber', f_scale=2.0, max_nfev=200)
    return {n: rot(sol.x, n) for n in R}


def cp_lines(edges, shots, index, max_points=DEFAULT_POINTS, sizes=None, cons=None):
    """'c n.. N.. x y X Y t0' lines. index maps shot index -> image number in the project; sizes
    (optional) maps image number -> (w, h) the project declares, used if it differs from the file;
    cons (optional, from consistent_points) restricts each pair to its parallax-free matches."""
    out, counts = [], []
    for e in edges:
        i, j = e['i'], e['j']
        if i not in index or j not in index: continue
        q0, q1, sc = e['cp']
        if cons is not None:
            m = cons[id(e)][1]
            if m.sum() < 3: continue
            q0, q1, sc = q0[m], q1[m], sc[m]
        a, b = shots[i], shots[j]
        k = select_points(q0, q1, sc, a.w, a.h, max_points)
        s0, s1 = a.stored_points(q0[k]), b.stored_points(q1[k])
        for p, s, n in ((s0, a, index[i]), (s1, b, index[j])):
            if sizes and n in sizes and sizes[n] != (s.raw_w, s.raw_h):
                p *= (sizes[n][0] / s.raw_w, sizes[n][1] / s.raw_h)
        for (x, y), (X, Y) in zip(s0, s1):
            out.append(f'c n{index[i]} N{index[j]} x{x:.4f} y{y:.4f} X{X:.4f} Y{Y:.4f} t0')
        counts.append(len(k))
    return out, counts


# ───────────────────────────────────────────── orientation in Hugin's terms
def _stored_rotation(R, o):
    """Our rotation maps upright-image rays to the world; Hugin's image is the stored one."""
    M = {3: [[-1, 0], [0, -1]], 6: [[0, -1], [1, 0]], 8: [[0, 1], [-1, 0]]}.get(o)
    if M is None: return R            # 1, or a mirrored orientation Hugin can't express as a rotation
    M3 = np.eye(3)
    M3[:2, :2] = M
    return R @ M3


def yaw_pitch_roll(R):
    """Hugin's y, p, r (degrees) for a camera-to-world rotation in our axes (x right, y down, z forward)."""
    yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[1, 2]))))
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rr = (Ry @ Rx).T @ R
    roll = math.degrees(math.atan2(Rr[1, 0], Rr[0, 0]))
    return yaw, pitch, roll


def _hfov_from_exif(info, w, h):
    """Horizontal field of view of the stored image width from the 35mm-equivalent focal length."""
    f35 = info.get('focal35')
    if not f35: return None
    half_long = 18.0                  # half of 36mm, the long side of a full frame
    half = half_long * (w / max(w, h))
    return math.degrees(2 * math.atan(half / f35))


# ───────────────────────────────────────────── a complete project from a stitch
def horizon_cp_lines(layout, index, xs, ys, pairs=24):
    """A traced horizon (output pixels, not flipped) as Hugin "horizontal line" control points (type
    t2): consecutive points along the line, ~pairs+1 of them spread over its length, each pair saying
    "these two lie on one horizontal line". Points in different photos tie the photos' tilts together
    — exactly what removes a step; the optimiser takes them into account by itself."""
    allp, _, keep = layout.horizon_steps(xs, ys)          # without the unreliable bits next to each step
    smp = [allp[i] for i in keep if allp[i][0] in index]
    if len(smp) < 2: return []
    pick = [smp[i] for i in np.unique(np.linspace(0, len(smp) - 1, pairs + 1).round().astype(int))]
    out = []
    for (na, xa, ya, _), (nb, xb, yb, _) in zip(pick[:-1], pick[1:]):
        (sxa, sya), = layout.shots[na].stored_points([[xa, ya]])
        (sxb, syb), = layout.shots[nb].stored_points([[xb, yb]])
        out.append(f'c n{index[na]} N{index[nb]} x{sxa:.4f} y{sya:.4f} X{sxb:.4f} Y{syb:.4f} t2')
    return out


def write_pto(path, layout, max_points=DEFAULT_POINTS, lens='auto', log=None, horizon=None):
    """Writes a Hugin project for a finished layout: its photos, control points, and starting
    lens/positions. lens: 'auto' (start from EXIF — or our estimate without EXIF — and adjust it to
    the focal length most matches agree with) or 'exif' (EXIF as is). Control points are limited
    to matches consistent with a single viewpoint, the model Hugin optimises (like cpclean).
    Returns a summary dict."""
    log = log or (lambda *a: None)
    shots = layout.shots
    nodes = [n for n in sorted(set(layout.nodes) | {layout.ref})]
    index = {n: k for k, n in enumerate(nodes)}
    infos = {n: E.exif_info(shots[n].path) for n in nodes}
    edges = [e for e in layout.edges if e['i'] in index and e['j'] in index]

    # focal lengths in full-resolution pixels: EXIF where the camera reports it, else ours
    focal = {}
    for n in nodes:
        s = shots[n]
        hv = _hfov_from_exif(infos[n], s.raw_w, s.raw_h)
        focal[n] = (s.raw_w / 2) / math.tan(math.radians(hv) / 2) if hv else (layout.f or 0.85 * max(s.w, s.h))
    exif_any = any(_hfov_from_exif(infos[n], 1, 1) for n in nodes)
    fscale = 1.0
    if lens == 'auto' and not exif_any:
        # no EXIF: our own estimate is biased by parallax, so take the focal most matches agree with.
        # (With EXIF this search is worse than the EXIF value on scenes with near foreground.)
        fscale, _ = best_focal_scale(edges, shots, focal)
        focal = {n: f * fscale for n, f in focal.items()}
    cons, n_cons = consistent_points(edges, shots, focal)
    n_all = sum(len(e['cp'][0]) for e in edges)
    log('info', f'Hugin export: lens from {"EXIF" if exif_any else f"matches (×{fscale:.3f} of our estimate)"}; '
                f'{n_cons}/{n_all} matches consistent with a single viewpoint (the rest are parallax or mismatches)')
    rot = solve_rotations(edges, shots, nodes, layout.ref, focal, cons)

    # world: centre the yaws so Hugin's panorama is centred on the photos
    yprs = {}
    for n in nodes:
        R = rot.get(n)
        yprs[n] = yaw_pitch_roll(_stored_rotation(R, shots[n].orientation)) if R is not None else None
    known = [v for v in yprs.values() if v is not None]
    if known:
        ys = np.radians([v[0] for v in known])
        mid = math.degrees(math.atan2(np.sin(ys).mean(), np.cos(ys).mean()))
        yprs = {n: (None if v is None else (((v[0] - mid + 180) % 360) - 180, v[1], v[2])) for n, v in yprs.items()}

    # lenses: photos of the same stored size and focal length share one (like Hugin's pto_gen). The
    # field of view is across the stored width, so a photo stored portrait needs its own lens entry.
    groups, lens_of = {}, {}
    for n in nodes:
        s, inf = shots[n], infos[n]
        key = (s.raw_w, s.raw_h, round(inf.get('focal') or 0, 1))
        lens_of[n] = groups.setdefault(key, index[n])
    first = {}
    for n in nodes: first.setdefault(lens_of[n], n)
    hfov = {n: math.degrees(2 * math.atan(shots[n].raw_w / 2 / focal[first[lens_of[n]]])) for n in nodes}
    f = focal[layout.ref]

    # panorama: our projection, wide enough for the photos, ~full resolution at the centre
    mode = layout.mode
    span = 0.0
    if known:
        ys = [yprs[n][0] for n in nodes if yprs[n] is not None]
        span = max(ys) - min(ys) + max(hfov.values())
    if mode == 'rectilinear':
        pv = min(150.0, max(span, max(hfov.values())) * 1.1)
    else:
        pv = 360.0 if span > 300 else min(360.0, span * 1.1 or 120.0)
    px_per_deg = (f or shots[layout.ref].w / max(1e-6, math.radians(hfov[layout.ref]))) * math.pi / 180
    if mode == 'rectilinear':
        pw = int(round(2 * math.tan(math.radians(pv / 2)) * px_per_deg * 180 / math.pi))
    else:
        pw = int(round(pv * px_per_deg))
    ph = int(round(pw * layout.H / max(1, layout.W))) if mode != 'equirectangular' or pv < 360 else pw // 2
    pw, ph = max(100, min(pw, 300000)), max(50, min(ph, 150000))

    # output exposure = the photos' (as pto_gen does); E0 against EXIF Eev ~11 renders white
    eevs = [infos[n]['eev'] for n in nodes if 'eev' in infos[n]]
    pe = float(np.median(eevs)) if eevs else 0.0
    lines = ['# hugin project file', '#hugin_ptoversion 2',
             f'p f{PTO_PROJ.get(mode, 2)} w{pw} h{ph} v{pv:.6g}  k0 E{pe:.10g} R0 n"TIFF_m c:LZW r:CROP"', 'm i0', '',
             '# image lines']
    for n in nodes:
        s, inf, k, lk = shots[n], infos[n], index[n], lens_of[n]
        y, p, r = yprs[n] or (0.0, 0.0, 0.0)
        eev = inf.get('eev', 0.0)
        if inf.get('focal') and inf.get('focal35'):
            lines.append(f'#-hugin  cropFactor={inf["focal35"] / inf["focal"]:.6g}')
        if lk == k:
            lens_part = (f'v{hfov[n]:.10g} Ra0 Rb0 Rc0 Rd0 Re0 Eev{eev:.10g} Er1 Eb1 r{r:.10g} p{p:.10g} y{y:.10g} '
                         'TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0')
        else:
            lens_part = (f'v={lk} Ra={lk} Rb={lk} Rc={lk} Rd={lk} Re={lk} Eev{eev:.10g} Er1 Eb1 r{r:.10g} p{p:.10g} y{y:.10g} '
                         f'TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a={lk} b={lk} c={lk} d={lk} e={lk} g={lk} t={lk} '
                         f'Va={lk} Vb={lk} Vc={lk} Vd={lk} Vx={lk} Vy={lk}')
        lines.append(f'i w{s.raw_w} h{s.raw_h} f0 {lens_part}  Vm5 n"{os.path.abspath(s.path)}"')
        if yprs[n] is None and n != layout.ref:
            log('warn', f'{s.name}: no starting position — Hugin will place it from the control points')

    ref = index[layout.ref]
    lines += ['', '', '# specify variables that should be optimized']
    for k in sorted(set(lens_of.values())):
        lines.append(f'v v{k}')
    for n in nodes:
        if index[n] != ref:
            lines += [f'v y{index[n]}', f'v p{index[n]}', f'v r{index[n]}']
    lines += ['v', '', '', '# control points']
    cps, counts = cp_lines(edges, shots, index, max_points, cons=cons)
    lines += cps
    n_hz = 0
    if horizon is not None:
        hz = horizon_cp_lines(layout, index, *horizon)
        lines += hz
        n_hz = len(hz)
        log('info', f'Hugin export: the traced horizon as {n_hz} horizontal-line control points')
    lines += ['', f'#hugin_optimizeReferenceImage {ref}', '#hugin_blender enblend', '#hugin_remapper nona',
              '#hugin_enblendOptions ', '#hugin_enfuseOptions ', '#hugin_hdrmergeOptions ',
              '#hugin_outputLDRBlended true', '#hugin_outputImageType tif', '#hugin_outputImageTypeCompression LZW',
              '#hugin_optimizerMasterSwitch 1', '#hugin_optimizerPhotoMasterSwitch 21', '']
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lines))
    return {'images': len(nodes), 'pairs': len(counts), 'points': sum(counts), 'hfov': hfov[layout.ref],
            'focal_source': 'EXIF' if exif_any else 'matches', 'focal_scale': fscale,
            'consistent': (n_cons, n_all), 'projection': mode, 'horizon_points': n_hz}


# ───────────────────────────────────────────── Hugin's control point detector interface
_I_RE = re.compile(r'\bn"([^"]*)"')
_W_RE = re.compile(r'(?:^|\s)w(\d+)')
_H_RE = re.compile(r'(?:^|\s)h(\d+)')
_V_RE = re.compile(r'(?:^|\s)v(=?)([-\d.e+]+)')


def read_pto_images(path):
    """[(file path, w, h, hfov°)] for the project's image lines, in order (hfov follows v=N links;
    None if absent)."""
    base = os.path.dirname(os.path.abspath(path))
    out = []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if not line.startswith('i '): continue
            m = _I_RE.search(line)
            if not m: raise ValueError(f'image line without a file name: {line.strip()[:80]}')
            p = m.group(1)
            if not os.path.isabs(p): p = os.path.join(base, p)
            w, h, v = _W_RE.search(line), _H_RE.search(line), _V_RE.search(line)
            hv = None
            if v:
                hv = out[int(float(v.group(2)))][3] if v.group(1) and int(float(v.group(2))) < len(out) else (None if v.group(1) else float(v.group(2)))
            out.append((os.path.normpath(p), int(w.group(1)) if w else None, int(h.group(1)) if h else None, hv))
    return out


def add_control_points(pto_in, pto_out, job, matcher='auto', model_path=None, device='cuda:0',
                       match_res=None, max_points=DEFAULT_POINTS, window=0, prune=True):
    """Hugin detector entry: reads pto_in, matches every pair of its images, writes pto_out =
    pto_in plus the new control points. prune: keep only matches consistent with a single
    viewpoint (at the focal length most matches agree with, searched around the project's).
    Returns (pairs connected, points written)."""
    imgs = read_pto_images(pto_in)
    if len(imgs) < 2: raise RuntimeError('the project has fewer than two images')
    shots = []
    for k, (p, w, h, _) in enumerate(imgs):
        job.check()
        job.progress(0.05 * k / len(imgs), f'loading {os.path.basename(p)}')
        shots.append(E.Shot(p))
    st = E.Stitcher(shots, job, matcher=matcher, model_path=model_path, device=device, match_res=match_res)
    edges = st.match_all(window=window)
    sizes = {k: (w, h) for k, (_, w, h, _) in enumerate(imgs) if w and h}
    for k, s in enumerate(shots):
        if k in sizes and sizes[k] != (s.raw_w, s.raw_h):
            job.log('warn', f'{s.name}: project says {sizes[k][0]}×{sizes[k][1]}, file is {s.raw_w}×{s.raw_h} — points scaled')
    cons = None
    if prune and edges:
        job.progress(0.97, 'keeping matches consistent with a single viewpoint')
        focal = {}
        for k, (s, (_, w, h, hv)) in enumerate(zip(shots, imgs)):
            hv = hv if hv and 1 < hv < 179 else 50.0
            focal[k] = (s.raw_w / 2) / math.tan(math.radians(hv) / 2)
        n_all = sum(len(e['cp'][0]) for e in edges)
        scale = 1.0
        cons, n_cons = consistent_points(edges, shots, focal)
        if n_cons < 0.5 * n_all:
            # the project's lens looks wrong (no EXIF?): judge consistency at the focal most matches agree with
            scale, _ = best_focal_scale(edges, shots, focal)
            cons, n_cons = consistent_points(edges, shots, {k: f * scale for k, f in focal.items()})
        job.log('info', f'{n_cons}/{n_all} matches kept'
                        f'{"" if scale == 1.0 else f" (judged at ×{scale:.3f} the project focal length)"}')
        for e in edges:
            _, m, thr = cons[id(e)]
            how = (f'within {thr:.0f}px of a single viewpoint' if thr is not None
                   else 'all kept — strong parallax, no single-viewpoint fit covers the overlap')
            job.log('info', f'  {shots[e["i"]].name} × {shots[e["j"]].name}: {int(m.sum())}/{len(m)} matches, {how}')
    cps, counts = cp_lines(edges, shots, {k: k for k in range(len(shots))}, max_points, sizes, cons)

    with open(pto_in, encoding='utf-8', errors='replace') as fh:
        lines = fh.read().splitlines()
    at = next((k + 1 for k, l in enumerate(lines) if l.strip().lower() == '# control points'), None)
    if at is None:
        at = next((k for k, l in enumerate(lines) if l.startswith('#hugin_')), len(lines))
    lines[at:at] = cps
    with open(pto_out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lines) + '\n')
    return len(counts), sum(counts)
