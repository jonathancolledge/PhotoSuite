# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Bayer demosaicing for RAW Demosaic desktop: line-for-line ports of the web app's (rawdemosaic.html)
AMaZE, RCD, Malvar-He-Cutler and bilinear, compiled with Numba and run tile-parallel.

AMaZE and RCD are ports of RawTherapee's amaze_demosaic_RT.cc (Emil Martinec, Ingo Weyrich) and
rcd_demosaic.cc (Luis Sanz Rodriguez, Ingo Weyrich), both GPL-3.0, scalar path, tiled exactly as there.
The border interpolation follows RawTherapee's border_interpolate. Their notices, kept as GPL section 4 asks:

    AMaZE: copyright (c) 2008-2010 Emil Martinec <ejmartin@uchicago.edu>, optimized for speed by Ingo Weyrich
    RCD:   Copyright (c) 2017-2020 Luis Sanz Rodriguez and Ingo Weyrich
    Both are free software: you can redistribute and/or modify them under the terms of the GNU General Public
    License as published by the Free Software Foundation, either version 3 of the License, or (at your option)
    any later version. https://github.com/RawTherapee/RawTherapee

Input for every function: a white-balanced Bayer mosaic, float32 H x W in [0, 1] (clip point 1), and
`pat`, the 2x2 CFA pattern as 4 colour ids (0 R, 1 G, 2 B) in row order. Output: float32 H x W x 3.
"""
import math

import numpy as np
from numba import njit, prange

_CACHE = True


@njit(inline='always')
def _clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


@njit(inline='always')
def _median3(a, b, c):
    return max(min(a, b), min(max(a, b), c))


# ───────────────────────────────────────────── AMaZE
@njit(cache=_CACHE)
def _amaze_tile(cfa_in, W, H, pat, top, left, out):
    ts = 160
    tsh = ts // 2
    v1 = ts; v2 = 2 * ts; v3 = 3 * ts
    p1 = -ts + 1; p2 = -2 * ts + 2; p3 = -3 * ts + 3
    m1 = ts + 1; m2 = 2 * ts + 2; m3 = 3 * ts + 3
    eps = 1e-5; epssq = 1e-10; arthresh = 0.75; nyqthresh = 0.5
    clip_pt = 1.0; clip_pt8 = 0.8
    g_odd0 = 0.14659727707323927; g_odd1 = 0.103592713382435; g_odd2 = 0.0732036125103057; g_odd3 = 0.0365543548389495
    gg0 = nyqthresh * 0.07384411893421103; gg1 = nyqthresh * 0.06207511968171489; gg2 = nyqthresh * 0.0521818194747806
    gg3 = nyqthresh * 0.03687419286733595; gg4 = nyqthresh * 0.03099732204057846; gg5 = nyqthresh * 0.018413194161458882
    ge0 = 0.13719494435797422; ge1 = 0.05640252782101291
    gq0 = 0.169917; gq1 = 0.108947; gq2 = 0.069855; gq3 = 0.0287182
    winx = 0; winy = 0; width = W; height = H

    f0 = pat[0]; f1 = pat[1]
    if f0 == 1:
        if f1 == 0:
            ey = 0; ex = 1
        else:
            ey = 1; ex = 0
    else:
        if f0 == 0:
            ey = 0; ex = 0
        else:
            ey = 1; ex = 1

    n = ts * ts; nh = ts * tsh
    rgbgreen = np.zeros(n, np.float32); delhvsqsum = np.zeros(n, np.float32)
    dirwts0 = np.zeros(n, np.float32); dirwts1 = np.zeros(n, np.float32)
    vcd = np.zeros(n, np.float32); hcd = np.zeros(n, np.float32)
    vcdalt = np.zeros(n, np.float32); hcdalt = np.zeros(n, np.float32)
    cddiffsq = np.zeros(n, np.float32); dgintv = np.zeros(n, np.float32); dginth = np.zeros(n, np.float32)
    cfa = np.zeros(n, np.float32)
    hvwt = np.zeros(nh, np.float32); delp = np.zeros(nh, np.float32); delm = np.zeros(nh, np.float32)
    rbint = np.zeros(nh, np.float32); Dgrbsq1m = np.zeros(nh, np.float32); Dgrbsq1p = np.zeros(nh, np.float32)
    pmwt = np.zeros(nh, np.float32); rbm = np.zeros(nh, np.float32); rbp = np.zeros(nh, np.float32)
    nyqutest = np.zeros(nh, np.float32)
    Dgrb0 = np.zeros(nh, np.float32); Dgrb1 = np.zeros(nh, np.float32)
    Dgrb2h = np.zeros(nh, np.float32); Dgrb2v = np.zeros(nh, np.float32)
    nyquist = np.zeros(nh, np.uint8); nyquist2 = np.zeros(nh, np.uint8)

    bottom = min(top + ts, winy + height + 16)
    right = min(left + ts, winx + width + 16)
    rr1 = bottom - top; cc1 = right - left
    rrmin = 16 if top < winy else 0
    ccmin = 16 if left < winx else 0
    rrmax = winy + height - top if bottom > winy + height else rr1
    ccmax = winx + width - left if right > winx + width else cc1

    # tile, with mirrored 16-pixel borders at the image edges
    if rrmin > 0:
        for rr in range(16):
            row = 32 - rr + top
            for cc in range(ccmin, ccmax):
                v = cfa_in[row, cc + left]; cfa[rr * ts + cc] = v; rgbgreen[rr * ts + cc] = v
    for rr in range(rrmin, rrmax):
        for cc in range(ccmin, ccmax):
            v = cfa_in[rr + top, cc + left]; cfa[rr * ts + cc] = v; rgbgreen[rr * ts + cc] = v
    if rrmax < rr1:
        for rr in range(16):
            for cc in range(ccmin, ccmax):
                v = cfa_in[winy + height - rr - 2, left + cc]; cfa[(rrmax + rr) * ts + cc] = v; rgbgreen[(rrmax + rr) * ts + cc] = v
    if ccmin > 0:
        for rr in range(rrmin, rrmax):
            for cc in range(16):
                v = cfa_in[rr + top, 32 - cc + left]; cfa[rr * ts + cc] = v; rgbgreen[rr * ts + cc] = v
    if ccmax < cc1:
        for rr in range(rrmin, rrmax):
            for cc in range(16):
                v = cfa_in[top + rr, winx + width - cc - 2]; cfa[rr * ts + ccmax + cc] = v; rgbgreen[rr * ts + ccmax + cc] = v
    if rrmin > 0 and ccmin > 0:
        for rr in range(16):
            for cc in range(16):
                v = cfa_in[winy + 32 - rr, winx + 32 - cc]; cfa[rr * ts + cc] = v; rgbgreen[rr * ts + cc] = v
    if rrmax < rr1 and ccmax < cc1:
        for rr in range(16):
            for cc in range(16):
                v = cfa_in[winy + height - rr - 2, winx + width - cc - 2]
                cfa[(rrmax + rr) * ts + ccmax + cc] = v; rgbgreen[(rrmax + rr) * ts + ccmax + cc] = v
    if rrmin > 0 and ccmax < cc1:
        for rr in range(16):
            for cc in range(16):
                v = cfa_in[winy + 32 - rr, winx + width - cc - 2]; cfa[rr * ts + ccmax + cc] = v; rgbgreen[rr * ts + ccmax + cc] = v
    if rrmax < rr1 and ccmin > 0:
        for rr in range(16):
            for cc in range(16):
                v = cfa_in[winy + height - rr - 2, winx + 32 - cc]; cfa[(rrmax + rr) * ts + cc] = v; rgbgreen[(rrmax + rr) * ts + cc] = v

    # horizontal and vertical gradients
    for rr in range(2, rr1 - 2):
        for cc in range(2, cc1 - 2):
            indx = rr * ts + cc
            delh = abs(cfa[indx + 1] - cfa[indx - 1]); delv = abs(cfa[indx + v1] - cfa[indx - v1])
            dirwts0[indx] = eps + abs(cfa[indx + v2] - cfa[indx]) + abs(cfa[indx] - cfa[indx - v2]) + delv
            dirwts1[indx] = eps + abs(cfa[indx + 2] - cfa[indx]) + abs(cfa[indx] - cfa[indx - 2]) + delh
            delhvsqsum[indx] = delh * delh + delv * delv
    # vertical and horizontal colour differences
    for rr in range(4, rr1 - 4):
        fcswitch = (pat[(rr & 1) * 2] & 1) == 1   # fc(rr, 4)
        for cc in range(4, cc1 - 4):
            indx = rr * ts + cc
            c0 = cfa[indx]
            cru = cfa[indx - v1] * (dirwts0[indx - v2] + dirwts0[indx]) / (dirwts0[indx - v2] * (eps + c0) + dirwts0[indx] * (eps + cfa[indx - v2]))
            crd = cfa[indx + v1] * (dirwts0[indx + v2] + dirwts0[indx]) / (dirwts0[indx + v2] * (eps + c0) + dirwts0[indx] * (eps + cfa[indx + v2]))
            crl = cfa[indx - 1] * (dirwts1[indx - 2] + dirwts1[indx]) / (dirwts1[indx - 2] * (eps + c0) + dirwts1[indx] * (eps + cfa[indx - 2]))
            crr = cfa[indx + 1] * (dirwts1[indx + 2] + dirwts1[indx]) / (dirwts1[indx + 2] * (eps + c0) + dirwts1[indx] * (eps + cfa[indx + 2]))
            guha = cfa[indx - v1] + (c0 - cfa[indx - v2]) / 2; gdha = cfa[indx + v1] + (c0 - cfa[indx + v2]) / 2
            glha = cfa[indx - 1] + (c0 - cfa[indx - 2]) / 2; grha = cfa[indx + 1] + (c0 - cfa[indx + 2]) / 2
            guar = c0 * cru if abs(1 - cru) < arthresh else guha
            gdar = c0 * crd if abs(1 - crd) < arthresh else gdha
            glar = c0 * crl if abs(1 - crl) < arthresh else glha
            grar = c0 * crr if abs(1 - crr) < arthresh else grha
            hwt = dirwts1[indx - 1] / (dirwts1[indx - 1] + dirwts1[indx + 1])
            vwt = dirwts0[indx - v1] / (dirwts0[indx + v1] + dirwts0[indx - v1])
            Gintvha = vwt * gdha + (1 - vwt) * guha; Ginthha = hwt * grha + (1 - hwt) * glha
            if fcswitch:
                vcd[indx] = c0 - (vwt * gdar + (1 - vwt) * guar); hcd[indx] = c0 - (hwt * grar + (1 - hwt) * glar)
                vcdalt[indx] = c0 - Gintvha; hcdalt[indx] = c0 - Ginthha
            else:
                vcd[indx] = (vwt * gdar + (1 - vwt) * guar) - c0; hcd[indx] = (hwt * grar + (1 - hwt) * glar) - c0
                vcdalt[indx] = Gintvha - c0; hcdalt[indx] = Ginthha - c0
            fcswitch = not fcswitch
            if c0 > clip_pt8 or Gintvha > clip_pt8 or Ginthha > clip_pt8:   # Hamilton-Adams near clipping
                guar = guha; gdar = gdha; glar = glha; grar = grha
                vcd[indx] = vcdalt[indx]; hcd[indx] = hcdalt[indx]
            dgintv[indx] = min((guha - gdha) ** 2, (guar - gdar) ** 2)
            dginth[indx] = min((glha - grha) ** 2, (glar - grar) ** 2)
    for rr in range(4, rr1 - 4):
        c = pat[(rr & 1) * 2] & 1
        for cc in range(4, cc1 - 4):
            indx = rr * ts + cc
            a0 = hcd[indx - 2]; a1 = hcd[indx]; a2 = hcd[indx + 2]
            hcdvar = 3 * (a0 * a0 + a1 * a1 + a2 * a2) - (a0 + a1 + a2) ** 2
            a0 = hcdalt[indx - 2]; a1 = hcdalt[indx]; a2 = hcdalt[indx + 2]
            hcdaltvar = 3 * (a0 * a0 + a1 * a1 + a2 * a2) - (a0 + a1 + a2) ** 2
            a0 = vcd[indx - v2]; a1 = vcd[indx]; a2 = vcd[indx + v2]
            vcdvar = 3 * (a0 * a0 + a1 * a1 + a2 * a2) - (a0 + a1 + a2) ** 2
            a0 = vcdalt[indx - v2]; a1 = vcdalt[indx]; a2 = vcdalt[indx + v2]
            vcdaltvar = 3 * (a0 * a0 + a1 * a1 + a2 * a2) - (a0 + a1 + a2) ** 2
            if hcdaltvar < hcdvar:
                hcd[indx] = hcdalt[indx]
            if vcdaltvar < vcdvar:
                vcd[indx] = vcdalt[indx]
            c0 = cfa[indx]
            if c:   # G site
                Ginth = -hcd[indx] + c0; Gintv = -vcd[indx] + c0
                if hcd[indx] > 0:
                    if 3 * hcd[indx] > Ginth + c0:
                        hcd[indx] = -_median3(Ginth, cfa[indx - 1], cfa[indx + 1]) + c0
                    else:
                        hwt = 1 - 3 * hcd[indx] / (eps + Ginth + c0)
                        hcd[indx] = hwt * hcd[indx] + (1 - hwt) * (-_median3(Ginth, cfa[indx - 1], cfa[indx + 1]) + c0)
                if vcd[indx] > 0:
                    if 3 * vcd[indx] > Gintv + c0:
                        vcd[indx] = -_median3(Gintv, cfa[indx - v1], cfa[indx + v1]) + c0
                    else:
                        vwt = 1 - 3 * vcd[indx] / (eps + Gintv + c0)
                        vcd[indx] = vwt * vcd[indx] + (1 - vwt) * (-_median3(Gintv, cfa[indx - v1], cfa[indx + v1]) + c0)
                if Ginth > clip_pt:
                    hcd[indx] = -_median3(Ginth, cfa[indx - 1], cfa[indx + 1]) + c0
                if Gintv > clip_pt:
                    vcd[indx] = -_median3(Gintv, cfa[indx - v1], cfa[indx + v1]) + c0
            else:   # R or B site
                Ginth = hcd[indx] + c0; Gintv = vcd[indx] + c0
                if hcd[indx] < 0:
                    if 3 * hcd[indx] < -(Ginth + c0):
                        hcd[indx] = _median3(Ginth, cfa[indx - 1], cfa[indx + 1]) - c0
                    else:
                        hwt = 1 + 3 * hcd[indx] / (eps + Ginth + c0)
                        hcd[indx] = hwt * hcd[indx] + (1 - hwt) * (_median3(Ginth, cfa[indx - 1], cfa[indx + 1]) - c0)
                if vcd[indx] < 0:
                    if 3 * vcd[indx] < -(Gintv + c0):
                        vcd[indx] = _median3(Gintv, cfa[indx - v1], cfa[indx + v1]) - c0
                    else:
                        vwt = 1 + 3 * vcd[indx] / (eps + Gintv + c0)
                        vcd[indx] = vwt * vcd[indx] + (1 - vwt) * (_median3(Gintv, cfa[indx - v1], cfa[indx + v1]) - c0)
                if Ginth > clip_pt:
                    hcd[indx] = _median3(Ginth, cfa[indx - 1], cfa[indx + 1]) - c0
                if Gintv > clip_pt:
                    vcd[indx] = _median3(Gintv, cfa[indx - v1], cfa[indx + v1]) - c0
                cddiffsq[indx] = (vcd[indx] - hcd[indx]) ** 2
            c = 0 if c else 1
    for rr in range(6, rr1 - 6):
        for cc in range(6 + (pat[(rr & 1) * 2] & 1), cc1 - 6, 2):
            indx = rr * ts + cc
            uave = vcd[indx] + vcd[indx - v1] + vcd[indx - v2] + vcd[indx - v3]
            dave = vcd[indx] + vcd[indx + v1] + vcd[indx + v2] + vcd[indx + v3]
            lave = hcd[indx] + hcd[indx - 1] + hcd[indx - 2] + hcd[indx - 3]
            rave = hcd[indx] + hcd[indx + 1] + hcd[indx + 2] + hcd[indx + 3]
            Dgrbvvaru = (vcd[indx] - uave) ** 2 + (vcd[indx - v1] - uave) ** 2 + (vcd[indx - v2] - uave) ** 2 + (vcd[indx - v3] - uave) ** 2
            Dgrbvvard = (vcd[indx] - dave) ** 2 + (vcd[indx + v1] - dave) ** 2 + (vcd[indx + v2] - dave) ** 2 + (vcd[indx + v3] - dave) ** 2
            Dgrbhvarl = (hcd[indx] - lave) ** 2 + (hcd[indx - 1] - lave) ** 2 + (hcd[indx - 2] - lave) ** 2 + (hcd[indx - 3] - lave) ** 2
            Dgrbhvarr = (hcd[indx] - rave) ** 2 + (hcd[indx + 1] - rave) ** 2 + (hcd[indx + 2] - rave) ** 2 + (hcd[indx + 3] - rave) ** 2
            hwt = dirwts1[indx - 1] / (dirwts1[indx - 1] + dirwts1[indx + 1])
            vwt = dirwts0[indx - v1] / (dirwts0[indx + v1] + dirwts0[indx - v1])
            vcdvar = epssq + vwt * Dgrbvvard + (1 - vwt) * Dgrbvvaru
            hcdvar = epssq + hwt * Dgrbhvarr + (1 - hwt) * Dgrbhvarl
            Dgrbvvaru = dgintv[indx] + dgintv[indx - v1] + dgintv[indx - v2]
            Dgrbvvard = dgintv[indx] + dgintv[indx + v1] + dgintv[indx + v2]
            Dgrbhvarl = dginth[indx] + dginth[indx - 1] + dginth[indx - 2]
            Dgrbhvarr = dginth[indx] + dginth[indx + 1] + dginth[indx + 2]
            vcdvar1 = epssq + vwt * Dgrbvvard + (1 - vwt) * Dgrbvvaru
            hcdvar1 = epssq + hwt * Dgrbhvarr + (1 - hwt) * Dgrbhvarl
            varwt = hcdvar / (vcdvar + hcdvar); diffwt = hcdvar1 / (vcdvar1 + hcdvar1)
            if (0.5 - varwt) * (0.5 - diffwt) > 0 and abs(0.5 - diffwt) < abs(0.5 - varwt):
                hvwt[indx >> 1] = varwt
            else:
                hvwt[indx >> 1] = diffwt
    # Nyquist texture test
    for rr in range(6, rr1 - 6):
        for cc in range(6 + (pat[(rr & 1) * 2] & 1), cc1 - 6, 2):
            indx = rr * ts + cc
            C = cddiffsq; D = delhvsqsum
            nyqutest[indx >> 1] = (g_odd0 * C[indx] + g_odd1 * (C[indx - m1] + C[indx + p1] + C[indx - p1] + C[indx + m1]) +
                                   g_odd2 * (C[indx - v2] + C[indx - 2] + C[indx + 2] + C[indx + v2]) +
                                   g_odd3 * (C[indx - m2] + C[indx + p2] + C[indx - p2] + C[indx + m2])) - \
                (gg0 * D[indx] + gg1 * (D[indx - v1] + D[indx + 1] + D[indx - 1] + D[indx + v1]) +
                 gg2 * (D[indx - m1] + D[indx + p1] + D[indx - p1] + D[indx + m1]) +
                 gg3 * (D[indx - v2] + D[indx - 2] + D[indx + 2] + D[indx + v2]) +
                 gg4 * (D[indx - v2 - 1] + D[indx - v2 + 1] + D[indx - ts - 2] + D[indx - ts + 2] + D[indx + ts - 2] + D[indx + ts + 2] + D[indx + v2 - 1] + D[indx + v2 + 1]) +
                 gg5 * (D[indx - m2] + D[indx + p2] + D[indx - p2] + D[indx + m2]))
    nystartrow = 0; nyendrow = 0; nystartcol = ts + 1; nyendcol = 0
    for rr in range(6, rr1 - 6):
        for cc in range(6 + (pat[(rr & 1) * 2] & 1), cc1 - 6, 2):
            indx = rr * ts + cc
            if nyqutest[indx >> 1] > 0:
                nyquist[indx >> 1] = 1
                if nystartrow == 0:
                    nystartrow = rr
                nyendrow = rr
                if nystartcol > cc:
                    nystartcol = cc
                if nyendcol < cc:
                    nyendcol = cc
    do_nyquist = nystartrow != nyendrow and nystartcol != nyendcol
    if do_nyquist:
        nyendrow += 1; nyendcol += 1
        nystartcol -= nystartcol & 1
        nystartrow = max(8, nystartrow); nyendrow = min(rr1 - 8, nyendrow)
        nystartcol = max(8, nystartcol); nyendcol = min(cc1 - 8, nyendcol)
        for rr in range(nystartrow, nyendrow):
            for indx in range(rr * ts + nystartcol + (pat[(rr & 1) * 2] & 1), rr * ts + nyendcol, 2):
                t = (nyquist[(indx - v2) >> 1] + nyquist[(indx - m1) >> 1] + nyquist[(indx + p1) >> 1] + nyquist[(indx - 2) >> 1] +
                     nyquist[(indx + 2) >> 1] + nyquist[(indx - p1) >> 1] + nyquist[(indx + m1) >> 1] + nyquist[(indx + v2) >> 1])
                if t > 4:
                    nyquist2[indx >> 1] = 1
                elif t < 4:
                    nyquist2[indx >> 1] = 0
                else:
                    nyquist2[indx >> 1] = nyquist[indx >> 1]
        # area interpolation in Nyquist texture
        for rr in range(nystartrow, nyendrow):
            for indx in range(rr * ts + nystartcol + (pat[(rr & 1) * 2] & 1), rr * ts + nyendcol, 2):
                if not nyquist2[indx >> 1]:
                    continue
                sumcfa = 0.0; sumh = 0.0; sumv = 0.0; sumsqh = 0.0; sumsqv = 0.0; areawt = 0.0
                for i in range(-6, 7, 2):
                    indx1 = indx + i * ts - 6
                    for j in range(-6, 7, 2):
                        if nyquist2[indx1 >> 1]:
                            ct = cfa[indx1]
                            sumcfa += ct; sumh += cfa[indx1 - 1] + cfa[indx1 + 1]; sumv += cfa[indx1 - v1] + cfa[indx1 + v1]
                            sumsqh += (ct - cfa[indx1 - 1]) ** 2 + (ct - cfa[indx1 + 1]) ** 2
                            sumsqv += (ct - cfa[indx1 - v1]) ** 2 + (ct - cfa[indx1 + v1]) ** 2
                            areawt += 1
                        indx1 += 2
                sumh = sumcfa - sumh / 2; sumv = sumcfa - sumv / 2; areawt = areawt / 2
                hcdvar = epssq + abs(areawt * sumsqh - sumh * sumh); vcdvar = epssq + abs(areawt * sumsqv - sumv * sumv)
                hvwt[indx >> 1] = hcdvar / (vcdvar + hcdvar)
    # G at R/B sites
    for rr in range(8, rr1 - 8):
        for indx in range(rr * ts + 8 + (pat[(rr & 1) * 2] & 1), rr * ts + cc1 - 8, 2):
            h1 = indx >> 1
            hvwtalt = (hvwt[(indx - m1) >> 1] + hvwt[(indx + p1) >> 1] + hvwt[(indx - p1) >> 1] + hvwt[(indx + m1) >> 1]) / 4
            if abs(0.5 - hvwt[h1]) < abs(0.5 - hvwtalt):
                hvwt[h1] = hvwtalt
            Dgrb0[h1] = hvwt[h1] * vcd[indx] + (1 - hvwt[h1]) * hcd[indx]
            rgbgreen[indx] = cfa[indx] + Dgrb0[h1]
            if nyquist2[h1]:
                Dgrb2h[h1] = (rgbgreen[indx] - (rgbgreen[indx - 1] + rgbgreen[indx + 1]) / 2) ** 2
                Dgrb2v[h1] = (rgbgreen[indx] - (rgbgreen[indx - v1] + rgbgreen[indx + v1]) / 2) ** 2
            else:
                Dgrb2h[h1] = 0; Dgrb2v[h1] = 0
    # refine Nyquist areas using G curvatures
    if do_nyquist:
        for rr in range(nystartrow, nyendrow):
            for indx in range(rr * ts + nystartcol + (pat[(rr & 1) * 2] & 1), rr * ts + nyendcol, 2):
                if not nyquist2[indx >> 1]:
                    continue
                A = Dgrb2h
                gvarh = epssq + (gq0 * A[indx >> 1] + gq1 * (A[(indx - m1) >> 1] + A[(indx + p1) >> 1] + A[(indx - p1) >> 1] + A[(indx + m1) >> 1]) +
                                 gq2 * (A[(indx - v2) >> 1] + A[(indx - 2) >> 1] + A[(indx + 2) >> 1] + A[(indx + v2) >> 1]) +
                                 gq3 * (A[(indx - m2) >> 1] + A[(indx + p2) >> 1] + A[(indx - p2) >> 1] + A[(indx + m2) >> 1]))
                A = Dgrb2v
                gvarv = epssq + (gq0 * A[indx >> 1] + gq1 * (A[(indx - m1) >> 1] + A[(indx + p1) >> 1] + A[(indx - p1) >> 1] + A[(indx + m1) >> 1]) +
                                 gq2 * (A[(indx - v2) >> 1] + A[(indx - 2) >> 1] + A[(indx + 2) >> 1] + A[(indx + v2) >> 1]) +
                                 gq3 * (A[(indx - m2) >> 1] + A[(indx + p2) >> 1] + A[(indx - p2) >> 1] + A[(indx + m2) >> 1]))
                Dgrb0[indx >> 1] = (hcd[indx] * gvarv + vcd[indx] * gvarh) / (gvarv + gvarh)
                rgbgreen[indx] = cfa[indx] + Dgrb0[indx >> 1]
    for rr in range(6, rr1 - 6):
        if (pat[(rr & 1) * 2] & 1) == 0:
            for cc in range(6, cc1 - 6, 2):
                indx = rr * ts + cc
                delp[indx >> 1] = abs(cfa[indx + p1] - cfa[indx - p1]); delm[indx >> 1] = abs(cfa[indx + m1] - cfa[indx - m1])
                Dgrbsq1p[indx >> 1] = (cfa[indx + 1] - cfa[indx + 1 - p1]) ** 2 + (cfa[indx + 1] - cfa[indx + 1 + p1]) ** 2
                Dgrbsq1m[indx >> 1] = (cfa[indx + 1] - cfa[indx + 1 - m1]) ** 2 + (cfa[indx + 1] - cfa[indx + 1 + m1]) ** 2
        else:
            for cc in range(6, cc1 - 6, 2):
                indx = rr * ts + cc
                Dgrbsq1p[indx >> 1] = (cfa[indx] - cfa[indx - p1]) ** 2 + (cfa[indx] - cfa[indx + p1]) ** 2
                Dgrbsq1m[indx >> 1] = (cfa[indx] - cfa[indx - m1]) ** 2 + (cfa[indx] - cfa[indx + m1]) ** 2
                delp[indx >> 1] = abs(cfa[indx + 1 + p1] - cfa[indx + 1 - p1]); delm[indx >> 1] = abs(cfa[indx + 1 + m1] - cfa[indx + 1 - m1])
    # diagonal interpolation correction
    for rr in range(8, rr1 - 8):
        for cc in range(8 + (pat[(rr & 1) * 2] & 1), cc1 - 8, 2):
            indx = rr * ts + cc; i1 = indx >> 1
            c0 = cfa[indx]
            crse = 2 * cfa[indx + m1] / (eps + c0 + cfa[indx + m2]); crnw = 2 * cfa[indx - m1] / (eps + c0 + cfa[indx - m2])
            crne = 2 * cfa[indx + p1] / (eps + c0 + cfa[indx + p2]); crsw = 2 * cfa[indx - p1] / (eps + c0 + cfa[indx - p2])
            rbse = c0 * crse if abs(1 - crse) < arthresh else cfa[indx + m1] + (c0 - cfa[indx + m2]) / 2
            rbnw = c0 * crnw if abs(1 - crnw) < arthresh else cfa[indx - m1] + (c0 - cfa[indx - m2]) / 2
            rbne = c0 * crne if abs(1 - crne) < arthresh else cfa[indx + p1] + (c0 - cfa[indx + p2]) / 2
            rbsw = c0 * crsw if abs(1 - crsw) < arthresh else cfa[indx - p1] + (c0 - cfa[indx - p2]) / 2
            wtse = eps + delm[i1] + delm[(indx + m1) >> 1] + delm[(indx + m2) >> 1]
            wtnw = eps + delm[i1] + delm[(indx - m1) >> 1] + delm[(indx - m2) >> 1]
            wtne = eps + delp[i1] + delp[(indx + p1) >> 1] + delp[(indx + p2) >> 1]
            wtsw = eps + delp[i1] + delp[(indx - p1) >> 1] + delp[(indx - p2) >> 1]
            rbm[i1] = (wtse * rbnw + wtnw * rbse) / (wtse + wtnw)
            rbp[i1] = (wtne * rbsw + wtsw * rbne) / (wtne + wtsw)
            A = Dgrbsq1m
            rbvarm = epssq + (ge0 * (A[(indx - v1) >> 1] + A[(indx - 1) >> 1] + A[(indx + 1) >> 1] + A[(indx + v1) >> 1]) +
                              ge1 * (A[(indx - v2 - 1) >> 1] + A[(indx - v2 + 1) >> 1] + A[(indx - 2 - v1) >> 1] + A[(indx + 2 - v1) >> 1] +
                                     A[(indx - 2 + v1) >> 1] + A[(indx + 2 + v1) >> 1] + A[(indx + v2 - 1) >> 1] + A[(indx + v2 + 1) >> 1]))
            A = Dgrbsq1p
            rbvarp = epssq + (ge0 * (A[(indx - v1) >> 1] + A[(indx - 1) >> 1] + A[(indx + 1) >> 1] + A[(indx + v1) >> 1]) +
                              ge1 * (A[(indx - v2 - 1) >> 1] + A[(indx - v2 + 1) >> 1] + A[(indx - 2 - v1) >> 1] + A[(indx + 2 - v1) >> 1] +
                                     A[(indx - 2 + v1) >> 1] + A[(indx + 2 + v1) >> 1] + A[(indx + v2 - 1) >> 1] + A[(indx + v2 + 1) >> 1]))
            pmwt[i1] = rbvarm / (rbvarp + rbvarm)
            if rbp[i1] < c0:
                if 2 * rbp[i1] < c0:
                    rbp[i1] = _median3(rbp[i1], cfa[indx - p1], cfa[indx + p1])
                else:
                    pwt = 2 * (c0 - rbp[i1]) / (eps + rbp[i1] + c0)
                    rbp[i1] = pwt * rbp[i1] + (1 - pwt) * _median3(rbp[i1], cfa[indx - p1], cfa[indx + p1])
            if rbm[i1] < c0:
                if 2 * rbm[i1] < c0:
                    rbm[i1] = _median3(rbm[i1], cfa[indx - m1], cfa[indx + m1])
                else:
                    mwt = 2 * (c0 - rbm[i1]) / (eps + rbm[i1] + c0)
                    rbm[i1] = mwt * rbm[i1] + (1 - mwt) * _median3(rbm[i1], cfa[indx - m1], cfa[indx + m1])
            if rbp[i1] > clip_pt:
                rbp[i1] = _median3(rbp[i1], cfa[indx - p1], cfa[indx + p1])
            if rbm[i1] > clip_pt:
                rbm[i1] = _median3(rbm[i1], cfa[indx - m1], cfa[indx + m1])
    for rr in range(10, rr1 - 10):
        for cc in range(10 + (pat[(rr & 1) * 2] & 1), cc1 - 10, 2):
            indx = rr * ts + cc; i1 = indx >> 1
            pmwtalt = (pmwt[(indx - m1) >> 1] + pmwt[(indx + p1) >> 1] + pmwt[(indx - p1) >> 1] + pmwt[(indx + m1) >> 1]) / 4
            if abs(0.5 - pmwt[i1]) < abs(0.5 - pmwtalt):
                pmwt[i1] = pmwtalt
            rbint[i1] = (cfa[indx] + rbm[i1] * (1 - pmwt[i1]) + rbp[i1] * pmwt[i1]) / 2   # R+B, interpolated
    for rr in range(12, rr1 - 12):
        for cc in range(12 + (pat[(rr & 1) * 2] & 1), cc1 - 12, 2):
            indx = rr * ts + cc; i1 = indx >> 1
            if abs(0.5 - pmwt[indx >> 1]) < abs(0.5 - hvwt[indx >> 1]):
                continue
            rb = rbint[i1]
            cru = cfa[indx - v1] * 2 / (eps + rb + rbint[i1 - v1]); crd = cfa[indx + v1] * 2 / (eps + rb + rbint[i1 + v1])
            crl = cfa[indx - 1] * 2 / (eps + rb + rbint[i1 - 1]); crr = cfa[indx + 1] * 2 / (eps + rb + rbint[i1 + 1])
            gu = rb * cru if abs(1 - cru) < arthresh else cfa[indx - v1] + (rb - rbint[i1 - v1]) / 2
            gd = rb * crd if abs(1 - crd) < arthresh else cfa[indx + v1] + (rb - rbint[i1 + v1]) / 2
            gl = rb * crl if abs(1 - crl) < arthresh else cfa[indx - 1] + (rb - rbint[i1 - 1]) / 2
            gr = rb * crr if abs(1 - crr) < arthresh else cfa[indx + 1] + (rb - rbint[i1 + 1]) / 2
            Gintv = (dirwts0[indx - v1] * gd + dirwts0[indx + v1] * gu) / (dirwts0[indx + v1] + dirwts0[indx - v1])
            Ginth = (dirwts1[indx - 1] * gr + dirwts1[indx + 1] * gl) / (dirwts1[indx - 1] + dirwts1[indx + 1])
            if Gintv < rb:
                if 2 * Gintv < rb:
                    Gintv = _median3(Gintv, cfa[indx - v1], cfa[indx + v1])
                else:
                    vwt = 2 * (rb - Gintv) / (eps + Gintv + rb)
                    Gintv = vwt * Gintv + (1 - vwt) * _median3(Gintv, cfa[indx - v1], cfa[indx + v1])
            if Ginth < rb:
                if 2 * Ginth < rb:
                    Ginth = _median3(Ginth, cfa[indx - 1], cfa[indx + 1])
                else:
                    hwt = 2 * (rb - Ginth) / (eps + Ginth + rb)
                    Ginth = hwt * Ginth + (1 - hwt) * _median3(Ginth, cfa[indx - 1], cfa[indx + 1])
            if Ginth > clip_pt:
                Ginth = _median3(Ginth, cfa[indx - 1], cfa[indx + 1])
            if Gintv > clip_pt:
                Gintv = _median3(Gintv, cfa[indx - v1], cfa[indx + v1])
            rgbgreen[indx] = Ginth * (1 - hvwt[i1]) + Gintv * hvwt[i1]
            Dgrb0[indx >> 1] = rgbgreen[indx] - cfa[indx]
    # chrominance: split G-B from G-R, then interpolate each diagonally
    for rr in range(13 - ey, rr1 - 12, 2):
        for i1 in range((rr * ts + 13 - ex) >> 1, (rr * ts + cc1 - 12) >> 1):
            Dgrb1[i1] = Dgrb0[i1]; Dgrb0[i1] = 0
    for rr in range(14, rr1 - 14):
        cc0 = 14 + (pat[(rr & 1) * 2] & 1)
        c = 1 - (pat[(rr & 1) * 2 + (cc0 & 1)] >> 1)
        D = Dgrb1 if c == 1 else Dgrb0
        for cc in range(cc0, cc1 - 14, 2):
            indx = rr * ts + cc
            nw1 = D[(indx - m1) >> 1]; se1 = D[(indx + m1) >> 1]; ne1 = D[(indx + p1) >> 1]; sw1 = D[(indx - p1) >> 1]
            nw3 = D[(indx - m3) >> 1]; se3 = D[(indx + m3) >> 1]; ne3 = D[(indx + p3) >> 1]; sw3 = D[(indx - p3) >> 1]
            wtnw = 1 / (eps + abs(nw1 - se1) + abs(nw1 - nw3) + abs(se1 - nw3))
            wtne = 1 / (eps + abs(ne1 - sw1) + abs(ne1 - ne3) + abs(sw1 - ne3))
            wtsw = 1 / (eps + abs(sw1 - ne1) + abs(sw1 - se3) + abs(ne1 - sw3))
            wtse = 1 / (eps + abs(se1 - nw1) + abs(se1 - sw3) + abs(nw1 - se3))
            D[indx >> 1] = (wtnw * (1.325 * nw1 - 0.175 * nw3 - 0.075 * D[(indx - m1 - 2) >> 1] - 0.075 * D[(indx - m1 - v2) >> 1]) +
                            wtne * (1.325 * ne1 - 0.175 * ne3 - 0.075 * D[(indx + p1 + 2) >> 1] - 0.075 * D[(indx + p1 + v2) >> 1]) +
                            wtsw * (1.325 * sw1 - 0.175 * sw3 - 0.075 * D[(indx - p1 - 2) >> 1] - 0.075 * D[(indx - p1 - v2) >> 1]) +
                            wtse * (1.325 * se1 - 0.175 * se3 - 0.075 * D[(indx + m1 + 2) >> 1] - 0.075 * D[(indx + m1 + v2) >> 1])) / (wtnw + wtne + wtsw + wtse)
    # write the tile's centre
    for rr in range(16, rr1 - 16):
        row = rr + top
        if row < 0 or row >= H:
            continue
        odd = (pat[(rr & 1) * 2] & 1) == 1
        for cc in range(16, cc1 - 16):
            col = left + cc
            indx = rr * ts + cc
            at_g = ((cc & 1) == 0) == odd
            if at_g:   # R, B at a G site from the four neighbours' colour differences
                hu = hvwt[(indx - v1) >> 1]; hr = hvwt[(indx + 1) >> 1]; hl = hvwt[(indx - 1) >> 1]; hd = hvwt[(indx + v1) >> 1]
                temp = 1 / (hu + 2 - hr - hl + hd)
                out[row, col, 0] = max(0.0, rgbgreen[indx] - (hu * Dgrb0[(indx - v1) >> 1] + (1 - hr) * Dgrb0[(indx + 1) >> 1] + (1 - hl) * Dgrb0[(indx - 1) >> 1] + hd * Dgrb0[(indx + v1) >> 1]) * temp)
                out[row, col, 2] = max(0.0, rgbgreen[indx] - (hu * Dgrb1[(indx - v1) >> 1] + (1 - hr) * Dgrb1[(indx + 1) >> 1] + (1 - hl) * Dgrb1[(indx - 1) >> 1] + hd * Dgrb1[(indx + v1) >> 1]) * temp)
            else:
                out[row, col, 0] = max(0.0, rgbgreen[indx] - Dgrb0[indx >> 1])
                out[row, col, 2] = max(0.0, rgbgreen[indx] - Dgrb1[indx >> 1])
            if 0 <= col < W:
                out[row, col, 1] = max(0.0, rgbgreen[indx])


@njit(parallel=True, cache=_CACHE)
def _amaze(cfa_in, pat):
    H, W = cfa_in.shape
    out = np.zeros((H, W, 3), np.float32)
    ts = 160
    tops = np.arange(-16, H, ts - 32)
    lefts = np.arange(-16, W, ts - 32)
    nt = tops.size * lefts.size
    for k in prange(nt):
        _amaze_tile(cfa_in, W, H, pat, tops[k // lefts.size], lefts[k % lefts.size], out)
    return out


def demosaic_amaze(mosaic, pat):
    """AMaZE. mosaic: float32 H x W white-balanced, clip point 1. pat: 4 colour ids."""
    return _amaze(np.ascontiguousarray(mosaic, np.float32), np.asarray(pat, np.int64))


# ───────────────────────────────────────────── RCD
@njit(cache=_CACHE)
def _rcd_tile(cfa_in, W, H, pat, tr, tc, numTh, numTw, out):
    tileBorder = 9; rcdBorder = 9; TS = 194; TSN = TS - 2 * tileBorder
    w1 = TS; w2 = 2 * TS; w3 = 3 * TS; w4 = 4 * TS
    eps = 1e-5; epssq = 1e-10
    rowStart = tr * TSN; rowEnd = min(rowStart + TS, H)
    if rowStart + tileBorder == rowEnd - tileBorder:
        return
    colStart = tc * TSN; colEnd = min(colStart + TS, W)
    if colStart + tileBorder == colEnd - tileBorder:
        return
    tileRows = min(rowEnd - rowStart, TS); tilecols = min(colEnd - colStart, TS)
    n = TS * TS
    cfa = np.zeros(n, np.float32)
    rgb = np.zeros((3, n), np.float32)
    VH = np.zeros(n, np.float32)
    PQ = np.zeros(n // 2, np.float32)
    lpf = PQ   # shared, as in RawTherapee: the low-pass is used up before the P/Q weights are written
    Phpf = np.zeros(n // 2, np.float32); Qhpf = np.zeros(n // 2, np.float32)
    V0 = np.zeros(TS, np.float32); V1 = np.zeros(TS, np.float32); V2 = np.zeros(TS, np.float32)
    bufH = np.zeros(TS, np.float32)
    for row in range(rowStart, rowEnd):
        c0 = pat[(row & 1) * 2 + (colStart & 1)]; c1 = pat[(row & 1) * 2 + ((colStart + 1) & 1)]
        indx = (row - rowStart) * TS
        for col in range(colStart, colEnd):
            v = _clamp01(cfa_in[row, col])
            cfa[indx] = v; rgb[c0, indx] = v; rgb[c1, indx] = v
            indx += 1
    # Step 1: vertical/horizontal discrimination
    for row in range(3, min(tileRows - 3, 5)):
        bv = V0 if row == 3 else V1
        for col in range(4, tilecols - 4):
            indx = row * TS + col
            t = (cfa[indx - w3] - cfa[indx - w1] - cfa[indx + w1] + cfa[indx + w3]) - 3 * (cfa[indx - w2] + cfa[indx + w2]) + 6 * cfa[indx]
            bv[col - 4] = t * t
    for row in range(4, tileRows - 4):
        for col in range(3, tilecols - 3):
            indx = row * TS + col
            t = (cfa[indx - 3] - cfa[indx - 1] - cfa[indx + 1] + cfa[indx + 3]) - 3 * (cfa[indx - 2] + cfa[indx + 2]) + 6 * cfa[indx]
            bufH[col - 3] = t * t
        for col in range(4, tilecols - 4):
            indx = (row + 1) * TS + col
            t = (cfa[indx - w3] - cfa[indx - w1] - cfa[indx + w1] + cfa[indx + w3]) - 3 * (cfa[indx - w2] + cfa[indx + w2]) + 6 * cfa[indx]
            V2[col - 4] = t * t
        for col in range(4, tilecols - 4):
            indx = row * TS + col
            Vs = max(epssq, V0[col - 4] + V1[col - 4] + V2[col - 4])
            Hs = max(epssq, bufH[col - 4] + bufH[col - 3] + bufH[col - 2])
            VH[indx] = Vs / (Vs + Hs)
        t_ = V0; V0 = V2; V2 = t_   # swap(V0, V2)
        t_ = V0; V0 = V1; V1 = t_   # swap(V0, V1)
    # Step 2: low pass filter
    for row in range(2, tileRows - 2):
        col = 2 + (pat[(row & 1) * 2] & 1)
        indx = row * TS + col; lp = indx >> 1
        while col < tilecols - 2:
            lpf[lp] = cfa[indx] + 0.5 * (cfa[indx - w1] + cfa[indx + w1] + cfa[indx - 1] + cfa[indx + 1]) + \
                0.25 * (cfa[indx - w1 - 1] + cfa[indx - w1 + 1] + cfa[indx + w1 - 1] + cfa[indx + w1 + 1])
            col += 2; indx += 2; lp += 1
    # Step 3: green at red/blue
    for row in range(4, tileRows - 4):
        col = 4 + (pat[(row & 1) * 2] & 1)
        indx = row * TS + col; lp = indx >> 1
        while col < tilecols - 4:
            ci = cfa[indx]
            N = eps + (abs(cfa[indx - w1] - cfa[indx + w1]) + abs(ci - cfa[indx - w2])) + (abs(cfa[indx - w1] - cfa[indx - w3]) + abs(cfa[indx - w2] - cfa[indx - w4]))
            S = eps + (abs(cfa[indx - w1] - cfa[indx + w1]) + abs(ci - cfa[indx + w2])) + (abs(cfa[indx + w1] - cfa[indx + w3]) + abs(cfa[indx + w2] - cfa[indx + w4]))
            Wg = eps + (abs(cfa[indx - 1] - cfa[indx + 1]) + abs(ci - cfa[indx - 2])) + (abs(cfa[indx - 1] - cfa[indx - 3]) + abs(cfa[indx - 2] - cfa[indx - 4]))
            E = eps + (abs(cfa[indx - 1] - cfa[indx + 1]) + abs(ci - cfa[indx + 2])) + (abs(cfa[indx + 1] - cfa[indx + 3]) + abs(cfa[indx + 2] - cfa[indx + 4]))
            l = lpf[lp]
            NE_ = cfa[indx - w1] * (l + l) / (eps + l + lpf[lp - w1])
            SE_ = cfa[indx + w1] * (l + l) / (eps + l + lpf[lp + w1])
            WE_ = cfa[indx - 1] * (l + l) / (eps + l + lpf[lp - 1])
            EE_ = cfa[indx + 1] * (l + l) / (eps + l + lpf[lp + 1])
            Vest = (S * NE_ + N * SE_) / (N + S)
            Hest = (Wg * EE_ + E * WE_) / (E + Wg)
            vc = VH[indx]; vn = 0.25 * ((VH[indx - w1 - 1] + VH[indx - w1 + 1]) + (VH[indx + w1 - 1] + VH[indx + w1 + 1]))
            disc = vn if abs(0.5 - vc) < abs(0.5 - vn) else vc
            rgb[1, indx] = disc * (Hest - Vest) + Vest
            col += 2; indx += 2; lp += 1
    # Step 4.0: P/Q diagonal high pass
    for row in range(3, tileRows - 3):
        col = 3; indx = row * TS + col; i2 = indx >> 1
        while col < tilecols - 3:
            t = (cfa[indx - w3 - 3] - cfa[indx - w1 - 1] - cfa[indx + w1 + 1] + cfa[indx + w3 + 3]) - 3 * (cfa[indx - w2 - 2] + cfa[indx + w2 + 2]) + 6 * cfa[indx]
            Phpf[i2] = t * t
            t = (cfa[indx - w3 + 3] - cfa[indx - w1 + 1] - cfa[indx + w1 - 1] + cfa[indx + w3 - 3]) - 3 * (cfa[indx - w2 + 2] + cfa[indx + w2 - 2]) + 6 * cfa[indx]
            Qhpf[i2] = t * t
            col += 2; indx += 2; i2 += 1
    # Step 4.1: P/Q discrimination
    for row in range(4, tileRows - 4):
        col = 4 + (pat[(row & 1) * 2] & 1)
        indx = row * TS + col; i2 = indx >> 1; i3 = (indx - w1 - 1) >> 1; i4 = (indx + w1 - 1) >> 1
        while col < tilecols - 4:
            Ps = max(epssq, Phpf[i3] + Phpf[i2] + Phpf[i4 + 1])
            Qs = max(epssq, Qhpf[i3 + 1] + Qhpf[i2] + Qhpf[i4])
            PQ[i2] = Ps / (Ps + Qs)
            col += 2; indx += 2; i2 += 1; i3 += 1; i4 += 1
    # Step 4.2: red/blue at blue/red
    for row in range(4, tileRows - 4):
        col0 = 4 + (pat[(row & 1) * 2] & 1)
        ci = 2 - pat[(row & 1) * 2 + (col0 & 1)]
        col = col0; indx = row * TS + col; pq = indx >> 1; pq2 = (indx - w1 - 1) >> 1; pq3 = (indx + w1 - 1) >> 1
        while col < tilecols - 4:
            pc = PQ[pq]; pn = 0.25 * (PQ[pq2] + PQ[pq2 + 1] + PQ[pq3] + PQ[pq3 + 1])
            disc = pn if abs(0.5 - pc) < abs(0.5 - pn) else pc
            NW = eps + abs(rgb[ci, indx - w1 - 1] - rgb[ci, indx + w1 + 1]) + abs(rgb[ci, indx - w1 - 1] - rgb[ci, indx - w3 - 3]) + abs(rgb[1, indx] - rgb[1, indx - w2 - 2])
            NE = eps + abs(rgb[ci, indx - w1 + 1] - rgb[ci, indx + w1 - 1]) + abs(rgb[ci, indx - w1 + 1] - rgb[ci, indx - w3 + 3]) + abs(rgb[1, indx] - rgb[1, indx - w2 + 2])
            SW = eps + abs(rgb[ci, indx - w1 + 1] - rgb[ci, indx + w1 - 1]) + abs(rgb[ci, indx + w1 - 1] - rgb[ci, indx + w3 - 3]) + abs(rgb[1, indx] - rgb[1, indx + w2 - 2])
            SE = eps + abs(rgb[ci, indx - w1 - 1] - rgb[ci, indx + w1 + 1]) + abs(rgb[ci, indx + w1 + 1] - rgb[ci, indx + w3 + 3]) + abs(rgb[1, indx] - rgb[1, indx + w2 + 2])
            NWe = rgb[ci, indx - w1 - 1] - rgb[1, indx - w1 - 1]; NEe = rgb[ci, indx - w1 + 1] - rgb[1, indx - w1 + 1]
            SWe = rgb[ci, indx + w1 - 1] - rgb[1, indx + w1 - 1]; SEe = rgb[ci, indx + w1 + 1] - rgb[1, indx + w1 + 1]
            Pe = (NW * SEe + SE * NWe) / (NW + SE); Qe = (NE * SWe + SW * NEe) / (NE + SW)
            rgb[ci, indx] = rgb[1, indx] + disc * (Qe - Pe) + Pe
            col += 2; indx += 2; pq += 1; pq2 += 1; pq3 += 1
    # Step 4.3: red and blue at green
    for row in range(4, tileRows - 4):
        col = 4 + (pat[(row & 1) * 2 + 1] & 1)
        indx = row * TS + col
        while col < tilecols - 4:
            vc = VH[indx]; vn = 0.25 * ((VH[indx - w1 - 1] + VH[indx - w1 + 1]) + (VH[indx + w1 - 1] + VH[indx + w1 + 1]))
            disc = vn if abs(0.5 - vc) < abs(0.5 - vn) else vc
            g1 = rgb[1, indx]
            N1 = eps + abs(g1 - rgb[1, indx - w2]); S1 = eps + abs(g1 - rgb[1, indx + w2])
            W1 = eps + abs(g1 - rgb[1, indx - 2]); E1 = eps + abs(g1 - rgb[1, indx + 2])
            gm = rgb[1, indx - w1]; gp = rgb[1, indx + w1]; gl = rgb[1, indx - 1]; gr = rgb[1, indx + 1]
            for c in (0, 2):
                SN = abs(rgb[c, indx - w1] - rgb[c, indx + w1]); EW = abs(rgb[c, indx - 1] - rgb[c, indx + 1])
                Ng = N1 + SN + abs(rgb[c, indx - w1] - rgb[c, indx - w3])
                Sg = S1 + SN + abs(rgb[c, indx + w1] - rgb[c, indx + w3])
                Wg = W1 + EW + abs(rgb[c, indx - 1] - rgb[c, indx - 3])
                Eg = E1 + EW + abs(rgb[c, indx + 1] - rgb[c, indx + 3])
                Ne = rgb[c, indx - w1] - gm; Se = rgb[c, indx + w1] - gp; We = rgb[c, indx - 1] - gl; Ee = rgb[c, indx + 1] - gr
                Ve = (Ng * Se + Sg * Ne) / (Ng + Sg); He = (Eg * We + Wg * Ee) / (Eg + Wg)
                rgb[c, indx] = g1 + disc * (He - Ve) + Ve
            col += 2; indx += 2
    fv = rowStart + (rcdBorder if tr == 0 else tileBorder); lv = rowEnd - (rcdBorder if tr == numTh - 1 else tileBorder)
    fh = colStart + (rcdBorder if tc == 0 else tileBorder); lh = colEnd - (rcdBorder if tc == numTw - 1 else tileBorder)
    for row in range(fv, lv):
        idx = (row - rowStart) * TS + fh - colStart
        for col in range(fh, lh):
            out[row, col, 0] = max(0.0, rgb[0, idx]); out[row, col, 1] = max(0.0, rgb[1, idx]); out[row, col, 2] = max(0.0, rgb[2, idx])
            idx += 1


@njit(parallel=True, cache=_CACHE)
def _rcd(cfa_in, pat):
    H, W = cfa_in.shape
    out = np.zeros((H, W, 3), np.float32)
    TSN = 194 - 18
    numTh = (H + TSN - 1) // TSN; numTw = (W + TSN - 1) // TSN
    for k in prange(numTh * numTw):
        _rcd_tile(cfa_in, W, H, pat, k // numTw, k % numTw, numTh, numTw, out)
    _border_interpolate(cfa_in, pat, 2, out, 9)
    return out


def demosaic_rcd(mosaic, pat):
    """RCD (Ratio Corrected Demosaicing)."""
    return _rcd(np.ascontiguousarray(mosaic, np.float32), np.asarray(pat, np.int64))


# ───────────────────────────────────────────── bilinear / border
@njit(cache=_CACHE)
def _border_interpolate(cfa_in, pat, P, out, border):
    """Same-colour average over the 3x3 neighbourhood (RawTherapee's border_interpolate);
    border = 0 => the whole image (bilinear)."""
    H, W = cfa_in.shape
    s = np.zeros(3); cnt = np.zeros(3, np.int64)
    for y in range(H):
        inside = y >= border and y < H - border
        x = 0
        while x < W:
            if inside and x == border and border > 0:
                x = W - border
                continue
            s[:] = 0; cnt[:] = 0
            for dy in range(-1, 2):
                yy = y + dy
                if yy < 0 or yy >= H:
                    continue
                for dx in range(-1, 2):
                    xx = x + dx
                    if xx < 0 or xx >= W:
                        continue
                    c = pat[(yy % P) * P + (xx % P)]
                    s[c] += _clamp01(cfa_in[yy, xx]); cnt[c] += 1
            own = pat[(y % P) * P + (x % P)]
            for c in range(3):
                if c == own:
                    out[y, x, c] = _clamp01(cfa_in[y, x])
                else:
                    out[y, x, c] = s[c] / cnt[c] if cnt[c] else 0.0
            x += 1


@njit(parallel=True, cache=_CACHE)
def _bilinear(cfa_in, pat):
    H, W = cfa_in.shape
    out = np.zeros((H, W, 3), np.float32)
    for y in prange(H):
        for x in range(W):
            s0 = 0.0; s1 = 0.0; s2 = 0.0; n0 = 0; n1 = 0; n2 = 0
            for dy in range(-1, 2):
                yy = y + dy
                if yy < 0 or yy >= H:
                    continue
                for dx in range(-1, 2):
                    xx = x + dx
                    if xx < 0 or xx >= W:
                        continue
                    c = pat[(yy & 1) * 2 + (xx & 1)]
                    v = _clamp01(cfa_in[yy, xx])
                    if c == 0:
                        s0 += v; n0 += 1
                    elif c == 1:
                        s1 += v; n1 += 1
                    else:
                        s2 += v; n2 += 1
            own = pat[(y & 1) * 2 + (x & 1)]
            v = _clamp01(cfa_in[y, x])
            out[y, x, 0] = v if own == 0 else (s0 / n0 if n0 else 0.0)
            out[y, x, 1] = v if own == 1 else (s1 / n1 if n1 else 0.0)
            out[y, x, 2] = v if own == 2 else (s2 / n2 if n2 else 0.0)
    return out


def demosaic_bilinear(mosaic, pat):
    """Bilinear: the same-colour mean of each pixel's 3x3 neighbourhood."""
    return _bilinear(np.ascontiguousarray(mosaic, np.float32), np.asarray(pat, np.int64))


# ───────────────────────────────────────────── Malvar-He-Cutler
@njit(inline='always')
def _reflect(i, n):
    if n == 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i
        if i >= n:
            i = 2 * n - 2 - i
    return i


@njit(parallel=True, cache=_CACHE)
def _malvar(m, pat, clip):
    H, W = m.shape
    out = np.empty((H, W, 3), np.float32)
    for y in prange(H):
        ym2 = _reflect(y - 2, H); ym1 = _reflect(y - 1, H); yp1 = _reflect(y + 1, H); yp2 = _reflect(y + 2, H)
        for x in range(W):
            xm2 = _reflect(x - 2, W); xm1 = _reflect(x - 1, W); xp1 = _reflect(x + 1, W); xp2 = _reflect(x + 2, W)
            c = m[y, x]; N = m[ym1, x]; S = m[yp1, x]; Wv = m[y, xm1]; E = m[y, xp1]
            N2 = m[ym2, x]; S2 = m[yp2, x]; W2 = m[y, xm2]; E2 = m[y, xp2]
            D = m[ym1, xm1] + m[ym1, xp1] + m[yp1, xm1] + m[yp1, xp1]
            col = pat[(y & 1) * 2 + (x & 1)]
            if col == 1:
                g = c
                hor = (5 * c + 4 * (Wv + E) - (W2 + E2) - D + 0.5 * (N2 + S2)) / 8
                ver = (5 * c + 4 * (N + S) - (N2 + S2) - D + 0.5 * (W2 + E2)) / 8
                if pat[(y & 1) * 2 + ((x + 1) & 1)] == 0:
                    r = hor; b = ver
                else:
                    r = ver; b = hor
            else:
                gg = (4 * c + 2 * (N + S + Wv + E) - (N2 + S2 + W2 + E2)) / 8
                opp = (6 * c + 2 * D - 1.5 * (N2 + S2 + W2 + E2)) / 8
                g = gg
                if col == 0:
                    r = c; b = opp
                else:
                    b = c; r = opp
            if clip:
                out[y, x, 0] = _clamp01(r); out[y, x, 1] = _clamp01(g); out[y, x, 2] = _clamp01(b)
            else:
                out[y, x, 0] = r; out[y, x, 1] = g; out[y, x, 2] = b
    return out


def demosaic_malvar(mosaic, pat, clip=True):
    """Malvar-He-Cutler (2004) 5x5 kernels, mirror boundaries (as colour_demosaicing, used by RawForge)."""
    return _malvar(np.ascontiguousarray(mosaic, np.float32), np.asarray(pat, np.int64), clip)


DEMOSAICS = {'amaze': demosaic_amaze, 'rcd': demosaic_rcd, 'mhc': demosaic_malvar, 'bilinear': demosaic_bilinear}
