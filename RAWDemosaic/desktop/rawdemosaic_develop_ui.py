# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""RAW Demosaic desktop — the Develop tab: the web app's Develop panel in tkinter.

Rendering runs on its own thread (the newest request wins, so sliders never queue up); the engine
(rawdemosaic_develop.Develop) is guarded by a lock that exports and AI tools also take. Settings use the web
app's structure, and settings files are interchangeable with it.
"""
import base64
import copy
import hashlib
import io
import json
import math
import os
import threading
import time
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

import rawdemosaic_develop as V
import rawdemosaic_colour as CM
import rawdemosaic_engine as E

ALT = 0x20000 if os.name == 'nt' else 0x8   # the Alt key's bit in a Tk event's state

APP = 'RAW Demosaic'
ASPECTS = [('orig', 'Original'), ('free', 'Free'), ('1', '1 : 1'), ('1.5', '3 : 2'), ('0.6667', '2 : 3'), ('1.3333', '4 : 3'), ('0.75', '3 : 4'),
           ('1.25', '5 : 4'), ('0.8', '4 : 5'), ('1.7778', '16 : 9'), ('0.5625', '9 : 16')]
AI_MODELS = [('lama', 'LaMa (best general)'), ('mst', 'MST (buildings, straight lines)'), ('migan', 'MI-GAN (fast)')]
TOOL_HINTS = {
    'crop': 'Drag the corners or edges of the frame, or drag inside it to move it. Enter (or Done) applies; Esc cancels.',
    'straighten': 'Draw a line along something that should be level (a horizon) or upright (a wall edge). Esc cancels.',
    'wb': 'Click something that should be neutral (white or grey). Esc cancels.',
    'spot': 'Click a blemish to remove it; drag the circles to adjust; Delete or double-click removes the selected spot. Esc or Done finishes.',
    'brush': 'Paint the mask (Erase removes). Middle-drag or Space-drag pans; [ and ] change the size; Esc finishes.',
    'fill': 'Click an area to select everything of a similar colour joined to it. Erase mode or Alt subtracts. Esc finishes.',
    'lasso': 'Drag round the area to select (hold Alt to subtract). Middle-drag pans; Esc finishes.',
    'mlasso': 'Click along the edge of the area: the outline snaps to edges. Click the first point, double-click or Enter closes (Alt subtracts); Backspace removes the last point; Esc cancels.',
    'poly': 'Click the corners of the area (straight sides). Click the first point, double-click or Enter closes (Alt subtracts); Backspace removes the last point; Esc cancels.',
    'rect': 'Drag a rectangle over the area (hold Alt to subtract). Middle-drag pans; Esc finishes.',
    'grad': 'Drag from where the adjustment is full to where it fades out. Drag the end circles to change angle and feather, the middle circle to move, '
            'the centre line to rotate, an outer line to change the feather. Delete or double-click the middle removes it. Esc finishes.',
    'sam': 'Click the object to select it; right-click (or Alt-click) excludes; drag for a box; Shift-click starts another selection. Esc finishes.',
    'guides': 'Drag along an edge to draw a guide (up to four), drag an end to adjust it, double-click a guide to delete it. Enter (or Done) applies; Esc cancels.',
}
SV_TOOLS = ('guides', 'spot', 'brush', 'sam', 'lasso', 'mlasso', 'poly', 'rect', 'grad', 'fill')   # shown on the uncorrected picture
RETOUCH_TOOLS = ('spot', 'brush', 'sam', 'lasso', 'mlasso', 'poly', 'rect', 'grad', 'fill')
MASK_TOOLS = ('brush', 'lasso', 'mlasso', 'poly', 'rect', 'grad', 'sam', 'fill')
LASSOS = ('lasso', 'mlasso', 'poly')
TONE_RANGE = {k: (-200, 200) for k in ('highlights', 'shadows', 'midtones')}   # wider than the other sliders; the same strength per step
GROUPS = {
    'tone': ('Light & white balance', ['temp', 'tint', 'exposure', 'contrast', 'highlights', 'shadows', 'midtones', 'whites', 'blacks', 'brightness'], True),
    'presence': ('Presence', ['dehaze', 'vibrance', 'saturation', 'clarity', 'localContrast', 'micro'], True),
    'colour': ('Curves, equalizer, mixer, grading', ['curves', 'eq', 'hsl', 'grading'], True),
    'detail': ('Sharpening', ['usmAmount', 'usmRadius', 'usmThreshold'], True),
    'lens': ('Lens corrections, chromatic aberration', ['lens', 'ca'], True),
    'geometry': ('Crop, rotation, perspective, outpainting', ['geometry', 'aspect', 'outpaint'], False),
    'retouch': ('Spot removal and AI removals', ['retouch'], False),
    'masks': ('Masks and local adjustments', ['masks'], False),
}
BR_MIN, BR_RANGE = 0.0002, 1000


def identity_curve():
    return [[0, 0], [1, 1]]


def dev_defaults():
    return dict(
        geometry=dict(rot90=0, angle=0, crop=dict(x=0, y=0, w=1, h=1), auto=True, keepEmpty=False, persp=dict(guides=[], vertical=0, horizontal=0, aspect=0)), aspect='orig',
        lens=dict(key='', dist=True, tca=True, vig=True, distAmt=100, vigAmt=100, focal=0, aperture=0, crop=0),
        temp=0, tint=0, exposure=0, contrast=15, highlights=0, shadows=0, midtones=0, whites=0, blacks=0, brightness=0,
        dehaze=0, vibrance=0, saturation=15, clarity=0, localContrast=0, micro=0,
        hsl=dict(hue=[0] * 8, sat=[0] * 8, lum=[0] * 8),
        grading=dict(shadows=dict(hue=230, sat=0), midtones=dict(hue=60, sat=0), highlights=dict(hue=70, sat=0), balance=0),
        curves=dict(m=identity_curve(), r=identity_curve(), g=identity_curve(), b=identity_curve()),
        eq=dict(hue=[0] * 8, sat=[0] * 8, bri=[0] * 8),
        usmAmount=0, usmRadius=1, usmThreshold=2,
        retouch=[], masks=[], outpaint=None,
        ca=dict(auto=None, red=0, blue=0, purple=0, purpleHue=315, green=0, greenHue=145),
    )


DEFAULTS = dev_defaults()


def get_p(path, o):
    for k in path: o = o[k]
    return o


def set_p(path, v, o):
    for k in path[:-1]: o = o[k]
    o[path[-1]] = v


def new_id():
    return os.urandom(4).hex()


def frame_of(region, src_hw, flip, is_rgb):
    """Where the developed image sits in the camera frame (display orientation), from the processed region."""
    if region is None: return dict(x=0, y=0, w=1, h=1)
    y0, x0, h, w = region
    H, W = src_hw
    pts = []
    for x, y in ((x0, y0), (x0 + w, y0 + h)):
        c, r = x / W, y / H
        if flip & 1: c = 1 - c
        if flip & 2: r = 1 - r
        pts.append((r, c) if flip & 4 else (c, r))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return dict(x=min(xs), y=min(ys), w=max(xs) - min(xs), h=max(ys) - min(ys))


# ───────────────────────────────────────────── widgets
class Slider:
    """Label (double-click resets) + scale + number box. on_change(live) while dragging, on_commit(label) at the end."""
    def __init__(self, ui, parent, label, get, set_, lo, hi, step, dec=0, unit='', default=0, hist=None, swatch=None, title='', bar=None):
        self.ui, self.get, self.set, self.lo, self.hi, self.step, self.dec, self.unit = ui, get, set_, lo, hi, step, dec, unit
        self.default, self.hist = default, hist or label
        C = ui.C
        row = ttk.Frame(parent, style='Card.TFrame'); row.pack(fill='x', pady=1)
        self.row = row
        if swatch:
            tk.Label(row, bg=swatch, width=2, height=1, relief='flat').pack(side='left', padx=(0, 4))
        self.lab = ttk.Label(row, text=label, style='Card.TLabel', width=15 if not swatch else 12)
        self.lab.pack(side='left')
        self.var = tk.DoubleVar(value=0)
        self.bar, self.barc = bar, None   # bar(v): the colour at value v, shown in a strip under the scale
        sp = row
        if bar:
            sp = ttk.Frame(row, style='Card.TFrame'); sp.pack(side='left', fill='x', expand=True, padx=4)
        self.scale = ttk.Scale(sp, from_=lo, to=hi, variable=self.var, orient='horizontal', command=self._moved)
        if not bar:
            self.scale.pack(side='left', fill='x', expand=True, padx=4)
        else:
            self.scale.pack(fill='x')
            self.barc = tk.Canvas(sp, height=7, bg=C['card'], highlightthickness=0, bd=0)
            self.barc.pack(fill='x')
            self.barc.bind('<Configure>', lambda e: self.draw_bar())
        self.num = tk.StringVar()
        e = ttk.Entry(row, textvariable=self.num, width=7, justify='right')
        e.pack(side='left')
        e.bind('<Return>', self._typed); e.bind('<FocusOut>', self._typed)
        self.lab.bind('<Double-Button-1>', lambda ev: self._reset())
        self.scale.bind('<ButtonRelease-1>', lambda ev: self._commit())
        self._loading = False
        self._dragged = False
        self.histf = None   # a function giving the history label, when it depends on other settings
        if title: ui.tip(self.lab, title + '. Double-click to reset.')
        else: ui.tip(self.lab, 'Double-click to reset.')

    def fmt(self, v):
        return f'{v:.{self.dec}f}'

    def _q(self, v):
        v = round(round(v / self.step) * self.step, 6)
        return max(self.lo, min(self.hi, v))

    def _moved(self, _):
        if self._loading or not self.ui.ready(): return
        v = self._q(self.var.get())
        if v == self.get(): return
        self.set(v); self._dragged = True
        self.num.set(self.fmt(v)); self._style(v); self._mark()
        self.ui.changed(self)

    def _commit(self):
        if not self._dragged: return
        self._dragged = False
        v = self.get()
        if self.histf: self.hist = self.histf()
        self.ui.commit(f'{self.hist} {"+" if self.lo < 0 and v > 0 else ""}{self.fmt(v)}{self.unit}')

    def _typed(self, _=None):
        if not self.ui.ready(): return
        try: v = self._q(float(self.num.get()))
        except ValueError: self.sync(); return
        if v == self.get(): self.sync(); return
        self.set(v); self.sync(); self.ui.changed(self)
        self.ui.commit(f'{self.hist} {"+" if self.lo < 0 and v > 0 else ""}{self.fmt(v)}{self.unit}')

    def _reset(self):
        if not self.ui.ready(): return
        self.set(self.default); self.sync(); self.ui.changed(self); self.ui.commit(f'{self.hist} reset')

    def _style(self, v):
        self.lab.configure(foreground=self.ui.C['accent'] if v != self.default else self.ui.C['text'])

    def sync(self):
        self._loading = True
        try:
            v = self.get() if self.ui.params is not None else self.default
            self.var.set(v); self.num.set(self.fmt(v)); self._style(v)
        finally:
            self._loading = False
        self.draw_bar()

    def _bar_x(self, v):
        """Canvas x of value v: where the scale puts its thumb centre (the trough ends are inset by half a thumb)."""
        try:
            a, b = self.scale.coords(self.lo)[0], self.scale.coords(self.hi)[0]
        except Exception:
            a, b = 0, 0
        w = self.barc.winfo_width()
        if b - a < 4: a, b = 8, w - 8
        return a + (b - a) * (v - self.lo) / (self.hi - self.lo)

    def draw_bar(self):
        cv = self.barc
        if cv is None or cv.winfo_width() < 10: return
        cv.delete('all')
        n = 48
        for i in range(n):
            v0 = self.lo + (self.hi - self.lo) * i / n; v1 = self.lo + (self.hi - self.lo) * (i + 1) / n
            x0, x1 = self._bar_x(v0), self._bar_x(v1)
            cv.create_rectangle(x0, 1, x1 + 1, 7, fill=self.bar((v0 + v1) / 2), outline='')
        cv.create_polygon(0, 0, 0, 0, 0, 0, fill=self.ui.C['text'], outline=self.ui.C['card'], tags='mk')
        self._mark()

    def _mark(self):
        if self.barc is None: return
        v = self.get() if self.ui.params is not None else self.default
        x = self._bar_x(v)
        self.barc.coords('mk', x - 4, 0, x + 4, 0, x, 6)


class Section:
    """A collapsible card; with `key`, an on/off tick that switches the panel's adjustments off."""
    def __init__(self, ui, parent, title, key=None, open_=False):
        C = ui.C
        self.ui, self.key = ui, key
        self.frame = ttk.Frame(parent, style='Card.TFrame', padding=(8, 6))
        self.frame.pack(fill='x', pady=(0, 6))
        head = ttk.Frame(self.frame, style='Card.TFrame'); head.pack(fill='x')
        self.on = tk.BooleanVar(value=True)
        if key:
            ttk.Checkbutton(head, variable=self.on, style='Card.TCheckbutton', command=self._toggled).pack(side='left')
            ui.tip(head.winfo_children()[-1], "Turn this panel's adjustments on or off")
        self.btn = tk.Label(head, text=('▾ ' if open_ else '▸ ') + title, bg=C['card'], fg=C['accent'], font=('Segoe UI', 10, 'bold'), cursor='hand2')
        self.btn.pack(side='left', fill='x')
        self.btn.bind('<Button-1>', lambda e: self.toggle())
        self.title = title
        self.body = ttk.Frame(self.frame, style='Card.TFrame')
        self.is_open = open_
        if open_: self.body.pack(fill='x', pady=(4, 0))
        self.on_open = None

    def toggle(self):
        self.is_open = not self.is_open
        self.btn.configure(text=('▾ ' if self.is_open else '▸ ') + self.title)
        if self.is_open:
            self.body.pack(fill='x', pady=(4, 0))
            if self.on_open: self.on_open()
        else: self.body.pack_forget()

    def _toggled(self):
        self.ui.section_switched(self)


class CurveEditor:
    TABS = [('m', 'RGB'), ('r', 'Red'), ('g', 'Green'), ('b', 'Blue'), ('eq:hue', 'Hue'), ('eq:sat', 'Saturation'), ('eq:bri', 'Brightness')]
    HINTS = {'m': 'Tone curve: click to add a point, drag to move it, double-click a point to remove it.', 'r': 'Red channel curve.',
             'g': 'Green channel curve.', 'b': 'Blue channel curve.',
             'eq:hue': 'Colour equalizer — hue: shift each hue towards its neighbours. Drag the nodes; double-click resets one.',
             'eq:sat': 'Colour equalizer — saturation of each hue. Drag the nodes; double-click resets one.',
             'eq:bri': 'Colour equalizer — brightness of each hue (e.g. darken a blue sky). Double-click resets a node.'}

    def __init__(self, ui, parent):
        self.ui, C = ui, ui.C
        self.tab = 'm'; self.drag = None
        tabs = ttk.Frame(parent, style='Card.TFrame'); tabs.pack(fill='x')
        self.tab_btns = {}
        for k, l in self.TABS:
            b = tk.Label(tabs, text=l, bg=C['card2'], fg=C['text'], padx=6, pady=2, cursor='hand2')
            b.pack(side='left', padx=(0, 2)); b.bind('<Button-1>', lambda e, k=k: self.set_tab(k)); self.tab_btns[k] = b
        self.cv = tk.Canvas(parent, height=int(220 * ui.scale), bg=C['input'], highlightthickness=0)
        self.cv.pack(fill='x', pady=(4, 2))
        self.hint = ttk.Label(parent, text='', style='Muted.TLabel', wraplength=int(440 * ui.scale), justify='left'); self.hint.pack(anchor='w')
        ui.link(parent, 'Reset this curve', self.reset)
        self.cv.bind('<Configure>', lambda e: self.draw())
        self.cv.bind('<ButtonPress-1>', self.press); self.cv.bind('<B1-Motion>', self.move); self.cv.bind('<ButtonRelease-1>', self.release)
        self.cv.bind('<Double-Button-1>', self.dbl); self.cv.bind('<Motion>', self.hover)
        self.set_tab('m')

    def set_tab(self, k):
        self.tab = k
        for kk, b in self.tab_btns.items(): b.configure(bg=self.ui.C['accent'] if kk == k else self.ui.C['card2'], fg='#111' if kk == k else self.ui.C['text'])
        self.hint.configure(text=self.HINTS[k]); self.draw()

    def frame(self):
        W, H = max(50, self.cv.winfo_width()), max(50, self.cv.winfo_height())
        pad = 8; eq = self.tab.startswith('eq:'); top, bot = pad, H - pad - (14 if eq else 0)
        X = lambda u: pad + u * (W - 2 * pad); U = lambda x: (x - pad) / (W - 2 * pad)
        Y = (lambda v: (top + bot) / 2 - v * (bot - top) / 2) if eq else (lambda v: bot - v * (bot - top))
        Vv = (lambda y: ((top + bot) / 2 - y) / ((bot - top) / 2)) if eq else (lambda y: (bot - y) / (bot - top))
        return W, H, pad, eq, X, U, Y, Vv

    def draw(self):
        cv, C, p = self.cv, self.ui.C, self.ui.params
        cv.delete('all')
        if p is None: return
        W, H, pad, eq, X, U, Y, Vv = self.frame()
        if eq:
            vals = p['eq'][self.tab[3:]]; fn = V.periodic_curve(vals)
            for i in range(0, 360, 4): cv.create_rectangle(X(i / 360), H - pad - 10, X((i + 4) / 360) + 1, H - pad, fill=V.oklab_hex(0.72, 0.13, i), width=0)
            for v in (-1, -0.5, 0.5, 1): cv.create_line(pad, Y(v), W - pad, Y(v), fill=C['border'])
            cv.create_line(pad, Y(0), W - pad, Y(0), fill=C['muted'])
            pts = [c for h in range(0, 361, 3) for c in (X(h / 360), Y(max(-1, min(1, fn(h)))))]
            cv.create_line(*pts, fill=C['text'], width=2, smooth=False)
            for i, h in enumerate(V.BAND_HUES):
                x, y = X(h / 360), Y(vals[i]); cv.create_oval(x - 6, y - 6, x + 6, y + 6, fill=V.oklab_hex(0.7, 0.14, h), outline=C['text'])
            return
        hist = self.ui.hist
        if hist is not None:
            h = hist[{'m': 3, 'r': 0, 'g': 1, 'b': 2}[self.tab]].astype(float)
            mx = max(1, h[1:255].max())
            poly = [X(0), Y(0)] + [c for i in range(256) for c in (X(i / 255), Y(min(1, h[i] / mx)))] + [X(1), Y(0)]
            cv.create_polygon(*poly, fill=C['card2'], outline='')
        for i in range(1, 4):
            cv.create_line(X(i / 4), Y(0), X(i / 4), Y(1), fill=C['border']); cv.create_line(X(0), Y(i / 4), X(1), Y(i / 4), fill=C['border'])
        cv.create_rectangle(X(0), Y(1), X(1), Y(0), outline=C['border'])
        cv.create_line(X(0), Y(0), X(1), Y(1), fill=C['border'], dash=(3, 4))
        pts = p['curves'][self.tab]; fn = V.monotone_curve(pts)
        col = {'m': C['text'], 'r': '#e53935', 'g': '#43a047', 'b': '#1e88e5'}[self.tab]
        u = np.linspace(0, 1, 201); y = np.clip(fn(u), 0, 1)
        cv.create_line(*[c for a, b in zip(u, y) for c in (X(a), Y(b))], fill=col, width=2)
        for q in pts: cv.create_rectangle(X(q[0]) - 4, Y(q[1]) - 4, X(q[0]) + 4, Y(q[1]) + 4, fill=col, outline='')

    def nearest(self, x, y):
        W, H, pad, eq, X, U, Y, Vv = self.frame(); p = self.ui.params
        if eq:
            best, bd = -1, 16
            for i, h in enumerate(V.BAND_HUES):
                d = abs(X(h / 360) - x)
                if d < bd: bd, best = d, i
            return best
        for i, q in enumerate(p['curves'][self.tab]):
            if math.hypot(X(q[0]) - x, Y(q[1]) - y) < 10: return i
        return -1

    def press(self, e):
        if not self.ui.ready(): return
        W, H, pad, eq, X, U, Y, Vv = self.frame()
        i = self.nearest(e.x, e.y)
        if not eq and i < 0:
            pts = self.ui.params['curves'][self.tab]; u = min(1, max(0, U(e.x)))
            if any(abs(q[0] - u) < 0.02 for q in pts): return
            pts.append([u, min(1, max(0, Vv(e.y)))]); pts.sort(key=lambda q: q[0])
            i = next(k for k, q in enumerate(pts) if q[0] == u)
        if i < 0: return
        self.drag = i; self.move(e)

    def move(self, e):
        if self.drag is None: return
        W, H, pad, eq, X, U, Y, Vv = self.frame(); p = self.ui.params
        if eq: p['eq'][self.tab[3:]][self.drag] = round(max(-1, min(1, Vv(e.y))) * 100) / 100
        else:
            pts = p['curves'][self.tab]; i = self.drag
            lo = 0 if i == 0 else pts[i - 1][0] + 0.01; hi = 1 if i == len(pts) - 1 else pts[i + 1][0] - 0.01
            pts[i] = [min(hi, max(lo, U(e.x))), min(1, max(0, Vv(e.y)))]
        self.draw(); self.ui.changed()

    def release(self, e):
        if self.drag is None: return
        i = self.drag; self.drag = None
        name = dict(self.TABS)[self.tab]
        self.ui.commit(f'Equalizer {name.lower()}: {V.BAND_NAMES[i]}' if self.tab.startswith('eq:') else f'Curve ({name})')

    def dbl(self, e):
        if not self.ui.ready(): return
        i = self.nearest(e.x, e.y)
        if i < 0: return
        p = self.ui.params; name = dict(self.TABS)[self.tab]
        if self.tab.startswith('eq:'):
            p['eq'][self.tab[3:]][i] = 0; self.draw(); self.ui.changed(); self.ui.commit(f'Equalizer {name.lower()}: {V.BAND_NAMES[i]} reset')
        else:
            pts = p['curves'][self.tab]
            if 0 < i < len(pts) - 1:
                del pts[i]; self.draw(); self.ui.changed(); self.ui.commit(f'Curve ({name}): point removed')

    def hover(self, e):
        if self.drag is not None or self.ui.params is None: return
        W, H, pad, eq, X, U, Y, Vv = self.frame(); u = min(1, max(0, U(e.x)))
        if eq: self.hint.configure(text=f'Hue {round(u * 360)}°: {V.periodic_curve(self.ui.params["eq"][self.tab[3:]])(u * 360) * 100:.0f}%')
        else: self.hint.configure(text=f'Input {round(u * 255)} → output {round(min(1, max(0, float(V.monotone_curve(self.ui.params["curves"][self.tab])(u)))) * 255)}')

    def reset(self):
        if not self.ui.ready(): return
        p = self.ui.params; name = dict(self.TABS)[self.tab]
        if self.tab.startswith('eq:'): p['eq'][self.tab[3:]] = [0] * 8
        else: p['curves'][self.tab] = identity_curve()
        self.draw(); self.ui.changed(); self.ui.commit(f'{name} {"equalizer" if self.tab.startswith("eq:") else "curve"} reset')


# ───────────────────────────────────────────── the tab
class DevelopTab:
    def __init__(self, app, nb):
        self.app, self.C, self.scale = app, app.C if hasattr(app, 'C') else None, app.ui_scale
        import rawdemosaic_desktop as D
        self.C = D.C
        self.root = app.root
        self.dv = None; self.lock = threading.RLock()
        self.params = None; self.history = []; self.h_index = 0
        self.info = {}; self.img_w = self.img_h = 0; self.frame = dict(x=0, y=0, w=1, h=1); self.frame_diag = 0
        self.bitmaps, self.fills, self.op_layers, self.rasters = {}, {}, {}, {}
        self.sel_mask = None; self.sel_spot = -1; self.tool = None; self.tool_snap = None
        self.brush = dict(size=0.03, feather=0.5, flow=1.0, erase=False); self.spot = dict(size=0.012, feather=0.4, mode='heal')
        self.show_overlay = False; self.sam = None; self.sam_prompt = None; self.lasso = None; self.lw = None; self.snap = None; self.grad_sel = None
        self.zoom = 'fit'; self.z = 1.0; self.cx = self.cy = 0.0; self.before = False; self.space = False
        self.hist = None; self.lens_db = None; self.camera = None; self.lens_cands = []
        self.sliders = []; self.sections = {}; self.clip = None
        self._lut_memo = [None, None, None, None]; self._persp = [None, None, '']
        self._req = None; self._req_ev = threading.Event(); self._shown = None; self._photo = None; self._drag = None
        self._last_line = None; self._timing = ''; self.rendering = False
        self._build(nb)
        threading.Thread(target=self._render_loop, daemon=True).start()

    # ─── helpers
    def ready(self):
        return self.dv is not None and self.params is not None

    def tip(self, w, text):
        pass

    def link(self, parent, text, cmd):
        l = tk.Label(parent, text=text, bg=self.C['card'], fg=self.C['info'], cursor='hand2', font=('Segoe UI', 9, 'underline'))
        l.pack(anchor='w'); l.bind('<Button-1>', lambda e: cmd())
        return l

    def btn(self, parent, text, cmd, side='left', accent=False):
        b = ttk.Button(parent, text=text, command=cmd, style='Accent.TButton' if accent else 'Small.TButton')
        b.pack(side=side, padx=(0, 4), pady=1)
        return b

    def row(self, parent):
        r = ttk.Frame(parent, style='Card.TFrame'); r.pack(fill='x', pady=2)
        return r

    def muted(self, parent, text, wrap=440):
        l = ttk.Label(parent, text=text, style='Muted.TLabel', wraplength=int(wrap * self.scale), justify='left'); l.pack(anchor='w', pady=(2, 0))
        return l

    # ─── layout
    def _build(self, nb):
        app = self.app
        tab = ttk.Frame(nb, padding=(0, 8, 0, 0))
        nb.add(tab, text='  4 · Develop  ')
        self.tab = tab
        left, pinned = app._scroll_column(tab, width=500)
        right = ttk.Frame(tab); right.pack(side='left', fill='both', expand=True, padx=(10, 0))
        self._outer, self._right = left.master.master, right
        C = self.C
        # top: open / histogram / history
        c = app._card(left, 'Develop')
        r = self.row(c)
        self.btn(r, 'Open image…', self.open_image)
        self.btn(r, 'Use selected result', self.use_result)
        self.on_right = tk.BooleanVar(value=bool(app.settings.get('dev_right', False)))
        ttk.Checkbutton(r, text='Controls on the right', variable=self.on_right, style='Card.TCheckbutton',
                        command=self.place_controls).pack(side='right')
        self.title_lbl = self.muted(c, 'Process a RAW (tab 1) — the result opens here — or open a TIFF/JPEG/PNG.')
        self.hist_cv = tk.Canvas(c, height=int(70 * self.scale), bg='#15171b', highlightthickness=0); self.hist_cv.pack(fill='x', pady=(6, 2))
        self.hist_cv.bind('<Configure>', lambda e: self.draw_hist())
        r = self.row(c)
        self.undo_btn = self.btn(r, '↶ Undo', lambda: self.goto(self.h_index - 1))
        self.redo_btn = self.btn(r, '↷ Redo', lambda: self.goto(self.h_index + 1))
        self.btn(r, 'Reset all', self.reset_all)
        self.timing = ttk.Label(r, text='', style='Muted.TLabel'); self.timing.pack(side='right')
        # sections
        s = self.sec(left, 'history', 'History', open_=True)
        self.hist_list = app._scrolled_listbox(s.body, 5)
        self.hist_list.bind('<<ListboxSelect>>', lambda e: self._hist_clicked())
        s = self.sec(left, 'settings', 'Copy, paste, save settings')
        r = self.row(s.body)
        for t, cmd in (('Copy', self.copy_settings), ('Paste', self.paste_settings), ('Save…', self.save_settings), ('Load…', self.load_settings)): self.btn(r, t, cmd)
        self.groups = {}
        for k, (label, keys, on) in GROUPS.items():
            v = tk.BooleanVar(value=on); self.groups[k] = v
            ttk.Checkbutton(s.body, text=label, variable=v, style='Card.TCheckbutton').pack(anchor='w')
        self.muted(s.body, 'Same format as the web version. AI removals and outpainting are saved as recipes and recreated when applied.')
        self._build_crop(left); self._build_lens(left); self._build_ca(left); self._build_transform(left); self._build_spots(left); self._build_masks(left)
        self._build_basic(left); self._build_presence(left)
        s = self.sec(left, 'curves', 'Curves & colour equalizer', key='curves')
        self.curve = CurveEditor(self, s.body)
        s.on_open = self.curve.draw
        s = self.sec(left, 'mixer', 'Colour mixer', key='mixer')
        r = self.row(s.body); ttk.Label(r, text='Adjust', style='Card.TLabel', width=8).pack(side='left')
        self.mixer_mode = tk.StringVar(value='Hue')
        cb = ttk.Combobox(r, textvariable=self.mixer_mode, state='readonly', values=['Hue', 'Saturation', 'Luminance'], width=12); cb.pack(side='left')
        cb.bind('<<ComboboxSelected>>', lambda e: self._sync_mixer())
        self.mixer = []
        for i, b in enumerate(V.BAND_NAMES):
            mk = lambda i=i: {'Hue': 'hue', 'Saturation': 'sat', 'Luminance': 'lum'}[self.mixer_mode.get()]
            self.mixer.append(self.slider(s.body, b, None, -100, 100, 1, get=lambda i=i, mk=mk: self.params['hsl'][mk()][i],
                                          set_=lambda v, i=i, mk=mk: self.params['hsl'][mk()].__setitem__(i, v), swatch=V.oklab_hex(0.7, 0.14, V.BAND_HUES[i]),
                                          hist=None, histf=lambda i=i: f'{self.mixer_mode.get()}: {V.BAND_NAMES[i]}', bar=lambda v, i=i: self.mixer_colour(i, v)))
        s = self.sec(left, 'grading', 'Colour grading', key='grading')
        for k, l in (('shadows', 'Shadows'), ('midtones', 'Midtones'), ('highlights', 'Highlights')):
            self.slider(s.body, f'{l} hue', ['grading', k, 'hue'], 0, 360, 1, unit='°')
            self.slider(s.body, f'{l} amount', ['grading', k, 'sat'], 0, 100, 1)
        self.slider(s.body, 'Balance', ['grading', 'balance'], -100, 100, 1)
        s = self.sec(left, 'sharpen', 'Sharpening (unsharp mask)', key='sharpen')
        self.slider(s.body, 'Amount', ['usmAmount'], 0, 300, 1, unit='%', hist='Sharpen amount')
        self.slider(s.body, 'Radius', ['usmRadius'], 0.3, 5, 0.1, dec=1, unit=' px', hist='Sharpen radius')
        self.slider(s.body, 'Threshold', ['usmThreshold'], 0, 30, 1, hist='Sharpen threshold')
        self._build_sr(left)
        self._build_proof(left)
        self.muted(left, 'Double-click a slider\'s name to reset it. Microcontrast and sharpening work at pixel scale: judge them at 100%. '
                         'Hold Before (or \\) to compare; Ctrl+Z / Ctrl+Y undo and redo.')
        # pinned: super-resolution and export
        import rawdemosaic_desktop as D
        sr = ttk.Frame(pinned, style='Card.TFrame', padding=(8, 4)); sr.pack(fill='x', pady=(4, 0))
        ttk.Checkbutton(sr, text='Upscale', variable=self.sr_on, style='Card.TCheckbutton').pack(side='left')
        ttk.Combobox(sr, textvariable=self.sr_target, state='readonly', width=22, values=[v for _, v in D.SR_TARGETS]).pack(side='left', padx=6)
        self.sr_models_lbl = ttk.Label(sr, text='', style='Muted.TLabel'); self.sr_models_lbl.pack(side='left')
        for v in (self.sr_on, self.sr_target, self.sr_down): v.trace_add('write', lambda *a: self.refresh_sr())
        self.conv_lbl = ttk.Label(pinned, text='', style='Muted.TLabel')
        r = ttk.Frame(pinned); r.pack(fill='x', pady=(4, 0))
        self.save_tif_btn = ttk.Button(r, text='Save TIFF…', style='Accent.TButton', command=lambda: self.export('tiff'))
        self.save_tif_btn.pack(side='left', fill='x', expand=True)
        self.save_jpg_btn = ttk.Button(r, text='Save JPEG…', command=lambda: self.export('jpeg'))
        self.save_jpg_btn.pack(side='left', padx=(4, 0))
        # right: view toolbar + canvas
        bar = ttk.Frame(right); bar.pack(fill='x', pady=(0, 4))
        self.view_name = ttk.Label(bar, text='', style='Tag.TLabel'); self.view_name.pack(side='left')
        self.before_btn = ttk.Button(bar, text='Before', style='Small.TButton'); self.before_btn.pack(side='right')
        self.before_btn.bind('<ButtonPress-1>', lambda e: self.set_before(True)); self.before_btn.bind('<ButtonRelease-1>', lambda e: self.set_before(False))
        ttk.Checkbutton(bar, text='Soft proof', variable=self.proof_on, command=self._proof_changed).pack(side='right', padx=(0, 8))
        self.zoom_lbl = ttk.Label(bar, text='Fit', style='Tag.TLabel', width=6); self.zoom_lbl.pack(side='right', padx=6)
        self.zoom_var = tk.IntVar(value=0)
        self.ZOOMS = ['fit', 0.1, 0.17, 0.25, 0.33, 0.5, 0.67, 1, 1.5, 2, 3, 4]
        zs = ttk.Scale(bar, from_=0, to=len(self.ZOOMS) - 1, variable=self.zoom_var, orient='horizontal', length=int(180 * self.scale),
                       command=lambda v: self.set_zoom(self.ZOOMS[int(round(float(v)))]))
        zs.pack(side='right')
        self.hint = ttk.Label(right, text='', style='Status.TLabel', wraplength=900); self.hint.pack(fill='x')
        pc = app._card(right, None, fill='both', expand=True)
        self.cv = tk.Canvas(pc, bg=C['checker'], highlightthickness=0, cursor='arrow')
        self.cv.pack(fill='both', expand=True)
        self._bind_view()
        self.sync_all()
        if self.on_right.get(): self.place_controls()
        self.refresh_sr(); self._proof_note()

    def place_controls(self):
        """Develop's controls on the left (as the other tabs) or on the right (as the web version)."""
        outer, right = self._outer, self._right
        outer.pack_forget(); right.pack_forget()
        if self.on_right.get():
            outer.pack(side='right', fill='y')
            right.pack(side='left', fill='both', expand=True, padx=(0, 10))
        else:
            outer.pack(side='left', fill='y')
            right.pack(side='left', fill='both', expand=True, padx=(10, 0))
        self.app.settings['dev_right'] = self.on_right.get()

    # ─── super-resolution on export (Develop's own, separate from tab 3's for Process and Blend files)
    def _build_sr(self, left):
        import rawdemosaic_desktop as D
        dsr = self.app.settings.get('dev_sr') or {}
        self.sr_on = tk.BooleanVar(value=bool(dsr.get('enabled', False)))
        self.sr_target = tk.StringVar(value=D.label_of(D.SR_TARGETS, str(dsr.get('target', '2'))))
        self.sr_down = tk.StringVar(value=D.label_of(D.DOWNSCALE, dsr.get('downscale', 'lanczos')))
        self.sr_rows = [(m['model'], float(m['weight'])) for m in dsr.get('models', [])]
        self.sr_each = tk.BooleanVar(value=bool(dsr.get('save_each', False)))
        s = self.sec(left, 'sr', 'Super-resolution (on export)')
        b = s.body
        ttk.Checkbutton(b, text='Upscale the developed image when saving', variable=self.sr_on, style='Card.TCheckbutton').pack(anchor='w')
        ttk.Checkbutton(b, text="Also save each model's own result (…_sr_<model>), when blending several", variable=self.sr_each,
                        style='Card.TCheckbutton', command=self.refresh_sr).pack(anchor='w')
        for lab, var, pairs in (('Output size', self.sr_target, D.SR_TARGETS), ('Scale models by', self.sr_down, D.DOWNSCALE)):
            r = self.row(b); ttk.Label(r, text=lab, style='Card.TLabel', width=15).pack(side='left')
            ttk.Combobox(r, textvariable=var, state='readonly', values=[v for _, v in pairs]).pack(side='left', fill='x', expand=True)
        self.muted(b, 'Several models can be blended: each is scaled to the output size first (e.g. Real-ESRGAN ×4 and RealSR ×4 at 50 % each, '
                      'then down to ×2). Weights are relative. Separate from tab 3\'s, which is for Process and Blend files.')
        self.sr_list = self.app._scrolled_listbox(b, 4)
        self.sr_list.bind('<<ListboxSelect>>', lambda e: self._sr_selected())
        r = self.row(b)
        self.sr_pick, self.sr_w = tk.StringVar(), tk.StringVar(value='1')
        self.sr_box = ttk.Combobox(r, textvariable=self.sr_pick, state='readonly'); self.sr_box.pack(side='left', fill='x', expand=True)
        ttk.Label(r, text=' weight', style='Card.TLabel').pack(side='left')
        ttk.Spinbox(r, textvariable=self.sr_w, from_=0, to=100, increment=0.1, width=5).pack(side='left', padx=(4, 0))
        r = self.row(b)
        for t, cmd in (('Add', self.sr_add), ('Update selected', self.sr_update), ('Remove', self.sr_remove)): self.btn(r, t, cmd)

    # ─── soft proofing, and converting exports to a profile (e.g. a lab's printer profile)
    def _build_proof(self, left):
        pf = self.app.settings.get('proof') or {}
        self.proof_on = tk.BooleanVar(value=False)   # starts off: the normal view is the screen's
        self.proof_path = pf.get('path', '')
        self.proof_label = tk.StringVar()
        self.proof_intent = tk.StringVar(value={k: l for k, l, _ in CM.INTENTS}.get(pf.get('intent', 'relative'), 'Relative colorimetric'))
        self.proof_bpc = tk.BooleanVar(value=pf.get('bpc', True))
        self.proof_paper = tk.BooleanVar(value=pf.get('paper', False))
        self.proof_gamut = tk.BooleanVar(value=pf.get('gamut', False))
        self.proof_convert = tk.BooleanVar(value=pf.get('convert', False))
        s = self.sec(left, 'proof', 'Soft proof & convert to a profile')
        b = s.body
        r = self.row(b); ttk.Label(r, text='Profile', style='Card.TLabel', width=15).pack(side='left')
        self.proof_box = ttk.Combobox(r, textvariable=self.proof_label, state='readonly'); self.proof_box.pack(side='left', fill='x', expand=True)
        self.proof_box.bind('<<ComboboxSelected>>', lambda e: self._proof_picked())
        r = self.row(b); ttk.Label(r, text='Rendering intent', style='Card.TLabel', width=15).pack(side='left')
        cb = ttk.Combobox(r, textvariable=self.proof_intent, state='readonly', values=[l for _, l, _ in CM.INTENTS]); cb.pack(side='left', fill='x', expand=True)
        cb.bind('<<ComboboxSelected>>', lambda e: self._proof_changed())
        for text, var in (('Black point compensation', self.proof_bpc),
                          ('Soft proof: show the image as it will print', self.proof_on),
                          ('Simulate paper and ink (the paper\'s white, the ink\'s black)', self.proof_paper),
                          ('Gamut warning: colours the profile cannot reproduce in magenta', self.proof_gamut),
                          ('Convert to this profile when saving (the TIFF and JPEG embed it)', self.proof_convert)):
            ttk.Checkbutton(b, text=text, variable=var, style='Card.TCheckbutton', command=self._proof_changed).pack(anchor='w')
        self.muted(b, 'Perceptual fits every colour into what the printer can make; relative colorimetric keeps the colours it can make exactly and '
                      'clips the rest (use it with black point compensation). Convert when a lab or printer driver asks for files in its profile: '
                      'many labs want sRGB, and give profiles for proofing only. Profiles are listed from the profiles folder next to the app '
                      'and the system\'s colour folder.')
        self._fill_proof_profiles()

    def _fill_proof_profiles(self):
        found = CM.list_profiles(space=None)
        items = CM.labelled([i for i in found if i['cls'] == 'prtr'] + [i for i in found if i['cls'] not in ('prtr', 'scnr')])
        p0 = self.proof_path
        if p0 and not any(os.path.normcase(p) == os.path.normcase(p0) for _, p in items) and CM.info(p0):
            items += CM.labelled([CM.info(p0)])
        self._proof_items = items
        self.proof_box.configure(values=[l for l, _ in items] + ['Browse…'])
        self.proof_label.set(next((l for l, p in items if os.path.normcase(p) == os.path.normcase(p0 or '?')), ''))

    def _proof_picked(self):
        l = self.proof_label.get()
        if l == 'Browse…':
            p = filedialog.askopenfilename(title='Printer or output profile', initialdir=os.path.join(E.APP_ROOT, 'profiles'),
                                           filetypes=[('ICC profiles', '*.icc *.icm'), ('All files', '*.*')])
            if p and CM.info(p) is None: messagebox.showerror(APP, f'{os.path.basename(p)} is not an ICC profile.'); p = ''
            if p: self.proof_path = os.path.normpath(p)
            self._fill_proof_profiles()
        else:
            self.proof_path = dict(self._proof_items).get(l, '')
        self._proof_changed()

    def _intent_key(self):
        return next((k for k, l, _ in CM.INTENTS if l == self.proof_intent.get()), 'relative')

    def _proof_changed(self):
        if (self.proof_on.get() or self.proof_convert.get()) and not self.proof_path:
            self.proof_on.set(False); self.proof_convert.set(False)
            messagebox.showinfo(APP, 'Choose a profile under "Soft proof & convert to a profile" first.')
        if self.proof_convert.get():
            try: CM.check_output_profile(self.proof_path)
            except ValueError as ex: self.proof_convert.set(False); messagebox.showerror(APP, str(ex))
        self.app.settings['proof'] = dict(path=self.proof_path, intent=self._intent_key(), bpc=self.proof_bpc.get(), paper=self.proof_paper.get(),
                                          gamut=self.proof_gamut.get(), convert=self.proof_convert.get())
        self._proof_note(); self.update_view_name(); self.request()

    def _proof_note(self):
        i = CM.info(self.proof_path) if self.proof_path else None
        if self.proof_convert.get() and i:
            self.conv_lbl.configure(text=f'Saving converted to {i["desc"]} ({self.proof_intent.get().lower()})')
            self.conv_lbl.pack(fill='x', pady=(4, 0), before=self.save_tif_btn.master)
        else:
            self.conv_lbl.pack_forget()

    def proof_cfg(self):
        """The soft proof for the view, or None."""
        if not self.proof_on.get() or not self.proof_path or not os.path.isfile(self.proof_path): return None
        return dict(path=self.proof_path, intent=self._intent_key(), bpc=self.proof_bpc.get(), paper=self.proof_paper.get(), gamut=self.proof_gamut.get())

    def refresh_sr_models(self, labels):
        self.sr_box.configure(values=labels)
        if not self.sr_pick.get() and labels:
            self.sr_pick.set(next((l for l in labels if 'RealESRGAN_x4plus' in l), labels[0]))

    def _sr_weight(self):
        try: return max(0.0, float(self.sr_w.get()))
        except ValueError: return 1.0

    def sr_add(self):
        if self.sr_pick.get():
            self.sr_rows.append((self.sr_pick.get(), self._sr_weight())); self.sr_on.set(True); self.refresh_sr()

    def sr_update(self):
        sel = self.sr_list.curselection()
        if sel and self.sr_pick.get():
            self.sr_rows[sel[0]] = (self.sr_pick.get(), self._sr_weight()); self.refresh_sr(); self.sr_list.selection_set(sel[0])

    def sr_remove(self):
        sel = self.sr_list.curselection()
        if sel: del self.sr_rows[sel[0]]; self.refresh_sr()

    def _sr_selected(self):
        sel = self.sr_list.curselection()
        if sel:
            m, w = self.sr_rows[sel[0]]; self.sr_pick.set(m); self.sr_w.set(f'{w:g}')

    def refresh_sr(self):
        """The model list, the summary beside the save buttons, and the remembered settings."""
        import rawdemosaic_desktop as D
        self.sr_list.delete(0, 'end')
        tot = sum(w for _, w in self.sr_rows) or 1
        for m, w in self.sr_rows: self.sr_list.insert('end', f'{100 * w / tot:5.1f} %   {m}')
        used = [m.split('/')[-1] for m, w in self.sr_rows if w > 0]
        self.sr_models_lbl.configure(text='' if not self.sr_on.get() else ' + '.join(used) if used else 'add models under Super-resolution')
        self.app.settings['dev_sr'] = dict(enabled=self.sr_on.get(), target=D.key_of(D.SR_TARGETS, self.sr_target.get(), '2'), save_each=self.sr_each.get(),
                                           downscale=D.key_of(D.DOWNSCALE, self.sr_down.get(), 'lanczos'),
                                           models=[dict(model=m, weight=w) for m, w in self.sr_rows])

    def sr_opts(self):
        d = self.app.settings['dev_sr']
        return dict(enabled=d['enabled'], target=d['target'], downscale=d['downscale'], save_each=d.get('save_each', False),
                    models=[dict(path=self.app.pth_path(m['model']), weight=m['weight']) for m in d['models']])

    def sec(self, parent, name, title, key=None, open_=False):
        s = Section(self, parent, title, key, open_)
        self.sections[name] = s
        return s

    def slider(self, parent, label, path, lo, hi, step, dec=0, unit='', hist=None, get=None, set_=None, swatch=None, title='', histf=None, bar=None):
        g = get or (lambda: get_p(path, self.params))
        st = set_ or (lambda v: set_p(path, v, self.params))
        default = 0 if get else get_p(path, DEFAULTS)
        s = Slider(self, parent, label, g, st, lo, hi, step, dec, unit, default, hist, swatch, title, bar)
        s.path = path
        s.histf = histf
        self.sliders.append(s)
        return s

    def _build_crop(self, left):
        s = self.sec(left, 'crop', 'Crop & straighten', open_=True)
        r = self.row(s.body)
        self.crop_btn = self.btn(r, 'Crop…', lambda: self.end_crop(True) if self.tool == 'crop' else self.set_tool('crop'))
        self.straighten_btn = self.btn(r, 'Straighten…', lambda: self.set_tool(None if self.tool == 'straighten' else 'straighten'))
        self.btn(r, '⟲', lambda: self.rotate90(-1)); self.btn(r, '⟳', lambda: self.rotate90(1))
        r = self.row(s.body); ttk.Label(r, text='Aspect', style='Card.TLabel', width=8).pack(side='left')
        self.aspect = tk.StringVar(value='Original')
        cb = ttk.Combobox(r, textvariable=self.aspect, state='readonly', values=[v for _, v in ASPECTS], width=10); cb.pack(side='left')
        cb.bind('<<ComboboxSelected>>', lambda e: self.aspect_changed())
        self.btn(r, 'Reset', self.crop_reset)
        self.slider(s.body, 'Angle', ['geometry', 'angle'], -45, 45, 0.05, dec=2, unit='°', title='Positive turns the picture clockwise')
        self.keep_empty = tk.BooleanVar()
        ttk.Checkbutton(s.body, text="Keep empty areas (don't crop them out: fill them with Outpaint)", variable=self.keep_empty,
                        style='Card.TCheckbutton', command=self.keep_empty_changed).pack(anchor='w')
        r = self.row(s.body)
        self.outpaint_btn = self.btn(r, 'Outpaint empty areas', self.outpaint)
        self.outpaint_note = ttk.Label(r, text='', style='Muted.TLabel'); self.outpaint_note.pack(side='left')

    def _build_lens(self, left):
        s = self.sec(left, 'lens', 'Lens corrections', key='lens')
        s.on_open = lambda: self.detect_lens(False) if self.ready() and self.lens_db is None else None
        self.lens_status = self.muted(s.body, 'Uses the Lensfun database of lens profiles.')
        self.lens_sel = tk.StringVar()
        self.lens_box = ttk.Combobox(s.body, textvariable=self.lens_sel, state='readonly'); self.lens_box.pack(fill='x', pady=2)
        self.lens_box.bind('<<ComboboxSelected>>', lambda e: self.lens_chosen())
        r = self.row(s.body); ttk.Label(r, text='Search', style='Card.TLabel', width=8).pack(side='left')
        self.lens_search = tk.StringVar(); e = ttk.Entry(r, textvariable=self.lens_search); e.pack(side='left', fill='x', expand=True)
        self.lens_search.trace_add('write', lambda *a: self.render_lens_ui())
        r = self.row(s.body)
        self.lens_vars = {}
        for k, l, w in (('focal', 'Focal (mm)', 6), ('aperture', 'f/', 5), ('crop', 'Crop factor', 5)):
            ttk.Label(r, text=l, style='Card.TLabel').pack(side='left')
            v = tk.StringVar(); self.lens_vars[k] = v
            en = ttk.Entry(r, textvariable=v, width=w); en.pack(side='left', padx=(2, 8))
            en.bind('<Return>', lambda e, k=k: self.lens_num(k)); en.bind('<FocusOut>', lambda e, k=k: self.lens_num(k))
        r = self.row(s.body)
        self.lens_chk = {}
        for k, l in (('dist', 'Distortion'), ('tca', 'Colour fringes'), ('vig', 'Vignetting')):
            v = tk.BooleanVar(value=True); self.lens_chk[k] = v
            ttk.Checkbutton(r, text=l, variable=v, style='Card.TCheckbutton', command=lambda k=k, l=l: self.lens_check(k, l)).pack(side='left')
        self.slider(s.body, 'Distortion', ['lens', 'distAmt'], 0, 200, 1, unit='%', hist='Distortion amount', title='100% is the full Lensfun correction')
        self.slider(s.body, 'Vignetting', ['lens', 'vigAmt'], 0, 200, 1, unit='%', hist='Vignetting amount')

    def _build_ca(self, left):
        s = self.sec(left, 'ca', 'Chromatic aberration', key='ca')
        self.ca_auto = tk.BooleanVar()
        ttk.Checkbutton(s.body, text='Remove chromatic aberration (automatic: measures the colour fringes)', variable=self.ca_auto,
                        style='Card.TCheckbutton', command=self.ca_toggled).pack(anchor='w')
        self.ca_note = self.muted(s.body, '')
        self.slider(s.body, 'Red / cyan', ['ca', 'red'], -100, 100, 1, hist='CA red/cyan')
        self.slider(s.body, 'Blue / yellow', ['ca', 'blue'], -100, 100, 1, hist='CA blue/yellow')
        ttk.Label(s.body, text='Defringe (purple and green edges)', style='Sub.TLabel').pack(anchor='w', pady=(4, 0))
        self.slider(s.body, 'Purple amount', ['ca', 'purple'], 0, 100, 1, hist='Defringe purple')
        self.slider(s.body, 'Purple hue', ['ca', 'purpleHue'], 250, 360, 1, unit='°', hist='Defringe purple hue')
        self.slider(s.body, 'Green amount', ['ca', 'green'], 0, 100, 1, hist='Defringe green')
        self.slider(s.body, 'Green hue', ['ca', 'greenHue'], 90, 200, 1, unit='°', hist='Defringe green hue')

    def _build_transform(self, left):
        s = self.sec(left, 'transform', 'Transform (perspective)', key='transform')
        r = self.row(s.body)
        self.guides_btn = self.btn(r, 'Guides…', lambda: self.end_guides(True) if self.tool == 'guides' else self.set_tool('guides'))
        self.btn(r, 'Clear guides', self.clear_guides)
        self.guides_hint = self.muted(s.body, 'Draw up to four guides on edges that should be vertical or horizontal: one line levels the picture; two verticals make '
                                              'converging verticals parallel and upright (two horizontals likewise); verticals and horizontals together correct both.')
        self.slider(s.body, 'Vertical', ['geometry', 'persp', 'vertical'], -100, 100, 1, title='Keystone: right widens the top')
        self.slider(s.body, 'Horizontal', ['geometry', 'persp', 'horizontal'], -100, 100, 1)
        self.slider(s.body, 'Aspect', ['geometry', 'persp', 'aspect'], -100, 100, 1)

    def _build_spots(self, left):
        s = self.sec(left, 'spots', 'Spot removal', key='spots')
        r = self.row(s.body)
        self.spots_btn = self.btn(r, 'Spots…', lambda: self.set_tool(None if self.tool == 'spot' else 'spot'))
        self.spot_count = ttk.Label(r, text='', style='Muted.TLabel'); self.spot_count.pack(side='left')
        r = self.row(s.body); ttk.Label(r, text='Mode', style='Card.TLabel', width=8).pack(side='left')
        self.spot_mode = tk.StringVar(value='Heal (match surroundings)')
        cb = ttk.Combobox(r, textvariable=self.spot_mode, state='readonly', values=['Heal (match surroundings)', 'Clone (exact copy)']); cb.pack(side='left', fill='x', expand=True)
        cb.bind('<<ComboboxSelected>>', lambda e: self.spot_mode_changed())
        self.spot_size = self.tool_slider(s.body, 'Size (% diag.)', 0.05, 10, 0.05, lambda: (self.cur_spot() or {}).get('r', self.spot['size']) * 100,
                                          lambda v: self.spot_set('r', v / 100), 'Spot size', dec=2)
        self.spot_feather = self.tool_slider(s.body, 'Feather (%)', 2, 100, 1, lambda: (self.cur_spot() or {}).get('feather', self.spot['feather']) * 100,
                                             lambda v: self.spot_set('feather', v / 100), 'Spot feather')
        r = self.row(s.body)
        self.btn(r, 'Delete selected', lambda: self.delete_spot(self.sel_spot)); self.btn(r, 'Clear all', self.clear_spots)

    def tool_slider(self, parent, label, lo, hi, step, get, set_, hist, dec=0):
        """A slider for a tool setting (not a develop parameter)."""
        s = Slider(self, parent, label, get, set_, lo, hi, step, dec, '', get(), hist)
        s.default = None
        s._style = lambda v: None
        s._reset = lambda: None
        s._commit = lambda: (setattr(s, '_dragged', False), self.commit(hist) if self.cur_spot_changed() else None)
        return s

    def _build_masks(self, left):
        s = self.sec(left, 'masks', 'Masks, local adjustments & AI removal', key='masks')
        b = s.body
        r = self.row(b)
        self.mask_sel = tk.StringVar()
        self.mask_box = ttk.Combobox(r, textvariable=self.mask_sel, state='readonly'); self.mask_box.pack(side='left', fill='x', expand=True)
        self.mask_box.bind('<<ComboboxSelected>>', lambda e: self.mask_selected())
        self.btn(r, 'New', self.new_mask_cmd); self.btn(r, 'Delete', self.delete_mask)
        r = self.row(b)
        self.tool_btns = {}
        for k, l in (('brush', 'Brush'), ('erase', 'Erase'), ('sam', 'Select (SAM2)'), ('fill', 'Fill'), (None, None),
                     ('lasso', 'Lasso'), ('mlasso', 'Magnetic'), ('poly', 'Polygon'), ('rect', 'Rectangle'), ('grad', 'Gradient')):
            if k is None: r = self.row(b); continue
            self.tool_btns[k] = self.btn(r, l, lambda k=k: self.mask_tool(k))
        r = self.row(b)
        self.overlay_var = tk.BooleanVar()
        ttk.Checkbutton(r, text='Show the mask', variable=self.overlay_var, style='Card.TCheckbutton', command=self.overlay_toggled).pack(side='left')
        self.brush_size = Slider(self, b, 'Brush size', lambda: math.log(self.brush['size'] / BR_MIN) / math.log(BR_RANGE) * 100,
                                 lambda v: self.set_brush_size(BR_MIN * BR_RANGE ** (v / 100)), 0, 100, 0.5, 0, '', 0, 'Brush size')
        self._tool_only(self.brush_size)
        self.brush_size_lbl = ttk.Label(b, text='', style='Muted.TLabel'); self.brush_size_lbl.pack(anchor='e')
        for key, lab, lo, hi, st, f in (('feather', 'Brush feather %', 0, 100, 1, 100), ('flow', 'Flow %', 5, 100, 1, 100)):
            sl = Slider(self, b, lab, lambda key=key, f=f: self.brush[key] * f, lambda v, key=key, f=f: self.brush.__setitem__(key, v / f), lo, hi, st, 0, '', 0, lab)
            self._tool_only(sl)
        self.fill_tol = tk.IntVar(value=12); self.lasso_feather = tk.DoubleVar(value=2)
        for lab, var, lo, hi in (('Fill tolerance', self.fill_tol, 0, 100), ('Edge feather px', self.lasso_feather, 0, 60)):
            r = self.row(b); ttk.Label(r, text=lab, style='Card.TLabel', width=15).pack(side='left')
            ttk.Scale(r, from_=lo, to=hi, variable=var, orient='horizontal').pack(side='left', fill='x', expand=True, padx=4)
            ttk.Label(r, textvariable=var, style='Card.TLabel', width=5).pack(side='left')
        self.sam_hint = self.muted(b, '')
        self.mask_body = ttk.Frame(b, style='Card.TFrame'); self.mask_body.pack(fill='x')
        r = self.row(self.mask_body)
        self.mask_name = tk.StringVar(); en = ttk.Entry(r, textvariable=self.mask_name); en.pack(side='left', fill='x', expand=True)
        en.bind('<Return>', lambda e: self.rename_mask()); en.bind('<FocusOut>', lambda e: self.rename_mask())
        self.mask_invert = tk.BooleanVar()
        ttk.Checkbutton(r, text='Invert', variable=self.mask_invert, style='Card.TCheckbutton', command=self.invert_mask).pack(side='left')
        ttk.Label(self.mask_body, text='Parts of this mask', style='Sub.TLabel').pack(anchor='w', pady=(4, 0))
        self.parts = self.app._scrolled_listbox(self.mask_body, 4)
        self.parts.bind('<<ListboxSelect>>', lambda e: self.part_picked())
        r = self.row(self.mask_body)
        self.btn(r, 'Add / subtract', self.part_toggle); self.btn(r, 'Edit points', self.part_edit); self.btn(r, 'Delete part', self.part_delete)
        self.slider(self.mask_body, 'Grow / shrink', None, -300, 300, 1, unit=' px', hist='Mask grow/shrink',
                    get=lambda: (self.cur_mask() or {}).get('grow', 0) or 0, set_=lambda v: self.cur_mask() and self.cur_mask().__setitem__('grow', v))
        self.slider(self.mask_body, 'Mask feather', None, 0, 300, 1, unit=' px', hist='Mask feather',
                    get=lambda: (self.cur_mask() or {}).get('feather', 0) or 0, set_=lambda v: self.cur_mask() and self.cur_mask().__setitem__('feather', v))
        ttk.Label(self.mask_body, text='Adjust inside the mask', style='Sub.TLabel').pack(anchor='w', pady=(4, 0))
        for k, l, lo, hi, st, dec in (('exposure', 'Exposure', -4, 4, 0.01, 2), ('contrast', 'Contrast', -100, 100, 1, 0), ('highlights', 'Highlights', *TONE_RANGE['highlights'], 1, 0),
                                      ('shadows', 'Shadows', *TONE_RANGE['shadows'], 1, 0), ('midtones', 'Midtones', *TONE_RANGE['midtones'], 1, 0), ('whites', 'Whites', -100, 100, 1, 0),
                                      ('blacks', 'Blacks', -100, 100, 1, 0), ('temp', 'Temperature', -200, 200, 1, 0), ('tint', 'Tint', -150, 150, 1, 0),
                                      ('saturation', 'Saturation', -100, 100, 1, 0), ('clarity', 'Clarity', -100, 100, 1, 0), ('dehaze', 'Dehaze', -100, 100, 1, 0),
                                      ('micro', 'Texture', -100, 100, 1, 0), ('sharpness', 'Sharpness', -100, 200, 1, 0)):
            self.slider(self.mask_body, l, None, lo, hi, st, dec=dec, hist='Mask ' + l.lower(),
                        get=lambda k=k: (self.cur_mask() or {}).get('adj', {}).get(k, 0) or 0,
                        set_=lambda v, k=k: self.cur_mask() and self.cur_mask()['adj'].__setitem__(k, v))
        r = self.row(b); ttk.Label(r, text='AI model', style='Card.TLabel', width=9).pack(side='left')
        self.ai_model = tk.StringVar(value=AI_MODELS[0][1])
        ttk.Combobox(r, textvariable=self.ai_model, state='readonly', values=[v for _, v in AI_MODELS]).pack(side='left', fill='x', expand=True)
        r = self.row(b)
        self.btn(r, 'Remove masked area (inpaint)', self.ai_remove, accent=True)
        self.muted(b, 'Paint or select (SAM2) what to remove, then Remove. For edits, use the sliders above: they apply only inside the mask.')

    def _tool_only(self, s):
        s.default = None; s._style = lambda v: None; s._reset = lambda: None
        s._commit = lambda: setattr(s, '_dragged', False)
        s._moved_orig = s._moved

        def moved(_):
            if s._loading: return
            v = s._q(s.var.get()); s.set(v); s.num.set(s.fmt(v)); self.draw_overlay()
        s._moved = moved
        s.scale.configure(command=moved)

    def _build_basic(self, left):
        s = self.sec(left, 'basic', 'Light & white balance', key='light', open_=True)
        r = self.row(s.body)
        self.auto_btn = self.btn(r, 'Auto tone', self.auto_tone, accent=True)
        self.autowb_btn = self.btn(r, 'Auto WB', self.auto_wb)
        self.wb_btn = self.btn(r, 'Pick white…', lambda: self.set_tool(None if self.tool == 'wb' else 'wb'))
        self.slider(s.body, 'Temperature', ['temp'], -200, 200, 1, title='Warmer (right) or cooler, in mireds from the camera\'s white balance')
        self.slider(s.body, 'Tint', ['tint'], -150, 150, 1, title='Magenta (right) or green (left)')
        self.slider(s.body, 'Exposure', ['exposure'], -5, 5, 0.01, dec=2, unit=' EV')
        for k, l in (('contrast', 'Contrast'), ('highlights', 'Highlights'), ('shadows', 'Shadows'), ('midtones', 'Midtones'), ('whites', 'Whites'),
                     ('blacks', 'Blacks'), ('brightness', 'Brightness')):
            self.slider(s.body, l, [k], *TONE_RANGE.get(k, (-100, 100)), 1)

    def _build_presence(self, left):
        s = self.sec(left, 'presence', 'Presence & detail', key='presence', open_=True)
        for k, l in (('dehaze', 'Dehaze'), ('vibrance', 'Vibrance'), ('saturation', 'Saturation'), ('clarity', 'Clarity'),
                     ('localContrast', 'Local contrast'), ('micro', 'Microcontrast')):
            self.slider(s.body, l, [k], -100, 100, 1)

    # ─── loading an image
    def load(self, linear, info, keep=False):
        """linear: Rec.2020 float32 H x W x 3 (display orientation). info: name, path, frame, frame_diag, make, model,
        lens, lens_make, focal, aperture, f35, is_rgb, alpha, stem, quick."""
        same = keep and self.params is not None and self.info.get('path') == info.get('path')
        same_dims = same and (self.img_w, self.img_h) == (linear.shape[1], linear.shape[0])
        if self.tool: self.set_tool(None)
        with self.lock:
            self.dv = V.Develop(linear)
            self.dv.fills.clear()
        self.info = info; self.img_h, self.img_w = linear.shape[:2]
        self.frame = info.get('frame') or dict(x=0, y=0, w=1, h=1)
        self.frame_diag = info.get('frame_diag') or math.hypot(self.img_w, self.img_h)
        self.rasters.clear(); self.snap = None; self.lw = None; self.sam = None
        if not same:
            self.bitmaps.clear(); self.fills.clear(); self.op_layers.clear(); self.sel_mask = None; self.sel_spot = -1
            self.params = dev_defaults()
            self.params['lens'].update(focal=round((info.get('focal') or 0) * 10) / 10, aperture=round((info.get('aperture') or 0) * 10) / 10)
            self.camera = None; self.lens_cands = []
            self.history = [('Processed image', json.dumps(self.params))]; self.h_index = 0
        elif not same_dims:
            self.params['geometry']['auto'] = True; self.refit_crop(); self.params['outpaint'] = None
            self.history[self.h_index] = (self.history[self.h_index][0], json.dumps(self.params))
        self.zoom = 'fit'; self.zoom_var.set(0); self.zoom_lbl.configure(text='Fit')
        name = info.get('name', '')
        self.title_lbl.configure(text=f'{name} — {self.img_w} × {self.img_h}' + (' (quick preview: settings carry over to the full image)' if info.get('quick') else ''))
        if info.get('alpha') is not None and len(self.history) == 1 and not any(m['name'] == 'Transparent areas' for m in self.params['masks']):
            W, H = self.raster_dims(); A = info['alpha']
            a = np.asarray(Image.fromarray(np.round((1 - A) * 255).astype(np.uint8)).resize((W, H), Image.NEAREST))
            ref = new_id(); self.bitmaps[ref] = dict(w=W, h=H, data=a, frame=dict(self.frame))
            m = self.new_mask('Transparent areas'); m['comps'].append(dict(type='fill', ref=ref, erase=False, label='Transparent areas'))
            self.history[0] = (self.history[0][0], json.dumps(self.params))
        self.sync_all()
        if len(self.history) == 1 and not self.lens_cands: self.detect_lens(True)
        self.request()

    def open_image(self):
        p = filedialog.askopenfilename(filetypes=[('Images', '*.tif *.tiff *.jpg *.jpeg *.png *.bmp *.webp'), ('All files', '*.*')],
                                       initialdir=self.app.settings.get('dev_dir'), title='Open an image to develop')
        if not p: return
        self.app._remember_dir('dev_dir', p)

        def work(job):
            job.progress(0.1, f'Reading {os.path.basename(p)}')
            rgb, info, alpha = E.load_image_linear(p)
            return rgb, dict(name=os.path.basename(p), path=p, stem=os.path.splitext(p)[0], is_rgb=True, alpha=alpha,
                             make=info.get('make', ''), model=info.get('model', ''))
        self.app.run_task('Opening', work, lambda r: self.load(r[0], r[1]))

    def use_result(self):
        self.app.develop_selected_result()

    # ─── history
    def commit(self, label):
        if self.params is None: return
        snap = json.dumps(self.params)
        if snap == self.history[self.h_index][1]: return
        del self.history[self.h_index + 1:]
        self.history.append((label, snap)); self.h_index = len(self.history) - 1
        self.render_history()

    def goto(self, i):
        if not self.ready() or i < 0 or i >= len(self.history): return
        if self.tool in ('crop', 'guides'): self.end_tool(False)
        self.h_index = i; self.params = json.loads(self.history[i][1])
        self.clear_sam_prompt()
        self.sync_all(); self.changed()

    def render_history(self):
        lb = self.hist_list
        lb.delete(0, 'end')
        for i, (l, _) in enumerate(self.history):
            lb.insert('end', ('▸ ' if i == self.h_index else '   ') + l)
            if i > self.h_index: lb.itemconfigure(i, foreground=self.C['muted'])
        lb.see(self.h_index)
        self.undo_btn.configure(state='normal' if self.h_index > 0 else 'disabled')
        self.redo_btn.configure(state='normal' if self.h_index < len(self.history) - 1 else 'disabled')

    def _hist_clicked(self):
        sel = self.hist_list.curselection()
        if sel and sel[0] != self.h_index: self.goto(sel[0])

    def reset_all(self):
        if not self.ready(): return
        if self.tool in ('crop', 'guides'): self.end_tool(False)
        lens = self.params['lens']
        self.clear_sam_prompt()
        self.params = dev_defaults(); self.params['lens'].update(key=lens['key'], focal=lens['focal'], aperture=lens['aperture'], crop=lens['crop'])
        self.refit_crop(); self.sync_all(); self.changed(); self.commit('Reset all')

    # ─── syncing the controls
    def sync_all(self):
        p = self.params
        for s in self.sliders: s.sync()
        if p is None: return
        self.ca_auto.set(bool(p['ca'].get('auto')))
        if not p['ca'].get('auto'): self.ca_note.configure(text='')
        off = p.get('off') or {}
        for name, s in self.sections.items():
            if s.key: s.on.set(not off.get(s.key))
        self.aspect.set(dict(ASPECTS).get(p['aspect'], 'Original'))
        self.keep_empty.set(bool(p['geometry'].get('keepEmpty')))
        self._sync_mixer(); self.render_mask_ui(); self.render_spot_ui(); self.render_outpaint_ui(); self.render_lens_ui(); self.show_brush_size()
        self.curve.draw(); self.render_history(); self.update_view_name()

    def mixer_colour(self, i, v):
        """The colour band i goes to at slider value v, as Develop moves it (hue ±180°, chroma ×0…2, lightness ±30 %)."""
        h, mode = V.BAND_HUES[i], self.mixer_mode.get()
        if mode == 'Hue': return V.oklab_hex(0.7, 0.14, h + v * 1.8)
        if mode == 'Saturation': return V.oklab_hex(0.7, 0.14 * (1 + v / 100), h)
        return V.oklab_hex(0.7 * (1 + v / 100 * 0.3), 0.14, h)

    def _sync_mixer(self):
        for s in self.mixer: s.sync()

    def changed(self, src=None):
        """A parameter changed (live): refit the crop if the geometry moved, then render."""
        path = getattr(src, 'path', None) if src is not None else None
        if path and (path[0] == 'lens' or (path[0] == 'geometry' and path[1] in ('angle', 'persp'))): self.refit_crop()
        if path and path[0] == 'geometry' and self.tool == 'crop': self.draw_overlay()
        self.request()

    def section_switched(self, s):
        if not self.ready(): s.on.set(True); return
        off = dict(self.params.get('off') or {})
        if s.on.get(): off.pop(s.key, None)
        else: off[s.key] = True
        self.params['off'] = off
        self.changed(); self.commit(f'{s.title} {"on" if s.on.get() else "off"}')

    # ─── geometry (the page side)
    def canvas_dims(self, g=None):
        g = g or self.params['geometry']
        return (self.img_h, self.img_w) if g['rot90'] % 2 else (self.img_w, self.img_h)

    def full_dims(self, p=None):
        p = p or self.params
        g = p['geometry']; CW, CH = self.canvas_dims(g)
        return max(1, V.jround(g['crop']['w'] * CW)), max(1, V.jround(g['crop']['h'] * CH))

    def hinv(self, g):
        CW, CH = self.canvas_dims(g)
        k = json.dumps([g.get('persp'), CW, CH])
        if self._persp[0] != k:
            v, err = V.hinv_for(g, CW, CH)
            self._persp = [k, v, err]
        return self._persp[1]

    def crop_valid(self, c, g):
        if c['x'] < -1e-6 or c['y'] < -1e-6 or c['x'] + c['w'] > 1 + 1e-6 or c['y'] + c['h'] > 1 + 1e-6 or c['w'] <= 0 or c['h'] <= 0: return False
        if g.get('keepEmpty'): return True
        lens = self.lens_params()
        CW, CH = self.canvas_dims(g)
        N = 8 if lens and lens.get('dist') else 1
        pts = []
        for k in range(N + 1):
            f = k / N
            for u, v in ((c['x'] + c['w'] * f, c['y']), (c['x'] + c['w'] * f, c['y'] + c['h']), (c['x'], c['y'] + c['h'] * f), (c['x'] + c['w'], c['y'] + c['h'] * f)):
                pts.append(((u - 0.5) * CW, (v - 0.5) * CH))
        P = V.map_points(dict(rot90=g['rot90'], angle=g['angle'], Hinv=self.hinv(g)), lens, self.img_w, self.img_h, 1.0, pts)
        if np.isnan(P).any(): return False
        return bool((P[:, 0] >= -0.5).all() and (P[:, 1] >= -0.5).all() and (P[:, 0] <= self.img_w + 0.5).all() and (P[:, 1] <= self.img_h + 0.5).all())

    def aspect_n(self, p=None):
        p = p or self.params
        CW, CH = self.canvas_dims(p['geometry'])
        if p['aspect'] == 'free': return None
        a = CW / CH if p['aspect'] == 'orig' else float(p['aspect'])
        return a * CH / CW

    def fit_crop(self, g, r, cx, cy, max_scale):
        w0, h0 = (1, 1 / r) if r >= 1 else (r, 1)

        def at(s):
            w, h = w0 * s, h0 * s
            return dict(x=min(1 - w, max(0, cx - w / 2)), y=min(1 - h, max(0, cy - h / 2)), w=w, h=h)
        if self.crop_valid(at(max_scale), g): return at(max_scale)
        lo, hi = 0.0, max_scale
        for _ in range(40):
            m = (lo + hi) / 2
            if self.crop_valid(at(m), g): lo = m
            else: hi = m
        if lo < 0.05 and (cx != 0.5 or cy != 0.5): return self.fit_crop(g, r, 0.5, 0.5, max_scale)
        return at(lo)

    def refit_crop(self, p=None):
        p = p or self.params
        g = p['geometry']; c = g['crop']; r = self.aspect_n(p) or c['w'] / c['h']
        if g.get('auto'): g['crop'] = self.fit_crop(g, r, 0.5, 0.5, 1)
        else: g['crop'] = self.fit_crop(g, r, c['x'] + c['w'] / 2, c['y'] + c['h'] / 2, min(1, c['w'] / (1 if r >= 1 else r)))

    def rotate90(self, d):
        if not self.ready(): return
        g = self.params['geometry']; c = g['crop']
        g['rot90'] = (g['rot90'] + (1 if d > 0 else 3)) % 4
        g['crop'] = dict(x=1 - c['y'] - c['h'], y=c['x'], w=c['h'], h=c['w']) if d > 0 else dict(x=c['y'], y=1 - c['x'] - c['w'], w=c['h'], h=c['w'])
        g['persp']['guides'] = [([1 - y1, x1, 1 - y2, x2] if d > 0 else [y1, 1 - x1, y2, 1 - x2]) for x1, y1, x2, y2 in g['persp']['guides']]
        a = self.params['aspect']
        if a not in ('orig', 'free'):
            inv = next((k for k, _ in ASPECTS if k not in ('orig', 'free') and abs(float(k) - 1 / float(a)) < 0.01), None)
            if inv: self.params['aspect'] = inv
        self.sync_all(); self.refit_crop(); self.changed(); self.commit('Rotate 90° right' if d > 0 else 'Rotate 90° left')

    def aspect_changed(self):
        if not self.ready(): return
        p = self.params; c = p['geometry']['crop']
        p['aspect'] = next(k for k, v in ASPECTS if v == self.aspect.get())
        r = self.aspect_n(p)
        if r: p['geometry']['crop'] = self.fit_crop(p['geometry'], r, c['x'] + c['w'] / 2, c['y'] + c['h'] / 2, 1)
        self.changed(); self.draw_overlay(); self.commit('Crop aspect ' + self.aspect.get().replace(' ', ''))

    def crop_reset(self):
        if not self.ready(): return
        g = self.params['geometry']
        g['angle'] = 0; g['auto'] = True; self.params['aspect'] = 'orig'; g['crop'] = dict(x=0, y=0, w=1, h=1)
        self.refit_crop(); self.sync_all(); self.changed(); self.commit('Crop & angle reset')

    def keep_empty_changed(self):
        if not self.ready(): return
        g = self.params['geometry']; g['keepEmpty'] = self.keep_empty.get(); g['auto'] = True
        if g['keepEmpty']: g['crop'] = dict(x=0, y=0, w=1, h=1)
        else: self.refit_crop()
        self.changed(); self.render_outpaint_ui(); self.commit('Keep empty areas' if g['keepEmpty'] else 'Crop out empty areas')

    def drag_crop(self, st, h, du, dv):
        g = self.params['geometry']; r = self.aspect_n(); MIN = 0.03
        L, T, R, B = st['x'], st['y'], st['x'] + st['w'], st['y'] + st['h']
        if h == 'move':
            dx = max(-L, min(1 - R, du)); dy = max(-T, min(1 - B, dv)); L += dx; R += dx; T += dy; B += dy
        else:
            if 'w' in h: L = min(R - MIN, L + du)
            if 'e' in h: R = max(L + MIN, R + du)
            if 'n' in h: T = min(B - MIN, T + dv)
            if 's' in h: B = max(T + MIN, B + dv)
            if r:
                w, hh = R - L, B - T
                if len(h) == 2:
                    if w / hh > r: hh = w / r
                    else: w = hh * r
                    if 'w' in h: L = R - w
                    else: R = L + w
                    if 'n' in h: T = B - hh
                    else: B = T + hh
                elif h in ('e', 'w'):
                    cy = st['y'] + st['h'] / 2; hh = w / r; T = cy - hh / 2; B = cy + hh / 2
                else:
                    cx = st['x'] + st['w'] / 2; w = hh * r; L = cx - w / 2; R = cx + w / 2
        cand = dict(x=L, y=T, w=R - L, h=B - T)
        if self.crop_valid(cand, g): return cand
        lerp = lambda t: {k: st[k] + (cand[k] - st[k]) * t for k in 'xywh'}
        lo, hi = 0.0, 1.0
        for _ in range(30):
            m = (lo + hi) / 2
            if self.crop_valid(lerp(m), g): lo = m
            else: hi = m
        return lerp(lo)

    def straighten_from(self, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        if math.hypot(dx, dy) < 10: return
        phi = math.degrees(math.atan2(dy, dx))
        d = ((phi + 45) % 90 + 90) % 90 - 45
        g = self.params['geometry']
        g['angle'] = round(max(-45, min(45, g['angle'] - d)) * 100) / 100
        self.refit_crop(); self.sync_all(); self.changed()
        self.commit(f'Straighten ({"horizon" if abs(phi) < 45 or abs(phi) > 135 else "vertical"}) {"+" if g["angle"] > 0 else ""}{g["angle"]:.2f}°')

    def clear_guides(self):
        if not self.ready(): return
        self.params['geometry']['persp']['guides'] = []; self.params['geometry']['auto'] = True
        self.refit_crop(); self.changed(); self.draw_overlay(); self.commit('Guides cleared')

    def guides_summary(self):
        g = self.params['geometry']; CW, CH = self.canvas_dims(g); gs = g['persp']['guides']
        nv = sum(1 for x1, y1, x2, y2 in gs if abs((y2 - y1) * CH) > abs((x2 - x1) * CW)); nh = len(gs) - nv
        what = ('No guides yet.' if not gs else f'{nv} vertical + {nh} horizontal: full correction.' if nv and nh
                else f'{nv} vertical: made parallel and upright.' if nv >= 2 else f'{nh} horizontal: made parallel and level.' if nh >= 2
                else '1 vertical: rotated upright.' if nv else '1 horizontal: rotated level.')
        self.hinv(g)
        return what + (f" — Can't correct: {self._persp[2]}." if self._persp[2] else '')

    # ─── lens corrections
    def lens_params(self, p=None):
        p = p or self.params
        ls = p and p['lens']
        if not ls or not ls.get('key') or not self.lens_db or not self.frame_diag: return None
        L = self.lens_db.by_key.get(ls['key'])
        if not L: return None
        at = V.Lensfun.at(L, ls.get('focal'), ls.get('aperture'))
        crop_i = ls.get('crop') or 1; half = self.frame_diag / 2
        r = dict(unitD=half / ((L['crop'] / crop_i) * math.hypot(L['aspect'], 1)), unitV=half / (L['crop'] / crop_i),
                 distAmt=ls['distAmt'] / 100, vigAmt=ls['vigAmt'] / 100)
        if ls.get('dist') and at['dist']: r['dist'] = at['dist']
        if ls.get('tca') and at['tca']: r['tca'] = at['tca']
        if ls.get('vig') and at['vig']: r['vig'] = at['vig']
        return r if (r.get('dist') or r.get('tca') or r.get('vig')) else None

    def lens_folders(self):
        return [os.path.join(self.app.models_dir.get(), 'lensfun'), os.path.join(E.APP_ROOT, 'models', 'lensfun')]

    def detect_lens(self, initial):
        if self.lens_db is None:
            self.lens_status.configure(text='Loading the Lensfun lens database…')
            folders = self.lens_folders()   # Tk variables are read on the Tk thread only

            def work():
                db = V.Lensfun()
                ok = db.load(folders)
                self.app.q.put(('call', self._lens_loaded, db if ok else False, initial))
            threading.Thread(target=work, daemon=True).start()
            return
        if self.lens_db is False:
            self.lens_status.configure(text='The Lensfun database was not found (models-desktop\\lensfun): run desktop\\setup_models.py.'); return
        self._lens_loaded(self.lens_db, initial)

    def _lens_loaded(self, db, initial):
        self.lens_db = db
        if not db:
            self.lens_status.configure(text='The Lensfun database was not found (models-desktop\\lensfun): run desktop\\setup_models.py.'); return
        if not self.ready(): return
        i, ls = self.info, self.params['lens']
        cam = db.find_camera(i.get('make'), i.get('model'))
        self.camera = cam
        if not ls.get('crop'):
            ls['crop'] = cam['crop'] if cam else (round(i['f35'] / i['focal'] * 100) / 100 if i.get('f35') and i.get('focal') else 1)
        self.lens_cands = db.candidates(i.get('lens_make'), i.get('lens'), i.get('focal'), cam)
        if initial and not i.get('is_rgb') and not ls.get('key'):
            top = self.lens_cands[0] if self.lens_cands else None
            if top and (top[0] >= 1.2 or len(self.lens_cands) == 1): ls['key'] = top[1]['key']
        if initial and len(self.history) == 1:
            self.refit_crop(); self.history[0] = (self.history[0][0], json.dumps(self.params))
        self.render_lens_ui(); self.changed(type('P', (), {'path': ['lens']})())

    def render_lens_ui(self):
        if self.params is None: return
        ls, i, cam, db = self.params['lens'], self.info, self.camera, self.lens_db
        if db:
            camtxt = f"{cam['makerEn'] or cam['maker']} {cam['modelEn'] or cam['model']}" if cam else f"{i.get('make', '')} {i.get('model', '')}".strip() or 'unknown'
            self.lens_status.configure(text=f"Camera: {camtxt}{'' if cam else ' (not in Lensfun)'}" + (f" · lens in file: {i['lens']}" if i.get('lens') else '') +
                                       f' · {len(db.lenses)} profiles' + ('\nOrdinary images are often already corrected: only choose a lens if this one was not.' if i.get('is_rgb') else ''))
            q = self.lens_search.get().strip()
            if q:
                qt = db.norm(q); lst = sorted(((db.score(qt, L), L) for L in db.lenses), key=lambda x: -x[0]); lst = [x for x in lst if x[0] > 0][:60]
            else: lst = list(self.lens_cands[:60])
            cur = db.by_key.get(ls.get('key'))
            if cur and not any(L is cur for _, L in lst): lst.insert(0, (0, cur))
            self._lens_list = [L for _, L in lst]
            vals = ['(no lens correction)'] + [db.label(L) for L in self._lens_list]
            self.lens_box.configure(values=vals)
            self.lens_sel.set(db.label(cur) if cur else vals[0])
        for k, v in self.lens_vars.items(): v.set(f"{ls.get(k) or ''}")
        for k, v in self.lens_chk.items(): v.set(bool(ls.get(k)))

    def lens_chosen(self):
        if not self.ready() or not self.lens_db: return
        i = self.lens_box.current()
        L = self._lens_list[i - 1] if i > 0 else None
        self.params['lens']['key'] = L['key'] if L else ''
        self.changed(type('P', (), {'path': ['lens']})()); self.commit('Lens: ' + (L['modelEn'] or L['model']) if L else 'Lens correction off')

    def lens_num(self, k):
        if not self.ready(): return
        try: v = float(self.lens_vars[k].get() or 0)
        except ValueError: v = 0
        if v == (self.params['lens'].get(k) or 0): return
        self.params['lens'][k] = v
        self.changed(type('P', (), {'path': ['lens']})()); self.commit(f'Lens {k} {v:g}')

    def lens_check(self, k, label):
        if not self.ready(): return
        self.params['lens'][k] = self.lens_chk[k].get()
        self.changed(type('P', (), {'path': ['lens']})()); self.commit(f'Lens {label.lower()} {"on" if self.lens_chk[k].get() else "off"}')

    def ca_toggled(self):
        if not self.ready(): return
        if not self.ca_auto.get():
            self.params['ca']['auto'] = None; self.ca_note.configure(text='')
            self.changed(type('P', (), {'path': ['lens']})()); self.commit('Chromatic aberration: auto off'); return
        self.ca_note.configure(text='Measuring the colour fringes…')
        rt = self.worker_retouch(self.params)

        def work():
            try:
                with self.lock: r = self.dv.estimate_ca(rt)
            except Exception as ex: r = dict(ok=False, n=0, error=str(ex))
            self.app.q.put(('call', self._ca_done, r))
        threading.Thread(target=work, daemon=True).start()

    def _ca_done(self, r):
        if not r.get('ok'):
            self.ca_auto.set(False)
            self.ca_note.configure(text=r.get('error') or f"Not enough sharp edges away from the centre to measure ({r['n']} found); use the sliders instead."); return
        self.params['ca']['auto'] = {k: float(v) for k, v in r['ca'].items()}
        f = lambda v: f'{"+" if v >= 0 else ""}{v:.2f} px'
        lp = self.lens_params()
        self.ca_note.configure(text=f"Measured at {r['n']} edges: red {f(r['cornerR'])}, blue {f(r['cornerB'])} at the corners." + (' (Replaces the lens profile\'s colour-fringe correction.)' if lp and lp.get('tca') else ''))
        self.changed(type('P', (), {'path': ['lens']})()); self.commit('Chromatic aberration: auto')

    # ─── the engine's parameters
    def worker_params(self, p=None):
        p = p or self.params
        w = {k: copy.deepcopy(v) for k, v in p.items() if k not in ('curves', 'eq', 'aspect', 'lens', 'masks', 'retouch', 'ca', 'off')}
        g = p['geometry']
        w['geometry'] = dict(rot90=g['rot90'], angle=g['angle'], crop=dict(g['crop']), Hinv=self.hinv(g))
        w['retouch'] = self.worker_retouch(p)
        w['locals'] = self.worker_locals(p)
        lp = self.lens_params(p)
        ca = p.get('ca') or {}
        if ca.get('auto') or ca.get('red') or ca.get('blue'):
            a = ca.get('auto') or dict(vr=1, br=0, vb=1, bb=0); k = 2e-5
            hd = math.hypot(self.img_w, self.img_h) / 2
            lp = dict(lp or dict(unitD=hd, unitV=hd), tca=None, caUnit=hd,
                      ca=dict(vr=a['vr'] * (1 + ca['red'] * k), br=a['br'] * (1 + ca['red'] * k), vb=a['vb'] * (1 + ca['blue'] * k), bb=a['bb'] * (1 + ca['blue'] * k)))
        if lp: w['lens'] = lp
        if ca.get('purple') or ca.get('green'):
            w['defringe'] = dict(purple=ca.get('purple'), purpleHue=ca.get('purpleHue'), green=ca.get('green'), greenHue=ca.get('greenHue'))
        ck = json.dumps(p['curves'])
        if ck != self._lut_memo[0]: self._lut_memo[0:2] = [ck, V.curve_lut(p['curves'])]
        ek = json.dumps(p['eq'])
        if ek != self._lut_memo[2]: self._lut_memo[2:4] = [ek, V.eq_tables(p['eq'])]
        if self._lut_memo[1]: w['curveLut'] = self._lut_memo[1]
        if self._lut_memo[3]: w['eq'] = self._lut_memo[3]
        off = p.get('off') or {}
        if off.get('light'): w.update(temp=0, tint=0, exposure=0, contrast=0, highlights=0, shadows=0, midtones=0, whites=0, blacks=0, brightness=0)
        if off.get('presence'): w.update(dehaze=0, vibrance=0, saturation=0, clarity=0, localContrast=0, micro=0)
        if off.get('curves'): w.pop('curveLut', None); w.pop('eq', None)
        if off.get('mixer'): w['hsl'] = dict(hue=[], sat=[], lum=[])
        if off.get('grading'): w['grading'] = {}
        if off.get('sharpen'): w['usmAmount'] = 0
        if off.get('lens') and w.get('lens'):
            l = w['lens']; w['lens'] = dict(l, dist=None, tca=None, vig=None) if l.get('ca') else None
        if off.get('ca'):
            w.pop('defringe', None)
            if w.get('lens'):
                w['lens'] = dict(w['lens'], ca=None)
                if not w['lens'].get('dist') and not w['lens'].get('tca') and not w['lens'].get('vig'): w.pop('lens')
        if not w.get('lens'): w.pop('lens', None)
        if off.get('transform'): w['geometry']['Hinv'] = None
        if off.get('spots'): w['retouch'] = [x for x in w['retouch'] if x['kind'] != 'spot']
        if off.get('masks'): w['locals'] = []; w['retouch'] = [x for x in w['retouch'] if x['kind'] != 'fill']
        if p.get('outpaint'):
            o = p['outpaint']; L = self.op_layers.get(o['id'])
            if L and self.dv is not None: self.dv.op_layers[o['id']] = L
        return w

    # frame <-> image coordinates
    def to_img(self, x, y):
        F = self.frame; return (x - F['x']) / F['w'], (y - F['y']) / F['h']

    def to_frame(self, x, y):
        F = self.frame; return F['x'] + x * F['w'], F['y'] + y * F['h']

    def img_diag_scale(self):
        return self.frame_diag / math.hypot(self.img_w, self.img_h)

    def c2s(self, u, v):
        k = self.params['geometry']['rot90'] & 3
        return (u, v) if k == 0 else (v, 1 - u) if k == 1 else (1 - u, 1 - v) if k == 2 else (1 - v, u)

    def s2c(self, x, y):
        k = self.params['geometry']['rot90'] & 3
        return (x, y) if k == 0 else (1 - y, x) if k == 1 else (1 - x, 1 - y) if k == 2 else (y, 1 - x)

    def worker_retouch(self, p):
        ds = self.img_diag_scale()
        out = []
        for it in p.get('retouch') or []:
            if it['kind'] == 'spot':
                x, y = self.to_img(it['x'], it['y']); sx, sy = self.to_img(it['sx'], it['sy'])
                out.append(dict(kind='spot', x=x, y=y, sx=sx, sy=sy, r=it['r'] * ds, feather=it.get('feather'), mode=it.get('mode'), opacity=it.get('opacity', 1)))
            else:
                f = self.fills.get(it['id'])
                if f is not None and self.dv is not None and it['id'] not in self.dv.fills:
                    x0, y0 = self.to_img(f['rect']['x'], f['rect']['y']); x1, y1 = self.to_img(f['rect']['x'] + f['rect']['w'], f['rect']['y'] + f['rect']['h'])
                    self.dv.fills[it['id']] = dict(f, rect=dict(x=x0, y=y0, w=x1 - x0, h=y1 - y0))
                out.append(dict(kind='fill', id=it['id']))
        return out

    def raster_dims(self):
        s = min(1, 2048 / max(self.img_w, self.img_h))
        return max(1, V.jround(self.img_w * s)), max(1, V.jround(self.img_h * s))

    def mask_key(self, m):
        W, H = self.raster_dims()
        return 'm' + hashlib.md5(json.dumps([m['comps'], m.get('invert'), m.get('grow') or 0, m.get('feather') or 0, W, H, self.frame], sort_keys=True).encode()).hexdigest()[:16]

    def rasterize(self, m):
        key = self.mask_key(m)
        if key in self.rasters: return key, self.rasters[key]
        W, H = self.raster_dims()
        r = V.rasterize_mask(m, W, H, self.frame, self.frame_diag, self.img_w, self.bitmaps)
        self.rasters[key] = r
        while len(self.rasters) > 40: self.rasters.pop(next(iter(self.rasters)))
        return key, r

    def worker_locals(self, p):
        out = []
        for m in p.get('masks') or []:
            if not m['comps'] or not any(v for v in (m.get('adj') or {}).values()): continue
            key, r = self.rasterize(m)
            if self.dv is not None: self.dv.mask_rasters[key] = r
            out.append(dict(key=key, adj=m['adj']))
        return out

    # ─── views and rendering
    def view_kind(self):
        if self.tool in SV_TOOLS: return 'source'
        if self.tool == 'crop': return 'uncropped'
        return 'final'

    def view_dims(self, kind=None):
        kind = kind or self.view_kind()
        return self.canvas_dims() if kind in ('source', 'uncropped') else self.full_dims()

    def view_params(self, kind):
        wp = self.worker_params()
        if kind == 'source':
            wp['geometry'] = dict(rot90=wp['geometry']['rot90'], angle=0, crop=dict(x=0, y=0, w=1, h=1), Hinv=None)
            wp.pop('lens', None); wp['outpaint'] = None
        if kind == 'uncropped':
            wp['geometry'] = dict(wp['geometry'], crop=dict(x=0, y=0, w=1, h=1))
        return wp

    def request(self):
        """Ask the render thread for the current view (the newest request replaces any waiting one)."""
        if not self.ready() or not self.cv.winfo_ismapped(): return
        kind = self.view_kind()
        try: wp = self.view_params(kind)
        except Exception as ex:
            traceback.print_exc(); return
        cw, ch = max(10, self.cv.winfo_width()), max(10, self.cv.winfo_height())
        lw, lh = self.view_dims(kind)
        req = dict(kind=kind, params=wp, neutral=self.before, lw=lw, lh=lh, cw=cw, ch=ch, proof=self.proof_cfg())
        if self.zoom == 'fit' or kind == 'uncropped':
            s = min((cw - 8) / lw, (ch - 8) / lh, 1)
            req.update(mode='preview', max_side=max(64, min(2560, math.ceil(max(lw, lh) * s))))
        elif self.z < 1 and max(lw, lh) * self.z <= 2560:
            req.update(mode='preview', max_side=max(16, V.jround(max(lw, lh) * self.z)))
        else:
            z = self.z
            x0 = max(0, int(self.cx - cw / 2 / z)); y0 = max(0, int(self.cy - ch / 2 / z))
            x1 = min(lw, int(math.ceil(self.cx + cw / 2 / z)) + 1); y1 = min(lh, int(math.ceil(self.cy + ch / 2 / z)) + 1)
            req.update(mode='region', region=(x0, y0, max(1, x1 - x0), max(1, y1 - y0)))
        m = self.cur_mask()
        if kind == 'final' and m and m['comps'] and (self.show_overlay or self.tool in ('brush', 'sam')) and req['mode'] == 'preview':
            key, r = self.rasterize(m); self.dv.mask_rasters[key] = r; req['overlay'] = key
        req['hist'] = kind == 'final' and not self.before and req['mode'] == 'preview'
        self._req = req; self._req_ev.set()

    def _render_loop(self):
        while True:
            self._req_ev.wait(); self._req_ev.clear()
            req, self._req = self._req, None
            if req is None: continue
            self.rendering = True
            try:
                t0 = time.time()
                with self.lock:
                    dv = self.dv
                    if dv is None: continue
                    p = req['params']
                    if req['mode'] == 'preview':
                        t = dv.preview_target(p, req['max_side'], uncropped=req['kind'] != 'final')
                        crop = dict(x=0, y=0, w=1, h=1) if req['kind'] != 'final' else p['geometry']['crop']
                        px = t.data if req['neutral'] else V.develop(t, p, t.scale, dv.D, dv.airlight, None, dv.masks_for(t, p, crop, t.w, t.h, 0, 0))
                        ov = None
                        if req.get('overlay') and req['overlay'] in dv.mask_rasters:
                            k = 'ma:' + req['overlay']
                            if k not in t.cache: t.cache[k] = V.mask_alpha(dv.mask_rasters[req['overlay']], p['geometry'], p.get('lens'), crop, t.w, t.h, 0, 0, t.w, t.h, dv.W)
                            ov = t.cache[k]
                    else:
                        px = dv.render_region(p, req['region'], neutral=req['neutral']); ov = None
                    rgb = V.to_srgb8(np.ascontiguousarray(px, np.float32))
                scr = CM.screen_linear(np.ascontiguousarray(px, np.float32), req.get('proof'))   # None: rgb is right for the screen
                hist = None
                if req['hist']:
                    f = rgb.reshape(-1, 3)[::max(1, rgb.shape[0] * rgb.shape[1] // 250000)].astype(np.int64)
                    hist = [np.bincount(f[:, c], minlength=256) for c in range(3)] + [np.bincount(np.round(f @ [0.2126, 0.7152, 0.0722]).astype(np.int64), minlength=256)]
                self.app.q.put(('call', self._show, req, rgb, ov, hist, time.time() - t0, scr))
            except Exception as ex:
                traceback.print_exc()
                self.app.q.put(('log', 'err', f'Develop: {ex}'))
            finally:
                self.rendering = False

    def _show(self, req, rgb, ov, hist, dt, scr=None):
        self._shown = dict(req=req, rgb=rgb, ov=ov, scr=scr)
        self.timing.configure(text=f'{dt * 1000:.0f} ms')
        if hist is not None:
            self.hist = hist; self.draw_hist()
            if self.sections['curves'].is_open: self.curve.draw()
        self.redraw()

    def layout(self):
        """Where the view image sits on the canvas: (scale: canvas px per view px, x0, y0)."""
        sh = self._shown
        cw, ch = max(10, self.cv.winfo_width()), max(10, self.cv.winfo_height())
        kind = self.view_kind()
        lw, lh = self.view_dims(kind)
        if self.zoom == 'fit' or kind == 'uncropped':
            s = min((cw - 8) / lw, (ch - 8) / lh, 1)
            return s, (cw - lw * s) / 2, (ch - lh * s) / 2
        z = self.z
        return z, cw / 2 - self.cx * z, ch / 2 - self.cy * z

    def redraw(self):
        cv = self.cv
        cv.delete('all')
        sh = self._shown
        if not self.ready() or sh is None:
            cv.create_text(cv.winfo_width() / 2, cv.winfo_height() / 2, text='Nothing to develop yet', fill=self.C['muted'])
            return
        rgb, req = sh['rgb'] if sh.get('scr') is None else sh['scr'], sh['req']
        s, x0, y0 = self.layout()
        if req['mode'] == 'region':
            rx, ry, rw, rh = req['region']
            img = Image.fromarray(rgb)
            img = img.resize((max(1, round(rgb.shape[1] * s)), max(1, round(rgb.shape[0] * s))), Image.NEAREST if s >= 1 else Image.BOX)
            pos = (x0 + rx * s, y0 + ry * s)
        else:
            lw, lh = req['lw'], req['lh']
            tw, th = max(1, round(lw * s)), max(1, round(lh * s))
            img = Image.fromarray(rgb)
            if img.size != (tw, th):
                img = img.resize((tw, th), Image.BOX if tw < img.width else Image.NEAREST if s > 1.5 * img.width / lw else Image.BILINEAR)
            cw, ch = cv.winfo_width(), cv.winfo_height()
            # crop to what is visible (large zoomed previews)
            vx0, vy0 = max(0, int(-x0)), max(0, int(-y0)); vx1, vy1 = min(tw, int(cw - x0) + 1), min(th, int(ch - y0) + 1)
            if vx1 > vx0 and vy1 > vy0 and (vx0 or vy0 or vx1 < tw or vy1 < th): img = img.crop((vx0, vy0, vx1, vy1))
            pos = (x0 + vx0, y0 + vy0)
            if sh['ov'] is not None and req['kind'] == 'final':
                a = (sh['ov'].astype(np.float32) * 0.5).astype(np.uint8)
                o = Image.fromarray(np.dstack([np.full_like(a, 255), np.full_like(a, 40), np.full_like(a, 60), a]), 'RGBA').resize((tw, th), Image.BILINEAR)
                if (vx0, vy0, vx1, vy1) != (0, 0, tw, th): o = o.crop((vx0, vy0, vx1, vy1))
                img = Image.alpha_composite(img.convert('RGBA'), o)
        if req['kind'] == 'source' and self._mask_visible():
            img = self._source_overlay(img, s, pos)
        if self.tool == 'crop' and req['kind'] == 'uncropped':
            c = self.params['geometry']['crop']
            (cx0, cy0), (cx1, cy1) = self.from_uv(c['x'], c['y']), self.from_uv(c['x'] + c['w'], c['y'] + c['h'])
            a = np.asarray(img.convert('RGB')).astype(np.float32)
            yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
            inside = (xx + pos[0] >= cx0) & (xx + pos[0] < cx1) & (yy + pos[1] >= cy0) & (yy + pos[1] < cy1)
            img = Image.fromarray(np.where(inside[..., None], a, a * 0.4).astype(np.uint8))
        self._photo = ImageTk.PhotoImage(img)
        cv.create_image(*pos, image=self._photo, anchor='nw')
        self.draw_overlay(redraw_image=False)

    def _mask_visible(self):
        m = self.cur_mask()
        return bool(m and m['comps'] and (self.show_overlay or self.tool in ('brush', 'sam', 'grad')))

    def _source_overlay(self, img, s, pos):
        """The selected mask's raster over the uncorrected (rot90 only) view."""
        key, r = self.rasterize(self.cur_mask())
        k = self.params['geometry']['rot90'] & 3
        a = np.rot90(r, -k) if k else r      # raster (image orientation) -> canvas orientation
        lw, lh = self.view_dims('source')
        cw, ch = self.cv.winfo_width(), self.cv.winfo_height()
        _, x0, y0 = self.layout()
        full = Image.fromarray((a * 0.5).astype(np.uint8)).resize((max(1, round(lw * s)), max(1, round(lh * s))), Image.BILINEAR)
        full = full.crop((int(pos[0] - x0), int(pos[1] - y0), int(pos[0] - x0) + img.width, int(pos[1] - y0) + img.height))
        al = np.asarray(full)
        o = Image.fromarray(np.dstack([np.full_like(al, 255), np.full_like(al, 40), np.full_like(al, 60), al]), 'RGBA')
        return Image.alpha_composite(img.convert('RGBA'), o)

    def draw_hist(self):
        cv = self.hist_cv; cv.delete('all')
        W, H = max(20, cv.winfo_width()), max(20, cv.winfo_height())
        if self.hist is None: return
        h = self.hist
        mx = max(1, max(int(h[c][1:255].max()) for c in range(3)))
        img = np.zeros((H, W, 3), np.float32)
        xs = (np.arange(W) / max(1, W - 1) * 255).astype(int)
        cols = [(230, 60, 60), (60, 200, 60), (70, 110, 255)]
        for c in range(3):
            hh = (np.minimum(1, np.sqrt(h[c][xs] / mx)) * (H - 4)).astype(int)
            mask = np.arange(H)[:, None] >= (H - hh)[None, :]
            img[mask] += np.array(cols[c]) * 0.75
        im = Image.fromarray(np.clip(img + 21, 0, 255).astype(np.uint8))
        self._hphoto = ImageTk.PhotoImage(im)
        cv.create_image(0, 0, image=self._hphoto, anchor='nw')
        tot = max(1, int(h[3].sum()))
        lo, hi = h[3][0] / tot, (h[0][255] + h[1][255] + h[2][255]) / 3 / tot
        if lo > 0.002: cv.create_text(4, 4, anchor='nw', text=f'▲ {lo * 100:.1f}% black', fill='#fff', font=('Segoe UI', 8))
        if hi > 0.002: cv.create_text(W - 4, 4, anchor='ne', text=f'clipped {hi * 100:.1f}% ▲', fill='#fff', font=('Segoe UI', 8))

    def update_view_name(self):
        if not self.ready(): self.view_name.configure(text=''); return
        w, h = self.view_dims()
        t = {'crop': 'Crop & straighten', 'guides': 'Guides (shown without the perspective correction)'}.get(self.tool) or \
            ('Retouch view (without crop, rotation and lens corrections)' if self.tool in SV_TOOLS else 'Before Develop' if self.before else 'Developed result')
        self.view_name.configure(text=f'{t}   ·   {w} × {h}' + (' (quick preview)' if self.info.get('quick') else '') +
                                 (f'   ·   soft proof: {CM.info(self.proof_path)["desc"]}' if self.proof_cfg() else ''))

    def set_zoom(self, z, focus=None):
        """z: 'fit' or a scale; focus: (canvas x, y) to keep still."""
        if not self.ready() or self.tool == 'crop': self.zoom_var.set(0); return
        lw, lh = self.view_dims()
        s, x0, y0 = self.layout()
        cw, ch = self.cv.winfo_width(), self.cv.winfo_height()
        fx, fy = focus or (cw / 2, ch / 2)
        vx, vy = (fx - x0) / s, (fy - y0) / s
        if z == 'fit':
            self.zoom = 'fit'
        else:
            self.zoom, self.z = 'z', float(z)
            self.cx, self.cy = vx - (fx - cw / 2) / self.z, vy - (fy - ch / 2) / self.z
            self.clamp_center()
        idx = self.ZOOMS.index(z) if z in self.ZOOMS else next((i for i, v in enumerate(self.ZOOMS) if v != 'fit' and v >= self.z), 0)
        self.zoom_var.set(idx); self.zoom_lbl.configure(text='Fit' if z == 'fit' else f'{round(self.z * 100)}%')
        self.redraw(); self.request(); self.update_view_name()

    def clamp_center(self):
        lw, lh = self.view_dims()
        self.cx, self.cy = min(max(self.cx, 0), lw), min(max(self.cy, 0), lh)

    def set_before(self, on):
        if not self.ready() or self.before == on: return
        self.before = on; self.update_view_name(); self.request()

    # canvas <-> view coordinates (u, v in 0..1 of the view image)
    def to_uv(self, X, Y):
        s, x0, y0 = self.layout(); lw, lh = self.view_dims()
        return (X - x0) / (lw * s), (Y - y0) / (lh * s)

    def from_uv(self, u, v):
        s, x0, y0 = self.layout(); lw, lh = self.view_dims()
        return x0 + u * lw * s, y0 + v * lh * s

    def view_wh(self):
        s, x0, y0 = self.layout(); lw, lh = self.view_dims()
        return lw * s, lh * s

    # ─── overlays
    def draw_overlay(self, redraw_image=True):
        cv = self.cv
        cv.delete('ov')
        if not self.ready() or self._shown is None: return
        C = self.C
        W, H = self.view_wh()
        P = lambda u, v: self.from_uv(u, v)
        t = self.tool
        if t == 'crop':
            c = self.params['geometry']['crop']
            (x0, y0), (x1, y1) = P(c['x'], c['y']), P(c['x'] + c['w'], c['y'] + c['h'])
            for k in (1, 2):
                cv.create_line(x0 + (x1 - x0) * k / 3, y0, x0 + (x1 - x0) * k / 3, y1, fill='#ffffff', dash=(2, 4), tags='ov')
                cv.create_line(x0, y0 + (y1 - y0) * k / 3, x1, y0 + (y1 - y0) * k / 3, fill='#ffffff', dash=(2, 4), tags='ov')
            cv.create_rectangle(x0, y0, x1, y1, outline=C['accent'], width=2, tags='ov')
            for hx, hy in ((0, 0), (0.5, 0), (1, 0), (0, 0.5), (1, 0.5), (0, 1), (0.5, 1), (1, 1)):
                X, Y = x0 + (x1 - x0) * hx, y0 + (y1 - y0) * hy
                cv.create_rectangle(X - 5, Y - 5, X + 5, Y + 5, fill=C['accent'], outline='#000', tags='ov')
        if t == 'guides':
            CW, CH = self.canvas_dims()
            for x1, y1, x2, y2 in self.params['geometry']['persp']['guides']:
                vert = abs((y2 - y1) * CH) > abs((x2 - x1) * CW)
                (a, b), (c, d) = P(x1, y1), P(x2, y2)
                cv.create_line(a, b, c, d, fill=C['info'] if vert else C['warn'], width=2, tags='ov')
                for X, Y in ((a, b), (c, d)): cv.create_oval(X - 5, Y - 5, X + 5, Y + 5, fill='#fff', outline='#000', tags='ov')
        if t == 'spot':
            k = W / self.view_dims()[0] * self.frame_diag
            for i, q in enumerate(self.params['retouch']):
                if q['kind'] != 'spot': continue
                R = max(2, q['r'] * k)
                tx, ty = P(*self.s2c(*self.to_img(q['x'], q['y']))); sx, sy = P(*self.s2c(*self.to_img(q['sx'], q['sy'])))
                cv.create_line(tx, ty, sx, sy, fill='#fff', dash=(3, 3), tags='ov')
                cv.create_oval(sx - R, sy - R, sx + R, sy + R, outline=C['info'], width=1, dash=(4, 3), tags='ov')
                cv.create_oval(tx - R, ty - R, tx + R, ty + R, outline=C['accent'] if i == self.sel_spot else '#fff', width=2, tags='ov')
        if t in LASSOS and self.lasso:
            pts = self.lasso['pts'] + self.lasso.get('preview', [])
            if len(pts) > 1:
                step = max(1, len(pts) // 3000)
                cv.create_line(*[c for u, v in pts[::step] + [pts[-1]] for c in P(u, v)], fill=C['warn'], width=2, tags='ov')
            if self.lasso.get('polygon') and len(pts) > 2:
                cv.create_line(*P(*pts[-1]), *P(*pts[0]), fill=C['warn'], dash=(3, 3), tags='ov')
            if self.lasso.get('magnetic') or self.lasso.get('polygon'):
                for i in self.lasso['anchors']:
                    X, Y = P(*self.lasso['pts'][i]); cv.create_rectangle(X - 3, Y - 3, X + 3, Y + 3, fill=C['warn'], outline='#000', tags='ov')
        if t == 'rect' and self._drag and self._drag.get('kind') == 'rect':
            (a, b), (c, d) = P(*self._drag['p']), P(*self._drag['cur'])
            cv.create_rectangle(a, b, c, d, outline=C['warn'], width=2, tags='ov')
        if t == 'grad':
            self.draw_grads()
        if t in ('brush', 'spot') and getattr(self, 'cursor', None):
            X, Y = self.cursor
            R = (self.brush['size'] if t == 'brush' else self.spot['size']) * W / self.view_dims()[0] * self.frame_diag
            cv.create_oval(X - R, Y - R, X + R, Y + R, outline='#fff', tags='ov')
            if t == 'brush' and self.brush['feather'] > 0:
                r2 = R * (1 - self.brush['feather'] * 0.5); cv.create_oval(X - r2, Y - r2, X + r2, Y + r2, outline='#fff', dash=(2, 3), tags='ov')
        if t == 'sam' and self.sam_prompt:
            pr = self.sam_prompt
            if pr.get('box'):
                (a, b), (c, d) = P(pr['box'][0], pr['box'][1]), P(pr['box'][2], pr['box'][3])
                cv.create_rectangle(a, b, c, d, outline=C['info'], width=2, tags='ov')
            if self._drag and self._drag.get('kind') == 'sam' and self._drag.get('cur'):
                a, b = self._drag['p']; c, d = self._drag['cur']
                cv.create_rectangle(a, b, c, d, outline=C['info'], dash=(4, 3), tags='ov')
            for u, v, l in pr['points']:
                X, Y = P(u, v); cv.create_oval(X - 6, Y - 6, X + 6, Y + 6, fill=C['ok'] if l else C['err'], outline='#000', width=2, tags='ov')
        if t == 'straighten':
            (ax, ay), (bx, by) = P(0, 0), P(1, 1)
            for k in range(1, 8):
                cv.create_line(ax, ay + (by - ay) * k / 8, bx, ay + (by - ay) * k / 8, fill='#cccccc', dash=(2, 6), tags='ov')
                cv.create_line(ax + (bx - ax) * k / 8, ay, ax + (bx - ax) * k / 8, by, fill='#cccccc', dash=(2, 6), tags='ov')
            if self._last_line: cv.create_line(*self._last_line, fill=C['warn'], width=2, tags='ov')

    # ─── tools
    def view_to_src(self, kind, x, y):
        """A point of the source or final view (view pixels) -> source position (0…1), or None."""
        CW, CH = self.canvas_dims()
        if kind == 'source': return self.c2s(x / CW, y / CH)
        g = self.params['geometry']; c = g['crop']
        P = V.map_points(dict(rot90=g['rot90'], angle=g['angle'], Hinv=self.hinv(g)), self.lens_params(), self.img_w, self.img_h, 1.0,
                         [(c['x'] * CW + x - CW / 2, c['y'] * CH + y - CH / 2)])
        if np.isnan(P).any(): return None
        return P[0, 0] / self.img_w, P[0, 1] / self.img_h

    def src_to_view(self, kind, sx, sy):
        """Source position (0…1) -> point of the source or final view (view pixels), or None."""
        CW, CH = self.canvas_dims()
        if kind == 'source':
            u, v = self.s2c(sx, sy); return u * CW, v * CH
        g = self.params['geometry']; c = g['crop']
        a = V._geo_args(dict(rot90=g['rot90'], angle=g['angle'], Hinv=self.hinv(g)), self.lens_params(), self.img_w, self.img_h, 1.0)
        TW, TH = self.full_dims()
        qx, qy, _ = V._locate(sx * self.img_w - 0.5, sy * self.img_h - 0.5, TW, TH, max(4, int(min(TW, TH) / 200)),
                              c['x'] * a[0], c['y'] * a[1], 1.0, *a[2:], a[0], a[1], self.img_w, self.img_h)
        return None if not np.isfinite(qx) else (qx + 0.5, qy + 0.5)

    def set_tool(self, t):
        if not self.ready(): return
        if self.tool == 'crop' and t != 'crop': self.end_crop(True, from_set_tool=True)
        if self.tool == 'guides' and t != 'guides': self.end_guides(True, from_set_tool=True)
        k0 = self.view_kind()
        self.tool = t; self._last_line = None
        k1 = self.view_kind()
        if self.zoom != 'fit' and k0 != k1 and {k0, k1} == {'source', 'final'}:   # zoomed in: stay on the same place
            try:
                q = self.view_to_src(k0, self.cx, self.cy)
                q = q and self.src_to_view(k1, *q)
                if q: self.cx, self.cy = q; self.clamp_center()
            except Exception:
                traceback.print_exc()
        if t in ('crop', 'guides'):
            self.tool_snap = json.dumps(self.params)
            self.zoom = 'fit'; self.zoom_var.set(0); self.zoom_lbl.configure(text='Fit')
        if t == 'sam':
            self.sam_prompt = dict(points=[], box=None, comp=None, erase=self.brush['erase'])
            self.sam_hint.configure(text='Click the object to select (right-click or Alt-click excludes, drag for a box).')
            self.sam_prepare()
        if t != 'sam' and self.sam_prompt is not None:
            if self.sam_prompt.get('comp') is not None: self.commit('SAM2 selection')
            self.sam_prompt = None
        if t not in LASSOS: self.lasso = None
        if t == 'mlasso': self.lasso_prepare()
        if t == 'grad': self.grad_sel = next((c for c in reversed((self.cur_mask() or {}).get('comps', [])) if c['type'] == 'grad'), None)
        self.spots_btn.configure(text='Done' if t == 'spot' else 'Spots…')
        self.crop_btn.configure(text='Done' if t == 'crop' else 'Crop…')
        self.guides_btn.configure(text='Done' if t == 'guides' else 'Guides…')
        for b, on in ((self.straighten_btn, t == 'straighten'), (self.wb_btn, t == 'wb')):
            b.configure(style='Accent.TButton' if on else 'Small.TButton')
        self.hint.configure(text=(TOOL_HINTS.get(t, '') + (' ' + self.guides_summary() if t == 'guides' else '')) if t else '')
        self.cv.configure(cursor='crosshair' if t in ('straighten', 'wb', 'sam', 'spot', 'brush', 'lasso', 'mlasso', 'poly', 'rect', 'grad', 'fill', 'guides') else 'arrow')
        self.render_mask_ui(); self.update_view_name()
        self._shown = None if self._shown is None else self._shown
        self.request(); self.draw_overlay()

    def end_tool(self, apply):
        if self.tool == 'crop': self.end_crop(apply)
        elif self.tool == 'guides': self.end_guides(apply)
        elif self.tool: self.set_tool(None)

    def end_crop(self, apply, from_set_tool=False):
        snap = self.tool_snap; self.tool_snap = None
        if not apply and snap: self.params = json.loads(snap); self.sync_all()
        if not from_set_tool: self.tool = None; self.set_tool(None)
        if apply and snap and snap != json.dumps(self.params): self.commit('Crop')
        self.request()

    def end_guides(self, apply, from_set_tool=False):
        snap = self.tool_snap; self.tool_snap = None
        if not apply and snap: self.params = json.loads(snap); self.sync_all()
        if not from_set_tool: self.tool = None; self.set_tool(None)
        if apply and snap and snap != json.dumps(self.params):
            g = self.params['geometry']
            if g['persp']['guides']: g['angle'] = 0
            g['auto'] = True; self.refit_crop(); self.sync_all()
            n = len(g['persp']['guides'])
            self.commit(f'Guided upright ({n} line{"s" if n > 1 else ""})' if n else 'Guides removed')
        self.request()

    def _bind_view(self):
        cv = self.cv
        cv.bind('<Configure>', lambda e: (self.redraw(), self.request()))
        import rawdemosaic_desktop as D
        D.bind_wheel(cv, self._wheel)
        for b in (1, 2, 3):
            cv.bind(f'<ButtonPress-{b}>', lambda e, b=b: self._press(e, b))
            cv.bind(f'<B{b}-Motion>', lambda e, b=b: self._motion(e, b))
            cv.bind(f'<ButtonRelease-{b}>', lambda e, b=b: self._release(e, b))
        cv.bind('<Motion>', self._hover)
        cv.bind('<Double-Button-1>', self._dbl)
        cv.bind('<Leave>', lambda e: (setattr(self, 'cursor', None), self.draw_overlay()))
        r = self.root
        r.bind_all('<Key>', self._key, add='+')
        r.bind_all('<KeyRelease>', self._keyup, add='+')

    def active(self):
        try: return self.app.nb.index(self.app.nb.select()) == self.app.nb.index(self.tab)
        except Exception: return False

    def _key(self, e):
        if not self.active() or not self.ready(): return
        w = e.widget
        typing = isinstance(w, (tk.Entry, ttk.Entry, ttk.Combobox, ttk.Spinbox, tk.Text))
        ctrl = bool(e.state & 0x4)
        k = e.keysym
        if ctrl and not typing:
            if k.lower() == 'z' and not (e.state & 0x1): self.goto(self.h_index - 1); return 'break'
            if k.lower() == 'y' or (k.lower() == 'z' and e.state & 0x1): self.goto(self.h_index + 1); return 'break'
        if typing: return
        if k == 'Escape' and self.tool in LASSOS and self.lasso: self.lasso = None; self.draw_overlay()
        elif k == 'Escape' and self.tool: self.end_tool(False)
        elif k == 'Return' and self.tool in ('crop', 'guides'): self.end_tool(True)
        elif k == 'backslash': self.set_before(True)
        elif k in ('Delete', 'BackSpace') and self.tool == 'spot': self.delete_spot(self.sel_spot)
        elif k in ('Delete', 'BackSpace') and self.tool == 'grad': self.grad_delete(self.grad_sel)
        elif k in ('Delete', 'BackSpace') and self.tool == 'sam' and self.sam_prompt:
            pr = self.sam_prompt
            if pr['points']: pr['points'].pop()
            else: pr['box'] = None
            self.draw_overlay(); self.sam_update()
        elif k == 'space' and self.tool in RETOUCH_TOOLS: self.space = True
        elif k in ('bracketleft', 'bracketright') and self.tool == 'brush': self.set_brush_size(self.brush['size'] * (1.25 if k == 'bracketright' else 0.8))
        elif self.tool in ('mlasso', 'poly') and self.lasso and k in ('BackSpace', 'Delete'): self.lasso_back()
        elif self.tool in ('mlasso', 'poly') and self.lasso and k == 'Return': self.lasso_close(bool(e.state & ALT))

    def _keyup(self, e):
        if e.keysym == 'backslash': self.set_before(False)
        if e.keysym == 'space': self.space = False

    def _wheel(self, e, d):
        if not self.ready() or self.tool == 'crop': return
        lw, lh = self.view_dims()
        cw, ch = self.cv.winfo_width(), self.cv.winfo_height()
        fit = min((cw - 8) / lw, (ch - 8) / lh, 1)
        cur = fit if self.zoom == 'fit' else self.z
        levels = [z for z in self.ZOOMS if z != 'fit']
        if d > 0: nxt = next((z for z in levels if z > cur * 1.01), levels[-1])
        else:
            nxt = next((z for z in reversed(levels) if z < cur * 0.99), 'fit')
            if nxt != 'fit' and nxt <= fit * 1.01: nxt = 'fit'
        self.set_zoom(nxt, (e.x, e.y))

    def _hover(self, e):
        if self.tool in ('brush', 'spot'): self.cursor = (e.x, e.y); self.draw_overlay()
        if self.tool == 'mlasso' and self.lasso and not self._drag: self.lasso_move(*self.to_uv(e.x, e.y))
        if self.tool == 'poly' and self.lasso and not self._drag: self.lasso['preview'] = [self.to_uv(e.x, e.y)]; self.draw_overlay()
        if self.tool == 'grad' and not self._drag:
            h = self.grad_hit(e.x, e.y)
            self.cv.configure(cursor={'mid': 'fleur', 'rot': 'exchange', 'aline': 'sb_v_double_arrow', 'bline': 'sb_v_double_arrow'}.get(h and h[1], 'crosshair'))
        if self.tool == 'crop' and not self._drag:
            h = self.crop_hit(e.x, e.y)
            self.cv.configure(cursor={'n': 'sb_v_double_arrow', 's': 'sb_v_double_arrow', 'e': 'sb_h_double_arrow', 'w': 'sb_h_double_arrow',
                                      'nw': 'size_nw_se', 'se': 'size_nw_se', 'ne': 'size_ne_sw', 'sw': 'size_ne_sw', 'move': 'fleur'}.get(h, 'arrow'))

    def crop_hit(self, X, Y):
        c = self.params['geometry']['crop']
        (L, T), (R, B) = self.from_uv(c['x'], c['y']), self.from_uv(c['x'] + c['w'], c['y'] + c['h'])
        tol = 12
        if X < L - tol or X > R + tol or Y < T - tol or Y > B + tol: return None
        h = ('n' if abs(Y - T) < tol else 's' if abs(Y - B) < tol else '') + ('w' if abs(X - L) < tol else 'e' if abs(X - R) < tol else '')
        return h or ('move' if L < X < R and T < Y < B else None)

    def guide_hit(self, X, Y):
        gs = self.params['geometry']['persp']['guides']
        for i in range(len(gs) - 1, -1, -1):
            x1, y1, x2, y2 = gs[i]
            if math.dist((X, Y), self.from_uv(x1, y1)) < 12: return i, 0
            if math.dist((X, Y), self.from_uv(x2, y2)) < 12: return i, 1
        for i in range(len(gs) - 1, -1, -1):
            (ax, ay), (bx, by) = self.from_uv(gs[i][0], gs[i][1]), self.from_uv(gs[i][2], gs[i][3])
            L2 = (bx - ax) ** 2 + (by - ay) ** 2
            t = max(0, min(1, ((X - ax) * (bx - ax) + (Y - ay) * (by - ay)) / (L2 or 1)))
            if math.hypot(X - ax - t * (bx - ax), Y - ay - t * (by - ay)) < 8: return i, -1
        return None

    def _press(self, e, b):
        if not self.ready() or self._shown is None: return
        t = self.tool
        u, v = self.to_uv(e.x, e.y)
        alt = bool(e.state & ALT); shift = bool(e.state & 0x1)
        if b == 2 or (self.space and b == 1):
            self._drag = dict(kind='pan', p=(e.x, e.y), c=(self.cx, self.cy), moved=False); return
        if t == 'sam' and b in (1, 3):
            self._drag = dict(kind='sam', p=(e.x, e.y), neg=b == 3 or alt, shift=shift); return
        if b != 1: return
        if t == 'spot':
            h = self.spot_hit(e.x, e.y)
            if h:
                self.sel_spot = h[0]; s0 = self.params['retouch'][h[0]]
                self._drag = dict(kind='spot', i=h[0], part=h[1], p=(u, v), start=(s0['x'], s0['y'], s0['sx'], s0['sy']), moved=False)
                self.render_spot_ui(); self.draw_overlay()
            else: self.add_spot(u, v)
            return
        if t == 'fill': self.fill_at(u, v, alt); return
        if t == 'lasso':
            self.lasso = dict(pts=[(u, v)], anchors=[0], magnetic=False, preview=[]); self._drag = dict(kind='lasso', erase=alt, last=(e.x, e.y)); return
        if t == 'mlasso':
            now = time.time(); lc = getattr(self, '_last_click', None); self._last_click = (now, e.x, e.y)
            if self.lasso and lc and now - lc[0] < 0.4 and math.hypot(e.x - lc[1], e.y - lc[2]) < 6: self.lasso_close(alt); return
            self.lasso_erase = alt; self.lasso_click(u, v); return
        if t == 'poly':
            now = time.time(); lc = getattr(self, '_last_click', None); self._last_click = (now, e.x, e.y)
            if self.lasso and lc and now - lc[0] < 0.4 and math.hypot(e.x - lc[1], e.y - lc[2]) < 6: self.lasso_close(alt); return
            self.lasso_erase = alt; self.poly_click(u, v); return
        if t == 'rect':
            self._drag = dict(kind='rect', p=(u, v), cur=(u, v), erase=alt); return
        if t == 'grad':
            self.grad_press(e, alt); return
        if t == 'brush':
            m = self.ensure_mask(); x, y = self.to_frame(*self.c2s(u, v))
            c = dict(type='brush', erase=self.brush['erase'], size=self.brush['size'], feather=self.brush['feather'], flow=self.brush['flow'], pts=[[x, y]])
            m['comps'].append(c); self._drag = dict(kind='brush', c=c, last=(e.x, e.y)); self.stroke_preview(c); return
        if t == 'crop':
            h = self.crop_hit(e.x, e.y)
            if h: self._drag = dict(kind='crop', h=h, p=(u, v), start=dict(self.params['geometry']['crop']))
            return
        if t == 'guides':
            gs = self.params['geometry']['persp']['guides']; h = self.guide_hit(e.x, e.y)
            if h: self._drag = dict(kind='guide', i=h[0], end=h[1], p=(u, v), start=list(gs[h[0]]))
            elif len(gs) < 4:
                gs.append([u, v, u, v]); self._drag = dict(kind='guide', i=len(gs) - 1, end=1, p=(u, v), start=[u, v, u, v], new=True)
            return
        if t == 'straighten': self._drag = dict(kind='line', p=(e.x, e.y)); return
        if t == 'wb': self.set_tool(None); self.pick_white(u, v); return
        self._drag = dict(kind='pan', p=(e.x, e.y), c=(self.cx, self.cy), moved=False)

    def _motion(self, e, b):
        d = self._drag
        if not d: return
        u, v = self.to_uv(e.x, e.y)
        k = d['kind']
        if k == 'pan':
            dx, dy = e.x - d['p'][0], e.y - d['p'][1]
            if abs(dx) + abs(dy) > 3: d['moved'] = True
            if self.zoom != 'fit':
                self.cx, self.cy = d['c'][0] - dx / self.z, d['c'][1] - dy / self.z; self.clamp_center()
                self.redraw(); self.request()
        elif k == 'lasso':
            if math.hypot(e.x - d['last'][0], e.y - d['last'][1]) >= 2:
                d['last'] = (e.x, e.y); self.lasso['pts'].append((u, v)); self.draw_overlay()
        elif k == 'spot':
            s1 = self.params['retouch'][d['i']]
            a, bb = self.to_frame(*self.c2s(u, v)); a0, b0 = self.to_frame(*self.c2s(*d['p'])); st = d['start']
            if d['part'] == 'target': s1['x'], s1['y'] = st[0] + a - a0, st[1] + bb - b0
            else: s1['sx'], s1['sy'] = st[2] + a - a0, st[3] + bb - b0
            d['moved'] = True; self.draw_overlay(); self.changed()
        elif k == 'brush':
            if math.hypot(e.x - d['last'][0], e.y - d['last'][1]) < 2: return
            d['last'] = (e.x, e.y); d['c']['pts'].append(list(self.to_frame(*self.c2s(u, v)))); self.stroke_preview(d['c'])
        elif k == 'sam':
            d['cur'] = (e.x, e.y); self.draw_overlay()
        elif k == 'rect':
            d['cur'] = (u, v); self.draw_overlay()
        elif k == 'grad':
            self.grad_motion(d, e)
        elif k == 'crop':
            self.params['geometry']['crop'] = self.drag_crop(d['start'], d['h'], u - d['p'][0], v - d['p'][1])
            self.params['geometry']['auto'] = False; self.redraw(); self.update_view_name()
        elif k == 'line':
            self._last_line = (d['p'][0], d['p'][1], e.x, e.y); self.draw_overlay()
        elif k == 'guide':
            cl = lambda x: max(0.0, min(1.0, x)); g = self.params['geometry']['persp']['guides']; s0 = d['start']; du, dv = u - d['p'][0], v - d['p'][1]
            g[d['i']] = ([cl(s0[0] + du), cl(s0[1] + dv), s0[2], s0[3]] if d['end'] == 0 else [s0[0], s0[1], cl(s0[2] + du), cl(s0[3] + dv)] if d['end'] == 1
                         else [cl(s0[0] + du), cl(s0[1] + dv), cl(s0[2] + du), cl(s0[3] + dv)])
            self.draw_overlay(); self.hint.configure(text=TOOL_HINTS['guides'] + ' ' + self.guides_summary())

    def _release(self, e, b):
        d, self._drag = self._drag, None
        if not d: return
        u, v = self.to_uv(e.x, e.y)
        k = d['kind']
        if k == 'spot':
            if d['moved']: self.commit('Spot moved')
        elif k == 'brush':
            self.changed(); self.render_mask_ui(); self.commit('Mask erased' if d['c']['erase'] else 'Mask painted')
        elif k == 'lasso': self.lasso_add(d['erase'] or bool(e.state & ALT))
        elif k == 'rect': self.rect_add(d['p'], (u, v), d['erase'] or bool(e.state & ALT))
        elif k == 'grad': self.grad_release(d)
        elif k == 'sam':
            pr = self.sam_prompt
            if not pr: return
            if d['shift'] and pr.get('comp') is not None: self.commit('SAM2 selection'); pr.update(points=[], box=None, comp=None)
            W, H = self.view_wh()
            hit = next((i for i, q in enumerate(pr['points']) if math.hypot(q[0] * W - (e.x - self.from_uv(0, 0)[0]), q[1] * H - (e.y - self.from_uv(0, 0)[1])) < 9), -1)
            if math.hypot(e.x - d['p'][0], e.y - d['p'][1]) > 8:
                (a, bb) = self.to_uv(*d['p'])
                pr['box'] = [min(a, u), min(bb, v), max(a, u), max(bb, v)]
            elif hit >= 0: del pr['points'][hit]
            else: pr['points'].append([u, v, 0 if d['neg'] else 1])
            self.draw_overlay(); self.sam_update()
        elif k == 'line':
            self._last_line = None; self.set_tool(None); self.straighten_from(d['p'], (e.x, e.y))
        elif k == 'guide':
            g = self.params['geometry']['persp']['guides']; x1, y1, x2, y2 = g[d['i']]
            W, H = self.view_wh()
            if d.get('new') and math.hypot((x2 - x1) * W, (y2 - y1) * H) < 12: del g[d['i']]
            self.draw_overlay(); self.hint.configure(text=TOOL_HINTS['guides'] + ' ' + self.guides_summary())
        elif k == 'crop': self.request()
        elif k == 'pan' and not d['moved'] and self.tool is None:
            self.set_zoom('fit' if self.zoom != 'fit' else 1, (e.x, e.y))

    def _dbl(self, e):
        if self.tool == 'grad':
            h = self.grad_hit(e.x, e.y)
            if h and h[1] == 'mid': self.grad_delete(h[0])
        elif self.tool in ('mlasso', 'poly') and self.lasso: self.lasso_close(bool(e.state & ALT))
        elif self.tool == 'spot':
            h = self.spot_hit(e.x, e.y)
            if h: self.delete_spot(h[0])
        elif self.tool == 'guides':
            h = self.guide_hit(e.x, e.y)
            if h:
                del self.params['geometry']['persp']['guides'][h[0]]; self.draw_overlay()
                self.hint.configure(text=TOOL_HINTS['guides'] + ' ' + self.guides_summary())

    # ─── white balance and auto
    def pick_white(self, u, v):
        if u < 0 or v < 0 or u > 1 or v > 1: return
        wp = self.worker_params()

        def work():
            with self.lock: r = self.dv.pick_wb(wp, u, v)
            self.app.q.put(('call', self._wb_done, r, False))
        threading.Thread(target=work, daemon=True).start()

    def _wb_done(self, r, auto):
        self.autowb_btn.configure(state='normal')
        if not r: return
        self.params['temp'], self.params['tint'] = r
        self.sync_all(); self.changed()
        self.commit(f'White balance {"auto" if auto else "picked"} ({"+" if r[0] > 0 else ""}{round(r[0])}, {"+" if r[1] > 0 else ""}{round(r[1])})')

    def auto_wb(self):
        if not self.ready(): return
        self.autowb_btn.configure(state='disabled'); wp = self.worker_params()

        def work():
            with self.lock: r = self.dv.auto_wb(wp)
            self.app.q.put(('call', self._wb_done, r, True))
        threading.Thread(target=work, daemon=True).start()

    def auto_tone(self):
        if not self.ready(): return
        self.auto_btn.configure(state='disabled'); wp = self.worker_params()

        def work():
            try:
                with self.lock: r = self.dv.auto_tone(wp)
            except Exception as ex:
                traceback.print_exc(); r = None
            self.app.q.put(('call', self._auto_done, r))
        threading.Thread(target=work, daemon=True).start()

    def _auto_done(self, r):
        self.auto_btn.configure(state='normal')
        if not r: return
        for k in ('exposure', 'contrast', 'highlights', 'shadows', 'whites', 'blacks', 'vibrance'): self.params[k] = r[k]
        self.params['brightness'] = 0
        self.sync_all(); self.changed(); self.commit('Auto tone')

    # ─── spots
    def cur_spot(self):
        if self.params is None: return None
        r = self.params['retouch']
        return r[self.sel_spot] if 0 <= self.sel_spot < len(r) and r[self.sel_spot]['kind'] == 'spot' else None

    def cur_spot_changed(self):
        return self.cur_spot() is not None

    def spot_set(self, k, v):
        if k == 'r': self.spot['size'] = v
        else: self.spot['feather'] = v
        s = self.cur_spot()
        if s: s[k] = v; self.changed()
        self.draw_overlay()

    def spot_mode_changed(self):
        self.spot['mode'] = 'heal' if self.spot_mode.get().startswith('Heal') else 'clone'
        s = self.cur_spot()
        if s: s['mode'] = self.spot['mode']; self.changed(); self.commit('Spot: ' + s['mode'])

    def render_spot_ui(self):
        if self.params is None: return
        n = sum(1 for x in self.params['retouch'] if x['kind'] == 'spot')
        self.spot_count.configure(text=f'{n} spot{"s" if n != 1 else ""}' if n else 'No spots yet')
        s = self.cur_spot()
        self.spot_mode.set('Heal (match surroundings)' if (s['mode'] if s else self.spot['mode']) == 'heal' else 'Clone (exact copy)')
        self.spot_size.sync(); self.spot_feather.sync()

    def spot_hit(self, X, Y):
        lst = self.params['retouch']; W, H = self.view_wh(); k = W / self.view_dims()[0] * self.frame_diag
        for i in range(len(lst) - 1, -1, -1):
            s = lst[i]
            if s['kind'] != 'spot': continue
            R = max(6, s['r'] * k)
            if math.dist((X, Y), self.from_uv(*self.s2c(*self.to_img(s['x'], s['y'])))) < R: return i, 'target'
            if math.dist((X, Y), self.from_uv(*self.s2c(*self.to_img(s['sx'], s['sy'])))) < R: return i, 'source'
        return None

    def add_spot(self, u, v):
        xi, yi = self.c2s(u, v); x, y = self.to_frame(xi, yi); r = self.spot['size']
        s = dict(kind='spot', x=x, y=y, sx=min(1, x + 3 * r), sy=y, r=r, feather=self.spot['feather'], mode=self.spot['mode'], opacity=1)
        rt = self.worker_retouch(self.params)
        self.params['retouch'].append(s); self.sel_spot = len(self.params['retouch']) - 1
        self.draw_overlay(); self.render_spot_ui()
        try:
            with self.lock:
                self.dv.sync_retouch(rt)
                sx, sy = V.spot_source(self.dv.prev, xi, yi, r * self.img_diag_scale())
            s['sx'], s['sy'] = self.to_frame(sx, sy)
        except Exception:
            traceback.print_exc()
        self.changed(); self.draw_overlay(); self.commit('Spot added')

    def delete_spot(self, i):
        lst = self.params['retouch'] if self.params else []
        if not (0 <= i < len(lst)) or lst[i]['kind'] != 'spot': return
        del lst[i]; self.sel_spot = -1; self.changed(); self.draw_overlay(); self.render_spot_ui(); self.commit('Spot removed')

    def clear_spots(self):
        if not self.ready(): return
        self.params['retouch'] = [x for x in self.params['retouch'] if x['kind'] != 'spot']; self.sel_spot = -1
        self.changed(); self.draw_overlay(); self.render_spot_ui(); self.commit('Spots cleared')

    # ─── masks
    def cur_mask(self):
        if self.params is None: return None
        return next((m for m in self.params['masks'] if m['id'] == self.sel_mask), None)

    def new_mask(self, name=None):
        m = dict(id=new_id(), name=name or f"Mask {len(self.params['masks']) + 1}", invert=False, comps=[], adj={})
        self.params['masks'].append(m); self.sel_mask = m['id']
        return m

    def ensure_mask(self):
        m = self.cur_mask()
        if not m: m = self.new_mask(); self.render_mask_ui()
        return m

    def new_mask_cmd(self):
        if not self.ready(): return
        if self.sam_prompt and self.sam_prompt.get('comp') is not None: self.commit('SAM2 selection')
        self.clear_sam_prompt(); self.new_mask(); self.render_mask_ui(); self.commit('New mask')

    def delete_mask(self):
        m = self.cur_mask()
        if not m: return
        self.clear_sam_prompt()
        self.params['masks'] = [x for x in self.params['masks'] if x is not m]; self.sel_mask = None
        self.render_mask_ui(); self.changed(); self.commit(f"Mask deleted ({m['name']})")

    def mask_selected(self):
        ms = self.params['masks'] if self.params else []
        i = self.mask_box.current()
        if not (0 <= i < len(ms)): return
        if self.sam_prompt and self.sam_prompt.get('comp') is not None: self.commit('SAM2 selection')
        self.clear_sam_prompt()
        self.sel_mask = ms[i]['id']; self.render_mask_ui(); self.request()

    def rename_mask(self):
        m = self.cur_mask()
        if m and self.mask_name.get() and self.mask_name.get() != m['name']:
            m['name'] = self.mask_name.get(); self.render_mask_ui(); self.commit('Mask renamed')

    def invert_mask(self):
        m = self.cur_mask()
        if m: m['invert'] = self.mask_invert.get(); self.changed(); self.commit(f"Mask {'inverted' if m['invert'] else 'not inverted'}")

    def overlay_toggled(self):
        self.show_overlay = self.overlay_var.get(); self.request()

    def mask_tool(self, k):
        if not self.ready(): return
        if k == 'erase':
            self.brush['erase'] = not self.brush['erase']
            if self.tool not in MASK_TOOLS: self.ensure_mask(); self.set_tool('brush')
            if self.sam_prompt and self.sam_prompt.get('comp') is None: self.sam_prompt['erase'] = self.brush['erase']
            self.render_mask_ui(); return
        if self.tool == k: self.set_tool(None)
        else:
            self.ensure_mask(); self.set_tool(k)
            if k == 'fill': threading.Thread(target=self.retouch_snapshot, daemon=True).start()
        self.render_mask_ui()

    def render_mask_ui(self):
        if self.params is None: return
        ms = self.params['masks']
        if not any(m['id'] == self.sel_mask for m in ms): self.sel_mask = ms[-1]['id'] if ms else None
        self.mask_box.configure(values=[m['name'] + ('' if m['comps'] else ' (empty)') for m in ms] or ['(no masks)'])
        m = self.cur_mask()
        self.mask_sel.set((m['name'] + ('' if m['comps'] else ' (empty)')) if m else '(no masks)')
        if m:
            if not self.mask_body.winfo_manager(): self.mask_body.pack(fill='x', after=self.sam_hint)
            self.mask_name.set(m['name']); self.mask_invert.set(bool(m.get('invert')))
        elif self.mask_body.winfo_manager(): self.mask_body.pack_forget()
        self.overlay_var.set(self.show_overlay)
        for k, b in self.tool_btns.items():
            on = self.brush['erase'] if k == 'erase' else self.tool == k
            b.configure(style='Accent.TButton' if on else 'Small.TButton', text=('Erase: on' if self.brush['erase'] else 'Erase') if k == 'erase' else b.cget('text'))
        for s in self.sliders:
            if s.path is None and s.row.master is self.mask_body: s.sync()
        self.parts.delete(0, 'end')
        if m:
            nb = ns = nf = 0; n = {}
            for c in m['comps']:
                if c['type'] == 'sam':
                    ns += 1; pr = c.get('prompt')
                    name = f'SAM2 selection {ns}' + (f" ({len(pr['points'])} point{'s' if len(pr['points']) != 1 else ''}{', box' if pr.get('box') else ''})" if pr else '')
                elif c['type'] in ('poly', 'grad'):
                    kind = 'Gradient' if c['type'] == 'grad' else {'rect': 'Rectangle', 'polygon': 'Polygon'}.get(c.get('shape')) or ('Magnetic lasso' if c.get('magnetic') else 'Lasso')
                    n[kind] = n.get(kind, 0) + 1; name = f'{kind} {n[kind]}'
                elif c['type'] == 'fill': nf += 1; name = c.get('label') or f'Fill {nf}'
                else: nb += 1; name = f'Brush stroke {nb}'
                editing = self.sam_prompt is not None and self.sam_prompt.get('comp') is c
                self.parts.insert('end', f"{'−' if c.get('erase') else '+'} {name}{'  (editing)' if editing else ''}")

    def _sel_part(self):
        m = self.cur_mask(); sel = self.parts.curselection()
        return (m, sel[0]) if m and sel and sel[0] < len(m['comps']) else (m, None)

    def part_toggle(self):
        m, i = self._sel_part()
        if i is None: return
        c = m['comps'][i]; c['erase'] = not c.get('erase'); self.changed(); self.render_mask_ui()
        self.commit(f"Mask part set to {'subtract' if c['erase'] else 'add'}")

    def part_delete(self):
        m, i = self._sel_part()
        if i is None: return
        c = m['comps'][i]
        if self.sam_prompt and self.sam_prompt.get('comp') is c: self.sam_prompt.update(comp=None, points=[], box=None); self.draw_overlay()
        del m['comps'][i]; self.changed(); self.render_mask_ui(); self.commit(f"Mask part deleted ({m['name']})")

    def part_picked(self):
        m, i = self._sel_part()
        if i is not None and self.tool == 'grad' and m['comps'][i]['type'] == 'grad': self.grad_sel = m['comps'][i]; self.draw_overlay()

    def part_edit(self):
        m, i = self._sel_part()
        if i is None: return
        c = m['comps'][i]
        if c['type'] == 'grad':
            if self.tool != 'grad': self.set_tool('grad')
            self.grad_sel = c; self.draw_overlay(); return
        if c['type'] != 'sam' or not c.get('prompt'): return
        if self.tool != 'sam': self.set_tool('sam')
        elif self.sam_prompt and self.sam_prompt.get('comp') not in (None, c): self.commit('SAM2 selection')
        self.sam_prompt = dict(self.prompt_from_frame(c['prompt']), comp=c, erase=c.get('erase'))
        self.draw_overlay(); self.render_mask_ui()
        self.sam_hint.configure(text='Editing this selection: click to add points, right-click to exclude, click a point (or Backspace) to remove it, drag for a new box.')

    def show_brush_size(self):
        px = self.brush['size'] * (self.frame_diag or 3000)
        self.brush_size_lbl.configure(text=(f'{px:.1f} px' if px < 10 else f'{round(px)} px'))
        self.brush_size.sync()

    def set_brush_size(self, v):
        self.brush['size'] = min(BR_MIN * BR_RANGE, max(BR_MIN, v)); self.show_brush_size(); self.draw_overlay()

    def stroke_preview(self, c):
        """A brush stroke in progress, drawn straight on the canvas."""
        W, H = self.view_wh(); r = c['size'] * self.frame_diag * W / self.view_dims()[0]
        pts = c['pts'][-2:]
        cs = [self.from_uv(*self.s2c(*self.to_img(*q))) for q in pts]
        if len(cs) == 1: cs = cs * 2
        self.cv.create_line(*cs[0], *cs[1], fill='#ff2850' if not c['erase'] else '#2850ff', width=max(1, 2 * r), capstyle='round', tags='ov')

    # lassos
    def lasso_add(self, erase):
        L, self.lasso = self.lasso, None
        if not L or len(L['pts']) < 3: self.draw_overlay(); return
        m = self.ensure_mask(); step = max(1, len(L['pts']) // 2000)
        pts = [list(self.to_frame(*self.c2s(u, v))) for u, v in L['pts'][::step]]
        sub = bool(erase) != bool(self.brush['erase'])
        c = dict(type='poly', pts=pts, erase=sub, featherPx=float(self.lasso_feather.get()), magnetic=bool(L.get('magnetic')))
        if L.get('polygon'): c['shape'] = 'polygon'
        m['comps'].append(c)
        self.changed(); self.render_mask_ui(); self.draw_overlay()
        self.commit(f"{'Magnetic lasso' if L.get('magnetic') else 'Polygon' if L.get('polygon') else 'Lasso'}{' (subtract)' if sub else ''}")

    def poly_click(self, u, v):
        L = self.lasso
        if not L:
            self.lasso = dict(pts=[(u, v)], anchors=[0], polygon=True, preview=[]); self.draw_overlay(); return
        W, H = self.view_wh(); first = L['pts'][0]
        if len(L['pts']) > 2 and math.hypot((u - first[0]) * W, (v - first[1]) * H) < 9: self.lasso_close(); return
        L['pts'].append((u, v)); L['anchors'].append(len(L['pts']) - 1); L['preview'] = []
        self.draw_overlay()

    def rect_add(self, p, q, erase):
        W, H = self.view_wh()
        if abs(q[0] - p[0]) * W < 3 or abs(q[1] - p[1]) * H < 3: self.draw_overlay(); return
        m = self.ensure_mask()
        pts = [list(self.to_frame(*self.c2s(u, v))) for u, v in ((p[0], p[1]), (q[0], p[1]), (q[0], q[1]), (p[0], q[1]))]
        sub = bool(erase) != bool(self.brush['erase'])
        m['comps'].append(dict(type='poly', pts=pts, erase=sub, featherPx=float(self.lasso_feather.get()), magnetic=False, shape='rect'))
        self.changed(); self.render_mask_ui(); self.draw_overlay()
        self.commit(f"Rectangle{' (subtract)' if sub else ''}")

    # linear gradients: a (full effect) and b (none) in frame coordinates, worked on in canvas pixels
    def grad_xy(self, q):
        return self.from_uv(*self.s2c(*self.to_img(*q)))

    def grad_frame(self, X, Y):
        return list(self.to_frame(*self.c2s(*self.to_uv(X, Y))))

    def grads(self):
        return [c for c in (self.cur_mask() or {}).get('comps', []) if c['type'] == 'grad']

    def draw_grads(self):
        cv, C = self.cv, self.C
        gs = self.grads()
        if not any(c is self.grad_sel for c in gs): self.grad_sel = None
        E = 4 * (self.cv.winfo_width() + self.cv.winfo_height())
        for c in gs:
            (ax, ay), (bx, by) = self.grad_xy(c['a']), self.grad_xy(c['b'])
            mx, my = (ax + bx) / 2, (ay + by) / 2
            if c is self.grad_sel:
                L = math.hypot(bx - ax, by - ay) or 1; nx, ny = -(by - ay) / L * E, (bx - ax) / L * E
                col = C['err'] if c.get('erase') else '#ffffff'
                for (X, Y), dash in (((ax, ay), ()), ((mx, my), (6, 4)), ((bx, by), (2, 4))):
                    cv.create_line(X - nx, Y - ny, X + nx, Y + ny, fill='#000000', width=3, tags='ov')
                    cv.create_line(X - nx, Y - ny, X + nx, Y + ny, fill=col, width=1, dash=dash, tags='ov')
                cv.create_line(ax, ay, bx, by, fill=col, dash=(2, 3), tags='ov')
                for X, Y in ((ax, ay), (bx, by)): cv.create_oval(X - 6, Y - 6, X + 6, Y + 6, fill='#fff', outline='#000', width=2, tags='ov')
            cv.create_oval(mx - 7, my - 7, mx + 7, my + 7, fill=C['accent'] if c is self.grad_sel else '#888888', outline='#000', width=2, tags='ov')

    def grad_hit(self, X, Y):
        """(gradient, part): part 'a' / 'b' (an end), 'mid', 'rot' (the centre line), 'aline' / 'bline' (an outer line); or None."""
        gs = self.grads(); sel = self.grad_sel if any(c is self.grad_sel for c in gs) else None
        if sel:
            for part in ('a', 'b'):
                if math.dist((X, Y), self.grad_xy(sel[part])) < 10: return sel, part
        for c in ([sel] if sel else []) + [c for c in reversed(gs) if c is not sel]:
            (ax, ay), (bx, by) = self.grad_xy(c['a']), self.grad_xy(c['b'])
            if math.dist((X, Y), ((ax + bx) / 2, (ay + by) / 2)) < 10: return c, 'mid'
        if sel:
            (ax, ay), (bx, by) = self.grad_xy(sel['a']), self.grad_xy(sel['b'])
            L = math.hypot(bx - ax, by - ay) or 1; dx, dy = (bx - ax) / L, (by - ay) / L
            t = (X - ax) * dx + (Y - ay) * dy   # distance along a -> b
            for part, at in (('rot', L / 2), ('aline', 0), ('bline', L)):
                if abs(t - at) < 6: return sel, part
        return None

    def grad_press(self, e, alt):
        h = self.grad_hit(e.x, e.y)
        if h:
            c, part = h; self.grad_sel = c
        else:
            m = self.ensure_mask(); q = self.grad_frame(e.x, e.y)
            c = dict(type='grad', a=q, b=list(q), erase=bool(alt) != bool(self.brush['erase']))
            m['comps'].append(c); self.grad_sel = c; part = 'b'; self.render_mask_ui()
        self._drag = dict(kind='grad', c=c, part=part, p=(e.x, e.y), a=self.grad_xy(c['a']), b=self.grad_xy(c['b']), new=not h, moved=False)
        self.draw_overlay()

    def grad_motion(self, d, e):
        c, part = d['c'], d['part']; (ax, ay), (bx, by) = d['a'], d['b']
        dx, dy = e.x - d['p'][0], e.y - d['p'][1]
        if part == 'a': A, B = (ax + dx, ay + dy), (bx, by)
        elif part == 'b': A, B = (ax, ay), (bx + dx, by + dy)
        elif part == 'mid': A, B = (ax + dx, ay + dy), (bx + dx, by + dy)
        elif part == 'rot':   # turn about the middle, keeping the length
            mx, my = (ax + bx) / 2, (ay + by) / 2
            r = math.atan2(e.y - my, e.x - mx) - math.atan2(d['p'][1] - my, d['p'][0] - mx)
            cs, sn = math.cos(r), math.sin(r)
            turn = lambda x, y: (mx + (x - mx) * cs - (y - my) * sn, my + (x - mx) * sn + (y - my) * cs)
            A, B = turn(ax, ay), turn(bx, by)
        else:   # an outer line: along the gradient's direction only, never past the other line
            L = math.hypot(bx - ax, by - ay) or 1; ux, uy = (bx - ax) / L, (by - ay) / L
            s = dx * ux + dy * uy
            if part == 'aline': s = min(s, L - 2); A, B = (ax + s * ux, ay + s * uy), (bx, by)
            else: s = max(s, 2 - L); A, B = (ax, ay), (bx + s * ux, by + s * uy)
        if math.dist(A, B) < 2: return
        c['a'], c['b'] = self.grad_frame(*A), self.grad_frame(*B)
        d['moved'] = True; self.draw_overlay(); self.changed()

    def grad_release(self, d):
        c = d['c']
        if d['new'] and math.dist(self.grad_xy(c['a']), self.grad_xy(c['b'])) < 4:   # a click, not a drag
            m = self.cur_mask()
            if m: m['comps'] = [x for x in m['comps'] if x is not c]
            self.grad_sel = None; self.render_mask_ui(); self.draw_overlay(); return
        if d['moved']:
            self.changed(); self.render_mask_ui()
            self.commit(('Gradient' + (' (subtract)' if c.get('erase') else '')) if d['new'] else 'Gradient adjusted')

    def grad_delete(self, c):
        m = self.cur_mask()
        if not m or c is None or not any(x is c for x in m['comps']): return
        m['comps'] = [x for x in m['comps'] if x is not c]; self.grad_sel = None
        self.changed(); self.render_mask_ui(); self.draw_overlay(); self.commit(f"Gradient deleted ({m['name']})")

    def retouch_snapshot(self):
        """The picture as shown in the retouch view (for the magnetic lasso and the fill tool)."""
        p = self.params
        key = json.dumps([p['geometry']['rot90'], p['retouch'], self.frame, self.img_w, p['exposure'], p['contrast'], p['temp'], p['tint']])
        if self.snap and self.snap[0] == key: return self.snap[1]
        wp = self.view_params('source')
        with self.lock:
            t = self.dv.preview_target(wp, 2560, uncropped=True)
            rgb = V.to_srgb8(np.ascontiguousarray(V.develop(t, wp, t.scale, self.dv.D, self.dv.airlight), np.float32))
        self.snap = (key, rgb)
        return rgb

    def lasso_prepare(self):
        def work():
            try:
                rgb = self.retouch_snapshot()
                self.lw = V.edge_cost(rgb)
                V.live_wire(self.lw, (0.1, 0.1), (0.12, 0.12))   # compile now
            except Exception: traceback.print_exc()
        threading.Thread(target=work, daemon=True).start()

    def lasso_click(self, u, v):
        L = self.lasso
        if not L:
            self.lasso = dict(pts=[(u, v)], anchors=[0], magnetic=True, preview=[]); self.draw_overlay(); return
        W, H = self.view_wh()
        first = L['pts'][0]
        if len(L['anchors']) > 2 and math.hypot((u - first[0]) * W, (v - first[1]) * H) < 9: self.lasso_close(); return
        seg = V.live_wire(self.lw, L['pts'][-1], (u, v)) if self.lw is not None else [(u, v)]
        L['pts'] += seg; L['anchors'].append(len(L['pts']) - 1); L['preview'] = []
        self.draw_overlay()

    def lasso_close(self, erase=None):
        L = self.lasso
        if not L: return
        if L.get('magnetic') and len(L['pts']) > 1 and self.lw is not None: L['pts'] += V.live_wire(self.lw, L['pts'][-1], L['pts'][0])
        self.lasso_add(getattr(self, 'lasso_erase', False) if erase is None else erase)

    def lasso_back(self):
        L = self.lasso
        if not L: return
        L['anchors'].pop()
        if not L['anchors']: self.lasso = None
        else: del L['pts'][L['anchors'][-1] + 1:]
        self.draw_overlay()

    def lasso_move(self, u, v):
        L = self.lasso
        if not L or not L.get('magnetic') or self.lw is None: return
        self._lw_pending = (u, v)
        if getattr(self, '_lw_busy', False): return
        self._lw_busy = True

        def go():
            self._lw_busy = False
            Lc = self.lasso; q = getattr(self, '_lw_pending', None); self._lw_pending = None
            if not Lc or q is None: return
            pv = V.live_wire(self.lw, Lc['pts'][-1], q)
            seg = 0.06 * max(self.lw.shape)
            if len(pv) > seg:
                keep = len(pv) - round(seg / 3)
                Lc['pts'] += pv[:keep]; Lc['anchors'].append(len(Lc['pts']) - 1); pv = pv[keep:]
            Lc['preview'] = pv; self.draw_overlay()
        self.root.after(15, go)

    # fill (bucket)
    def canvas_mask_to_bitmap(self, mk):
        """A soft image-orientation bitmap at raster size from a mask over the (rot90) canvas."""
        W, H = self.raster_dims(); h, w = mk.shape
        yy, xx = np.mgrid[0:H, 0:W]
        u, v = self.s2c_arr((xx + 0.5) / W, (yy + 0.5) / H)
        fx = np.clip(u * w - 0.5, 0, w - 1.001); fy = np.clip(v * h - 0.5, 0, h - 1.001)
        a = V._sample3(mk[..., None].astype(np.float64), fx, fy)[..., 0]
        return dict(w=W, h=H, data=np.floor(a).astype(np.uint8), frame=dict(self.frame))

    def s2c_arr(self, x, y):
        k = self.params['geometry']['rot90'] & 3
        return (x, y) if k == 0 else (1 - y, x) if k == 1 else (1 - x, 1 - y) if k == 2 else (y, 1 - x)

    def fill_at(self, u, v, erase):
        from scipy.ndimage import label
        try: S = self.retouch_snapshot()
        except Exception as ex: self.app.log('err', f'Fill: {ex}'); return
        h, w = S.shape[:2]
        x0, y0 = min(w - 1, max(0, int(u * w))), min(h - 1, max(0, int(v * h)))
        tol = self.fill_tol.get() * 2.55
        ok = np.abs(S.astype(np.int16) - S[y0, x0].astype(np.int16)).max(-1) <= tol
        lab, _ = label(ok)
        mk = np.where(lab == lab[y0, x0], 255, 0).astype(np.uint8)
        ref = new_id(); m = self.ensure_mask()
        self.bitmaps[ref] = self.canvas_mask_to_bitmap(mk)
        sub = bool(erase) != bool(self.brush['erase'])
        m['comps'].append(dict(type='fill', ref=ref, erase=sub, tol=self.fill_tol.get()))
        self.changed(); self.render_mask_ui()
        self.commit(f"Fill{' (subtract)' if sub else ''} ({(mk > 0).mean() * 100:.1f}% of the picture)")

    # SAM2
    def sam_prepare(self):
        p = self.params
        key = json.dumps([p['geometry']['rot90'], p['retouch'], self.frame, self.img_w])
        if self.sam and self.sam[0] == key: return
        self.sam = (key, None)
        self.app.status.configure(text='SAM2 is looking at the picture…')
        wp = self.view_params('source')
        self.app.models.set(self.app.models_dir.get(), self.app.device_key())

        def work():
            try:
                with self.lock:
                    t = self.dv.preview_target(wp, 1024, uncropped=True)
                    rgb = V.to_srgb8(np.ascontiguousarray(V.develop(t, wp, t.scale, self.dv.D, self.dv.airlight), np.float32))
                pred = self.app.models.sam2('large')
                import torch
                with torch.inference_mode(): pred.set_image(rgb)
                self.app.q.put(('call', self._sam_ready, key, pred, rgb.shape))
            except Exception as ex:
                traceback.print_exc(); self.app.q.put(('log', 'err', f'SAM2: {ex}')); self.sam = None
        threading.Thread(target=work, daemon=True).start()

    def _sam_ready(self, key, pred, shape):
        if self.sam and self.sam[0] == key:
            self.sam = (key, pred, shape); self.app.status.configure(text='SAM2 ready.')
            if self.sam_prompt and (self.sam_prompt['points'] or self.sam_prompt.get('box')): self.sam_update()

    def prompt_to_frame(self, pr):
        pts = [[*self.to_frame(*self.c2s(u, v)), l] for u, v, l in pr['points']]
        box = None
        if pr.get('box'): box = [*self.to_frame(*self.c2s(pr['box'][0], pr['box'][1])), *self.to_frame(*self.c2s(pr['box'][2], pr['box'][3]))]
        return dict(points=pts, box=box)

    def prompt_from_frame(self, q):
        pts = [[*self.s2c(*self.to_img(x, y)), l] for x, y, l in q.get('points') or []]
        box = None
        if q.get('box'):
            a = self.s2c(*self.to_img(q['box'][0], q['box'][1])); b = self.s2c(*self.to_img(q['box'][2], q['box'][3]))
            box = [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]
        return dict(points=pts, box=box)

    def clear_sam_prompt(self):
        if not self.sam_prompt: return
        self.sam_prompt.update(points=[], box=None, comp=None)
        self.sam_hint.configure(text='Click the object to select (right-click excludes, drag for a box).' if self.tool == 'sam' else '')
        self.draw_overlay()

    def sam_update(self):
        pr = self.sam_prompt
        if not pr: return
        if not pr['points'] and not pr.get('box'):
            m = self.cur_mask()
            if m and pr.get('comp') is not None:
                m['comps'] = [c for c in m['comps'] if c is not pr['comp']]; pr['comp'] = None; self.changed(); self.render_mask_ui()
            return
        if not self.sam or len(self.sam) < 3 or self.sam[1] is None:
            self.sam_prepare(); return
        key, pred, shape = self.sam
        h, w = shape[:2]
        import torch
        pts = np.array([[q[0] * w, q[1] * h] for q in pr['points']], np.float32) if pr['points'] else None
        lab = np.array([q[2] for q in pr['points']], np.int32) if pr['points'] else None
        box = np.array([pr['box'][0] * w, pr['box'][1] * h, pr['box'][2] * w, pr['box'][3] * h], np.float32) if pr.get('box') else None
        try:
            with torch.inference_mode():
                masks, scores, _ = pred.predict(point_coords=pts, point_labels=lab, box=box, multimask_output=True, return_logits=True)
        except Exception as ex:
            self.app.log('err', f'SAM2: {ex}'); return
        best = int(np.argmax(scores)); L = masks[best].astype(np.float32)
        W, H = self.raster_dims(); mh, mw = L.shape
        yy, xx = np.mgrid[0:H, 0:W]
        u, v = self.s2c_arr((xx + 0.5) / W, (yy + 0.5) / H)
        fx = np.clip(u * mw - 0.5, 0, mw - 1.001); fy = np.clip(v * mh - 0.5, 0, mh - 1.001)
        l = V._sample3(L[..., None].astype(np.float64), fx, fy)[..., 0]
        a = np.floor(255 / (1 + np.exp(-2 * l))).astype(np.uint8)
        ref = new_id(); self.bitmaps[ref] = dict(w=W, h=H, data=a, frame=dict(self.frame))
        m = self.ensure_mask()
        if pr.get('comp') is None or not any(c is pr['comp'] for c in m['comps']):
            pr['comp'] = dict(type='sam', ref=ref, erase=bool(pr.get('erase'))); m['comps'].append(pr['comp'])
        else: pr['comp']['ref'] = ref
        pr['comp']['prompt'] = self.prompt_to_frame(pr)
        self.changed(); self.render_mask_ui()
        self.sam_hint.configure(text=f'Selection score {scores[best] * 100:.0f}%. Click to add, right-click to exclude, drag a box; click a point (or Backspace) to remove it; Shift-click starts another selection.')

    # AI removal and outpainting
    def ai_model_key(self):
        return next(k for k, v in AI_MODELS if v == self.ai_model.get())

    def inpaint_mask(self, m, model, job, existing=None):
        key, r = self.rasterize(m)
        ys, xs = np.nonzero(r > 100)
        if not len(ys): raise ValueError('the mask is empty')
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        H, W = r.shape
        side = max(x1 - x0, y1 - y0) + 1; mg = max(side * 0.6, 24); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; half = side / 2 + mg
        rect = dict(x=max(0, (cx - half) / W), y=max(0, (cy - half) / H))
        rect['w'] = min(1, (cx + half) / W) - rect['x']; rect['h'] = min(1, (cy + half) / H) - rect['y']
        with self.lock: reg = self.dv.source_region(self.worker_retouch(self.params), rect, 1024 if model == 'migan' else 768)
        rr = reg['rect']
        yy, xx = np.mgrid[0:reg['h'], 0:reg['w']]
        u = rr['x'] + (xx + 0.5) / reg['w'] * rr['w']; v = rr['y'] + (yy + 0.5) / reg['h'] * rr['h']
        hole = (r[np.minimum(H - 1, (v * H).astype(int)), np.minimum(W - 1, (u * W).astype(int))] > 100).astype(np.uint8)
        hole = V.dilate(hole, max(2, round(max(reg['w'], reg['h']) / 250)))
        job.progress(0.3, f'{dict(AI_MODELS)[model].split(" (")[0]}: filling…')
        out = self.inpainter().inpaint(model, V.lin_to_display(reg['data']), hole)
        (fx0, fy0), (fx1, fy1) = self.to_frame(rr['x'], rr['y']), self.to_frame(rr['x'] + rr['w'], rr['y'] + rr['h'])
        fid = existing or new_id()
        self.fills[fid] = dict(rect=dict(x=fx0, y=fy0, w=fx1 - fx0, h=fy1 - fy0), w=reg['w'], h=reg['h'], data=V.display_to_lin(out), alpha=V.feather_alpha(hole, 2))
        with self.lock: self.dv.fills.pop(fid, None)
        bm = 'f' + fid
        self.bitmaps[bm] = dict(w=W, h=H, data=r.copy(), frame=dict(self.frame))
        return dict(kind='fill', id=fid, model=model, mask=bm)

    def inpainter(self):
        if getattr(self, '_inp', None) is None or self._inp.models is not self.app.models: self._inp = V.Inpainter(self.app.models)
        return self._inp

    def ai_remove(self):
        m = self.cur_mask(); model = self.ai_model_key()
        if not self.ready() or not m or not m['comps']:
            messagebox.showinfo(APP, 'Make a mask over what to remove first (brush, lasso or SAM2), then Remove.'); return
        if self.tool: self.end_tool(True)

        def done(it):
            self.params['retouch'].append(it); self.changed()
            self.commit(f"Removed with {dict(AI_MODELS)[model].split(' (')[0]} ({m['name']})")
        self.app.run_task('AI removal', lambda job: self.inpaint_mask(m, model, job), done)

    def outpaint(self):
        if not self.ready(): return
        if not self.params['geometry'].get('keepEmpty'):
            messagebox.showinfo(APP, 'Tick "Keep empty areas" first (after a perspective, rotation or lens correction).'); return
        if self.tool: self.end_tool(True)
        model = self.ai_model_key(); wp = self.worker_params()

        def work(job):
            with self.lock: gkey, wins = self.dv.outpaint_windows(wp)
            if not wins: raise ValueError('there are no empty areas (turn on "Keep empty areas", after a perspective or lens correction)')
            patches = []
            for i, w in enumerate(wins):
                job.check(); job.progress(i / len(wins), f'Outpainting area {i + 1} of {len(wins)}')
                hole = V.dilate((w['valid'] == 0).astype(np.uint8), 2)
                out = self.inpainter().inpaint(model, V.lin_to_display(w['data']), hole)
                patches.append(dict(u0=w['u0'], v0=w['v0'], u1=w['u1'], v1=w['v1'], w=w['w'], h=w['h'], data=V.display_to_lin(out)))
            oid = new_id()
            self.op_layers[oid] = dict(gkey=gkey, patches=patches)
            return dict(id=oid, gkey=gkey, model=model)

        def done(o):
            self.params['outpaint'] = o; self.changed(); self.render_outpaint_ui(); self.commit(f"Outpainted with {dict(AI_MODELS)[model].split(' (')[0]}")
        self.app.run_task('Outpainting', work, done)

    def render_outpaint_ui(self):
        if self.params is None: return
        o = self.params.get('outpaint')
        self.outpaint_btn.configure(state='normal' if self.params['geometry'].get('keepEmpty') else 'disabled')
        note = ''
        if o:
            L = self.op_layers.get(o['id'])
            wp = self.worker_params() if self.dv is not None else None
            note = ('Outpainted areas will be recreated.' if not L else
                    'The geometry has changed since the outpainting: outpaint again.' if wp and L['gkey'] != V.Develop.canvas_key(wp['geometry'], wp.get('lens')) else 'The empty areas are outpainted.')
        self.outpaint_note.configure(text=note)

    # ─── settings (the web app's format)
    def chosen_groups(self):
        return [k for k, v in self.groups.items() if v.get()]

    @staticmethod
    def bitmap_png(bm):
        a = bm['data']
        im = Image.fromarray(np.dstack([np.full_like(a, 255)] * 3 + [a]), 'RGBA')
        b = io.BytesIO(); im.save(b, 'PNG')
        return base64.b64encode(b.getvalue()).decode('ascii')

    @staticmethod
    def png_bitmap(b64, w, h):
        im = Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGBA').resize((w, h), Image.BILINEAR)
        return np.asarray(im)[..., 3].copy()

    def settings_object(self, groups):
        params = {}
        for g in groups:
            for k in GROUPS[g][1]:
                if k in self.params: params[k] = copy.deepcopy(self.params[k])
        refs = set()
        for m in params.get('masks') or []:
            for c in m['comps']:
                if c['type'] in ('sam', 'fill'): refs.add(c['ref'])
        for it in params.get('retouch') or []:
            if it['kind'] == 'fill': refs.add(it['mask'])
        bitmaps = {r: dict(w=bm['w'], h=bm['h'], frame=bm['frame'], png=self.bitmap_png(bm)) for r in refs if (bm := self.bitmaps.get(r))}
        if params.get('outpaint') and params['outpaint']['id'] not in self.op_layers: params.pop('outpaint')
        return dict(app='RAWDemosaic', kind='develop-settings', version=3, **{'from': self.info.get('name', '')}, groups=groups, params=params, bitmaps=bitmaps)

    def apply_settings(self, obj, groups):
        if not obj or obj.get('kind') != 'develop-settings': raise ValueError('this is not a RAWDemosaic settings file')
        if self.tool: self.end_tool(True)
        self.clear_sam_prompt()
        for r, b in (obj.get('bitmaps') or {}).items():
            if r not in self.bitmaps: self.bitmaps[r] = dict(w=b['w'], h=b['h'], frame=b['frame'], data=self.png_bitmap(b['png'], b['w'], b['h']))
        applied = []
        for g in groups:
            keys = [k for k in GROUPS[g][1] if k in obj['params']]
            if not keys: continue
            for k in keys: self.params[k] = copy.deepcopy(obj['params'][k])
            ver = obj.get('version') or 1
            if 'hsl' in keys and ver < 3:   # the mixer's hue ±100 was ±30° (version 1), ±60° (version 2); now ±180°
                self.params['hsl']['hue'] = [v * (30 if ver < 2 else 60) / 180 for v in self.params['hsl']['hue']]
            applied.append(GROUPS[g][0])
        if not applied: raise ValueError('nothing to apply with the groups ticked')
        self.params.setdefault('outpaint', None)
        self.refit_crop(); self.sync_all(); self.changed(type('P', (), {'path': ['lens']})())
        fills = [it for it in self.params['retouch'] if it['kind'] == 'fill' and it['id'] not in self.fills]
        op = self.params.get('outpaint') if self.params.get('outpaint') and self.params['outpaint']['id'] not in self.op_layers else None
        if fills or op:
            def work(job):
                for i, it in enumerate(fills):
                    job.progress(i / max(1, len(fills)), f'Recreating AI removal {i + 1} of {len(fills)}')
                    bm = self.bitmaps.get(it['mask'])
                    if bm: self.inpaint_mask(dict(comps=[dict(type='sam', ref=it['mask'])], invert=False, id='tmp'), it['model'], job, it['id'])
                return True

            def done(_):
                if op:
                    self.params['outpaint'] = None
                    self.outpaint_model = op['model']; self.ai_model.set(dict(AI_MODELS)[op['model']]); self.outpaint()
                self.changed()
            self.app.run_task('Recreating AI removals', work, done)
        return applied

    def copy_settings(self):
        if not self.ready(): return
        obj = self.settings_object(self.chosen_groups()); self.clip = obj
        txt = json.dumps(obj)
        try: self.root.clipboard_clear(); self.root.clipboard_append(txt)
        except Exception: pass
        self.app.log('ok', 'copied: ' + ', '.join(GROUPS[g][0] for g in obj['groups']))

    def paste_settings(self):
        if not self.ready(): return
        obj = self.clip
        if not obj:
            try: obj = json.loads(self.root.clipboard_get())
            except Exception: obj = None
        if not obj: messagebox.showinfo(APP, 'Nothing copied yet.'); return
        try:
            a = self.apply_settings(obj, self.chosen_groups()); self.commit('Settings pasted'); self.app.log('ok', 'pasted: ' + ', '.join(a))
        except Exception as ex: messagebox.showerror(APP, f'Paste failed: {ex}')

    def save_settings(self):
        if not self.ready(): return
        p = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('Develop settings', '*.json')], initialdir=self.app.settings.get('json_dir'),
                                         initialfile=f"{os.path.splitext(self.info.get('name', 'image'))[0]}.develop.json", title='Save develop settings')
        if not p: return
        self.app._remember_dir('json_dir', p)
        with open(p, 'w', encoding='utf-8') as f: json.dump(self.settings_object(self.chosen_groups()), f, indent=1)
        self.app.log('ok', f'develop settings saved to {os.path.basename(p)}')

    def load_settings(self):
        if not self.ready(): return
        p = filedialog.askopenfilename(filetypes=[('Develop settings', '*.json')], initialdir=self.app.settings.get('json_dir'), title='Load develop settings')
        if not p: return
        try:
            with open(p, encoding='utf-8') as f: obj = json.load(f)
            a = self.apply_settings(obj, self.chosen_groups()); self.commit('Settings loaded'); self.app.log('ok', f'loaded {os.path.basename(p)}: ' + ', '.join(a))
        except Exception as ex: messagebox.showerror(APP, f'Could not load the settings: {ex}')

    # ─── export
    def export(self, kind):
        if not self.ready(): return
        if self.tool: self.end_tool(True)
        try: opts = self.app.opts()
        except Exception as ex: messagebox.showerror(APP, str(ex)); return
        stem0 = self.info.get('stem') or os.path.splitext(self.info.get('path') or 'image')[0]
        d = opts['out_dir'] or os.path.dirname(stem0)
        base = os.path.basename(stem0) + '_developed'
        ext = '.tif' if kind == 'tiff' else '.jpg'
        if opts.get('unique', True): base = os.path.basename(E.unique_stem(os.path.join(d, base), (ext,)))
        p = filedialog.asksaveasfilename(defaultextension=ext, initialdir=d, initialfile=base + ext,
                                         filetypes=[('TIFF 16-bit', '*.tif')] if kind == 'tiff' else [('JPEG', '*.jpg')], title='Save the developed image')
        if not p: return
        try: o = dict(opts, tiff=kind == 'tiff', jpeg=kind == 'jpeg', sr=self.sr_opts())
        except Exception as ex: messagebox.showerror(APP, str(ex)); return
        if o['sr']['enabled'] and not any(m['weight'] > 0 for m in o['sr']['models']):
            messagebox.showinfo(APP, 'Add one or more models under Super-resolution (on export), or untick Upscale.'); return
        if self.proof_convert.get():
            try: CM.check_output_profile(self.proof_path)
            except ValueError as ex: messagebox.showerror(APP, str(ex)); return
            o['convert'] = dict(path=self.proof_path, intent=self._intent_key(), bpc=self.proof_bpc.get())
        wp = self.worker_params()

        def work(job):
            job.progress(0.05, f'Developing at full resolution ({"×".join(map(str, self.full_dims()))})')
            with self.lock: full = self.dv.render_full(wp)
            return E.save_outputs(full, os.path.splitext(p)[0], o, self.app.models, job)
        self.app.run_task('Exporting', work, lambda files: self.app.log('ok', 'saved ' + ', '.join(os.path.basename(f) for f in files)))
