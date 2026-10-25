# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""RAW Demosaic desktop — tkinter front end for rawdemosaic_engine.

All heavy work runs on one worker thread; it talks to the UI only through a queue that the Tk loop drains
every 50 ms, so the window never freezes and Tk is only ever touched from the main thread.
"""
import copy
import json
import math
import os
import queue
import threading
import time
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

import rawdemosaic_engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(HERE, 'rawdemosaic_settings.json')
APP = 'RAW Demosaic'
ALT = 0x20000 if os.name == 'nt' else 0x8   # the Alt key's bit in a Tk event's state


def wheel_delta(e):
    """Mouse-wheel direction (+1 up, -1 down) from <MouseWheel> (Windows, macOS) or Button-4/5 (Linux)."""
    if getattr(e, 'num', None) == 4: return 1
    if getattr(e, 'num', None) == 5: return -1
    return 1 if e.delta > 0 else -1


def bind_wheel(widget, fn, all_=False):
    """Bind fn(event, direction) to the mouse wheel on every platform."""
    b = widget.bind_all if all_ else widget.bind
    for seq in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
        b(seq, lambda e: fn(e, wheel_delta(e)))


def unbind_wheel(widget):
    for seq in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
        widget.unbind_all(seq)


def open_folder(d):
    import subprocess
    import sys
    if os.name == 'nt': os.startfile(d)
    else: subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', d])

C = dict(bg='#121417', card='#1d2025', card2='#252930', text='#e2e5ea', muted='#9aa3ad', border='#353b44',
         input='#15181c', accent='#ff6e55', accent_hover='#ff8670', ok='#a5d6a7', err='#ef9a9a', info='#80deea',
         warn='#ffd98a', track='#2c3138', checker='#262a30')

IN_TYPES = [('RAW and images', ' '.join('*' + e for e in E.RAW_EXTS + E.IMAGE_EXTS)), ('RAW files', ' '.join('*' + e for e in E.RAW_EXTS)),
            ('Images', ' '.join('*' + e for e in E.IMAGE_EXTS)), ('All files', '*.*')]
IMG_TYPES = [('Images', ' '.join('*' + e for e in E.IMAGE_EXTS)), ('All files', '*.*')]
RAW_DENOISE = [('none', 'None'), ('nind', 'RAW NIND-Unet (joint demosaic + denoise, Bayer)'), ('rawforge', 'RawForge')]
DEMOSAIC = [('amaze', 'AMaZE'), ('rcd', 'RCD'), ('mhc', 'Malvar (MHC)'), ('bilinear', 'Bilinear'),
            ('xt3', 'X-Trans: Markesteijn 3-pass'), ('xt1', 'X-Trans: Markesteijn 1-pass')]
POST_DENOISE = [('none', 'None'), ('nind_rgb', 'NIND-Unet'), ('nind_linear', 'NIND-Unet (linear / X-Trans)')]
BM3D = [('none', 'None'), ('raw', 'RAW (each CFA plane, before demosaic)'), ('rgb', 'RGB (after demosaic)')]
BOOST_STAGE = [('before', 'Before RAW denoise'), ('after', 'After RAW denoise')]
SR_TARGETS = [('2', '×2'), ('4', '×4'), ('1', '×1 (same size: upscale, then back down)')]
DOWNSCALE = [('lanczos', 'Lanczos-3'), ('box', 'Box (mean)')]
RF_MODELS = E.BAYER_RF + E.XTRANS_RF


def fmt_mp(w, h):
    return f'{w}×{h} ({w * h / 1e6:.0f} MP)'


def key_of(pairs, label, default):
    return next((k for k, v in pairs if v == label), default)


def label_of(pairs, key):
    return dict(pairs).get(key, pairs[0][1])


class Result:
    """A finished image held for viewing: an sRGB uint8 display copy (src: the input file; kind: blend/column)."""
    def __init__(self, title, img, files=(), note='', src=None, kind='blend', quick=False, size=None):
        self.title, self.img, self.files, self.note = title, img, list(files), note
        self.src, self.kind, self.quick = src, kind, quick
        self.size = size or (img.shape[1], img.shape[0])   # the image's real size (the copy shown may be reduced)

    def label(self):
        w, h = self.size
        return f'{self.title} — {w}×{h}'


# ───────────────────────────────────────────── zoomable image view
class ImageView:
    """Canvas showing an RGB uint8 image: fit or zoom (wheel, around the pointer), drag to pan,
    double-click for fit / 100 %. on_click(x, y, button, shift) gets image coordinates;
    overlay(view) draws on top (use view.to_canvas)."""
    def __init__(self, canvas, on_click=None, on_drag=None, overlay=None, pan_button=1):
        self.cv, self.on_click, self.on_drag, self.overlay = canvas, on_click, on_drag, overlay
        self.img = None; self.pil = None; self.label = ''
        self.scale = None; self.cx = self.cy = 0.0
        self._photo = None; self._fitcache = None; self._drag = None
        self.pan_button = pan_button
        cv = canvas
        cv.bind('<Configure>', lambda e: self.redraw())
        bind_wheel(cv, self._wheel)
        cv.bind('<Double-Button-1>', self._toggle)
        for b in (1, 2, 3):
            cv.bind(f'<ButtonPress-{b}>', lambda e, b=b: self._press(e, b))
            cv.bind(f'<B{b}-Motion>', lambda e, b=b: self._motion(e, b))
            cv.bind(f'<ButtonRelease-{b}>', lambda e, b=b: self._release(e, b))

    def set_image(self, img, label='', keep_view=False):
        self.img = img
        self.pil = Image.fromarray(img) if img is not None else None
        self.label = label
        self._fitcache = None
        if not keep_view or img is None:
            self.scale = None
            if img is not None: self.cx, self.cy = img.shape[1] / 2, img.shape[0] / 2
        self.redraw()

    def fit_scale(self):
        h, w = self.img.shape[:2]
        cw, ch = max(10, self.cv.winfo_width()), max(10, self.cv.winfo_height())
        return min((cw - 8) / w, (ch - 8) / h)

    def cur_scale(self):
        return self.scale if self.scale else self.fit_scale()

    def to_canvas(self, x, y):
        s = self.cur_scale()
        cw, ch = self.cv.winfo_width(), self.cv.winfo_height()
        cx, cy = (self.img.shape[1] / 2, self.img.shape[0] / 2) if self.scale is None else (self.cx, self.cy)
        return cw / 2 + (x - cx) * s, ch / 2 + (y - cy) * s

    def to_image(self, X, Y):
        s = self.cur_scale()
        cw, ch = self.cv.winfo_width(), self.cv.winfo_height()
        cx, cy = (self.img.shape[1] / 2, self.img.shape[0] / 2) if self.scale is None else (self.cx, self.cy)
        return cx + (X - cw / 2) / s, cy + (Y - ch / 2) / s

    def redraw(self):
        cv = self.cv
        cv.delete('all')
        cw, ch = max(10, cv.winfo_width()), max(10, cv.winfo_height())
        if self.img is None:
            cv.create_text(cw / 2, ch / 2, text=self.label or 'Nothing to show yet', fill=C['muted'])
            return
        h, w = self.img.shape[:2]
        s = self.cur_scale()
        if self.scale is None:
            tw, th = max(1, int(w * s)), max(1, int(h * s))
            if not self._fitcache or self._fitcache[0] != (tw, th):
                self._fitcache = ((tw, th), self.pil.resize((tw, th), Image.BOX if s < 1 else Image.NEAREST, reducing_gap=3.0 if s < 0.5 else None))
            self._photo = ImageTk.PhotoImage(self._fitcache[1])
            cv.create_image((cw - tw) / 2, (ch - th) / 2, image=self._photo, anchor='nw')
        else:
            x0, y0 = self.cx - cw / 2 / s, self.cy - ch / 2 / s
            ix0, iy0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
            ix1, iy1 = min(w, int(np.ceil(x0 + cw / s)) + 1), min(h, int(np.ceil(y0 + ch / s)) + 1)
            if ix1 > ix0 and iy1 > iy0:
                crop = self.pil.crop((ix0, iy0, ix1, iy1))
                dw, dh = max(1, round((ix1 - ix0) * s)), max(1, round((iy1 - iy0) * s))
                crop = crop.resize((dw, dh), Image.BOX if s < 1 else Image.NEAREST)
                self._photo = ImageTk.PhotoImage(crop)
                cv.create_image((ix0 - x0) * s, (iy0 - y0) * s, image=self._photo, anchor='nw')
        if self.overlay: self.overlay(self)
        zoom = 'fit' if self.scale is None else f'{s * 100:.0f}%'
        cv.create_text(10, ch - 6, anchor='sw', text=f'{self.label}   ·   {w}×{h}   ·   {zoom}', fill=C['text'], font=('Segoe UI', 9))

    def _clamp_center(self):
        h, w = self.img.shape[:2]
        self.cx, self.cy = min(max(self.cx, 0), w), min(max(self.cy, 0), h)

    def _wheel(self, e, d):
        if self.img is None: return
        s0 = self.cur_scale()
        ix, iy = self.to_image(e.x, e.y)
        s1 = s0 * (1.25 if d > 0 else 0.8)
        fit = self.fit_scale()
        if s1 <= fit * 1.0001:
            self.scale = None
        else:
            s1 = min(s1, 16.0)
            cw, ch = self.cv.winfo_width(), self.cv.winfo_height()
            self.scale = s1
            self.cx, self.cy = ix - (e.x - cw / 2) / s1, iy - (e.y - ch / 2) / s1
            self._clamp_center()
        self.redraw()

    def _toggle(self, e):
        if self.img is None: return
        if self.scale is None:
            ix, iy = self.to_image(e.x, e.y)
            self.scale, self.cx, self.cy = 1.0, ix, iy
            self._clamp_center()
        else:
            self.scale = None
        self.redraw()

    def _press(self, e, b):
        if self.img is None: return
        self._drag = (e.x, e.y, self.cx, self.cy, b, False)
        if self.on_drag and b == 1 and (e.state & 0x1):
            ix, iy = self.to_image(e.x, e.y)
            self.on_drag('start', ix, iy)

    def _motion(self, e, b):
        if not self._drag: return
        x0, y0, cx, cy, bb, moved = self._drag
        if abs(e.x - x0) + abs(e.y - y0) > 3: moved = True
        self._drag = (x0, y0, cx, cy, bb, moved)
        if self.on_drag and b == 1 and (e.state & 0x1):
            ix, iy = self.to_image(e.x, e.y)
            self.on_drag('move', ix, iy)
            return
        if moved and (b == self.pan_button or b == 2) and self.scale is not None:
            s = self.scale
            self.cx, self.cy = cx - (e.x - x0) / s, cy - (e.y - y0) / s
            self._clamp_center()
            self.redraw()

    def _release(self, e, b):
        if not self._drag: return
        x0, y0, cx, cy, bb, moved = self._drag
        self._drag = None
        if self.on_drag and b == 1 and (e.state & 0x1):
            ix, iy = self.to_image(e.x, e.y)
            self.on_drag('end', ix, iy)
            return
        if not moved and self.on_click:
            ix, iy = self.to_image(e.x, e.y)
            h, w = self.img.shape[:2]
            if 0 <= ix < w and 0 <= iy < h: self.on_click(ix, iy, b, bool(e.state & 0x1))


# ───────────────────────────────────────────── the app
class App:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.job = None
        self.busy = False
        self.settings = E.load_json(SETTINGS_PATH, {}) or {}
        s = self.settings
        if s.get('defaults', 1) < 2:   # version 2: ProPhoto RGB and column TIFFs by default
            s.setdefault('out', {}).update(space='prophoto', save_columns=True)
            s['defaults'] = 2
        self.files = []                          # input paths (Process tab)
        self.src_cache = {}                      # path -> RawFile / ImageInput (the last two)
        self.results = []
        self.blend_files = []                    # [(path, weight)]
        self.session_tiffs = []                  # column TIFFs written this session
        self.mask_cfg = None                     # SAM2 mask for the Blend files tab
        self.columns = [E.new_column(**c) for c in s.get('columns', [])] or E.default_columns()
        self.devices = [('cpu', 'cpu')]
        self.models = E.Models(s.get('models_dir', E.DEFAULT_MODELS), s.get('device', 'cuda:0'))
        self.pth = []

        self.ui_scale = max(1.0, root.winfo_fpixels('1i') / 96)
        root.title('RAW Demosaic desktop')
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(s.get('geometry') or f'{int(sw * 0.85)}x{int(sh * 0.82)}+{int(sw * 0.07)}+{int(sh * 0.05)}')
        root.minsize(int(980 * self.ui_scale), int(600 * self.ui_scale))
        if s.get('zoomed'): root.state('zoomed')
        root.configure(bg=C['bg'])
        self._style()
        self._build()
        self._refresh_models()
        self._show_column(0)
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        root.after(50, self._poll)
        threading.Thread(target=self._probe, daemon=True).start()

    # ─────────────────────────────── settings
    def _collect_out(self):
        sr_models = [dict(model=m, weight=w) for m, w in self.sr_rows]
        return dict(space=key_of([(k, E.SPACES[k]['label']) for k in E.OUTPUT_SPACES], self.out_space.get(), 'srgb'),
                    tiff=self.save_tiff.get(), jpeg=self.save_jpeg.get(), jpeg_quality=int(self.jpeg_q.get()),
                    save_columns=self.save_cols.get(), out_dir=self.out_dir.get().strip(),
                    restore=list(self.restore_chain), sr=dict(enabled=self.sr_on.get(), target=key_of(SR_TARGETS, self.sr_target.get(), '2'), save_each=self.sr_each.get(),
                                                              downscale=key_of(DOWNSCALE, self.sr_down.get(), 'lanczos'), models=sr_models))

    def _save_settings(self):
        s = self.settings
        zoomed = self.root.state() == 'zoomed'
        s.update(zoomed=zoomed, columns=self.columns, blend_mode=key_of(list(E.BLEND_MODES.items()), self.blend_mode.get(), 'mean'),
                 out=self._collect_out(), models_dir=self.models_dir.get(), device=self.device_key())
        if not zoomed: s['geometry'] = self.root.geometry()
        try:
            with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
                json.dump(s, f, indent=1)
        except Exception:
            pass

    def _remember_dir(self, key, path):
        if path: self.settings[key] = os.path.dirname(path) if os.path.isfile(path) else path

    # ─────────────────────────────── look
    def _style(self):
        st = ttk.Style(self.root)
        st.theme_use('clam')
        f = ('Segoe UI', 10)
        self.root.option_add('*Font', f)
        st.configure('.', background=C['bg'], foreground=C['text'], fieldbackground=C['input'], bordercolor=C['border'],
                     lightcolor=C['border'], darkcolor=C['border'], troughcolor=C['track'], font=f,
                     selectbackground=C['accent'], selectforeground='#000', insertcolor=C['text'])
        st.configure('Card.TFrame', background=C['card'])
        st.configure('Card.TLabel', background=C['card'], foreground=C['text'])
        st.configure('Muted.TLabel', background=C['card'], foreground=C['muted'], font=('Segoe UI', 9))
        st.configure('Head.TLabel', background=C['card'], foreground=C['accent'], font=('Segoe UI', 11, 'bold'))
        st.configure('Sub.TLabel', background=C['card'], foreground=C['info'], font=('Segoe UI', 9, 'bold'))
        st.configure('Title.TLabel', background=C['bg'], foreground=C['text'], font=('Segoe UI', 16, 'bold'))
        st.configure('Tag.TLabel', background=C['bg'], foreground=C['muted'])
        st.configure('Status.TLabel', background=C['bg'], foreground=C['muted'], font=('Segoe UI', 9))
        st.configure('Card.TCheckbutton', background=C['card'], foreground=C['text'])
        st.map('Card.TCheckbutton', background=[('active', C['card'])], indicatorcolor=[('selected', C['accent']), ('!selected', C['input'])])
        st.configure('TButton', background=C['card2'], foreground=C['text'], padding=(10, 5), borderwidth=1)
        st.map('TButton', background=[('disabled', C['card']), ('active', C['border'])], foreground=[('disabled', C['muted'])])
        st.configure('Small.TButton', padding=(6, 2))
        st.configure('Accent.TButton', background=C['accent'], foreground='#111', font=('Segoe UI', 10, 'bold'), padding=(14, 7))
        st.map('Accent.TButton', background=[('disabled', C['card2']), ('active', C['accent_hover'])], foreground=[('disabled', C['muted'])])
        st.configure('TCombobox', fieldbackground=C['input'], background=C['card2'], foreground=C['text'], arrowcolor=C['text'])
        st.map('TCombobox', fieldbackground=[('readonly', C['input']), ('disabled', C['card'])], foreground=[('readonly', C['text']), ('disabled', C['muted'])],
               selectbackground=[('readonly', C['input'])], selectforeground=[('readonly', C['text'])])
        self.root.option_add('*TCombobox*Listbox.background', C['input'])
        self.root.option_add('*TCombobox*Listbox.foreground', C['text'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', C['accent'])
        st.configure('TEntry', fieldbackground=C['input'], foreground=C['text'])
        st.configure('TSpinbox', fieldbackground=C['input'], foreground=C['text'], arrowcolor=C['text'], background=C['card2'])
        st.map('TSpinbox', fieldbackground=[('disabled', C['card'])], foreground=[('disabled', C['muted'])])
        st.configure('Horizontal.TProgressbar', background=C['accent'], troughcolor=C['track'], thickness=10)
        st.configure('TNotebook', background=C['bg'], borderwidth=0)
        st.configure('TNotebook.Tab', background=C['card'], foreground=C['muted'], padding=(16, 6))
        st.map('TNotebook.Tab', background=[('selected', C['card2'])], foreground=[('selected', C['accent'])])
        st.configure('Vertical.TScrollbar', background=C['card2'], troughcolor=C['card'], arrowcolor=C['muted'])
        st.configure('Horizontal.TScale', background=C['card'], troughcolor=C['track'])

    def _card(self, parent, title=None, **pack):
        fr = ttk.Frame(parent, style='Card.TFrame', padding=10)
        fr.pack(**({'fill': 'x', 'pady': (0, 8)} | pack))
        if title:
            ttk.Label(fr, text=title, style='Head.TLabel').pack(anchor='w', pady=(0, 6))
        return fr

    def _listbox(self, parent, height):
        return tk.Listbox(parent, height=height, bg=C['input'], fg=C['text'], selectbackground=C['accent'],
                          selectforeground='#000', highlightthickness=1, highlightbackground=C['border'],
                          highlightcolor=C['accent'], relief='flat', activestyle='none', exportselection=False)

    def _scrolled_listbox(self, parent, height, **pack):
        lf = ttk.Frame(parent, style='Card.TFrame')
        lf.pack(**({'fill': 'x'} | pack))
        lb = self._listbox(lf, height)
        sb = ttk.Scrollbar(lf, orient='vertical', command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        lb.pack(side='left', fill='both', expand=True)
        sb.pack(side='left', fill='y')
        return lb

    def _row(self, parent, label, widget_factory, width=14):
        r = ttk.Frame(parent, style='Card.TFrame')
        r.pack(fill='x', pady=2)
        ttk.Label(r, text=label, style='Card.TLabel', width=width).pack(side='left')
        w = widget_factory(r)
        w.pack(side='left', fill='x', expand=True)
        return w

    def _combo(self, parent, var, pairs_or_values, width=None):
        vals = [v for _, v in pairs_or_values] if pairs_or_values and isinstance(pairs_or_values[0], tuple) else list(pairs_or_values)
        cb = ttk.Combobox(parent, textvariable=var, state='readonly', values=vals)
        if width: cb.configure(width=width)
        return cb

    def _spin(self, parent, var, lo, hi, inc, width=7, fmt=None):
        sp = ttk.Spinbox(parent, textvariable=var, from_=lo, to=hi, increment=inc, width=width)
        if fmt: sp.configure(format=fmt)
        return sp

    def _button_row(self, parent, buttons, pady=(6, 0)):
        br = ttk.Frame(parent, style='Card.TFrame')
        br.pack(fill='x', pady=pady)
        out = []
        for i, (text, cmd) in enumerate(buttons):
            b = ttk.Button(br, text=text, command=cmd, style='Small.TButton')
            b.pack(side='left', padx=(0 if i == 0 else 4, 0))
            out.append(b)
        return out

    def _scroll_column(self, tab, width=470):
        """Fixed-width left column that scrolls when it is taller than the window. Returns (inner, pinned)."""
        outer = ttk.Frame(tab, width=int(width * self.ui_scale))
        outer.pack(side='left', fill='y')
        outer.pack_propagate(False)
        pinned = ttk.Frame(outer)
        pinned.pack(side='bottom', fill='x')
        cv = tk.Canvas(outer, bg=C['bg'], highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient='vertical', command=cv.yview)
        inner = ttk.Frame(cv)
        win = cv.create_window(0, 0, window=inner, anchor='nw')
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side='left', fill='both', expand=True)

        def relayout(_=None):
            cv.configure(scrollregion=(0, 0, inner.winfo_reqwidth(), inner.winfo_reqheight()))
            need = inner.winfo_reqheight() > cv.winfo_height() + 1
            if need and not sb.winfo_ismapped():
                sb.pack(side='right', fill='y', before=cv)
            elif not need and sb.winfo_ismapped():
                sb.pack_forget(); cv.yview_moveto(0)
            cv.itemconfigure(win, width=cv.winfo_width())
        inner.bind('<Configure>', relayout)
        cv.bind('<Configure>', relayout)

        def wheel(e, d):
            if sb.winfo_ismapped(): cv.yview_scroll(-d, 'units')
        outer.bind('<Enter>', lambda e: bind_wheel(self.root, wheel, all_=True))
        outer.bind('<Leave>', lambda e: unbind_wheel(self.root))
        return inner, pinned

    # ─────────────────────────────── layout
    def _build(self):
        top = ttk.Frame(self.root, padding=(14, 10, 14, 4))
        top.pack(fill='x')
        ttk.Label(top, text='RAW DEMOSAIC', style='Title.TLabel').pack(side='left')
        ttk.Label(top, text='  desktop · the web app\'s pipeline with full-precision models, batches, BM3D and masked blending',
                  style='Tag.TLabel').pack(side='left', pady=(5, 0))

        bottom = ttk.Frame(self.root, padding=(14, 0, 14, 10))
        bottom.pack(side='bottom', fill='x')
        pr = ttk.Frame(bottom)
        pr.pack(fill='x', pady=(0, 4))
        self.progress = ttk.Progressbar(pr, maximum=1000, style='Horizontal.TProgressbar')
        self.progress.pack(side='left', fill='x', expand=True)
        self.cancel_btn = ttk.Button(pr, text='Cancel', command=self.cancel, state='disabled')
        self.cancel_btn.pack(side='left', padx=(8, 0))
        self.status = ttk.Label(bottom, text='Ready.', style='Status.TLabel')
        self.status.pack(fill='x')
        lf = ttk.Frame(bottom)
        lf.pack(fill='x', pady=(4, 0))
        self.log_text = tk.Text(lf, height=6, bg=C['input'], fg=C['text'], relief='flat', wrap='word',
                                highlightthickness=1, highlightbackground=C['border'], font=('Consolas', 9), state='disabled')
        sb = ttk.Scrollbar(lf, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        self.log_text.pack(side='left', fill='x', expand=True)
        sb.pack(side='left', fill='y')
        for tag in ('ok', 'err', 'info', 'warn'):
            self.log_text.tag_configure(tag, foreground=C[tag])
        self.log_text.tag_configure('time', foreground=C['muted'])

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill='both', expand=True, padx=14, pady=(4, 8))
        self._build_process_tab()
        self._build_blend_tab()
        self._build_output_tab()
        import rawdemosaic_develop_ui as DU
        self.develop = DU.DevelopTab(self, self.nb)
        self.dev_payload = None          # the last processed file's blend, for Develop
        self.nb.bind('<<NotebookTabChanged>>', lambda e: self.root.after(30, self.develop.request))

    # ─── tab 1: process
    def _build_process_tab(self):
        tab = ttk.Frame(self.nb, padding=(0, 8, 0, 0))
        self.nb.add(tab, text='  1 · Process  ')
        left, pinned = self._scroll_column(tab)
        right = ttk.Frame(tab)
        right.pack(side='left', fill='both', expand=True, padx=(10, 0))
        s = self.settings

        c = self._card(left, 'Files')
        self.file_list = self._scrolled_listbox(c, 6, fill='both', expand=True)
        self.file_list.configure(selectmode='extended')
        self.file_list.bind('<Delete>', lambda e: self.remove_files())
        self.file_list.bind('<<ListboxSelect>>', lambda e: self.root.after_idle(self._file_selected))
        self._button_row(c, [('Add files…', self.add_files), ('Add folder…', self.add_folder), ('Remove', self.remove_files), ('Clear', self.clear_files)])
        self.file_count = ttk.Label(c, text='No files. RAW files and JPEG/TIFF/PNG images can be mixed.', style='Muted.TLabel')
        self.file_count.pack(anchor='w', pady=(4, 0))
        self.file_info = ttk.Label(c, text='', style='Muted.TLabel', justify='left', wraplength=int(430 * self.ui_scale))
        self.file_info.pack(anchor='w', pady=(4, 0))

        c = self._card(left, 'Columns')
        ttk.Label(c, text='Each column is a separate processing of the file; the results are blended by weight.',
                  style='Muted.TLabel', wraplength=int(430 * self.ui_scale), justify='left').pack(anchor='w', pady=(0, 4))
        self.col_list = self._scrolled_listbox(c, 4)
        self.col_list.bind('<<ListboxSelect>>', lambda e: self._show_column(self._sel_col()))
        self._button_row(c, [('Add', self.add_column), ('Duplicate', self.dup_column), ('Remove', self.remove_column),
                             ('▲', lambda: self.move_column(-1)), ('▼', lambda: self.move_column(1))])
        self._button_row(c, [('Load settings…', self.load_settings_file), ('Save settings…', self.save_settings_file), ('Defaults', self.default_cols)], pady=(4, 6))
        self._build_column_editor(c)

        c = self._card(left, 'Blend')
        self.blend_mode = tk.StringVar(value=E.BLEND_MODES.get(s.get('blend_mode', 'mean'), 'Weighted mean'))
        self._row(c, 'Blend mode', lambda p: self._combo(p, self.blend_mode, list(E.BLEND_MODES.values())))
        ttk.Label(c, text='Blending is done in linear light. Median and detail-adaptive keep every column in memory.\n'
                          'Output colour space, JPEG, column TIFFs and super-resolution: see tab 3.',
                  style='Muted.TLabel', justify='left').pack(anchor='w', pady=(4, 0))

        br = ttk.Frame(pinned)
        br.pack(fill='x', pady=(4, 0))
        self.preview_btn = ttk.Button(br, text='Quick preview', command=lambda: self.process(quick=True))
        self.preview_btn.pack(side='left')
        self.process_btn = ttk.Button(br, text='Process selected', style='Accent.TButton', command=lambda: self.process(which='selected'))
        self.process_btn.pack(side='left', fill='x', expand=True, padx=4)
        self.batch_btn = ttk.Button(br, text='Process all', style='Accent.TButton', command=lambda: self.process(which='all'))
        self.batch_btn.pack(side='left')

        c = self._card(right, 'Results')
        self.result_list = self._scrolled_listbox(c, 4)
        # keep the zoom and position when switching between results of the same size (e.g. the columns), to compare them
        self.result_list.bind('<<ListboxSelect>>', lambda e: self.root.after_idle(lambda: self._show_selected_result(keep_view=True)))
        self._button_row(c, [('Develop', self.develop_selected_result), ('Open folder', self.open_result_folder),
                             ('Column TIFFs → Blend files', self.send_to_blend), ('Discard', self.discard_result)])
        pc = self._card(right, None, fill='both', expand=True)
        cv = tk.Canvas(pc, bg=C['checker'], highlightthickness=0)
        cv.pack(fill='both', expand=True)
        self.view = ImageView(cv)
        self.view.set_image(None, 'Add files, then Quick preview or Process. Wheel zooms, double-click toggles 100 %, drag pans.')

    def _build_column_editor(self, parent):
        ed = ttk.Frame(parent, style='Card.TFrame')
        ed.pack(fill='x')
        self._col_loading = False
        self.cv_name, self.cv_on, self.cv_weight = tk.StringVar(), tk.BooleanVar(), tk.StringVar()
        self.cv_rawdn, self.cv_rf1, self.cv_rf2 = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.cv_lumi, self.cv_chroma, self.cv_dem, self.cv_post = tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.cv_bm3d, self.cv_sigma = tk.StringVar(), tk.StringVar()
        self.cv_boost, self.cv_stage = tk.BooleanVar(), tk.StringVar()
        self.cv_gains = [tk.StringVar() for _ in range(10)]
        r = ttk.Frame(ed, style='Card.TFrame'); r.pack(fill='x', pady=2)
        ttk.Label(r, text='Name', style='Card.TLabel', width=14).pack(side='left')
        ttk.Entry(r, textvariable=self.cv_name, width=16).pack(side='left', fill='x', expand=True)
        ttk.Label(r, text='  Weight', style='Card.TLabel').pack(side='left')
        self._spin(r, self.cv_weight, 0, 100, 0.1, 6).pack(side='left', padx=(4, 0))
        ttk.Checkbutton(r, text='On', variable=self.cv_on, style='Card.TCheckbutton').pack(side='left', padx=(8, 0))
        self._row(ed, 'Demosaic', lambda p: self._combo(p, self.cv_dem, DEMOSAIC))
        self.rawdn_row = self._row(ed, 'RAW denoise', lambda p: self._combo(p, self.cv_rawdn, RAW_DENOISE)).master
        self.rf_frame = ttk.Frame(ed, style='Card.TFrame')
        self.rf_frame.pack(fill='x')
        self._row(self.rf_frame, '  RawForge model', lambda p: self._combo(p, self.cv_rf1, RF_MODELS))
        self._row(self.rf_frame, '  then (optional)', lambda p: self._combo(p, self.cv_rf2, ['(none)'] + RF_MODELS))
        r = ttk.Frame(self.rf_frame, style='Card.TFrame'); r.pack(fill='x', pady=2)
        ttk.Label(r, text='  Blend back', style='Card.TLabel', width=14).pack(side='left')
        ttk.Label(r, text='luminance', style='Muted.TLabel').pack(side='left')
        self._spin(r, self.cv_lumi, 0, 1, 0.05, 5).pack(side='left', padx=4)
        ttk.Label(r, text='chroma', style='Muted.TLabel').pack(side='left')
        self._spin(r, self.cv_chroma, 0, 1, 0.05, 5).pack(side='left', padx=4)
        self._row(ed, 'After demosaic', lambda p: self._combo(p, self.cv_post, POST_DENOISE))
        r = ttk.Frame(ed, style='Card.TFrame'); r.pack(fill='x', pady=2)
        ttk.Label(r, text='BM3D', style='Card.TLabel', width=14).pack(side='left')
        self._combo(r, self.cv_bm3d, BM3D).pack(side='left', fill='x', expand=True)
        ttk.Label(r, text=' σ', style='Card.TLabel').pack(side='left')
        self._spin(r, self.cv_sigma, 0.001, 0.5, 0.005, 6).pack(side='left', padx=(4, 0))
        r = ttk.Frame(ed, style='Card.TFrame'); r.pack(fill='x', pady=(6, 2))
        ttk.Checkbutton(r, text='Detail boost', variable=self.cv_boost, style='Card.TCheckbutton').pack(side='left')
        self._combo(r, self.cv_stage, BOOST_STAGE, 18).pack(side='right')
        g = ttk.Frame(ed, style='Card.TFrame'); g.pack(fill='x')
        for i, v in enumerate(self.cv_gains):
            f = ttk.Frame(g, style='Card.TFrame')
            f.grid(row=i // 5, column=i % 5, padx=2, pady=1, sticky='w')
            ttk.Label(f, text=f'L{i + 1}', style='Muted.TLabel', width=3).pack(side='left')
            self._spin(f, v, 0, 10, 0.1, 4).pack(side='left')
        ttk.Label(ed, text='Band gains, finest first (1 = unchanged). L1–L3 sharpen fine detail; higher levels are broader.',
                  style='Muted.TLabel', wraplength=int(430 * self.ui_scale), justify='left').pack(anchor='w', pady=(2, 0))
        allv = [self.cv_name, self.cv_on, self.cv_weight, self.cv_rawdn, self.cv_rf1, self.cv_rf2, self.cv_lumi, self.cv_chroma,
                self.cv_dem, self.cv_post, self.cv_bm3d, self.cv_sigma, self.cv_boost, self.cv_stage] + self.cv_gains
        for v in allv:
            v.trace_add('write', lambda *a: self._column_edited())

    # ─── tab 2: blend files
    def _build_blend_tab(self):
        tab = ttk.Frame(self.nb, padding=(0, 8, 0, 0))
        self.nb.add(tab, text='  2 · Blend files  ')
        left, pinned = self._scroll_column(tab)
        right = ttk.Frame(tab)
        right.pack(side='left', fill='both', expand=True, padx=(10, 0))
        c = self._card(left, 'Files to blend')
        ttk.Label(c, text='Column TIFFs saved by Process (…_col1.tif, …), or any finished images, re-blended in linear\n'
                          'light with new weights. Their ICC profiles are honoured; sizes are matched to the largest.',
                  style='Muted.TLabel', justify='left').pack(anchor='w', pady=(0, 4))
        self.bl_list = self._scrolled_listbox(c, 6)
        self.bl_list.bind('<<ListboxSelect>>', lambda e: self._bl_selected())
        self.bl_list.bind('<Delete>', lambda e: self.bl_remove())
        self._button_row(c, [('Add files…', self.bl_add), ('This session\'s column TIFFs', self.send_to_blend), ('Remove', self.bl_remove), ('Clear', self.bl_clear)])
        r = ttk.Frame(c, style='Card.TFrame'); r.pack(fill='x', pady=(6, 0))
        ttk.Label(r, text='Weight of selected', style='Card.TLabel', width=17).pack(side='left')
        self.bl_weight = tk.StringVar(value='1')
        self._spin(r, self.bl_weight, 0, 100, 0.1, 7).pack(side='left')
        self.bl_weight.trace_add('write', lambda *a: self._bl_weight_edited())
        self.bl_mode = tk.StringVar(value='Weighted mean')
        self.bl_mode_box = self._row(c, 'Blend mode', lambda p: self._combo(p, self.bl_mode, list(E.BLEND_MODES.values())), 17)

        c = self._card(left, 'SAM2 mask')
        ttk.Label(c, text='Restrict one file to (or exclude it from) an area you select with SAM2: inside the mask that file\n'
                          'is used alone, outside it the other files share the image by their weights.',
                  style='Muted.TLabel', justify='left').pack(anchor='w', pady=(0, 4))
        self.mask_file = tk.StringVar(value='(no mask)')
        self.mask_box = self._row(c, 'Masked file', lambda p: self._combo(p, self.mask_file, ['(no mask)']), 17)
        self.mask_file.trace_add('write', lambda *a: self._mask_file_changed())
        self._button_row(c, [('Edit mask…', self.edit_mask), ('Clear mask', self.clear_mask)])
        self.mask_info = ttk.Label(c, text='No mask.', style='Muted.TLabel')
        self.mask_info.pack(anchor='w', pady=(4, 0))

        c = self._card(left, 'Save as')
        self.bl_name = tk.StringVar(value=self.settings.get('blend_name', ''))
        self._row(c, 'File name', lambda p: ttk.Entry(p, textvariable=self.bl_name), 17)
        ttk.Label(c, text='Without an extension; empty = first file\'s name + _blend. Saved to the output folder (tab 3),\n'
                          'or next to the first file, with the same colour space, JPEG and super-resolution options.',
                  style='Muted.TLabel', justify='left').pack(anchor='w', pady=(2, 0))
        self.blend_btn = ttk.Button(pinned, text='Blend and save', style='Accent.TButton', command=self.run_blend)
        self.blend_btn.pack(fill='x', pady=(4, 0))
        pc = self._card(right, None, fill='both', expand=True)
        cv = tk.Canvas(pc, bg=C['checker'], highlightthickness=0)
        cv.pack(fill='both', expand=True)
        self.bl_view = ImageView(cv)
        self.bl_view.set_image(None, 'Add two or more files, set weights, then Blend and save.')

    # ─── tab 3: output & models
    def _build_output_tab(self):
        tab = ttk.Frame(self.nb, padding=(0, 8, 0, 0))
        self.nb.add(tab, text='  3 · Output & models  ')
        cols = ttk.Frame(tab)
        cols.pack(fill='both', expand=True)
        left = ttk.Frame(cols); left.pack(side='left', fill='both', expand=True)
        right = ttk.Frame(cols); right.pack(side='left', fill='both', expand=True, padx=(10, 0))
        o = self.settings.get('out', {})

        c = self._card(left, 'Output')
        self.out_space = tk.StringVar(value=E.SPACES[o.get('space', 'prophoto')]['label'])
        self._row(c, 'Colour space', lambda p: self._combo(p, self.out_space, [E.SPACES[k]['label'] for k in E.OUTPUT_SPACES]))
        ttk.Label(c, text='16-bit TIFFs carry the space\'s ICC profile. ProPhoto keeps every colour the camera recorded.',
                  style='Muted.TLabel').pack(anchor='w')
        self.save_tiff = tk.BooleanVar(value=o.get('tiff', True))
        ttk.Checkbutton(c, text='Save a 16-bit TIFF', variable=self.save_tiff, style='Card.TCheckbutton').pack(anchor='w', pady=(6, 0))
        r = ttk.Frame(c, style='Card.TFrame'); r.pack(fill='x')
        self.save_jpeg = tk.BooleanVar(value=o.get('jpeg', False))
        ttk.Checkbutton(r, text='Save a JPEG (sRGB) too, quality', variable=self.save_jpeg, style='Card.TCheckbutton').pack(side='left')
        self.jpeg_q = tk.IntVar(value=o.get('jpeg_quality', 95))
        ttk.Scale(r, from_=50, to=100, variable=self.jpeg_q, orient='horizontal', command=lambda v: self.jpeg_q.set(round(float(v)))).pack(side='left', fill='x', expand=True, padx=6)
        ttk.Label(r, textvariable=self.jpeg_q, style='Card.TLabel', width=4).pack(side='left')
        self.save_cols = tk.BooleanVar(value=o.get('save_columns', True))
        ttk.Checkbutton(c, text='Also save a 16-bit TIFF of each column (…_col1.tif …), for re-blending later',
                        variable=self.save_cols, style='Card.TCheckbutton').pack(anchor='w')
        r = ttk.Frame(c, style='Card.TFrame'); r.pack(fill='x', pady=(6, 0))
        ttk.Label(r, text='Output folder', style='Card.TLabel', width=14).pack(side='left')
        self.out_dir = tk.StringVar(value=o.get('out_dir', ''))
        ttk.Entry(r, textvariable=self.out_dir).pack(side='left', fill='x', expand=True)
        ttk.Button(r, text='…', width=3, command=self.choose_out_dir).pack(side='left', padx=(4, 0))
        ttk.Label(c, text='Empty: next to each input file.', style='Muted.TLabel').pack(anchor='w')

        c = self._card(left, 'Models')
        r = ttk.Frame(c, style='Card.TFrame'); r.pack(fill='x', pady=2)
        ttk.Label(r, text='Models folder', style='Card.TLabel', width=14).pack(side='left')
        self.models_dir = tk.StringVar(value=self.settings.get('models_dir', E.DEFAULT_MODELS))
        ttk.Entry(r, textvariable=self.models_dir).pack(side='left', fill='x', expand=True)
        ttk.Button(r, text='…', width=3, command=self.choose_models_dir).pack(side='left', padx=(4, 0))
        self.device = tk.StringVar(value='')
        self.device_box = self._row(c, 'Run models on', lambda p: self._combo(p, self.device, ['cpu']))
        self._button_row(c, [('Free GPU memory', self.free_gpu), ('Rescan models', self._refresh_models)])
        ttk.Label(c, text='Every model runs in full precision: float32 weights (RawForge\'s original TorchScript, not its\n'
                          'float16 ONNX), no mixed precision, TF32 off. Fill the folder with desktop\\setup_models.py.',
                  style='Muted.TLabel', justify='left').pack(anchor='w', pady=(4, 0))

        c = self._card(right, 'Restoration (before super-resolution)')
        ttk.Label(c, text='Models applied in turn to the finished image, e.g. Restormer deblurring or NAFNet denoising.',
                  style='Muted.TLabel').pack(anchor='w', pady=(0, 4))
        self.restore_chain = list(o.get('restore', []))
        self.rs_list = self._scrolled_listbox(c, 3)
        r = ttk.Frame(c, style='Card.TFrame'); r.pack(fill='x', pady=(6, 0))
        self.rs_pick = tk.StringVar()
        self.rs_box = self._combo(r, self.rs_pick, [''])
        self.rs_box.pack(side='left', fill='x', expand=True)
        for t, cmd in (('Add', self.rs_add), ('Remove', self.rs_remove), ('▲', lambda: self.rs_move(-1)), ('▼', lambda: self.rs_move(1))):
            ttk.Button(r, text=t, command=cmd, style='Small.TButton').pack(side='left', padx=(4, 0))

        c = self._card(right, 'Super-resolution')
        sr = o.get('sr', {})
        self.sr_on = tk.BooleanVar(value=sr.get('enabled', False))
        ttk.Checkbutton(c, text='Upscale the result (Process and Blend files; Develop has its own)', variable=self.sr_on, style='Card.TCheckbutton').pack(anchor='w')
        self.sr_each = tk.BooleanVar(value=bool(sr.get('save_each', False)))
        ttk.Checkbutton(c, text="Also save each model's own result (…_sr_<model>.tif), when blending several", variable=self.sr_each,
                        style='Card.TCheckbutton').pack(anchor='w')
        self.sr_target = tk.StringVar(value=label_of(SR_TARGETS, str(sr.get('target', '2'))))
        self._row(c, 'Output size', lambda p: self._combo(p, self.sr_target, SR_TARGETS))
        self.sr_down = tk.StringVar(value=label_of(DOWNSCALE, sr.get('downscale', 'lanczos')))
        self._row(c, 'Scale models by', lambda p: self._combo(p, self.sr_down, DOWNSCALE))
        ttk.Label(c, text='Several models can be blended: each is scaled to the output size first (e.g. Real-ESRGAN ×4 and\n'
                          'RealSR ×4 at 50 % each, then down to ×2). Weights are relative.',
                  style='Muted.TLabel', justify='left').pack(anchor='w', pady=(2, 4))
        self.sr_rows = [(m['model'], float(m['weight'])) for m in sr.get('models', [])]
        self.sr_list = self._scrolled_listbox(c, 4)
        self.sr_list.bind('<<ListboxSelect>>', lambda e: self._sr_selected())
        r = ttk.Frame(c, style='Card.TFrame'); r.pack(fill='x', pady=(6, 0))
        self.sr_pick, self.sr_w = tk.StringVar(), tk.StringVar(value='1')
        self.sr_box = self._combo(r, self.sr_pick, [''])
        self.sr_box.pack(side='left', fill='x', expand=True)
        ttk.Label(r, text=' weight', style='Card.TLabel').pack(side='left')
        self._spin(r, self.sr_w, 0, 100, 0.1, 5).pack(side='left', padx=(4, 0))
        self._button_row(c, [('Add', self.sr_add), ('Update selected', self.sr_update), ('Remove', self.sr_remove)])
        self._refresh_lists()

    # ─────────────────────────────── helpers
    def device_key(self):
        return next((k for k, v in self.devices if v == self.device.get()), self.settings.get('device', 'cuda:0'))

    def log(self, level, msg):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', time.strftime('%H:%M:%S '), 'time')
        self.log_text.insert('end', msg + '\n', level if level in ('ok', 'err', 'info', 'warn') else '')
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def _probe(self):
        devs = E.list_devices()
        self.q.put(('call', self._set_devices, devs))
        try:   # compile the demosaics once now (Numba caches them on disk), so the first run starts at once
            z = np.random.default_rng(0).random((96, 96), np.float32)
            for f in E.DM.DEMOSAICS.values(): f(z, [0, 1, 1, 2])
        except Exception:
            traceback.print_exc()

    def _set_devices(self, devs):
        self.devices = devs
        self.device_box.configure(values=[v for _, v in devs])
        want = self.settings.get('device', devs[0][0])
        self.device.set(next((v for k, v in devs if k == want), devs[0][1]))
        self.log('info', 'devices: ' + ', '.join(v for _, v in devs) + ('' if len(devs) > 1 else ' (no CUDA GPU found)'))

    def _refresh_models(self):
        root = self.models_dir.get() if hasattr(self, 'models_dir') else E.DEFAULT_MODELS
        self.pth = E.find_pth_models(root)
        labels = [l for l, _ in self.pth]
        self.rs_box.configure(values=[l for l in labels if l.startswith('restore/')] + [l for l in labels if l.startswith('sr/')])
        self.sr_box.configure(values=[l for l in labels if l.startswith('sr/')] + [l for l in labels if l.startswith('restore/')])
        if hasattr(self, 'develop'): self.develop.refresh_sr_models([l for l in labels if l.startswith('sr/')] + [l for l in labels if l.startswith('restore/')])
        if not self.sr_pick.get() and labels:
            self.sr_pick.set(next((l for l in labels if 'RealESRGAN_x4plus' in l), labels[0]))
        if not os.path.isdir(root):
            self.log('warn', f'models folder {root} not found — run desktop\\setup_models.py')
        elif not labels:
            self.log('warn', f'no restoration / super-resolution models in {root}')

    def pth_path(self, label):
        p = dict(self.pth).get(label)
        if not p: raise ValueError(f'Model {label} is not in the models folder')
        return p

    def _refresh_lists(self):
        self.rs_list.delete(0, 'end')
        for i, m in enumerate(self.restore_chain): self.rs_list.insert('end', f'{i + 1}. {m}')
        self.sr_list.delete(0, 'end')
        tot = sum(w for _, w in self.sr_rows) or 1
        for m, w in self.sr_rows: self.sr_list.insert('end', f'{100 * w / tot:5.1f} %   {m}')

    def opts(self):
        o = self._collect_out()
        o['restore'] = [self.pth_path(m) for m in o['restore']]
        o['sr'] = dict(o['sr'], models=[dict(path=self.pth_path(m['model']), weight=m['weight']) for m in o['sr']['models']])
        o['out_space'] = o['space']
        o['blend_mode'] = key_of(list(E.BLEND_MODES.items()), self.blend_mode.get(), 'mean')
        if o['out_dir'] and not os.path.isdir(o['out_dir']):
            os.makedirs(o['out_dir'], exist_ok=True)
        return o

    # ─────────────────────────────── worker plumbing
    def run_task(self, label, fn, done=None):
        """fn(job) runs on a worker thread; done(result) then runs on the Tk thread."""
        if self.busy:
            messagebox.showinfo(APP, 'Something is already running — wait for it or press Cancel.')
            return
        self.busy = True
        job = E.Job(log=lambda l, m: self.q.put(('log', l, m)), progress=lambda f, t=None: self.q.put(('progress', f, t)))
        self.job = job
        self._set_busy_ui(True)
        self.progress['value'] = 0
        self.status.configure(text=label + '…')
        t0 = time.time()
        self.models.set(self.models_dir.get(), self.device_key())

        def work():
            try:
                res = fn(job)
                self.q.put(('done', done, res, time.time() - t0))
            except E.Cancelled:
                self.q.put(('cancelled',))
            except MemoryError:
                self.q.put(('error', 'Out of memory — try fewer columns, the weighted-mean blend, or a smaller super-resolution.', traceback.format_exc()))
            except Exception as ex:
                msg = str(ex) or type(ex).__name__
                if 'out of memory' in msg.lower(): msg = 'The GPU ran out of memory — press Free GPU memory, or choose the other GPU or the CPU.\n\n' + msg
                self.q.put(('error', msg, traceback.format_exc()))
        threading.Thread(target=work, daemon=True).start()

    def _set_busy_ui(self, busy):
        st = 'disabled' if busy else 'normal'
        for b in (self.preview_btn, self.process_btn, self.batch_btn, self.blend_btn):
            b.configure(state=st)
        self.cancel_btn.configure(state='normal' if busy else 'disabled')

    def cancel(self):
        if self.job:
            self.job.cancel_event.set()
            self.status.configure(text='Cancelling (at the next tile or stage)…')

    def _poll(self):
        last_prog = None
        try:
            for _ in range(500):
                m = self.q.get_nowait()
                kind = m[0]
                if kind == 'log':
                    self.log(m[1], m[2])
                elif kind == 'progress':
                    last_prog = m
                elif kind == 'call':
                    m[1](*m[2:])
                else:
                    last_prog = None
                    self._finish(m)
        except queue.Empty:
            pass
        if last_prog:
            self.progress['value'] = max(0, min(1, last_prog[1])) * 1000
            if last_prog[2]: self.status.configure(text=last_prog[2])
        self.root.after(50, self._poll)

    def _finish(self, m):
        self.busy = False
        self.job = None
        kind = m[0]
        self._set_busy_ui(False)
        if kind == 'done':
            _, cb, res, dt = m
            self.progress['value'] = 1000
            self.status.configure(text=f'Done in {dt:.1f} s.')
            if cb:
                try: cb(res)
                except Exception as ex:
                    self.log('err', f'{ex}'); traceback.print_exc()
        elif kind == 'cancelled':
            self.progress['value'] = 0
            self.status.configure(text='Cancelled.')
            self.log('warn', 'cancelled')
        else:
            self.progress['value'] = 0
            self.status.configure(text='Failed: ' + m[1].splitlines()[0])
            self.log('err', m[1])
            print(m[2])
            messagebox.showerror(APP, m[1])

    # ─────────────────────────────── files
    def _set_files(self, paths):
        seen, out = set(), []
        for p in paths:
            k = os.path.normcase(os.path.abspath(p))
            if k not in seen:
                seen.add(k); out.append(os.path.abspath(p))
        self.files = out
        self.file_list.delete(0, 'end')
        for p in out: self.file_list.insert('end', os.path.basename(p))
        n_raw = sum(1 for p in out if os.path.splitext(p)[1].lower() in E.RAW_EXTS)
        self.file_count.configure(text=(f'{len(out)} file{"s" if len(out) != 1 else ""} ({n_raw} RAW).' if out else 'No files.'))
        if out and not self.file_list.curselection():
            self.file_list.selection_set(0); self._file_selected()

    def add_files(self):
        ps = filedialog.askopenfilenames(filetypes=IN_TYPES, initialdir=self.settings.get('in_dir'), title='RAW files or images to process')
        if ps:
            self._remember_dir('in_dir', ps[0])
            self._set_files(self.files + [os.path.normpath(p) for p in ps])

    def add_folder(self):
        d = filedialog.askdirectory(initialdir=self.settings.get('in_dir'), title='Folder of RAW files')
        if d:
            self._remember_dir('in_dir', d)
            ps = sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(E.RAW_EXTS + E.IMAGE_EXTS))
            self._set_files(self.files + [os.path.normpath(p) for p in ps])

    def remove_files(self):
        sel = set(self.file_list.curselection())
        self._set_files([p for i, p in enumerate(self.files) if i not in sel])

    def clear_files(self):
        self._set_files([])
        self.file_info.configure(text='')

    def _file_selected(self):
        sel = self.file_list.curselection()
        if not sel: return
        path = self.files[sel[0]]
        self.file_info.configure(text='Reading…')

        def work():
            try:
                src = self._get_src(path)
                rows = src.summary()
                txt = '\n'.join(f'{k}: {v}' for k, v in rows if v) + ''.join('\n⚠ ' + w for w in src.warnings)
                thumb = src.rgb if src.is_rgb else src.thumbnail()
                if src.is_rgb: thumb = E.display_image(thumb, 2400)
                self.q.put(('call', self._show_file_info, path, txt, thumb))
            except Exception as ex:
                self.q.put(('call', self._show_file_info, path, f'Could not read this file: {ex}', None))
        threading.Thread(target=work, daemon=True).start()

    def _show_file_info(self, path, txt, thumb):
        sel = self.file_list.curselection()
        if not sel or self.files[sel[0]] != path: return
        self.file_info.configure(text=txt)
        if thumb is not None and not self.busy:
            self.view.set_image(thumb, f'{os.path.basename(path)} — embedded preview' if not path.lower().endswith(E.IMAGE_EXTS) else os.path.basename(path))

    _src_lock = threading.Lock()

    def _get_src(self, path):
        with self._src_lock:
            src = self.src_cache.get(path)
            if src is None:
                src = E.open_input(path)
                self.src_cache[path] = src
                while len(self.src_cache) > 2:
                    self.src_cache.pop(next(iter(self.src_cache)))
            return src

    # ─────────────────────────────── columns
    def _sel_col(self):
        sel = self.col_list.curselection()
        return sel[0] if sel else (0 if self.columns else -1)

    def _refresh_columns(self, select=None):
        self.col_list.delete(0, 'end')
        tot = sum(c['weight'] for c in self.columns if c['enabled']) or 1
        for i, c in enumerate(self.columns):
            pct = f'{100 * c["weight"] / tot:4.0f} %' if c['enabled'] else '  off'
            self.col_list.insert('end', f'{pct}   {i + 1}. {c["name"]} — {E.describe_column(c, False, False)}')
            if not c['enabled']: self.col_list.itemconfigure(i, foreground=C['muted'])
        if select is not None and 0 <= select < len(self.columns):
            self.col_list.selection_clear(0, 'end'); self.col_list.selection_set(select); self.col_list.see(select)

    def _show_column(self, i):
        self._refresh_columns(select=i)
        if i < 0 or i >= len(self.columns): return
        c = self.columns[i]
        self._col_loading = True
        try:
            self.cv_name.set(c['name']); self.cv_on.set(c['enabled']); self.cv_weight.set(f'{c["weight"]:g}')
            self.cv_rawdn.set(label_of(RAW_DENOISE, c['rawDenoise'])); self.cv_rf1.set(c['rf1']); self.cv_rf2.set(c['rf2'] or '(none)')
            self.cv_lumi.set(f'{c["lumi"]:g}'); self.cv_chroma.set(f'{c["chroma"]:g}')
            self.cv_dem.set(label_of(DEMOSAIC, c['demosaic'])); self.cv_post.set(label_of(POST_DENOISE, c['postDenoise']))
            self.cv_bm3d.set(label_of(BM3D, c['bm3d']['mode'])); self.cv_sigma.set(f'{c["bm3d"]["sigma"]:g}')
            self.cv_boost.set(c['boost']['enabled']); self.cv_stage.set(label_of(BOOST_STAGE, c['boost']['stage']))
            for v, g in zip(self.cv_gains, c['boost']['gains']): v.set(f'{g:g}')
        finally:
            self._col_loading = False
        self._update_rf_visibility()

    def _update_rf_visibility(self):
        if key_of(RAW_DENOISE, self.cv_rawdn.get(), 'none') == 'rawforge':
            if not self.rf_frame.winfo_manager(): self.rf_frame.pack(fill='x', after=self.rawdn_row)
        elif self.rf_frame.winfo_manager():
            self.rf_frame.pack_forget()

    def _column_edited(self):
        if self._col_loading: return
        i = self._sel_col()
        if i < 0 or i >= len(self.columns): return
        c = self.columns[i]

        def num(v, d):
            try: return float(v.get())
            except (ValueError, tk.TclError): return d
        c['name'] = self.cv_name.get() or 'Column'
        c['enabled'] = bool(self.cv_on.get())
        c['weight'] = max(0.0, num(self.cv_weight, c['weight']))
        c['rawDenoise'] = key_of(RAW_DENOISE, self.cv_rawdn.get(), 'none')
        c['rf1'] = self.cv_rf1.get() or c['rf1']
        c['rf2'] = '' if self.cv_rf2.get() in ('', '(none)') else self.cv_rf2.get()
        c['lumi'] = min(1, max(0, num(self.cv_lumi, c['lumi']))); c['chroma'] = min(1, max(0, num(self.cv_chroma, c['chroma'])))
        c['demosaic'] = key_of(DEMOSAIC, self.cv_dem.get(), 'amaze')
        c['postDenoise'] = key_of(POST_DENOISE, self.cv_post.get(), 'none')
        c['bm3d'] = dict(mode=key_of(BM3D, self.cv_bm3d.get(), 'none'), sigma=max(1e-4, num(self.cv_sigma, c['bm3d']['sigma'])))
        c['boost'] = dict(enabled=bool(self.cv_boost.get()), stage=key_of(BOOST_STAGE, self.cv_stage.get(), 'after'),
                          gains=[max(0.0, num(v, g)) for v, g in zip(self.cv_gains, c['boost']['gains'])])
        self._update_rf_visibility()
        self._refresh_columns(select=i)

    def add_column(self):
        self.columns.append(E.new_column(name=f'Column {len(self.columns) + 1}'))
        self._show_column(len(self.columns) - 1)

    def dup_column(self):
        i = self._sel_col()
        if i < 0: return
        c = copy.deepcopy(self.columns[i]); c['name'] += ' (copy)'
        self.columns.insert(i + 1, c)
        self._show_column(i + 1)

    def remove_column(self):
        i = self._sel_col()
        if i < 0 or len(self.columns) <= 1: return
        del self.columns[i]
        self._show_column(min(i, len(self.columns) - 1))

    def move_column(self, d):
        i = self._sel_col(); j = i + d
        if i < 0 or not 0 <= j < len(self.columns): return
        self.columns[i], self.columns[j] = self.columns[j], self.columns[i]
        self._show_column(j)

    def default_cols(self):
        if messagebox.askyesno(APP, 'Replace the columns with the defaults (AMaZE, RAW NIND-Unet, RawForge)?'):
            self.columns = E.default_columns(); self._show_column(0)

    def save_settings_file(self):
        p = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('Settings', '*.json')], initialdir=self.settings.get('json_dir'), title='Save settings')
        if not p: return
        self._remember_dir('json_dir', p)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(dict(app='RAW Demosaic desktop', columns=self.columns, blend_mode=key_of(list(E.BLEND_MODES.items()), self.blend_mode.get(), 'mean'),
                           out=self._collect_out()), f, indent=1)
        self.log('ok', f'settings saved to {os.path.basename(p)}')

    def load_settings_file(self):
        p = filedialog.askopenfilename(filetypes=[('Settings', '*.json')], initialdir=self.settings.get('json_dir'), title='Load settings')
        if not p: return
        self._remember_dir('json_dir', p)
        self.apply_settings_file(p)

    def apply_settings_file(self, p):
        """Columns, blend mode and output options from a file written by Save settings…; False if it holds none."""
        d = E.load_json(p)
        if not d or not isinstance(d.get('columns'), list):
            messagebox.showerror(APP, f'{os.path.basename(p)} holds no column settings.'); return False
        self.columns = [E.new_column(**c) for c in d['columns']] or E.default_columns()
        if d.get('blend_mode') in E.BLEND_MODES: self.blend_mode.set(E.BLEND_MODES[d['blend_mode']])
        o = d.get('out')
        if o:
            if o.get('space') in E.OUTPUT_SPACES: self.out_space.set(E.SPACES[o['space']]['label'])
            for var, k in ((self.save_tiff, 'tiff'), (self.save_jpeg, 'jpeg'), (self.save_cols, 'save_columns')):
                if k in o: var.set(bool(o[k]))
            if 'jpeg_quality' in o: self.jpeg_q.set(int(o['jpeg_quality']))
            if 'restore' in o: self.restore_chain = list(o['restore'])
            sr = o.get('sr')
            if sr:
                self.sr_on.set(bool(sr.get('enabled'))); self.sr_target.set(label_of(SR_TARGETS, str(sr.get('target', '2'))))
                self.sr_down.set(label_of(DOWNSCALE, sr.get('downscale', 'lanczos')))
                self.sr_rows = [(m['model'], float(m['weight'])) for m in sr.get('models', [])]
            self._refresh_lists()
        self._show_column(0)
        self.log('ok', f'settings loaded from {os.path.basename(p)}')
        return True

    def open_from_command_line(self, files, settings=None, run=False):
        """Files handed over on the command line (e.g. by Photo Library): load them, apply a settings file,
        and with run start Process all — a batch."""
        if settings and not self.apply_settings_file(settings): run = False
        if files:
            self._set_files(self.files + [os.path.normpath(p) for p in files])
            self.log('info', f'{len(files)} file{"s" if len(files) != 1 else ""} from the command line')
        if run and self.files:
            self.process(which='all')

    # ─────────────────────────────── processing
    def process(self, quick=False, which='selected'):
        if not self.files:
            messagebox.showinfo(APP, 'Add a RAW file or image first.'); return
        sel = list(self.file_list.curselection()) or [0]
        paths = [self.files[i] for i in sel] if which == 'selected' or quick else list(self.files)
        if quick: paths = paths[:1]
        try:
            opts = self.opts()
        except Exception as ex:
            messagebox.showerror(APP, str(ex)); return
        columns = copy.deepcopy(self.columns)
        if not any(c['enabled'] and c['weight'] > 0 for c in columns):
            messagebox.showinfo(APP, 'Turn on at least one column with a weight above 0.'); return
        if not quick and not (opts['tiff'] or opts['jpeg']):
            messagebox.showinfo(APP, 'Choose TIFF and/or JPEG output on tab 3.'); return
        self._save_settings()

        def work(job):
            done = []
            for k, p in enumerate(paths):
                job.check()
                name = os.path.basename(p)
                job.log('info', f'{"quick preview of" if quick else "processing"} {name} ({k + 1}/{len(paths)})')
                src = self._get_src(p)
                for w in src.warnings: job.log('warn', f'{name}: {w}')
                sub = E.Job(log=job.log, progress=lambda f, t=None, k=k: job.progress((k + max(0, min(1, f))) / len(paths), f'{name}: {t}' if t else None))
                sub.cancel_event = job.cancel_event
                res = E.process(src, columns, opts, self.models, sub, quick=quick)
                big = E.display_image(res['blend'], 0 if (k == len(paths) - 1) else 2400)
                last = k == len(paths) - 1
                size = (res['blend'].shape[1], res['blend'].shape[0])
                cols = [(c['name'], img if last else img[::max(1, -(-max(img.shape[:2]) // 2400))][:, ::max(1, -(-max(img.shape[:2]) // 2400))])
                        for c, img in res['columns']]
                if k == len(paths) - 1:
                    self.q.put(('call', self._to_develop, p, res, src, quick))
                job.log('ok', f'{name}: {"preview" if quick else "done"} in {res["seconds"]:.1f} s' + (f' → {", ".join(os.path.basename(f) for f in res["files"])}' if res['files'] else ''))
                self.q.put(('call', self._add_file_results, name, big, cols, res['files'], quick, p, size))
                res = None
                done.append(p)
            return done
        self.run_task('Quick preview' if quick else f'Processing {len(paths)} file{"s" if len(paths) > 1 else ""}', work)

    def _add_file_results(self, name, big, cols, files, quick, src=None, size=None):
        tag = ' (quick preview)' if quick else ''
        start = len(self.results)
        self.results.append(Result(f'{name}{tag} — blend', big, [f for f in files if '_col' not in os.path.basename(f)], src=src, quick=quick, size=size))
        for i, (cname, img) in enumerate(cols):
            self.results.append(Result(f'{name}{tag} — column {i + 1}: {cname}', img, [f for f in files if f.endswith(f'_col{i + 1}.tif')], src=src, kind='column', quick=quick, size=size))
        self.session_tiffs += [f for f in files if '_col' in os.path.basename(f) and f.endswith('.tif')]
        for r in self.results[start:]: self.result_list.insert('end', r.label())
        # keep RAM in check: older blends' full-size display copies are reduced
        for r in self.results[:start]:
            if r.img.shape[0] * r.img.shape[1] > 6e6:
                f = int(np.ceil(max(r.img.shape[:2]) / 2400)); r.img = np.ascontiguousarray(r.img[::f, ::f])
        self.result_list.selection_clear(0, 'end'); self.result_list.selection_set(start); self.result_list.see('end')
        self._show_selected_result(keep_view=True)

    # ─── to Develop
    @staticmethod
    def dev_info(path, src, res=None, quick=False):
        info = dict(name=os.path.basename(path), path=path, stem=os.path.splitext(path)[0], quick=quick, is_rgb=src.is_rgb)
        if src.is_rgb:
            info.update(make=src.info.get('make', ''), model=src.info.get('model', ''), alpha=src.alpha)
        else:
            info.update(make=src.make, model=src.model, lens=src.lens, lens_make=src.lens_make, focal=src.focal, aperture=src.aperture,
                        f35=src.f35, iso=src.iso)
            info['frame_diag'] = math.hypot(src.W, src.H)
            if res is not None:
                import rawdemosaic_develop_ui as DU
                info['frame'] = DU.frame_of(res['region'], res['src_hw'], res['flip'], False)
        return info

    def _to_develop(self, path, res, src, quick):
        info = self.dev_info(path, src, res, quick)
        if quick and info.get('alpha') is not None and res.get('region'):
            y0, x0, h, w = res['region']; info['alpha'] = info['alpha'][y0:y0 + h, x0:x0 + w]
        self.dev_payload = (path, quick, res['blend'], info)
        self.develop.load(res['blend'], info, keep=True)

    def develop_selected_result(self):
        r = self.selected_result()
        if r is None:
            messagebox.showinfo(APP, 'Select a result on tab 1 first.'); return
        if self.dev_payload and r.src == self.dev_payload[0] and r.quick == self.dev_payload[1] and r.kind == 'blend':
            p, q, blend, info = self.dev_payload
            self.develop.load(blend, info, keep=True); self.nb.select(3); return
        tif = next((f for f in r.files if f.lower().endswith('.tif')), None)
        if not tif:
            messagebox.showinfo(APP, 'This result is not in memory and has no saved TIFF: process it again.'); return

        def work(job):
            rgb, info, alpha = E.load_image_linear(tif)
            dinfo = dict(name=os.path.basename(tif), path=r.src or tif, stem=os.path.splitext(tif)[0], is_rgb=True, alpha=alpha)
            if r.src:
                try:
                    src = self._get_src(r.src)
                    dinfo = dict(self.dev_info(r.src, src), name=os.path.basename(tif), stem=os.path.splitext(tif)[0], alpha=alpha)
                except Exception:
                    pass
            return rgb, dinfo
        self.run_task('Opening in Develop', work, lambda res: (self.develop.load(res[0], res[1], keep=True), self.nb.select(3)))

    def selected_result(self):
        sel = self.result_list.curselection()
        return self.results[sel[0]] if sel and sel[0] < len(self.results) else None

    def _show_selected_result(self, keep_view=False):
        r = self.selected_result()
        if not r: return
        # same size, give or take the pixel a reduced copy can differ by (blend box-averaged, columns strided)
        same = self.view.img is not None and all(abs(a - b) <= 2 for a, b in zip(self.view.img.shape[:2], r.img.shape[:2]))
        self.view.set_image(r.img, r.title, keep_view=keep_view and same)

    def discard_result(self):
        sel = self.result_list.curselection()
        if not sel: return
        i = sel[0]
        del self.results[i]
        self.result_list.delete(i)
        if self.results:
            self.result_list.selection_set(min(i, len(self.results) - 1)); self._show_selected_result()
        else:
            self.view.set_image(None, 'Nothing to show yet')

    def open_result_folder(self):
        r = self.selected_result()
        d = os.path.dirname(r.files[0]) if r and r.files else (self.out_dir.get().strip() or (os.path.dirname(self.files[0]) if self.files else ''))
        if d and os.path.isdir(d): open_folder(d)

    # ─────────────────────────────── blend files
    def _bl_refresh(self, select=None):
        self.bl_list.delete(0, 'end')
        tot = sum(w for _, w in self.blend_files) or 1
        for p, w in self.blend_files: self.bl_list.insert('end', f'{100 * w / tot:5.1f} %   {os.path.basename(p)}')
        names = ['(no mask)'] + [f'{i + 1}. {os.path.basename(p)}' for i, (p, _) in enumerate(self.blend_files)]
        self.mask_box.configure(values=names)
        if self.mask_file.get() not in names: self.mask_file.set('(no mask)')
        if select is not None and 0 <= select < len(self.blend_files):
            self.bl_list.selection_set(select)

    def bl_add(self, paths=None):
        ps = paths or filedialog.askopenfilenames(filetypes=IMG_TYPES, initialdir=self.settings.get('blend_dir'), title='Images to blend')
        if not ps: return
        self._remember_dir('blend_dir', ps[0])
        have = {os.path.normcase(p) for p, _ in self.blend_files}
        for p in ps:
            if os.path.normcase(os.path.abspath(p)) not in have: self.blend_files.append((os.path.abspath(p), 1.0))
        self._bl_refresh(select=len(self.blend_files) - 1)

    def send_to_blend(self):
        # the selected result's file, else the file processed last (never a mix of different photos)
        stem_of = lambda f: os.path.join(os.path.dirname(f), os.path.basename(f).rsplit('_col', 1)[0])
        r = self.selected_result()
        stem = stem_of(r.files[0].rsplit('.', 1)[0]) if r and r.files else (stem_of(self.session_tiffs[-1]) if self.session_tiffs else None)
        tiffs = [f for f in self.session_tiffs if stem_of(f) == stem]
        if not tiffs:
            messagebox.showinfo(APP, 'No column TIFFs were saved this session. Tick "Also save a 16-bit TIFF of each column" on tab 3, '
                                     'process a file, or add earlier …_col1.tif files with Add files.'); return
        self.blend_files = []
        self.bl_add(tiffs)
        self.nb.select(1)

    def bl_remove(self):
        sel = self.bl_list.curselection()
        if not sel: return
        del self.blend_files[sel[0]]
        if self.mask_cfg and self.mask_cfg['path'] not in [p for p, _ in self.blend_files]: self.clear_mask()
        self._bl_refresh(select=min(sel[0], len(self.blend_files) - 1))

    def bl_clear(self):
        self.blend_files = []; self.clear_mask(); self._bl_refresh()

    def _bl_selected(self):
        sel = self.bl_list.curselection()
        if not sel: return
        p, w = self.blend_files[sel[0]]
        self._bl_loading = True
        self.bl_weight.set(f'{w:g}')
        self._bl_loading = False

        def work():
            try:
                rgb, info = E.load_for_blend(p)
                self.q.put(('call', lambda: (not self.busy) and self.bl_view.set_image(E.display_image(rgb, 2400), f'{os.path.basename(p)} — {info["profile"]}')))
            except Exception as ex:
                self.q.put(('log', 'err', f'{os.path.basename(p)}: {ex}'))
        threading.Thread(target=work, daemon=True).start()

    def _bl_weight_edited(self):
        if getattr(self, '_bl_loading', False): return
        sel = self.bl_list.curselection()
        if not sel: return
        try: w = max(0.0, float(self.bl_weight.get()))
        except ValueError: return
        p, _ = self.blend_files[sel[0]]
        self.blend_files[sel[0]] = (p, w)
        self._bl_refresh(select=sel[0])

    def _mask_index(self):
        v = self.mask_file.get()
        if v == '(no mask)': return None
        try: return int(v.split('.', 1)[0]) - 1
        except ValueError: return None

    def _mask_file_changed(self):
        i = self._mask_index()
        if self.mask_cfg and (i is None or self.blend_files[i][0] != self.mask_cfg['path']):
            self.clear_mask(keep_choice=True)
        self._update_mask_ui()

    def _update_mask_ui(self):
        if self.mask_cfg:
            m = self.mask_cfg
            n = sum(1 for s in m['selections'] if s['points'] or s['box'])
            self.mask_info.configure(text=f'{n} selection{"s" if n != 1 else ""} on {os.path.basename(m["path"])}; grow {m["grow"]:+d} px, '
                                          f'feather {m["feather"]} px{", excluded" if m["exclude"] else ""}. Weighted mean is used.')
            self.bl_mode.set('Weighted mean'); self.bl_mode_box.configure(state='disabled')
        else:
            self.mask_info.configure(text='No mask.')
            self.bl_mode_box.configure(state='readonly')

    def clear_mask(self, keep_choice=False):
        self.mask_cfg = None
        if not keep_choice: self.mask_file.set('(no mask)')
        self._update_mask_ui()

    def edit_mask(self):
        i = self._mask_index()
        if i is None:
            messagebox.showinfo(APP, 'Choose the file to mask first ("Masked file").'); return
        if self.busy:
            messagebox.showinfo(APP, 'Wait for the current job to finish.'); return
        self.models.set(self.models_dir.get(), self.device_key())
        MaskEditor(self, self.blend_files[i][0], self.mask_cfg if self.mask_cfg and self.mask_cfg['path'] == self.blend_files[i][0] else None)

    def set_mask(self, cfg):
        self.mask_cfg = cfg
        self._update_mask_ui()

    def run_blend(self):
        if len(self.blend_files) < 2:
            messagebox.showinfo(APP, 'Add at least two files to blend.'); return
        if sum(w for _, w in self.blend_files) <= 0:
            messagebox.showinfo(APP, 'Give at least one file a weight above 0.'); return
        try:
            opts = self.opts()
        except Exception as ex:
            messagebox.showerror(APP, str(ex)); return
        if not (opts['tiff'] or opts['jpeg']):
            messagebox.showinfo(APP, 'Choose TIFF and/or JPEG output on tab 3.'); return
        paths, weights = [p for p, _ in self.blend_files], [w for _, w in self.blend_files]
        mode = key_of(list(E.BLEND_MODES.items()), self.bl_mode.get(), 'mean')
        name = self.bl_name.get().strip()
        first = paths[0]
        stem0 = os.path.splitext(os.path.basename(first))[0]
        stem0 = stem0.split('_col')[0] if '_col' in stem0 else stem0
        out_stem = os.path.join(opts['out_dir'] or os.path.dirname(first), name or stem0 + '_blend')
        if any(os.path.exists(out_stem + e) for e in ('.tif', '.jpg')) and not messagebox.askyesno(APP, f'{os.path.basename(out_stem)}.tif/.jpg exists. Replace it?'):
            return
        mcfg = copy.deepcopy(self.mask_cfg) if self.mask_cfg else None
        self.settings['blend_name'] = name
        self._save_settings()

        def work(job):
            mask, mi = None, None
            if mcfg:
                mi = paths.index(mcfg['path'])
                job.progress(0, 'Building the mask')
                fw, fh = mcfg['full_size']
                m = Image.fromarray((mcfg['union'] * 255).astype(np.uint8)).resize((fw, fh), Image.BILINEAR)
                mask = E.expand_feather(np.asarray(m) >= 128, mcfg['grow'], mcfg['feather'])
            out, files = E.blend_files(paths, weights, mode, opts, self.models, job, mask=mask, mask_index=mi,
                                       exclude=bool(mcfg and mcfg['exclude']), out_stem=out_stem)
            return E.display_image(out, 0), files

        def done(res):
            img, files = res
            self.bl_view.set_image(img, os.path.basename(out_stem) + ' — blend')
            self.results.append(Result(os.path.basename(out_stem) + ' — blend', img[::max(1, int(np.ceil(max(img.shape[:2]) / 2400))), ::max(1, int(np.ceil(max(img.shape[:2]) / 2400)))], files))
            self.result_list.insert('end', self.results[-1].label())
        self.run_task('Blending', work, done)

    # ─────────────────────────────── output tab
    def choose_out_dir(self):
        d = filedialog.askdirectory(initialdir=self.out_dir.get() or self.settings.get('in_dir'), title='Output folder')
        if d: self.out_dir.set(os.path.normpath(d))

    def choose_models_dir(self):
        d = filedialog.askdirectory(initialdir=self.models_dir.get() or E.DEFAULT_MODELS, title='models-desktop folder')
        if d:
            self.models_dir.set(os.path.normpath(d)); self._refresh_models()

    def free_gpu(self):
        if self.busy: return
        self.models.release()
        self.log('info', 'models unloaded, GPU memory freed')

    def rs_add(self):
        m = self.rs_pick.get()
        if m: self.restore_chain.append(m); self._refresh_lists()

    def rs_remove(self):
        sel = self.rs_list.curselection()
        if sel: del self.restore_chain[sel[0]]; self._refresh_lists()

    def rs_move(self, d):
        sel = self.rs_list.curselection()
        if not sel: return
        i, j = sel[0], sel[0] + d
        if 0 <= j < len(self.restore_chain):
            self.restore_chain[i], self.restore_chain[j] = self.restore_chain[j], self.restore_chain[i]
            self._refresh_lists(); self.rs_list.selection_set(j)

    def _sr_weight(self):
        try: return max(0.0, float(self.sr_w.get()))
        except ValueError: return 1.0

    def sr_add(self):
        m = self.sr_pick.get()
        if not m: return
        self.sr_rows.append((m, self._sr_weight())); self.sr_on.set(True); self._refresh_lists()

    def sr_update(self):
        sel = self.sr_list.curselection()
        if sel and self.sr_pick.get():
            self.sr_rows[sel[0]] = (self.sr_pick.get(), self._sr_weight()); self._refresh_lists(); self.sr_list.selection_set(sel[0])

    def sr_remove(self):
        sel = self.sr_list.curselection()
        if sel: del self.sr_rows[sel[0]]; self._refresh_lists()

    def _sr_selected(self):
        sel = self.sr_list.curselection()
        if sel:
            m, w = self.sr_rows[sel[0]]; self.sr_pick.set(m); self.sr_w.set(f'{w:g}')

    # ─────────────────────────────── shutdown
    def on_close(self):
        if self.busy and not messagebox.askyesno(APP, 'Something is still running. Quit anyway?'):
            return
        if self.job: self.job.cancel_event.set()
        self._save_settings()
        self.root.destroy()


# ───────────────────────────────────────────── SAM2 mask editor
class MaskEditor:
    """Click to add, right-click to exclude, Shift-drag a box. Several selections are combined; grow/shrink
    and feather apply to the combined mask (in full-resolution pixels)."""
    MAX_SIDE = 2400

    def __init__(self, app, path, cfg):
        self.app, self.path = app, path
        self.win = tk.Toplevel(app.root)
        self.win.title(f'SAM2 mask — {os.path.basename(path)}')
        self.win.configure(bg=C['bg'])
        sw, sh = app.root.winfo_screenwidth(), app.root.winfo_screenheight()
        self.win.geometry(f'{int(sw * 0.75)}x{int(sh * 0.78)}+{int(sw * 0.12)}+{int(sh * 0.08)}')
        self.win.transient(app.root)
        self.sel = [dict(points=[], box=None)] if not cfg else copy.deepcopy(cfg['selections'])
        self.masks = [None] * len(self.sel)
        self.cur = 0
        self.box_drag = None
        self.predictor = None
        self.display = None
        self.factor = 1.0
        side = ttk.Frame(self.win, style='Card.TFrame', padding=10, width=int(320 * app.ui_scale))
        side.pack(side='right', fill='y', padx=(0, 10), pady=10)
        side.pack_propagate(False)
        ttk.Label(side, text='Selections', style='Head.TLabel').pack(anchor='w')
        ttk.Label(side, text='Click: add · right-click: exclude\nShift-drag: box · wheel: zoom\nmiddle-drag: pan · double-click: 100 %',
                  style='Muted.TLabel', justify='left').pack(anchor='w', pady=(0, 6))
        self.lb = app._listbox(side, 6)
        self.lb.pack(fill='x')
        self.lb.bind('<<ListboxSelect>>', lambda e: self._pick())
        app._button_row(side, [('New selection', self.add_sel), ('Delete', self.del_sel), ('Clear points', self.clear_pts)])
        self.grow = tk.StringVar(value=str(cfg['grow'] if cfg else 0))
        self.feather = tk.StringVar(value=str(cfg['feather'] if cfg else 8))
        self.exclude = tk.BooleanVar(value=cfg['exclude'] if cfg else False)
        r = ttk.Frame(side, style='Card.TFrame'); r.pack(fill='x', pady=(12, 2))
        ttk.Label(r, text='Grow / shrink (px)', style='Card.TLabel', width=17).pack(side='left')
        app._spin(r, self.grow, -500, 500, 1, 6).pack(side='left')
        r = ttk.Frame(side, style='Card.TFrame'); r.pack(fill='x', pady=2)
        ttk.Label(r, text='Feather (px)', style='Card.TLabel', width=17).pack(side='left')
        app._spin(r, self.feather, 0, 1000, 1, 6).pack(side='left')
        ttk.Checkbutton(side, text='Exclude this area instead\n(use the file everywhere else)', variable=self.exclude, style='Card.TCheckbutton').pack(anchor='w', pady=(6, 0))
        for v in (self.grow, self.feather):
            v.trace_add('write', lambda *a: self._schedule_preview())
        self.show_soft = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text='Preview grow and feather', variable=self.show_soft, style='Card.TCheckbutton',
                        command=self._schedule_preview).pack(anchor='w', pady=(6, 0))
        self.info = ttk.Label(side, text='Loading the image…', style='Muted.TLabel', wraplength=int(290 * app.ui_scale), justify='left')
        self.info.pack(anchor='w', pady=(10, 0))
        br = ttk.Frame(side, style='Card.TFrame'); br.pack(side='bottom', fill='x')
        ttk.Button(br, text='Cancel', command=self.win.destroy).pack(side='right')
        ttk.Button(br, text='Use mask', style='Accent.TButton', command=self.accept).pack(side='right', padx=4)
        pc = ttk.Frame(self.win, style='Card.TFrame', padding=4)
        pc.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        cv = tk.Canvas(pc, bg=C['checker'], highlightthickness=0)
        cv.pack(fill='both', expand=True)
        self.view = ImageView(cv, on_click=self.click, on_drag=self.drag, overlay=self.draw_overlay, pan_button=2)
        self.view.set_image(None, 'Loading…')
        self._soft = None; self._after = None
        self._refresh_list()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            rgb, info = E.load_for_blend(self.path)
            full = (rgb.shape[1], rgb.shape[0])
            disp = E.display_image(rgb, self.MAX_SIDE)
            rgb = None
            self.app.q.put(('call', self._loaded, disp, full))
            pred = self.app.models.sam2('large')
            import torch
            with torch.inference_mode():
                pred.set_image(disp)
            self.app.q.put(('call', self._encoded, pred))
        except Exception as ex:
            traceback.print_exc()
            self.app.q.put(('call', lambda: self.info.configure(text=f'SAM2 could not start: {ex}')))

    def _loaded(self, disp, full):
        if not self.win.winfo_exists(): return
        self.display, self.full = disp, full
        self.factor = full[0] / disp.shape[1]
        self.view.set_image(disp, os.path.basename(self.path))
        self.info.configure(text='Encoding the image with SAM2.1 (large, float32)…')

    def _encoded(self, pred):
        if not self.win.winfo_exists(): return
        self.predictor = pred
        self.info.configure(text='Ready: click the area to select.')
        for i in range(len(self.sel)): self._predict(i)
        self.view.redraw()

    def _refresh_list(self):
        self.lb.delete(0, 'end')
        for i, s in enumerate(self.sel):
            n = len(s['points'])
            self.lb.insert('end', f'Selection {i + 1}: {n} point{"s" if n != 1 else ""}{", box" if s["box"] else ""}')
        self.lb.selection_set(self.cur)

    def _pick(self):
        sel = self.lb.curselection()
        if sel: self.cur = sel[0]; self.view.redraw()

    def add_sel(self):
        self.sel.append(dict(points=[], box=None)); self.masks.append(None)
        self.cur = len(self.sel) - 1; self._refresh_list(); self.view.redraw()

    def del_sel(self):
        if len(self.sel) == 1:
            self.clear_pts(); return
        del self.sel[self.cur]; del self.masks[self.cur]
        self.cur = min(self.cur, len(self.sel) - 1); self._refresh_list(); self._schedule_preview()

    def clear_pts(self):
        self.sel[self.cur] = dict(points=[], box=None); self.masks[self.cur] = None
        self._refresh_list(); self._schedule_preview()

    def click(self, x, y, button, shift):
        if self.predictor is None or shift: return
        s = self.sel[self.cur]
        # clicking an existing point removes it
        r = min(8 / self.view.cur_scale(), 0.02 * max(self.display.shape[:2]))
        for k, (px, py, pos) in enumerate(s['points']):
            if (px - x) ** 2 + (py - y) ** 2 <= r * r:
                del s['points'][k]; break
        else:
            s['points'].append((float(x), float(y), button == 1))
        self._predict(self.cur); self._refresh_list(); self._schedule_preview()

    def drag(self, phase, x, y):
        if self.predictor is None: return
        if phase == 'start': self.box_drag = (x, y, x, y)
        elif phase == 'move' and self.box_drag: self.box_drag = self.box_drag[:2] + (x, y)
        elif phase == 'end' and self.box_drag:
            x0, y0 = self.box_drag[:2]
            self.box_drag = None
            h, w = self.display.shape[:2]
            b = (max(0, min(x0, x)), max(0, min(y0, y)), min(w, max(x0, x)), min(h, max(y0, y)))
            if b[2] - b[0] > 3 and b[3] - b[1] > 3:
                self.sel[self.cur]['box'] = b
                self._predict(self.cur); self._refresh_list(); self._schedule_preview()
        self.view.redraw()

    def _predict(self, i):
        s = self.sel[i]
        if not s['points'] and not s['box']:
            self.masks[i] = None; return
        import torch
        pts = np.array([[p[0], p[1]] for p in s['points']], np.float32) if s['points'] else None
        lab = np.array([1 if p[2] else 0 for p in s['points']], np.int32) if s['points'] else None
        box = np.array(s['box'], np.float32) if s['box'] else None
        multi = box is None and len(s['points']) <= 1
        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(point_coords=pts, point_labels=lab, box=box, multimask_output=multi)
        self.masks[i] = (masks[int(np.argmax(scores))] if multi else masks[0]) > 0.5

    def union(self):
        ms = [m for m in self.masks if m is not None]
        if not ms: return None
        out = np.zeros_like(ms[0], bool)
        for m in ms: out |= m
        return out

    def _schedule_preview(self):
        if self._after: self.win.after_cancel(self._after)
        self._after = self.win.after(150, self._update_soft)

    def _update_soft(self):
        self._after = None
        u = self.union()
        self._soft = None
        if u is not None and self.show_soft.get():
            try: g, f = int(self.grow.get()) / self.factor, int(self.feather.get()) / self.factor
            except ValueError: g, f = 0, 0
            self._soft = E.expand_feather(u, int(round(g)), f)
        self._ov = None
        self.view.redraw()

    def draw_overlay(self, view):
        if self.display is None: return
        u = self.union()
        cv = view.cv
        if u is not None:
            soft = self._soft if self._soft is not None else u.astype(np.float32)
            if self.exclude.get(): soft = 1 - soft
            key = (id(soft), view.cur_scale(), view.cx, view.cy, cv.winfo_width(), cv.winfo_height())
            if getattr(self, '_ov', None) is None or self._ov[0] != key:
                h, w = soft.shape
                rgba = np.zeros((h, w, 4), np.uint8)
                rgba[..., 0], rgba[..., 1], rgba[..., 2] = 255, 110, 85
                rgba[..., 3] = (soft * 110).astype(np.uint8)
                pil = Image.fromarray(rgba)
                s = view.cur_scale()
                if view.scale is None:
                    tw, th = max(1, int(w * s)), max(1, int(h * s))
                    img = pil.resize((tw, th), Image.BILINEAR)
                    pos = ((cv.winfo_width() - tw) / 2, (cv.winfo_height() - th) / 2)
                else:
                    cw, ch = cv.winfo_width(), cv.winfo_height()
                    x0, y0 = view.cx - cw / 2 / s, view.cy - ch / 2 / s
                    ix0, iy0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
                    ix1, iy1 = min(w, int(np.ceil(x0 + cw / s)) + 1), min(h, int(np.ceil(y0 + ch / s)) + 1)
                    img = pil.crop((ix0, iy0, ix1, iy1)).resize((max(1, round((ix1 - ix0) * s)), max(1, round((iy1 - iy0) * s))), Image.NEAREST)
                    pos = ((ix0 - x0) * s, (iy0 - y0) * s)
                self._ov = (key, ImageTk.PhotoImage(img), pos)
            cv.create_image(*self._ov[2], image=self._ov[1], anchor='nw')
        for i, s in enumerate(self.sel):
            on = i == self.cur
            for px, py, pos in s['points']:
                X, Y = view.to_canvas(px, py)
                cv.create_oval(X - 6, Y - 6, X + 6, Y + 6, fill=C['ok'] if pos else C['err'], outline='#000' if on else C['muted'], width=2 if on else 1)
            if s['box']:
                (X0, Y0), (X1, Y1) = view.to_canvas(*s['box'][:2]), view.to_canvas(*s['box'][2:])
                cv.create_rectangle(X0, Y0, X1, Y1, outline=C['info'] if on else C['muted'], width=2 if on else 1, dash=() if on else (3, 3))
        if self.box_drag:
            (X0, Y0), (X1, Y1) = view.to_canvas(*self.box_drag[:2]), view.to_canvas(*self.box_drag[2:])
            cv.create_rectangle(X0, Y0, X1, Y1, outline=C['warn'], width=2, dash=(4, 3))

    def accept(self):
        u = self.union()
        if u is None:
            messagebox.showinfo(APP, 'Select something first (click on the image).', parent=self.win); return
        try: grow, feather = int(self.grow.get()), int(self.feather.get())
        except ValueError:
            messagebox.showerror(APP, 'Grow and feather must be whole numbers.', parent=self.win); return
        self.app.set_mask(dict(path=self.path, selections=copy.deepcopy(self.sel), union=u.astype(np.float32), full_size=self.full,
                               grow=grow, feather=max(0, feather), exclude=bool(self.exclude.get())))
        self.win.destroy()


def main():
    import argparse
    ap = argparse.ArgumentParser(description='RAW Demosaic desktop')
    ap.add_argument('files', nargs='*', help='RAW files or images to load')
    ap.add_argument('--list', help='a text file naming more files, one per line')
    ap.add_argument('--settings', help='a settings file saved with Save settings… to apply')
    ap.add_argument('--process', action='store_true', help='start Process all at once (a batch)')
    args = ap.parse_args()
    files = list(args.files)
    if args.list:
        with open(args.list, encoding='utf-8') as f:
            files += [ln.strip() for ln in f if ln.strip()]
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)   # crisp text on scaled displays
    except Exception:
        pass
    root = tk.Tk()
    app = App(root)
    if files or args.settings:
        root.after(300, lambda: app.open_from_command_line([p for p in files if os.path.isfile(p)], args.settings, args.process))
    root.mainloop()


if __name__ == '__main__':
    main()
