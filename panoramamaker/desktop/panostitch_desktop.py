# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""PANOSTITCH desktop — tkinter front end for panostitch_engine.

All heavy work runs on one worker thread; it talks to the UI only through a queue that the Tk
loop drains every 50ms, so the window never freezes and Tk is only ever touched from the main thread.
"""
import base64
import json
import math
import os
import queue
import threading
import time
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import cv2
import numpy as np

import panostitch_engine as E
import panostitch_hugin as PH

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = os.name == 'nt'
UI_FONT, MONO_FONT = ('Segoe UI', 'Consolas') if WINDOWS else ('DejaVu Sans', 'DejaVu Sans Mono')
# installed with the other desktop apps (a release folder holding suite.json): its shared environment and models
SUITE = os.path.normpath(os.path.join(HERE, '..', '..'))
IN_SUITE = os.path.isfile(os.path.join(SUITE, 'suite.json'))
ENV_ROOT = SUITE if IN_SUITE else HERE
# Linux gets its own environment and settings, so one folder can serve both systems (e.g. WSL on /mnt/c)
VENV_PYTHON = os.path.join(ENV_ROOT, '.venv', 'Scripts', 'python.exe') if WINDOWS else os.path.join(ENV_ROOT, '.venv-linux', 'bin', 'python')
SETTINGS_PATH = os.path.join(HERE, 'panostitch_settings.json' if os.name == 'nt' else 'panostitch_settings_linux.json')
DEFAULT_MODELS = os.path.join(SUITE, 'models', 'panoramamaker') if IN_SUITE else os.path.normpath(os.path.join(HERE, '..', 'models'))

C = dict(bg='#121417', card='#1d2025', card2='#252930', text='#e2e5ea', muted='#9aa3ad', border='#353b44',
         input='#15181c', accent='#ff6e55', accent_hover='#ff8670', ok='#a5d6a7', err='#ef9a9a', info='#80deea',
         warn='#ffd98a', track='#2c3138', checker='#262a30')

PROJECTIONS = [('auto', 'Auto (rectilinear → cylindrical → equirectangular)'), ('rectilinear', 'Rectilinear'),
               ('cylindrical', 'Cylindrical'), ('equirectangular', 'Equirectangular (360°)')]
MATCHERS = [('auto', 'AUTO (ALIKED→SIFT)'), ('aliked', 'ALIKED + LightGlue only'), ('sift', 'SIFT only (no model)')]
SCALES = ['100%', '75%', '50%', '25%']
# (label, Stitcher match_res): None = full resolution — ALIKED at the most the GPU holds, every
# match then refined on the full-resolution photos; SIFT runs on the full photos directly
# parallax warp stiffness (label, value): lower bends more freely to the matches
WARP_STIFFNESS = [('Flexible', 0.1), ('Normal', 1.0), ('Stiff', 10.0)]
# seam blend width (label, Gaussian sigma in seam-map pixels; the seam map is ~2000px across)
# Multi-band: brightness/colour fade over up to a quarter of the panorama, detail switches within the
# first value's width (soft: wider, for moving clouds or water). Values: (sigma, multi-band)
# Blending is independent of seam cutting: seam cutting decides WHERE photos meet (graph cut; without
# it, at the line between photo centres), blending decides HOW. 'Feather (classic)' = no seams at all.
SEAM_BLEND = [('Multi-band (full pyramid)', (1.0, 'full')), ('Multi-band', (1.0, True)),
              ('Multi-band soft', (4.0, True)), ('Sharp seam', (0.8, False)),
              ('Normal', (2.0, False)), ('Soft seam', (5.0, False)), ('Very soft', (12.0, False)), ('Widest', (30.0, False)),
              ('Feather (classic)', (None, False))]
# how much of the traced horizon's shape is straightened (label, smoothing as a fraction of the width):
# hill silhouettes are part of a traced skyline — usually only the overall tilt/curve should go
HZ_DETAIL = [('Overall tilt & curve', 0.05), ('Medium', 0.015), ('Every bump (exact)', 0.003)]
# aligned layers: "Blended colours" = each photo's own detail with the finished panorama's brightness and
# colour (so layers composited by hand match the blend); "Original colours" = the photo as warped
LAYER_KINDS = ['Blended colours', 'Original colours']
ADDP_MATCHERS = [('ALIKED + LightGlue', 'aliked'), ('SIFT', 'sift')]
ADDP_BLENDS = [('Seam + multi-band (full pyramid)', 'multiband'), ('Feathered (simple)', 'feather')]
HZ_METHOD = [('Level the camera (rotation)', 'rotate'), ('Fix steps (level each photo)', 'photos'),
             ('Straighten the line (move columns)', 'columns')]
MATCH_RES_OPTS = [('Full resolution (refined)', None), ('2048 px', 2048), ('1280 px (fastest)', 1280)]
IMG_TYPES = [('Images', ' '.join('*' + e for e in E.IMAGE_EXTS)), ('All files', '*.*')]
SAVE_TYPES = [('TIFF 16-bit + alpha', '*.tif'), ('PNG + alpha', '*.png'), ('JPEG', '*.jpg')]


def fmt_mp(w, h):
    return f'{w}×{h} ({w * h / 1e6:.0f} MP)'


class Result:
    """A finished image held in RAM: a stitch, an add-photo result or an opened file."""
    def __init__(self, title, rgb, alpha, layout=None, mode=None):
        self.title, self.rgb, self.alpha, self.layout = title, rgb, alpha, layout
        self.mode = mode or (layout.mode if layout else None)
        self.hz_lines = None            # horizon lines traced on it, when kept from the Horizon tab

    @property
    def size(self):
        return self.rgb.shape[1], self.rgb.shape[0]

    def label(self):
        w, h = self.size
        return f'{self.title} — {fmt_mp(w, h)}'


class App:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.job = None
        self.busy = False
        self.shot_cache = {}              # path -> Shot (full-resolution pixels kept, re-stitching is instant to start)
        self.results = []                 # Result list shown in the results box
        self.addp_base = None             # Result used as the base on the Add photo tab
        self.pole_eq = None               # Result used on the Zenith/Nadir tab
        self.devices = ['cpu']
        self.settings = self._load_settings()
        if not self.settings.get('defaults_v2'):
            # 2026-10-21: parallax warp, seam cutting and full-pyramid multi-band became the defaults —
            # applied once over older saved settings (which had them off); later choices are kept
            self.settings.update(warp=True, seams=True, seam_blend='Multi-band (full pyramid)', defaults_v2=True)
        if not self.settings.get('defaults_v3'):
            # 2026-10-22: Hugin control points default to all good matches (the old cap of 40 thinned them
            # into clusters); an old saved 40 is moved over once
            if str(self.settings.get('cp_per_pair', '40')) == '40': self.settings['cp_per_pair'] = '0'
            self.settings['defaults_v3'] = True
        self._previews = {}               # canvas -> PhotoImage (kept alive)

        self.ui_scale = max(1.0, root.winfo_fpixels('1i') / 96)
        if not WINDOWS: self._pick_linux_fonts()
        root.title('PANOSTITCH desktop')
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(self.settings.get('geometry') or f'{int(sw * 0.85)}x{int(sh * 0.82)}+{int(sw * 0.07)}+{int(sh * 0.05)}')
        root.minsize(int(900 * self.ui_scale), int(560 * self.ui_scale))
        if self.settings.get('zoomed'):
            try:
                root.state('zoomed') if WINDOWS else root.attributes('-zoomed', True)   # Linux Tk has no 'zoomed' state
            except tk.TclError:
                pass
        root.configure(bg=C['bg'])
        self._style()
        self._build()
        self._refresh_models()
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        root.after(50, self._poll)
        threading.Thread(target=self._probe_devices, daemon=True).start()

    @staticmethod
    def _pick_linux_fonts():
        """Linux installs differ: use the first good font that's really there (Tk silently falls back to
        a bare bitmap font for a missing name), else Tk's own default fonts."""
        global UI_FONT, MONO_FONT
        fams = set(tkfont.families())
        ui = next((f for f in ('DejaVu Sans', 'Noto Sans', 'Ubuntu', 'Cantarell', 'Liberation Sans', 'FreeSans') if f in fams), None)
        mono = next((f for f in ('DejaVu Sans Mono', 'Noto Sans Mono', 'Ubuntu Mono', 'Liberation Mono', 'FreeMono') if f in fams), None)
        UI_FONT = ui or tkfont.nametofont('TkDefaultFont').actual('family')
        MONO_FONT = mono or tkfont.nametofont('TkFixedFont').actual('family')

    # ─────────────────────────────── settings
    def _load_settings(self):
        try:
            with open(SETTINGS_PATH, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_settings(self):
        s = self.settings
        zoomed = self.root.state() == 'zoomed'
        s.update(zoomed=zoomed, models_dir=self.models_dir.get(), model=self.model.get(), device_auto_offered=True,
                 match_res=self.match_res.get(), cp_per_pair=self.cp_per_pair.get(),
                 warp=self.warp.get(), warp_stiff=self.warp_stiff.get(),
                 seams=self.seams.get(), seam_blend=self.seam_blend.get(),
                 device=self.device.get(), matcher=self.matcher_key(), projection=self.projection_key(),
                 scale=self.scale.get(), exposure=self.exposure.get(), flip=self.flip.get(),
                 layer_fmt=self.layer_fmt.get())
        if not zoomed: s['geometry'] = self.root.geometry()
        try:
            with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
                json.dump(s, f, indent=1)
        except Exception:
            pass

    def _remember_dir(self, key, path):
        if path:
            self.settings[key] = os.path.dirname(path) if os.path.isfile(path) else path

    # ─────────────────────────────── look
    def _style(self):
        st = ttk.Style(self.root)
        st.theme_use('clam')
        f = (UI_FONT, 10)
        self.root.option_add('*Font', f)
        st.configure('.', background=C['bg'], foreground=C['text'], fieldbackground=C['input'], bordercolor=C['border'],
                     lightcolor=C['border'], darkcolor=C['border'], troughcolor=C['track'], font=f,
                     selectbackground=C['accent'], selectforeground='#000', insertcolor=C['text'])
        st.configure('Card.TFrame', background=C['card'])
        st.configure('Card.TLabel', background=C['card'], foreground=C['text'])
        st.configure('Muted.TLabel', background=C['card'], foreground=C['muted'], font=(UI_FONT, 9))
        st.configure('Head.TLabel', background=C['card'], foreground=C['accent'], font=(UI_FONT, 11, 'bold'))
        st.configure('Title.TLabel', background=C['bg'], foreground=C['text'], font=(UI_FONT, 16, 'bold'))
        st.configure('Sub.TLabel', background=C['bg'], foreground=C['muted'])
        st.configure('Status.TLabel', background=C['bg'], foreground=C['muted'], font=(UI_FONT, 9))
        st.configure('Card.TCheckbutton', background=C['card'], foreground=C['text'])
        st.map('Card.TCheckbutton', background=[('active', C['card'])], indicatorcolor=[('selected', C['accent']), ('!selected', C['input'])])
        st.configure('Card.TRadiobutton', background=C['card'], foreground=C['text'])
        st.map('Card.TRadiobutton', background=[('active', C['card'])], indicatorcolor=[('selected', C['accent']), ('!selected', C['input'])])
        st.configure('TButton', background=C['card2'], foreground=C['text'], padding=(10, 5), borderwidth=1)
        st.map('TButton', background=[('disabled', C['card']), ('active', C['border'])], foreground=[('disabled', C['muted'])])
        st.configure('Accent.TButton', background=C['accent'], foreground='#111', font=(UI_FONT, 10, 'bold'), padding=(14, 7))
        st.map('Accent.TButton', background=[('disabled', C['card2']), ('active', C['accent_hover'])], foreground=[('disabled', C['muted'])])
        st.configure('TCombobox', fieldbackground=C['input'], background=C['card2'], foreground=C['text'], arrowcolor=C['text'])
        st.map('TCombobox', fieldbackground=[('readonly', C['input'])], foreground=[('readonly', C['text'])],
               selectbackground=[('readonly', C['input'])], selectforeground=[('readonly', C['text'])])
        self.root.option_add('*TCombobox*Listbox.background', C['input'])
        self.root.option_add('*TCombobox*Listbox.foreground', C['text'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', C['accent'])
        st.configure('TEntry', fieldbackground=C['input'], foreground=C['text'])
        st.configure('TSpinbox', fieldbackground=C['input'], foreground=C['text'], arrowcolor=C['text'], background=C['card2'])
        st.configure('Horizontal.TProgressbar', background=C['accent'], troughcolor=C['track'], thickness=10)
        st.configure('TNotebook', background=C['bg'], borderwidth=0)
        st.configure('TNotebook.Tab', background=C['card'], foreground=C['muted'], padding=(16, 6))
        st.map('TNotebook.Tab', background=[('selected', C['card2'])], foreground=[('selected', C['accent'])])
        st.configure('Vertical.TScrollbar', background=C['card2'], troughcolor=C['card'], arrowcolor=C['muted'])

    def _card(self, parent, title=None, **pack):
        fr = ttk.Frame(parent, style='Card.TFrame', padding=10)
        fr.pack(**({'fill': 'x', 'pady': (0, 8)} | pack))
        if title:
            ttk.Label(fr, text=title, style='Head.TLabel').pack(anchor='w', pady=(0, 6))
        return fr

    def _listbox(self, parent, height):
        lb = tk.Listbox(parent, height=height, bg=C['input'], fg=C['text'], selectbackground=C['accent'],
                        selectforeground='#000', highlightthickness=1, highlightbackground=C['border'],
                        highlightcolor=C['accent'], relief='flat', activestyle='none', exportselection=False)
        return lb

    def _row(self, parent, label, widget_factory):
        r = ttk.Frame(parent, style='Card.TFrame')
        r.pack(fill='x', pady=2)
        ttk.Label(r, text=label, style='Card.TLabel', width=13).pack(side='left')
        w = widget_factory(r)
        w.pack(side='left', fill='x', expand=True)
        return w

    def _scroll_column(self, tab):
        """Fixed-width left column that scrolls when it is taller than the window (high DPI, small
        screens). Returns (inner, pinned): inner scrolls, pinned stays visible at the bottom."""
        outer = ttk.Frame(tab, width=int(440 * self.ui_scale))
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

        def wheel(e):
            if sb.winfo_ismapped(): cv.yview_scroll(-1 if e.delta > 0 else 1, 'units')
        # Windows/macOS send <MouseWheel>; X11 sends buttons 4 (up) and 5 (down)
        wheel_events = (('<MouseWheel>', wheel), ('<Button-4>', lambda e: wheel(type('E', (), {'delta': 1}))),
                        ('<Button-5>', lambda e: wheel(type('E', (), {'delta': -1}))))
        outer.bind('<Enter>', lambda e: [self.root.bind_all(ev, fn) for ev, fn in wheel_events])
        outer.bind('<Leave>', lambda e: [self.root.unbind_all(ev) for ev, _ in wheel_events])
        return inner, pinned

    def _preview_canvas(self, parent):
        cv = tk.Canvas(parent, bg=C['checker'], highlightthickness=0)
        cv.bind('<Configure>', lambda e, c=cv: self._redraw_preview(c))
        return cv

    # ─────────────────────────────── layout
    def _build(self):
        top = ttk.Frame(self.root, padding=(14, 10, 14, 4))
        top.pack(fill='x')
        ttk.Label(top, text='PANOSTITCH', style='Title.TLabel').pack(side='left')
        ttk.Label(top, text='  desktop · full VRAM and system RAM, no browser limits', style='Sub.TLabel').pack(side='left', pady=(5, 0))
        ttk.Button(top, text='Save project…', command=self.save_project).pack(side='right')
        ttk.Button(top, text='Open project…', command=self.open_project).pack(side='right', padx=6)
        self.root.bind('<Control-s>', lambda e: self.save_project())
        self.root.bind('<Control-o>', lambda e: self.open_project())

        # bottom: progress + log (packed before the notebook so it keeps its height)
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
        self.log_text = tk.Text(lf, height=5, bg=C['input'], fg=C['text'], relief='flat', wrap='word',
                                highlightthickness=1, highlightbackground=C['border'], font=(MONO_FONT, 9), state='disabled')
        sb = ttk.Scrollbar(lf, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        self.log_text.pack(side='left', fill='x', expand=True)
        sb.pack(side='left', fill='y')
        for tag in ('ok', 'err', 'info', 'warn'):
            self.log_text.tag_configure(tag, foreground=C[tag])
        self.log_text.tag_configure('time', foreground=C['muted'])

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill='both', expand=True, padx=14, pady=(4, 8))
        self._build_stitch_tab()
        self._build_addp_tab()
        self._build_pole_tab()
        self._build_horizon_tab()

    def _build_horizon_tab(self):
        tab = ttk.Frame(self.nb, padding=(0, 8, 0, 0))
        self.nb.add(tab, text='  4 · Horizon  ')
        left, pinned = self._scroll_column(tab)
        right = ttk.Frame(tab)
        right.pack(side='left', fill='both', expand=True, padx=(10, 0))
        self.hz_src, self.hz_lines, self.hz_active, self.hz_live, self.hz_history = None, [], None, [], []
        self.hz_states, self.hz_pos = [], -1      # straightening history: [{'result', 'lines', 'label'}]
        self.hz_view = None          # (scale, ox, oy, tw, th) of the displayed image on the canvas

        c = self._card(left, 'Panorama')
        br = ttk.Frame(c, style='Card.TFrame')
        br.pack(fill='x')
        ttk.Button(br, text='Use selected result', command=lambda: self.send_to('hz')).pack(side='left')
        ttk.Button(br, text='Open file…', command=lambda: self._open_result('Open a panorama', self._set_hz_src)).pack(side='left', padx=4)
        self.hz_lbl = ttk.Label(c, text='No panorama loaded.', style='Muted.TLabel')
        self.hz_lbl.pack(anchor='w', pady=(4, 0))

        c = self._card(left, 'Trace the horizon')
        self.hz_mode = tk.StringVar(value='magnetic')
        ttk.Radiobutton(c, text='Magnetic — click along the horizon; each piece snaps to the\nedge between your clicks.',
                        value='magnetic', variable=self.hz_mode, style='Card.TRadiobutton', command=self._hz_mode_changed).pack(anchor='w')
        ttk.Radiobutton(c, text='Freehand — click to add straight pieces, or drag to draw.', value='freehand', variable=self.hz_mode,
                        style='Card.TRadiobutton', command=self._hz_mode_changed).pack(anchor='w', pady=(4, 0))
        br = ttk.Frame(c, style='Card.TFrame')
        br.pack(fill='x', pady=(6, 0))
        ttk.Button(br, text='Undo', command=self.hz_undo).pack(side='left')
        self.root.bind('<Control-z>', lambda e: self.hz_undo() if self.nb.index('current') == 3 else None)
        ttk.Button(br, text='Clear', command=self.hz_clear).pack(side='left', padx=4)
        ttk.Label(c, text='Switch tools at any time; each click carries the current line on.\n'
                          'Double-click or Enter ends the line — the next click starts a new one.\n'
                          'Click a line\'s end (circle) to carry that line on, or reach another\n'
                          'line\'s end to join them. Lines may cover different parts of the\n'
                          'width; beyond them the correction is held level.', style='Muted.TLabel').pack(anchor='w', pady=(6, 0))
        self.hz_method = tk.StringVar(value=self.settings.get('hz_method', HZ_METHOD[0][0]))
        if self.hz_method.get() not in dict(HZ_METHOD): self.hz_method.set(HZ_METHOD[0][0])
        self._row(c, 'Method', lambda p: ttk.Combobox(p, textvariable=self.hz_method, state='readonly', values=[v for v, _ in HZ_METHOD]))
        ttk.Label(c, text='Level the camera: turns the whole scene (fixes tilt — the sine-wave horizon\n'
                          'of a sphere — and keeps verticals vertical). Fix steps: turns each photo a\n'
                          'little so the horizon carries on smoothly where one photo meets the next.\n'
                          'Straighten the line: moves columns up/down to take out remaining wiggles.',
                  style='Muted.TLabel').pack(anchor='w', pady=(2, 4))
        self.hz_detail = tk.StringVar(value=self.settings.get('hz_detail', HZ_DETAIL[0][0]))
        if self.hz_detail.get() not in dict(HZ_DETAIL): self.hz_detail.set(HZ_DETAIL[0][0])
        self._row(c, 'Line detail', lambda p: ttk.Combobox(p, textvariable=self.hz_detail, state='readonly', values=[v for v, _ in HZ_DETAIL]))

        self.hz_btn = ttk.Button(pinned, text='Straighten horizon (re-blend)', style='Accent.TButton', command=self.hz_run)
        self.hz_btn.pack(fill='x', pady=(4, 0))

        c = self._card(right, 'Result')          # same place as the Stitch tab's results and Save
        br = ttk.Frame(c, style='Card.TFrame')
        br.pack(fill='x')
        self.hz_back_btn = ttk.Button(br, text='◀ Back', command=self.hz_back, state='disabled')
        self.hz_back_btn.pack(side='left')
        self.hz_fwd_btn = ttk.Button(br, text='Forward ▶', command=self.hz_forward, state='disabled')
        self.hz_fwd_btn.pack(side='left', padx=(4, 0))
        ttk.Label(br, text='  History', style='Card.TLabel').pack(side='left')
        self.hz_hist = tk.StringVar()
        self.hz_hist_box = ttk.Combobox(br, textvariable=self.hz_hist, state='readonly', width=34)
        self.hz_hist_box.pack(side='left', padx=(4, 0))
        self.hz_hist_box.bind('<<ComboboxSelected>>', self._hz_pick_state)
        br = ttk.Frame(c, style='Card.TFrame')
        br.pack(fill='x', pady=(6, 0))
        self.hz_save_btn = ttk.Button(br, text='Save…', command=lambda: self.save_image_dialog(self.hz_src), state='disabled')
        self.hz_save_btn.pack(side='left')
        ttk.Button(br, text='Keep in results list', command=self.hz_keep).pack(side='left', padx=4)
        ttk.Button(br, text='Save aligned layers…', command=lambda: self.save_layers(self.hz_src)).pack(side='left', padx=(12, 0))
        ttk.Combobox(br, textvariable=self.layer_kind, state='readonly', values=LAYER_KINDS, width=16).pack(side='left', padx=(4, 0))
        ttk.Checkbutton(br, text='seam masks', variable=self.layer_masks, style='Card.TCheckbutton').pack(side='left', padx=(4, 0))
        ttk.Button(br, text='Export Hugin project with this horizon…', command=self.hz_export_hugin).pack(side='left', padx=(4, 0))
        ttk.Label(br, text='Straightening works on the version shown; Back/History return to earlier ones.',
                  style='Muted.TLabel').pack(side='left', padx=(10, 0))

        pc = self._card(right, None, fill='both', expand=True)
        zb = ttk.Frame(pc, style='Card.TFrame')
        zb.pack(fill='x', pady=(0, 6))
        ttk.Button(zb, text='−', width=3, command=lambda: self.hz_zoom_by(0.8)).pack(side='left')
        ttk.Button(zb, text='+', width=3, command=lambda: self.hz_zoom_by(1.25)).pack(side='left', padx=(4, 0))
        ttk.Button(zb, text='Fit', command=lambda: self.hz_zoom_set(1.0)).pack(side='left', padx=(8, 0))
        ttk.Button(zb, text='100%', command=lambda: self.hz_zoom_set(1.0 / self.hz_fit)).pack(side='left', padx=(4, 0))
        self.hz_zoom_lbl = ttk.Label(zb, text='', style='Card.TLabel')
        self.hz_zoom_lbl.pack(side='left', padx=(10, 0))
        ttk.Label(zb, text='mouse wheel: zoom · right- or middle-drag: move around', style='Muted.TLabel').pack(side='right')
        self.hz_zoom, self.hz_center, self.hz_fit = 1.0, None, 1.0
        self.hz_canvas = tk.Canvas(pc, bg=C['checker'], highlightthickness=0, cursor='crosshair')
        self.hz_canvas.pack(fill='both', expand=True)
        cv = self.hz_canvas
        cv.bind('<MouseWheel>', self._hz_wheel)
        cv.bind('<Button-4>', lambda e: self._hz_wheel(e, 1))       # X11 wheel
        cv.bind('<Button-5>', lambda e: self._hz_wheel(e, -1))
        for b in (2, 3):
            cv.bind(f'<ButtonPress-{b}>', self._hz_pan_start)
            cv.bind(f'<B{b}-Motion>', self._hz_pan_move)
        cv.bind('<Configure>', lambda e: self._hz_redraw(full=True))
        cv.bind('<Button-1>', self._hz_press)
        cv.bind('<B1-Motion>', self._hz_drag)
        cv.bind('<Motion>', self._hz_motion)
        cv.bind('<Double-Button-1>', lambda e: self._hz_finish())
        self.root.bind('<Return>', lambda e: self._hz_finish() if self.nb.index('current') == 3 else None)

    def _build_stitch_tab(self):
        tab = ttk.Frame(self.nb, padding=(0, 8, 0, 0))
        self.nb.add(tab, text='  1 · Stitch  ')
        left, pinned = self._scroll_column(tab)
        right = ttk.Frame(tab)
        right.pack(side='left', fill='both', expand=True, padx=(10, 0))

        c = self._card(left, 'Photos')
        lf = ttk.Frame(c, style='Card.TFrame')
        lf.pack(fill='both', expand=True)
        self.photo_list = self._listbox(lf, 8)
        self.photo_list.configure(selectmode='extended')
        sb = ttk.Scrollbar(lf, orient='vertical', command=self.photo_list.yview)
        self.photo_list.configure(yscrollcommand=sb.set)
        self.photo_list.pack(side='left', fill='both', expand=True)
        sb.pack(side='left', fill='y')
        self.photo_list.bind('<Delete>', lambda e: self.remove_photos())
        self.photo_paths = []
        br = ttk.Frame(c, style='Card.TFrame')
        br.pack(fill='x', pady=(6, 0))
        ttk.Button(br, text='Add photos…', command=self.add_photos).pack(side='left')
        ttk.Button(br, text='Add folder…', command=self.add_folder).pack(side='left', padx=4)
        ttk.Button(br, text='Remove', command=self.remove_photos).pack(side='left')
        ttk.Button(br, text='Clear', command=self.clear_photos).pack(side='left', padx=4)
        self.photo_count = ttk.Label(c, text='No photos.', style='Muted.TLabel')
        self.photo_count.pack(anchor='w', pady=(4, 0))

        c = self._card(left, 'Options')
        s = self.settings
        self.scale = tk.StringVar(value=s.get('scale', '100%'))
        self._row(c, 'Output size', lambda p: ttk.Combobox(p, textvariable=self.scale, values=SCALES, width=10))
        self.projection = tk.StringVar(value=dict(PROJECTIONS).get(s.get('projection', 'auto'), PROJECTIONS[0][1]))
        self._row(c, 'Projection', lambda p: ttk.Combobox(p, textvariable=self.projection, state='readonly', values=[v for _, v in PROJECTIONS]))
        self.matcher = tk.StringVar(value=dict(MATCHERS).get(s.get('matcher', 'auto'), MATCHERS[0][1]))
        self._row(c, 'Matcher', lambda p: ttk.Combobox(p, textvariable=self.matcher, state='readonly', values=[v for _, v in MATCHERS]))
        self.model = tk.StringVar(value=s.get('model', ''))
        self.model_box = self._row(c, 'Model', lambda p: ttk.Combobox(p, textvariable=self.model, state='readonly', width=10))
        self.models_dir = tk.StringVar(value=s.get('models_dir', DEFAULT_MODELS))
        r = ttk.Frame(c, style='Card.TFrame')
        r.pack(fill='x', pady=2)
        ttk.Label(r, text='Models folder', style='Card.TLabel', width=13).pack(side='left')
        ttk.Entry(r, textvariable=self.models_dir).pack(side='left', fill='x', expand=True)
        ttk.Button(r, text='…', width=3, command=self.choose_models_dir).pack(side='left', padx=(4, 0))
        # empty until the GPU probe picks the first ('auto'); older settings predate 'auto', so offer it once
        self.device = tk.StringVar(value=s.get('device', '') if s.get('device_auto_offered') else '')
        self.device_box = self._row(c, 'Run model on', lambda p: ttk.Combobox(p, textvariable=self.device, state='readonly', values=['cpu']))
        self.match_res = tk.StringVar(value=s.get('match_res', MATCH_RES_OPTS[0][0]))
        self._row(c, 'Match at', lambda p: ttk.Combobox(p, textvariable=self.match_res, state='readonly', values=[v for v, _ in MATCH_RES_OPTS]))
        self.exposure = tk.BooleanVar(value=s.get('exposure', False))
        ttk.Checkbutton(c, text='Match exposure between photos', variable=self.exposure, style='Card.TCheckbutton').pack(anchor='w', pady=(6, 0))
        self.warp = tk.BooleanVar(value=s.get('warp', True))
        r = ttk.Frame(c, style='Card.TFrame')
        r.pack(fill='x')
        ttk.Checkbutton(r, text='Correct parallax (local warp)', variable=self.warp, style='Card.TCheckbutton').pack(side='left')
        self.warp_stiff = tk.StringVar(value=s.get('warp_stiff', WARP_STIFFNESS[1][0]))
        ttk.Combobox(r, textvariable=self.warp_stiff, state='readonly', values=[v for v, _ in WARP_STIFFNESS], width=10).pack(side='right')
        self.seams = tk.BooleanVar(value=s.get('seams', True))
        self.seams_chk = ttk.Checkbutton(c, text='Seam cutting (sharpest photo wins)', variable=self.seams, style='Card.TCheckbutton')
        self.seams_chk.pack(anchor='w')
        self.seam_blend = tk.StringVar(value=s.get('seam_blend', SEAM_BLEND[0][0]))
        if self.seam_blend.get() not in dict(SEAM_BLEND): self.seam_blend.set(SEAM_BLEND[0][0])
        self._row(c, 'Blending', lambda p: ttk.Combobox(p, textvariable=self.seam_blend, state='readonly',
                                                       values=[v for v, _ in SEAM_BLEND], width=22))
        # the classic feather uses no seams, so seam cutting has nothing to do with it
        self.seam_blend.trace_add('write', lambda *a: self.seams_chk.configure(
            state='disabled' if dict(SEAM_BLEND).get(self.seam_blend.get(), (1, 0))[0] is None else 'normal'))
        self.seam_blend.set(self.seam_blend.get())
        self.flip = tk.BooleanVar(value=s.get('flip', False))
        ttk.Checkbutton(c, text='Flip vertically (not normally needed here)', variable=self.flip, style='Card.TCheckbutton').pack(anchor='w')

        self.stitch_btn = ttk.Button(pinned, text='Stitch panorama', style='Accent.TButton', command=self.stitch)
        self.stitch_btn.pack(fill='x', pady=(4, 0))

        c = self._card(right, 'Results')
        self.result_list = self._listbox(c, 3)
        self.result_list.pack(fill='x')
        self.result_list.bind('<<ListboxSelect>>', lambda e: self.root.after_idle(self._show_selected_result))
        br = ttk.Frame(c, style='Card.TFrame')
        br.pack(fill='x', pady=(6, 0))
        self.save_btn = ttk.Button(br, text='Save panorama…', command=self.save_result, state='disabled')
        self.save_btn.pack(side='left')
        self.layers_btn = ttk.Button(br, text='Save aligned layers…', command=self.save_layers, state='disabled')
        self.layers_btn.pack(side='left', padx=4)
        self.layer_fmt = tk.StringVar(value=s.get('layer_fmt', 'PNG'))
        ttk.Combobox(br, textvariable=self.layer_fmt, state='readonly', values=['PNG', 'TIFF 16-bit'], width=11).pack(side='left')
        self.layer_kind = tk.StringVar(value=s.get('layer_kind', LAYER_KINDS[0]))
        if self.layer_kind.get() not in LAYER_KINDS: self.layer_kind.set(LAYER_KINDS[0])
        ttk.Combobox(br, textvariable=self.layer_kind, state='readonly', values=LAYER_KINDS, width=16).pack(side='left', padx=(4, 0))
        self.layer_masks = tk.BooleanVar(value=s.get('layer_masks', True))
        ttk.Checkbutton(br, text='seam masks', variable=self.layer_masks, style='Card.TCheckbutton').pack(side='left', padx=(4, 0))
        ttk.Button(br, text='→ Add photo', command=lambda: self.send_to('addp')).pack(side='right')
        ttk.Button(br, text='→ Zenith/Nadir', command=lambda: self.send_to('pole')).pack(side='right', padx=4)
        ttk.Button(br, text='→ Horizon', command=lambda: self.send_to('hz')).pack(side='right')
        ttk.Button(br, text='Discard', command=self.discard_result).pack(side='right')
        br = ttk.Frame(c, style='Card.TFrame')
        br.pack(fill='x', pady=(6, 0))
        self.hugin_btn = ttk.Button(br, text='Export Hugin project…', command=self.export_hugin, state='disabled')
        self.hugin_btn.pack(side='left')
        ttk.Label(br, text='  points per pair (0 = all)', style='Card.TLabel').pack(side='left')
        self.cp_per_pair = tk.StringVar(value=str(s.get('cp_per_pair', PH.DEFAULT_POINTS)))
        ttk.Spinbox(br, from_=0, to=5000, increment=10, textvariable=self.cp_per_pair, width=6).pack(side='left', padx=(4, 0))
        ttk.Button(br, text='Use as Hugin’s control point detector…', command=self.hugin_detector_help).pack(side='right')
        pc = self._card(right, None, fill='both', expand=True)
        self.stitch_preview = self._preview_canvas(pc)
        self.stitch_preview.pack(fill='both', expand=True)

    def _build_addp_tab(self):
        tab = ttk.Frame(self.nb, padding=(0, 8, 0, 0))
        self.nb.add(tab, text='  2 · Add a photo  ')
        left, pinned = self._scroll_column(tab)
        right = ttk.Frame(tab)
        right.pack(side='left', fill='both', expand=True, padx=(10, 0))

        c = self._card(left, 'Panorama')
        ttk.Label(c, text='Add a photo to a finished panorama — any size, matched with SIFT and\n'
                          'blended in; the canvas grows if the photo reaches past the edge.', style='Muted.TLabel').pack(anchor='w')
        br = ttk.Frame(c, style='Card.TFrame')
        br.pack(fill='x', pady=(6, 0))
        ttk.Button(br, text='Use selected result', command=lambda: self.send_to('addp')).pack(side='left')
        ttk.Button(br, text='Open panorama file…', command=self.addp_open_base).pack(side='left', padx=4)
        self.addp_base_lbl = ttk.Label(c, text='No panorama loaded.', style='Muted.TLabel')
        self.addp_base_lbl.pack(anchor='w', pady=(4, 0))

        c = self._card(left, 'Photo to add')
        self.addp_photo = tk.StringVar()
        r = ttk.Frame(c, style='Card.TFrame')
        r.pack(fill='x')
        ttk.Entry(r, textvariable=self.addp_photo).pack(side='left', fill='x', expand=True)
        ttk.Button(r, text='…', width=3, command=self.addp_choose_photo).pack(side='left', padx=(4, 0))
        s = self.settings
        self.addp_matcher = tk.StringVar(value=s.get('addp_matcher', ADDP_MATCHERS[0][0]))
        if self.addp_matcher.get() not in dict(ADDP_MATCHERS): self.addp_matcher.set(ADDP_MATCHERS[0][0])
        self._row(c, 'Matcher', lambda p: ttk.Combobox(p, textvariable=self.addp_matcher, state='readonly', values=[v for v, _ in ADDP_MATCHERS]))
        self.addp_blend = tk.StringVar(value=s.get('addp_blend', ADDP_BLENDS[0][0]))
        if self.addp_blend.get() not in dict(ADDP_BLENDS): self.addp_blend.set(ADDP_BLENDS[0][0])
        self._row(c, 'Blending', lambda p: ttk.Combobox(p, textvariable=self.addp_blend, state='readonly', values=[v for v, _ in ADDP_BLENDS]))
        self.addp_mode = tk.StringVar(value=s.get('addp_mode', 'top'))
        ttk.Radiobutton(c, text='Photo on top — e.g. to replace a ghosted area', value='top',
                        variable=self.addp_mode, style='Card.TRadiobutton').pack(anchor='w', pady=(6, 0))
        ttk.Radiobutton(c, text='Only fill empty areas — e.g. to fill a gap or extend sky', value='fill',
                        variable=self.addp_mode, style='Card.TRadiobutton').pack(anchor='w')
        ttk.Radiobutton(c, text='Sharpest wins — keep whichever is sharper (seam blending)', value='sharp',
                        variable=self.addp_mode, style='Card.TRadiobutton').pack(anchor='w')
        self.addp_btn = ttk.Button(pinned, text='Add photo', style='Accent.TButton', command=self.addp_run)
        self.addp_btn.pack(fill='x', pady=(4, 0))

        c = self._card(left, 'Result')
        ttk.Label(c, text='Each added photo becomes the new base, so you can add several in turn.', style='Muted.TLabel').pack(anchor='w')
        br = ttk.Frame(c, style='Card.TFrame')
        br.pack(fill='x', pady=(6, 0))
        self.addp_save_btn = ttk.Button(br, text='Save…', command=lambda: self.save_image_dialog(self.addp_base), state='disabled')
        self.addp_save_btn.pack(side='left')
        ttk.Button(br, text='Keep in results list', command=self.addp_keep).pack(side='left', padx=4)

        pc = self._card(right, None, fill='both', expand=True)
        self.addp_preview = self._preview_canvas(pc)
        self.addp_preview.pack(fill='both', expand=True)

    def _build_pole_tab(self):
        tab = ttk.Frame(self.nb, padding=(0, 8, 0, 0))
        self.nb.add(tab, text='  3 · Zenith / Nadir  ')
        left, pinned = self._scroll_column(tab)
        right = ttk.Frame(tab)
        right.pack(side='left', fill='both', expand=True, padx=(10, 0))

        c = self._card(left, 'Equirectangular panorama')
        br = ttk.Frame(c, style='Card.TFrame')
        br.pack(fill='x')
        ttk.Button(br, text='Use selected result', command=lambda: self.send_to('pole')).pack(side='left')
        ttk.Button(br, text='Open file…', command=self.pole_open_eq).pack(side='left', padx=4)
        self.pole_eq_lbl = ttk.Label(c, text='No panorama loaded.', style='Muted.TLabel')
        self.pole_eq_lbl.pack(anchor='w', pady=(4, 0))

        c = self._card(left, '1 · Extract')
        self.pole_fov = tk.StringVar(value=str(self.settings.get('pole_fov', 110)))
        self._row(c, 'Field of view °', lambda p: ttk.Spinbox(p, from_=60, to=170, increment=5, textvariable=self.pole_fov, width=8))
        self.pole_size = tk.StringVar(value=str(self.settings.get('pole_size', 'auto')))
        self._row(c, 'Patch size px', lambda p: ttk.Combobox(p, textvariable=self.pole_size, values=['auto', '2048', '4096', '6144', '8192'], width=8))
        ttk.Label(c, text='Saves zenith (top) and nadir (bottom) patches as flat images to edit\n'
                          '(e.g. remove the tripod). Keep the same field of view to reimport.', style='Muted.TLabel').pack(anchor='w', pady=(4, 0))
        self.pole_extract_btn = ttk.Button(c, text='Extract and save patches…', command=self.pole_extract)
        self.pole_extract_btn.pack(anchor='w', pady=(6, 0))

        c = self._card(left, '2 · Reimport edited patches')
        self.pole_zen = tk.StringVar()
        self.pole_nad = tk.StringVar()
        for label, var in (('Zenith patch', self.pole_zen), ('Nadir patch', self.pole_nad)):
            r = ttk.Frame(c, style='Card.TFrame')
            r.pack(fill='x', pady=2)
            ttk.Label(r, text=label, style='Card.TLabel', width=13).pack(side='left')
            ttk.Entry(r, textvariable=var).pack(side='left', fill='x', expand=True)
            ttk.Button(r, text='…', width=3, command=lambda v=var: self._choose_into(v)).pack(side='left', padx=(4, 0))
        ttk.Label(c, text='Either may be left empty.', style='Muted.TLabel').pack(anchor='w')
        self.pole_import_btn = ttk.Button(c, text='Composite into panorama', style='Accent.TButton', command=self.pole_import)
        self.pole_import_btn.pack(fill='x', pady=(6, 0))
        br = ttk.Frame(c, style='Card.TFrame')
        br.pack(fill='x', pady=(6, 0))
        self.pole_save_btn = ttk.Button(br, text='Save…', command=lambda: self.save_image_dialog(self.pole_eq), state='disabled')
        self.pole_save_btn.pack(side='left')
        ttk.Button(br, text='Keep in results list', command=self.pole_keep).pack(side='left', padx=4)

        pc = self._card(right, None, fill='both', expand=True)
        self.pole_preview = self._preview_canvas(pc)
        self.pole_preview.pack(fill='both', expand=True)

    # ─────────────────────────────── helpers
    def matcher_key(self):
        return {v: k for k, v in MATCHERS}.get(self.matcher.get(), 'auto')

    def projection_key(self):
        return {v: k for k, v in PROJECTIONS}.get(self.projection.get(), 'auto')

    def scale_value(self):
        t = self.scale.get().strip().rstrip('%').strip()
        v = float(t) / 100
        if not 0.01 <= v <= 4: raise ValueError('output size must be between 1% and 400%')
        return v

    def log(self, level, msg):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', time.strftime('%H:%M:%S '), 'time')
        self.log_text.insert('end', msg + '\n', level if level in ('ok', 'err', 'info', 'warn') else '')
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def _refresh_models(self):
        models = E.find_models(self.models_dir.get())
        self.models = models
        keys = list(models)
        self.model_box.configure(values=keys)
        if self.model.get() not in keys:
            # k2048: 4× the browser default's keypoints, still ~1s per pair on a GPU
            self.model.set('k2048' if 'k2048' in keys else (keys[-1] if keys else ''))
        if not keys:
            self.log('warn', f'no raco_aliked_lightglue_pipeline_k*.onnx models in {self.models_dir.get()} — SIFT only')

    def choose_models_dir(self):
        d = filedialog.askdirectory(initialdir=self.models_dir.get() or HERE, title='Folder with the ALIKED ONNX models')
        if d:
            self.models_dir.set(os.path.normpath(d))
            self._refresh_models()

    def _probe_devices(self):
        devs = E.list_devices()
        self.q.put(('call', self._set_devices, devs))

    def _set_devices(self, devs):
        self.devices = devs
        self.device_box.configure(values=devs)
        if self.device.get() not in devs:
            self.device.set(devs[0])
        self.log('info', 'model devices: ' + ', '.join(devs) + ('' if len(devs) > 1 else ' (no CUDA GPU found by ONNX Runtime)'))

    def _choose_into(self, var):
        p = filedialog.askopenfilename(filetypes=IMG_TYPES, initialdir=self.settings.get('open_dir'))
        if p:
            var.set(os.path.normpath(p))
            self._remember_dir('open_dir', p)

    # ─────────────────────────────── worker plumbing
    def run_task(self, label, fn, done=None):
        """fn(job) runs on a worker thread; done(result) then runs on the Tk thread."""
        if self.busy:
            messagebox.showinfo('PANOSTITCH', 'Something is already running — wait for it or press Cancel.')
            return
        self.busy = True
        job = E.Job(log=lambda l, m: self.q.put(('log', l, m)),
                    progress=lambda f, t=None: self.q.put(('progress', f, t)))
        self.job = job
        self._set_busy_ui(True)
        self.progress['value'] = 0
        self.status.configure(text=label + '…')
        t0 = time.time()

        def work():
            try:
                res = fn(job)
                self.q.put(('done', done, res, time.time() - t0))
            except E.Cancelled:
                self.q.put(('cancelled',))
            except MemoryError:
                self.q.put(('error', 'Out of memory — try a smaller output size.', traceback.format_exc()))
            except Exception as ex:
                self.q.put(('error', str(ex) or type(ex).__name__, traceback.format_exc()))
        threading.Thread(target=work, daemon=True).start()

    def _set_busy_ui(self, busy):
        st = 'disabled' if busy else 'normal'
        for b in (self.stitch_btn, self.addp_btn, self.pole_extract_btn, self.pole_import_btn, self.hz_btn):
            b.configure(state=st)
        self.cancel_btn.configure(state='normal' if busy else 'disabled')
        if not busy:
            self._update_buttons()

    def _update_buttons(self):
        has = self.selected_result() is not None
        st = 'normal' if (has and not self.busy) else 'disabled'
        self.save_btn.configure(state=st)
        r = self.selected_result()
        self.layers_btn.configure(state='normal' if (r and r.layout and not self.busy) else 'disabled')
        self.hugin_btn.configure(state='normal' if (r and r.layout and getattr(r.layout, 'edges', None) and not self.busy) else 'disabled')
        self.addp_save_btn.configure(state='normal' if (self.addp_base and not self.busy) else 'disabled')
        self.pole_save_btn.configure(state='normal' if (self.pole_eq and not self.busy) else 'disabled')
        self.hz_save_btn.configure(state='normal' if (self.hz_src and not self.busy) else 'disabled')

    def cancel(self):
        if self.job:
            self.job.cancel_event.set()
            self.status.configure(text='Cancelling…')

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
                    last_prog = None        # progress queued before the finish is stale
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
        if kind == 'done':
            _, cb, res, dt = m
            self.progress['value'] = 1000
            self.status.configure(text=f'Done in {dt:.1f}s.')
            self._set_busy_ui(False)
            if cb:
                try:
                    cb(res)
                except Exception as ex:
                    self.log('err', f'{ex}')
                    traceback.print_exc()
        elif kind == 'cancelled':
            self.progress['value'] = 0
            self.status.configure(text='Cancelled.')
            self.log('warn', 'cancelled')
            self._set_busy_ui(False)
        else:
            self.progress['value'] = 0
            self.status.configure(text='Failed: ' + m[1])
            self.log('err', m[1])
            print(m[2])
            self._set_busy_ui(False)
            messagebox.showerror('PANOSTITCH', m[1])

    # ─────────────────────────────── previews
    def show_preview(self, canvas, result):
        canvas._result = result
        self._redraw_preview(canvas)

    def _redraw_preview(self, canvas):
        canvas.delete('all')
        r = getattr(canvas, '_result', None)
        cw, ch = max(10, canvas.winfo_width()), max(10, canvas.winfo_height())
        if r is None:
            canvas.create_text(cw / 2, ch / 2, text='Nothing to show yet', fill=C['muted'])
            return
        rgb, alpha = r.rgb, r.alpha
        h, w = rgb.shape[:2]
        spans = r.layout.photo_spans() if r.layout is not None else []
        row_h = int(18 * self.ui_scale)
        font = tkfont.Font(family=UI_FONT, size=8)
        # first pass at the bar-less scale to place names in rows; each row holds names that don't collide
        s = min((cw - 8) / w, (ch - 8 - 2 * row_h) / h)
        rows_end, placed = [], []
        for name, l, c, rr, cy in spans:
            label = os.path.splitext(name)[0]
            half = font.measure(label) / 2 + 6
            xc = (cw - w * s) / 2 + c * s
            row = next((k for k, end in enumerate(rows_end) if xc - half > end), len(rows_end))
            if row == len(rows_end): rows_end.append(0)
            rows_end[row] = xc + half
            placed.append((label, l, c, rr, cy, row))
        bar = len(rows_end) * row_h + 6 if spans else 0
        s = min((cw - 8) / w, (ch - 8 - bar) / h)
        tw, th = max(1, int(w * s)), max(1, int(h * s))
        key = (id(r), tw, th)
        cache = getattr(canvas, '_cache', None)
        if cache and cache[0] == key:
            img = cache[1]
        else:
            # downscale in steps of whole pixels first (cheap) so INTER_AREA on a 1GP image stays quick
            step = max(1, int(1 / s) // 2)
            small = E.to_uint8(rgb[::step, ::step])
            small = cv2.resize(small, (tw, th), interpolation=cv2.INTER_AREA)
            if alpha is not None:
                a = cv2.resize(alpha[::step, ::step], (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255
                yy, xx = np.mgrid[0:th, 0:tw]
                chk = np.where(((yy // 10 + xx // 10) % 2)[..., None] == 0, 38, 46).astype(np.float32)
                small = (small * a + chk * (1 - a)).astype(np.uint8)
            ok, buf = cv2.imencode('.png', small[..., ::-1])
            img = tk.PhotoImage(data=base64.b64encode(buf.tobytes()).decode('ascii'), format='png')
            canvas._cache = (key, img)
        x0, y0 = (cw - tw) / 2, (bar + 4) if spans else (ch - th) / 2   # labelled: sit right under the name bar
        canvas.create_image(x0, y0, image=img, anchor='nw')
        palette = (C['accent'], C['info'], C['warn'], C['ok'])
        for i, (label, l, c, rr, cy, row) in enumerate(placed):
            col = palette[i % len(palette)]
            ty = 2 + row * row_h
            ly = ty + row_h - 3
            xl, xc, xr, yc = x0 + l * s, x0 + c * s, x0 + rr * s, y0 + cy * s
            canvas.create_line(xl, ly, xr, ly, fill=col)
            canvas.create_line(xl, ly - 4, xl, ly + 1, fill=col)
            canvas.create_line(xr, ly - 4, xr, ly + 1, fill=col)
            canvas.create_line(xc, ly, xc, yc, fill=col, dash=(2, 3))
            canvas.create_oval(xc - 4, yc - 4, xc + 4, yc + 4, outline='#000', fill=col)
            canvas.create_text(xc, ty, anchor='n', text=label, fill=col, font=font)
        canvas.create_text(10, ch - 6, anchor='sw', text=r.label() + (f' · {r.mode}' if r.mode else ''),
                           fill=C['text'], font=(UI_FONT, 9))

    # ─────────────────────────────── photos
    def _set_photos(self, paths):
        seen, out = set(), []
        for p in paths:
            k = os.path.normcase(os.path.abspath(p))
            if k not in seen:
                seen.add(k); out.append(os.path.abspath(p))
        self.photo_paths = out
        self.photo_list.delete(0, 'end')
        for p in out:
            self.photo_list.insert('end', os.path.basename(p))
        self.photo_count.configure(text=f'{len(out)} photo{"s" if len(out) != 1 else ""}.' if out else 'No photos.')
        live = {os.path.normcase(p) for p in out}
        for k in [k for k in self.shot_cache if os.path.normcase(k) not in live]:
            del self.shot_cache[k]          # free RAM held by photos no longer in the list

    def add_photos(self):
        ps = filedialog.askopenfilenames(filetypes=IMG_TYPES, initialdir=self.settings.get('photo_dir'), title='Photos to stitch')
        if ps:
            self._remember_dir('photo_dir', ps[0])
            self._set_photos(self.photo_paths + [os.path.normpath(p) for p in ps])

    def add_folder(self):
        d = filedialog.askdirectory(initialdir=self.settings.get('photo_dir'), title='Folder of photos to stitch')
        if d:
            self._remember_dir('photo_dir', d)
            ps = sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(E.IMAGE_EXTS))
            self._set_photos(self.photo_paths + [os.path.normpath(p) for p in ps])

    def remove_photos(self):
        sel = set(self.photo_list.curselection())
        self._set_photos([p for i, p in enumerate(self.photo_paths) if i not in sel])

    def clear_photos(self):
        self._set_photos([])

    # ─────────────────────────────── stitch
    def stitch(self):
        if len(self.photo_paths) < 2:
            messagebox.showinfo('PANOSTITCH', 'Add at least two photos.')
            return
        try:
            scale = self.scale_value()
        except ValueError as ex:
            messagebox.showerror('PANOSTITCH', str(ex)); return
        matcher = self.matcher_key()
        model_path = self.models.get(self.model.get())
        if matcher == 'aliked' and not model_path:
            messagebox.showerror('PANOSTITCH', 'ALIKED needs a model file — choose the models folder, or use SIFT.'); return
        opts = dict(matcher=matcher, model_path=model_path, device=self.device.get(), projection=self.projection_key(),
                    exposure=self.exposure.get(), flip=self.flip.get(), scale=scale,
                    match_res=dict(MATCH_RES_OPTS).get(self.match_res.get()),
                    warp=self.warp.get(), warp_stiffness=dict(WARP_STIFFNESS).get(self.warp_stiff.get(), 1.0),
                    seams=self.seams.get() and dict(SEAM_BLEND).get(self.seam_blend.get(), (1.0, True))[0] is not None,
                    seam_blend=dict(SEAM_BLEND).get(self.seam_blend.get(), (1.0, True))[0],
                    multiband=dict(SEAM_BLEND).get(self.seam_blend.get(), (1.0, True))[1])
        paths = list(self.photo_paths)
        self._save_settings()

        def work(job):
            shots = []
            for k, p in enumerate(paths):
                job.check()
                s = self.shot_cache.get(p)
                if s is None:
                    job.progress(0.02 * k / len(paths), f'loading {os.path.basename(p)}')
                    try:
                        s = E.Shot(p)
                    except Exception as ex:
                        job.log('warn', f'{os.path.basename(p)}: could not load ({ex}) — skipped'); continue
                    self.shot_cache[p] = s
                shots.append(s)
            if len(shots) < 2: raise RuntimeError('fewer than two photos could be loaded')
            job.log('info', f'loaded {len(shots)} photos: ' + ', '.join(f'{s.w}×{s.h}' for s in shots[:6]) + (' …' if len(shots) > 6 else ''))
            st = E.Stitcher(shots, job, **opts)
            layouts, unmatched = st.run()
            if not layouts: raise RuntimeError('no group of photos could be laid out')
            out = []
            for k, L in enumerate(layouts):
                job.log('info', f'panorama {k + 1}: {L.mode}, {len(L.nodes)} photos, {fmt_mp(L.W, L.H)}, '
                                f'{L.W * L.H * 6 / 1e9:.1f} GB in RAM')
                f0 = 0.5 + 0.5 * k / len(layouts)
                rgb, alpha = L.render(job, f0, f0 + 0.5 / len(layouts))
                out.append(Result(f'Panorama {len(self.results) + k + 1}', rgb, alpha, layout=L))
                job.log('ok', f'panorama {k + 1} rendered: {", ".join(L.names)}')
            return out
        self.run_task('Stitching', work, self._stitch_done)

    def _stitch_done(self, results):
        for r in results:
            self.results.append(r)
            self.result_list.insert('end', r.label() + f' · {r.mode} · {len(r.layout.nodes)} photos')
        self.result_list.selection_clear(0, 'end')
        self.result_list.selection_set(len(self.results) - len(results))
        self.result_list.see('end')
        self._show_selected_result()

    def selected_result(self):
        sel = self.result_list.curselection()
        return self.results[sel[0]] if sel and sel[0] < len(self.results) else None

    def _show_selected_result(self):
        self.show_preview(self.stitch_preview, self.selected_result())
        self._update_buttons()

    def discard_result(self):
        sel = self.result_list.curselection()
        if not sel: return
        i = sel[0]
        del self.results[i]
        self.result_list.delete(i)
        if self.results:
            self.result_list.selection_set(min(i, len(self.results) - 1))
        self._show_selected_result()

    def _add_result(self, r, extra=''):
        self.results.append(r)
        self.result_list.insert('end', r.label() + extra)

    # ─────────────────────────────── projects
    def _options(self):
        """The stitching / tool options a project carries (not machine things like folders or GPU)."""
        return {'scale': self.scale.get(), 'projection': self.projection_key(), 'matcher': self.matcher_key(),
                'model': self.model.get(), 'match_res': self.match_res.get(), 'exposure': self.exposure.get(),
                'flip': self.flip.get(), 'warp': self.warp.get(), 'warp_stiff': self.warp_stiff.get(),
                'seams': self.seams.get(), 'seam_blend': self.seam_blend.get(), 'cp_per_pair': self.cp_per_pair.get(),
                'layer_fmt': self.layer_fmt.get(), 'layer_kind': self.layer_kind.get(), 'layer_masks': self.layer_masks.get(),
                'addp_matcher': self.addp_matcher.get(), 'addp_blend': self.addp_blend.get(),
                'addp_mode': self.addp_mode.get(), 'hz_method': self.hz_method.get(), 'hz_detail': self.hz_detail.get()}

    def _apply_options(self, o):
        plain = {'scale': self.scale, 'model': self.model, 'match_res': self.match_res, 'exposure': self.exposure,
                 'flip': self.flip, 'warp': self.warp, 'warp_stiff': self.warp_stiff, 'seams': self.seams,
                 'seam_blend': self.seam_blend, 'cp_per_pair': self.cp_per_pair, 'layer_fmt': self.layer_fmt,
                 'layer_kind': self.layer_kind, 'layer_masks': self.layer_masks,
                 'addp_matcher': self.addp_matcher, 'addp_blend': self.addp_blend, 'addp_mode': self.addp_mode,
                 'hz_method': self.hz_method, 'hz_detail': self.hz_detail}
        for k, var in plain.items():
            if k in o:
                try: var.set(o[k])
                except Exception: pass
        if o.get('projection') in dict(PROJECTIONS): self.projection.set(dict(PROJECTIONS)[o['projection']])
        if o.get('matcher') in dict(MATCHERS): self.matcher.set(dict(MATCHERS)[o['matcher']])
        if self.model.get() not in self.models: self._refresh_models()

    def save_project(self):
        if self.busy:
            messagebox.showinfo('PANOSTITCH', 'Wait for the current task to finish (or cancel it) first.'); return
        p = filedialog.asksaveasfilename(filetypes=[('PANOSTITCH project', '*.pstitch')], defaultextension='.pstitch',
                                         initialfile=self.settings.get('project_name', 'panorama.pstitch'),
                                         initialdir=self.settings.get('project_dir'), title='Save project')
        if not p: return
        if not p.lower().endswith('.pstitch'): p += '.pstitch'
        pixels = False
        if any(r.layout is not None for r in self.results):
            ans = messagebox.askyesnocancel('PANOSTITCH', 'Also store the finished panoramas in the project?\n\n'
                                            'Yes — a much bigger file, but it opens instantly.\n'
                                            'No — small; stitched results are re-rendered from the photos when it opens.')
            if ans is None: return
            pixels = ans
        self._remember_dir('project_dir', p)
        self.settings['project_name'] = os.path.basename(p)
        results, photos, opts = list(self.results), list(self.photo_paths), self._options()
        sel = self.result_list.curselection()

        def work(job):
            meta = {'format': 'PANOSTITCH project', 'version': 1, 'photos': photos, 'options': opts,
                    'selected': sel[0] if sel else None, 'results': []}
            arrays = {}
            for i, r in enumerate(results):
                job.check()
                job.progress(0.1 + 0.6 * i / max(1, len(results)), f'collecting {r.title}')
                ent = {'title': r.title, 'mode': r.mode, 'layout': None, 'pixels': False,
                       'hz_lines': [[[float(x), float(y)] for x, y in l] for l in (r.hz_lines or [])] or None}
                if r.layout is not None:
                    lm, la = E.layout_state(r.layout)
                    ent['layout'] = lm
                    arrays.update({f'r{i}_{k}': v for k, v in la.items()})
                if r.layout is None or pixels:
                    ent['pixels'] = True
                    arrays[f'r{i}_rgb'] = r.rgb
                    if r.alpha is not None: arrays[f'r{i}_alpha'] = r.alpha
                meta['results'].append(ent)
            meta['project_dir'] = os.path.dirname(os.path.abspath(p))
            job.progress(0.75, f'writing {os.path.basename(p)}')
            tmp = p + '.saving'
            with open(tmp, 'wb') as fh:                       # written aside, then swapped in: a failed save
                np.savez(fh, meta=np.frombuffer(json.dumps(meta).encode('utf-8'), np.uint8), **arrays)
            os.replace(tmp, p)                                # never leaves a half-written project behind
            return os.path.getsize(p)
        self.run_task('Saving project', work, lambda n: self.log('ok', f'project saved: {p} ({n / 1e6:.0f} MB, '
                                                                       f'{len(results)} result{"s" if len(results) != 1 else ""}, {len(photos)} photos)'))

    def _resolve_paths(self, paths, proj_dir, saved_dir):
        """Maps saved photo paths to existing files: as saved, else at the same place relative to the
        project, else by name next to the project, else by name in a folder the user points to."""
        out, missing = {}, []
        for q in paths:
            cands = [q]
            if saved_dir:
                try: cands.append(os.path.join(proj_dir, os.path.relpath(q, saved_dir)))
                except ValueError: pass
            cands.append(os.path.join(proj_dir, os.path.basename(q)))
            hit = next((c for c in cands if os.path.isfile(c)), None)
            if hit: out[q] = os.path.normpath(hit)
            else: missing.append(q)
        while missing:
            names = ', '.join(os.path.basename(m) for m in missing[:4]) + (' …' if len(missing) > 4 else '')
            if not messagebox.askokcancel('PANOSTITCH', f'{len(missing)} photo(s) used by this project can\'t be found ({names}).\n\n'
                                                        'Choose the folder they are in now?'):
                return None
            d = filedialog.askdirectory(title='Folder with the missing photos', initialdir=proj_dir)
            if not d: return None
            still = []
            for q in missing:
                c = os.path.join(d, os.path.basename(q))
                if os.path.isfile(c): out[q] = os.path.normpath(c)
                else: still.append(q)
            missing = still
        return out

    def open_project(self):
        if self.busy:
            messagebox.showinfo('PANOSTITCH', 'Wait for the current task to finish (or cancel it) first.'); return
        p = filedialog.askopenfilename(filetypes=[('PANOSTITCH project', '*.pstitch'), ('All files', '*.*')],
                                       initialdir=self.settings.get('project_dir'), title='Open project')
        if not p: return
        try:
            z = np.load(p, allow_pickle=False)
            meta = json.loads(bytes(z['meta']).decode('utf-8'))
            if meta.get('format') != 'PANOSTITCH project': raise ValueError('not a PANOSTITCH project')
        except Exception as ex:
            messagebox.showerror('PANOSTITCH', f'Could not open {os.path.basename(p)}: {ex}'); return
        if self.results and not messagebox.askyesno('PANOSTITCH', 'Opening a project replaces the current photos and results. Continue?'):
            return
        needed = list(meta['photos'])
        for ent in meta['results']:
            if ent['layout']: needed += ent['layout']['shot_paths']   # stitched results keep their photos' links
        needed = list(dict.fromkeys(needed))
        where = self._resolve_paths(needed, os.path.dirname(os.path.abspath(p)), meta.get('project_dir'))
        if where is None: return
        self._remember_dir('project_dir', p)
        self.settings['project_name'] = os.path.basename(p)

        def work(job):
            shots = {}
            def shot(q):
                f = where[q]
                if f not in shots:
                    job.check()
                    job.progress(0.05, f'loading {os.path.basename(f)}')
                    shots[f] = self.shot_cache.get(f) or E.Shot(f)
                return shots[f]
            out = []
            n = len(meta['results'])
            for i, ent in enumerate(meta['results']):
                job.check()
                L = None
                if ent['layout']:
                    lm = ent['layout']
                    pre = f'r{i}_'
                    arr = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre) and k[len(pre):] not in ('rgb', 'alpha')}
                    L = E.layout_from_state(lm, arr, [shot(q) for q in lm['shot_paths']])
                if ent['pixels']:
                    rgb = z[f'r{i}_rgb']
                    alpha = z[f'r{i}_alpha'] if f'r{i}_alpha' in z.files else None
                else:
                    f0, f1 = 0.1 + 0.85 * i / n, 0.1 + 0.85 * (i + 1) / n
                    job.log('info', f'project: re-rendering {ent["title"]} from the photos')
                    rgb, alpha = L.render(job, f0, f1)
                res = Result(ent['title'], rgb, alpha, layout=L, mode=ent['mode'])
                if ent.get('hz_lines'): res.hz_lines = [[tuple(p) for p in l] for l in ent['hz_lines']]
                out.append(res)
            z.close()
            for f, s in shots.items(): self.shot_cache.setdefault(f, s)
            return [where.get(q, q) for q in meta['photos']], meta.get('options', {}), out, meta.get('selected')

        def done(res):
            photos, opts, results, sel = res
            self._apply_options(opts)
            self._set_photos(photos)
            self.results = []
            self.result_list.delete(0, 'end')
            for r in results:
                self._add_result(r, f' · {r.mode} · {len(r.layout.nodes)} photos' if r.layout else (f' · {r.mode}' if r.mode else ''))
            if results:
                self.result_list.selection_set(sel if sel is not None and sel < len(results) else 0)
            self._show_selected_result()
            self.nb.select(0)
            self.log('ok', f'project opened: {p} — {len(photos)} photos, {len(results)} result{"s" if len(results) != 1 else ""}')
        self.run_task('Opening project', work, done)

    # ─────────────────────────────── saving
    def save_result(self):
        r = self.selected_result()
        if r: self.save_image_dialog(r)

    def save_image_dialog(self, r):
        if r is None: return
        base = r.title.lower().replace(' ', '_')
        p = filedialog.asksaveasfilename(filetypes=SAVE_TYPES, defaultextension='.tif', initialfile=base + '.tif',
                                         initialdir=self.settings.get('save_dir'), title='Save panorama')
        if not p: return
        if os.path.splitext(p)[1].lower() not in ('.tif', '.tiff', '.png', '.jpg', '.jpeg'):
            p += '.tif'
        self._remember_dir('save_dir', p)

        def work(job):
            job.progress(0.1, f'saving {os.path.basename(p)}')
            E.save_image(p, r.rgb, r.alpha)
            return p
        self.run_task('Saving', work, lambda path: self.log('ok', f'saved {path} ({os.path.getsize(path) / 1e6:.0f} MB)'))

    def save_layers(self, r=None):
        """Saves one image per photo, the panorama's size and aligned with it (also for a straightened
        version, from the Horizon tab): "Blended colours" (default) matches each to the finished blend's
        brightness and colour; with "seam masks", each also gets a grey mask of where the seams gave it
        the picture."""
        r = r or self.selected_result()
        if not r: return
        if not r.layout:
            messagebox.showinfo('PANOSTITCH', 'Layers need a stitched result — this one has no photos behind it '
                                              '(it was opened from a file, or a photo was added to it).'); return
        d = filedialog.askdirectory(initialdir=self.settings.get('save_dir'), title='Folder for the aligned layers')
        if not d: return
        self._remember_dir('save_dir', d)
        ext = '.tif' if self.layer_fmt.get().startswith('TIFF') else '.png'
        blended = self.layer_kind.get() == LAYER_KINDS[0]
        masks = self.layer_masks.get() and r.layout.seams is not None
        self.settings.update(layer_kind=self.layer_kind.get(), layer_masks=self.layer_masks.get())
        L = r.layout
        names = [self._layer_name(L, n, ext) for n in L.nodes]
        if masks: names += [self._layer_name(L, n, '_mask.png') for n in L.nodes]
        existing = [f for f in names if os.path.exists(os.path.join(d, f))]
        if existing and not messagebox.askyesno('PANOSTITCH', f'{len(existing)} layer file(s) already exist in that folder. Overwrite them?'):
            return
        sigma = max(8.0, 0.004 * max(L.W, L.H))            # "smooth" for colour matching: ~0.4% of the panorama

        def work(job):
            saved = []
            for k, n in enumerate(L.nodes):
                job.check()
                name = self._layer_name(L, n, ext)
                job.progress(k / len(L.nodes), f'layer {k + 1}/{len(L.nodes)}: {name}')
                rgb, alpha = L.render_layer(n, job)
                m = L.seam_mask_layer(n, job) if L.seams is not None else None
                if blended:
                    rgb = E.harmonise_layer(rgb, alpha, r.rgb, r.alpha, sigma, shown=m)
                E.save_image(os.path.join(d, name), rgb, alpha, bit16=True)
                saved.append(name)
                if masks:
                    mname = self._layer_name(L, n, '_mask.png')
                    ok, buf = cv2.imencode('.png', m)
                    buf.tofile(os.path.join(d, mname))
                    saved.append(mname)
                job.log('ok', f'saved layer {name}' + (' and its seam mask' if masks else ''))
            return saved
        kind = 'colours matched to the blended panorama' if blended else 'original colours'
        self.run_task('Saving layers', work, lambda s: self.log(
            'ok', f'{len(L.nodes)} aligned layers ({kind}){" with seam masks" if masks else ""} saved to {d} — '
                  'stack them in your editor; they line up exactly with the panorama'))

    # ─────────────────────────────── Hugin
    def export_hugin(self):
        r = self.selected_result()
        if not r or not r.layout or not getattr(r.layout, 'edges', None): return
        try:
            pts = int(self.cp_per_pair.get())
            if not (pts == 0 or 3 <= pts <= 100000): raise ValueError
        except ValueError:
            messagebox.showerror('PANOSTITCH', 'Points per pair must be 0 (all good matches) or a whole number from 3 up.'); return
        first = r.layout.shots[r.layout.nodes[0]].path
        p = filedialog.asksaveasfilename(filetypes=[('Hugin project', '*.pto')], defaultextension='.pto',
                                         initialfile=r.title.lower().replace(' ', '_') + '.pto',
                                         initialdir=self.settings.get('hugin_dir') or os.path.dirname(first),
                                         title='Export Hugin project')
        if not p: return
        self._remember_dir('hugin_dir', p)
        L = r.layout

        def work(job):
            job.progress(0.2, 'choosing control points and starting positions for Hugin')
            return PH.write_pto(p, L, max_points=pts, log=job.log)

        def done(info):
            self.log('ok', f'Hugin project saved: {p} — {info["images"]} photos, {info["points"]} control points on '
                           f'{info["pairs"]} pairs, lens from {info["focal_source"]} ({info["hfov"]:.1f}° wide). '
                           'In Hugin: open it, optimise (e.g. Positions, View and Barrel), then stitch.')
        self.run_task('Exporting Hugin project', work, done)

    def hugin_detector_help(self):
        prog = VENV_PYTHON
        args = f'"{os.path.join(HERE, "panostitch_cpfind.py")}" --model {self.model.get() or "k2048"} -o %o %s'
        win = tk.Toplevel(self.root)
        win.title('Use PANOSTITCH as Hugin’s control point detector')
        win.configure(bg=C['card'])
        win.transient(self.root)
        fr = ttk.Frame(win, style='Card.TFrame', padding=16)
        fr.pack(fill='both', expand=True)
        ttk.Label(fr, text='In Hugin: File → Preferences → Control Points → Add…, then fill in:', style='Card.TLabel').pack(anchor='w')
        win._vars = []          # Tk forgets a StringVar's text once Python drops the variable
        for label, value in (('Label', 'PANOSTITCH ALIKED+LightGlue'), ('Type', 'All images at once'),
                             ('Program', prog), ('Arguments', args)):
            row = ttk.Frame(fr, style='Card.TFrame')
            row.pack(fill='x', pady=3)
            ttk.Label(row, text=label, style='Card.TLabel', width=10).pack(side='left')
            var = tk.StringVar(value=value)
            win._vars.append(var)
            ttk.Entry(row, textvariable=var, width=90, state='readonly').pack(side='left', fill='x', expand=True)
            ttk.Button(row, text='Copy', width=6,
                       command=lambda v=value: (self.root.clipboard_clear(), self.root.clipboard_append(v))).pack(side='left', padx=(4, 0))
        ttk.Label(fr, text='(Program and Arguments go on the Detector tab, with “One step detector” selected.)\n'
                           'Then pick it in the Photos tab’s “Feature matching” list and press Create control points.\n'
                           'Options for Arguments: --res 2048 (faster), --points N (cap per pair; default: all good matches), --device cuda:1,\n'
                           '--window N (long sequences: only match photos up to N apart), --matcher sift, --no-prune.',
                  style='Muted.TLabel', justify='left').pack(anchor='w', pady=(8, 0))
        ttk.Button(fr, text='Close', command=win.destroy).pack(anchor='e', pady=(10, 0))

    @staticmethod
    def _layer_name(L, n, ext):
        k = L.nodes.index(n)
        return f'layer_{k + 1:02d}_{os.path.splitext(L.shots[n].name)[0]}{ext}'

    # ─────────────────────────────── load a panorama file
    def _open_result(self, title, done):
        p = filedialog.askopenfilename(filetypes=IMG_TYPES, initialdir=self.settings.get('open_dir'), title=title)
        if not p: return
        self._remember_dir('open_dir', p)

        def work(job):
            job.progress(0.2, f'loading {os.path.basename(p)}')
            rgb, alpha = E.load_image(p)
            return Result(os.path.splitext(os.path.basename(p))[0], rgb, alpha)
        self.run_task('Loading', work, done)

    def send_to(self, where):
        r = self.selected_result()
        if r is None:
            messagebox.showinfo('PANOSTITCH', 'Select a result on the Stitch tab first.'); return
        if where == 'addp':
            self._set_addp_base(r)
            self.nb.select(1)
        elif where == 'hz':
            self._set_hz_src(r)
            self.nb.select(3)
        else:
            self._set_pole_eq(r)
            self.nb.select(2)

    # ─────────────────────────────── add photo
    def _set_addp_base(self, r):
        self.addp_base = r
        self.addp_base_lbl.configure(text=r.label())
        self.show_preview(self.addp_preview, r)
        self._update_buttons()

    def addp_open_base(self):
        self._open_result('Open a panorama', self._set_addp_base)

    def addp_choose_photo(self):
        self._choose_into(self.addp_photo)

    def addp_run(self):
        base, p = self.addp_base, self.addp_photo.get().strip()
        if base is None:
            messagebox.showinfo('PANOSTITCH', 'Choose a panorama first.'); return
        if not p or not os.path.isfile(p):
            messagebox.showinfo('PANOSTITCH', 'Choose the photo to add.'); return
        mode = self.addp_mode.get()
        matcher = dict(ADDP_MATCHERS).get(self.addp_matcher.get(), 'aliked')
        blend = dict(ADDP_BLENDS).get(self.addp_blend.get(), 'multiband')
        model_path, device = self.models.get(self.model.get()), self.device.get() or 'auto'
        if matcher == 'aliked' and not model_path:
            messagebox.showerror('PANOSTITCH', 'ALIKED needs a model file — choose the models folder on the Stitch tab, or use SIFT.'); return
        self.settings.update(addp_matcher=self.addp_matcher.get(), addp_blend=self.addp_blend.get(), addp_mode=mode)

        def work(job):
            job.progress(0.01, f'loading {os.path.basename(p)}')
            photo, _ = E.load_image(p)
            rgb, alpha, info = E.add_photo(base.rgb, base.alpha, photo, mode, job, os.path.basename(p),
                                           matcher=matcher, model_path=model_path, device=device, blend=blend)
            job.log('ok', f'{os.path.basename(p)} added ({info["inliers"]} inliers) — now {fmt_mp(*info["size"])}')
            t = base.title if base.title.endswith('(edited)') else base.title + ' (edited)'
            return Result(t, rgb, alpha, mode=base.mode)
        self.run_task('Adding photo', work, self._set_addp_base)

    def addp_keep(self):
        if self.addp_base and self.addp_base not in self.results:
            self._add_result(self.addp_base)
            self.log('info', f'{self.addp_base.title} added to the results list')

    # ─────────────────────────────── zenith / nadir
    def _set_pole_eq(self, r):
        w, h = r.size
        self.pole_eq = r
        note = '' if abs(w / h - 2) < 0.02 else '  ⚠ not 2:1 — is this equirectangular?'
        self.pole_eq_lbl.configure(text=r.label() + note)
        self.show_preview(self.pole_preview, r)
        self._update_buttons()

    def pole_open_eq(self):
        self._open_result('Open an equirectangular panorama', self._set_pole_eq)

    def _pole_fov(self):
        v = float(self.pole_fov.get())
        if not 30 <= v <= 175: raise ValueError('field of view must be between 30° and 175°')
        return v

    def pole_extract(self):
        r = self.pole_eq
        if r is None:
            messagebox.showinfo('PANOSTITCH', 'Load an equirectangular panorama first.'); return
        try:
            fov = self._pole_fov()
            sz = self.pole_size.get().strip()
            # auto: keep the equator's pixel density (W / 2π per radian) at the patch centre
            size = int(sz) if sz.isdigit() else int(min(16384, max(512, round(r.size[0] / (2 * np.pi) * 2 * np.tan(np.radians(fov) / 2) / 8) * 8)))
        except ValueError as ex:
            messagebox.showerror('PANOSTITCH', str(ex)); return
        d = filedialog.askdirectory(initialdir=self.settings.get('save_dir'), title='Folder for the zenith/nadir patches')
        if not d: return
        self._remember_dir('save_dir', d)
        self.settings.update(pole_fov=fov, pole_size=sz)
        base = r.title.lower().replace(' ', '_')
        names = {True: f'{base}_zenith_fov{fov:g}.tif', False: f'{base}_nadir_fov{fov:g}.tif'}
        existing = [n for n in names.values() if os.path.exists(os.path.join(d, n))]
        if existing and not messagebox.askyesno('PANOSTITCH', 'Overwrite ' + ', '.join(existing) + '?'):
            return

        def work(job):
            out = []
            for k, zen in enumerate((True, False)):
                job.check()
                job.progress(0.1 + 0.45 * k, f'extracting {"zenith" if zen else "nadir"} ({size}×{size})')
                patch = E.extract_pole(r.rgb, zen, fov, size)
                pth = os.path.join(d, names[zen])
                E.save_image(pth, patch)
                out.append(pth)
            return out

        def done(paths):
            self.pole_zen.set(paths[0]); self.pole_nad.set(paths[1])
            self.log('ok', f'saved {os.path.basename(paths[0])} and {os.path.basename(paths[1])} to {d} '
                           f'({size}×{size}, {fov:g}°) — edit them, then reimport below')
        self.run_task('Extracting poles', work, done)

    def pole_import(self):
        r = self.pole_eq
        if r is None:
            messagebox.showinfo('PANOSTITCH', 'Load an equirectangular panorama first.'); return
        zp, np_ = self.pole_zen.get().strip(), self.pole_nad.get().strip()
        if not zp and not np_:
            messagebox.showinfo('PANOSTITCH', 'Choose an edited zenith and/or nadir patch.'); return
        for p in (zp, np_):
            if p and not os.path.isfile(p):
                messagebox.showerror('PANOSTITCH', f'File not found: {p}'); return
        try:
            fov = self._pole_fov()
        except ValueError as ex:
            messagebox.showerror('PANOSTITCH', str(ex)); return

        def work(job):
            pats = []
            for p in (zp, np_):
                if not p: pats.append(None); continue
                job.progress(0.1 + 0.2 * len(pats), f'loading {os.path.basename(p)}')
                img, _ = E.load_image(p)
                if abs(img.shape[0] - img.shape[1]) > 2:
                    raise RuntimeError(f'{os.path.basename(p)} is not square — patches must keep their size')
                pats.append(img)
            job.progress(0.6, 'compositing into the panorama')
            rgb = E.composite_poles(r.rgb, pats[0], pats[1], fov)
            t = r.title if r.title.endswith('(poles)') else r.title + ' (poles)'
            return Result(t, rgb, r.alpha, mode=r.mode)

        def done(res):
            self._set_pole_eq(res)
            self.log('ok', 'patches composited — save the result, or keep it in the results list')
        self.run_task('Compositing poles', work, done)

    def pole_keep(self):
        if self.pole_eq and self.pole_eq not in self.results:
            self._add_result(self.pole_eq)
            self.log('info', f'{self.pole_eq.title} added to the results list')

    # ─────────────────────────────── horizon
    def _set_hz_src(self, r):
        """A new picture for the Horizon tab: starts a fresh straightening history with it."""
        self.hz_states = [{'result': r, 'lines': [list(l) for l in (r.hz_lines or [])], 'label': 'Original'}]
        self.hz_pos = -1
        self.hz_zoom, self.hz_center = 1.0, None
        self._hz_show_state(0)

    # ── straightening history: every straightened version, with the lines traced on it
    HZ_MAX_STATES = 13                  # the original + the last 12 straightenings (each holds a panorama in RAM)

    def _hz_show_state(self, i):
        """Shows history state i: its picture and the lines that were traced on it."""
        if 0 <= self.hz_pos < len(self.hz_states):              # remember the lines on the state we leave
            self.hz_states[self.hz_pos]['lines'] = [list(l) for l in self.hz_lines]
        self.hz_pos = i
        st = self.hz_states[i]
        r = self.hz_src = st['result']
        self.hz_lbl.configure(text=r.label() + ('' if r.layout else '  (from a file: shifted, not re-blended)'))
        self.hz_lines, self.hz_active, self.hz_live, self.hz_history = [list(l) for l in st['lines']], None, [], []
        self._hz_redraw(full=True)
        self._hz_history_ui()
        self._update_buttons()

    def _hz_history_ui(self):
        if not hasattr(self, 'hz_hist_box'): return
        names = [f'{k}. {st["label"]}' if k else st['label'] for k, st in enumerate(self.hz_states)]
        self.hz_hist_box.configure(values=names)
        self.hz_hist.set(names[self.hz_pos])
        self.hz_back_btn.configure(state='normal' if self.hz_pos > 0 else 'disabled')
        self.hz_fwd_btn.configure(state='normal' if self.hz_pos < len(self.hz_states) - 1 else 'disabled')

    def hz_back(self):
        if self.hz_pos > 0 and not self.busy: self._hz_show_state(self.hz_pos - 1)

    def hz_forward(self):
        if self.hz_pos < len(self.hz_states) - 1 and not self.busy: self._hz_show_state(self.hz_pos + 1)

    def _hz_pick_state(self, e=None):
        names = list(self.hz_hist_box.cget('values'))
        if self.hz_hist.get() in names and not self.busy:
            self._hz_show_state(names.index(self.hz_hist.get()))

    def _hz_push_state(self, res, label):
        """A new straightened version: later states (if we'd gone back) are replaced, as with undo."""
        self.hz_states[self.hz_pos]['lines'] = [list(l) for l in self.hz_lines]
        del self.hz_states[self.hz_pos + 1:]
        base = self.hz_states[0]['result'].title
        res.title = f'{base} (level {len(self.hz_states)})'      # each version its own name in the results list
        self.hz_states.append({'result': res, 'lines': [], 'label': label})
        while len(self.hz_states) > self.HZ_MAX_STATES:
            del self.hz_states[1]                               # forget the oldest step, never the original
        self.hz_pos = -1
        self._hz_show_state(len(self.hz_states) - 1)

    # Tracing: any number of separate lines. A click with no line in progress starts a new one — or, if
    # it lands on either end of an existing line, carries that line on from there. While a line is in
    # progress, each click adds a piece to it (magnetic: snapped to the edge; freehand: straight, or
    # drawn by dragging); clicking on another line's end joins the two. Double-click / Enter ends the line.
    @property
    def hz_pts(self):
        """All traced points (every line), for straightening."""
        return [p for line in self.hz_lines for p in line]

    def _hz_snapshot(self):
        self.hz_history.append(([list(l) for l in self.hz_lines], self.hz_active))
        del self.hz_history[:-200]

    def _hz_mode_changed(self):
        self.hz_live = []
        self._hz_redraw()

    def hz_clear(self):
        if self.hz_lines: self._hz_snapshot()
        self.hz_lines, self.hz_active, self.hz_live = [], None, []
        self._hz_redraw(full=True)

    def hz_undo(self):
        if self.hz_history:
            self.hz_lines, self.hz_active = self.hz_history.pop()
        self.hz_live = []
        self._hz_redraw()

    def _hz_near_end(self, p, exclude=None):
        """(line index, 'start'|'end') of a line end within a few screen pixels of p, or None."""
        s = self.hz_view[0]
        best = None
        for i, line in enumerate(self.hz_lines):
            if i == exclude or not line: continue
            for which, q in (('end', line[-1]), ('start', line[0])):
                d = math.hypot((q[0] - p[0]) * s, (q[1] - p[1]) * s)
                if d <= 9 and (best is None or d < best[0]): best = (d, i, which)
        return best and best[1:]

    def _hz_piece(self, a, b):
        """Points from a to b (excluding a): snapped to the edge (magnetic) or a straight piece. The
        magnetic path is found on the full-resolution panorama around the two points, at the scale
        you're viewing (never finer than the image itself) — zoom in for a more precise snap."""
        if self.hz_mode.get() == 'freehand': return [b]
        r = self.hz_src
        H, W = r.rgb.shape[:2]
        s = self.hz_view[0]
        pad_x, pad_y = 20 / s, 60 / s                      # search a band around the two points
        x0, x1 = int(max(0, min(a[0], b[0]) - pad_x)), int(min(W, max(a[0], b[0]) + pad_x + 1))
        y0, y1 = int(max(0, min(a[1], b[1]) - pad_y)), int(min(H, max(a[1], b[1]) + pad_y + 1))
        if x1 - x0 < 2 or y1 - y0 < 2: return [b]
        k = min(1.0, s, 3000.0 / (x1 - x0))                # working scale: view scale, capped
        crop = E.to_uint8(r.rgb[y0:y1, x0:x1])
        if k < 0.999:
            crop = cv2.resize(crop, (max(2, round((x1 - x0) * k)), max(2, round((y1 - y0) * k))), interpolation=cv2.INTER_AREA)
        edge = E.edge_strength(crop)
        xs, ys = E.magnetic_path(edge, (a[0] - x0) * k, (a[1] - y0) * k, (b[0] - x0) * k, (b[1] - y0) * k)
        return [(x / k + x0, y / k + y0) for x, y in zip(xs[1:], ys[1:])] or [b]

    # ── zoom and pan
    def hz_zoom_by(self, f, about=None):
        """Zoom by factor f about a canvas point (default: the canvas centre), 1 = fit the window."""
        if not self.hz_view: return
        cv = self.hz_canvas
        cw, ch = max(10, cv.winfo_width()), max(10, cv.winfo_height())
        s, ox, oy = self.hz_view[:3]
        ex, ey = about if about else (cw / 2, ch / 2)
        p = ((ex - ox) / s, (ey - oy) / s)                 # the image point under the pointer stays put
        z = float(np.clip(self.hz_zoom * f, 1.0, 8.0 / self.hz_fit))
        s2 = self.hz_fit * z
        self.hz_zoom = z
        self.hz_center = ((cw / 2 - (ex - p[0] * s2)) / s2, (ch / 2 - (ey - p[1] * s2)) / s2)
        self._hz_redraw()

    def hz_zoom_set(self, z):
        if not self.hz_view: return
        self.hz_zoom_by(z / self.hz_zoom)

    def _hz_wheel(self, e, delta=None):
        d = delta if delta is not None else e.delta
        self.hz_zoom_by(1.25 if d > 0 else 0.8, (e.x, e.y))

    def _hz_pan_start(self, e):
        self._hz_pan = (e.x, e.y, self.hz_center)

    def _hz_pan_move(self, e):
        if not getattr(self, '_hz_pan', None) or not self.hz_view or self.hz_center is None: return
        x0, y0, (cx, cy) = self._hz_pan
        s = self.hz_view[0]
        self.hz_center = (cx - (e.x - x0) / s, cy - (e.y - y0) / s)
        self._hz_later(self._hz_redraw)

    def _hz_later(self, fn):
        """Run fn once the app is idle — a burst of mouse events then costs one redraw, not one each,
        so the view never falls behind the mouse."""
        if not getattr(self, '_hz_pending', None):
            def go():
                self._hz_pending = None
                fn()
            self._hz_pending = self.root.after_idle(go)

    def _hz_to_img(self, e):
        s, ox, oy, tw, th = self.hz_view
        return ((e.x - ox) / s, (e.y - oy) / s)

    def _hz_inside(self, e):
        if not self.hz_view: return False
        s, ox, oy, tw, th = self.hz_view
        return ox <= e.x < ox + tw and oy <= e.y < oy + th

    def _hz_press(self, e):
        if not self._hz_inside(e) or self.busy: return
        p = self._hz_to_img(e)
        self._hz_snapshot()
        if self.hz_active is None:
            near = self._hz_near_end(p)
            if near:                                    # carry an existing line on from this end
                i, which = near
                if which == 'start': self.hz_lines[i].reverse()
                self.hz_active = i
            else:                                       # a new, separate line
                self.hz_lines.append([p])
                self.hz_active = len(self.hz_lines) - 1
        else:
            line = self.hz_lines[self.hz_active]
            near = self._hz_near_end(p, exclude=self.hz_active)
            if near:                                    # reached another line's end: join the two
                j, which = near
                other = self.hz_lines[j] if which == 'start' else self.hz_lines[j][::-1]
                line += self._hz_piece(line[-1], other[0]) + other[1:]
                del self.hz_lines[j]
                if j < self.hz_active: self.hz_active -= 1
            else:
                line += self._hz_piece(line[-1], p)
        self.hz_live = []
        self._hz_redraw()

    def _hz_drag(self, e):
        if self.hz_mode.get() != 'freehand' or self.hz_active is None or not self._hz_inside(e): return
        line = self.hz_lines[self.hz_active]
        p = self._hz_to_img(e)
        s = self.hz_view[0]
        if abs(p[0] - line[-1][0]) * s >= 2 or abs(p[1] - line[-1][1]) * s >= 2:
            line.append(p)
            self._hz_redraw()

    def _hz_motion(self, e):
        if self.hz_active is None or not self._hz_inside(e): return
        self._hz_mouse = self._hz_to_img(e)

        def update():                                   # only for the latest mouse position
            if self.hz_active is None: return
            a, b = self.hz_lines[self.hz_active][-1], self._hz_mouse
            self.hz_live = [a, b] if self.hz_mode.get() == 'freehand' else [a] + self._hz_piece(a, b)
            self._hz_redraw()
        self._hz_later(update)

    def _hz_finish(self):
        """Ends the line in progress (double-click / Enter): the next click starts a new one."""
        if self.hz_active is not None:
            self._hz_snapshot()
            self.hz_active = None
        self.hz_live = []
        self._hz_redraw()

    def _hz_samples(self):
        """The traced lines as one height per image column: each line is interpolated along its own
        x range and, where lines overlap, their heights are averaged (no zig-zag between them)."""
        r = self.hz_src
        W = r.rgb.shape[1]
        acc, cnt = np.zeros(W), np.zeros(W)
        for line in self.hz_lines:
            if len(line) < 2: continue
            xs = np.array([p[0] for p in line]); ys = np.array([p[1] for p in line])
            o = np.argsort(xs, kind='stable')
            xs, ys = xs[o], ys[o]
            c0, c1 = int(math.ceil(max(0, xs[0]))), int(math.floor(min(W - 1, xs[-1])))
            if c1 < c0: continue
            cols = np.arange(c0, c1 + 1)
            acc[cols] += np.interp(cols, xs, ys); cnt[cols] += 1
        have = cnt > 0
        cols = np.nonzero(have)[0]
        return cols.astype(np.float64), acc[have] / cnt[have]

    def _hz_raw(self):
        """Every traced line, one point per image column along it, NOT averaged where lines overlap —
        for fixing steps, where each point must count for the photo it falls in (overshoots at a step
        are then recognised and left out by the engine)."""
        r = self.hz_src
        W = r.rgb.shape[1]
        X, Y = [], []
        for line in self.hz_lines:
            if len(line) < 2: continue
            xs = np.array([p[0] for p in line]); ys = np.array([p[1] for p in line])
            o = np.argsort(xs, kind='stable')
            xs, ys = xs[o], ys[o]
            c0, c1 = int(math.ceil(max(0, xs[0]))), int(math.floor(min(W - 1, xs[-1])))
            if c1 < c0: continue
            cols = np.arange(c0, c1 + 1)
            X.append(cols.astype(np.float64)); Y.append(np.interp(cols, xs, ys))
        if not X: return np.zeros(0), np.zeros(0)
        X, Y = np.concatenate(X), np.concatenate(Y)
        # A line that overshoots a step carries on at its own height where the picture shows the horizon
        # elsewhere: keep only points with a horizon edge within a few pixels (the magnetic tool's edge map)
        if getattr(self, 'hz_view', None):
            s = self.hz_fit                                 # the edge map is of the whole picture, fitted
            edge = self._hz_edge()
            eh, ew = edge.shape
            px = np.clip((X * s).astype(int), 0, ew - 1)
            py = (Y * s).astype(int)
            near = np.max(np.stack([edge[np.clip(py + d, 0, eh - 1), px] for d in range(-3, 4)]), axis=0)
            on_edge = near >= 0.25
            if on_edge.mean() >= 0.3:                       # a very faint horizon: keep everything
                X, Y = X[on_edge], Y[on_edge]
        return X, Y

    def _hz_edge(self):
        """Edge map of the whole picture as fitted to the window (for checking traced points)."""
        key = (id(self.hz_src), self._hz_small.shape)
        if getattr(self, '_hz_edge_key', None) != key:
            self._hz_edge_map = E.edge_strength(self._hz_small)
            self._hz_edge_key = key
        return self._hz_edge_map

    @staticmethod
    def _photo(img):
        """RGB uint8 -> Tk image, as raw PPM (no compression: ~4x faster than PNG — smooth panning)."""
        img = np.ascontiguousarray(img)
        return tk.PhotoImage(data=b'P6 %d %d 255\n' % (img.shape[1], img.shape[0]) + img.tobytes(), format='PPM')

    def _hz_redraw(self, full=False):
        cv = self.hz_canvas
        r = self.hz_src
        cw, ch = max(10, cv.winfo_width()), max(10, cv.winfo_height())
        if r is None:
            cv.delete('all')
            cv.create_text(cw / 2, ch / 2, text='Use a stitched result or open a panorama', fill=C['muted'])
            self.hz_view = None
            return
        h, w = r.rgb.shape[:2]
        fs = min((cw - 8) / w, (ch - 8) / h)                   # "fit" scale
        self.hz_fit = fs
        ftw, fth = max(1, int(w * fs)), max(1, int(h * fs))
        fkey = (id(r), ftw, fth)
        if full or getattr(self, '_hz_img_key', None) != fkey:   # the fitted picture (also the edge check's)
            step = max(1, int(1 / fs) // 2)
            small = cv2.resize(E.to_uint8(r.rgb[::step, ::step]), (ftw, fth), interpolation=cv2.INTER_AREA)
            if r.alpha is not None:
                a = cv2.resize(r.alpha[::step, ::step], (ftw, fth), interpolation=cv2.INTER_NEAREST) > 0
                small = np.where(a[..., None], small, np.uint8(40))
            self._hz_small = small
            self._hz_photo = self._photo(small)
            self._hz_img_key = fkey
        z = max(1.0, getattr(self, 'hz_zoom', 1.0))
        s = fs * z
        tw, th = w * s, h * s
        # where the view is centred (image pixels), kept so the picture can't be dragged away
        cx, cy = self.hz_center if (z > 1.0 and self.hz_center) else (w / 2, h / 2)
        cx = w / 2 if tw <= cw else float(np.clip(cx, cw / 2 / s, w - cw / 2 / s))
        cy = h / 2 if th <= ch else float(np.clip(cy, ch / 2 / s, h - ch / 2 / s))
        self.hz_center = (cx, cy)
        ox, oy = round(cw / 2 - cx * s), round(ch / 2 - cy * s)
        self.hz_view = (s, ox, oy, tw, th)
        cv.delete('all')
        if z <= 1.0:
            cv.create_image(ox, oy, image=self._hz_photo, anchor='nw')
        else:                                                    # only the visible part, from full resolution
            ix0, iy0 = max(0, int((-ox) / s)), max(0, int((-oy) / s))
            ix1, iy1 = min(w, int(math.ceil((cw - ox) / s)) + 1), min(h, int(math.ceil((ch - oy) / s)) + 1)
            vkey = (id(r), round(s, 6), ix0, iy0, ix1, iy1)
            if getattr(self, '_hz_view_key', None) != vkey and ix1 > ix0 and iy1 > iy0:
                if getattr(self, '_hz_u8_key', None) != id(r):
                    # one 8-bit display copy per picture (uncovered parts already grey): each view is
                    # then just a slice and a resize, quick enough to pan smoothly
                    u8 = E.to_uint8(r.rgb)
                    if r.alpha is not None:
                        u8 = u8.copy() if u8 is r.rgb else u8
                        u8[r.alpha == 0] = 40
                    self._hz_u8, self._hz_u8_key = u8, id(r)
                step = max(1, int(1 / s) // 2)
                crop = self._hz_u8[iy0:iy1:step, ix0:ix1:step]
                size = (max(1, round((ix1 - ix0) * s)), max(1, round((iy1 - iy0) * s)))
                interp = cv2.INTER_AREA if s < 1 else (cv2.INTER_NEAREST if s >= 3 else cv2.INTER_LINEAR)
                self._hz_vphoto = self._photo(cv2.resize(crop, size, interpolation=interp))
                self._hz_view_key = vkey
            if getattr(self, '_hz_view_key', None) == vkey:
                cv.create_image(ox + ix0 * s, oy + iy0 * s, image=self._hz_vphoto, anchor='nw')
        if hasattr(self, 'hz_zoom_lbl'):
            self.hz_zoom_lbl.configure(text=f'{s * 100:.0f}% of full size' + ('' if z > 1.0 else ' (fit)'))
        P = lambda q: (ox + q[0] * s, oy + q[1] * s)
        for i, line in enumerate(self.hz_lines):
            if len(line) >= 2:
                cv.create_line(*[v for q in line for v in P(q)], fill=C['accent'], width=2)
            ends = [line[0]] if len(line) == 1 else [line[0], line[-1]]
            for k, q in enumerate(ends):                # line ends: click one to carry the line on from there
                x, y = P(q)
                live_end = i == self.hz_active and q is line[-1]
                cv.create_oval(x - 4, y - 4, x + 4, y + 4, outline='#fff', width=1, fill=C['warn'] if live_end else '')
        if len(self.hz_live) >= 2:
            cv.create_line(*[v for q in self.hz_live for v in P(q)], fill=C['warn'], width=1, dash=(4, 3))
        cv.create_text(10, ch - 6, anchor='sw', text=r.label(), fill=C['text'], font=(UI_FONT, 9))

    def hz_run(self):
        r = self.hz_src
        if r is None:
            messagebox.showinfo('PANOSTITCH', 'Choose a panorama first.'); return
        xs, ys = self._hz_samples()
        if len(xs) < 5:
            messagebox.showinfo('PANOSTITCH', 'Trace the horizon across the panorama first.'); return
        smooth = dict(HZ_DETAIL).get(self.hz_detail.get(), 0.05)
        method = dict(HZ_METHOD).get(self.hz_method.get(), 'rotate')
        self.settings.update(hz_detail=self.hz_detail.get(), hz_method=self.hz_method.get())
        title = r.title if r.title.endswith('(level)') else r.title + ' (level)'
        raw = self._hz_raw()

        def report(info):
            job_log = self.q.put
            job_log(('log', 'info', f'horizon: camera tilted back by {info["tilt_deg"]:.2f}°; the traced line sits within '
                                    f'±{info["spread_deg"]:.2f}° (≈{info["spread_px"]:.0f}px) of level afterwards — what\'s left is '
                                    f'hills or wiggles ("Straighten the line" can take those out)'))

        def work(job):
            if method == 'photos':
                if r.layout is None:
                    raise RuntimeError('"Fix steps" needs a stitched result (it turns the photos) — this one was opened '
                                       'from a file; use "Straighten the line" with "Every bump (exact)" instead')
                L = r.layout
                rx, ry = raw
                yu = (L.H - 1 - ry) if L.flip else ry
                L2, info = L.level_photos(rx, yu, job)
                pairs = ', '.join(f'{b:.0f}→{a:.0f}px' for b, a in zip(info['steps_before_px'], info['steps_after_px']))
                turned = ', '.join(f'{k} {v:.2f}°' for k, v in info['turns_deg'].items() if v >= 0.01) or 'none'
                self.q.put(('log', 'info', f'fix steps: {info["boundaries"]} photo boundaries on the line, steps {pairs}; '
                                           f'photos turned: {turned}'))
                rgb, a = L2.render(job, 0.3, 1.0)
                return Result(title, rgb, a, layout=L2, mode=r.mode)
            if method == 'rotate':
                if r.layout is not None:
                    L = r.layout
                    yu = (L.H - 1 - ys) if L.flip else ys
                    R, info = L.level_rotation(xs, yu)
                    report(info)
                    job.progress(0.05, 'levelling: re-framing and finding seams again' if L.seams else 'levelling: re-framing')
                    L2 = L.rotated(R, job)
                    rgb, a = L2.render(job, 0.3, 1.0)
                    return Result(title, rgb, a, layout=L2, mode=r.mode)
                job.progress(0.2, 'rotating the sphere')
                rgb, a, info = E.level_equirect_image(r.rgb, r.alpha, xs, ys)
                report(info)
                return Result(title, rgb, a, mode='equirectangular')
            if r.layout is not None:
                L = r.layout
                yu = (L.H - 1 - ys) if L.flip else ys             # displayed rows -> render rows
                L2 = L.leveled(xs, yu, smooth)
                job.log('info', f'horizon: columns moved by up to {np.abs(L2.hz["d"]).max():.0f}px; canvas {L.W}×{L2.H}')
                rgb, a = L2.render(job, 0.05, 1.0)
                return Result(title, rgb, a, layout=L2, mode=r.mode)
            job.progress(0.3, 'shifting columns')
            rgb, a = E.level_image(r.rgb, r.alpha, xs, ys, smooth)
            return Result(title, rgb, a, mode=r.mode)

        label = self.hz_method.get().split(' (')[0] + ('' if method != 'columns' else f' — {self.hz_detail.get().split(" (")[0].lower()}')

        def done(res):
            self._hz_push_state(res, label)
            self.log('ok', 'horizon straightened — trace again to refine; "Back" returns to the version before')
        self.run_task('Straightening the horizon', work, done)

    def hz_export_hugin(self):
        """Hugin project for the result shown, with the traced horizon as horizontal-line control points."""
        r = self.hz_src
        if r is None or r.layout is None or not getattr(r.layout, 'edges', None):
            messagebox.showinfo('PANOSTITCH', 'This needs a stitched result (not one opened from a file).'); return
        xs, ys = self._hz_raw()
        if len(xs) < 5:
            messagebox.showinfo('PANOSTITCH', 'Trace the horizon first.'); return
        try:
            pts = int(self.cp_per_pair.get())
        except ValueError:
            pts = PH.DEFAULT_POINTS
        L = r.layout
        yu = (L.H - 1 - ys) if L.flip else ys
        first = L.shots[L.nodes[0]].path
        p = filedialog.asksaveasfilename(filetypes=[('Hugin project', '*.pto')], defaultextension='.pto',
                                         initialfile=r.title.lower().replace(' ', '_') + '_horizon.pto',
                                         initialdir=self.settings.get('hugin_dir') or os.path.dirname(first),
                                         title='Export Hugin project with the horizon')
        if not p: return
        self._remember_dir('hugin_dir', p)

        def work(job):
            job.progress(0.2, 'choosing control points and starting positions for Hugin')
            return PH.write_pto(p, L, max_points=pts, log=job.log, horizon=(xs, yu))

        def done(info):
            self.log('ok', f'Hugin project saved: {p} — {info["points"]} control points plus {info["horizon_points"]} '
                           'horizontal-line points along your horizon. In Hugin, optimise (Positions) and the step closes.')
        self.run_task('Exporting Hugin project', work, done)

    def hz_keep(self):
        """Keeps the version shown in the results list — with its traced lines, which come back when it's
        sent to the Horizon tab again (and are saved with projects)."""
        r = self.hz_src
        if not r: return
        r.hz_lines = [list(l) for l in self.hz_lines]
        if r not in self.results:
            self._add_result(r, f' · {r.mode}' if r.mode else '')
            self.log('info', f'{r.title} added to the results list')
        else:
            self.log('info', f'{r.title}: its traced lines are remembered')

    # ─────────────────────────────── shutdown
    def on_close(self):
        if self.busy and not messagebox.askyesno('PANOSTITCH', 'Something is still running. Quit anyway?'):
            return
        if self.job: self.job.cancel_event.set()
        self._save_settings()
        self.root.destroy()


def main():
    import argparse
    ap = argparse.ArgumentParser(description='PANOSTITCH desktop')
    ap.add_argument('files', nargs='*', help='photos to stitch')
    ap.add_argument('--list', help='a text file naming more photos, one per line (e.g. from Photo Library)')
    args = ap.parse_args()
    files = list(args.files)
    if args.list:
        with open(args.list, encoding='utf-8') as f:
            files += [ln.strip() for ln in f if ln.strip()]
    files = [os.path.normpath(p) for p in files if os.path.isfile(p)]
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)   # crisp text on scaled displays
    except Exception:
        pass
    root = tk.Tk()
    app = App(root)
    if files:
        def load():
            app._set_photos(app.photo_paths + files)
            app.log('info', f'{len(files)} photo{"s" if len(files) != 1 else ""} from the command line')
        root.after(300, load)
    root.mainloop()


if __name__ == '__main__':
    main()
