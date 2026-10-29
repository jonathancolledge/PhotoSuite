# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Photo Library desktop — a catalog of photos that stay where they are on disk.

Folders are added, never moved: the app only reads the photos (EXIF, thumbnails, AI) and keeps what it learns in
its own SQLite catalog, so it cannot change or delete a photo. Browse by the folder tree as it is on disk or by
the date each photo was taken; filter by rating, tags, people and any EXIF value; save filters as virtual folders;
rate and tag many photos at once; send a selection to RAW Demosaic (open, or batch with a settings file) or to
PANOSTITCH.

As in the RAW Demosaic and PANOSTITCH desktop apps, heavy work runs on one worker thread that talks to the UI only
through a queue the Tk loop drains every 50 ms. Grid thumbnails are decoded on a few reader threads the same way,
and the grid draws only the cells on screen, so a library of any size scrolls smoothly.
"""
import calendar
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
from collections import OrderedDict, defaultdict
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image, ImageTk

import photolibrary_ai as AI
import photolibrary_catalog as L

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = os.name == 'nt'
UI_FONT, MONO_FONT = ('Segoe UI', 'Consolas') if WINDOWS else ('DejaVu Sans', 'DejaVu Sans Mono')
SETTINGS_PATH = os.path.join(HERE, 'photolibrary_settings.json' if WINDOWS else 'photolibrary_settings_linux.json')
APP = 'Photo Library'
DEFAULT_PANOSTITCH = os.path.normpath(os.path.join(L.APP_ROOT, '..', 'panoramamaker', 'desktop'))
PANO_EXTS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.jfif')
CTRL, SHIFT = 0x4, 0x1

C = dict(bg='#121417', card='#1d2025', card2='#252930', text='#e2e5ea', muted='#9aa3ad', border='#353b44',
         input='#15181c', accent='#ff6e55', accent_hover='#ff8670', ok='#a5d6a7', err='#ef9a9a', info='#80deea',
         warn='#ffd98a', track='#2c3138', checker='#262a30', star_off='#3d434c')

ZOOMS = [96, 128, 160, 200, 256, 320, 400, 512, 640, 800, 1024, 0]   # cell width; 0 = one photo filling the view
RATING_FILTER = [('Any', None), ('Unrated', 'unrated'), ('1★ or more', 1), ('2★ or more', 2), ('3★ or more', 3),
                 ('4★ or more', 4), ('5★', 5)]
THUMB_MODES = [('Embedded preview if 1024 px or more, else RCD', 1024), ('Embedded preview if 512 px or more, else RCD', 512),
               ('Any embedded preview, else RCD', 1), ('Always RCD demosaic', None)]
GPS_FILTER = [('Any', None), ('With GPS', True), ('Without GPS', False)]
KIND_FILTER = [('Any', None), ('RAW (any)', 'raw'), ('Not RAW', 'image')]   # + each catalogued file type
SEARCH_MODES = ['Keywords', 'Meaning (AI)']


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
    if WINDOWS: os.startfile(d)
    else: subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', d])


def show_in_explorer(path):
    if WINDOWS: subprocess.Popen(['explorer', '/select,', os.path.normpath(path)])
    else: open_folder(os.path.dirname(path))


def key_of(pairs, label, default=None):
    return next((k for l, k in pairs if l == label), default)


def label_of(pairs, key):
    return next((l for l, k in pairs if k == key), pairs[0][0])


def num_or_none(s, parse=float):
    s = str(s).strip()
    if not s: return None
    try:
        return parse(s)
    except (ValueError, ZeroDivisionError):
        return None


def fmt_size(n):
    return f'{n / 2 ** 20:.1f} MB' if n >= 2 ** 20 else f'{n / 1024:.0f} KB'


# ───────────────────────────────────────────── thumbnails for the grids
class ThumbLoader:
    """Decodes grid thumbnails on reader threads, newest request first; results go to the UI queue."""
    def __init__(self, cat, out_q, workers=3):
        self.cat, self.q = cat, out_q
        self.todo = OrderedDict()
        self.cv = threading.Condition()
        for _ in range(workers):
            threading.Thread(target=self._run, daemon=True).start()

    def request(self, key, job):
        with self.cv:
            if key in self.todo: self.todo.move_to_end(key)
            else: self.todo[key] = job
            while len(self.todo) > 600: self.todo.popitem(last=False)
            self.cv.notify()

    def forget(self, prefix, keep):
        """Drop queued requests of one grid that are no longer on screen."""
        with self.cv:
            for k in [k for k in self.todo if k[0] == prefix and k not in keep]:
                del self.todo[k]

    def _run(self):
        while True:
            with self.cv:
                while not self.todo: self.cv.wait()
                key, job = self.todo.popitem(last=True)
            try:
                im = self._load(job)
            except Exception:
                im = None
            self.q.put(('thumb_img', key, im))

    def _load(self, job):
        if job[0] == 'photo':
            _, pid, w, h = job
            p = self.cat.best_thumb(pid, max(w, h))
            if not p: return None
            im = Image.open(p)
            im.draft('RGB', (w, h))
            im = im.convert('RGB')
            s = min(w / im.width, h / im.height)
            if abs(s - 1) > 0.01:
                im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS if s < 1 else Image.BICUBIC)
            return im
        _, pid, box, size = job   # a face: square crop around the box, from the 1024 px thumbnail
        p = self.cat.best_thumb(pid, 1024)
        if not p: return None
        im = Image.open(p).convert('RGB')
        W, H = im.size
        x0, y0, x1, y1 = box
        cx, cy, half = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H, max((x1 - x0) * W, (y1 - y0) * H) * 0.7
        return im.crop((round(cx - half), round(cy - half), round(cx + half), round(cy + half))).resize((size, size), Image.LANCZOS)


# ───────────────────────────────────────────── the photo grid
class PhotoGrid:
    """A canvas that draws only the visible cells of a list of photo ids: from many small thumbnails up to one
    photo filling the view. Click selects, Ctrl+click toggles, Shift+click selects a range, dragging draws a
    selection box; arrows move, 0–5 rate, Enter / double-click shows one photo, Ctrl+wheel resizes."""
    GAP = 6
    PREFIX = 'p'

    def __init__(self, app, parent):
        self.app = app
        self.cv = tk.Canvas(parent, bg=C['bg'], highlightthickness=0, takefocus=1)
        self.sb = ttk.Scrollbar(parent, orient='vertical', command=self._yview)
        self.cv.pack(side='left', fill='both', expand=True)
        self.sb.pack(side='left', fill='y')
        self.ids, self.rows, self.pos = [], {}, {}
        self.sel, self.anchor, self.focus = set(), None, None
        self.zoom, self.prev_zoom, self.top = 4, 4, 0
        self.cache = OrderedDict()
        self.failed = set()
        self.empty_text = ''
        self._redraw_due = False
        self._press = self._band = None
        self.f_small, self.f_star, self.f_badge = (UI_FONT, 8), (UI_FONT, 11), (UI_FONT, 7, 'bold')
        from tkinter import font as tkfont
        self.star_w = tkfont.Font(font=self.f_star).measure('★') + 1
        self.char_w = tkfont.Font(font=self.f_small).measure('_DSC4471.NEF') / 12   # a typical file name's average
        cv = self.cv
        cv.bind('<Configure>', lambda e: self.request_redraw())
        cv.bind('<ButtonPress-1>', self._on_press)
        cv.bind('<B1-Motion>', self._on_motion)
        cv.bind('<ButtonRelease-1>', self._on_release)
        cv.bind('<Double-Button-1>', self._on_double)
        cv.bind('<Button-3>', self._on_menu)
        cv.bind('<Key>', self._on_key)
        cv.bind('<Control-a>', lambda e: self.select_all() or 'break')
        bind_wheel(cv, self._on_wheel)

    # ─── data
    def set_items(self, ids, rows, keep=True):
        self.ids, self.rows = ids, rows
        self.pos = {p: i for i, p in enumerate(ids)}
        if keep:
            self.sel &= set(self.pos)
        else:
            self.sel, self.top = set(), 0
        fpid = getattr(self, '_focus_pid', None)
        self.focus = self.pos.get(fpid) if fpid is not None else None
        self.anchor = self.focus
        self.request_redraw()
        self._changed()

    def selected_ids(self):
        if len(self.sel) == 1: return list(self.sel)
        return [p for p in self.ids if p in self.sel]

    def select_ids(self, pids):
        self.sel = {p for p in pids if p in self.pos}
        if self.sel:
            i = min(self.pos[p] for p in self.sel)
            self._set_focus(i); self.anchor = i
            self.ensure_visible(i)
        self._changed(); self.request_redraw()

    def select_all(self):
        self.sel = set(self.ids)
        self._changed(); self.request_redraw()

    def invalidate(self, pids):
        pids = set(pids)
        for k in [k for k in self.cache if k[1] in pids]: del self.cache[k]
        self.failed = {k for k in self.failed if k[1] not in pids}
        self.request_redraw()

    def _set_focus(self, i):
        self.focus = i
        self._focus_pid = self.ids[i] if i is not None and 0 <= i < len(self.ids) else None

    def _changed(self):
        self.app.selection_changed()

    # ─── geometry
    def geom(self):
        W, H = max(60, self.cv.winfo_width()), max(60, self.cv.winfo_height())
        g = self.GAP
        z = ZOOMS[self.zoom]
        if z == 0:
            band = int(26 * self.app.ui_scale)
            return W, H, g, 1, W - 2 * g, H - 2 * g, band
        cols = max(1, (W - g) // (z + g))
        cw = (W - g) // cols - g
        band = int(22 * self.app.ui_scale) if cw >= 110 else 0
        return W, H, g, cols, cw, cw + band, band

    def single(self):
        return ZOOMS[self.zoom] == 0

    def _content(self, H, g, cols, ch):
        rows = -(-len(self.ids) // cols)
        return g + rows * (ch + g)

    def ensure_visible(self, i):
        W, H, g, cols, cw, ch, band = self.geom()
        y = g + (i // cols) * (ch + g)
        if self.single() or y < self.top: self.top = y - g
        elif y + ch > self.top + H: self.top = y + ch + g - H
        self.request_redraw()

    def hit(self, x, y):
        """Canvas point -> (index or None, part) where part is ('star', n) on a cell's rating stars."""
        W, H, g, cols, cw, ch, band = self.geom()
        yy = y + self.top - g
        c, r = int((x - g) // (cw + g)), int(yy // (ch + g))
        if yy < 0 or c < 0 or c >= cols: return None, None
        lx, ly = x - g - c * (cw + g), yy - r * (ch + g)
        i = r * cols + c
        if lx > cw or ly > ch or i >= len(self.ids): return None, None
        if band and ly > ch - band and 4 <= lx < 8 + 5 * self.star_w:
            return i, ('star', min(5, max(1, int((lx - 8) // self.star_w) + 1)))
        return i, None

    def in_rect(self, x0, y0, x1, y1):
        """Ids of cells meeting a rectangle in content coordinates."""
        W, H, g, cols, cw, ch, band = self.geom()
        x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
        c0, c1 = max(0, int((x0 - g) // (cw + g))), min(cols - 1, int((x1 - g) // (cw + g)))
        r0, r1 = max(0, int((y0 - g) // (ch + g))), int((y1 - g) // (ch + g))
        out = set()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                i = r * cols + c
                if i < len(self.ids): out.add(self.ids[i])
        return out

    # ─── drawing
    def request_redraw(self):
        if not self._redraw_due:
            self._redraw_due = True
            self.cv.after(10, self.redraw)

    def redraw(self):
        self._redraw_due = False
        cv = self.cv
        cv.delete('all')
        W, H, g, cols, cw, ch, band = self.geom()
        n = len(self.ids)
        content = self._content(H, g, cols, ch)
        self.top = max(0, min(self.top, content - H))
        if not n:
            cv.create_text(W / 2, H / 2, text=self.empty_text, fill=C['muted'], justify='center', font=(UI_FONT, 11))
            self.sb.set(0, 1)
            return
        pitch = ch + g
        want = set()
        for r in range(self.top // pitch, min(-(-n // cols), (self.top + H) // pitch + 1)):
            for c in range(cols):
                i = r * cols + c
                if i >= n: break
                key = self.draw_cell(i, g + c * (cw + g), g + r * pitch - self.top, cw, ch, band)
                if key: want.add(key)
        self.app.loader.forget(self.PREFIX, want)
        while len(self.cache) > 900: self.cache.popitem(last=False)
        if self._band:
            x0, y0, x1, y1 = self._band
            cv.create_rectangle(x0, y0 - self.top, x1, y1 - self.top, outline=C['accent'], dash=(4, 3), width=1)
        self.sb.set(self.top / content, min(1, (self.top + H) / content))

    def cell_job(self, pid, bw, bh):
        row = self.rows.get(pid)
        if not row or row['thumb'] != 1: return None, None
        return (self.PREFIX, pid, bw, bh), ('photo', pid, bw, bh)

    def draw_cell(self, i, x, y, cw, ch, band):
        cv = self.cv
        pid = self.ids[i]
        row = self.rows.get(pid) or {}
        sel = pid in self.sel
        cv.create_rectangle(x, y, x + cw, y + ch, fill=C['card2'] if sel else C['card'], outline=C['accent'] if sel else C['card'], width=2)
        ih = ch - band
        bw, bh = max(8, cw - 8), max(8, ih - 8)
        key, job = self.cell_job(pid, bw, bh)
        ph = self.cache.get(key) if key else None
        if ph is not None:
            self.cache.move_to_end(key)
            cv.create_image(x + cw / 2, y + ih / 2, image=ph)
        else:
            st = row.get('thumb', 0)
            msg = 'no preview' if st == -1 or key in self.failed else '' if key else 'making thumbnail…' if st == 0 else ''
            if key and key not in self.failed: self.app.loader.request(key, job)
            if cw >= 90:
                cv.create_text(x + cw / 2, y + ih / 2, text=(msg + '\n' if msg else '') + str(row.get('name', '')), fill=C['muted'],
                               font=self.f_small, width=cw - 12, justify='center')
        if cw >= 100:
            bx = x + 6
            if row.get('is_raw'):
                cv.create_rectangle(bx, y + 6, bx + 30, y + 19, fill='#000', outline='', stipple='gray50')
                cv.create_text(bx + 15, y + 12, text='RAW', fill=C['info'], font=self.f_badge); bx += 34
            if row.get('missing'):
                cv.create_text(bx, y + 12, anchor='w', text='⚠ missing', fill=C['warn'], font=self.f_badge)
        if band:
            yb = y + ih + band / 2 - 1
            r = row.get('rating') or 0
            for k in range(5):
                cv.create_text(x + 8 + k * self.star_w, yb, anchor='w', text='★', font=self.f_star, fill=C['warn'] if k < r else C['star_off'])
            room = cw - 20 - 5 * self.star_w
            if room > 6 * self.char_w:
                name = str(row.get('name', ''))
                if self.single() and row.get('taken'): name = f'{name}   ·   {row["taken"]}   ·   {i + 1} of {len(self.ids)}'
                fit = int(room / self.char_w)
                cv.create_text(x + cw - 6, yb, anchor='e', text=name if len(name) <= fit else '…' + name[-fit + 1:], fill=C['muted'], font=self.f_small)
        if i == self.focus and self.cv.focus_get() == self.cv:
            cv.create_rectangle(x + 3, y + 3, x + cw - 3, y + ch - 3, outline=C['muted'], dash=(2, 3))
        return key

    def got(self, key, im):
        if im is None:
            self.failed.add(key)
        else:
            self.cache[key] = ImageTk.PhotoImage(im)
        self.request_redraw()

    # ─── scrolling and zoom
    def _yview(self, *a):
        W, H, g, cols, cw, ch, band = self.geom()
        content = self._content(H, g, cols, ch)
        if a[0] == 'moveto':
            self.top = int(float(a[1]) * content)
        elif a[0] == 'scroll':
            step = H - g if a[2] == 'pages' or self.single() else max(40, (ch + g) // 2)
            self.top += int(a[1]) * step
        self.request_redraw()

    def set_zoom(self, z, around=None):
        z = max(0, min(len(ZOOMS) - 1, z))
        if z == self.zoom: return
        i = around if around is not None else self.focus if self.focus is not None else self._first_visible()
        self.zoom = z
        self.app.zoom_changed(z)
        if i is not None: self.ensure_visible(i)
        self.request_redraw()

    def _first_visible(self):
        if not self.ids: return None
        W, H, g, cols, cw, ch, band = self.geom()
        return min(len(self.ids) - 1, (self.top // (ch + g)) * cols)

    def toggle_single(self, i=None):
        i = self.focus if i is None else i
        if self.single():
            self.set_zoom(self.prev_zoom if ZOOMS[self.prev_zoom] else 4, around=i)
        elif i is not None:
            self.prev_zoom = self.zoom
            self.sel = {self.ids[i]}; self._set_focus(i); self.anchor = i
            self._changed()
            self.set_zoom(len(ZOOMS) - 1, around=i)

    def _on_wheel(self, e, d):
        if e.state & CTRL:
            i, _ = self.hit(e.x, e.y)
            self.set_zoom(self.zoom + (1 if d > 0 else -1), around=i)
        elif self.single():
            self.move(-d)
        else:
            W, H, g, cols, cw, ch, band = self.geom()
            self.top -= d * max(60, (ch + g) // 2)
            self.request_redraw()

    # ─── mouse
    def _on_press(self, e):
        self.cv.focus_set()
        i, part = self.hit(e.x, e.y)
        if part and part[0] == 'star':
            pid = self.ids[i]
            cur = (self.rows.get(pid) or {}).get('rating', 0)
            targets = self.selected_ids() if pid in self.sel and len(self.sel) > 1 else [pid]
            self.app.set_rating(targets, 0 if cur == part[1] else part[1])
            self._press = None
            return
        self._press = (e.x, e.y + self.top, i, e.state, set(self.sel))
        self._band = None

    def _on_motion(self, e):
        if not self._press or self.single(): return
        x0, y0, i, state, base = self._press
        if self._band is None and abs(e.x - x0) + abs(e.y + self.top - y0) < 6: return
        H = self.cv.winfo_height()
        if e.y > H - 10: self.top += 25
        elif e.y < 10: self.top = max(0, self.top - 25)
        self._band = (x0, y0, e.x, e.y + self.top)
        hits = self.in_rect(*self._band)
        self.sel = (base | hits) if state & CTRL else hits
        self._changed()
        self.request_redraw()

    def _on_release(self, e):
        if not self._press: return
        x0, y0, i, state, base = self._press
        self._press = None
        if self._band is not None:
            self._band = None
            self.request_redraw()
            return
        self.click(i, state)

    def click(self, i, state=0):
        if i is None:
            if not state & CTRL: self.sel = set()
        elif state & SHIFT and self.anchor is not None:
            a, b = sorted((self.anchor, i))
            rng = set(self.ids[a:b + 1])
            self.sel = (self.sel | rng) if state & CTRL else rng
            self._set_focus(i)
        elif state & CTRL:
            self.sel ^= {self.ids[i]}
            self._set_focus(i); self.anchor = i
        else:
            self.sel = {self.ids[i]}
            self._set_focus(i); self.anchor = i
        if i is not None and self.single(): self.ensure_visible(i)
        self._changed()
        self.request_redraw()

    def _on_double(self, e):
        i, part = self.hit(e.x, e.y)
        if i is not None and not part: self.toggle_single(i)

    def _on_menu(self, e):
        i, _ = self.hit(e.x, e.y)
        if i is not None and self.ids[i] not in self.sel: self.click(i)
        self.app.grid_menu(e)

    # ─── keys
    def move(self, step, extend=False):
        n = len(self.ids)
        if not n: return
        i = self.focus if self.focus is not None else -1 if step > 0 else n
        j = max(0, min(n - 1, i + step))
        if extend and self.anchor is not None:
            a, b = sorted((self.anchor, j))
            self.sel = set(self.ids[a:b + 1])
        else:
            self.sel = {self.ids[j]}; self.anchor = j
        self._set_focus(j)
        self.ensure_visible(j)
        self._changed()

    def _on_key(self, e):
        k = e.keysym
        W, H, g, cols, cw, ch, band = self.geom()
        page = cols * max(1, H // (ch + g))
        steps = {'Left': -1, 'Right': 1, 'Up': -1 if self.single() else -cols, 'Down': 1 if self.single() else cols,
                 'Prior': -1 if self.single() else -page, 'Next': 1 if self.single() else page, 'Home': -10 ** 9, 'End': 10 ** 9}
        if k in steps:
            self.move(steps[k], bool(e.state & SHIFT))
            return 'break'
        if e.char and e.char in '012345' and not e.state & CTRL and self.PREFIX == 'p':
            ids = self.selected_ids()
            if ids: self.app.set_rating(ids, int(e.char))
            return 'break'
        if k in ('Return', 'space', 'KP_Enter'):
            self.toggle_single(); return 'break'
        if k == 'Escape':
            if self.single(): self.toggle_single()
            else:
                self.sel = set(); self._changed(); self.request_redraw()
            return 'break'
        if e.char in ('+', '='): self.set_zoom(self.zoom + 1); return 'break'
        if e.char in ('-', '_'): self.set_zoom(self.zoom - 1); return 'break'


class FaceGrid(PhotoGrid):
    """The same grid for face crops (ids are face ids): one fixed size, no stars."""
    PREFIX = 'f'

    def geom(self):
        W, H = max(60, self.cv.winfo_width()), max(60, self.cv.winfo_height())
        g, z = self.GAP, int(112 * self.app.ui_scale)
        cols = max(1, (W - g) // (z + g))
        cw = (W - g) // cols - g
        return W, H, g, cols, cw, cw, 0

    def single(self):
        return False

    def cell_job(self, fid, bw, bh):
        r = self.rows.get(fid)
        if not r: return None, None
        s = min(bw, bh)
        return (self.PREFIX, fid, s), ('face', r['photo_id'], (r['x0'], r['y0'], r['x1'], r['y1']), s)

    def draw_cell(self, i, x, y, cw, ch, band):
        key = super().draw_cell(i, x, y, cw, ch, band)
        r = self.rows.get(self.ids[i]) or {}
        if r.get('assigned') == 'auto':   # suggested, not yet confirmed
            self.cv.create_oval(x + cw - 22, y + 6, x + cw - 6, y + 22, fill=C['warn'], outline='')
            self.cv.create_text(x + cw - 14, y + 14, text='?', fill='#000', font=self.f_badge)
        return key

    def set_zoom(self, z, around=None):
        pass

    def _changed(self):
        pass

    def toggle_single(self, i=None):
        i = self.focus if i is None else i
        if i is not None: self.app.show_face_photo(self.ids[i])

    def _on_menu(self, e):
        pass


# ───────────────────────────────────────────── the app
class App:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.job = None
        self.busy = False
        self.settings = self._load_settings()
        s = self.settings
        self.cat = L.Catalog(s.get('library_dir') or L.DEFAULT_LIBRARY)
        self.ai = None
        self.loader = ThumbLoader(self.cat, self.q)
        self.scope = None                 # what the Browse trees point at
        self.results = None               # (label, [(id, score)]) from a search by meaning / similar photos
        self._tree_spec = object()
        self._filter_due = self._sel_due = None
        self._loading_filter = self._quiet_tree = self._desc_mixed = False
        self._sel_rows, self._tag_names, self._tag_people, self.vfilters = {}, [], set(), []
        self.cat.sync_person_tags()   # names given before person tags existed, or by another copy of the app
        self.devices = [('cpu', 'CPU')]
        self.persons = {}
        self.kind_filter = list(KIND_FILTER)

        self.ui_scale = max(1.0, root.winfo_fpixels('1i') / 96)
        root.title('Photo Library desktop')
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(s.get('geometry') or f'{int(sw * 0.85)}x{int(sh * 0.82)}+{int(sw * 0.07)}+{int(sh * 0.05)}')
        root.minsize(int(1100 * self.ui_scale), int(640 * self.ui_scale))
        if s.get('zoomed'):
            try:
                root.state('zoomed') if WINDOWS else root.attributes('-zoomed', True)
            except tk.TclError:
                pass
        root.configure(bg=C['bg'])
        self._style()
        self._build()
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        root.after(50, self._poll)
        root.after(80, self.refresh_all)
        threading.Thread(target=self._probe, daemon=True).start()
        self._clean_send_lists()

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
        s.update(zoomed=zoomed, zoom=self.grid.zoom, sort=self.sort.get(), desc=self.desc.get(), recursive=self.recursive.get(),
                 browse_tab=self.browse.index('current'), library_dir=self.lib_dir.get().strip() or L.DEFAULT_LIBRARY,
                 rawdemosaic_dir=self.rd_dir.get().strip(), panostitch_dir=self.ps_dir.get().strip(), batch_settings=self.batch_json.get().strip(),
                 thumb_mode=key_of(THUMB_MODES, self.thumb_mode.get(), 1024), models_dir=self.models_dir.get().strip(),
                 device=self.device_key(), ai_faces=self.ai_faces.get(), ai_embed=self.ai_embed.get(), ai_caption=self.ai_caption.get(),
                 ai_auto=self.ai_auto.get())
        if not zoomed: s['geometry'] = self.root.geometry()
        try:
            with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
                json.dump(s, f, indent=1)
        except Exception:
            pass

    def min_preview(self):
        return key_of(THUMB_MODES, self.thumb_mode.get(), 1024)

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
        st.configure('Big.TLabel', background=C['card'], foreground=C['text'], font=(UI_FONT, 11, 'bold'))
        st.configure('Info.TLabel', background=C['card'], foreground=C['info'], font=(UI_FONT, 9, 'bold'))
        st.configure('Title.TLabel', background=C['bg'], foreground=C['text'], font=(UI_FONT, 16, 'bold'))
        st.configure('Sub.TLabel', background=C['bg'], foreground=C['muted'])
        st.configure('Status.TLabel', background=C['bg'], foreground=C['muted'], font=(UI_FONT, 9))
        st.configure('Card.TCheckbutton', background=C['card'], foreground=C['text'])
        st.map('Card.TCheckbutton', background=[('active', C['card'])], indicatorcolor=[('selected', C['accent']), ('!selected', C['input'])])
        st.configure('TButton', background=C['card2'], foreground=C['text'], padding=(10, 5), borderwidth=1)
        st.map('TButton', background=[('disabled', C['card']), ('active', C['border'])], foreground=[('disabled', C['muted'])])
        st.configure('Small.TButton', padding=(6, 2))
        st.configure('Accent.TButton', background=C['accent'], foreground='#111', font=(UI_FONT, 10, 'bold'), padding=(14, 7))
        st.map('Accent.TButton', background=[('disabled', C['card2']), ('active', C['accent_hover'])], foreground=[('disabled', C['muted'])])
        st.configure('TCombobox', fieldbackground=C['input'], background=C['card2'], foreground=C['text'], arrowcolor=C['text'])
        st.map('TCombobox', fieldbackground=[('readonly', C['input']), ('disabled', C['card'])], foreground=[('readonly', C['text']), ('disabled', C['muted'])],
               selectbackground=[('readonly', C['input'])], selectforeground=[('readonly', C['text'])])
        self.root.option_add('*TCombobox*Listbox.background', C['input'])
        self.root.option_add('*TCombobox*Listbox.foreground', C['text'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', C['accent'])
        st.configure('TEntry', fieldbackground=C['input'], foreground=C['text'])
        st.configure('Horizontal.TProgressbar', background=C['accent'], troughcolor=C['track'], thickness=10)
        st.configure('TNotebook', background=C['bg'], borderwidth=0)
        st.configure('TNotebook.Tab', background=C['card'], foreground=C['muted'], padding=(16, 6))
        st.map('TNotebook.Tab', background=[('selected', C['card2'])], foreground=[('selected', C['accent'])])
        st.configure('Card.TNotebook', background=C['card'], borderwidth=0)
        st.configure('Card.TNotebook.Tab', background=C['input'], foreground=C['muted'], padding=(12, 4))
        st.map('Card.TNotebook.Tab', background=[('selected', C['card2'])], foreground=[('selected', C['accent'])])
        st.configure('Vertical.TScrollbar', background=C['card2'], troughcolor=C['card'], arrowcolor=C['muted'])
        st.configure('Horizontal.TScale', background=C['card'], troughcolor=C['track'])
        st.configure('Treeview', background=C['input'], fieldbackground=C['input'], foreground=C['text'], bordercolor=C['border'],
                     rowheight=int(22 * self.ui_scale))
        st.map('Treeview', background=[('selected', C['accent'])], foreground=[('selected', '#000')])

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

    def _tree(self, parent, height):
        lf = ttk.Frame(parent, style='Card.TFrame')
        lf.pack(fill='both', expand=True)
        t = ttk.Treeview(lf, columns=('n',), show='tree', height=height, selectmode='browse')
        t.column('#0', stretch=True, width=200)
        t.column('n', width=int(56 * self.ui_scale), anchor='e', stretch=False)
        t.tag_configure('empty', foreground=C['muted'])
        sb = ttk.Scrollbar(lf, orient='vertical', command=t.yview)
        t.configure(yscrollcommand=sb.set)
        t.pack(side='left', fill='both', expand=True)
        sb.pack(side='left', fill='y')
        return t

    def _row(self, parent, label, widget_factory, width=10):
        r = ttk.Frame(parent, style='Card.TFrame')
        r.pack(fill='x', pady=2)
        ttk.Label(r, text=label, style='Card.TLabel', width=width).pack(side='left')
        w = widget_factory(r)
        w.pack(side='left', fill='x', expand=True)
        return w

    def _range_row(self, parent, label, lo, hi, hint='', width=10):
        r = ttk.Frame(parent, style='Card.TFrame')
        r.pack(fill='x', pady=2)
        ttk.Label(r, text=label, style='Card.TLabel', width=width).pack(side='left')
        ttk.Entry(r, textvariable=lo, width=9).pack(side='left', fill='x', expand=True)
        ttk.Label(r, text=' – ', style='Muted.TLabel').pack(side='left')
        ttk.Entry(r, textvariable=hi, width=9).pack(side='left', fill='x', expand=True)
        if hint: ttk.Label(r, text=' ' + hint, style='Muted.TLabel', width=4).pack(side='left')

    def _combo(self, parent, var, pairs_or_values, width=None, editable=False):
        vals = [l for l, _ in pairs_or_values] if pairs_or_values and isinstance(pairs_or_values[0], tuple) else list(pairs_or_values)
        cb = ttk.Combobox(parent, textvariable=var, state='normal' if editable else 'readonly', values=vals)
        if width: cb.configure(width=width)
        return cb

    def _button_row(self, parent, buttons, pady=(6, 0)):
        br = ttk.Frame(parent, style='Card.TFrame')
        br.pack(fill='x', pady=pady)
        out = []
        for i, (text, cmd) in enumerate(buttons):
            b = ttk.Button(br, text=text, command=cmd, style='Small.TButton')
            b.pack(side='left', padx=(0 if i == 0 else 4, 0))
            out.append(b)
        return out

    def _scroll_column(self, tab, width=340, side='left'):
        """Fixed-width column that scrolls when it is taller than the window. Returns (inner, pinned)."""
        outer = ttk.Frame(tab, width=int(width * self.ui_scale))
        outer.pack(side=side, fill='y')
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
            if sb.winfo_ismapped() and not isinstance(e.widget, (ttk.Treeview, tk.Listbox, tk.Text)):
                cv.yview_scroll(-d, 'units')
        outer.bind('<Enter>', lambda e: bind_wheel(self.root, wheel, all_=True))
        outer.bind('<Leave>', lambda e: unbind_wheel(self.root))
        return inner, pinned

    # ─────────────────────────────── layout
    def _build(self):
        top = ttk.Frame(self.root, padding=(14, 10, 14, 4))
        top.pack(fill='x')
        ttk.Label(top, text='PHOTO LIBRARY', style='Title.TLabel').pack(side='left')
        ttk.Label(top, text='  desktop · your photos stay where they are: the catalog only reads them', style='Sub.TLabel').pack(side='left', pady=(5, 0))
        ttk.Button(top, text='Rescan all', command=self.rescan_all).pack(side='right')
        ttk.Button(top, text='Add folder…', style='Accent.TButton', command=self.add_folder).pack(side='right', padx=6)

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
        self.log_text = tk.Text(lf, height=4, bg=C['input'], fg=C['text'], relief='flat', wrap='word',
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
        self._build_settings_vars()
        self._build_library_tab()
        self._build_people_tab()
        self._build_settings_tab()
        self.nb.bind('<<NotebookTabChanged>>', lambda e: self._tab_changed())

    def _build_settings_vars(self):
        s = self.settings
        self.lib_dir = tk.StringVar(value=s.get('library_dir') or L.DEFAULT_LIBRARY)
        self.rd_dir = tk.StringVar(value=s.get('rawdemosaic_dir') or L.DEFAULT_RAWDEMOSAIC)
        self.ps_dir = tk.StringVar(value=s.get('panostitch_dir') or DEFAULT_PANOSTITCH)
        self.batch_json = tk.StringVar(value=s.get('batch_settings', ''))
        self.thumb_mode = tk.StringVar(value=label_of(THUMB_MODES, s.get('thumb_mode', 1024)))
        self.models_dir = tk.StringVar(value=s.get('models_dir') or AI.DEFAULT_MODELS)
        self.device = tk.StringVar(value='')
        self.ai_faces = tk.BooleanVar(value=s.get('ai_faces', True))
        self.ai_embed = tk.BooleanVar(value=s.get('ai_embed', True))
        self.ai_caption = tk.BooleanVar(value=s.get('ai_caption', True))
        self.ai_auto = tk.BooleanVar(value=s.get('ai_auto', True))
        self.recursive = tk.BooleanVar(value=s.get('recursive', True))
        self.sort = tk.StringVar(value=s.get('sort', 'Date taken'))
        self.desc = tk.BooleanVar(value=s.get('desc', False))

    # ─── tab 1: library
    def _build_library_tab(self):
        tab = ttk.Frame(self.nb, padding=(0, 8, 0, 0))
        self.nb.add(tab, text='  Library  ')
        left, _ = self._scroll_column(tab, 330)
        right, rpinned = self._scroll_column(tab, 330, side='right')
        mid = ttk.Frame(tab)
        mid.pack(side='left', fill='both', expand=True, padx=10)

        c = self._card(left, 'Browse')
        self.browse = ttk.Notebook(c, style='Card.TNotebook')
        self.browse.pack(fill='both', expand=True)
        f = ttk.Frame(self.browse, style='Card.TFrame', padding=(0, 6, 0, 0))
        self.browse.add(f, text='Folders')
        self.ftree = self._tree(f, 14)
        ttk.Checkbutton(f, text='Include subfolders', variable=self.recursive, style='Card.TCheckbutton',
                        command=self._scope_changed).pack(anchor='w', pady=(4, 0))
        f = ttk.Frame(self.browse, style='Card.TFrame', padding=(0, 6, 0, 0))
        self.browse.add(f, text='Dates')
        self.dtree = self._tree(f, 15)
        f = ttk.Frame(self.browse, style='Card.TFrame', padding=(0, 6, 0, 0))
        self.browse.add(f, text='Virtual folders')
        ttk.Label(f, text='Saved filters: each shows every photo in the library that matches it.',
                  style='Muted.TLabel', wraplength=int(290 * self.ui_scale), justify='left').pack(anchor='w', pady=(0, 4))
        self.vlist = self._scrolled_listbox(f, 12, fill='both', expand=True)
        self._button_row(f, [('Save filter as…', self.save_virtual), ('Update', self.update_virtual), ('Delete', self.delete_virtual)])
        self.ftree.bind('<<TreeviewSelect>>', lambda e: self._scope_changed())
        self.dtree.bind('<<TreeviewSelect>>', lambda e: self._scope_changed())
        self.vlist.bind('<<ListboxSelect>>', lambda e: self._virtual_selected())
        self.browse.bind('<<NotebookTabChanged>>', lambda e: self._scope_changed())
        self.ftree.bind('<Button-3>', self._folder_menu)

        c = self._card(left, 'Filter')
        self.f_rating, self.f_tags, self.f_tagmode = tk.StringVar(value='Any'), tk.StringVar(), tk.StringVar(value='all')
        self.f_camera, self.f_lens, self.f_person = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.f_gps, self.f_kind = tk.StringVar(value='Any'), tk.StringVar(value='Any')
        self.f_iso = (tk.StringVar(), tk.StringVar()); self.f_ap = (tk.StringVar(), tk.StringVar())
        self.f_focal = (tk.StringVar(), tk.StringVar()); self.f_exp = (tk.StringVar(), tk.StringVar())
        self.f_date = (tk.StringVar(), tk.StringVar())
        self._row(c, 'Rating', lambda p: self._combo(p, self.f_rating, RATING_FILTER))
        r = ttk.Frame(c, style='Card.TFrame'); r.pack(fill='x', pady=2)
        ttk.Label(r, text='Tags', style='Card.TLabel', width=10).pack(side='left')
        ttk.Entry(r, textvariable=self.f_tags).pack(side='left', fill='x', expand=True)
        self._combo(r, self.f_tagmode, ['all', 'any'], 4).pack(side='left', padx=(4, 0))
        r = ttk.Frame(c, style='Card.TFrame'); r.pack(fill='x', pady=2)
        ttk.Label(r, text='', style='Card.TLabel', width=10).pack(side='left')
        self.f_tagpick = tk.StringVar()
        self.f_tagpick_box = self._combo(r, self.f_tagpick, ['(add a tag)'])
        self.f_tagpick_box.pack(side='left', fill='x', expand=True)
        self.f_tagpick_box.bind('<<ComboboxSelected>>', lambda e: self._pick_filter_tag())
        self.f_camera_box = self._row(c, 'Camera', lambda p: self._combo(p, self.f_camera, ['']))
        self.f_lens_box = self._row(c, 'Lens', lambda p: self._combo(p, self.f_lens, ['']))
        self.f_person_box = self._row(c, 'Person', lambda p: self._combo(p, self.f_person, ['']))
        self._range_row(c, 'ISO', *self.f_iso)
        self._range_row(c, 'Aperture', *self.f_ap, hint='f/')
        self._range_row(c, 'Focal', *self.f_focal, hint='mm')
        self._range_row(c, 'Shutter', *self.f_exp, hint='s')
        self._range_row(c, 'Date', *self.f_date)
        self._row(c, 'Location', lambda p: self._combo(p, self.f_gps, GPS_FILTER))
        self.f_kind_box = self._row(c, 'File type', lambda p: self._combo(p, self.f_kind, KIND_FILTER))
        ttk.Label(c, text='Ranges: either end may be empty. Shutter as 1/250 or 2; dates as 2024-05-31 (or just 2024).',
                  style='Muted.TLabel', wraplength=int(290 * self.ui_scale), justify='left').pack(anchor='w', pady=(4, 0))
        self._button_row(c, [('Clear filter', self.clear_filter), ('Save as virtual folder…', self.save_virtual)])
        for v in (self.f_rating, self.f_tags, self.f_tagmode, self.f_camera, self.f_lens, self.f_person, self.f_gps, self.f_kind,
                  *self.f_iso, *self.f_ap, *self.f_focal, *self.f_exp, *self.f_date):
            v.trace_add('write', lambda *a: self._filter_changed())

        tb = ttk.Frame(mid, style='Card.TFrame', padding=(8, 6))
        tb.pack(fill='x', pady=(0, 8))
        bar = ttk.Frame(tb, style='Card.TFrame')
        bar.pack(fill='x')
        self.search_var = tk.StringVar()
        se = ttk.Entry(bar, textvariable=self.search_var, width=30)
        se.pack(side='left', fill='x', expand=True)
        se.bind('<Return>', lambda e: self.search())
        self.search_var.trace_add('write', lambda *a: self.search_mode.get() == SEARCH_MODES[0] and self._filter_changed())
        self.search_mode = tk.StringVar(value=SEARCH_MODES[0])
        self._combo(bar, self.search_mode, SEARCH_MODES, 13).pack(side='left', padx=4)
        ttk.Button(bar, text='Search', style='Small.TButton', command=self.search).pack(side='left')
        self.results_lbl = ttk.Label(bar, text='', style='Info.TLabel')
        self.results_lbl.pack(side='left', padx=(10, 0))
        self.results_clear = ttk.Button(bar, text='✕ Show all', style='Small.TButton', command=self.clear_results)
        bar = ttk.Frame(tb, style='Card.TFrame')
        bar.pack(fill='x', pady=(6, 0))
        self.zoom_var = tk.IntVar(value=int(self.settings.get('zoom', 4)))
        self.zoom_lbl = ttk.Label(bar, text='', style='Muted.TLabel', width=9)
        self.zoom_lbl.pack(side='right')
        ttk.Scale(bar, from_=0, to=len(ZOOMS) - 1, variable=self.zoom_var, orient='horizontal', length=int(150 * self.ui_scale),
                  command=lambda v: self.grid.set_zoom(round(float(v)))).pack(side='right', padx=(4, 4))
        ttk.Label(bar, text='Size', style='Card.TLabel').pack(side='right', padx=(12, 0))
        ttk.Label(bar, text='Sort', style='Card.TLabel').pack(side='left')
        sc = self._combo(bar, self.sort, list(L.SORTS), 11)
        sc.pack(side='left', padx=4)
        sc.bind('<<ComboboxSelected>>', lambda e: self.refresh())
        ttk.Checkbutton(bar, text='Descending', variable=self.desc, style='Card.TCheckbutton', command=lambda: self.refresh()).pack(side='left')

        gc = ttk.Frame(mid, style='Card.TFrame', padding=4)
        gc.pack(fill='both', expand=True)
        self.grid = PhotoGrid(self, gc)
        self.grid.zoom = self.zoom_var.get()
        self.zoom_changed(self.grid.zoom)

        c = self._card(right, 'Selection')
        self.sel_title = ttk.Label(c, text='Nothing selected', style='Big.TLabel', wraplength=int(290 * self.ui_scale), justify='left')
        self.sel_title.pack(anchor='w')
        self.sel_info = ttk.Label(c, text='', style='Muted.TLabel', wraplength=int(300 * self.ui_scale), justify='left')
        self.sel_info.pack(anchor='w', pady=(4, 0))
        sr = ttk.Frame(c, style='Card.TFrame'); sr.pack(fill='x', pady=(8, 0))
        self.stars = []
        for k in range(1, 6):
            lb = tk.Label(sr, text='★', font=(UI_FONT, 18), bg=C['card'], fg=C['star_off'], cursor='hand2')
            lb.pack(side='left')
            lb.bind('<Button-1>', lambda e, k=k: self._star_clicked(k))
            self.stars.append(lb)
        lb = tk.Label(sr, text='  clear', font=(UI_FONT, 9), bg=C['card'], fg=C['muted'], cursor='hand2')
        lb.pack(side='left', pady=(6, 0))
        lb.bind('<Button-1>', lambda e: self._star_clicked(0))
        self.map_btn = ttk.Button(c, text='Show location on a map', style='Small.TButton', command=self.open_map)
        self.caption_lbl = ttk.Label(c, text='', style='Muted.TLabel', wraplength=int(300 * self.ui_scale), justify='left')

        c = self._card(right, 'Tags')
        r = ttk.Frame(c, style='Card.TFrame'); r.pack(fill='x')
        self.tag_add = tk.StringVar()
        self.tag_box = self._combo(r, self.tag_add, [''], editable=True)
        self.tag_box.pack(side='left', fill='x', expand=True)
        self.tag_box.bind('<Return>', lambda e: self.add_tags())
        ttk.Button(r, text='Add', style='Small.TButton', command=self.add_tags).pack(side='left', padx=(4, 0))
        self.tag_list = self._scrolled_listbox(c, 6, pady=(6, 0))
        self.tag_list.configure(selectmode='extended')
        self.tag_list.bind('<Delete>', lambda e: self.remove_tags())
        self.tag_list.bind('<Double-Button-1>', lambda e: self._filter_by_tag())
        self._button_row(c, [('Remove', self.remove_tags), ('Filter by it', self._filter_by_tag)])
        ttk.Label(c, text='Comma-separated tags are added to every selected photo. AI tags come from the captions.',
                  style='Muted.TLabel', wraplength=int(290 * self.ui_scale), justify='left').pack(anchor='w', pady=(4, 0))

        c = self._card(right, 'Description')
        self.desc_text = tk.Text(c, height=4, bg=C['input'], fg=C['text'], relief='flat', wrap='word', insertbackground=C['text'],
                                 highlightthickness=1, highlightbackground=C['border'], highlightcolor=C['accent'], font=(UI_FONT, 10))
        self.desc_text.pack(fill='x')
        self._button_row(c, [('Save description', self.save_description), ('Use AI caption', self.use_caption)])

        c = self._card(right, 'Send to')
        for text, cmd in (('Open in RAW Demosaic', self.open_rawdemosaic), ('Batch process in RAW Demosaic…', self.batch_rawdemosaic),
                          ('Stitch in PANOSTITCH', self.open_panostitch), ('Show in Explorer' if WINDOWS else 'Show in folder', self.reveal),
                          ('Find similar photos (AI)', self.find_similar)):
            ttk.Button(c, text=text, command=cmd).pack(fill='x', pady=2)

    # ─── tab 2: people
    def _build_people_tab(self):
        tab = ttk.Frame(self.nb, padding=(0, 8, 0, 0))
        self.nb.add(tab, text='  People  ')
        left, _ = self._scroll_column(tab, 330)
        right = ttk.Frame(tab)
        right.pack(side='left', fill='both', expand=True, padx=(10, 0))
        c = self._card(left, 'People')
        self.p_list = self._scrolled_listbox(c, 10)
        self.p_list.bind('<<ListboxSelect>>', lambda e: self._people_selected('person'))
        self._button_row(c, [('Rename…', self.rename_person), ('Show their photos', self.person_photos)])
        c = self._card(left, 'Unnamed faces')
        ttk.Label(c, text='Faces found by the AI that nobody has named. Group similar faces, pick a group, select the\n'
                          'faces of one person and name them; other faces like them are then suggested (marked ?).',
                  style='Muted.TLabel', wraplength=int(290 * self.ui_scale), justify='left').pack(anchor='w', pady=(0, 4))
        self.g_list = self._scrolled_listbox(c, 12)
        self.g_list.bind('<<ListboxSelect>>', lambda e: self._people_selected('group'))
        self._button_row(c, [('Group similar faces', self.group_faces), ('Suggest names', self.suggest_names)])

        bar = ttk.Frame(right, style='Card.TFrame', padding=(8, 6))
        bar.pack(fill='x', pady=(0, 8))
        self.face_title = ttk.Label(bar, text='Pick a person or a group on the left.', style='Big.TLabel')
        self.face_title.pack(side='left')
        ttk.Button(bar, text='Show photo', style='Small.TButton', command=lambda: self.fgrid.toggle_single()).pack(side='right')
        ttk.Button(bar, text='Not this person', style='Small.TButton', command=self.unname_faces).pack(side='right', padx=4)
        ttk.Button(bar, text='Confirm', style='Small.TButton', command=self.confirm_faces).pack(side='right')
        ttk.Button(bar, text='Name selected', style='Small.TButton', command=self.name_faces).pack(side='right', padx=4)
        self.face_name = tk.StringVar()
        self.face_name_box = self._combo(bar, self.face_name, [''], 18, editable=True)
        self.face_name_box.pack(side='right')
        self.face_name_box.bind('<Return>', lambda e: self.name_faces())
        gc = ttk.Frame(right, style='Card.TFrame', padding=4)
        gc.pack(fill='both', expand=True)
        self.fgrid = FaceGrid(self, gc)
        self.fgrid.empty_text = 'No faces here yet.\n\nFaces are found when the AI runs (Settings & AI).'
        self._face_source = None

    # ─── tab 3: settings and AI
    def _build_settings_tab(self):
        tab = ttk.Frame(self.nb, padding=(0, 8, 0, 0))
        self.nb.add(tab, text='  Settings & AI  ')
        cols = ttk.Frame(tab)
        cols.pack(fill='both', expand=True)
        left = ttk.Frame(cols); left.pack(side='left', fill='both', expand=True)
        right = ttk.Frame(cols); right.pack(side='left', fill='both', expand=True, padx=(10, 0))

        c = self._card(left, 'Library')
        r = ttk.Frame(c, style='Card.TFrame'); r.pack(fill='x', pady=2)
        ttk.Label(r, text='Catalog folder', style='Card.TLabel', width=16).pack(side='left')
        ttk.Entry(r, textvariable=self.lib_dir).pack(side='left', fill='x', expand=True)
        ttk.Button(r, text='…', width=3, command=lambda: self._choose_dir(self.lib_dir, 'Catalog folder')).pack(side='left', padx=(4, 0))
        ttk.Label(c, text='Holds catalog.sqlite (ratings, tags, EXIF, faces, AI vectors) and the thumbnails. A change takes\n'
                          'effect when the app is restarted. Back this folder up: it is everything the app knows.',
                  style='Muted.TLabel', justify='left').pack(anchor='w', pady=(2, 6))
        ttk.Label(c, text='Folders in the library', style='Card.TLabel').pack(anchor='w')
        self.root_list = self._scrolled_listbox(c, 5, pady=(2, 0))
        self.root_list.configure(selectmode='extended')
        self._button_row(c, [('Add folder…', self.add_folder), ('Rescan selected', self.rescan_selected), ('Remove from library…', self.remove_roots),
                             ('Forget missing photos', self.forget_missing)])
        ttk.Label(c, text='Removing a folder forgets its photos\' ratings, tags and descriptions; the files are never touched.',
                  style='Muted.TLabel').pack(anchor='w', pady=(4, 0))

        c = self._card(left, 'Thumbnails')
        self._row(c, 'RAW thumbnails', lambda p: self._combo(p, self.thumb_mode, THUMB_MODES), 16)
        ttk.Label(c, text='Thumbnails are 128, 512 and 1024 px. A RAW file\'s embedded JPEG is used when it is big enough;\n'
                          'otherwise the RAW data is demosaiced with RCD by RAW Demosaic\'s own engine (Bayer; X-Trans uses\n'
                          'LibRaw\'s 1-pass Markesteijn). Right-click photos → Rebuild thumbnails to redo them.',
                  style='Muted.TLabel', justify='left').pack(anchor='w', pady=(2, 0))

        c = self._card(left, 'Other apps')
        for label, var, title in (('RAW Demosaic', self.rd_dir, 'RAW Demosaic\'s desktop folder'),
                                  ('PANOSTITCH', self.ps_dir, 'PANOSTITCH\'s desktop folder')):
            r = ttk.Frame(c, style='Card.TFrame'); r.pack(fill='x', pady=2)
            ttk.Label(r, text=label, style='Card.TLabel', width=16).pack(side='left')
            ttk.Entry(r, textvariable=var).pack(side='left', fill='x', expand=True)
            ttk.Button(r, text='…', width=3, command=lambda v=var, t=title: self._choose_dir(v, t)).pack(side='left', padx=(4, 0))
        r = ttk.Frame(c, style='Card.TFrame'); r.pack(fill='x', pady=2)
        ttk.Label(r, text='Batch settings', style='Card.TLabel', width=16).pack(side='left')
        ttk.Entry(r, textvariable=self.batch_json).pack(side='left', fill='x', expand=True)
        ttk.Button(r, text='…', width=3, command=self._choose_batch_json).pack(side='left', padx=(4, 0))
        ttk.Label(c, text='The folders that hold rawdemosaic_desktop.py and panostitch_desktop.py; each app runs in its own\n'
                          'Python environment. Batch settings: a file saved with RAW Demosaic\'s "Save settings…".',
                  style='Muted.TLabel', justify='left').pack(anchor='w', pady=(2, 0))

        c = self._card(right, 'AI models')
        r = ttk.Frame(c, style='Card.TFrame'); r.pack(fill='x', pady=2)
        ttk.Label(r, text='Models folder', style='Card.TLabel', width=16).pack(side='left')
        ttk.Entry(r, textvariable=self.models_dir).pack(side='left', fill='x', expand=True)
        ttk.Button(r, text='…', width=3, command=lambda: self._choose_dir(self.models_dir, 'AI models folder')).pack(side='left', padx=(4, 0))
        self.device_box = self._row(c, 'Run models on', lambda p: self._combo(p, self.device, ['CPU']), 16)
        self.models_info = ttk.Label(c, text='', style='Card.TLabel', justify='left')
        self.models_info.pack(anchor='w', pady=(6, 0))
        self._button_row(c, [('Download missing models', self.download_models), ('Free GPU memory', self.free_gpu)])
        ttk.Label(c, text='Faces: InsightFace buffalo_l — SCRFD detector + ArcFace (non-commercial licence). Search by meaning:\n'
                          'SigLIP2 so400m. Captions: Florence-2 large. About 6 GB in all; they run on the thumbnails.',
                  style='Muted.TLabel', justify='left').pack(anchor='w', pady=(4, 0))

        c = self._card(right, 'AI on photos')
        ttk.Checkbutton(c, text='Find faces (for People)', variable=self.ai_faces, style='Card.TCheckbutton').pack(anchor='w')
        ttk.Checkbutton(c, text='Image embeddings (for search by meaning and similar photos)', variable=self.ai_embed, style='Card.TCheckbutton').pack(anchor='w')
        ttk.Checkbutton(c, text='Captions and tags', variable=self.ai_caption, style='Card.TCheckbutton').pack(anchor='w')
        ttk.Checkbutton(c, text='Run them automatically on new photos after their thumbnails are made', variable=self.ai_auto,
                        style='Card.TCheckbutton').pack(anchor='w', pady=(6, 0))
        self._button_row(c, [('Run on photos not done yet', lambda: self.run_ai()), ('Redo on selected photos', lambda: self.run_ai(self.grid.selected_ids()))])

        c = self._card(right, 'Library statistics')
        self.stats_lbl = ttk.Label(c, text='', style='Card.TLabel', justify='left')
        self.stats_lbl.pack(anchor='w')

    # ─────────────────────────────── helpers
    def log(self, level, msg):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', time.strftime('%H:%M:%S '), 'time')
        self.log_text.insert('end', msg + '\n', level if level in ('ok', 'err', 'info', 'warn') else '')
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def device_key(self):
        return next((k for k, v in self.devices if v == self.device.get()), self.settings.get('device', 'cuda:0'))

    def _probe(self):
        devs = AI.list_devices()
        self.q.put(('call', self._set_devices, devs))

    def _set_devices(self, devs):
        self.devices = devs
        self.device_box.configure(values=[v for _, v in devs])
        want = self.settings.get('device', devs[0][0])
        self.device.set(next((v for k, v in devs if k == want), devs[0][1]))

    def _choose_dir(self, var, title):
        d = filedialog.askdirectory(initialdir=var.get() or None, title=title)
        if d: var.set(os.path.normpath(d))

    def _choose_batch_json(self):
        p = filedialog.askopenfilename(filetypes=[('Settings', '*.json')], initialdir=os.path.dirname(self.batch_json.get()) or None,
                                       title='RAW Demosaic settings for batches')
        if p: self.batch_json.set(os.path.normpath(p))

    def _tab_changed(self):
        i = self.nb.index('current')
        if i == 1: self.refresh_people()
        elif i == 2: self.refresh_settings_info()

    # ─────────────────────────────── worker plumbing
    def run_task(self, label, fn, done=None):
        """fn(job) runs on a worker thread; done(result) then runs on the Tk thread."""
        if self.busy:
            messagebox.showinfo(APP, 'Something is already running — wait for it or press Cancel.')
            return False
        self.busy = True
        job = L.Job(log=lambda l, m: self.q.put(('log', l, m)), progress=lambda f, t=None: self.q.put(('progress', f, t)),
                    post=lambda *m: self.q.put(m))
        self.job = job
        self.cancel_btn.configure(state='normal')
        self.progress['value'] = 0
        self.status.configure(text=label + '…')
        t0 = time.time()

        def work():
            try:
                res = fn(job)
                self.q.put(('done', done, res, time.time() - t0))
            except L.Cancelled:
                self.q.put(('cancelled',))
            except Exception as ex:
                msg = str(ex) or type(ex).__name__
                if 'out of memory' in msg.lower(): msg = 'The GPU ran out of memory — press Free GPU memory, or choose the other GPU or the CPU.\n\n' + msg
                self.q.put(('error', msg, traceback.format_exc()))
        threading.Thread(target=work, daemon=True).start()
        return True

    def cancel(self):
        if self.job:
            self.job.cancel_event.set()
            self.status.configure(text='Cancelling…')

    def _poll(self):
        last_prog, thumbs, ai_ids = None, [], []
        try:
            for _ in range(800):
                m = self.q.get_nowait()
                kind = m[0]
                if kind == 'thumb_img':
                    (self.grid if m[1][0] == 'p' else self.fgrid).got(m[1], m[2])
                elif kind == 'log':
                    self.log(m[1], m[2])
                elif kind == 'progress':
                    last_prog = m
                elif kind == 'thumbs':
                    thumbs += m[1]
                elif kind == 'ai':
                    ai_ids += m[1]
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
        if thumbs: self._thumbs_ready(thumbs)
        if ai_ids and set(ai_ids) & self.grid.sel: self.selection_changed()
        self.root.after(50, self._poll)

    def _finish(self, m):
        self.busy = False
        self.job = None
        self.cancel_btn.configure(state='disabled')
        kind = m[0]
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
            self.refresh_all()
        else:
            self.progress['value'] = 0
            self.status.configure(text='Failed: ' + m[1].splitlines()[0])
            self.log('err', m[1])
            print(m[2])
            messagebox.showerror(APP, m[1])
            self.refresh_all()

    def _thumbs_ready(self, pids):
        rows = self.cat.rows(pids, 'id, thumb')
        for pid, r in rows.items():
            if pid in self.grid.rows: self.grid.rows[pid]['thumb'] = r['thumb']
        self.grid.invalidate(pids)

    # ─────────────────────────────── refreshing
    def refresh_all(self):
        self.refresh_lists()
        self._tree_spec = object()
        self.refresh()

    def refresh_lists(self):
        tags = self.cat.all_tags()
        self.tag_box.configure(values=tags)
        self.f_tagpick_box.configure(values=['(add a tag)'] + tags)
        self.f_camera_box.configure(values=[''] + self.cat.distinct('camera'))
        self.f_lens_box.configure(values=[''] + self.cat.distinct('lens'))
        self.kind_filter = KIND_FILTER + L.file_types(self.cat.distinct('ext'))
        self.f_kind_box.configure(values=[l for l, _ in self.kind_filter])
        self.persons = {r['name']: r['id'] for r in self.cat.q('SELECT id, name FROM persons ORDER BY name COLLATE NOCASE')}
        self.f_person_box.configure(values=[''] + list(self.persons))
        self.face_name_box.configure(values=list(self.persons))
        self.vfilters = self.cat.saved_filters()
        sel = self.vlist.curselection()
        self.vlist.delete(0, 'end')
        for name, spec in self.vfilters: self.vlist.insert('end', f'{name}   —   {L.describe(spec)}')
        if sel and sel[0] < len(self.vfilters): self.vlist.selection_set(sel[0])
        self.root_list.delete(0, 'end')
        for r in self.cat.roots(): self.root_list.insert('end', r)

    def collect_spec(self):
        sp = {}
        r = key_of(RATING_FILTER, self.f_rating.get())
        if r == 'unrated': sp['rating_exact'] = 0
        elif r: sp['rating_min'] = r
        tags = [t.strip() for t in self.f_tags.get().split(',') if t.strip()]
        if tags: sp['tags'], sp['tag_mode'] = tags, self.f_tagmode.get()
        for k, v in (('camera', self.f_camera), ('lens', self.f_lens)):
            if v.get(): sp[k] = v.get()
        if self.f_person.get() in self.persons:
            sp['person'], sp['person_name'] = self.persons[self.f_person.get()], self.f_person.get()
        for k, (lo, hi), parse in (('iso', self.f_iso, float), ('aperture', self.f_ap, float), ('focal', self.f_focal, float),
                                   ('exposure', self.f_exp, L.parse_shutter)):
            a, b = num_or_none(lo.get(), parse), num_or_none(hi.get(), parse)
            if a is not None or b is not None: sp[k] = [a, b]
        d0, d1 = self.f_date[0].get().strip(), self.f_date[1].get().strip()
        if d0: sp['date_from'] = d0
        if d1: sp['date_to'] = d1
        g = key_of(GPS_FILTER, self.f_gps.get())
        if g is not None: sp['gps'] = g
        k = key_of(self.kind_filter, self.f_kind.get())
        if k: sp['kind'] = k
        if self.search_mode.get() == SEARCH_MODES[0] and self.search_var.get().strip():
            sp['text'] = self.search_var.get().strip()
        return sp

    def apply_spec(self, sp):
        self._loading_filter = True
        try:
            self.f_rating.set('Unrated' if sp.get('rating_exact') == 0 else label_of(RATING_FILTER, sp.get('rating_min')))
            self.f_tags.set(', '.join(sp.get('tags') or []))
            self.f_tagmode.set(sp.get('tag_mode', 'all'))
            self.f_camera.set(sp.get('camera', '')); self.f_lens.set(sp.get('lens', ''))
            self.f_person.set(sp.get('person_name', '') if sp.get('person_name') in self.persons else '')
            for k, (lo, hi) in (('iso', self.f_iso), ('aperture', self.f_ap), ('focal', self.f_focal), ('exposure', self.f_exp)):
                a, b = (list(sp.get(k) or []) + [None, None])[:2]
                fmt = (lambda v: L.fmt_exposure(v).replace(' s', '')) if k == 'exposure' else (lambda v: f'{v:g}')
                lo.set('' if a is None else fmt(a)); hi.set('' if b is None else fmt(b))
            self.f_date[0].set(sp.get('date_from', '')); self.f_date[1].set(sp.get('date_to', ''))
            self.f_gps.set(label_of(GPS_FILTER, sp.get('gps')))
            kind = sp.get('kind')
            if kind and kind not in [v for _, v in self.kind_filter]:   # a saved filter's type no longer catalogued: still offer it
                self.kind_filter.append((L.kind_label(kind), kind))
                self.f_kind_box.configure(values=[l for l, _ in self.kind_filter])
            self.f_kind.set(label_of(self.kind_filter, kind))
            if self.search_mode.get() == SEARCH_MODES[0]: self.search_var.set(sp.get('text', ''))
        finally:
            self._loading_filter = False
        self.refresh(keep=False)

    def clear_filter(self):
        self.apply_spec({})

    def _filter_changed(self):
        if self._loading_filter: return
        if self._filter_due: self.root.after_cancel(self._filter_due)
        self._filter_due = self.root.after(350, lambda: self.refresh(keep=True))

    def _pick_filter_tag(self):
        t = self.f_tagpick.get()
        self.f_tagpick.set('(add a tag)')
        if not t or t == '(add a tag)': return
        cur = [x.strip() for x in self.f_tags.get().split(',') if x.strip()]
        if t not in cur: self.f_tags.set(', '.join(cur + [t]))

    def refresh(self, keep=True):
        self._filter_due = None
        spec = self.collect_spec()
        try:
            if self.results:
                ranked = [i for i, _ in self.results[1]]
                ids = self.cat.query_ids(spec, None, ids=ranked)
            else:
                ids = self.cat.query_ids(spec, self.scope, self.sort.get(), self.desc.get())
        except Exception as ex:
            self.status.configure(text=f'Filter error: {ex}')
            return
        rows = {k: dict(r) for k, r in self.cat.rows(ids, 'id, name, rating, is_raw, thumb, missing, taken').items()}
        self.grid.empty_text = ('Nothing matches.' if self.cat.roots() else
                                'No photos yet.\n\nAdd folder… (top right) catalogues a folder and its subfolders.\nThe photos stay where they are.')
        self.grid.set_items(ids, rows, keep)
        if spec != self._tree_spec:
            self._tree_spec = spec
            self.refresh_trees()
        what = self.results[0] if self.results else self._scope_label()
        self.status.configure(text=f'{len(ids)} photo{"s" if len(ids) != 1 else ""} · {what} · {L.describe(spec)}')

    def _scope_label(self):
        sc = self.scope
        if not sc: return 'whole library'
        if sc['kind'] == 'folder': return sc['path'] + (' and subfolders' if sc.get('recursive', True) else '')
        return sc.get('prefix') or 'all dates'

    def refresh_trees(self):
        spec = self._tree_spec if isinstance(self._tree_spec, dict) else self.collect_spec()
        # folders, with counts that include subfolders
        counts = self.cat.folder_counts(spec)
        roots, folders = self.cat.roots(), self.cat.folders()
        nc = os.path.normcase
        total = defaultdict(int)
        for f, n in counts.items():
            p = f
            while True:
                total[nc(p)] += n
                q = os.path.dirname(p)
                if q == p: break
                p = q
        rootset = {nc(r) for r in roots}
        kids = defaultdict(list)
        for f in folders:
            if nc(f) not in rootset: kids[nc(os.path.dirname(f))].append(f)
        t = self.ftree
        opened = {i for i in self._all_items(t) if t.item(i, 'open')} if t.get_children() else {'__first__'}
        sel = t.selection()
        self._quiet_tree = True
        t.delete(*t.get_children())
        t.insert('', 'end', iid='__all__', text='All photos', values=(sum(counts.values()),))

        def add(parent, path, label):
            n = total.get(nc(path), 0)
            t.insert(parent, 'end', iid=path, text=label, values=(n or '',), tags=() if n else ('empty',),
                     open=path in opened or ('__first__' in opened and parent == ''))
            for ch in sorted(kids.get(nc(path), []), key=lambda p: os.path.basename(p).lower()):
                add(path, ch, os.path.basename(ch))
        for r in roots:
            add('', r, r)
        # dates: year > month > day, newest first
        dc = self.cat.date_counts(spec)
        d = self.dtree
        dopened = {i for i in self._all_items(d) if d.item(i, 'open')}
        dsel = d.selection()
        d.delete(*d.get_children())
        d.insert('', 'end', iid='d:', text='All dates', values=(sum(n for _, n in dc),))
        years = defaultdict(lambda: defaultdict(list))
        for day, n in dc:
            if day: years[day[:4]][day[5:7]].append((day, n))
        for y in sorted(years, reverse=True):
            d.insert('', 'end', iid=f'd:{y}', text=y, values=(sum(n for m in years[y].values() for _, n in m),), open=f'd:{y}' in dopened)
            for mo in sorted(years[y], reverse=True):
                days = years[y][mo]
                d.insert(f'd:{y}', 'end', iid=f'd:{y}-{mo}', text=calendar.month_name[int(mo)] if mo.isdigit() and 1 <= int(mo) <= 12 else mo,
                         values=(sum(n for _, n in days),), open=f'd:{y}-{mo}' in dopened)
                for day, n in sorted(days, reverse=True):
                    try:
                        wd = calendar.day_abbr[calendar.weekday(int(day[:4]), int(day[5:7]), int(day[8:10]))]
                    except ValueError:
                        wd = ''
                    d.insert(f'd:{y}-{mo}', 'end', iid=f'd:{day}', text=f'{int(day[8:10])} {wd}', values=(n,))
        for tree, s in ((t, sel), (d, dsel)):
            if s and tree.exists(s[0]):
                tree.selection_set(s[0]); tree.see(s[0])
        self.root.after_idle(lambda: setattr(self, '_quiet_tree', False))

    @staticmethod
    def _all_items(t, parent=''):
        out = []
        for i in t.get_children(parent):
            out.append(i); out += App._all_items(t, i)
        return out

    def _scope_changed(self):
        if getattr(self, '_quiet_tree', False): return
        tab = self.browse.index('current')
        scope = None
        if tab == 0:
            s = self.ftree.selection()
            if s and s[0] != '__all__': scope = {'kind': 'folder', 'path': s[0], 'recursive': self.recursive.get()}
        elif tab == 1:
            s = self.dtree.selection()
            if s and s[0] != 'd:': scope = {'kind': 'date', 'prefix': s[0][2:]}
        if scope != self.scope or self.results:
            self.scope = scope
            self.results = None
            self._show_results_label()
            self.refresh(keep=False)

    def _virtual_selected(self):
        s = self.vlist.curselection()
        if not s or s[0] >= len(self.vfilters): return
        self.scope, self.results = None, None
        self._show_results_label()
        self.apply_spec(self.vfilters[s[0]][1])

    def save_virtual(self):
        spec = self.collect_spec()
        name = simpledialog.askstring(APP, f'Name of the virtual folder\n({L.describe(spec)}):', parent=self.root)
        if not name: return
        self.cat.save_filter(name.strip(), spec)
        self.refresh_lists()
        self.browse.select(2)
        self.log('ok', f'virtual folder "{name.strip()}" saved')

    def update_virtual(self):
        s = self.vlist.curselection()
        if not s: return
        name = self.vfilters[s[0]][0]
        self.cat.save_filter(name, self.collect_spec())
        self.refresh_lists()
        self.log('ok', f'virtual folder "{name}" now uses the current filter')

    def delete_virtual(self):
        s = self.vlist.curselection()
        if not s: return
        name = self.vfilters[s[0]][0]
        if messagebox.askyesno(APP, f'Delete the virtual folder "{name}"? (Only the saved filter goes; no photos are affected.)'):
            self.cat.delete_filter(name)
            self.refresh_lists()

    def _folder_menu(self, e):
        iid = self.ftree.identify_row(e.y)
        if not iid or iid == '__all__': return
        self.ftree.selection_set(iid)
        m = self._menu()
        m.add_command(label='Open folder', command=lambda: open_folder(iid))
        m.add_command(label='Rescan this folder', command=lambda: self._rescan([iid]))
        m.tk_popup(e.x_root, e.y_root)

    def _menu(self):
        return tk.Menu(self.root, tearoff=0, bg=C['card2'], fg=C['text'], activebackground=C['accent'], activeforeground='#000',
                       relief='flat', bd=1)

    # ─────────────────────────────── grid
    def zoom_changed(self, z):
        self.zoom_var.set(z)
        self.zoom_lbl.configure(text='one photo' if ZOOMS[z] == 0 else f'{ZOOMS[z]} px')

    def grid_menu(self, e):
        ids = self.grid.selected_ids()
        if not ids: return
        m = self._menu()
        rm = self._menu()
        for k in range(6):
            rm.add_command(label='Unrated' if k == 0 else '★' * k, command=lambda k=k: self.set_rating(self.grid.selected_ids(), k))
        m.add_cascade(label='Rating', menu=rm)
        m.add_command(label='Add tags…', command=lambda: self.tag_box.focus_set())
        m.add_separator()
        m.add_command(label='Open in RAW Demosaic', command=self.open_rawdemosaic)
        m.add_command(label='Batch process in RAW Demosaic…', command=self.batch_rawdemosaic)
        m.add_command(label='Stitch in PANOSTITCH', command=self.open_panostitch)
        m.add_separator()
        m.add_command(label='Open with the default app', command=self.open_default)
        m.add_command(label='Show in Explorer' if WINDOWS else 'Show in folder', command=self.reveal)
        m.add_command(label='Find similar photos (AI)', command=self.find_similar)
        m.add_separator()
        m.add_command(label='Rebuild thumbnails', command=self.rebuild_thumbs)
        m.add_command(label='Run AI again', command=lambda: self.run_ai(self.grid.selected_ids()))
        m.tk_popup(e.x_root, e.y_root)

    def selection_changed(self):
        if self._sel_due: self.root.after_cancel(self._sel_due)
        self._sel_due = self.root.after(60, self._show_selection)

    def _show_selection(self):
        self._sel_due = None
        ids = self.grid.selected_ids()
        n = len(ids)
        self.map_btn.pack_forget()
        self.caption_lbl.pack_forget()
        self._desc_mixed = False
        self.desc_text.delete('1.0', 'end')
        self.tag_list.delete(0, 'end')
        self._tag_names = []
        if not n:
            self.sel_title.configure(text='Nothing selected')
            self.sel_info.configure(text=f'{len(self.grid.ids)} photos shown.\n\nClick selects; Ctrl+click and Shift+click add; drag draws a box. '
                                         f'Keys 0–5 rate, Enter or double-click shows one photo, Ctrl+wheel resizes, right-click for more.')
            self._paint_stars(None)
            self._sel_rows = {}
            return
        rows = self.cat.rows(ids[:20000])
        self._sel_rows = rows
        ratings = {r['rating'] for r in rows.values()}
        self._paint_stars(ratings.pop() if len(ratings) == 1 else None)
        if n == 1:
            r = rows.get(ids[0])
            if r is None: return
            self.sel_title.configure(text=r['name'])
            lines = [r['folder'], '']
            lines.append(f'Taken {r["taken"] or "?"}' + {'file': ' (file date: no EXIF date)', 'datetime': ' (when the file was written: no capture date)',
                                                          'original': ' (from the original it was made from)'}.get(r['taken_src'], ''))
            gear = ' · '.join(x for x in (r['camera'], r['lens']) if x)
            if gear: lines.append(gear)
            exp = [L.fmt_exposure(r['exposure']), f'f/{r["aperture"]:g}' if r['aperture'] else '', f'ISO {r["iso"]}' if r['iso'] else '',
                   (f'{r["focal"]:g} mm' + (f' ({r["focal35"]:g} mm equiv.)' if r['focal35'] and abs(r['focal35'] - r['focal']) > 0.5 else '')) if r['focal'] else '']
            if any(exp): lines.append(' · '.join(x for x in exp if x))
            if r['width']: lines.append(f'{r["width"]} × {r["height"]} ({r["width"] * r["height"] / 1e6:.1f} MP) · {fmt_size(r["size"] or 0)}')
            src = {'preview': 'embedded preview', 'rcd': 'RCD demosaic', 'image': 'the image'}.get(r['thumb_src'], 'not made yet' if r['thumb'] == 0 else 'failed')
            lines.append(('RAW · ' if r['is_raw'] else '') + f'thumbnail from {src}')
            if r['lat'] is not None:
                lines.append(f'GPS {r["lat"]:.5f}, {r["lon"]:.5f}' + (f', {r["alt"]:.0f} m' if r['alt'] is not None else ''))
            faces = self.cat.q('SELECT p.name FROM faces f LEFT JOIN persons p ON p.id = f.person_id WHERE f.photo_id = ?', (ids[0],))
            if faces:
                named = sorted({f['name'] for f in faces if f['name']})
                lines.append(f'{len(faces)} face{"s" if len(faces) != 1 else ""}' + (': ' + ', '.join(named) if named else ''))
            if r['missing']: lines.append('⚠ The file is no longer there (moved or deleted outside the app).')
            self.sel_info.configure(text='\n'.join(lines))
            if r['lat'] is not None: self.map_btn.pack(anchor='w', pady=(6, 0))
            if r['caption']:
                self.caption_lbl.configure(text='AI caption: ' + r['caption'])
                self.caption_lbl.pack(anchor='w', pady=(6, 0))
            self.desc_text.insert('1.0', r['description'] or '')
        else:
            self.sel_title.configure(text=f'{n} photos selected')
            dates = sorted(r['taken'] for r in rows.values() if r['taken'])
            raw = sum(r['is_raw'] for r in rows.values())
            lines = []
            if dates: lines.append(f'{dates[0][:10]} → {dates[-1][:10]}')
            lines.append(f'{raw} RAW, {n - raw} other')
            cams = sorted({r['camera'] for r in rows.values() if r['camera']})
            if cams: lines.append(', '.join(cams[:4]) + (' …' if len(cams) > 4 else ''))
            lines.append('\nRating, tags and description apply to all of them.')
            self.sel_info.configure(text='\n'.join(lines))
            descs = {r['description'] for r in rows.values()}
            if len(descs) == 1:
                self.desc_text.insert('1.0', descs.pop())
            else:
                self._desc_mixed = True
                self.desc_text.insert('1.0', '(different on these photos — saving sets them all to what is written here)')
        self._tag_people = set()
        for name, cnt, src in self.cat.tag_counts(ids):
            self._tag_names.append(name)
            if src == {'person'}: self._tag_people.add(name)
            self.tag_list.insert('end', f'{name}' + (f'   {cnt}/{n}' if n > 1 else '') +
                                 {frozenset({'ai'}): '   · AI', frozenset({'person'}): '   · person'}.get(frozenset(src), ''))

    def _paint_stars(self, r):
        for k, lb in enumerate(self.stars, 1):
            lb.configure(fg=C['warn'] if r and k <= r else C['star_off'])

    def _star_clicked(self, k):
        ids = self.grid.selected_ids()
        if not ids: return
        cur = {r['rating'] for r in self._sel_rows.values()}
        self.set_rating(ids, 0 if cur == {k} else k)

    def set_rating(self, ids, r):
        self.cat.set_rating(ids, r)
        for pid in ids:
            if pid in self.grid.rows: self.grid.rows[pid]['rating'] = r
        self.grid.request_redraw()
        if set(ids) & self.grid.sel: self.selection_changed()
        self.status.configure(text=f'{len(ids)} photo{"s" if len(ids) != 1 else ""} rated {r}★' if r else f'{len(ids)} unrated')

    def add_tags(self):
        ids = self.grid.selected_ids()
        names = [t.strip() for t in self.tag_add.get().split(',') if t.strip()]
        if not ids or not names: return
        self.cat.add_tags(ids, names)
        self.tag_add.set('')
        self.refresh_lists()
        self.selection_changed()
        self.log('ok', f'tagged {len(ids)} photo{"s" if len(ids) != 1 else ""}: {", ".join(names)}')

    def remove_tags(self):
        ids = self.grid.selected_ids()
        names = [self._tag_names[i] for i in self.tag_list.curselection()]
        if not ids or not names: return
        people = [n for n in names if n in self._tag_people]
        if people:
            messagebox.showinfo(APP, f'{", ".join(people)}: a person\'s name comes from the faces named on the People tab. '
                                     'To take it off a photo, select the face there and press "Not this person".')
        self.cat.remove_tags(ids, names)
        self.refresh_lists()
        self.selection_changed()

    def _filter_by_tag(self):
        s = self.tag_list.curselection()
        if s: self.f_tags.set(self._tag_names[s[0]])

    def save_description(self):
        ids = self.grid.selected_ids()
        if not ids: return
        text = self.desc_text.get('1.0', 'end').strip()
        if self._desc_mixed and text.startswith('(different on these photos'): return
        self.cat.set_description(ids, text)
        self.log('ok', f'description saved for {len(ids)} photo{"s" if len(ids) != 1 else ""}')
        self.selection_changed()

    def use_caption(self):
        ids = self.grid.selected_ids()
        rows = [r for r in self._sel_rows.values() if r['caption']]
        if not rows: return
        with self.cat.tx() as c:
            c.executemany("UPDATE photos SET description = caption WHERE id = ? AND caption != ''", [(r['id'],) for r in rows])
        self.log('ok', f'AI caption copied to the description of {len(rows)} photo{"s" if len(rows) != 1 else ""}')
        self.selection_changed()

    def open_map(self):
        ids = self.grid.selected_ids()
        r = self._sel_rows.get(ids[0]) if ids else None
        if r and r['lat'] is not None:
            import webbrowser
            webbrowser.open(f'https://www.openstreetmap.org/?mlat={r["lat"]:.6f}&mlon={r["lon"]:.6f}#map=15/{r["lat"]:.6f}/{r["lon"]:.6f}')

    def _paths(self, ids=None):
        ids = self.grid.selected_ids() if ids is None else ids
        rows = self.cat.rows(ids, 'id, path, missing')
        return [rows[i]['path'] for i in ids if i in rows and not rows[i]['missing']]

    def reveal(self):
        p = self._paths()
        if p: show_in_explorer(p[0])

    def open_default(self):
        for p in self._paths()[:10]:
            if WINDOWS: os.startfile(p)
            else: subprocess.Popen(['xdg-open', p])

    # ─────────────────────────────── search
    def search(self):
        text = self.search_var.get().strip()
        if self.search_mode.get() == SEARCH_MODES[0]:
            self.refresh(keep=False); return
        if not text: return
        self.status.configure(text=f'Searching for "{text}" (the first search loads SigLIP2)…')
        ai = self._ai()

        def work():
            try:
                hits = ai.search_text(text)
                self.q.put(('call', self._show_results, f'meaning: "{text}"', hits))
            except Exception as ex:
                self.q.put(('call', self._search_failed, ex))
        threading.Thread(target=work, daemon=True).start()

    def find_similar(self):
        ids = self.grid.selected_ids()
        if not ids: return
        hits = self._ai().search_similar(ids[0])
        if not hits:
            messagebox.showinfo(APP, 'This photo has no image embedding yet — run the AI on it (Settings & AI).'); return
        name = (self.grid.rows.get(ids[0]) or {}).get('name', '')
        self._show_results(f'similar to {name}', hits)

    def _search_failed(self, ex):
        self.status.configure(text=f'Search failed: {ex}')
        self.log('err', f'search: {ex}')

    def _show_results(self, label, hits):
        self.results = (label, hits)
        self._show_results_label()
        self.nb.select(0)
        self.refresh(keep=False)
        if not hits: self.log('warn', f'{label}: nothing found (have image embeddings been made? Settings & AI)')

    def _show_results_label(self):
        if self.results:
            self.results_lbl.configure(text=f'{len(self.results[1])} results · {self.results[0]}')
            self.results_clear.pack(side='left', padx=(6, 0))
        else:
            self.results_lbl.configure(text='')
            self.results_clear.pack_forget()

    def clear_results(self):
        self.results = None
        self._show_results_label()
        self.refresh(keep=False)

    # ─────────────────────────────── adding and scanning
    def add_folder(self):
        d = filedialog.askdirectory(initialdir=self.settings.get('add_dir'), title='Folder to add to the library (with its subfolders)')
        if not d: return
        d = os.path.normpath(d)
        self.settings['add_dir'] = os.path.dirname(d)
        lib = os.path.normcase(self.cat.dir)
        if os.path.normcase(d) == lib or lib.startswith(os.path.normcase(d).rstrip(os.sep) + os.sep):
            self.log('info', 'the library\'s own folder (thumbnails) is skipped while scanning')
        root = self.cat.add_root(d)
        self.refresh_lists()
        self._rescan([root] if os.path.normcase(root) == os.path.normcase(d) else [d])

    def rescan_all(self):
        roots = self.cat.roots()
        if not roots:
            self.add_folder(); return
        self._rescan(roots)

    def rescan_selected(self):
        roots = [self.root_list.get(i) for i in self.root_list.curselection()]
        if roots: self._rescan(roots)

    def _rescan(self, folders):
        min_prev, rd = self.min_preview(), self.rd_dir.get().strip()
        faces, embed, caption, auto = self.ai_faces.get(), self.ai_embed.get(), self.ai_caption.get(), self.ai_auto.get()
        ai = self._ai() if auto and (faces or embed or caption) else None

        def fn(job):
            new = []
            for f in folders:
                ids = self.cat.scan(f, job)
                job.post('call', self.refresh_all)
                self.cat.make_thumbnails(ids, job, min_prev, rd)
                new += ids
            if ai and new:
                have = ai.present()
                f, e, c = faces and have['faces'], embed and have['siglip'], caption and have['florence']
                want = (L.AI_FACES if f else 0) | (L.AI_EMBED if e else 0) | (L.AI_CAPTION if c else 0)
                if want:
                    rows = self.cat.rows(new, 'id, thumb, ai')
                    todo = [i for i in new if i in rows and rows[i]['thumb'] == 1 and rows[i]['ai'] & want != want]
                    ai.run(todo, job, f, e, c)
                else:
                    job.log('warn', 'AI skipped: its models are not downloaded yet (Settings & AI)')
            return len(new)
        self.run_task('Scanning', fn, done=lambda n: (self.refresh_all(), self.refresh_settings_info()))

    def remove_roots(self):
        roots = [self.root_list.get(i) for i in self.root_list.curselection()]
        if not roots: return
        n = sum(self.cat.q1('SELECT COUNT(*) FROM photos WHERE ' + L._under('path', r)[0], L._under('path', r)[1])[0] for r in roots)
        if not messagebox.askyesno(APP, f'Remove {", ".join(roots)} from the library?\n\nThe catalog forgets {n} photos — their ratings, tags, '
                                        f'descriptions, faces and AI data. The files on disk are not touched.'):
            return
        for r in roots: self.cat.remove_root(r)
        self.scope = None
        self.refresh_all()
        self.log('ok', f'removed {len(roots)} folder{"s" if len(roots) != 1 else ""} from the library ({n} photos forgotten)')

    def forget_missing(self):
        n = self.cat.q1('SELECT COUNT(*) FROM photos WHERE missing = 1')[0]
        if not n:
            messagebox.showinfo(APP, 'No catalogued photo is missing.'); return
        if messagebox.askyesno(APP, f'{n} catalogued photos are no longer on disk. Forget them (and their ratings and tags)?'):
            self.cat.forget_missing()
            self.refresh_all()

    def rebuild_thumbs(self):
        ids = self.grid.selected_ids()
        if not ids: return
        with self.cat.tx() as c:
            c.executemany('UPDATE photos SET thumb = 0 WHERE id = ?', [(i,) for i in ids])
        for i in ids:
            if i in self.grid.rows: self.grid.rows[i]['thumb'] = 0
        self.grid.invalidate(ids)
        min_prev, rd = self.min_preview(), self.rd_dir.get().strip()
        self.run_task('Thumbnails', lambda job: self.cat.make_thumbnails(ids, job, min_prev, rd))

    # ─────────────────────────────── AI
    def _ai(self):
        md, dev = self.models_dir.get().strip() or AI.DEFAULT_MODELS, self.device_key()
        if self.ai is None: self.ai = AI.AI(self.cat, md, dev)
        else: self.ai.configure(md, dev)
        return self.ai

    def run_ai(self, ids=None):
        f, e, c = self.ai_faces.get(), self.ai_embed.get(), self.ai_caption.get()
        if not (f or e or c):
            messagebox.showinfo(APP, 'Tick at least one of faces, embeddings or captions (Settings & AI).'); return
        ai = self._ai()
        have = ai.present()
        missing = [n for n, on, k in (('faces', f, 'faces'), ('SigLIP2', e, 'siglip'), ('Florence-2', c, 'florence')) if on and not have[k]]
        if missing:
            if messagebox.askyesno(APP, f'Not downloaded yet: {", ".join(missing)}. Download the missing models now?'):
                self.download_models()
            return
        want = (L.AI_FACES if f else 0) | (L.AI_EMBED if e else 0) | (L.AI_CAPTION if c else 0)

        def fn(job):
            if ids is None:
                todo = [r[0] for r in self.cat.q('SELECT id FROM photos WHERE thumb = 1 AND missing = 0 AND (ai & ?) != ? ORDER BY id', (want, want))]
            else:
                rows = self.cat.rows(ids, 'id, thumb')
                todo = [i for i in ids if i in rows and rows[i]['thumb'] == 1]
            job.log('info', f'AI on {len(todo)} photos')
            ai.run(todo, job, f, e, c)
            return len(todo)
        self.run_task('AI', fn, done=lambda n: (self.refresh_lists(), self.refresh_settings_info(), self.selection_changed()))

    def download_models(self):
        md = self.models_dir.get().strip() or AI.DEFAULT_MODELS
        which = [k for k, on in (('faces', self.ai_faces.get()), ('siglip', self.ai_embed.get()), ('florence', self.ai_caption.get())) if on] \
            or ['faces', 'siglip', 'florence']
        ok = False
        if 'faces' in which and not AI.SM.present(md)['faces']:
            ok = messagebox.askyesno(APP, AI.SM.FACES_NOTICE + '\n\nDownload them for non-commercial use?\n'
                                     '(No: the other models are still downloaded; People then stays empty.)')
        self.run_task('Downloading models', lambda job: AI.SM.download(md, which, job.log, accept_noncommercial=ok),
                      done=lambda r: self.refresh_settings_info())

    def free_gpu(self):
        if self.ai: self.ai.unload()
        self.log('info', 'AI models unloaded')

    def refresh_settings_info(self):
        have = AI.SM.present(self.models_dir.get().strip() or AI.DEFAULT_MODELS)
        self.models_info.configure(text='\n'.join(f'{"✓" if have[k] else "✗"}  {label}' for k, label in
                                                  (('faces', 'faces — buffalo_l (SCRFD + ArcFace)'), ('siglip', 'search by meaning — SigLIP2'),
                                                   ('florence', 'captions — Florence-2'))))
        st = self.cat.stats()
        nf = self.cat.q1('SELECT COUNT(*) FROM faces')[0]
        npers = self.cat.q1('SELECT COUNT(*) FROM persons')[0]
        self.stats_lbl.configure(text=f'{st["n"]} photos ({st["raw"]} RAW) in {len(self.cat.roots())} folder{"s" if len(self.cat.roots()) != 1 else ""}\n'
                                      f'{st["th"]} with thumbnails, {st["miss"]} missing on disk\n'
                                      f'AI: {st["f"]} searched for faces ({nf} found, {npers} people named), {st["e"]} embedded, {st["c"]} captioned\n'
                                      f'Catalog: {self.cat.db_path}')

    # ─────────────────────────────── people
    def refresh_people(self):
        self.cat.sync_person_tags()   # after every naming, confirming, renaming or "not this person"
        self.selection_changed()
        self._people = self.cat.q('SELECT p.id, p.name, COUNT(f.id) AS n, SUM(f.assigned = \'auto\') AS auto FROM persons p '
                                  'LEFT JOIN faces f ON f.person_id = p.id GROUP BY p.id ORDER BY p.name COLLATE NOCASE')
        self.p_list.delete(0, 'end')
        for r in self._people:
            self.p_list.insert('end', f'{r["name"]}   ({r["n"]}' + (f', {r["auto"]} to check)' if r['auto'] else ')'))
        n_un = self.cat.q1('SELECT COUNT(*) FROM faces WHERE person_id IS NULL')[0]
        self._groups = [(None, n_un)] + [(r[0], r[1]) for r in self.cat.q(
            'SELECT cluster, COUNT(*) AS n FROM faces WHERE person_id IS NULL AND cluster IS NOT NULL GROUP BY cluster HAVING n > 1 ORDER BY n DESC LIMIT 1000')]
        self.g_list.delete(0, 'end')
        for cid, n in self._groups:
            self.g_list.insert('end', f'All unnamed faces   ({n})' if cid is None else f'Group of {n}')
        self.refresh_lists()
        if self._face_source: self._load_faces(*self._face_source)

    def _people_selected(self, kind):
        lb = self.p_list if kind == 'person' else self.g_list
        s = lb.curselection()
        if not s: return
        (self.g_list if kind == 'person' else self.p_list).selection_clear(0, 'end')
        if kind == 'person':
            r = self._people[s[0]]
            self.face_name.set(r['name'])
            self._load_faces('person', r['id'], r['name'])
        else:
            cid, n = self._groups[s[0]]
            self._load_faces('group', cid, 'All unnamed faces' if cid is None else f'Group of {n}')

    def _load_faces(self, kind, key, title):
        self._face_source = (kind, key, title)
        if kind == 'person':
            rows = self.cat.q("SELECT id, photo_id, x0, y0, x1, y1, assigned FROM faces WHERE person_id = ? ORDER BY assigned = 'auto' DESC, id", (key,))
        elif key is None:
            rows = self.cat.q('SELECT id, photo_id, x0, y0, x1, y1, assigned FROM faces WHERE person_id IS NULL ORDER BY cluster, id LIMIT 20000')
        else:
            rows = self.cat.q('SELECT id, photo_id, x0, y0, x1, y1, assigned FROM faces WHERE person_id IS NULL AND cluster = ? ORDER BY id', (key,))
        self.face_title.configure(text=f'{title} — {len(rows)} face{"s" if len(rows) != 1 else ""}')
        self.fgrid.set_items([r['id'] for r in rows], {r['id']: dict(r) for r in rows})

    def name_faces(self):
        ids = self.fgrid.selected_ids()
        name = self.face_name.get().strip()
        if not ids:
            messagebox.showinfo(APP, 'Select the faces of one person first (click, Ctrl+click, Shift+click or drag).'); return
        if not name:
            messagebox.showinfo(APP, 'Type the person\'s name in the box first.'); return
        AI.name_faces(self.cat, ids, name)
        n = AI.suggest_people(self.cat)
        self.log('ok', f'{len(ids)} face{"s" if len(ids) != 1 else ""} named {name}' + (f'; {n} other faces suggested (marked ?)' if n else ''))
        self.refresh_people()

    def confirm_faces(self):
        ids = self.fgrid.selected_ids() or [i for i in self.fgrid.ids if self.fgrid.rows[i].get('assigned') == 'auto']
        with self.cat.tx() as c:
            c.executemany("UPDATE faces SET assigned = 'user' WHERE id = ? AND person_id IS NOT NULL", [(i,) for i in ids])
        self.refresh_people()

    def unname_faces(self):
        ids = self.fgrid.selected_ids()
        if not ids: return
        AI.unname_faces(self.cat, ids)
        self.refresh_people()

    def rename_person(self):
        s = self.p_list.curselection()
        if not s: return
        r = self._people[s[0]]
        name = simpledialog.askstring(APP, 'New name:', initialvalue=r['name'], parent=self.root)
        if not name or name.strip() == r['name']: return
        other = self.cat.q1('SELECT id FROM persons WHERE name = ? AND id != ?', (name.strip(), r['id']))
        with self.cat.tx() as c:
            if other:   # the same person under two names: merge
                c.execute('UPDATE faces SET person_id = ? WHERE person_id = ?', (other[0], r['id']))
                c.execute('DELETE FROM persons WHERE id = ?', (r['id'],))
            else:
                c.execute('UPDATE persons SET name = ? WHERE id = ?', (name.strip(), r['id']))
        self.refresh_people()

    def person_photos(self):
        s = self.p_list.curselection()
        if not s: return
        self.scope, self.results = None, None
        self._show_results_label()
        self.apply_spec({'person': self._people[s[0]]['id'], 'person_name': self._people[s[0]]['name']})
        self.nb.select(0)

    def show_face_photo(self, fid):
        r = self.fgrid.rows.get(fid)
        if not r: return
        self._show_results('a face\'s photo', [(r['photo_id'], 1.0)])
        self.grid.select_ids([r['photo_id']])
        self.grid.toggle_single(0)

    def group_faces(self):
        self.run_task('Grouping faces', lambda job: AI.group_faces(self.cat),
                      done=lambda n: (self.log('ok', f'unnamed faces in {n} groups (single faces included)'), self.refresh_people()))

    def suggest_names(self):
        n = AI.suggest_people(self.cat)
        self.log('ok', f'{n} unnamed faces matched to people you have named (marked ?)' if n else 'no new matches')
        self.refresh_people()

    # ─────────────────────────────── other apps
    def _send_list(self, app, paths):
        d = os.path.join(self.cat.dir, 'send')
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f'{app}_{time.strftime("%Y%m%d_%H%M%S")}.txt')
        with open(p, 'w', encoding='utf-8') as f:
            f.write('\n'.join(paths))
        return p

    def _clean_send_lists(self):
        d = os.path.join(self.cat.dir, 'send')
        try:
            for f in os.listdir(d):
                p = os.path.join(d, f)
                if time.time() - os.path.getmtime(p) > 86400: os.remove(p)
        except OSError:
            pass

    def _launch(self, name, folder, script, pythons, args):
        target = os.path.join(folder, script)
        if not os.path.isfile(target):
            messagebox.showerror(APP, f'{script} is not in {folder} — set {name}\'s folder in Settings & AI.'); return False
        if L.IN_SUITE:   # every app shares the release folder's environment
            pythons = [os.path.join(L.SUITE, '.venv', 'Scripts', 'pythonw.exe') if WINDOWS else os.path.join(L.SUITE, '.venv-linux', 'bin', 'python')]
        py = next((p for p in pythons if os.path.isfile(p)), None)
        if not py:
            messagebox.showerror(APP, f'{name}\'s Python environment was not found (looked for {pythons[0]}). Run its installer first.'); return False
        subprocess.Popen([py, target] + args, cwd=folder)
        return True

    def _rawdemosaic(self, extra):
        paths = self._paths()
        if not paths: return
        d = self.rd_dir.get().strip() or L.DEFAULT_RAWDEMOSAIC
        app_root = os.path.dirname(d)
        pys = ([os.path.join(app_root, '.venv', 'Scripts', 'pythonw.exe'), os.path.join(d, '.venv', 'Scripts', 'pythonw.exe')] if WINDOWS else
               [os.path.join(app_root, '.venv-linux', 'bin', 'python'), os.path.join(app_root, '.venv', 'bin', 'python')])
        if self._launch('RAW Demosaic', d, 'rawdemosaic_desktop.py', pys, extra + ['--list', self._send_list('rawdemosaic', paths)]):
            self.log('ok', f'{len(paths)} file{"s" if len(paths) != 1 else ""} sent to RAW Demosaic' + (' for batch processing' if '--process' in extra else ''))

    def open_rawdemosaic(self):
        self._rawdemosaic([])

    def batch_rawdemosaic(self):
        paths = self._paths()
        if not paths: return
        cur = self.batch_json.get().strip()
        p = filedialog.askopenfilename(filetypes=[('Settings', '*.json')], initialdir=os.path.dirname(cur) if cur else None,
                                       initialfile=os.path.basename(cur) if cur else None, title='RAW Demosaic settings for this batch')
        if not p: return
        self.batch_json.set(os.path.normpath(p))
        if not messagebox.askyesno(APP, f'Process {len(paths)} file{"s" if len(paths) != 1 else ""} in RAW Demosaic with {os.path.basename(p)}?\n\n'
                                        'RAW Demosaic opens with them loaded and these settings applied, and starts "Process all". '
                                        'Results are saved where the settings say (next to each photo when they name no output folder); '
                                        'rescan here afterwards to catalogue them.'):
            return
        self._rawdemosaic(['--settings', os.path.normpath(p), '--process'])

    def _developed(self, path):
        """For a RAW file, the newest image beside it whose name starts with the RAW's (e.g. _DSC4471.tif,
        _DSC4471_developed.tif) — PANOSTITCH reads images, not RAW files."""
        d, stem = os.path.dirname(path), os.path.splitext(os.path.basename(path))[0].lower()
        try:
            cands = [os.path.join(d, f) for f in os.listdir(d)
                     if f.lower().endswith(PANO_EXTS) and os.path.splitext(f)[0].lower().startswith(stem)
                     and not os.path.splitext(f)[0].lower().startswith(stem + '_col') and '_preview' not in f.lower()]
        except OSError:
            return None
        if not cands: return None
        return max(cands, key=lambda p: (os.path.splitext(p)[0].lower() == os.path.join(d, stem).lower(),
                                         os.path.splitext(p)[1].lower() in ('.tif', '.tiff'), os.path.getmtime(p)))

    def open_panostitch(self):
        paths = self._paths()
        if not paths: return
        out, raw_missing = [], []
        for p in paths:
            if p.lower().endswith(L.RAW_EXTS):
                dev = self._developed(p)
                if dev: out.append(dev)
                else: raw_missing.append(os.path.basename(p))
            else:
                out.append(p)
        out = list(dict.fromkeys(out))
        if raw_missing:
            msg = (f'PANOSTITCH reads images, not RAW files. {len(raw_missing)} RAW file{"s have" if len(raw_missing) != 1 else " has"} no developed '
                   f'image beside {"them" if len(raw_missing) != 1 else "it"} ({", ".join(raw_missing[:5])}{" …" if len(raw_missing) > 5 else ""}).\n\n'
                   'Develop them first (Batch process in RAW Demosaic), rescan, then send them again.')
            if len(out) < 2:
                messagebox.showinfo(APP, msg); return
            if not messagebox.askyesno(APP, msg + f'\n\nSend the other {len(out)} images now?'): return
        if len(out) < 2:
            messagebox.showinfo(APP, 'Select at least two photos to stitch.'); return
        d = self.ps_dir.get().strip() or DEFAULT_PANOSTITCH
        pys = [os.path.join(d, '.venv', 'Scripts', 'pythonw.exe')] if WINDOWS else [os.path.join(d, '.venv-linux', 'bin', 'python')]
        if self._launch('PANOSTITCH', d, 'panostitch_desktop.py', pys, ['--list', self._send_list('panostitch', out)]):
            self.log('ok', f'{len(out)} images sent to PANOSTITCH')

    # ─────────────────────────────── shutdown
    def on_close(self):
        if self.busy and not messagebox.askyesno(APP, 'Something is still running. Quit anyway?'):
            return
        if self.job: self.job.cancel_event.set()
        self._save_settings()
        self.root.destroy()


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)   # crisp text on scaled displays
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
