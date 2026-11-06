# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Photo Library catalog: the SQLite database, folder scanning, EXIF metadata and thumbnails.

The photos themselves are only ever read — never written, moved or deleted. Thumbnails (128, 512 and
1024 px on the long side, sRGB JPEG) live in the library folder, keyed by photo id. A RAW file's
thumbnail comes from its embedded JPEG preview when that is big enough, otherwise from a quick RCD
demosaic through RAW Demosaic's own engine (rawdemosaic_engine), the same pipeline that app uses.
"""
import io
import json
import logging
import math
import os
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime

import numpy as np
from PIL import Image, ImageCms, ImageOps

import focuspoints as FP

HERE = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(HERE)
DEFAULT_LIBRARY = os.path.join(APP_ROOT, 'library')
DEFAULT_RAWDEMOSAIC = os.path.normpath(os.path.join(APP_ROOT, '..', 'RAWDemosaic', 'desktop'))
FALLBACK_RAWDEMOSAIC = os.path.join(APP_ROOT, 'Example', 'desktop')
# installed with the other desktop apps (a release folder holding suite.json): one shared Python environment
SUITE = os.path.dirname(APP_ROOT)
IN_SUITE = os.path.isfile(os.path.join(SUITE, 'suite.json'))

RAW_EXTS = ('.nef', '.nrw', '.cr2', '.cr3', '.crw', '.arw', '.srf', '.sr2', '.raf', '.dng', '.orf', '.rw2', '.pef',
            '.srw', '.x3f', '.3fr', '.fff', '.iiq', '.mos', '.mef', '.mrw', '.erf', '.kdc', '.dcr', '.raw', '.rwl', '.gpr')
IMAGE_EXTS = ('.jpg', '.jpeg', '.jfif', '.png', '.tif', '.tiff', '.bmp', '.webp')
SUPPORTED_EXTS = RAW_EXTS + IMAGE_EXTS
THUMB_SIZES = (128, 512, 1024)
# thumbnail formats: file extension, Pillow format, save options. At about the same fidelity as JPEG 88 (PSNR to the
# original, measured on this library) WebP 85 is ~35% smaller and AVIF 65-70 ~45-50% smaller; both decode in < 20 ms.
THUMB_FORMATS = {'jpeg': ('.jpg', 'JPEG', dict(optimize=True, progressive=True)),
                 'webp': ('.webp', 'WEBP', dict(method=6)),
                 'avif': ('.avif', 'AVIF', dict(speed=6))}
THUMB_QUALITY = {'jpeg': 88, 'webp': 85, 'avif': 65}   # the defaults; thumbnails written before this were all jpeg88
SKIP_DIRS = {'.venv', '.venv-linux', '__pycache__', '.git', '$recycle.bin', 'system volume information'}

logging.getLogger('exifread').setLevel(logging.CRITICAL)   # it logs "File format not recognized" for RAF, CR3 …
Image.MAX_IMAGE_PIXELS = None                               # panoramas are big on purpose


def norm(p):
    return os.path.normpath(os.path.abspath(p))


# ───────────────────────────────────────────── RAW Demosaic's engine
_E = None
_E_lock = threading.Lock()


def engine(rawdemosaic_dir=None):
    """rawdemosaic_engine (and its Numba demosaics), imported from the RAW Demosaic app's own folder."""
    global _E
    with _E_lock:
        if _E is None:
            # this app's own Numba cache, so it and RAW Demosaic (another environment) don't overwrite each other's
            os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(HERE, '__pycache__', 'numba'))
            for d in (rawdemosaic_dir, DEFAULT_RAWDEMOSAIC, FALLBACK_RAWDEMOSAIC):
                if d and os.path.isfile(os.path.join(d, 'rawdemosaic_engine.py')):
                    if d not in sys.path: sys.path.append(d)   # appended, so it never shadows this app's modules
                    import rawdemosaic_engine
                    _E = rawdemosaic_engine
                    break
            else:
                raise RuntimeError('RAW Demosaic\'s engine (rawdemosaic_engine.py) was not found — set its folder in Settings.')
        return _E


# ───────────────────────────────────────────── jobs
class Cancelled(Exception):
    pass


class DiskFull(Exception):
    def __init__(self, folder):
        super().__init__(f'The disk holding the thumbnails is full ({folder}), so making them stopped. Free some space, '
                         'then Settings & AI → Thumbnails → Rebuild thumbnails from the originals… → Missing ones makes the rest.')


def _disk_full(ex):
    import errno
    return isinstance(ex, OSError) and ex.errno == errno.ENOSPC


class Job:
    """Progress/log sink handed to long operations. All methods are safe to call from a worker thread."""
    def __init__(self, log=None, progress=None, post=None):
        self._log, self._progress, self._post = log, progress, post
        self.cancel_event = threading.Event()

    def log(self, level, msg):
        if self._log: self._log(level, msg)

    def progress(self, frac, label=None):
        if self._progress: self._progress(frac, label)

    def post(self, *msg):
        """Send a message to the UI thread (e.g. ('thumbs', ids) so the grid can redraw them)."""
        if self._post: self._post(*msg)

    def check(self):
        if self.cancel_event.is_set(): raise Cancelled()


# ───────────────────────────────────────────── metadata
def _num(tag):
    """An exifread tag's first value as a float (None if missing or x/0)."""
    if tag is None: return None
    v = tag.values[0] if isinstance(tag.values, (list, tuple)) and tag.values else tag.values
    try:
        if hasattr(v, 'denominator') and v.denominator == 0: return None
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _str(tag):
    if tag is None: return ''
    s = str(tag.printable if hasattr(tag, 'printable') else tag).replace('\x00', '').strip()
    return '' if set(s) <= set('- ') else s


def _gps(tags):
    def deg(t, ref):
        if t is None: return None
        try:
            d, m, s = (float(v) if getattr(v, 'denominator', 1) else 0.0 for v in t.values[:3])
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        x = d + m / 60 + s / 3600
        return -x if _str(ref).upper()[:1] in ('S', 'W') else x
    lat = deg(tags.get('GPS GPSLatitude'), tags.get('GPS GPSLatitudeRef'))
    lon = deg(tags.get('GPS GPSLongitude'), tags.get('GPS GPSLongitudeRef'))
    if lat is None or lon is None or (lat == 0 and lon == 0): return None, None, None
    alt = _num(tags.get('GPS GPSAltitude'))
    if alt is not None and _num(tags.get('GPS GPSAltitudeRef')) == 1: alt = -alt
    return lat, lon, alt


def _exif_date(s):
    s = (s or '').strip().replace('\x00', '')
    for fmt in ('%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y:%m:%d %H:%M', '%Y-%m-%dT%H:%M:%S'):
        try:
            d = datetime.strptime(s[:19], fmt)
            if d.year > 1900: return d.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            pass
    return None


def exif_taken(tags):
    """exifread tags -> (date taken, source). 'exif' is the capture time; 'datetime' is only the TIFF DateTime tag,
    which editors set to when they wrote the file — a developed copy then takes its original's date instead."""
    for k in ('EXIF DateTimeOriginal', 'Image DateTimeOriginal', 'EXIF DateTimeDigitized'):
        d = _exif_date(_str(tags.get(k)))
        if d: return d, 'exif'
    d = _exif_date(_str(tags.get('Image DateTime')))
    return (d, 'datetime') if d else (None, None)


def camera_name(make, model):
    """'NIKON CORPORATION' + 'NIKON D810' -> 'NIKON D810'; 'FUJIFILM' + 'X-T3' -> 'FUJIFILM X-T3'."""
    make, model = (make or '').strip(), (model or '').strip()
    if not model: return make
    first = make.split()[0] if make else ''
    return model if not first or first.lower() in model.lower() else f'{first} {model}'


def read_metadata(path, st=None):
    """EXIF of a photo -> dict of catalog columns. exifread first; for RAW files whose container it can't
    parse (RAF, CR3 …) the embedded JPEG's EXIF; rawpy (LibRaw) fills what is still missing."""
    import exifread
    st = st or os.stat(path)
    ext = os.path.splitext(path)[1].lower()
    is_raw = ext in RAW_EXTS
    m = dict(size=st.st_size, mtime=st.st_mtime, is_raw=int(is_raw), width=None, height=None, taken=None, taken_src=None,
             make='', model='', lens='', focal=None, focal35=None, aperture=None, exposure=None, iso=None,
             lat=None, lon=None, alt=None, focus='')
    tags = {}
    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False) or {}
    except Exception:
        tags = {}
    raw, exif_ok = None, len(tags) >= 5
    hl = _headerless(path) if is_raw else None
    if hl:   # a bare sensor dump (SJCAM .RAW): LibRaw mistakes it for another format, and there is no EXIF
        m['make'], m['model'], m['width'], m['height'] = hl['make'], hl['model'], hl['W'], hl['H']
        m['taken'], m['taken_src'] = datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S'), 'file'
        m['camera'] = camera_name(m['make'], m['model'])
        return m
    if is_raw:
        try:
            import rawpy
            raw = rawpy.imread(path)
            if len(tags) < 5:   # e.g. Fuji RAF: read the EXIF the camera wrote into the embedded JPEG
                try:
                    t = raw.extract_thumb()
                    if t.format == rawpy.ThumbFormat.JPEG:
                        tags = exifread.process_file(io.BytesIO(t.data), details=True) or tags   # maker note: Fuji's focus point
                except Exception:
                    pass
        except Exception:
            raw = None

    def g(*keys):
        for k in keys:
            if k in tags and _str(tags[k]): return tags[k]
        return None
    m['make'] = _str(g('Image Make'))
    m['model'] = _str(g('Image Model'))
    m['lens'] = _str(g('EXIF LensModel', 'Image LensModel', 'MakerNote LensModel', 'EXIF LensSpecification'))
    m['focal'] = _num(g('EXIF FocalLength', 'Image FocalLength'))
    m['focal35'] = _num(g('EXIF FocalLengthIn35mmFilm', 'Image FocalLengthIn35mmFilm'))
    m['aperture'] = _num(g('EXIF FNumber', 'Image FNumber'))
    m['exposure'] = _num(g('EXIF ExposureTime', 'Image ExposureTime'))
    iso = _num(g('EXIF ISOSpeedRatings', 'Image ISOSpeedRatings', 'EXIF PhotographicSensitivity'))
    m['iso'] = int(iso) if iso else None
    m['taken'], m['taken_src'] = exif_taken(tags)
    if is_raw and m['taken_src'] == 'datetime': m['taken_src'] = 'exif'   # a camera's own file: DateTime is the capture
    m['lat'], m['lon'], m['alt'] = _gps(tags)
    orientation = int(_num(tags.get('Image Orientation')) or 1)

    flip = None
    if raw is not None:
        try:
            s = raw.sizes
            flip = s.flip
            w, h = int(s.width), int(s.height)
            m['width'], m['height'] = (h, w) if s.flip in (5, 6) else (w, h)
            # rawpy's .other decodes the whole sensor image first (~0.1-0.9 s a file): only when EXIF left gaps
            if not all(m[k] for k in ('iso', 'exposure', 'aperture', 'focal', 'taken')):
                o = raw.other
                if not m['iso'] and o.iso_speed: m['iso'] = int(o.iso_speed)
                if not m['exposure'] and o.shutter_speed: m['exposure'] = float(o.shutter_speed)
                if not m['aperture'] and o.aperture: m['aperture'] = round(float(o.aperture), 1)
                if not m['focal'] and o.focal_length: m['focal'] = float(o.focal_length)
                if not m['taken'] and o.timestamp:
                    ts = o.timestamp
                    ts = ts if isinstance(ts, datetime) else datetime.fromtimestamp(float(ts))
                    if ts.year > 1970: m['taken'], m['taken_src'] = ts.strftime('%Y-%m-%d %H:%M:%S'), 'raw'
            if not m['lens'] and not exif_ok:   # .lens decodes the image too, and finds no more than EXIF did
                ln = raw.lens
                m['lens'] = (ln.model or '').strip()
                if not m['lens'] and ln.min_focal:
                    lo, hi = round(float(ln.min_focal)), round(float(ln.max_focal or 0))
                    m['lens'] = f'{lo}{"-" + str(hi) if hi and hi != lo else ""}mm'
        except Exception:
            pass
        finally:
            raw.close()
        notes = tags if 'FUJI' in m['make'].upper() else None   # the embedded JPEG's EXIF, read with its maker note
        if 'NIKON' in m['make'].upper():   # the maker note: the focus point, and the lens's range when it has no name
            try:
                with open(path, 'rb') as f:
                    notes = exifread.process_file(f, details=True)
                t = notes.get('MakerNote LensMinMaxFocalMaxAperture')
                lo, hi, a0, a1 = (float(v) for v in t.values[:4])
                if lo and not m['lens']:
                    m['lens'] = (f'{lo:g}' + (f'-{hi:g}' if hi != lo else '') + f'mm f/{a0:g}' + (f'-{a1:g}' if a1 != a0 else ''))
            except Exception:
                pass
        if notes:
            boxes = FP.from_tags(notes, m['make'], m['model'], {0: 1, 3: 3, 5: 8, 6: 6}.get(flip, orientation))
            m['focus'] = FP.dumps(boxes) if boxes else ''
        if not m['make']:
            try:
                E = engine()
                with open(path, 'rb') as f:
                    head = f.read(1 << 20)
                info = E._raf_info(head) if head[:8] == b'FUJIFILM' else None
                if info: m['make'], m['model'] = info['make'], info['model']
            except Exception:
                pass
    else:
        try:
            with Image.open(path) as im:
                w, h = im.size
                if orientation in (5, 6, 7, 8): w, h = h, w
                m['width'], m['height'] = w, h
        except Exception:
            try:
                import tifffile
                with tifffile.TiffFile(path) as tf:
                    sh = tf.pages[0].shape
                    m['height'], m['width'] = int(sh[0]), int(sh[1])
            except Exception:
                pass
    if not m['taken']:
        m['taken'], m['taken_src'] = datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S'), 'file'
    m['camera'] = camera_name(m['make'], m['model'])
    return m


# ───────────────────────────────────────────── thumbnails
_SRGB = None


def _to_srgb(im):
    """PIL image -> RGB in sRGB, honouring an embedded ICC profile (Adobe RGB JPEGs etc.)."""
    global _SRGB
    icc = im.info.get('icc_profile')
    if im.mode not in ('RGB', 'L'):
        im = im.convert('RGBA').convert('RGB') if 'A' in im.mode or im.mode == 'P' else im.convert('RGB')
    if icc and im.mode == 'RGB':
        try:
            if _SRGB is None: _SRGB = ImageCms.createProfile('sRGB')
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            if 'srgb' not in (ImageCms.getProfileDescription(src) or '').lower():
                im = ImageCms.profileToProfile(im, src, _SRGB, outputMode='RGB')
        except Exception:
            pass
    return im.convert('RGB')


def _open_image(path, want=1024):
    """An ordinary image file -> (PIL RGB sRGB, oriented, long side >= want when the file is that big)."""
    ext = os.path.splitext(path)[1].lower()
    try:
        im = Image.open(path)
        if im.format == 'JPEG': im.draft('RGB', (want, want))   # decode at 1/2, 1/4 or 1/8 size: much faster
        if im.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F') or (ext in ('.tif', '.tiff') and im.mode not in ('RGB', 'RGBA', 'L', 'P')):
            raise ValueError('needs the engine')
        im.load()
        im = ImageOps.exif_transpose(im)
        return _to_srgb(im)
    except Exception:
        E = engine()   # 16-bit / float TIFF and PNG, with ICC profiles: the engine reads them colour-managed
        rgb, _, _ = E.load_image_linear(path)
        return Image.fromarray(E.display_image(rgb, want * 2))


def _headerless(path):
    """RAW Demosaic's entry for a headerless .RAW file (known by its size), or None."""
    if not path.lower().endswith('.raw'): return None
    try:
        return engine().headerless(path)
    except Exception:
        return None


def _raw_preview(path, min_side):
    """A RAW file's embedded JPEG preview, oriented, if its long side is at least min_side (else None)."""
    if _headerless(path): return None   # no preview in a bare sensor dump
    import rawpy
    with rawpy.imread(path) as r:
        try:
            t = r.extract_thumb()
        except Exception:
            return None
        flip = int(r.sizes.flip)
        W, H = int(r.sizes.width), int(r.sizes.height)
    if t.format == rawpy.ThumbFormat.JPEG:
        im = Image.open(io.BytesIO(t.data))
        if max(im.size) < min_side: return None
        im.draft('RGB', (1024, 1024))
        exif_o = im.getexif().get(0x112, 1)
        im = _to_srgb(ImageOps.exif_transpose(im))
    else:
        a = np.asarray(t.data)[..., :3]
        if max(a.shape[:2]) < min_side: return None
        if a.dtype != np.uint8: a = (a / max(1, a.max()) * 255).astype(np.uint8)
        im, exif_o = Image.fromarray(a), 1
    if exif_o == 1 and flip:
        # the preview is stored as the sensor reads; rotate it like the RAW unless it is already turned
        already = (flip in (5, 6)) and ((im.width > im.height) != (W > H))
        if not already:
            im = Image.fromarray(engine().orient(np.asarray(im), flip))
    return im


def _raw_rcd(path, rawdemosaic_dir=None):
    """A quick RCD demosaic of the whole RAW file through RAW Demosaic's engine (Bayer); X-Trans uses
    LibRaw's 1-pass Markesteijn, as that app's xt1 column does. -> PIL sRGB, long side <= 1024."""
    E = engine(rawdemosaic_dir)
    src = E.RawFile(path)
    wb = np.asarray(src.wb, np.float32)
    if src.is_xtrans:
        cam = src.xtrans_rgb(1).astype(np.float32) * np.float32(1 / (src.white - np.mean(src.black)))
        cam *= wb
    else:
        mosaic = E.normalize_mosaic(src.raw, src.pattern, src.P, src.black, src.white)
        wm = np.minimum(1, mosaic * wb[E.cfa_colour_map(src.pattern, 2, *mosaic.shape)]).astype(np.float32)
        del mosaic
        cam = E.DM.DEMOSAICS['rcd'](wm, src.pattern)
        del wm
    np.minimum(cam, 1, out=cam)
    if getattr(src, 'distortion', None): cam = E.correct_distortion(cam, src.distortion)   # RAW Demosaic's built-in lens profiles
    E.apply_matrix(cam, E.cam_to_working(src.xyz_to_cam), out=cam)
    np.maximum(cam, 0, out=cam)
    cam = E.orient(cam, src.flip)
    im = Image.fromarray(E.display_image(cam, 2048))   # box-averaged by a whole factor, then Lanczos to 1024
    im.thumbnail((1024, 1024), Image.LANCZOS)
    return im


def _raw_libraw(path):
    """LibRaw's own processing (camera white balance, no brightening), for what RAW Demosaic's engine does not
    demosaic: linear DNGs (already demosaiced, as Lightroom's Enhance, DxO and Topaz write them) and other sensors."""
    import rawpy
    with rawpy.imread(path) as r:
        a = r.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
    im = Image.fromarray(a)
    im.thumbnail((1024, 1024), Image.LANCZOS)
    return im


_ORIENT = {2: Image.FLIP_LEFT_RIGHT, 3: Image.ROTATE_180, 4: Image.FLIP_TOP_BOTTOM, 5: Image.TRANSPOSE,
           6: Image.ROTATE_270, 7: Image.TRANSVERSE, 8: Image.ROTATE_90}


def _tiff_preview(path):
    """The biggest RGB or JPEG image stored in a TIFF-based RAW (a DNG's previews), whatever its size, read without
    LibRaw (for files it rejects), or None."""
    import tifffile
    with tifffile.TiffFile(path) as tf:
        pages, todo = [], list(tf.pages)
        while todo:
            p = todo.pop()
            pages.append(p)
            try: todo += list(p.pages or [])   # SubIFDs
            except Exception: pass
        cands = sorted((p for p in pages if p.photometric in (2, 6) and p.samplesperpixel >= 3),
                       key=lambda p: -p.imagewidth * p.imagelength)
        orient = tf.pages[0].tags.valueof(274, 1)
        im = None
        for p in cands:
            try:
                if p.compression == 7 and len(p.dataoffsets) == 1:   # a whole JPEG stream
                    tf.filehandle.seek(p.dataoffsets[0])
                    im = Image.open(io.BytesIO(tf.filehandle.read(p.databytecounts[0])))
                    im.draft('RGB', (1024, 1024))
                    im = im.convert('RGB')
                else:
                    a = p.asarray()[..., :3]
                    if a.dtype != np.uint8: a = (a / max(1, a.max()) * 255).astype(np.uint8)
                    im = Image.fromarray(a)
                break
            except Exception:
                im = None
    if im is None: return None
    if orient in _ORIENT: im = im.transpose(_ORIENT[orient])
    return im


def _raw_fallback(path, why):
    """A RAW file the engine could not demosaic: LibRaw's processing, else its embedded preview however small.
    -> (PIL image, kind); re-raises why when nothing works."""
    try:
        return _raw_libraw(path), 'libraw'
    except Exception:
        pass
    for get in (lambda: _raw_preview(path, 0), lambda: _tiff_preview(path)):
        try:
            im = get()
            if im is not None: return im, 'preview'
        except Exception:
            pass
    raise why


def render_thumbnail(path, min_preview=1024, rawdemosaic_dir=None):
    """-> (PIL RGB image, long side <= 1024 unless the photo is smaller, source: 'image' | 'preview' | 'rcd' | 'libraw')."""
    if os.path.splitext(path)[1].lower() in RAW_EXTS:
        im = None
        if min_preview is not None:
            try:
                im = _raw_preview(path, min_preview)
            except Exception:
                im = None
        kind = 'preview'
        if im is None:
            try:
                im, kind = _raw_rcd(path, rawdemosaic_dir), 'rcd'
            except Exception as ex:
                im, kind = _raw_fallback(path, ex)
    else:
        im, kind = _open_image(path), 'image'
    if max(im.size) > 1024:
        im.thumbnail((1024, 1024), Image.LANCZOS)
    return im, kind


# ───────────────────────────────────────────── the database
SCHEMA = '''
CREATE TABLE IF NOT EXISTS roots (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL COLLATE NOCASE, added TEXT);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);   -- catalog upgrades done, etc.
CREATE TABLE IF NOT EXISTS folders (path TEXT PRIMARY KEY COLLATE NOCASE);
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL COLLATE NOCASE,
    folder TEXT NOT NULL COLLATE NOCASE,
    name TEXT NOT NULL,
    ext TEXT, is_raw INTEGER NOT NULL DEFAULT 0, size INTEGER, mtime REAL,
    width INTEGER, height INTEGER,
    taken TEXT, taken_src TEXT,
    make TEXT, model TEXT, camera TEXT, lens TEXT,
    focal REAL, focal35 REAL, aperture REAL, exposure REAL, iso INTEGER,
    lat REAL, lon REAL, alt REAL,
    rating INTEGER NOT NULL DEFAULT 0,
    description TEXT NOT NULL DEFAULT '',
    caption TEXT NOT NULL DEFAULT '',
    thumb INTEGER NOT NULL DEFAULT 0,      -- 0 to do, 1 done, -1 failed
    thumb_src TEXT,                        -- image | preview | rcd | libraw
    focus TEXT,                            -- where the camera focused (focuspoints.py JSON); '' none, NULL not read yet
    ai INTEGER NOT NULL DEFAULT 0,         -- bits: 1 faces, 2 image embedding, 4 caption
    missing INTEGER NOT NULL DEFAULT 0,
    added TEXT
);
CREATE INDEX IF NOT EXISTS photos_folder ON photos(folder);
CREATE INDEX IF NOT EXISTS photos_taken ON photos(taken);
CREATE INDEX IF NOT EXISTS photos_rating ON photos(rating);
CREATE INDEX IF NOT EXISTS photos_camera ON photos(camera);
CREATE TABLE IF NOT EXISTS tags (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL COLLATE NOCASE);
CREATE TABLE IF NOT EXISTS photo_tags (
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    source TEXT NOT NULL DEFAULT 'user',   -- user | ai
    PRIMARY KEY (photo_id, tag_id)
);
CREATE INDEX IF NOT EXISTS photo_tags_tag ON photo_tags(tag_id);
CREATE TABLE IF NOT EXISTS saved_filters (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL COLLATE NOCASE, spec TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS persons (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL COLLATE NOCASE);
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL,    -- 0..1 of the oriented image
    score REAL,
    person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    assigned TEXT,                         -- user | auto (NULL: nobody yet)
    cluster INTEGER,                       -- group of similar unnamed faces
    vec BLOB NOT NULL                      -- ArcFace embedding, 512 float32, unit length
);
CREATE TABLE IF NOT EXISTS person_refs (   -- faces of a person shared from another library: used for suggestions only
    id INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    source TEXT,                           -- the file it came from
    vec BLOB NOT NULL                      -- ArcFace embedding, 512 float32, unit length
);
CREATE INDEX IF NOT EXISTS person_refs_person ON person_refs(person_id);
CREATE INDEX IF NOT EXISTS faces_photo ON faces(photo_id);
CREATE INDEX IF NOT EXISTS faces_person ON faces(person_id);
CREATE TABLE IF NOT EXISTS image_vectors (
    photo_id INTEGER PRIMARY KEY REFERENCES photos(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    vec BLOB NOT NULL                      -- SigLIP2 image embedding, float16, unit length
);
'''

AI_FACES, AI_EMBED, AI_CAPTION = 1, 2, 4
SORTS = {'Date taken': 'taken {d}, name', 'File name': 'name {d}', 'Rating': 'rating {d}, taken', 'Folder': 'folder {d}, name',
         'Date added': 'id {d}'}


class Catalog:
    def __init__(self, library_dir=DEFAULT_LIBRARY):
        self.dir = norm(library_dir)
        os.makedirs(self.dir, exist_ok=True)
        self.db_path = os.path.join(self.dir, 'catalog.sqlite')
        self.thumb_dir = os.path.join(self.dir, 'thumbs')
        self._local = threading.local()
        self.lock = threading.RLock()        # one writer at a time (SQLite allows one anyway)
        c = self.conn()
        with self.lock:
            c.executescript(SCHEMA)
            # face boxes written by the first version were numpy float32s, which SQLite stored as 4-byte BLOBs
            bad = c.execute("SELECT id, x0, y0, x1, y1 FROM faces WHERE typeof(x0) = 'blob'").fetchall()
            c.executemany('UPDATE faces SET x0 = ?, y0 = ?, x1 = ?, y1 = ? WHERE id = ?',
                          [tuple(float(np.frombuffer(v, np.float32)[0]) for v in r[1:]) + (r[0],) for r in bad])
            cols = {r[1] for r in c.execute('PRAGMA table_info(photos)')}
            if 'focus' not in cols:
                c.execute('ALTER TABLE photos ADD COLUMN focus TEXT')   # catalogs made before focus points were read
            if 'thumb_spec' not in cols:
                c.execute('ALTER TABLE photos ADD COLUMN thumb_spec TEXT')   # format and quality, e.g. webp85 (NULL: jpeg88)
            c.commit()
        self.thumb_fmt, self.thumb_q = 'avif', THUMB_QUALITY['avif']

    def conn(self):
        c = getattr(self._local, 'c', None)
        if c is None:
            c = sqlite3.connect(self.db_path, timeout=60)
            c.row_factory = sqlite3.Row
            c.execute('PRAGMA journal_mode=WAL')
            c.execute('PRAGMA synchronous=NORMAL')
            c.execute('PRAGMA foreign_keys=ON')
            self._local.c = c
        return c

    def q(self, sql, params=()):
        return self.conn().execute(sql, params).fetchall()

    def q1(self, sql, params=()):
        return self.conn().execute(sql, params).fetchone()

    @contextmanager
    def tx(self):
        with self.lock:
            c = self.conn()
            try:
                yield c
                c.commit()
            except BaseException:
                c.rollback()
                raise

    def thumb_path(self, pid, size, fmt=None):
        """Where the thumbnail is written in this format (default: the one thumbnails are made in now)."""
        return os.path.join(self.thumb_dir, f'{pid // 1000:04d}', f'{pid}_{size}{THUMB_FORMATS[fmt or self.thumb_fmt][0]}')

    def find_thumb(self, pid, size):
        """The stored thumbnail of this size, in whichever format it was made (None if there is none)."""
        for f in [self.thumb_fmt] + [f for f in THUMB_FORMATS if f != self.thumb_fmt]:
            p = self.thumb_path(pid, size, f)
            if os.path.exists(p): return p
        return None

    def best_thumb(self, pid, px):
        """The smallest stored thumbnail at least px on its long side (else the biggest there is)."""
        for s in [s for s in THUMB_SIZES if s >= px] + list(reversed(THUMB_SIZES)):
            p = self.find_thumb(pid, s)
            if p: return p
        return None

    def set_thumb_format(self, fmt, quality):
        """The format and quality new (and recompressed) thumbnails are written in."""
        self.thumb_fmt = fmt if fmt in THUMB_FORMATS else 'jpeg'
        self.thumb_q = max(10, min(100, int(quality or THUMB_QUALITY[self.thumb_fmt])))

    def thumb_spec(self):
        return f'{self.thumb_fmt}{self.thumb_q}'

    def _encode_thumbs(self, im, fmt=None, q=None):
        """PIL RGB (long side <= 1024) -> {size: encoded bytes}, each size made from the next bigger one."""
        fmt, q = fmt or self.thumb_fmt, q or self.thumb_q
        _, pil, opts = THUMB_FORMATS[fmt]
        out = {}
        for s in reversed(THUMB_SIZES):
            if max(im.size) > s: im = im.copy(); im.thumbnail((s, s), Image.LANCZOS)
            b = io.BytesIO()
            im.save(b, pil, quality=q if s > 128 else max(10, q - 3), **opts)
            out[s] = b.getvalue()
        return out

    # ─── roots and folders
    def roots(self):
        return [r['path'] for r in self.q('SELECT path FROM roots ORDER BY path')]

    def root_of(self, path):
        p = os.path.normcase(norm(path))
        for r in self.roots():
            rc = os.path.normcase(r)
            if p == rc or p.startswith(rc.rstrip(os.sep) + os.sep): return r
        return None

    def add_root(self, path):
        """Register a folder (with its subfolders). Returns the root that covers it."""
        path = norm(path)
        inside = self.root_of(path)
        if inside: return inside
        pc = os.path.normcase(path).rstrip(os.sep) + os.sep
        with self.tx() as c:
            for r in self.roots():   # a new parent folder takes over roots inside it
                if os.path.normcase(r).startswith(pc): c.execute('DELETE FROM roots WHERE path = ?', (r,))
            c.execute('INSERT OR IGNORE INTO roots (path, added) VALUES (?, ?)', (path, datetime.now().isoformat(' ', 'seconds')))
        return path

    def remove_root(self, root):
        """Forget a root and everything catalogued under it (the files on disk are not touched)."""
        w, p = _under('path', root)
        fw, fp = _under('path', root)
        with self.tx() as c:
            c.execute('DELETE FROM photos WHERE ' + w, p)
            c.execute('DELETE FROM folders WHERE ' + fw, fp)
            c.execute('DELETE FROM roots WHERE path = ?', (root,))
            c.execute('DELETE FROM tags WHERE id NOT IN (SELECT tag_id FROM photo_tags)')

    def folders(self):
        return [r['path'] for r in self.q('SELECT path FROM folders')]

    # ─── scanning
    def scan(self, root, job, threads=8):
        """Walk root: add new photos, re-read changed ones, flag vanished ones as missing.
        Returns the ids whose thumbnails need making."""
        root = norm(root)
        job.progress(0, f'Scanning {root}…')
        w, p = _under('path', root)
        known = {os.path.normcase(r['path']): (r['id'], r['size'], r['mtime'], r['thumb'])
                 for r in self.q('SELECT id, path, size, mtime, thumb FROM photos WHERE ' + w, p)}
        lib = os.path.normcase(self.dir)
        dirs, todo, seen, n = [], [], set(), 0
        for dirpath, dirnames, filenames in os.walk(root):
            job.check()
            dirnames[:] = sorted(d for d in dirnames if d.lower() not in SKIP_DIRS and not d.startswith('.')
                                 and os.path.normcase(os.path.join(dirpath, d)) != lib)
            dirs.append(dirpath)
            for fn in filenames:
                if not fn.lower().endswith(SUPPORTED_EXTS) or fn.startswith('._'): continue
                fp = os.path.join(dirpath, fn)
                try:
                    st = os.stat(fp)
                except OSError:
                    continue
                k = os.path.normcase(fp)
                seen.add(k)
                n += 1
                old = known.get(k)
                if old is None or old[1] != st.st_size or abs((old[2] or 0) - st.st_mtime) > 1:
                    todo.append((fp, st))
            if len(dirs) % 50 == 0: job.progress(0, f'Scanning… {n} photos in {len(dirs)} folders')
        gone = [v[0] for k, v in known.items() if k not in seen]
        with self.tx() as c:
            c.execute('DELETE FROM folders WHERE ' + w, p)
            c.executemany('INSERT OR IGNORE INTO folders (path) VALUES (?)', [(d,) for d in dirs])
            c.execute('UPDATE photos SET missing = 0 WHERE missing = 1 AND ' + w, p)
            c.executemany('UPDATE photos SET missing = 1 WHERE id = ?', [(i,) for i in gone])
        if gone: job.log('warn', f'{len(gone)} catalogued photo{"s are" if len(gone) != 1 else " is"} no longer on disk (marked missing)')
        job.log('info', f'{root}: {n} photos in {len(dirs)} folders, {len(todo)} new or changed')
        now = datetime.now().isoformat(' ', 'seconds')
        done = 0
        with ThreadPoolExecutor(threads) as ex:
            for i in range(0, len(todo), 64):
                job.check()
                chunk = todo[i:i + 64]
                metas = list(ex.map(lambda a: _safe_meta(*a), chunk))
                with self.tx() as c:
                    for (fp, st), m in zip(chunk, metas):
                        c.execute(UPSERT, dict(m, path=fp, folder=os.path.dirname(fp), name=os.path.basename(fp),
                                               ext=os.path.splitext(fp)[1].lower(), added=now))
                done += len(chunk)
                job.progress(done / max(1, len(todo)), f'Reading EXIF {done}/{len(todo)}')
        self._recheck_dates(root, job)
        n = self.inherit_from_originals(root)
        if n: job.log('info', f'{n} developed or exported file{"s take" if n != 1 else " takes"} the date and camera details of the original')
        self._read_focus(root, job, threads)
        need = [r['id'] for r in self.q('SELECT id FROM photos WHERE thumb = 0 AND missing = 0 AND ' + w, p)]
        return need

    def _recheck_dates(self, root, job):
        """Once per folder: catalogues made before 'datetime' existed counted the TIFF DateTime tag (when an editor
        wrote the file) as the capture time. Re-read just the date of those non-RAW files; nothing else changes."""
        key = 'dates-v2:' + os.path.normcase(root)
        if self.q1('SELECT 1 FROM meta WHERE key = ?', (key,)): return
        import exifread
        w, p = _under('path', root)
        rows = self.q(f"SELECT id, path FROM photos WHERE is_raw = 0 AND missing = 0 AND taken_src = 'exif' AND {w}", p)
        fix = []
        for i, r in enumerate(rows):
            if i % 200 == 0: job.progress(i / max(1, len(rows)), f'Checking dates {i}/{len(rows)}')
            try:
                with open(r['path'], 'rb') as f:
                    d, src = exif_taken(exifread.process_file(f, details=False) or {})
            except Exception:
                continue
            if src == 'datetime': fix.append((r['id'],))
        with self.tx() as c:
            c.executemany("UPDATE photos SET taken_src = 'datetime' WHERE id = ?", fix)
            c.execute('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)', (key, datetime.now().isoformat(' ', 'seconds')))

    def _read_focus(self, root, job, threads):
        """RAW files catalogued before focus points were read: read just those (new files get them with their EXIF)."""
        w, p = _under('path', root)
        rows = self.q(f'SELECT id, path FROM photos WHERE is_raw = 1 AND missing = 0 AND focus IS NULL AND {w}', p)
        found = 0
        with ThreadPoolExecutor(threads) as ex:
            for i in range(0, len(rows), 64):
                job.check()
                chunk = rows[i:i + 64]
                res = list(ex.map(lambda r: _safe_focus(r['path']), chunk))
                found += sum(1 for b in res if b)
                with self.tx() as c:
                    c.executemany('UPDATE photos SET focus = ? WHERE id = ?', [(b, r['id']) for r, b in zip(chunk, res)])
                job.progress((i + len(chunk)) / len(rows), f'Reading focus points {i + len(chunk)}/{len(rows)}')
        if found: job.log('info', f'focus points found in {found} of {len(rows)} RAW files')

    def inherit_from_originals(self, root):
        """Developed or exported copies whose date is only the file's (RAW Demosaic's TIFFs, many editors' exports)
        take the date, camera, lens, exposure and GPS of the original they were made from: the photo in the same
        folder whose name theirs starts with, followed by nothing or a separator (_DSF7969_col1.tif and
        _DSF7969.tif <- _DSF7969.RAF). Their own values win where they have them. Returns how many changed."""
        w, p = _under('folder', root)
        rows = self.q(f'SELECT id, folder, name, is_raw, taken_src FROM photos WHERE missing = 0 AND {w}', p)
        folders = {}
        for r in rows: folders.setdefault(os.path.normcase(r['folder']), []).append(r)
        pairs = []
        for items in folders.values():
            # longest name first, so _DSF7969_col1 matches _DSF7969_col1.jpg before _DSF7969.RAF; RAW first on a tie
            originals = sorted(((os.path.splitext(r['name'])[0].lower(), r['id'], r['is_raw']) for r in items if r['taken_src'] in ('exif', 'raw')),
                               key=lambda t: (-len(t[0]), -t[2]))
            for r in items:
                if r['taken_src'] not in ('file', 'datetime', 'original'): continue
                stem = os.path.splitext(r['name'])[0].lower()
                src = next((oid for o, oid, _ in originals if oid != r['id'] and stem.startswith(o) and (len(stem) == len(o) or stem[len(o)] in '_- .(')), None)
                if src: pairs.append((r['id'], src))
        if not pairs: return 0
        cols = ('make', 'model', 'camera', 'lens', 'focal', 'focal35', 'aperture', 'exposure', 'iso', 'lat', 'lon', 'alt')
        src = self.rows(list({s for _, s in pairs}), 'id, taken, ' + ', '.join(cols))
        dst = self.rows([d for d, _ in pairs], 'id, taken, ' + ', '.join(cols))
        updates = []
        for d, s in pairs:
            a, b = dst[d], src[s]
            vals = [a[c] if a[c] not in (None, '') else b[c] for c in cols]
            if a['taken'] != b['taken'] or vals != [a[c] for c in cols]:
                updates.append((b['taken'], *vals, d))
        with self.tx() as c:
            c.executemany(f"UPDATE photos SET taken = ?, taken_src = 'original', {', '.join(c_ + ' = ?' for c_ in cols)} WHERE id = ?", updates)
        return len(updates)

    def make_thumbnails(self, ids, job, min_preview=1024, rawdemosaic_dir=None, threads=None):
        """Write the 128/512/1024 thumbnails for these photo ids (RCD runs one at a time: it uses every core)."""
        if not ids: return
        threads = threads or max(2, min(20, (os.cpu_count() or 4) - 2))   # Pillow's decode and resize release the GIL
        rows = self.rows(ids, 'id, path, is_raw')
        ids = [i for i in ids if i in rows]
        spec = self.thumb_spec()
        rcd_lock, count_lock = threading.Lock(), threading.Lock()
        t0, counter = time.time(), [0, 0]
        ready, full = [], []

        def one(pid):
            if job.cancel_event.is_set() or full: return
            r = rows[pid]
            try:
                im, kind = None, 'preview'
                if r['is_raw'] and min_preview is not None:
                    try:
                        im = _raw_preview(r['path'], min_preview)
                    except Exception:
                        im = None
                if im is None and r['is_raw']:
                    with rcd_lock:
                        im, kind = render_thumbnail(r['path'], None, rawdemosaic_dir)
                elif im is None:
                    im, kind = render_thumbnail(r['path'], min_preview, rawdemosaic_dir)
                if max(im.size) > 1024: im.thumbnail((1024, 1024), Image.LANCZOS)
                self._write_thumbs(pid, im)
                res = (1, kind, spec)
            except Exception as ex:
                if _disk_full(ex):   # not this photo's fault: leave it to be made later, and stop
                    full.append(ex); return
                res = (-1, None, None)
                job.log('err', f'{os.path.basename(r["path"])}: no thumbnail ({ex})')
            with self.tx() as c:
                c.execute('UPDATE photos SET thumb = ?, thumb_src = ?, thumb_spec = ? WHERE id = ?', (*res, pid))
            with count_lock:
                counter[0] += 1
                counter[1] += res[1] == 'rcd'
                ready.append(pid)

        with ThreadPoolExecutor(threads) as ex:
            futs = [ex.submit(one, i) for i in ids]
            last = 0
            for k, f in enumerate(futs):
                f.result()
                if time.time() - last > 0.25 or k == len(futs) - 1:
                    last = time.time()
                    job.progress((k + 1) / len(futs), f'Thumbnails {k + 1}/{len(futs)}{_eta(t0, k + 1, len(futs))}')
                    with count_lock:
                        batch = ready[:]; ready.clear()
                    if batch: job.post('thumbs', batch)
                if full:
                    for g in futs: g.cancel()
                    raise DiskFull(self.thumb_dir)
                if job.cancel_event.is_set():
                    for g in futs: g.cancel()
                    raise Cancelled()
        job.log('ok', f'{counter[0]} thumbnails made ({counter[1]} by RCD demosaic) in {time.time() - t0:.0f} s')

    def _write_thumbs(self, pid, im):
        """Write the three sizes in the current format, then remove any in another format. -> bytes written."""
        os.makedirs(os.path.dirname(self.thumb_path(pid, 128)), exist_ok=True)
        n = 0
        for s, data in self._encode_thumbs(im).items():
            p = self.thumb_path(pid, s)
            try:
                with open(p + '.part', 'wb') as f:
                    f.write(data)
                os.replace(p + '.part', p)
            except BaseException:
                try: os.remove(p + '.part')
                except OSError: pass
                raise
            n += len(data)
        for f in THUMB_FORMATS:
            if f != self.thumb_fmt:
                for s in THUMB_SIZES:
                    try: os.remove(self.thumb_path(pid, s, f))
                    except OSError: pass
        return n

    def _thumb_bytes(self, pid):
        """Bytes on disk of a photo's thumbnails, every format."""
        n = 0
        for f in THUMB_FORMATS:
            for s in THUMB_SIZES:
                try: n += os.path.getsize(self.thumb_path(pid, s, f))
                except OSError: pass
        return n

    def thumbs_on_disk(self):
        """(bytes, files) of the whole thumbnails folder."""
        n = k = 0
        if not os.path.isdir(self.thumb_dir): return 0, 0
        for d in os.scandir(self.thumb_dir):
            if d.is_dir():
                for e in os.scandir(d.path):
                    if e.is_file(): n += e.stat().st_size; k += 1
        return n, k

    def _thumb_source(self, pid):
        """The biggest stored thumbnail of a photo, decoded (to re-encode it), or None."""
        for s in reversed(THUMB_SIZES):
            p = self.find_thumb(pid, s)
            if p:
                with Image.open(p) as im:
                    return im.convert('RGB')
        return None

    def estimate_thumbs(self, fmt, quality, sample=60):
        """What recompressing every thumbnail to fmt/quality would save, from a random sample.
        -> dict(bytes, files, ratio, estimate, sample, todo)."""
        ids = [r[0] for r in self.q('SELECT id FROM photos WHERE thumb = 1 ORDER BY random() LIMIT ?', (sample,))]
        old = new = 0
        for pid in ids:
            im = self._thumb_source(pid)
            if im is None: continue
            old += self._thumb_bytes(pid)
            new += sum(len(b) for b in self._encode_thumbs(im, fmt, quality).values())
        n, k = self.thumbs_on_disk()
        ratio = new / old if old else 1.0
        spec = f'{fmt}{quality}'
        todo = self.q1("SELECT COUNT(*) FROM photos WHERE thumb = 1 AND COALESCE(thumb_spec, 'jpeg88') != ?", (spec,))[0]
        return dict(bytes=n, files=k, ratio=ratio, estimate=n * ratio, sample=len(ids), todo=todo)

    def recompress_thumbs(self, job, threads=None):
        """Re-encode every thumbnail not already in the current format and quality, from its own biggest size (the
        originals are not needed). Stops cleanly when cancelled; running it again carries on. -> (photos, before, after)."""
        threads = threads or max(2, min(20, (os.cpu_count() or 4) - 2))
        spec = self.thumb_spec()
        ids = [r[0] for r in self.q("SELECT id FROM photos WHERE thumb = 1 AND COALESCE(thumb_spec, 'jpeg88') != ?", (spec,))]
        if not ids: return 0, 0, 0
        t0, lock, done, sizes, full = time.time(), threading.Lock(), [], [0, 0], []

        def one(pid):
            if job.cancel_event.is_set() or full: return
            try:
                before = self._thumb_bytes(pid)
                im = self._thumb_source(pid)
                if im is None: return
                after = self._write_thumbs(pid, im)
            except Exception as ex:
                if _disk_full(ex): full.append(ex); return
                job.log('err', f'photo {pid}: thumbnail not recompressed ({ex})')
                return
            with lock:
                done.append(pid); sizes[0] += before; sizes[1] += after

        def save(batch):
            with self.tx() as c:
                c.executemany('UPDATE photos SET thumb_spec = ? WHERE id = ?', [(spec, p) for p in batch])

        with ThreadPoolExecutor(threads) as ex:
            futs = [ex.submit(one, i) for i in ids]
            last, saved = 0, 0
            for k, f in enumerate(futs):
                f.result()
                if time.time() - last > 0.5 or k == len(futs) - 1:
                    last = time.time()
                    with lock:
                        batch = done[saved:]; saved = len(done); b0, b1 = sizes
                    if batch: save(batch)
                    job.progress((k + 1) / len(futs), f'Recompressing thumbnails {k + 1}/{len(futs)}'
                                 + (f' — {fmt_bytes(b0)} → {fmt_bytes(b1)}' if b0 else '') + _eta(t0, k + 1, len(futs)))
                if job.cancel_event.is_set() or full:
                    for g in futs: g.cancel()
                    break
        with lock:
            batch = done[saved:]
        if batch: save(batch)
        if full: raise DiskFull(self.thumb_dir)
        job.log('ok', f'{len(done)} photos\' thumbnails recompressed to {spec}: {fmt_bytes(sizes[0])} → {fmt_bytes(sizes[1])} '
                      f'in {time.time() - t0:.0f} s' + (' (stopped: run it again to carry on)' if job.cancel_event.is_set() else ''))
        if job.cancel_event.is_set(): raise Cancelled()
        return len(done), sizes[0], sizes[1]

    REBUILD_KEY = 'thumb_rebuild_upto'

    def rebuild_left(self):
        """Photos an unfinished 'rebuild all thumbnails' still has to do, or None when none is under way."""
        r = self.q1('SELECT value FROM meta WHERE key = ?', (self.REBUILD_KEY,))
        if not r: return None
        return self.q1('SELECT COUNT(*) FROM photos WHERE missing = 0 AND id > ?', (int(r[0]),))[0]

    def thumb_files(self):
        """One walk of the thumbnails folder -> ({photo id: sizes on disk, any format}, [unfinished .part files])."""
        have, parts = defaultdict(set), []
        exts = {e for e, _, _ in THUMB_FORMATS.values()}
        if not os.path.isdir(self.thumb_dir): return have, parts
        for d in os.scandir(self.thumb_dir):
            if not d.is_dir(): continue
            for e in os.scandir(d.path):
                stem, ext = os.path.splitext(e.name)
                if ext == '.part': parts.append(e.path); continue
                if ext not in exts: continue
                try: pid, s = map(int, stem.split('_'))
                except ValueError: continue
                have[pid].add(s)
        return have, parts

    def thumbs_to_redo(self, have=None):
        """The photos on disk whose thumbnails a rebuild would redo, by id order:
        dict(missing=[never made, failed, or a size not on disk], format=[those, and the ones not in the current
        format and quality], all=[every photo])."""
        if have is None: have = self.thumb_files()[0]
        full, spec = set(THUMB_SIZES), self.thumb_spec()
        out = dict(missing=[], format=[], all=[])
        for pid, t, s in self.q("SELECT id, thumb, COALESCE(thumb_spec, 'jpeg88') FROM photos WHERE missing = 0 ORDER BY id"):
            out['all'].append(pid)
            if t != 1 or not full <= have.get(pid, set()):
                out['missing'].append(pid); out['format'].append(pid)
            elif s != spec:
                out['format'].append(pid)
        return out

    def rebuild_counts(self):
        """For the rebuild dialog: dict(missing, format, all: photos; parts: unfinished files; left: see rebuild_left)."""
        have, parts = self.thumb_files()
        return dict({k: len(v) for k, v in self.thumbs_to_redo(have).items()}, parts=len(parts), left=self.rebuild_left())

    def rebuild_thumbs(self, which, job, min_preview=1024, rawdemosaic_dir=None, restart=False, chunk=500):
        """Make thumbnails afresh from the originals, in the current format; nothing else about the photos is touched.
        which: 'missing', 'format' or 'all' (see thumbs_to_redo); missing photos keep the thumbnails they have.
        Unfinished (.part) files are removed first. After Cancel (or a full disk), running it again carries on:
        'missing' and 'format' by what is still to do, 'all' from the last chunk finished, kept in meta
        (restart=True begins again). -> photos done."""
        have, parts = self.thumb_files()
        for p in parts:
            try: os.remove(p)
            except OSError: pass
        if parts: job.log('info', f'{len(parts):,} unfinished thumbnail files removed')
        if which == 'all':
            with self.tx() as c:
                if restart: c.execute('DELETE FROM meta WHERE key = ?', (self.REBUILD_KEY,))
                c.execute('INSERT OR IGNORE INTO meta (key, value) VALUES (?, 0)', (self.REBUILD_KEY,))
            upto = int(self.q1('SELECT value FROM meta WHERE key = ?', (self.REBUILD_KEY,))[0])
            ids = [r[0] for r in self.q('SELECT id FROM photos WHERE missing = 0 AND id > ? ORDER BY id', (upto,))]
        else:
            ids = self.thumbs_to_redo(have)[which]
        t0, n = time.time(), len(ids)
        for k in range(0, n, chunk):
            part = ids[k:k + chunk]

            def progress(f, label=None, k=k, m=len(part)):
                d = k + round(f * m)
                job.progress(d / n, f'Rebuilding thumbnails {d:,}/{n:,}{_eta(t0, d, n) if d else ""}')
            sub = Job(log=lambda lvl, msg: lvl != 'ok' and job.log(lvl, msg), progress=progress, post=job.post)
            sub.cancel_event = job.cancel_event
            self.make_thumbnails(part, sub, min_preview, rawdemosaic_dir)   # raises Cancelled: this chunk is redone next time
            if which == 'all':
                with self.tx() as c:
                    c.execute('UPDATE meta SET value = ? WHERE key = ?', (part[-1], self.REBUILD_KEY))
        if which == 'all':
            with self.tx() as c:
                c.execute('DELETE FROM meta WHERE key = ?', (self.REBUILD_KEY,))
        job.log('ok', f'{n:,} photos\' thumbnails rebuilt from the originals as {self.thumb_spec()} in {time.time() - t0:.0f} s')
        return n

    def forget_missing(self):
        ids = [r['id'] for r in self.q('SELECT id FROM photos WHERE missing = 1')]
        with self.tx() as c:
            c.execute('DELETE FROM photos WHERE missing = 1')
            c.execute('DELETE FROM tags WHERE id NOT IN (SELECT tag_id FROM photo_tags)')
        self._delete_thumbs(ids)
        return len(ids)

    def _delete_thumbs(self, ids):
        for pid in ids:
            for f in THUMB_FORMATS:
                for s in THUMB_SIZES:
                    try: os.remove(self.thumb_path(pid, s, f))
                    except OSError: pass

    # ─── photos that were moved
    def find_moved(self, folder, job, threads=8):
        """Catalogued photos whose files are gone, looked for in folder (and its subfolders): each one found keeps its
        catalog entry — thumbnails, rating, tags, description, faces, AI — and just points at its new path.

        A match is the same file name and exact size; a renamed file needs the same size, date taken, camera and
        pixel size (its EXIF is read to check). Only unambiguous matches are made: when two files fit, the one whose
        folder names match the old ones wins, otherwise the photo is left missing. A photo already catalogued again at
        its new place (a scan after the move) is merged into its old entry: your rating, tags and description from
        either are kept, the newer entry's thumbnails and AI data go. Returns a summary dict."""
        folder = norm(folder)
        job.progress(0, 'Checking which catalogued photos are gone…')
        rows = self.q('SELECT id, path, folder, name, size, taken, model, width, height, missing FROM photos')
        exists = {}
        lost = []
        for i, r in enumerate(rows):
            if i % 2000 == 0: job.check()
            d = os.path.normcase(r['folder'])
            if d not in exists: exists[d] = os.path.isdir(r['folder'])
            if not exists[d] or not os.path.isfile(r['path']): lost.append(r)
        res = dict(lost=len(lost), by_name=0, by_exif=0, merged=0, ambiguous=0, still_missing=0, root_added=None)
        if not lost: return res

        job.progress(0, f'Looking for {len(lost)} photos in {folder}…')
        lib = os.path.normcase(self.dir)
        found = []                        # (path, stat)
        for dirpath, dirnames, filenames in os.walk(folder):
            job.check()
            dirnames[:] = sorted(d for d in dirnames if d.lower() not in SKIP_DIRS and not d.startswith('.')
                                 and os.path.normcase(os.path.join(dirpath, d)) != lib)
            for fn in filenames:
                if not fn.lower().endswith(SUPPORTED_EXTS) or fn.startswith('._'): continue
                fp = os.path.join(dirpath, fn)
                try: found.append((fp, os.stat(fp)))
                except OSError: pass
        by_name, by_size = defaultdict(list), defaultdict(list)
        for fp, st in found:
            by_name[(os.path.basename(fp).lower(), st.st_size)].append((fp, st))
            by_size[st.st_size].append((fp, st))

        def tail_match(old, new):
            """How many folder names, from the file upwards, the old and new paths share: a tie-breaker."""
            a, b = os.path.normcase(os.path.dirname(old)).split(os.sep)[::-1], os.path.normcase(os.path.dirname(new)).split(os.sep)[::-1]
            n = 0
            while n < min(len(a), len(b)) and a[n] == b[n]: n += 1
            return n

        def pick(r, cands):
            if len(cands) == 1: return cands[0]
            score = [tail_match(r['path'], fp) for fp, _ in cands]
            best = max(score)
            return cands[score.index(best)] if score.count(best) == 1 else None

        used, matches, unmatched, todo = set(), [], [], lost
        for _ in range(2):                # 1) same name and size; again for the unclear ones once the clear ones are taken
            unmatched = []
            for r in todo:
                cands = [c for c in by_name.get((r['name'].lower(), r['size']), []) if os.path.normcase(c[0]) not in used]
                hit = pick(r, cands) if cands else None
                if hit:
                    used.add(os.path.normcase(hit[0])); matches.append((r, hit)); res['by_name'] += 1
                else:
                    unmatched.append(r)
            todo = unmatched
        exif = {}
        for i, r in enumerate(unmatched):  # 2) renamed: same size, and the same date taken, camera and pixel size
            if i % 50 == 0: job.progress(i / max(1, len(unmatched)), f'Checking renamed files {i}/{len(unmatched)}')
            cands = [c for c in by_size.get(r['size'], []) if os.path.normcase(c[0]) not in used]
            if not r['taken'] or not cands or len(cands) > 20: continue
            ok = []
            for fp, st in cands:
                if fp not in exif: exif[fp] = _safe_meta(fp, st)
                m = exif[fp]
                if m['taken'] == r['taken'] and (m['model'] or '') == (r['model'] or '') and (m['width'], m['height']) == (r['width'], r['height']):
                    ok.append((fp, st))
            hit = pick(r, ok) if ok else None
            if hit:
                used.add(os.path.normcase(hit[0])); matches.append((r, hit)); res['by_exif'] += 1
        res['still_missing'] = len(lost) - len(matches)
        moved = {r['id'] for r, _ in matches}
        res['ambiguous'] = sum(1 for r in lost if r['id'] not in moved and len(by_name.get((r['name'].lower(), r['size']), [])) > 1)

        job.progress(1, f'Updating {len(matches)} catalog entries…')
        dup_ids = []
        with self.tx() as c:
            for r, (fp, st) in matches:
                dup = c.execute('SELECT id, rating, description FROM photos WHERE path = ? AND id != ?', (fp, r['id'])).fetchone()
                if dup:                   # catalogued again after the move: keep the old entry, bring over your edits
                    c.execute("UPDATE photos SET rating = CASE WHEN rating = 0 THEN ? ELSE rating END, "
                              "description = CASE WHEN description = '' THEN ? ELSE description END WHERE id = ?",
                              (dup['rating'], dup['description'], r['id']))
                    c.execute("INSERT OR IGNORE INTO photo_tags (photo_id, tag_id, source) SELECT ?, tag_id, source FROM photo_tags "
                              "WHERE photo_id = ? AND source = 'user'", (r['id'], dup['id']))
                    c.execute('DELETE FROM photos WHERE id = ?', (dup['id'],))   # its faces, vectors and tags go with it
                    dup_ids.append(dup['id']); res['merged'] += 1
                c.execute('UPDATE photos SET path = ?, folder = ?, name = ?, ext = ?, mtime = ?, missing = 0 WHERE id = ?',
                          (fp, os.path.dirname(fp), os.path.basename(fp), os.path.splitext(fp)[1].lower(), st.st_mtime, r['id']))
            c.executemany('UPDATE photos SET missing = 1 WHERE id = ?', [(r['id'],) for r in lost if r['id'] not in moved])
            c.execute('DELETE FROM tags WHERE id NOT IN (SELECT tag_id FROM photo_tags)')
        self._delete_thumbs(dup_ids)
        if res['merged']: self.sync_person_tags()
        if matches and not self.root_of(folder):
            res['root_added'] = self.add_root(folder)
        with self.tx() as c:              # library folders that are gone and now hold nothing
            for root in self.roots():
                w, p = _under('path', root)
                if not os.path.isdir(root) and not c.execute('SELECT 1 FROM photos WHERE ' + w + ' LIMIT 1', p).fetchone():
                    c.execute('DELETE FROM folders WHERE ' + w, p)
                    c.execute('DELETE FROM roots WHERE path = ?', (root,))
            gone = [(f,) for (f,) in c.execute('SELECT path FROM folders').fetchall() if not os.path.isdir(f)]
            c.executemany('DELETE FROM folders WHERE path = ? AND NOT EXISTS (SELECT 1 FROM photos WHERE folder = folders.path)', gone)
        return res

    # ─── queries
    def query_ids(self, spec, scope=None, sort='Date taken', desc=False, ids=None):
        w, p = where(spec, scope)
        if ids is not None:   # keep the given order (e.g. semantic search ranking)
            rows = self.q(f'SELECT id FROM photos WHERE {w}', p)
            have = {r['id'] for r in rows}
            return [i for i in ids if i in have]
        order = SORTS.get(sort, SORTS['Date taken']).format(d='DESC' if desc else 'ASC')
        return [r['id'] for r in self.q(f'SELECT id FROM photos WHERE {w} ORDER BY {order}', p)]

    def rows(self, ids, cols='*'):
        out = {}
        for i in range(0, len(ids), 900):
            part = ids[i:i + 900]
            for r in self.q(f'SELECT {cols} FROM photos WHERE id IN ({",".join("?" * len(part))})', part):
                out[r['id']] = r
        return out

    def folder_counts(self, spec):
        w, p = where(spec, None)
        return {r['folder']: r['n'] for r in self.q(f'SELECT folder, COUNT(*) AS n FROM photos WHERE {w} GROUP BY folder', p)}

    def date_counts(self, spec):
        w, p = where(spec, None)
        return [(r['d'], r['n']) for r in self.q(f'SELECT substr(taken, 1, 10) AS d, COUNT(*) AS n FROM photos WHERE {w} GROUP BY d ORDER BY d', p)]

    def distinct(self, col):
        return [r[0] for r in self.q(f"SELECT DISTINCT {col} FROM photos WHERE {col} IS NOT NULL AND {col} != '' ORDER BY {col} COLLATE NOCASE")]

    def all_tags(self):
        return [r['name'] for r in self.q('SELECT name FROM tags ORDER BY name COLLATE NOCASE')]

    def tag_counts(self, ids):
        """Tags on these photos -> [(name, count, sources)]."""
        out = []
        for i in range(0, len(ids), 900):
            part = ids[i:i + 900]
            out += self.q(f'SELECT t.name, COUNT(*) AS n, GROUP_CONCAT(DISTINCT pt.source) AS src FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id '
                          f'WHERE pt.photo_id IN ({",".join("?" * len(part))}) GROUP BY t.name', part)
        agg = {}
        for r in out:
            a = agg.setdefault(r['name'], [0, set()])
            a[0] += r['n']; a[1] |= set((r['src'] or '').split(','))
        return sorted(((k, v[0], v[1]) for k, v in agg.items()), key=lambda t: (-t[1], t[0].lower()))

    # ─── edits (catalog only — never the files)
    def set_rating(self, ids, rating):
        with self.tx() as c:
            c.executemany('UPDATE photos SET rating = ? WHERE id = ?', [(int(rating), i) for i in ids])

    def set_description(self, ids, text):
        with self.tx() as c:
            c.executemany('UPDATE photos SET description = ? WHERE id = ?', [(text, i) for i in ids])

    def _tag_id(self, c, name):
        c.execute('INSERT OR IGNORE INTO tags (name) VALUES (?)', (name,))
        return c.execute('SELECT id FROM tags WHERE name = ?', (name,)).fetchone()[0]

    def add_tags(self, ids, names, source='user'):
        names = [n.strip() for n in names if n.strip()]
        with self.tx() as c:
            for n in names:
                t = self._tag_id(c, n)
                # a user's tag replaces an AI one of the same name (so it survives re-captioning)
                c.executemany('INSERT INTO photo_tags (photo_id, tag_id, source) VALUES (?, ?, ?) '
                              "ON CONFLICT(photo_id, tag_id) DO UPDATE SET source = CASE WHEN excluded.source = 'user' THEN 'user' ELSE source END",
                              [(i, t, source) for i in ids])

    def remove_tags(self, ids, names):
        with self.tx() as c:
            for n in names:
                r = c.execute('SELECT id FROM tags WHERE name = ?', (n,)).fetchone()
                if r: c.executemany("DELETE FROM photo_tags WHERE photo_id = ? AND tag_id = ? AND source != 'person'", [(i, r[0]) for i in ids])
            c.execute('DELETE FROM tags WHERE id NOT IN (SELECT tag_id FROM photo_tags)')

    def sync_person_tags(self):
        """Every photo is tagged with the people whose faces are in it (source 'person'), so names work like any tag: in
        the tag list, the tag filter, keyword search and virtual folders. Named and suggested faces (marked ?) both
        count, as in the Person filter; "Not this person" on the People tab takes a wrong one off. Rebuilt in full:
        a few statements, even for many faces."""
        named = "FROM faces f JOIN persons p ON p.id = f.person_id WHERE f.assigned IN ('user', 'auto')"
        with self.tx() as c:
            c.execute("DELETE FROM photo_tags WHERE source = 'person'")
            c.execute(f'INSERT OR IGNORE INTO tags (name) SELECT DISTINCT p.name {named}')
            # a tag you added yourself with the same name stays yours (the primary key keeps the first row)
            c.execute("INSERT OR IGNORE INTO photo_tags (photo_id, tag_id, source) SELECT DISTINCT f.photo_id, t.id, 'person' "
                      + named.replace('WHERE', 'JOIN tags t ON t.name = p.name WHERE'))
            c.execute('DELETE FROM tags WHERE id NOT IN (SELECT tag_id FROM photo_tags)')

    def set_ai_tags(self, c, pid, names):
        c.execute("DELETE FROM photo_tags WHERE photo_id = ? AND source = 'ai'", (pid,))
        for n in names:
            t = self._tag_id(c, n)
            c.execute("INSERT OR IGNORE INTO photo_tags (photo_id, tag_id, source) VALUES (?, ?, 'ai')", (pid, t))

    # ─── saved filters ("virtual folders")
    def saved_filters(self):
        return [(r['name'], json.loads(r['spec'])) for r in self.q('SELECT name, spec FROM saved_filters ORDER BY name COLLATE NOCASE')]

    def save_filter(self, name, spec):
        with self.tx() as c:
            c.execute('INSERT INTO saved_filters (name, spec) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET spec = excluded.spec',
                      (name, json.dumps(spec)))

    def delete_filter(self, name):
        with self.tx() as c:
            c.execute('DELETE FROM saved_filters WHERE name = ?', (name,))

    def stats(self):
        r = self.q1('SELECT COUNT(*) AS n, SUM(is_raw) AS raw, SUM(thumb = 1) AS th, SUM(missing) AS miss, '
                    f'SUM(ai & {AI_FACES} > 0) AS f, SUM(ai & {AI_EMBED} > 0) AS e, SUM(ai & {AI_CAPTION} > 0) AS c FROM photos')
        return {k: (r[k] or 0) for k in r.keys()}


UPSERT = '''
INSERT INTO photos (path, folder, name, ext, is_raw, size, mtime, width, height, taken, taken_src, make, model, camera, lens,
                    focal, focal35, aperture, exposure, iso, lat, lon, alt, focus, added)
VALUES (:path, :folder, :name, :ext, :is_raw, :size, :mtime, :width, :height, :taken, :taken_src, :make, :model, :camera, :lens,
        :focal, :focal35, :aperture, :exposure, :iso, :lat, :lon, :alt, :focus, :added)
ON CONFLICT(path) DO UPDATE SET
    folder = excluded.folder, name = excluded.name, ext = excluded.ext, is_raw = excluded.is_raw, size = excluded.size,
    mtime = excluded.mtime, width = excluded.width, height = excluded.height, taken = excluded.taken, taken_src = excluded.taken_src,
    make = excluded.make, model = excluded.model, camera = excluded.camera, lens = excluded.lens, focal = excluded.focal,
    focal35 = excluded.focal35, aperture = excluded.aperture, exposure = excluded.exposure, iso = excluded.iso,
    lat = excluded.lat, lon = excluded.lon, alt = excluded.alt, focus = excluded.focus, thumb = 0, ai = 0, missing = 0
'''


def _safe_focus(path):
    try:
        boxes = FP.read(path)
        return FP.dumps(boxes) if boxes else ''
    except Exception:
        return ''


def _safe_meta(fp, st):
    try:
        return read_metadata(fp, st)
    except Exception:
        m = dict(size=st.st_size, mtime=st.st_mtime, is_raw=int(fp.lower().endswith(RAW_EXTS)), width=None, height=None,
                 taken=datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S'), taken_src='file', make='', model='',
                 camera='', lens='', focal=None, focal35=None, aperture=None, exposure=None, iso=None, lat=None, lon=None, alt=None,
                 focus=None)
        return m


def fmt_bytes(n):
    return f'{n / 2 ** 30:.2f} GB' if n >= 2 ** 30 else f'{n / 2 ** 20:.1f} MB' if n >= 2 ** 20 else f'{n / 1024:.0f} KB'


def _eta(t0, done, total):
    if done < 3 or done >= total: return ''
    left = (time.time() - t0) / done * (total - done)
    return f' — about {math.ceil(left / 5) * 5} s left' if left < 90 else f' — about {round(left / 60)} min left'


def _under(col, folder):
    """SQL: col is this folder or anything inside it (case-insensitive, like Windows paths)."""
    f = norm(folder).rstrip(os.sep)
    pre = f + os.sep
    return f'({col} = ? COLLATE NOCASE OR lower(substr({col}, 1, ?)) = lower(?))', (f, len(pre), pre)


# ───────────────────────────────────────────── filters
def parse_shutter(s):
    """'1/250', '0.004', '2' or '2s' -> seconds."""
    s = str(s).strip().lower().rstrip('s').strip()
    if not s: return None
    if '/' in s:
        a, b = s.split('/', 1)
        return float(a) / float(b)
    return float(s)


def where(spec, scope):
    """A filter spec (+ the browsed scope) -> (SQL WHERE, params).

    spec keys (all optional): rating_min, rating_exact, tags [names], tag_mode 'all'|'any', camera, lens,
    iso [lo, hi], aperture [lo, hi], focal [lo, hi], exposure [lo, hi] (seconds), date_from, date_to ('YYYY-MM-DD'),
    gps True/False, kind 'raw'|'image', text, person (id), ai_missing (bit), show_missing.
    scope: None | {'kind': 'folder', 'path', 'recursive'} | {'kind': 'date', 'prefix': 'YYYY' / 'YYYY-MM' / 'YYYY-MM-DD' / ''}."""
    spec = spec or {}
    w, p = [], []
    if not spec.get('show_missing'): w.append('missing = 0')
    if spec.get('rating_exact') is not None:
        w.append('rating = ?'); p.append(int(spec['rating_exact']))
    elif spec.get('rating_min'):
        w.append('rating >= ?'); p.append(int(spec['rating_min']))
    tags = [t for t in spec.get('tags') or [] if t]
    if tags:
        sub = 'SELECT pt.photo_id FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id WHERE t.name = ?'
        if spec.get('tag_mode') == 'any':
            w.append('id IN (' + ' UNION '.join([sub] * len(tags)) + ')'); p += tags
        else:
            for t in tags:
                w.append(f'id IN ({sub})'); p.append(t)
    for k in ('camera', 'lens'):
        if spec.get(k):
            w.append(f'{k} = ? COLLATE NOCASE'); p.append(spec[k])
    for k in ('iso', 'aperture', 'focal', 'exposure'):
        lo, hi = (list(spec.get(k) or []) + [None, None])[:2]
        if lo is not None: w.append(f'{k} >= ?'); p.append(float(lo) * (0.999 if k == 'exposure' else 1))
        if hi is not None: w.append(f'{k} <= ?'); p.append(float(hi) * (1.001 if k == 'exposure' else 1))
    if spec.get('date_from'): w.append('taken >= ?'); p.append(spec['date_from'])
    if spec.get('date_to'): w.append('taken < ?'); p.append(spec['date_to'] + '~')
    if spec.get('gps') is True: w.append('lat IS NOT NULL')
    elif spec.get('gps') is False: w.append('lat IS NULL')
    kind = spec.get('kind') or ''
    if kind == 'raw': w.append('is_raw = 1')
    elif kind == 'image': w.append('is_raw = 0')
    elif kind.startswith('ext:'):   # one file type, e.g. 'ext:.tif,.tiff'
        exts = [e for e in kind[4:].split(',') if e]
        w.append(f'ext IN ({",".join("?" * len(exts))})'); p += exts
    if spec.get('person'):
        w.append('id IN (SELECT photo_id FROM faces WHERE person_id = ?)'); p.append(int(spec['person']))
    if spec.get('text'):
        for word in str(spec['text']).split():
            like = f'%{word}%'
            w.append('(name LIKE ? OR description LIKE ? OR caption LIKE ? OR camera LIKE ? OR lens LIKE ? OR folder LIKE ? OR '
                     'id IN (SELECT pt.photo_id FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id WHERE t.name LIKE ?))')
            p += [like] * 7
    if spec.get('ai_missing'):
        w.append('(ai & ?) = 0'); p.append(int(spec['ai_missing']))
    if scope:
        if scope.get('kind') == 'folder':
            if scope.get('recursive', True):
                uw, up = _under('folder', scope['path']); w.append(uw); p += up
            else:
                w.append('folder = ?'); p.append(norm(scope['path']))
        elif scope.get('kind') == 'date':
            pre = scope.get('prefix') or ''
            if pre == 'undated': w.append('taken IS NULL')
            elif pre: w.append('substr(taken, 1, ?) = ?'); p += [len(pre), pre]
    return (' AND '.join(w) or '1'), p


def describe(spec):
    """A short human summary of a filter spec, for the status line and saved-filter list."""
    if not spec: return 'no filter'
    bits = []
    if spec.get('rating_exact') is not None: bits.append('unrated' if spec['rating_exact'] == 0 else f'{spec["rating_exact"]}★')
    elif spec.get('rating_min'): bits.append(f'{spec["rating_min"]}★+')
    if spec.get('tags'): bits.append((' or ' if spec.get('tag_mode') == 'any' else ' + ').join(spec['tags']))
    for k in ('camera', 'lens'):
        if spec.get(k): bits.append(spec[k])
    for k, u in (('iso', 'ISO '), ('aperture', 'f/'), ('focal', ''), ('exposure', '')):
        r = spec.get(k)
        if r and any(v is not None for v in r): bits.append(f'{u}{r[0] if r[0] is not None else ""}–{r[1] if r[1] is not None else ""}{" mm" if k == "focal" else " s" if k == "exposure" else ""}')
    if spec.get('date_from') or spec.get('date_to'): bits.append(f'{spec.get("date_from") or "…"} → {spec.get("date_to") or "…"}')
    if spec.get('gps') is not None: bits.append('with GPS' if spec['gps'] else 'no GPS')
    if spec.get('kind'): bits.append(kind_label(spec['kind']))
    if spec.get('text'): bits.append(f'"{spec["text"]}"')
    if spec.get('person_name'): bits.append(spec['person_name'])
    return ', '.join(bits) or 'no filter'


EXT_GROUPS = {'.jpeg': '.jpg', '.jfif': '.jpg', '.tiff': '.tif'}   # spellings of one format, filtered together
EXT_NAMES = {'.jpg': 'JPEG', '.tif': 'TIFF', '.png': 'PNG', '.webp': 'WebP', '.bmp': 'BMP'}


def file_types(exts):
    """Catalogued extensions -> [(label, kind)] for the file-type filter: images first, then each RAW format."""
    groups = {}
    for e in exts:
        e = e.lower()
        groups.setdefault(EXT_GROUPS.get(e, e), set()).add(e)
    out = []
    for main in sorted(groups, key=lambda e: (e in RAW_EXTS, EXT_NAMES.get(e, e.upper()))):
        out.append((kind_label('ext:' + ','.join(sorted(groups[main]))), 'ext:' + ','.join(sorted(groups[main]))))
    return out


def kind_label(kind):
    if kind == 'raw': return 'RAW (any)'
    if kind == 'image': return 'Not RAW'
    exts = kind[4:].split(',') if kind.startswith('ext:') else [kind]
    main = EXT_GROUPS.get(exts[0], exts[0])
    name = EXT_NAMES.get(main, main.lstrip('.').upper())
    return f'{name} ({", ".join(e.lstrip(".") for e in exts)})' if main not in RAW_EXTS else f'{name} (RAW)'


def fmt_exposure(s):
    if not s: return ''
    return f'{s:.1f} s' if s >= 1 else f'1/{round(1 / s)} s'
