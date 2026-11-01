# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Where the camera focused, read from a RAW file's maker notes — shared by Photo Library and RAW Demosaic (the two
copies of this file are kept identical).

Supported: Fujifilm (FocusPixel, the selected face/eye box and the detected faces, from the RAF's embedded JPEG) and
Nikon DSLRs with AFInfo2 version 0100 (D3/D4/D7x00/D800/D810 …): contrast-detect (LiveView) AF areas for all of
them, phase-detect points for the 51-point FX bodies whose point positions are known (D800, D800E, D810, D810A).

Result: a list of boxes [x, y, w, h, kind] — centre and size as fractions of the photo as displayed (after the EXIF
orientation is applied), kind 'focus' (where it focused), 'af' (other AF points used) or 'face' (faces the camera
found). An empty list means the file says nothing usable.

Sources: ExifTool's Nikon and FujiFilm tag tables (Phil Harvey, exiftool.org) for the tag layouts and the 51-point
numbering; the D800 point positions from the Focus-Points Lightroom plugin (github.com/FocusPointsLrC/Focus-Points,
Apache-2.0, map submitted by capricorn8) — the D810 has the same sensor size and the same 51-point AF module.
"""
import io
import json

# 51-point Nikon numbering (ExifTool %afPoints51): bit n-1 of AFPointsUsed is point n
AF_POINTS_51 = {
    1: 'C6', 2: 'B6', 3: 'A5', 4: 'D6', 5: 'E5', 6: 'C7', 7: 'B7', 8: 'A6', 9: 'D7', 10: 'E6',
    11: 'C5', 12: 'B5', 13: 'A4', 14: 'D5', 15: 'E4', 16: 'C8', 17: 'B8', 18: 'A7', 19: 'D8', 20: 'E7',
    21: 'C9', 22: 'B9', 23: 'A8', 24: 'D9', 25: 'E8', 26: 'C10', 27: 'B10', 28: 'A9', 29: 'D10', 30: 'E9',
    31: 'C11', 32: 'B11', 33: 'D11', 34: 'C4', 35: 'B4', 36: 'A3', 37: 'D4', 38: 'E3', 39: 'C3', 40: 'B3',
    41: 'A2', 42: 'D3', 43: 'E2', 44: 'C2', 45: 'B2', 46: 'A1', 47: 'D2', 48: 'E1', 49: 'C1', 50: 'B1', 51: 'D1'}

# point centre and size in pixels of the 7360 × 4912 FX frame (Focus-Points plugin, "nikon d800.txt")
_D800 = (7360, 4912, {
    'B1': (1850, 2136, 295, 240), 'C1': (1850, 2456, 295, 240), 'D1': (1850, 2776, 295, 240),
    'A1': (2195, 1816, 295, 240), 'B2': (2195, 2136, 295, 240), 'C2': (2195, 2456, 295, 240), 'D2': (2195, 2776, 295, 240), 'E1': (2195, 3096, 295, 240),
    'A2': (2540, 1816, 295, 240), 'B3': (2540, 2136, 295, 240), 'C3': (2540, 2456, 295, 240), 'D3': (2540, 2776, 295, 240), 'E2': (2540, 3096, 295, 240),
    'A3': (2885, 1816, 295, 240), 'B4': (2885, 2136, 295, 240), 'C4': (2885, 2456, 295, 240), 'D4': (2885, 2776, 295, 240), 'E3': (2885, 3096, 295, 240),
    'A4': (3285, 1716, 305, 240), 'B5': (3285, 2086, 305, 240), 'C5': (3285, 2456, 305, 240), 'D5': (3285, 2826, 305, 240), 'E4': (3285, 3196, 305, 240),
    'A5': (3680, 1716, 305, 240), 'B6': (3680, 2086, 305, 240), 'C6': (3680, 2456, 305, 240), 'D6': (3680, 2826, 305, 240), 'E5': (3680, 3196, 305, 240),
    'A6': (4075, 1716, 305, 240), 'B7': (4075, 2086, 305, 240), 'C7': (4075, 2456, 305, 240), 'D7': (4075, 2826, 305, 240), 'E6': (4075, 3196, 305, 240),
    'A7': (4475, 1816, 295, 240), 'B8': (4475, 2136, 295, 240), 'C8': (4475, 2456, 295, 240), 'D8': (4475, 2776, 295, 240), 'E7': (4475, 3096, 295, 240),
    'A8': (4820, 1816, 295, 240), 'B9': (4820, 2136, 295, 240), 'C9': (4820, 2456, 295, 240), 'D9': (4820, 2776, 295, 240), 'E8': (4820, 3096, 295, 240),
    'A9': (5165, 1816, 295, 240), 'B10': (5165, 2136, 295, 240), 'C10': (5165, 2456, 295, 240), 'D10': (5165, 2776, 295, 240), 'E9': (5165, 3096, 295, 240),
    'B11': (5510, 2136, 295, 240), 'C11': (5510, 2456, 295, 240), 'D11': (5510, 2776, 295, 240)})
NIKON_51_MAPS = {m: _D800 for m in ('NIKON D800', 'NIKON D800E', 'NIKON D810', 'NIKON D810A')}

FUJI_FOCUS_BOX = 0.035     # FocusPixel is a point: drawn as a square this fraction of the photo's longer side


def _vals(tags, key):
    t = tags.get(key)
    return list(t.values) if t is not None and hasattr(t, 'values') else None


def _orient(boxes, orientation):
    """Boxes in the sensor's landscape frame -> the frame the photo is shown in (EXIF orientation 1/3/6/8)."""
    out = []
    for x, y, w, h, k in boxes:
        if orientation == 3: x, y = 1 - x, 1 - y
        elif orientation == 6: x, y, w, h = 1 - y, x, h, w      # shown rotated 90° clockwise
        elif orientation == 8: x, y, w, h = y, 1 - x, h, w      # shown rotated 90° anticlockwise
        out.append([round(x, 4), round(y, 4), round(w, 4), round(h, 4), k])
    return out


def _overlap(a, b):
    """Intersection over union of two (x0, y0, x1, y1) boxes."""
    iw, ih = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / u if u > 0 else 0


def _fuji(tags):
    fw, fh = (_vals(tags, 'EXIF ExifImageWidth') or [0])[0], (_vals(tags, 'EXIF ExifImageLength') or [0])[0]
    fp = _vals(tags, 'MakerNote FocusPixel')
    if not (fw and fh and fp and len(fp) >= 2): return []
    s = FUJI_FOCUS_BOX * max(fw, fh)
    boxes = [(fp[0] / fw, fp[1] / fh, s / fw, s / fh, 'focus')]
    sel = _vals(tags, 'MakerNote Tag 0x4005')                    # FaceElementSelected: the face/eye it focused on
    if sel and len(sel) == 4 and sel[2] > sel[0] and sel[3] > sel[1]:
        boxes.append(((sel[0] + sel[2]) / 2 / fw, (sel[1] + sel[3]) / 2 / fh, (sel[2] - sel[0]) / fw, (sel[3] - sel[1]) / fh, 'focus'))
    faces = _vals(tags, 'MakerNote Tag 0x4103') or []             # FacePositions: left, top, right, bottom per face
    for i in range(0, len(faces) - 3, 4):
        x0, y0, x1, y1 = faces[i:i + 4]
        if x1 > x0 and y1 > y0 and not (sel and len(sel) == 4 and _overlap((x0, y0, x1, y1), sel) > 0.6):   # not the one focused on
            boxes.append(((x0 + x1) / 2 / fw, (y0 + y1) / 2 / fh, (x1 - x0) / fw, (y1 - y0) / fh, 'face'))
    return boxes


def _nikon(tags, model):
    v = _vals(tags, 'MakerNote AFInfo2')
    if not v or len(v) < 16 or bytes(v[:4]) != b'0100': return []
    method, schema, primary = v[4], v[6], v[7]
    if method == 1 and len(v) >= 0x1c:                         # contrast-detect (LiveView): an area in "AF image" pixels
        for order in ('big', 'little'):                          # maker-note byte order: Nikon writes big-endian
            w_, h_, x, y, aw, ah = (int.from_bytes(bytes(v[o:o + 2]), order) for o in range(0x10, 0x1c, 2))
            if w_ and h_ and 0 < x < w_ and 0 < y < h_ and aw <= w_ and ah <= h_:
                return [(x / w_, y / h_, max(aw, w_ * 0.02) / w_, max(ah, h_ * 0.02) / h_, 'focus')]
        return []
    m = NIKON_51_MAPS.get(model.upper().strip())
    if method != 0 or schema != 1 or not m: return []
    fw, fh, pts = m
    used = [n for n in range(1, 52) if v[8 + (n - 1) // 8] >> ((n - 1) % 8) & 1]
    boxes = []
    for n in sorted(set(used) | ({primary} if primary in AF_POINTS_51 else set())):
        x, y, w, h = pts[AF_POINTS_51[n]]
        boxes.append((x / fw, y / fh, w / fw, h / fh, 'focus' if n == primary or (not primary and len(used) == 1) else 'af'))
    return sorted(boxes, key=lambda b: b[4] != 'af')             # the focus point drawn last, on top


def from_tags(tags, make='', model='', orientation=None):
    """exifread tags (read with details=True: the maker note is needed) -> boxes, or [] when there's nothing."""
    make = (make or str(tags.get('Image Make', ''))).upper()
    model = model or str(tags.get('Image Model', ''))
    if orientation is None: orientation = (_vals(tags, 'Image Orientation') or [1])[0]
    try:
        boxes = _fuji(tags) if 'FUJI' in make else _nikon(tags, model) if 'NIKON' in make else []
    except Exception:
        boxes = []
    return _orient(boxes, orientation)


def read_unrotated(path):
    """(boxes in the sensor's own landscape frame, EXIF orientation) for a RAW file — for a view that may show the
    photo unrotated (an embedded preview) or rotated (a developed result). Fujifilm RAF: from its embedded JPEG."""
    import exifread
    with open(path, 'rb') as f:
        head = f.read(16)
        f.seek(0)
        if head.startswith(b'FUJIFILM'):
            import rawpy
            with rawpy.imread(path) as r:
                t = r.extract_thumb()
                orientation = {0: 1, 3: 3, 5: 8, 6: 6}.get(r.sizes.flip, 1)
            return from_tags(exifread.process_file(io.BytesIO(t.data), details=True), orientation=1), orientation
        tags = exifread.process_file(f, details=True)
        return from_tags(tags, orientation=1), (_vals(tags, 'Image Orientation') or [1])[0]


def read(path):
    """Boxes for a RAW file, in the frame the photo is shown in."""
    boxes, orientation = read_unrotated(path)
    return orient(boxes, orientation)


def orient(boxes, orientation):
    """Turn boxes from the sensor's frame to that of a photo shown with this EXIF orientation."""
    return _orient(boxes, orientation)


def dumps(boxes):
    return json.dumps(boxes, separators=(',', ':'))


def loads(s):
    try:
        return json.loads(s) if s else []
    except ValueError:
        return []
