# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""Photo Library AI: faces, search by meaning, and captions with tags.

  faces     InsightFace's buffalo_l models run directly with ONNX Runtime: SCRFD-10G finds faces and their five
            landmarks (it is the RetinaFace team's successor, so no separate RetinaFace is needed), each face is
            aligned to 112×112 and ArcFace R50 turns it into a 512-number vector. The insightface package itself
            is not used (it has no Windows wheels and needs a C++ compiler).
  meaning   SigLIP2 embeds every image; a typed phrase is embedded with the same model's text tower and the
            nearest images are found by cosine similarity.
  captions  Florence-2 writes a detailed caption; regular expressions over it give tags (source 'ai').

Every model works on the photo's 1024 px thumbnail, so RAW files need no decoding here. Vectors are stored in
the catalog's own SQLite file (faces.vec, image_vectors.vec) and searched with numpy: one file to back up, and
fast enough for a personal library (a few hundred thousand vectors).
"""
import json
import os
import re
import threading
import time

import numpy as np
from PIL import Image

import photolibrary_catalog as L
import photolibrary_setup_models as SM

DEFAULT_MODELS = SM.DEFAULT_MODELS
CAPTION_TASK = '<DETAILED_CAPTION>'
FACE_MATCH = 0.45      # ArcFace cosine: suggest the same person above this
FACE_GROUP = 0.50      # group unnamed faces above this


def list_devices():
    out = []
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append((f'cuda:{i}', f'cuda:{i} — {p.name} ({p.total_memory / 2 ** 30:.0f} GB)'))
    except Exception:
        pass
    return out + [('cpu', 'CPU')]


def _ort_providers(device):
    import onnxruntime as ort
    ort.set_default_logger_severity(3)   # errors only: it warns about plugin providers it doesn't use
    try:
        import torch   # noqa: F401 — loads the CUDA / cuDNN DLLs ONNX Runtime needs
        ort.preload_dlls()
    except Exception:
        pass
    if device.startswith('cuda') and 'CUDAExecutionProvider' in ort.get_available_providers():
        return [('CUDAExecutionProvider', {'device_id': int(device.split(':')[1] or 0)}), 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


# ───────────────────────────────────────────── faces
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def _similarity(src, dst):
    """Least-squares similarity transform (Umeyama) src -> dst, as a 2×3 matrix: InsightFace's face alignment."""
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    U, S, Vt = np.linalg.svd(d0.T @ s0 / len(src))
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: D[1, 1] = -1
    R = U @ D @ Vt
    scale = (S * np.diag(D)).sum() / s0.var(0).sum()
    t = md - scale * R @ ms
    return np.hstack([scale * R, t[:, None]]).astype(np.float32)


def _iou(a, b):
    """Overlap of two (x0, y0, x1, y1) boxes, 0..1."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes, scores, thr):
    order, keep = scores.argsort()[::-1], []
    x0, y0, x1, y1 = boxes.T
    area = (x1 - x0 + 1) * (y1 - y0 + 1)
    while order.size:
        i = order[0]; keep.append(i)
        xx0, yy0 = np.maximum(x0[i], x0[order[1:]]), np.maximum(y0[i], y0[order[1:]])
        xx1, yy1 = np.minimum(x1[i], x1[order[1:]]), np.minimum(y1[i], y1[order[1:]])
        inter = np.maximum(0, xx1 - xx0 + 1) * np.maximum(0, yy1 - yy0 + 1)
        order = order[1:][inter / (area[i] + area[order[1:]] - inter) <= thr]
    return keep


class FaceModel:
    def __init__(self, folder, device):
        import onnxruntime as ort
        prov = _ort_providers(device)
        self.det = ort.InferenceSession(os.path.join(folder, 'det_10g.onnx'), providers=prov)
        self.rec = ort.InferenceSession(os.path.join(folder, 'w600k_r50.onnx'), providers=prov)
        self.det_in, self.rec_in = self.det.get_inputs()[0].name, self.rec.get_inputs()[0].name
        self.on_gpu = 'CUDAExecutionProvider' in self.det.get_providers()

    def detect(self, rgb, thresh=0.5, size=640, min_px=16):
        """RGB uint8 -> [(box x0,y0,x1,y1 in pixels, score, 5×2 landmarks)]."""
        import cv2
        h, w = rgb.shape[:2]
        s = size / max(h, w)
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        canvas = np.zeros((size, size, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        blob = ((canvas.astype(np.float32) - 127.5) / 128).transpose(2, 0, 1)[None]
        outs = self.det.run(None, {self.det_in: blob})
        boxes, scores, kpss = [], [], []
        for i, stride in enumerate((8, 16, 32)):   # SCRFD: 3 strides, 2 anchors per cell; scores, boxes, landmarks
            sc, bb, kp = outs[i], outs[i + 3], outs[i + 6]
            if sc.ndim == 3: sc, bb, kp = sc[0], bb[0], kp[0]
            n = size // stride
            centres = np.stack(np.mgrid[:n, :n][::-1], axis=-1).reshape(-1, 2).astype(np.float32) * stride
            centres = np.repeat(centres, 2, axis=0)
            keep = np.where(sc.ravel() >= thresh)[0]
            if not keep.size: continue
            c, d, k = centres[keep], bb[keep] * stride, kp[keep] * stride
            boxes.append(np.hstack([c - d[:, :2], c + d[:, 2:]]))
            kpss.append(c[:, None, :] + k.reshape(-1, 5, 2))
            scores.append(sc.ravel()[keep])
        if not boxes: return []
        boxes, scores, kpss = np.vstack(boxes) / s, np.concatenate(scores), np.vstack(kpss) / s
        out = []
        for i in _nms(boxes, scores, 0.4):
            b = boxes[i]
            if min(b[2] - b[0], b[3] - b[1]) >= min_px:
                out.append((b, float(scores[i]), kpss[i]))
        return out

    def embed(self, rgb, kps):
        import cv2
        crop = cv2.warpAffine(rgb, _similarity(kps.astype(np.float32), ARCFACE_DST), (112, 112), borderValue=0)
        blob = ((crop.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        v = self.rec.run(None, {self.rec_in: blob})[0][0].astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)


# ───────────────────────────────────────────── SigLIP2
def _features(out):
    """get_*_features returns a tensor in transformers 4 and a model output in 5."""
    import torch
    if isinstance(out, torch.Tensor): return out
    for k in ('pooler_output', 'image_embeds', 'text_embeds'):
        v = getattr(out, k, None)
        if v is not None: return v
    return out[0]


class SigLIP:
    def __init__(self, folder, device):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.dtype = torch.float16 if device.startswith('cuda') else torch.float32
        try:
            m = AutoModel.from_pretrained(folder, dtype=self.dtype)
        except TypeError:
            m = AutoModel.from_pretrained(folder, torch_dtype=self.dtype)
        self.model = m.to(device).eval()
        self.proc = AutoProcessor.from_pretrained(folder)
        self.name = os.path.basename(json.load(open(os.path.join(folder, 'config.json'))).get('_name_or_path', '') or folder) or 'siglip2'
        self.scale = float(self.model.logit_scale.exp().item())
        self.bias = float(self.model.logit_bias.item())

    def images(self, ims):
        import torch
        with torch.inference_mode():
            x = self.proc(images=ims, return_tensors='pt')
            f = _features(self.model.get_image_features(pixel_values=x['pixel_values'].to(self.device, self.dtype)))
            f = torch.nn.functional.normalize(f.float(), dim=-1)
        return f.cpu().numpy()

    def text(self, q):
        import torch
        with torch.inference_mode():
            x = self.proc(text=[q.lower()], padding='max_length', max_length=64, truncation=True, return_tensors='pt')
            x = {k: v.to(self.device) for k, v in x.items() if k in ('input_ids', 'attention_mask')}
            f = _features(self.model.get_text_features(**x))
            f = torch.nn.functional.normalize(f.float(), dim=-1)
        return f.cpu().numpy()[0]

    def prob(self, cos):
        """SigLIP's own sigmoid: how likely the text describes the image."""
        return 1 / (1 + np.exp(-(cos * self.scale + self.bias)))


# ───────────────────────────────────────────── Florence-2
class Florence:
    def __init__(self, folder, device):
        import torch
        from transformers import AutoProcessor
        self.device = device
        self.dtype = torch.float16 if device.startswith('cuda') else torch.float32
        cfg = json.load(open(os.path.join(folder, 'config.json')))
        remote = 'auto_map' in cfg
        if remote:
            from transformers import AutoModelForCausalLM as M
        else:
            from transformers import Florence2ForConditionalGeneration as M
        kw = dict(trust_remote_code=True) if remote else {}
        try:
            m = M.from_pretrained(folder, dtype=self.dtype, **kw)
        except TypeError:
            m = M.from_pretrained(folder, torch_dtype=self.dtype, **kw)
        self.model = m.to(device).eval()
        self.proc = AutoProcessor.from_pretrained(folder, **kw)

    def captions(self, ims, task=CAPTION_TASK):
        import torch
        with torch.inference_mode():
            x = self.proc(text=[task] * len(ims), images=ims, return_tensors='pt')
            ids = self.model.generate(input_ids=x['input_ids'].to(self.device), pixel_values=x['pixel_values'].to(self.device, self.dtype),
                                      max_new_tokens=160, num_beams=3, do_sample=False)
            texts = self.proc.batch_decode(ids, skip_special_tokens=False)
        out = []
        for t, im in zip(texts, ims):
            p = self.proc.post_process_generation(t, task=task, image_size=im.size)
            out.append(_clean_caption(p.get(task, '') if isinstance(p, dict) else str(p)))
        return out


def _clean_caption(s):
    s = re.sub(r'<[^>]+>', '', s).strip()
    s = re.sub(r'^(the image (shows|is)|in this image,? (we can see|there is|there are)|this is an? image of)\s+', '', s, flags=re.I)
    return (s[:1].upper() + s[1:]) if s else s


# ───────────────────────────────────────────── caption -> tags
# canonical tag: pattern over the lower-case caption. Plural and common synonyms map to one tag, so that
# filtering by a tag finds every photo described with any of its words.
TAG_RULES = {
    'people': r'\b(people|persons?|man|men|woman|women|boys?|girls?|child(ren)?|kids?|bab(y|ies)|toddlers?|crowd|couple|family|lady|gentleman|guy)\b',
    'portrait': r'\b(portrait|selfie|close-?up of (a|the) (man|woman|person|face)|smiling)\b',
    'group': r'\b(group of (people|friends|men|women|children)|crowd|team)\b',
    'dog': r'\b(dogs?|pupp(y|ies))\b', 'cat': r'\b(cats?|kittens?)\b', 'bird': r'\b(birds?|gulls?|seagulls?|ducks?|swans?|geese|goose|owls?|eagles?|hawks?|pigeons?|robins?)\b',
    'horse': r'\b(horses?|pon(y|ies))\b', 'cattle': r'\b(cows?|cattle|bulls?|calf|calves)\b', 'sheep': r'\b(sheep|lambs?)\b',
    'insect': r'\b(insects?|butterfl(y|ies)|bees?|dragonfl(y|ies)|moths?|spiders?)\b', 'deer': r'\b(deer|stags?)\b', 'fish': r'\b(fish|fishes)\b',
    'animal': r'\b(animals?|wildlife|squirrels?|rabbits?|hares?|fox(es)?|hedgehogs?|badgers?|otters?|mice|mouse|rats?|monkeys?|bears?|lions?|tigers?|elephants?|giraffes?|zebras?|goats?|pigs?|seals?)\b',
    'sea': r'\b(sea|ocean|seascape|waves?|surf)\b', 'beach': r'\b(beach(es)?|sand|shore(line)?|coast(line)?)\b', 'lake': r'\b(lakes?|pond|reservoir|loch)\b',
    'river': r'\b(rivers?|stream|creek|canal)\b', 'waterfall': r'\b(waterfalls?|cascade)\b', 'water': r'\b(water)\b',
    'mountain': r'\b(mountains?|peaks?|summit|alps|mountain range)\b', 'hills': r'\b(hills?|hillside|moor(land)?|valley)\b', 'rocks': r'\b(rocks?|rocky|cliffs?|boulders?)\b',
    'sky': r'\b(sky|skies)\b', 'clouds': r'\b(clouds?|cloudy|overcast)\b', 'sunset': r'\b(sunset|dusk|golden hour)\b', 'sunrise': r'\b(sunrise|dawn)\b',
    'night': r'\b(night|nighttime|starry|stars|moon)\b', 'snow': r'\b(snow|snowy|ice|icy|frost|winter)\b', 'fog': r'\b(fog|foggy|mist|misty|haze)\b',
    'forest': r'\b(forest|woods|woodland|jungle)\b', 'trees': r'\b(trees?)\b', 'flowers': r'\b(flowers?|blossoms?|roses?|tulips?|daisies|bloom(ing)?)\b',
    'garden': r'\b(gardens?|park)\b', 'field': r'\b(fields?|meadow|farmland|crops?|grass(land)?)\b', 'landscape': r'\b(landscape|countryside|scenery|vista)\b',
    'building': r'\b(buildings?|architecture|facade)\b', 'church': r'\b(church(es)?|cathedral|chapel|abbey|steeple|spire)\b', 'castle': r'\b(castles?|fort(ress)?|ruins?)\b',
    'bridge': r'\b(bridges?)\b', 'tower': r'\b(towers?|lighthouse)\b', 'house': r'\b(house|houses|cottage|home)\b', 'street': r'\b(street|road|alley|sidewalk|pavement)\b',
    'city': r'\b(city|cityscape|skyline|town|urban|downtown)\b', 'interior': r'\b(room|interior|kitchen|bedroom|living room|hall(way)?|inside)\b',
    'car': r'\b(cars?|vehicles?|truck|van|suv)\b', 'boat': r'\b(boats?|ships?|yachts?|sailboats?|ferry|kayak|canoe)\b', 'train': r'\b(trains?|railway|locomotive|tram)\b',
    'aircraft': r'\b(planes?|aircraft|airplanes?|jet|helicopter)\b', 'bicycle': r'\b(bicycles?|bikes?|cyclist)\b', 'motorcycle': r'\b(motorcycles?|motorbikes?|scooter)\b',
    'food': r'\b(food|meal|dish|plate of|cake|bread|pizza|fruit|vegetables?|dinner|lunch|breakfast)\b', 'drink': r'\b(drinks?|coffee|tea|wine|beer|cup|glass of)\b',
    'sport': r'\b(sports?|football|soccer|rugby|tennis|golf|cricket|runners?|marathon|racing|stadium)\b', 'music': r'\b(music|concert|guitar|piano|band|singer|stage)\b',
    'wedding': r'\b(wedding|bride|groom)\b', 'party': r'\b(party|celebration|birthday|festival)\b',
    'text': r'\b(text|sign|signs|writing|words|letters|poster|logo)\b', 'statue': r'\b(statues?|sculptures?|monument|memorial)\b',
    'painting': r'\b(painting|artwork|mural|drawing)\b', 'reflection': r'\b(reflections?|reflected|mirror)\b',
    'black and white': r'\b(black and white|black-and-white|monochrome|grayscale|greyscale)\b', 'aerial': r'\b(aerial|from above|bird\'s eye)\b',
    'macro': r'\b(macro|close-?up)\b', 'silhouette': r'\b(silhouettes?)\b',
}
_TAG_RX = {k: re.compile(v, re.I) for k, v in TAG_RULES.items()}


def caption_tags(caption):
    return sorted(k for k, rx in _TAG_RX.items() if rx.search(caption or ''))


# ───────────────────────────────────────────── vector search
class ImageIndex:
    """Every image vector of one model as a float16 matrix in RAM, reloaded when the table changes."""
    def __init__(self, cat):
        self.cat, self._key, self.ids, self.mat = cat, None, np.zeros(0, np.int64), None

    def _load(self):
        r = self.cat.q1('SELECT COUNT(*), MAX(rowid), MIN(model) FROM image_vectors')
        key = tuple(r)
        if key != self._key:
            rows = self.cat.q('SELECT photo_id, vec FROM image_vectors')
            self.ids = np.array([x[0] for x in rows], np.int64)
            self.mat = np.stack([np.frombuffer(x[1], np.float16) for x in rows]) if rows else None
            self._key = key

    def vector(self, pid):
        r = self.cat.q1('SELECT vec FROM image_vectors WHERE photo_id = ?', (pid,))
        return np.frombuffer(r[0], np.float16).astype(np.float32) if r else None

    def search(self, vec, top=300):
        """-> [(photo id, cosine)], best first."""
        self._load()
        if self.mat is None: return []
        sims = np.empty(len(self.ids), np.float32)
        step = 65536
        for i in range(0, len(self.ids), step):
            sims[i:i + step] = self.mat[i:i + step].astype(np.float32) @ vec.astype(np.float32)
        k = min(top, len(sims))
        best = np.argpartition(-sims, k - 1)[:k]
        best = best[np.argsort(-sims[best])]
        return [(int(self.ids[i]), float(sims[i])) for i in best]


# ───────────────────────────────────────────── people
def _face_matrix(cat, where_sql='1', params=()):
    rows = cat.q(f'SELECT id, person_id, vec FROM faces WHERE {where_sql}', params)
    if not rows: return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 512), np.float32)
    return (np.array([r[0] for r in rows]), np.array([r[1] or 0 for r in rows]),
            np.stack([np.frombuffer(r[2], np.float32) for r in rows]))


def suggest_people(cat, thresh=FACE_MATCH):
    """Faces nobody has named: give each the person whose user-named faces it is most like (assigned='auto').
    Returns how many faces got a different suggestion than before."""
    nid, npid, nv = _face_matrix(cat, "assigned = 'user'")
    uid, upid, uv = _face_matrix(cat, "assigned IS NULL OR assigned = 'auto'")
    if not len(nid) or not len(uid): return 0
    updates = []
    for i in range(0, len(uid), 4096):
        s = uv[i:i + 4096] @ nv.T
        j = s.argmax(1)
        best = s[np.arange(len(j)), j]
        for k in range(len(j)):
            updates.append((int(npid[j[k]]) if best[k] >= thresh else None, 'auto' if best[k] >= thresh else None, int(uid[i + k])))
    with cat.tx() as c:
        c.executemany('UPDATE faces SET person_id = ?, assigned = ? WHERE id = ?', updates)
    return sum(1 for u, before in zip(updates, upid) if u[0] and u[0] != before)


def group_faces(cat, thresh=FACE_GROUP):
    """Group faces that have no person by similarity (union-find over cosine > thresh). Returns the group count."""
    ids, _, v = _face_matrix(cat, 'person_id IS NULL')
    n = len(ids)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(0, n, 2048):
        s = v[i:i + 2048] @ v.T
        a, b = np.nonzero(s > thresh)
        for x, y in zip(a + i, b):
            if x < y:
                rx, ry = find(x), find(y)
                if rx != ry: parent[max(rx, ry)] = min(rx, ry)
    roots = [find(i) for i in range(n)]
    with cat.tx() as c:
        c.execute('UPDATE faces SET cluster = NULL')
        c.executemany('UPDATE faces SET cluster = ? WHERE id = ?', [(int(ids[r]), int(ids[i])) for i, r in enumerate(roots)])
    return len(set(roots))


def name_faces(cat, face_ids, name):
    with cat.tx() as c:
        c.execute('INSERT OR IGNORE INTO persons (name) VALUES (?)', (name.strip(),))
        pid = c.execute('SELECT id FROM persons WHERE name = ?', (name.strip(),)).fetchone()[0]
        c.executemany("UPDATE faces SET person_id = ?, assigned = 'user', cluster = NULL WHERE id = ?", [(pid, int(f)) for f in face_ids])
    return pid


def unname_faces(cat, face_ids):
    with cat.tx() as c:
        c.executemany('UPDATE faces SET person_id = NULL, assigned = NULL WHERE id = ?', [(int(f),) for f in face_ids])
        c.execute('DELETE FROM persons WHERE id NOT IN (SELECT person_id FROM faces WHERE person_id IS NOT NULL)')


# ───────────────────────────────────────────── the runner
class AI:
    """Loads models on first use (on the worker thread) and runs them over photos."""
    def __init__(self, cat, models_dir=DEFAULT_MODELS, device='cuda:0'):
        self.cat, self.models_dir, self.device = cat, models_dir, device
        self.faces = self.siglip = self.florence = None
        self.index = ImageIndex(cat)
        self.lock = threading.RLock()

    def configure(self, models_dir, device):
        if (models_dir, device) != (self.models_dir, self.device):
            self.unload()
            self.models_dir, self.device = models_dir, device

    def present(self):
        return SM.present(self.models_dir)

    def unload(self):
        with self.lock:
            self.faces = self.siglip = self.florence = None
            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except Exception:
                pass

    def _need(self, part, job=None):
        if not self.present()[part]:
            raise RuntimeError(f'The {SM.PARTS[part]} model is not in {self.models_dir} — Settings & AI → Download models.')
        with self.lock:
            t0 = time.time()
            if part == 'faces' and self.faces is None:
                self.faces = FaceModel(os.path.join(self.models_dir, 'buffalo_l'), self.device)
                if job: job.log('info', f'face models loaded ({"GPU" if self.faces.on_gpu else "CPU"}, {time.time() - t0:.1f} s)')
            elif part == 'siglip' and self.siglip is None:
                self.siglip = SigLIP(os.path.join(self.models_dir, 'siglip2'), self.device)
                if job: job.log('info', f'SigLIP2 loaded on {self.device} ({time.time() - t0:.1f} s)')
            elif part == 'florence' and self.florence is None:
                self.florence = Florence(os.path.join(self.models_dir, 'florence2'), self.device)
                if job: job.log('info', f'Florence-2 loaded on {self.device} ({time.time() - t0:.1f} s)')
        return getattr(self, part)

    def _thumb(self, pid):
        p = self.cat.thumb_path(pid, 1024)
        if not os.path.exists(p): p = self.cat.best_thumb(pid, 1024)
        return Image.open(p).convert('RGB') if p else None

    def run(self, ids, job, faces=True, embed=True, caption=True, batch=8):
        """Run the chosen models over these photos (their thumbnails must exist)."""
        want = (L.AI_FACES if faces else 0) | (L.AI_EMBED if embed else 0) | (L.AI_CAPTION if caption else 0)
        if not want or not ids: return
        fm = self._need('faces', job) if faces else None
        sg = self._need('siglip', job) if embed else None
        fl = self._need('florence', job) if caption else None
        t0, nf, ncap = time.time(), 0, 0
        for i in range(0, len(ids), batch):
            job.check()
            part = ids[i:i + batch]
            ims = [(pid, self._thumb(pid)) for pid in part]
            ims = [(pid, im) for pid, im in ims if im is not None]
            if not ims: continue
            face_rows, vec_rows, caps = [], [], []
            if fm:
                for pid, im in ims:
                    a = np.asarray(im)
                    h, w = a.shape[:2]
                    for box, score, kps in fm.detect(a):
                        v = fm.embed(a, kps)
                        x0, y0, x1, y1 = (float(t) for t in box)   # numpy scalars would be stored as BLOBs
                        face_rows.append((pid, max(0.0, x0 / w), max(0.0, y0 / h), min(1.0, x1 / w), min(1.0, y1 / h), score, v.tobytes()))
            if sg:
                vecs = sg.images([im for _, im in ims])
                vec_rows = [(pid, sg.name, v.astype(np.float16).tobytes()) for (pid, _), v in zip(ims, vecs)]
            if fl:
                caps = list(zip([pid for pid, _ in ims], fl.captions([im for _, im in ims])))
            with self.cat.tx() as c:
                if fm:
                    # faces found again keep who they are: the old face that overlaps a new one most hands over its name
                    old = {}
                    for pid, _ in ims:
                        old[pid] = c.execute('SELECT x0, y0, x1, y1, person_id, assigned FROM faces WHERE photo_id = ? AND person_id IS NOT NULL',
                                             (pid,)).fetchall()
                    rows = []
                    for r in face_rows:
                        best = max(old.get(r[0], []), key=lambda o: _iou(o[:4], r[1:5]), default=None)
                        keep = best if best is not None and _iou(best[:4], r[1:5]) > 0.4 else (None, None, None, None, None, None)
                        rows.append(r + (keep[4], keep[5]))
                    c.executemany('DELETE FROM faces WHERE photo_id = ?', [(pid,) for pid, _ in ims])
                    c.executemany('INSERT INTO faces (photo_id, x0, y0, x1, y1, score, vec, person_id, assigned) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', rows)
                    nf += len(face_rows)
                if vec_rows:
                    c.executemany('INSERT OR REPLACE INTO image_vectors (photo_id, model, vec) VALUES (?, ?, ?)', vec_rows)
                for pid, cap in caps:
                    c.execute('UPDATE photos SET caption = ? WHERE id = ?', (cap, pid))
                    self.cat.set_ai_tags(c, pid, caption_tags(cap))
                    ncap += 1
                c.executemany('UPDATE photos SET ai = ai | ? WHERE id = ?', [(want, pid) for pid, _ in ims])
            done = min(len(ids), i + batch)
            job.progress(done / len(ids), f'AI {done}/{len(ids)}{L._eta(t0, done, len(ids))}')
            job.post('ai', [pid for pid, _ in ims])
        if fm:
            n = suggest_people(self.cat)
            self.cat.sync_person_tags()
            job.log('ok', f'{nf} faces found' + (f', {n} matched to people you have named' if n else ''))
        if caption: job.log('ok', f'{ncap} captions written')
        job.log('ok', f'AI done for {len(ids)} photos in {time.time() - t0:.0f} s')

    def search_text(self, text, job=None, top=300, min_prob=1e-3):
        sg = self._need('siglip', job)
        hits = self.index.search(sg.text(text), top)
        good = [(i, s) for i, s in hits if sg.prob(s) >= min_prob]
        return good if len(good) >= 12 else hits[:max(12, len(good))]

    def search_similar(self, pid, top=300):
        v = self.index.vector(pid)
        if v is None: return []
        return self.index.search(v, top)
