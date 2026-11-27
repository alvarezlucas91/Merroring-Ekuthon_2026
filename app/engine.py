"""Motor de la demo: recibe muestras de cada celular, arma ventanas, calcula embeddings,
compara con prototipos y aprende actividades nuevas con pocos segundos.

No guarda señal cruda en disco: solo prototipos (vectores) y su nombre.
"""
import json
import os
import threading
import time
import uuid
from collections import Counter, deque

import numpy as np

from dsp import G, TARGET_HZ, WIN, resample_uniform
from encoders import load_encoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.environ.get("PROTOTYPES_FILE", os.path.join(ROOT, "data", "prototypes.json"))
PALETTE = ["#2f6fdb", "#e0663a", "#1f9e74", "#c43d8c", "#d9a21b", "#6a4fd0", "#3aa6c4", "#8c6d3a",
           "#d44b4b", "#5c8a2e"]
UNKNOWN = "Desconocida"
STILL = "Sin movimiento"
BASE_ACTIVITY_LABELS = {
    "Caminar": "Walking",
    "Trotar": "Jogging",
    "Parado": "Standing",
    "Aplaudir": "Clapping",
    "Cepillarse los dientes": "Brushing teeth",
}


def activity_label(name):
    if name == UNKNOWN:
        return "Unknown"
    if name == STILL:
        return "Still"
    if name and name.endswith(" (base)"):
        base = BASE_ACTIVITY_LABELS.get(name.removesuffix(" (base)"))
        if base:
            return f"{base} (base)"
    return name


class Participant:
    def __init__(self, pid, name):
        self.id, self.name = pid, name
        self.raw = deque(maxlen=TARGET_HZ * 20 * 4)  # ~20 s a hasta 120 Hz: (t_seg, x, y, z, |giro| °/s) en g
        self.last_seen = time.time()
        self.connected = True
        self.activity, self.score, self.xy = None, 0.0, None
        self.votes = deque(maxlen=3)
        self.trail = deque(maxlen=8)
        self.rate_hz = 0.0
        self.recording = None  # dict(name, until, start, windows)
        self.last_emb = None

    def window(self):
        """Últimos 5 s remuestreados a 30 Hz, forma (3, 150), o None si no alcanza."""
        if len(self.raw) < 10:
            return None
        arr = np.array(self.raw)[:, :4]
        if arr[-1, 0] - arr[0, 0] < WIN / TARGET_HZ:
            return None
        tail = arr[arr[:, 0] >= arr[-1, 0] - WIN / TARGET_HZ - 0.5]
        segs = resample_uniform(tail[:, 0], tail[:, 1:])
        if not segs or len(segs[-1]) < WIN:
            return None
        span = arr[-1, 0] - arr[0, 0]
        self.rate_hz = (len(arr) - 1) / span if span > 0 else 0.0
        return segs[-1][-WIN:].T.astype(np.float32)


class Engine:
    def __init__(self, encoder="hybrid", threshold=None, record_seconds=20):
        self.encoder_name = encoder
        self.enc = load_encoder(encoder)
        th_file = os.path.join(ROOT, "models", "thresholds.json")
        default_th = 0.6
        if os.path.exists(th_file):
            default_th = json.load(open(th_file)).get(encoder, default_th)
        self.threshold = float(threshold if threshold is not None else default_th)
        self.record_seconds = record_seconds
        self.participants = {}
        # actividades: nombre -> {"color", "protos": [{"id","vec","by","pts"}]}
        self.activities = {}
        self.map_basis = None  # (mean, comps(2,d), scale)
        self.lock = threading.Lock()
        self.load()

    # ---------------------------------------------------------------- persistencia
    def save(self):
        out = {n: {"color": a["color"], "protos": [{"id": p["id"], "by": p["by"], "vec": p["vec"].tolist(),
                                                    "pts": p["pts"].tolist()} for p in a["protos"]]}
               for n, a in self.activities.items()}
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump({"encoder": self.encoder_name, "activities": out}, f)

    def load(self):
        if not os.path.exists(STATE_FILE):
            return
        d = json.load(open(STATE_FILE))
        if d.get("encoder") != self.encoder_name:
            print(f"prototypes.json es de otro encoder ({d.get('encoder')}); se ignora")
            return
        for n, a in d["activities"].items():
            self.activities[n] = {"color": a["color"], "protos": [
                {"id": p["id"], "by": p["by"], "vec": np.array(p["vec"], np.float32),
                 "pts": np.array(p["pts"], np.float32)} for p in a["protos"]]}
        self._refit_map()

    # ---------------------------------------------------------------- participantes
    def join(self, pid, name):
        with self.lock:
            p = self.participants.get(pid)
            if p is None:
                p = self.participants[pid] = Participant(pid, name or f"Participant {len(self.participants) + 1}")
            else:
                p.name = name or p.name
            p.connected, p.last_seen = True, time.time()
            return p

    def leave(self, pid):
        with self.lock:
            if pid in self.participants:
                self.participants[pid].connected = False

    def ingest(self, pid, samples):
        """samples: lista de [t_ms, ax, ay, az] o [t_ms, ax, ay, az, gx, gy, gz].

        Aceleración en m/s² (ya con la convención de signo calibrada); giroscopio en °/s.
        """
        p = self.participants.get(pid)
        if p is None or not samples:
            return
        p.last_seen = time.time()
        for s in samples:
            if len(s) not in (4, 7):
                continue
            try:
                v = [float(x) if x is not None else float("nan") for x in s]
            except (TypeError, ValueError):
                continue
            if not all(np.isfinite(v[:4])):
                continue
            w = float(np.sqrt(v[4] ** 2 + v[5] ** 2 + v[6] ** 2)) if len(v) == 7 else float("nan")
            p.raw.append((v[0] / 1000.0, v[1] / G, v[2] / G, v[3] / G, w))

    # ---------------------------------------------------------------- aprendizaje
    def start_recording(self, pid, name, seconds=None):
        name = (name or "").strip()[:40]
        p = self.participants.get(pid)
        if not p or not name:
            return False
        secs = seconds or self.record_seconds
        p.recording = {"name": name, "start": time.time() + 3, "until": time.time() + 3 + secs, "windows": []}
        return True

    def cancel_recording(self, pid):
        p = self.participants.get(pid)
        if p:
            p.recording = None

    def _finish_recording(self, p):
        rec, p.recording = p.recording, None
        W = rec["windows"]
        if len(W) < 2:
            return None
        Z = self.enc(np.stack(W))
        vec = Z.mean(0)
        vec /= np.linalg.norm(vec) + 1e-9
        with self.lock:
            a = self.activities.setdefault(rec["name"], {"color": PALETTE[len(self.activities) % len(PALETTE)],
                                                          "protos": []})
            a["protos"].append({"id": uuid.uuid4().hex[:8], "by": p.name, "vec": vec.astype(np.float32),
                                "pts": Z[:: max(1, len(Z) // 12)][:12]})
            self._refit_map()
            self.save()
        return rec["name"]

    def delete_activity(self, name):
        with self.lock:
            self.activities.pop(name, None)
            self._refit_map()
            self.save()

    def reset(self):
        with self.lock:
            self.activities.clear()
            self.map_basis = None
            self.save()

    def load_base(self, names=tuple(BASE_ACTIVITY_LABELS), per_class=8):
        """Precarga actividades desde WISDM (reloj en la muñeca). Experimental: el dominio es distinto."""
        f = os.path.join(ROOT, "data", "wisdm_watch_30hz.npz")
        if not os.path.exists(f):
            return False
        d = np.load(f)
        X, y, s, cls = d["X"], d["y"], d["subject"], list(d["names"])
        rng = np.random.default_rng(0)
        for n in names:
            c = cls.index(n)
            subs = rng.choice(np.unique(s[y == c]), per_class, replace=False)
            for sub in subs:
                m = np.where((y == c) & (s == sub))[0][:8]  # ~20 s
                Z = self.enc(X[m])
                vec = Z.mean(0)
                vec /= np.linalg.norm(vec) + 1e-9
                with self.lock:
                    a = self.activities.setdefault(f"{n} (base)", {
                        "color": PALETTE[len(self.activities) % len(PALETTE)], "protos": []})
                    a["protos"].append({"id": uuid.uuid4().hex[:8], "by": f"WISDM {sub}", "vec": vec,
                                        "pts": Z[:3]})
        with self.lock:
            self._refit_map()
            self.save()
        return True

    # ---------------------------------------------------------------- mapa 2D
    def _refit_map(self):
        pts = [p["pts"] for a in self.activities.values() for p in a["protos"]]
        if not pts:
            self.map_basis = None
            return
        P = np.concatenate(pts)
        mu = P.mean(0)
        if len(P) < 3:
            self.map_basis = None
            return
        _, _, vt = np.linalg.svd(P - mu, full_matrices=False)
        comps = vt[:2]
        proj = (P - mu) @ comps.T
        scale = np.abs(proj).max(0) + 1e-6
        self.map_basis = (mu, comps, scale)

    def _project(self, z):
        if self.map_basis is None:
            return None
        mu, comps, scale = self.map_basis
        xy = ((z - mu) @ comps.T) / scale
        return np.clip(xy, -1.3, 1.3).tolist()

    # ---------------------------------------------------------------- ciclo principal
    def tick(self):
        """Se llama 1 vez por segundo. Clasifica a todos y avanza las grabaciones."""
        now = time.time()
        live = [p for p in list(self.participants.values()) if p.connected and now - p.last_seen < 3]
        wins, owners = [], []
        for p in live:
            w = p.window()
            if w is not None:
                wins.append(w)
                owners.append(p)
        if wins:
            Z = self.enc(np.stack(wins))
            with self.lock:
                names = list(self.activities)
                protos = [(n, pr["vec"]) for n in names for pr in self.activities[n]["protos"]]
            for p, w, z in zip(owners, wins, Z):
                p.last_emb = z
                moving = float(np.linalg.norm(w, axis=0).std()) > 0.02
                if not protos:
                    act, score = None, 0.0
                else:
                    sims = {}
                    for n, v in protos:
                        sims[n] = max(sims.get(n, -1.0), float(z @ v))
                    act, score = max(sims.items(), key=lambda kv: kv[1])
                    if score < self.threshold:
                        act = UNKNOWN
                if not moving and act in (None, UNKNOWN):
                    act = STILL
                p.votes.append(act)
                voted = Counter(p.votes).most_common(1)[0][0]
                p.activity, p.score = voted, score
                p.xy = self._project(z)
                if p.xy:
                    p.trail.append(p.xy)
                rec = p.recording
                if rec and rec["start"] <= now <= rec["until"]:
                    rec["windows"].append(w)
        finished = []
        for p in live:
            if p.recording and now > p.recording["until"]:
                n = self._finish_recording(p)
                if n:
                    finished.append((p.id, n))
        return finished

    # ---------------------------------------------------------------- estado para el panel
    def snapshot(self):
        now = time.time()
        with self.lock:
            acts = []
            for n, a in self.activities.items():
                pts = [self._project(pt) for pr in a["protos"] for pt in pr["pts"]]
                cen = None
                if pts and pts[0] is not None:
                    cen = np.mean(np.array(pts), axis=0).tolist()
                acts.append({"name": n, "label": activity_label(n), "color": a["color"], "n": len(a["protos"]),
                             "by": sorted({pr["by"] for pr in a["protos"]}), "points": pts, "centroid": cen})
        parts = []
        for p in self.participants.values():
            if not p.connected and now - p.last_seen > 60:
                continue
            rec = None
            if p.recording:
                r = p.recording
                rec = {"name": r["name"], "countdown": max(0, round(r["start"] - now)),
                       "remaining": max(0, round(r["until"] - max(now, r["start"]))),
                       "windows": len(r["windows"])}
            parts.append({"id": p.id, "name": p.name, "online": p.connected and now - p.last_seen < 3,
                          "activity": p.activity, "activity_label": activity_label(p.activity),
                          "score": round(p.score, 3), "xy": p.xy,
                          "trail": list(p.trail), "rate": round(p.rate_hz), "recording": rec})
        return {"type": "state", "threshold": self.threshold, "encoder": self.encoder_name,
                "record_seconds": self.record_seconds, "activities": acts, "participants": parts,
                "colors": {a["name"]: a["color"] for a in acts}}

    def phone_state(self, pid):
        p = self.participants.get(pid)
        if not p:
            return {"type": "state"}
        now = time.time()
        rec = None
        if p.recording:
            r = p.recording
            rec = {"name": r["name"], "countdown": max(0, round(r["start"] - now)),
                   "remaining": max(0, round(r["until"] - max(now, r["start"])))}
        color = self.activities.get(p.activity, {}).get("color") if p.activity else None
        return {"type": "state", "activity": p.activity, "activity_label": activity_label(p.activity),
                "score": round(p.score, 3), "color": color,
                "recording": rec, "rate": round(p.rate_hz)}
