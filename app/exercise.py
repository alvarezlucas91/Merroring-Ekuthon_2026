"""Modo "Ejercicio guiado": mide con qué precisión una persona imita un movimiento de referencia.

Se pueden guardar varios movimientos de referencia. La referencia se graba a mano (Grabar referencia,
Terminar, Guardar). La imitación se cierra sola (ver Session) o con "Cerrar repetición".

Señales (no dependen de cómo esté puesto el celular):
  ángulo   grados que rotó el celular respecto de la posición de reposo (dirección de la gravedad)
  giro     velocidad de rotación en °/s (giroscopio; si no hay, se estima desde la gravedad)
  sacudida aceleración sin gravedad, en g

Fases: en la referencia, las pausas a mitad de movimiento (giro casi nulo) dividen la repetición.
Si no hay pausas, se divide en ida y vuelta en el punto de máximo ángulo.

Comparación: DTW (Dynamic Time Warping) alinea la curva del intento con la de la referencia.
Con ese alineamiento se ubica cada límite de fase en el intento y se comparan duraciones.
Puntaje con tempo = 40% tiempo + 30% amplitud + 30% forma (WEIGHTS).
Puntaje sin tempo = 60% forma + 40% amplitud (WEIGHTS_FREE): solo cuánto se parece, sin importar la velocidad.
"""
import json
import os
import time
from collections import deque

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_FILE = os.environ.get("EXERCISE_FILE", os.path.join(ROOT, "data", "exercises.json"))
OLD_REF_FILE = os.path.join(ROOT, "data", "exercise_reference.json")  # versión anterior (una sola referencia)

# ------------------------------------------------------------------ parámetros (ajustables desde el panel)
CFG = {
    "still_gyro": 18.0,     # °/s: por debajo se considera quieto
    "still_acc": 0.06,      # g de sacudida: por debajo se considera quieto
    "calib_sec": 1.5,       # segundos quieto para fijar el reposo
    "start_angle": 8.0,     # ° desde el reposo para considerar que empezó
    "start_gyro": 45.0,     # °/s para considerar que empezó (rotaciones horizontales)
    "end_angle": 10.0,      # ° para considerar que volvió al reposo
    "end_still_sec": 0.35,  # imitación: segundos quieto en el inicio para cerrar (si ya hizo todos los tramos)
    "vel_frac": 0.10,       # una fase empieza/termina al cruzar el 10% de la velocidad pico
    "hold_gyro_min": 6.0,   # °/s: piso del umbral de pausa (temblor normal al sostener)
    "hold_min_sec": 0.4,    # duración mínima de una pausa
    "min_amp": 15.0,        # ° mínimos para que cuente como repetición
    "max_rep_sec": 60.0,    # imitación: se descarta si una repetición dura más que esto
    "ref_max_sec": 120.0,   # referencia: se cierra sola si dura más que esto
    "long_still_min": 2.5,  # imitación: quieto en cualquier posición al menos esto para cerrar...
    "long_still_factor": 1.5,  # ...y al menos 1,5 veces la pausa más larga de la referencia
    "edge_angle": 1.0,      # ° desde el reposo para buscar hacia atrás el inicio
    "strict_time": 1.5,     # 1 = lineal; más alto castiga más los errores de tiempo
    "strict_amp": 1.5,      # ídem para la amplitud
}
WEIGHTS = {"tiempo": 0.4, "amplitud": 0.3, "forma": 0.3}
WEIGHTS_FREE = {"amplitud": 0.4, "forma": 0.6}
RES_HZ = 50       # remuestreo de cada repetición
_trapz = getattr(np, "trapezoid", None) or np.trapz
DTW_HZ = 10       # resolución del alineamiento
FINISH_HINT = "Return to the starting position and hold still to end the rep."
LEGACY_PHASE_LABELS = {"Ida": "Outward", "Vuelta": "Return", "Movimiento": "Movement", "Pausa": "Pause"}


def _translate_legacy_phases(phases):
    for phase in phases:
        label, _, number = phase[0].partition(" ")
        if label in LEGACY_PHASE_LABELS and (not number or number.isdecimal()):
            phase[0] = LEGACY_PHASE_LABELS[label] + (f" {number}" if number else "")


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _ema(prev, x, dt, tau):
    if prev is None or not np.isfinite(prev):
        return x
    a = dt / (tau + dt)
    return prev + a * (x - prev)


class SignalTracker:
    """Procesa muestras crudas una por una: gravedad filtrada, giro, sacudida, ángulo."""

    def __init__(self):
        self.g = None
        self.t = None
        self.w = None
        self.acc = None
        self.rest = None

    def push(self, t, a, w_gyro):
        a = np.asarray(a, dtype=np.float64)
        if self.g is None:
            self.g, self.t = a.copy(), t
            return None
        dt = t - self.t
        if dt <= 0 or dt > 0.5:
            self.t = t
            return None
        g_old = _unit(self.g)
        alpha = dt / (0.15 + dt)
        self.g = self.g + alpha * (a - self.g)
        g_new = _unit(self.g)
        if np.isfinite(w_gyro):
            w_raw = w_gyro
        else:  # sin giroscopio: velocidad angular desde el cambio de dirección de la gravedad
            w_raw = np.degrees(np.arccos(np.clip(g_old @ g_new, -1, 1))) / dt
        self.w = _ema(self.w, w_raw, dt, 0.08)
        self.acc = _ema(self.acc, float(np.linalg.norm(a - self.g)), dt, 0.08)
        self.t = t
        ang = float(np.degrees(np.arccos(np.clip(g_new @ self.rest, -1, 1)))) if self.rest is not None else 0.0
        return {"t": t, "dir": g_new, "angle": ang, "w": float(self.w), "acc": float(self.acc)}


def _resample(rows, hz=RES_HZ):
    """rows: lista de dict con t, angle, w, acc -> dict de arrays uniformes con t desde 0."""
    t = np.array([r["t"] for r in rows])
    t = t - t[0]
    grid = np.arange(0, t[-1] + 1e-9, 1.0 / hz)
    out = {"t": grid}
    for k in ("angle", "w", "acc"):
        out[k] = np.interp(grid, t, np.array([r[k] for r in rows]))
    return out


def trim(c):
    """Recorta la repetición donde la velocidad de giro cruza el 10% de su pico (inicio y fin reales)."""
    thr = max(CFG["vel_frac"] * float(c["w"].max()), CFG["hold_gyro_min"] * 0.7)
    idx = np.where(c["w"] >= thr)[0]
    if len(idx) < 2:
        return c
    a, b = idx[0], idx[-1] + 1
    out = {k: v[a:b].copy() for k, v in c.items()}
    out["t"] = out["t"] - out["t"][0]
    return out


def find_phases(c):
    """Divide la curva de referencia en fases. Devuelve lista de (nombre, inicio_s, fin_s, tipo)."""
    t, w, ang = c["t"], c["w"], c["angle"]
    dur = t[-1]
    hold = w < max(CFG["vel_frac"] * float(w.max()), CFG["hold_gyro_min"])
    # tramos quietos internos (no los extremos, que son reposo)
    segs, i = [], 0
    while i < len(t):
        if hold[i]:
            j = i
            while j + 1 < len(t) and hold[j + 1]:
                j += 1
            if t[j] - t[i] >= CFG["hold_min_sec"] and i > 0 and j < len(t) - 1:
                segs.append((t[i], t[j]))
            i = j + 1
        else:
            i += 1
    bounds, kinds = [0.0], []
    for a, b in segs:
        bounds += [a, b]
        kinds += ["mov", "pausa"]
    bounds.append(dur)
    kinds.append("mov")
    if not segs:  # sin pausas: ida y vuelta en el máximo
        tp = float(t[int(np.argmax(ang))])
        if 0.15 * dur < tp < 0.85 * dur:
            bounds, kinds = [0.0, tp, dur], ["mov", "mov"]
    phases, nm, np_ = [], 0, 0
    nmov = kinds.count("mov")
    # "Ida/Vuelta" solo si la pausa del medio no es en la posición inicial
    at_rest_pause = any(float(np.mean(ang[(t >= a) & (t <= b)])) < CFG["end_angle"] for a, b in segs)
    for k, (a, b) in zip(kinds, zip(bounds[:-1], bounds[1:])):
        if k == "mov":
            nm += 1
            if nmov == 2 and not at_rest_pause:
                name = "Outward" if nm == 1 else "Return"
            elif nmov == 1:
                name = "Movement"
            else:
                name = f"Movement {nm}"
        else:
            np_ += 1
            name = "Pause" if kinds.count("pausa") == 1 else f"Pause {np_}"
        phases.append([name, float(a), float(b), k])
    return phases


def count_bursts(c):
    """Cantidad de tramos de movimiento separados por pausas (sin dividir ida y vuelta)."""
    w, t = c["w"], c["t"]
    if len(w) < 3:
        return 0
    moving = w >= max(CFG["vel_frac"] * float(w.max()), CFG["hold_gyro_min"])
    n, i = 0, 0
    while i < len(w):
        if moving[i]:
            j = i
            while j + 1 < len(w) and (moving[j + 1] or _short_gap(moving, t, j + 1)):
                j += 1
            if t[j] - t[i] >= 0.15:
                n += 1
            i = j + 1
        else:
            i += 1
    return n


def _short_gap(moving, t, k):
    """True si en k empieza un hueco quieto más corto que una pausa (no separa tramos)."""
    j = k
    while j < len(moving) and not moving[j]:
        j += 1
    return j < len(moving) and t[j] - t[k] < CFG["hold_min_sec"]


def max_pause(phases):
    return max([b - a for _, a, b, k in phases if k == "pausa"], default=0.0)


def _features(c, hz=DTW_HZ):
    """Curva normalizada por sus propios máximos (la forma, sin la amplitud) a DTW_HZ."""
    step = max(1, int(round(RES_HZ / hz)))
    ang = c["angle"][::step]
    w = c["w"][::step]
    acc = c["acc"][::step]
    na = ang / (np.max(ang) + 1e-6)
    nw = w / (np.percentile(w, 95) + 1e-6)
    nacc = acc / (np.percentile(acc, 95) + 1e-6)
    return np.stack([na, 0.5 * nw, 0.25 * nacc], axis=1), c["t"][::step]


def dtw(A, B):
    """DTW clásico con pasos (1,0), (0,1), (1,1). Devuelve costo medio y camino [(i, j)]."""
    n, m = len(A), len(B)
    D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    C = np.full((n + 1, m + 1), np.inf)
    C[0, 0] = 0.0
    for i in range(1, n + 1):
        Ci, Cp, Di = C[i], C[i - 1], D[i - 1]
        for j in range(1, m + 1):
            Ci[j] = Di[j - 1] + min(Cp[j], Ci[j - 1], Cp[j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = np.argmin([C[i - 1, j - 1], C[i - 1, j], C[i, j - 1]])
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    cost = float(np.mean([D[a, b] for a, b in path]))
    return cost, path


def score(ref, att, tempo=True):
    """Compara una repetición contra la referencia. ref: dict con curve y phases; att: dict con curve.

    tempo=True: cuenta tiempo, amplitud y forma. tempo=False: solo forma y amplitud (los tiempos se informan).
    """
    rc, ac = ref["curve"], att["curve"]
    hz = DTW_HZ if max(rc["t"][-1], ac["t"][-1]) <= 20 else 5
    FA, tA = _features(rc, hz)
    FB, tB = _features(ac, hz)
    cost, path = dtw(FA, FB)
    # límite de fase en la referencia -> tiempo correspondiente en el intento
    match = {}
    for i, j in path:
        match.setdefault(i, []).append(j)

    def map_time(t_ref):
        i = int(np.clip(round(t_ref * hz), 0, len(tA) - 1))
        js = match.get(i) or [0]
        return float(tB[int(np.median(js))])

    phases = []
    tiempo_scores = []
    for name, a, b, kind in ref["phases"]:
        ta = 0.0 if a == 0 else map_time(a)
        tb = float(tB[-1]) if abs(b - rc["t"][-1]) < 1e-6 else map_time(b)
        d_ref, d_att = b - a, max(0.0, tb - ta)
        s = max(0.0, 1 - CFG["strict_time"] * abs(d_att - d_ref) / max(d_ref, 0.5))
        tiempo_scores.append(s)
        phases.append({"name": name, "kind": kind, "ref": round(d_ref, 2), "att": round(d_att, 2),
                       "start_att": round(ta, 2), "end_att": round(tb, 2), "score": round(s, 3)})
    tiempo = float(np.mean(tiempo_scores))

    pr, pa = float(rc["angle"].max()), float(ac["angle"].max())
    if pr >= CFG["min_amp"]:
        amp_ref, amp_att, unit = pr, pa, "°"
    else:  # movimiento casi sin inclinación (p. ej. girar en el plano horizontal): rotación total
        amp_ref = float(_trapz(rc["w"], rc["t"]))
        amp_att = float(_trapz(ac["w"], ac["t"]))
        unit = "° rotated"
    amplitud = max(0.0, 1 - CFG["strict_amp"] * abs(amp_att - amp_ref) / max(amp_ref, 10.0))
    forma = float(np.clip(1 - (cost - 0.04) / 0.30, 0, 1))
    if tempo:
        total = WEIGHTS["tiempo"] * tiempo + WEIGHTS["amplitud"] * amplitud + WEIGHTS["forma"] * forma
    else:
        total = WEIGHTS_FREE["amplitud"] * amplitud + WEIGHTS_FREE["forma"] * forma

    tips = []
    for p in phases if tempo else []:
        if p["ref"] >= 0.3 and abs(p["att"] - p["ref"]) / max(p["ref"], 0.5) > 0.2:
            pace = "faster" if p["att"] < p["ref"] else "slower"
            tips.append(f"{p['name']}: {pace} ({p['att']:.1f} s vs. {p['ref']:.1f} s reference)")
    if amplitud < 0.85:
        tips.append(f"Range: {'less' if amp_att < amp_ref else 'more'} movement "
                    f"({amp_att:.0f}{unit} vs. {amp_ref:.0f}{unit} reference)")
    if forma < 0.7:
        tips.append("Shape differs from the reference. Watch the two curves to see where they diverge.")
    return {"total": round(total, 3), "tiempo": round(tiempo, 3), "amplitud": round(amplitud, 3),
            "forma": round(forma, 3), "phases": phases, "amp_ref": round(amp_ref, 1),
            "amp_att": round(amp_att, 1), "amp_unit": unit, "dur_ref": round(float(rc["t"][-1]), 2),
            "dur_att": round(float(ac["t"][-1]), 2), "dtw_cost": round(cost, 4), "tips": tips, "tempo": tempo}


def _pack(c, hz=20):
    step = max(1, int(round(RES_HZ / hz)))
    return {"t": np.round(c["t"][::step], 2).tolist(), "angle": np.round(c["angle"][::step], 1).tolist()}


class Session:
    """Una persona grabando una referencia (kind='reference') o imitando (kind='attempt').

    Referencia: calibra (quieto), detecta el inicio del movimiento y graba hasta que se toca Terminar.
      Al terminar queda "pendiente" hasta que se le pone nombre y se guarda (o se descarta).
    Imitación: cada repetición se cierra sola cuando
      a) vuelve a la posición inicial y queda quieta 0,6 s, habiendo hecho tantos tramos de movimiento
         como la referencia, o
      b) queda quieta en cualquier posición más que 1,5 veces la pausa más larga de la referencia
         (mínimo 2,5 s), o
      c) se toca "Cerrar repetición".
    """

    def __init__(self, pid, name, kind, ref=None, tempo=True):
        self.pid, self.name, self.kind = pid, name, kind
        self.ref, self.tempo = ref, tempo
        self.state = "calib"
        self.trk = SignalTracker()
        self.last_t = None
        self.calib = deque()
        self.pre = deque(maxlen=200)
        self.rep = []
        self.still_since = None
        self.rest_checked = False
        self.reps = []
        self.pending = None
        self.msg = "Hold still"
        self.last_row = None
        self.started = time.time()
        self.long_still = 0.0
        if ref is not None:
            self.long_still = max(CFG["long_still_min"], CFG["long_still_factor"] * max_pause(ref["phases"]))

    # ------------------------------------------------------------ flujo de muestras
    def feed(self, raw, manager):
        for s in list(raw):
            t = s[0]
            if self.last_t is not None and t <= self.last_t:
                continue
            self.last_t = t
            row = self.trk.push(t, s[1:4], s[4] if len(s) > 4 else float("nan"))
            if row is None:
                continue
            self.last_row = row
            if self.state in ("pending", "done"):
                continue
            self._step(row, manager)

    def _still(self, r):
        return r["w"] < CFG["still_gyro"] and r["acc"] < CFG["still_acc"]

    def _step(self, r, manager):
        if self.state == "calib":
            if self._still(r):
                self.calib.append(r)
                while self.calib and r["t"] - self.calib[0]["t"] > CFG["calib_sec"]:
                    self.calib.popleft()
                if len(self.calib) > 5 and r["t"] - self.calib[0]["t"] >= CFG["calib_sec"] - 0.05:
                    self.trk.rest = _unit(np.mean([c["dir"] for c in self.calib], axis=0))
                    self.state = "ready"
                    self.msg = "Ready. Start the movement." if self.kind == "attempt" else \
                        "Ready to record. Start the movement."
            else:
                self.calib.clear()
            return
        if self.state == "ready":
            self.pre.append(r)
            if r["angle"] > CFG["start_angle"] or r["w"] > CFG["start_gyro"]:
                pre = list(self.pre)
                k = len(pre) - 1
                while k > 0 and (pre[k]["angle"] > CFG["edge_angle"] or pre[k]["w"] > CFG["hold_gyro_min"] * 0.7):
                    k -= 1
                self.rep = pre[k:]
                self.state, self.still_since, self.rest_checked = "rep", None, False
                self.msg = "Movement in progress…" if self.kind == "attempt" else "Recording…"
            return
        if self.state != "rep":
            return
        self.rep.append(r)
        dur = r["t"] - self.rep[0]["t"]
        if self.kind == "reference":
            if dur > CFG["ref_max_sec"]:
                self.finish_now(manager)
            return
        # ---------------- imitación: cierre automático
        if self._still(r):
            if self.still_since is None:
                self.still_since, self.rest_checked = len(self.rep) - 1, False
            still_t = r["t"] - self.rep[self.still_since]["t"]
            if r["angle"] < CFG["end_angle"] and still_t >= CFG["end_still_sec"] and not self.rest_checked:
                self.rest_checked = True
                c = trim(_resample(self.rep[: self.still_since + 1]))
                if count_bursts(c) >= self.ref.get("n_bursts", 1):
                    return self._finish_attempt(self.rep[: self.still_since + 1])
            if still_t >= self.long_still:
                return self._finish_attempt(self.rep[: self.still_since + 1])
        else:
            self.still_since = None
        if dur > float(self.ref["curve"]["t"][-1]) + 3 and self.msg != FINISH_HINT:
            self.msg = FINISH_HINT
        if dur > CFG["max_rep_sec"]:
            self._reset("Rep too long. Return to the starting position.")

    def _reset(self, msg):
        self.state, self.msg, self.rep, self.still_since = "ready", msg, [], None
        self.pre.clear()

    def _curve(self, rows):
        """Curva recortada y validada, o (None, motivo)."""
        if len(rows) < 10:
            return None, "No movement detected"
        c = trim(_resample(rows))
        moved = float(_trapz(c["w"], c["t"]))
        if c["angle"].max() < CFG["min_amp"] and moved < 40:
            return None, "Movement too small. Try again."
        if c["t"][-1] < 0.6:
            return None, "Movement too short. Try again."
        return c, ""

    # ------------------------------------------------------------ cierre
    def finish_now(self, manager):
        """Botón Terminar (referencia) o Cerrar repetición (imitación). Devuelve (ok, mensaje)."""
        if self.state != "rep":
            return False, "Waiting for movement" if self.state in ("ready", "calib") else ""
        if self.kind == "reference":
            c, why = self._curve(self.rep)
            if c is None:
                self._reset(why)
                return False, why
            self.pending = {"curve": c, "phases": find_phases(c), "n_bursts": count_bursts(c), "at": time.time()}
            self.state, self.msg = "pending", "Movement recorded. Enter a name to save it."
            return True, ""
        # imitación: cerrar en el último instante con movimiento
        rows = self.rep if self.still_since is None else self.rep[: self.still_since + 1]
        return self._finish_attempt(rows), ""

    def _finish_attempt(self, rows):
        c, why = self._curve(rows)
        if c is None:
            self._reset(why)
            return False
        res = score(self.ref, {"curve": c}, self.tempo)
        res["n"] = len(self.reps) + 1
        res["curve"] = _pack(c)
        res["at"] = time.time()
        self.reps.append(res)
        if self.on_result:
            self.on_result(self, res)
        self._reset(f"Rep {res['n']}: {round(res['total'] * 100)}%")
        return True

    on_result = None

    def view(self):
        live = None
        if self.state == "rep" and self.rep:
            t0 = self.rep[0]["t"]
            pts = self.rep[:: max(1, len(self.rep) // 300)]
            live = {"t": [round(p["t"] - t0, 2) for p in pts], "angle": [round(p["angle"], 1) for p in pts]}
        r = self.last_row or {}
        best = max((x["total"] for x in self.reps), default=None)
        pend = None
        if self.pending:
            c = self.pending["curve"]
            pend = {"curve": _pack(c), "phases": self.pending["phases"], "dur": round(float(c["t"][-1]), 2),
                    "peak": round(float(c["angle"].max()), 1), "at": self.pending["at"]}
        return {"pid": self.pid, "name": self.name, "kind": self.kind, "state": self.state, "msg": self.msg,
                "ref_id": self.ref["id"] if self.ref else None, "ref_name": self.ref["name"] if self.ref else None,
                "tempo": self.tempo, "angle": round(r.get("angle", 0.0), 1),
                "live": live, "reps": [{k: x[k] for k in ("n", "total", "tiempo", "amplitud", "forma")} for x in self.reps],
                "last": self.reps[-1] if self.reps else None, "best": best, "pending": pend}


def _pack(c, hz=20):
    step = max(1, int(round(RES_HZ / hz)))
    return {"t": np.round(c["t"][::step], 2).tolist(), "angle": np.round(c["angle"][::step], 1).tolist()}


class ExerciseManager:
    """Biblioteca de movimientos guardados, sesiones en curso y ranking por movimiento y modo."""

    def __init__(self):
        self.refs = {}       # id -> referencia (curve con arrays)
        self.results = {}    # id -> {"tempo"|"libre": {nombre: {best, last, n}}}
        self.sessions = {}
        self.version = 0     # sube cada vez que cambia la biblioteca
        self.load()

    # ------------------------------------------------------------ persistencia
    def load(self):
        try:
            if os.path.exists(REF_FILE):
                d = json.load(open(REF_FILE))
                for r in d.get("references", []):
                    r["curve"] = {k: np.array(v) for k, v in r["curve"].items()}
                    _translate_legacy_phases(r["phases"])
                    self.refs[r["id"]] = r
                self.results = d.get("results", {})
            elif os.path.exists(OLD_REF_FILE):  # migrar la referencia única de la versión anterior
                r = json.load(open(OLD_REF_FILE))
                r["curve"] = {k: np.array(v) for k, v in r["curve"].items()}
                _translate_legacy_phases(r["phases"])
                r["id"] = os.urandom(4).hex()
                r["n_bursts"] = count_bursts(r["curve"])
                self.refs[r["id"]] = r
                self.save()
        except Exception as e:  # noqa: BLE001
            print("no se pudieron leer los movimientos guardados:", e)

    def save(self):
        self.version += 1
        refs = []
        for r in self.refs.values():
            d = dict(r)
            d["curve"] = {k: v.tolist() for k, v in r["curve"].items()}
            refs.append(d)
        os.makedirs(os.path.dirname(REF_FILE), exist_ok=True)
        with open(REF_FILE, "w") as f:
            json.dump({"references": refs, "results": self.results}, f)

    # ------------------------------------------------------------ biblioteca
    def save_pending(self, pid, name, phase_names=None):
        s = self.sessions.get(pid)
        if not s or not s.pending:
            return None
        name = (str(name or "").strip() or "Movement")[:40]
        existing = {r["name"] for r in self.refs.values()}
        base, k = name, 2
        while name in existing:
            name, k = f"{base} ({k})", k + 1
        p = s.pending
        rid = os.urandom(4).hex()
        phases = [list(x) for x in p["phases"]]
        for ph, n in zip(phases, phase_names or []):
            n = str(n).strip()[:24]
            if n:
                ph[0] = n
        self.refs[rid] = {"id": rid, "name": name, "by": s.name, "curve": p["curve"], "phases": phases,
                          "n_bursts": p["n_bursts"], "at": time.time()}
        self.sessions.pop(pid, None)
        self.save()
        return rid

    def discard_pending(self, pid):
        s = self.sessions.get(pid)
        if s and s.kind == "reference":
            self.sessions.pop(pid, None)

    def delete(self, rid):
        if rid in self.refs:
            del self.refs[rid]
            self.results.pop(rid, None)
            for pid, s in list(self.sessions.items()):
                if s.ref and s.ref["id"] == rid:
                    self.sessions.pop(pid, None)
            self.save()

    def rename_phases(self, rid, names):
        r = self.refs.get(rid)
        if not r:
            return
        for p, n in zip(r["phases"], names):
            n = str(n).strip()[:24]
            if n:
                p[0] = n
        self.save()

    def reset_ranking(self, rid):
        self.results.pop(rid, None)
        self.save()

    def _record(self, session, res):
        mode = "tempo" if session.tempo else "libre"
        r = self.results.setdefault(session.ref["id"], {}).setdefault(mode, {})
        e = r.setdefault(session.name, {"best": 0.0, "last": 0.0, "n": 0})
        e["best"], e["last"], e["n"] = max(e["best"], res["total"]), res["total"], e["n"] + 1
        self.save()

    # ------------------------------------------------------------ sesiones
    def start_reference(self, participant):
        return self._start(Session(participant.id, participant.name, "reference"), participant)

    def start_attempt(self, participant, rid, tempo=True):
        ref = self.refs.get(rid)
        if ref is None:
            return None
        s = Session(participant.id, participant.name, "attempt", ref=ref, tempo=bool(tempo))
        s.on_result = self._record
        return self._start(s, participant)

    def _start(self, s, participant):
        s.last_t = participant.raw[-1][0] if participant.raw else None  # solo lo que llegue desde ahora
        self.sessions[participant.id] = s
        return s

    def finish(self, pid):
        s = self.sessions.get(pid)
        if not s:
            return False, ""
        return s.finish_now(self)

    def stop(self, pid):
        self.sessions.pop(pid, None)

    def update(self, participants):
        for pid, s in list(self.sessions.items()):
            p = participants.get(pid)
            if p is None:
                continue
            s.name = p.name
            s.feed(p.raw, self)

    # ------------------------------------------------------------ vistas
    def library(self):
        out = []
        for r in sorted(self.refs.values(), key=lambda x: x["at"]):
            res = self.results.get(r["id"], {})
            ranking = {m: sorted([{"name": n, **v} for n, v in res.get(m, {}).items()], key=lambda x: -x["best"])
                       for m in ("tempo", "libre")}
            out.append({"id": r["id"], "name": r["name"], "by": r["by"], "curve": _pack(r["curve"]),
                        "phases": r["phases"], "peak": round(float(r["curve"]["angle"].max()), 1),
                        "dur": round(float(r["curve"]["t"][-1]), 2), "ranking": ranking})
        return {"type": "library", "version": self.version, "items": out}

    def view(self):
        return {"type": "exercise", "version": self.version, "sessions": [s.view() for s in self.sessions.values()],
                "cfg": CFG, "weights": WEIGHTS, "weights_free": WEIGHTS_FREE}

    def phone_view(self, pid):
        s = self.sessions.get(pid)
        if not s:
            return None
        last = s.reps[-1]["total"] if s.reps else None
        return {"kind": s.kind, "state": s.state, "msg": s.msg, "last": last, "n": len(s.reps),
                "ref_name": s.ref["name"] if s.ref else None, "tempo": s.tempo,
                "tips": s.reps[-1]["tips"][:2] if s.reps else []}
