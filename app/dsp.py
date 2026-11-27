"""Procesamiento de señal compartido por el servidor en vivo y los scripts offline.

Convenciones (iguales para entrenamiento y para la demo en vivo):
- Acelerómetro con gravedad incluida, en g (m/s² / 9.81).
- 30 Hz.
- Ventanas de 5 s = 150 muestras, forma (3, 150).
"""
import numpy as np

G = 9.81
TARGET_HZ = 30
WIN_SEC = 5
WIN = TARGET_HZ * WIN_SEC  # 150
MAX_GAP_SEC = 0.5  # si faltan datos por más de esto, se corta el tramo


def resample_uniform(t_sec, xyz, hz=TARGET_HZ):
    """Interpola linealmente a una grilla uniforme.

    t_sec: (N,) tiempos en segundos (crecientes). xyz: (N, 3).
    Devuelve lista de tramos contiguos, cada uno (M, 3), cortando en huecos > MAX_GAP_SEC.
    """
    t_sec = np.asarray(t_sec, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    order = np.argsort(t_sec, kind="stable")
    t_sec, xyz = t_sec[order], xyz[order]
    keep = np.concatenate([[True], np.diff(t_sec) > 0])
    t_sec, xyz = t_sec[keep], xyz[keep]
    if len(t_sec) < 2:
        return []
    cuts = np.where(np.diff(t_sec) > MAX_GAP_SEC)[0] + 1
    out = []
    for seg_t, seg_x in zip(np.split(t_sec, cuts), np.split(xyz, cuts)):
        if len(seg_t) < 2:
            continue
        grid = np.arange(seg_t[0], seg_t[-1], 1.0 / hz)
        res = np.stack([np.interp(grid, seg_t, seg_x[:, k]) for k in range(3)], axis=1)
        out.append(res)
    return out


def sliding_windows(arr, win=WIN, step=WIN // 2):
    """arr: (M, 3) -> (K, 3, win)."""
    if len(arr) < win:
        return np.zeros((0, 3, win), dtype=np.float32)
    idx = np.arange(0, len(arr) - win + 1, step)
    w = np.stack([arr[i:i + win] for i in idx]).transpose(0, 2, 1)
    return w.astype(np.float32)


def handcrafted_features(w, hz=TARGET_HZ):
    """Features clásicas por ventana. w: (N, 3, T) en g -> (N, F)."""
    w = np.asarray(w, dtype=np.float64)
    mag = np.linalg.norm(w, axis=1, keepdims=True)
    ch = np.concatenate([w, mag], axis=1)  # (N, 4, T)
    feats = [
        ch.mean(-1), ch.std(-1), ch.min(-1), ch.max(-1),
        *[np.percentile(ch, q, axis=-1) for q in (10, 25, 50, 75, 90)],
        np.abs(np.diff(ch, axis=-1)).mean(-1),
    ]
    # Espectro de cada canal sin la componente continua
    x = ch - ch.mean(-1, keepdims=True)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / hz)
    total = spec[..., 1:].sum(-1) + 1e-9
    feats.append(freqs[spec[..., 1:].argmax(-1) + 1])  # frecuencia dominante
    for lo, hi in ((0.3, 1), (1, 3), (3, 6), (6, 15)):
        band = (freqs >= lo) & (freqs < hi)
        feats.append(spec[..., band].sum(-1) / total)  # potencia relativa por banda
    feats.append(np.log1p(total))
    # Correlación entre ejes
    xc = x[:, :3]
    sd = xc.std(-1) + 1e-9
    for a, b in ((0, 1), (0, 2), (1, 2)):
        feats.append(((xc[:, a] * xc[:, b]).mean(-1) / (sd[:, a] * sd[:, b]))[:, None])
    return np.concatenate([f if f.ndim == 2 else f[:, None] for f in feats], axis=1).astype(np.float32)
