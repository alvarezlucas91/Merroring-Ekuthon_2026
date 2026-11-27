"""Prepara el acelerómetro del smartwatch de WISDM 2019 en ventanas de 5 s a 30 Hz, en g.

Fuente original: UCI Machine Learning Repository, dataset 507 (CC BY 4.0).
  https://archive.ics.uci.edu/dataset/507
Si UCI no es accesible, el script baja los mismos archivos crudos desde un espejo en GitHub.

Uso:
  python scripts/prepare_wisdm.py                # baja (si hace falta) y procesa
  python scripts/prepare_wisdm.py --raw DIR      # DIR con data_16xx_accel_watch.txt
Salida: data/wisdm_watch_30hz.npz
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
from dsp import G, WIN, resample_uniform, sliding_windows  # noqa: E402

MIRROR = "https://github.com/DusanRandD/Human-Activity-Recognition.git"

ACTIVITIES = {
    "A": "Caminar", "B": "Trotar", "C": "Escaleras", "D": "Sentado", "E": "Parado",
    "F": "Tipear", "G": "Cepillarse los dientes", "H": "Tomar sopa", "I": "Comer papas fritas",
    "J": "Comer pasta", "K": "Tomar de una taza", "L": "Comer un sándwich", "M": "Patear una pelota",
    "O": "Jugar a atrapar", "P": "Picar una pelota", "Q": "Escribir a mano", "R": "Aplaudir",
    "S": "Doblar ropa",
}


def fetch_mirror(dest):
    if glob.glob(os.path.join(dest, "Dataset", "data_*_accel_watch.txt")):
        return os.path.join(dest, "Dataset")
    print("Bajando archivos crudos desde el espejo de GitHub...")
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout", MIRROR, dest], check=True)
    subprocess.run(["git", "-C", dest, "checkout", "HEAD", "--", "Dataset/"], check=True)
    return os.path.join(dest, "Dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None)
    ap.add_argument("--step", type=int, default=WIN // 2, help="paso entre ventanas (muestras)")
    args = ap.parse_args()

    raw = args.raw or fetch_mirror(os.path.join(ROOT, "data", "wisdm_raw"))
    files = sorted(glob.glob(os.path.join(raw, "data_*_accel_watch.txt")))
    assert files, f"No hay archivos en {raw}"
    codes = list(ACTIVITIES)
    X, Y, S, T = [], [], [], []
    for f in files:
        rows = []
        with open(f) as fh:
            for line in fh:
                p = line.strip().rstrip(";").split(",")
                if len(p) == 6:
                    rows.append(p)
        subj = np.array([int(r[0]) for r in rows])
        act = np.array([r[1] for r in rows])
        ts = np.array([int(r[2]) for r in rows], dtype=np.int64)
        xyz = np.array([[float(r[3]), float(r[4]), float(r[5])] for r in rows]) / G
        for code in codes:
            m = act == code
            if m.sum() < WIN:
                continue
            t = (ts[m] - ts[m].min()) / 1e9
            offset = 0.0
            for seg in resample_uniform(t, xyz[m]):
                w = sliding_windows(seg, step=args.step)
                X.append(w)
                Y += [codes.index(code)] * len(w)
                S += [subj[0]] * len(w)
                T += list(offset + np.arange(len(w)) * args.step / 30.0)
                offset += len(seg) / 30.0
        print(os.path.basename(f), "ok")
    X = np.concatenate(X)
    out = os.path.join(ROOT, "data", "wisdm_watch_30hz.npz")
    np.savez_compressed(out, X=X, y=np.array(Y), subject=np.array(S), t=np.array(T, dtype=np.float32),
                        names=np.array([ACTIVITIES[c] for c in codes]))
    print(f"{out}: {X.shape[0]} ventanas, {len(set(S))} personas, {len(codes)} actividades")


if __name__ == "__main__":
    main()
