"""Entrena el encoder propio (EkumenNet) con WISDM.

Dos modos:
  --mode eval    entrena SIN 6 actividades y SIN 10 personas, para medir honestamente cómo
                 aprende actividades nuevas con pocos segundos (lo usa evaluate_fewshot.py).
  --mode deploy  entrena con todo; es el modelo que usa la demo.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
from encoders import EkumenNet  # noqa: E402

HOLDOUT_CLASSES = [6, 12, 14, 15, 16, 17]  # cepillarse, patear, picar pelota, escribir, aplaudir, doblar ropa
TEST_SUBJECTS = list(range(1641, 1651))


def rot_batch(n, max_deg, rng):
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg, size=n))
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    K = K - K.transpose(0, 2, 1)
    I = np.eye(3)[None]
    return I + np.sin(ang)[:, None, None] * K + (1 - np.cos(ang))[:, None, None] * K @ K


def augment(x, rng):
    n = len(x)
    R = rot_batch(n, 30, rng).astype(np.float32)
    x = np.einsum("nij,njt->nit", R, x)
    x = x * rng.uniform(0.9, 1.1, size=(n, 1, 1)).astype(np.float32)
    x = x + rng.normal(0, 0.01, size=x.shape).astype(np.float32)
    shift = rng.integers(0, x.shape[-1], size=n)
    x = np.stack([np.roll(x[i], s, axis=-1) for i, s in enumerate(shift)])
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "deploy"], required=True)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    d = np.load(os.path.join(ROOT, "data", "wisdm_watch_30hz.npz"))
    X, y, subj = d["X"], d["y"], d["subject"]
    if args.mode == "eval":
        m = ~np.isin(y, HOLDOUT_CLASSES) & ~np.isin(subj, TEST_SUBJECTS)
    else:
        m = np.ones(len(y), bool)
    X, y = X[m], y[m]
    classes = np.unique(y)
    y = np.searchsorted(classes, y)
    print(f"modo {args.mode}: {len(X)} ventanas, {len(classes)} clases")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = EkumenNet(n_classes=len(classes))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = args.epochs * (len(X) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    for ep in range(args.epochs):
        t0, perm, tot, acc = time.time(), rng.permutation(len(X)), 0.0, 0.0
        net.train()
        nb = len(X) // 256
        for b in range(nb):
            i = perm[b * 256:(b + 1) * 256]
            xb = torch.from_numpy(augment(X[i], rng))
            yb = torch.from_numpy(y[i]).long()
            out = net(xb)
            loss = lossf(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            acc += (out.argmax(1) == yb).float().mean().item()
        print(f"época {ep + 1}/{args.epochs}  loss {tot / nb:.3f}  acc entren. {acc / nb:.3f}  {time.time() - t0:.0f}s", flush=True)

    net.eval()
    with torch.no_grad():
        Z = np.concatenate([net.embed(torch.from_numpy(X[i:i + 1024])).numpy() for i in range(0, len(X), 1024)])
    name = "ekumen_encoder.pt" if args.mode == "deploy" else "ekumen_encoder_eval.pt"
    torch.save({"state_dict": net.state_dict(), "n_classes": len(classes), "classes": classes,
                "center": Z.mean(0), "scale": Z.std(0) + 1e-6}, os.path.join(ROOT, "models", name))
    print("guardado", name)


if __name__ == "__main__":
    main()
