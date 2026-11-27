"""Evalúa cuántos segundos de grabación hacen falta para aprender una actividad NUEVA.

Usa las 6 actividades y las 10 personas que el encoder propio nunca vio al entrenar.
Compara tres encoders: features clásicas, CNN propia (Ekumen) y Oxford (si está disponible).

Protocolos:
  misma persona: la persona graba D segundos y después se la reconoce a ella misma.
  otra persona:  una persona graba D segundos y se reconoce a otra distinta.
  dos personas enseñan: dos personas graban D segundos cada una (un prototipo por grabación,
                 gana el más parecido) y se reconoce a una tercera.
Además mide la detección de "actividad desconocida" (AUROC) y propone un umbral.

Salida: results/fewshot_metrics.json, results/fewshot.png, models/thresholds.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from encoders import EkumenEncoder, FeatureEncoder, HybridEncoder, OxfordEncoder, MODELS  # noqa: E402
from train_encoder import HOLDOUT_CLASSES, TEST_SUBJECTS  # noqa: E402

DURATIONS = [5, 10, 20, 30]
REPS = 20
rng = np.random.default_rng(42)


def auroc(pos, neg):
    """Probabilidad de que un conocido tenga más similitud que un desconocido."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def proto(Z):
    p = Z.mean(0)
    return p / (np.linalg.norm(p) + 1e-9)


def support_query(t, D):
    """Elige un bloque de D s; devuelve máscaras de soporte y de consulta sin superposición."""
    t0 = rng.uniform(t.min(), max(t.min(), t.max() + 5 - D))
    sup = (t >= t0) & (t + 5 <= t0 + D + 1e-6)
    if not sup.any():
        sup = np.zeros_like(t, bool)
        sup[np.argmin(np.abs(t - t0))] = True
    qry = (t >= t0 + D) | (t + 5 <= t0)
    return sup, qry


def evaluate(Z, y, s, t):
    res = {"misma persona": {}, "otra persona": {}, "dos personas enseñan": {}}
    subs = TEST_SUBJECTS
    for D in DURATIONS:
        same, other, two = [], [], []
        for _ in range(REPS):
            for a in subs:
                protos, queries = [], []
                for c in HOLDOUT_CLASSES:
                    m = (s == a) & (y == c)
                    sup, qry = support_query(t[m], D)
                    protos.append(proto(Z[m][sup]))
                    queries.append((Z[m][qry], c))
                P = np.stack(protos)
                for Zq, c in queries:
                    pred = np.array(HOLDOUT_CLASSES)[(Zq @ P.T).argmax(1)]
                    same.append((pred == c).mean())
                b = rng.choice([x for x in subs if x != a])
                for c in HOLDOUT_CLASSES:
                    m = (s == b) & (y == c)
                    pred = np.array(HOLDOUT_CLASSES)[(Z[m] @ P.T).argmax(1)]
                    other.append((pred == c).mean())
                # dos personas enseñan (a y b), se reconoce a una tercera (q)
                b2, q = rng.choice([x for x in subs if x != a], 2, replace=False)
                P2 = []
                for c in HOLDOUT_CLASSES:
                    m = (s == b2) & (y == c)
                    sup, _ = support_query(t[m], D)
                    P2.append(proto(Z[m][sup]))
                P2 = np.stack(P2)
                for c in HOLDOUT_CLASSES:
                    m = (s == q) & (y == c)
                    sim = np.maximum(Z[m] @ P.T, Z[m] @ P2.T)
                    pred = np.array(HOLDOUT_CLASSES)[sim.argmax(1)]
                    two.append((pred == c).mean())
        res["misma persona"][D] = float(np.mean(same))
        res["otra persona"][D] = float(np.mean(other))
        res["dos personas enseñan"][D] = float(np.mean(two))
    # Desconocido: prototipos de 5 actividades (30 s, misma persona), la sexta es la "nueva"
    known, unknown = [], []
    for _ in range(REPS // 2):
        for a in subs:
            for left in HOLDOUT_CLASSES:
                P, kq = [], []
                for c in HOLDOUT_CLASSES:
                    m = (s == a) & (y == c)
                    if c == left:
                        continue
                    sup, qry = support_query(t[m], 30)
                    P.append(proto(Z[m][sup]))
                    kq.append(Z[m][qry])
                P = np.stack(P)
                known += list((np.concatenate(kq) @ P.T).max(1))
                m = (s == a) & (y == left)
                unknown += list((Z[m] @ P.T).max(1))
    known, unknown = np.array(known), np.array(unknown)
    thr = float(np.percentile(known, 10))  # deja pasar el 90% de lo conocido
    res["desconocido"] = {
        "auroc": float(auroc(known, unknown)),
        "umbral_sugerido": thr,
        "desconocidos_detectados_con_umbral": float((unknown < thr).mean()),
    }
    return res


def main():
    d = np.load(os.path.join(ROOT, "data", "wisdm_watch_30hz.npz"))
    X, y, s, t = d["X"], d["y"], d["subject"], d["t"]
    train_m = ~np.isin(y, HOLDOUT_CLASSES) & ~np.isin(s, TEST_SUBJECTS)
    test_m = np.isin(y, HOLDOUT_CLASSES) & np.isin(s, TEST_SUBJECTS)
    ref = X[train_m][rng.choice(train_m.sum(), 8000, replace=False)]

    fe = FeatureEncoder().fit_center(ref)
    ek = EkumenEncoder(os.path.join(MODELS, "ekumen_encoder_eval.pt"))
    encoders = {"Features clásicas": fe, "CNN propia (Ekumen)": ek,
                "Híbrido: features + CNN propia": HybridEncoder(fe, ek)}
    try:
        encoders["Oxford SSL (uso académico)"] = OxfordEncoder().fit_center(ref)
    except Exception as e:  # noqa: BLE001
        print("Oxford no disponible:", e)

    # Normalización para producción (se ajusta con todas las personas de entrenamiento)
    np.savez(os.path.join(MODELS, "features_norm.npz"), center=encoders["Features clásicas"].center,
             scale=encoders["Features clásicas"].scale)
    if "Oxford SSL (uso académico)" in encoders:
        o = encoders["Oxford SSL (uso académico)"]
        np.savez(os.path.join(MODELS, "oxford_norm.npz"), center=o.center, scale=o.scale)

    results = {}
    for name, enc in encoders.items():
        Z = enc(X[test_m])
        results[name] = evaluate(Z, y[test_m], s[test_m], t[test_m])
        r = results[name]
        print(f"\n{name}")
        for k in ("misma persona", "otra persona", "dos personas enseñan"):
            print(f"  {k}: " + "  ".join(f"{D}s={v:.2f}" for D, v in r[k].items()))
        print(f"  desconocido: AUROC={r['desconocido']['auroc']:.2f}  umbral={r['desconocido']['umbral_sugerido']:.2f}")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "fewshot_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    key = {"Features clásicas": "features", "CNN propia (Ekumen)": "ekumen",
           "Híbrido: features + CNN propia": "hybrid", "Oxford SSL (uso académico)": "oxford"}
    with open(os.path.join(MODELS, "thresholds.json"), "w") as f:
        json.dump({key[n]: r["desconocido"]["umbral_sugerido"] for n, r in results.items()}, f, indent=2)
    plot(results)


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Features clásicas": "#9aa0a8", "CNN propia (Ekumen)": "#7fa7e8",
              "Híbrido: features + CNN propia": "#1d4fb8", "Oxford SSL (uso académico)": "#d9822b"}
    titles = {"misma persona": "Reconoce a la misma persona", "otra persona": "Enseña una persona, reconoce a otra",
              "dos personas enseñan": "Enseñan dos personas, reconoce a una tercera"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, k in zip(axes, titles):
        for name, r in results.items():
            xs = list(r[k].keys())
            ys = [100 * v for v in r[k].values()]
            ax.plot(xs, ys, marker="o", lw=2.2, color=colors.get(name, "#444"), label=name)
            if name.startswith(("Híbrido", "Oxford")):
                ax.annotate(f"{ys[-1]:.0f}%", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, -3),
                            fontsize=9, color=colors.get(name, "#444"))
        ax.set_title(titles[k], fontsize=11, loc="left")
        ax.set_xlabel("Segundos grabados para enseñar cada actividad")
        ax.set_xticks(DURATIONS)
        ax.axhline(100 / 6, color="#bbb", lw=1, ls="--")
        ax.text(DURATIONS[0], 100 / 6 + 1.5, "azar (1 de 6)", fontsize=8, color="#999")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#eee")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Precisión en actividades nuevas (%)")
    axes[2].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Aprender una actividad nueva con pocos segundos (6 actividades y 10 personas nunca vistas)",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "fewshot.png"), dpi=160)
    print("results/fewshot.png")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        with open(os.path.join(ROOT, "results", "fewshot_metrics.json")) as f:
            r = json.load(f)
        plot({n: {k: ({int(D): v for D, v in d.items()} if k != "desconocido" else d) for k, d in x.items()}
              for n, x in r.items()})
    else:
        main()
