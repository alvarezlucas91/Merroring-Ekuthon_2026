"""Encoders: convierten ventanas (N, 3, 150) en vectores (embeddings) normalizados.

Tres opciones intercambiables:
- "features": features clásicas estandarizadas. Sin deep learning, sin restricciones de licencia.
- "ekumen":   CNN 1D propia entrenada con WISDM (CC BY 4.0). Ver scripts/train_encoder.py.
- "oxford":   modelo autosupervisado de Oxford (harnet5, UK Biobank). Licencia SOLO académica,
              no comercial: no usar en material para clientes sin revisar la licencia.
"""
import os

import numpy as np
import torch
import torch.nn as nn

from dsp import handcrafted_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
torch.set_num_threads(max(1, (os.cpu_count() or 2)))


def _l2(z):
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)


class _Base:
    name = "base"
    center = None
    scale = None

    def raw(self, w):
        raise NotImplementedError

    def fit_center(self, w):
        z = self.raw(w)
        self.center = z.mean(0)
        self.scale = z.std(0) + 1e-6
        return self

    def __call__(self, w):
        w = np.asarray(w, dtype=np.float32)
        if len(w) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        z = self.raw(w)
        if self.center is not None:
            z = (z - self.center) / self.scale
        return _l2(z).astype(np.float32)


class FeatureEncoder(_Base):
    name = "features"

    def raw(self, w):
        return handcrafted_features(w)

    @property
    def dim(self):
        return handcrafted_features(np.zeros((1, 3, 150))).shape[1]


# ---------------------------------------------------------------- CNN propia
class EkumenNet(nn.Module):
    """CNN 1D chica. embed() devuelve el vector de 128; forward() las clases (solo para entrenar)."""

    def __init__(self, n_classes=18, emb=128):
        super().__init__()

        def block(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())

        self.body = nn.Sequential(
            block(3, 32, 7), nn.MaxPool1d(2),
            block(32, 64, 5), nn.MaxPool1d(2),
            block(64, 128, 5), nn.MaxPool1d(2),
            block(128, 128, 3),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, emb),
        )
        # clasificador coseno: empuja a que el espacio sirva para comparar por distancia
        self.W = nn.Parameter(torch.randn(n_classes, emb) * 0.01)
        self.s = 16.0

    def embed(self, x):
        return self.body(x)

    def forward(self, x):
        e = nn.functional.normalize(self.embed(x), dim=1)
        W = nn.functional.normalize(self.W, dim=1)
        return self.s * e @ W.t()


class EkumenEncoder(_Base):
    name = "ekumen"
    dim = 128

    def __init__(self, path=os.path.join(MODELS, "ekumen_encoder.pt")):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = EkumenNet(n_classes=ck["n_classes"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.center, self.scale = ck.get("center"), ck.get("scale")

    @torch.no_grad()
    def raw(self, w):
        out = [self.net.embed(torch.from_numpy(w[i:i + 512])).numpy() for i in range(0, len(w), 512)]
        return np.concatenate(out)


class OxfordEncoder(_Base):
    """harnet5 de OxWearables/ssl-wearables. Copyright © 2022, University of Oxford. Uso académico."""
    name = "oxford"
    dim = 512

    def __init__(self, repo=os.environ.get("SSL_WEARABLES_REPO", os.path.join(ROOT, "vendor", "ssl-wearables"))):
        m = torch.hub.load(repo, "harnet5", source="local", class_num=5, pretrained=True, trust_repo=True)
        m.eval()
        self.net = m.feature_extractor

    @torch.no_grad()
    def raw(self, w):
        out = [self.net(torch.from_numpy(w[i:i + 512])).squeeze(-1).numpy() for i in range(0, len(w), 512)]
        return np.concatenate(out)


class HybridEncoder:
    """Concatena features clásicas y CNN propia. Es la mejor opción en la evaluación."""
    name = "hybrid"

    def __init__(self, features, ekumen):
        self.parts = [features, ekumen]
        self.dim = features.dim + ekumen.dim

    def __call__(self, w):
        z = np.concatenate([p(w) for p in self.parts], axis=1)
        return _l2(z).astype(np.float32)


def load_encoder(name):
    if name == "hybrid":
        return HybridEncoder(load_encoder("features"), load_encoder("ekumen"))
    if name == "features":
        enc = FeatureEncoder()
        p = os.path.join(MODELS, "features_norm.npz")
        if os.path.exists(p):
            d = np.load(p)
            enc.center, enc.scale = d["center"], d["scale"]
        return enc
    if name == "ekumen":
        return EkumenEncoder()
    if name == "oxford":
        enc = OxfordEncoder()
        p = os.path.join(MODELS, "oxford_norm.npz")
        if os.path.exists(p):
            d = np.load(p)
            enc.center, enc.scale = d["center"], d["scale"]
        return enc
    raise ValueError(name)
