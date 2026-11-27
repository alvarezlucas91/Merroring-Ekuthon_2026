"""Genera movimientos sintéticos (acelerómetro + giroscopio) para probar el modo Ejercicio guiado.

Un perfil es una lista de tramos:
  quieto:SEG              quieto en la posición actual
  mover:SEG:ANGULO        rotar suavemente hasta ANGULO grados desde el reposo
  pausa:SEG               sostener la posición

Ejemplo (subir la mano en 2 s, sostener 1 s, bajar en 2 s):
  python scripts/simulate_exercise.py --name Ana --profile "quieto:3,mover:2:90,pausa:1,mover:2:0,quieto:2" --repeat 3

Cada persona simulada se pone el celular con una orientación distinta (al azar), para verificar
que la comparación no depende de cómo esté puesto.
"""
import argparse
import asyncio
import json

import numpy as np

G = 9.81


def rot(axis, deg):
    axis = axis / np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def random_mount(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def parse_profile(spec):
    out = []
    for part in spec.split(","):
        f = part.strip().split(":")
        out.append((f[0], float(f[1]), float(f[2]) if len(f) > 2 else None))
    return out


def angle_curve(profile, hz=60):
    """Ángulo (°) en el tiempo según el perfil, con transiciones suaves (coseno)."""
    ang, cur = [], 0.0
    for kind, dur, target in profile:
        n = max(1, int(round(dur * hz)))
        if kind == "mover":
            u = (1 - np.cos(np.linspace(0, np.pi, n))) / 2
            seg = cur + (target - cur) * u
            cur = target
        else:
            seg = np.full(n, cur)
        ang.append(seg)
    return np.concatenate(ang)


def synth(profile, mount, hz=60, t0_ms=0.0, rng=None, noise=1.0, jerk=0.0):
    """Muestras [t_ms, ax, ay, az, gx, gy, gz] (m/s², °/s) para un celular montado con `mount`."""
    rng = rng or np.random.default_rng()
    theta = angle_curve(profile, hz)
    if jerk > 0:  # temblor / movimiento brusco
        theta = theta + jerk * np.sin(2 * np.pi * 4 * np.arange(len(theta)) / hz) * (np.gradient(theta) != 0)
    g0 = mount @ np.array([0.0, -1.0, 0.0])        # gravedad en el marco del celular, en reposo
    axis = mount @ np.array([1.0, 0.0, 0.0])       # eje de rotación del antebrazo, en el marco del celular
    dtheta = np.gradient(theta) * hz               # °/s
    ddtheta = np.gradient(dtheta) * hz
    tang = np.cross(axis, g0)
    out = []
    for k, th in enumerate(theta):
        g = rot(axis, -th) @ g0
        lin = 0.3 * np.radians(ddtheta[k]) / G * (rot(axis, -th) @ tang)  # aceleración tangencial (r = 0.3 m)
        a = (g + lin + rng.normal(0, 0.01 * noise, 3)) * G
        gyro = axis * dtheta[k] + rng.normal(0, 1.0 * noise, 3)
        out.append([round(t0_ms + k * 1000 / hz, 2), *a.round(4).tolist(), *gyro.round(3).tolist()])
    return out


async def run(url, name, profile, repeat, seed):
    import websockets
    rng = np.random.default_rng(seed)
    mount = random_mount(rng)
    samples = []
    for _ in range(repeat):
        t0 = samples[-1][0] + 1000 / 60 if samples else 0.0
        samples += synth(profile, mount, t0_ms=t0, rng=rng)
    async with websockets.connect(url + "/ws/phone") as ws:
        await ws.send(json.dumps({"type": "hello", "id": f"sim-ex-{name.lower()}", "name": name}))
        for i in range(0, len(samples), 12):
            await ws.send(json.dumps({"type": "samples", "s": samples[i:i + 12]}))
            await asyncio.sleep(0.2)
        while True:  # queda conectado y quieto
            t = samples[-1][0]
            still = synth([("quieto", 0.2, None)], mount, t0_ms=t + 1000 / 60, rng=rng)
            samples[-1] = still[-1]
            await ws.send(json.dumps({"type": "samples", "s": still}))
            await asyncio.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8000")
    ap.add_argument("--name", default="Simulado")
    ap.add_argument("--profile", default="quieto:3,mover:2:90,pausa:1,mover:2:0,quieto:2")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run(args.url, args.name, parse_profile(args.profile), args.repeat, args.seed))


if __name__ == "__main__":
    main()
