"""Celulares simulados: reproducen señales reales de WISDM en tiempo real contra el servidor.

Sirve para probar el panel sin celulares y como plan B en el escenario.

  python scripts/simulate_phones.py --url ws://localhost:8000 \
      --phone "Ana=Aplaudir:30,Caminar:30" --phone "Beto=Cepillarse los dientes:40"

Cada --phone es NOMBRE=ACTIVIDAD:SEGUNDOS,... y se repite en loop.
Actividades disponibles: las 18 de WISDM (ver scripts/prepare_wisdm.py).
"""
import argparse
import asyncio
import json
import os
import random

import numpy as np
import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 9.81
HZ = 30


class Library:
    """Señal continua por (persona, actividad), reconstruida desde las ventanas con 50% de solapamiento."""

    def __init__(self):
        d = np.load(os.path.join(ROOT, "data", "wisdm_watch_30hz.npz"))
        self.X, self.y, self.s, self.names = d["X"], d["y"], d["subject"], list(d["names"])

    def stream(self, activity, subject):
        c = self.names.index(activity)
        idx = np.where((self.y == c) & (self.s == subject))[0]
        w = self.X[idx]
        return np.concatenate([x[:, :75] for x in w], axis=1).T  # (T, 3) en g


async def phone(url, name, schedule, subject, lib, stop=None, log=True):
    pid = f"sim-{name.lower()}-{random.randint(1000, 9999)}"
    async with websockets.connect(url + "/ws/phone") as ws:
        await ws.send(json.dumps({"type": "hello", "id": pid, "name": name}))

        async def reader():
            async for m in ws:
                msg = json.loads(m)
                if log and msg.get("type") == "state" and msg.get("activity"):
                    print(f"  [{name}] ve: {msg['activity']} ({msg['score']:.2f})")

        rtask = asyncio.create_task(reader())
        t_ms = 0.0
        try:
            while stop is None or not stop.is_set():
                for act, secs in schedule:
                    sig = lib.stream(act, subject)
                    if log:
                        print(f"[{name}] haciendo: {act} ({secs}s)")
                    n = int(secs * HZ)
                    start = random.randint(0, max(0, len(sig) - n - 1))
                    chunk = sig[start:start + n]
                    for i in range(0, len(chunk), 6):  # paquetes de 200 ms
                        if stop is not None and stop.is_set():
                            return
                        part = chunk[i:i + 6]
                        s = []
                        for row in part:
                            s.append([round(t_ms, 1), *(row * G).round(4).tolist()])
                            t_ms += 1000 / HZ
                        await ws.send(json.dumps({"type": "samples", "s": s}))
                        await asyncio.sleep(0.2)
        finally:
            rtask.cancel()


def parse(spec):
    name, rest = spec.split("=", 1)
    sched = []
    for part in rest.split(","):
        a, secs = part.rsplit(":", 1)
        sched.append((a.strip(), float(secs)))
    return name.strip(), sched


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8000")
    ap.add_argument("--phone", action="append", required=True)
    args = ap.parse_args()
    lib = Library()
    subjects = list(range(1641, 1651))
    tasks = [phone(args.url, *parse(p), subject=subjects[i % len(subjects)], lib=lib) for i, p in enumerate(args.phone)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
