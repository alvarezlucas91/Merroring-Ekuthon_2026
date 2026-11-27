"""Prueba de punta a punta del modo Ejercicio guiado, con celulares simulados.

Ana graba tres movimientos de referencia (uno con pausas de 5 s y otro que termina en otra posición).
Otras cinco personas imitan distintos movimientos, con tempo y sin tempo. Cada una con el celular puesto distinto.

  python scripts/e2e_exercise.py            # prueba y muestra los puntajes
  python scripts/e2e_exercise.py --keep     # deja el servidor andando en :8000 para mirar el panel
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from simulate_exercise import parse_profile, random_mount, synth  # noqa: E402

REFS = {  # nombre: perfil (sin la quietud inicial)
    "Subir la mano": "mover:2:90,pausa:1,mover:2:0",
    "Con pausas de 5 s": "mover:2:60,pausa:5,mover:2:110,pausa:5,mover:2:0",
    "Termina arriba": "mover:2:80",
}
ATTEMPTS = [  # (persona, movimiento, con tempo, perfil, descripción)
    ("Beto", "Subir la mano", True, "mover:2:90,pausa:1,mover:2:0", "igual"),
    ("Caro", "Subir la mano", True, "mover:1:90,pausa:0.5,mover:1:0", "al doble de velocidad"),
    ("Dani", "Subir la mano", False, "mover:1:90,pausa:0.5,mover:1:0", "al doble de velocidad"),
    ("Eli", "Con pausas de 5 s", True, "mover:2:60,pausa:6,mover:2:110,pausa:4,mover:2:0", "pausas de 6 y 4 s"),
    ("Fede", "Termina arriba", True, "mover:2:80", "igual"),
]


class FakePhone:
    def __init__(self, url, name, seed):
        self.url, self.name = url, name
        self.rng = np.random.default_rng(seed)
        self.mount = random_mount(self.rng)
        self.queue = []
        self.t = 0.0
        self.pid = f"sim-{name.lower()}"

    def do(self, profile):
        s = synth(parse_profile(profile), self.mount, t0_ms=self.t, rng=self.rng)
        self.t = s[-1][0] + 1000 / 60
        self.queue += s

    async def run(self, stop):
        async with websockets.connect(self.url + "/ws/phone") as ws:
            await ws.send(json.dumps({"type": "hello", "id": self.pid, "name": self.name}))

            async def drain():  # leer lo que manda el servidor, como hace un celular real
                async for _ in ws:
                    pass
            reader = asyncio.create_task(drain())
            while not stop.is_set():
                if len(self.queue) < 12:
                    self.do("quieto:0.2")
                chunk, self.queue = self.queue[:12], self.queue[12:]
                await ws.send(json.dumps({"type": "samples", "s": chunk}))
                await asyncio.sleep(0.2)
            reader.cancel()


async def scenario(url, keep):
    stop = asyncio.Event()
    names = ["Ana"] + [x[0] for x in ATTEMPTS]
    phones = {n: FakePhone(url, n, i + 3) for i, n in enumerate(names)}
    tasks = [asyncio.create_task(p.run(stop)) for p in phones.values()]
    view, lib = {}, {}
    async with websockets.connect(url + "/ws/panel") as panel:
        async def listen():
            async for m in panel:
                d = json.loads(m)
                if d.get("type") == "exercise":
                    view.update(d)
                elif d.get("type") == "library":
                    lib.update(d)
        lt = asyncio.create_task(listen())
        await asyncio.sleep(2)
        print("Grabando referencias (Grabar referencia, movimiento, Terminar, Guardar):")
        for name, prof in REFS.items():
            await panel.send(json.dumps({"type": "ex_reference", "pid": "sim-ana"}))
            await asyncio.sleep(2.5)
            phones["Ana"].do(prof + ",quieto:1.5")
            dur = sum(float(x.split(":")[1]) for x in prof.split(",")) + 1.5
            await asyncio.sleep(dur + 1)
            await panel.send(json.dumps({"type": "ex_finish", "pid": "sim-ana"}))
            await asyncio.sleep(0.6)
            await panel.send(json.dumps({"type": "ex_save", "pid": "sim-ana", "name": name}))
            await asyncio.sleep(0.6)
        ids = {r["name"]: r["id"] for r in lib.get("items", [])}
        for r in lib.get("items", []):
            print(f"  {r['name']:20s} {[(p[0], round(p[2] - p[1], 1)) for p in r['phases']]}")
        for who, ref, tempo, _, _ in ATTEMPTS:
            await panel.send(json.dumps({"type": "ex_attempt", "pid": phones[who].pid, "ref_id": ids[ref], "tempo": tempo}))
        await asyncio.sleep(2.5)
        for who, _, _, prof, _ in ATTEMPTS:
            phones[who].do(prof + ",quieto:0.5," + prof)  # dos repeticiones
        await asyncio.sleep(42)
        print("\nImitaciones:")
        by = {s["name"]: s for s in view.get("sessions", [])}
        for who, ref, tempo, _, desc in ATTEMPTS:
            s = by.get(who)
            for rep in (s or {}).get("reps", []):
                print(f"  {who:5s} {ref:18s} {'con tempo' if tempo else 'sin tempo':9s} ({desc:20s}) rep {rep['n']}: "
                      f"total {rep['total']:.0%}  tiempo {rep['tiempo']:.0%}  amplitud {rep['amplitud']:.0%}  forma {rep['forma']:.0%}")
        if keep:
            print("\nServidor andando en http://localhost:8000/panel (Ctrl+C para cortar)")
            await asyncio.sleep(10 ** 6)
        lt.cancel()
    stop.set()
    await asyncio.gather(*tasks, return_exceptions=True)


def wait_server(port, timeout=60):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/panel", timeout=1)
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    raise RuntimeError("el servidor no arrancó")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    port = 8000 if args.keep else 8766
    tmpdir = tempfile.mkdtemp()
    env = dict(os.environ, PROTOTYPES_FILE=os.path.join(tmpdir, "p.json"), EXERCISE_FILE=os.path.join(tmpdir, "e.json"))
    srv = subprocess.Popen([sys.executable, os.path.join(ROOT, "app", "server.py"), "--port", str(port)], env=env)
    try:
        wait_server(port)
        asyncio.run(scenario(f"ws://localhost:{port}", args.keep))
    except KeyboardInterrupt:
        pass
    finally:
        srv.terminate()
        srv.wait()


if __name__ == "__main__":
    main()
