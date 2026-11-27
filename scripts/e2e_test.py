"""Prueba de punta a punta con celulares simulados (sin hardware).

1. Levanta el servidor en un puerto de prueba.
2. Tres "personas" enseñan Aplaudir, Cepillarse los dientes y Picar una pelota.
3. Otras cuatro personas distintas hacen esos movimientos + uno que nadie enseñó (Doblar ropa).
4. Mide qué detectó el panel para cada una.

  python scripts/e2e_test.py [--encoder hybrid]
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter

import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from simulate_phones import Library, phone  # noqa: E402

PORT = 8765
URL = f"ws://localhost:{PORT}"
TEACH = [("Ana", "Aplaudir", 1641), ("Beto", "Cepillarse los dientes", 1642), ("Caro", "Picar una pelota", 1643)]
TEST = [("Dani", "Aplaudir", 1647), ("Eli", "Cepillarse los dientes", 1648),
        ("Fede", "Picar una pelota", 1649), ("Gabi", "Doblar ropa", 1650)]


async def run(record_secs):
    lib = Library()
    seen = {}

    async with websockets.connect(URL + "/ws/panel") as panel:
        async def listen():
            async for m in panel:
                st = json.loads(m)
                if st.get("type") != "state":
                    continue
                for e in st.get("events", []):
                    print(f"  panel: aprendió '{e['name']}'")
                for p in st.get("participants", []):
                    if p["online"] and p["activity"]:
                        seen.setdefault(p["name"], []).append((time.time(), p["activity"], p["score"]))
                listen.last = st
        listen.last = None
        lt = asyncio.create_task(listen())

        print("Fase 1: enseñan")
        stop = asyncio.Event()
        tasks = [asyncio.create_task(phone(URL, n, [(a, 60)], s, lib, stop, log=False)) for n, a, s in TEACH]
        await asyncio.sleep(6)
        for p in listen.last["participants"]:
            act = dict((n, a) for n, a, _ in TEACH).get(p["name"])
            if act:
                await panel.send(json.dumps({"type": "record", "pid": p["id"], "name": act, "seconds": record_secs}))
        await asyncio.sleep(record_secs + 6)
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)

        print("Fase 2: personas nuevas")
        stop2 = asyncio.Event()
        t_start = time.time()
        tasks = [asyncio.create_task(phone(URL, n, [(a, 60)], s, lib, stop2, log=False)) for n, a, s in TEST]
        await asyncio.sleep(25)
        stop2.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        lt.cancel()

    print("\nResultado (desde que cada persona lleva 8 s conectada):")
    ok_all = 0
    for n, a, _ in TEST:
        preds = [x[1] for x in seen.get(n, []) if x[0] > t_start + 8]
        expected = a if a != "Doblar ropa" else "Desconocida"
        hit = sum(p == expected for p in preds) / max(1, len(preds))
        ok_all += hit
        sc = [x[2] for x in seen.get(n, []) if x[0] > t_start + 8]
        print(f"  {n:5s} parecido medio {sum(sc) / max(1, len(sc)):.2f}")
        print(f"  {n:5s} hace {a:24s} esperado {expected:24s} acierto {hit:4.0%}  {Counter(preds).most_common(3)}")
    print(f"Promedio: {ok_all / len(TEST):.0%}")


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
    ap.add_argument("--encoder", default="hybrid")
    ap.add_argument("--record-seconds", type=int, default=15)
    args = ap.parse_args()
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    os.unlink(tmp)
    env = dict(os.environ, PROTOTYPES_FILE=tmp)
    srv = subprocess.Popen([sys.executable, os.path.join(ROOT, "app", "server.py"), "--encoder", args.encoder,
                            "--port", str(PORT)], env=env)
    try:
        wait_server(PORT)
        asyncio.run(run(args.record_seconds))
    finally:
        srv.terminate()
        srv.wait()


if __name__ == "__main__":
    main()
