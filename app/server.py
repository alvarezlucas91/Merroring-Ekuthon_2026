"""Servidor de la demo.

  /        página para el celular del público (captura de sensores)
  /panel   panel para la pantalla grande
  /ws/phone, /ws/panel  WebSockets

Correr:  python app/server.py  [--encoder hybrid|features|ekumen|oxford] [--port 8000]
"""
import argparse
import asyncio
import io
import json
import os
import sys
import time
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn  # noqa: E402
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402

from engine import Engine  # noqa: E402
from exercise import CFG as EX_CFG, ExerciseManager  # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)
engine: Engine = None
exman: ExerciseManager = None
phones = {}   # pid -> websocket
panels = set()


@app.get("/")
def phone_page():
    return FileResponse(os.path.join(STATIC, "phone.html"), headers={"Cache-Control": "no-store"})


@app.get("/panel")
def panel_page():
    return FileResponse(os.path.join(STATIC, "panel.html"), headers={"Cache-Control": "no-store"})


@app.get("/qr.svg")
def qr(url: str):
    import qrcode
    import qrcode.image.svg
    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=12, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return Response(buf.getvalue(), media_type="image/svg+xml")


@app.websocket("/ws/phone")
async def ws_phone(ws: WebSocket):
    await ws.accept()
    pid = None
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("type") == "hello":
                pid = str(msg.get("id") or os.urandom(4).hex())[:32]
                engine.join(pid, str(msg.get("name", ""))[:30])
                phones[pid] = ws
                await ws.send_text(json.dumps({"type": "welcome", "id": pid}))
            elif msg.get("type") == "samples" and pid:
                engine.ingest(pid, msg.get("s", []))
    except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
        pass
    finally:
        if pid:
            engine.leave(pid)
            if phones.get(pid) is ws:
                phones.pop(pid, None)


@app.websocket("/ws/panel")
async def ws_panel(ws: WebSocket):
    await ws.accept()
    panels.add(ws)
    try:
        await ws.send_text(json.dumps(engine.snapshot()))
        await ws.send_text(json.dumps(exman.library()))
        await ws.send_text(json.dumps(exman.view()))
        while True:
            msg = json.loads(await ws.receive_text())
            t = msg.get("type")
            if t == "record":
                engine.start_recording(msg.get("pid"), msg.get("name"), msg.get("seconds"))
            elif t == "cancel":
                engine.cancel_recording(msg.get("pid"))
            elif t == "delete":
                engine.delete_activity(msg.get("name"))
            elif t == "threshold":
                engine.threshold = float(msg.get("value", engine.threshold))
            elif t == "record_seconds":
                engine.record_seconds = int(msg.get("value", engine.record_seconds))
            elif t == "load_base":
                engine.load_base()
            elif t == "reset":
                engine.reset()
            elif t == "ex_reference":
                p = engine.participants.get(msg.get("pid"))
                if p:
                    exman.start_reference(p)
            elif t == "ex_attempt":
                p = engine.participants.get(msg.get("pid"))
                if p:
                    exman.start_attempt(p, msg.get("ref_id"), msg.get("tempo", True))
            elif t == "ex_finish":
                ok, why = exman.finish(msg.get("pid"))
                if not ok and why:
                    await ws.send_text(json.dumps({"type": "toast", "text": why}))
            elif t == "ex_save":
                exman.save_pending(msg.get("pid"), msg.get("name"), msg.get("phases"))
            elif t == "ex_discard":
                exman.discard_pending(msg.get("pid"))
            elif t == "ex_stop":
                exman.stop(msg.get("pid"))
            elif t == "ex_delete":
                exman.delete(msg.get("ref_id"))
            elif t == "ex_rename":
                exman.rename_phases(msg.get("ref_id"), msg.get("names", []))
            elif t == "ex_reset_ranking":
                exman.reset_ranking(msg.get("ref_id"))
            elif t == "ex_cfg":
                k, v = msg.get("key"), msg.get("value")
                if k in EX_CFG:
                    EX_CFG[k] = float(v)
            await ws.send_text(json.dumps(exman.view()))
    except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
        pass
    finally:
        panels.discard(ws)


async def broadcast(ws_iter, payload_fn):
    dead = []
    for key, ws in ws_iter:
        try:
            # tiempo límite: un celular trabado (pestaña en segundo plano) no frena a los demás
            await asyncio.wait_for(ws.send_text(json.dumps(payload_fn(key))), timeout=0.5)
        except Exception:  # noqa: BLE001
            dead.append(key)
    return dead


def phone_payload(pid):
    st = engine.phone_state(pid)
    st["ex"] = exman.phone_view(pid)
    return st


async def loop():
    """10 veces por segundo: modo ejercicio. 1 vez por segundo: reconocimiento y estado completo."""
    k, lib_sent = 0, -1
    while True:
        t0 = time.time()
        try:
            exman.update(engine.participants)
            if exman.version != lib_sent:
                lib, lib_sent = exman.library(), exman.version
                await broadcast([(w, w) for w in list(panels)], lambda _: lib)
            active = any(s.state == "rep" for s in exman.sessions.values())
            if active or k % 3 == 0:
                view = exman.view()
                await broadcast([(w, w) for w in list(panels)], lambda _: view)
            if exman.sessions and k % 2 == 0:
                await broadcast([(pid, ws) for pid, ws in phones.items() if pid in exman.sessions], phone_payload)
            if k % 10 == 0:
                finished = engine.tick()
                snap = engine.snapshot()
                for pid, name in finished:
                    snap.setdefault("events", []).append({"kind": "learned", "pid": pid, "name": name})
                await broadcast([(w, w) for w in list(panels)], lambda _: snap)
                await broadcast(list(phones.items()), phone_payload)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("error en el ciclo:", repr(e))
        k += 1
        await asyncio.sleep(max(0.01, 0.1 - (time.time() - t0)))


def main():
    global engine, exman
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=os.environ.get("ENCODER", "hybrid"),
                    choices=["hybrid", "features", "ekumen", "oxford"])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--record-seconds", type=int, default=20)
    args = ap.parse_args()
    engine = Engine(args.encoder, args.threshold, args.record_seconds)
    exman = ExerciseManager()
    print(f"Encoder: {args.encoder}  umbral: {engine.threshold:.2f}")
    print(f"Celular:  http://localhost:{args.port}/      Panel: http://localhost:{args.port}/panel")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
