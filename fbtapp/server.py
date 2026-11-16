"""Local web server for the FBT app: the UI page, live MJPEG of every camera
with its 2D skeleton, a Server-Sent-Events state stream and a small JSON API.
The native window (macapp/, FBT.app) is a WKWebView on http://127.0.0.1:8765.

  .venv/bin/python -m fbtapp --browser            # without the app: opens the default browser
  .venv/bin/python -m fbtapp --cameras replay:data/rec_sit_lowres/cam0,replay:data/rec_sit_lowres/cam1,replay:data/rec_sit_lowres/cam2
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from fbt.engine import Engine
from fbt.skeleton import BONE_IDX, BONES, HALPE26

from . import replay

UI = os.path.join(os.path.dirname(__file__), "ui")
TYPES = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".png": "image/png", ".svg": "image/svg+xml"}
ENGINE: Engine | None = None


def _side(a: str, b: str):
    if a.startswith("left") and b.startswith("left"):
        return (255, 170, 60)      # BGR: blue-ish
    if a.startswith("right") and b.startswith("right"):
        return (60, 160, 255)      # orange
    return (120, 230, 120)


BONE_COLORS = [_side(a, b) for a, b in BONES]


def render_tile(img: np.ndarray, pose, max_side: int) -> np.ndarray:
    s = max_side / max(img.shape[:2])
    out = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    s = min(s, 1.0)
    if pose is not None:
        kp, sc = pose.keypoints * s, pose.scores
        for (a, b), col in zip(BONE_IDX, BONE_COLORS):
            if sc[a] > 0.3 and sc[b] > 0.3:
                cv2.line(out, tuple(map(int, kp[a])), tuple(map(int, kp[b])), col, 3, cv2.LINE_AA)
        for k in range(len(kp)):
            if sc[k] > 0.3:
                cv2.circle(out, tuple(map(int, kp[k])), 3, (255, 255, 255), -1, cv2.LINE_AA)
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "FBT/1"

    def log_message(self, *a):
        pass

    # ---------------- helpers
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    # ---------------- GET
    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path in ("/", "/index.html", "/app.js", "/style.css", "/replay", "/replay.html", "/replay.js"):
                path = "/replay.html" if path == "/replay" else path
                name = "index.html" if path == "/" else path[1:]
                with open(os.path.join(UI, name), "rb") as fh:
                    self._send(200, fh.read(), TYPES[os.path.splitext(name)[1]])
            elif path == "/health":
                self._send(200, b"ok", "text/plain")
            elif path == "/api/state":
                self._json(ENGINE.snapshot())
            elif path == "/api/meta":
                self._json({"bones": BONE_IDX, "names": HALPE26})
            elif path == "/api/sessions":
                self._json(replay.list_sessions(ENGINE.d))
            elif path.startswith("/api/replay/"):
                self._replay(path.split("/")[3])
            elif path == "/events":
                self._events()
            elif path.startswith("/video/"):
                self._video(int(path.split("/")[2]))
            else:
                self._send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _replay(self, name: str):
        try:
            p = replay.build(ENGINE.d, name)
        except (ValueError, OSError) as e:
            self._send(404, str(e).encode(), "text/plain")
            return
        with open(p, "rb") as fh:
            self._send(200, fh.read(), "application/octet-stream")

    def _events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        while True:
            setup = ENGINE.mode == "setup"
            msg = json.dumps(ENGINE.snapshot(with_skeleton=setup), ensure_ascii=False)
            self.wfile.write(f"data: {msg}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.1 if setup else 0.3)

    def _video(self, i: int):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        last_id, t_last = None, 0.0
        while ENGINE.mode == "setup":
            slots = ENGINE.slots
            if i >= len(slots):
                return
            s = slots[i]
            with s.lock:
                img, pose = s.last_img, s.last_pose
            if img is None or id(img) == last_id or time.perf_counter() - t_last < 1 / 15:
                time.sleep(0.02)
                if not s.online and img is not None and time.perf_counter() - t_last > 2:
                    return            # camera gone: the page reconnects when it is back
                continue
            last_id, t_last = id(img), time.perf_counter()
            tile = render_tile(img, pose, 640)
            ok, jpg = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if not ok:
                continue
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg.tobytes() + b"\r\n")
            self.wfile.flush()

    # ---------------- POST
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except ValueError:
            body = {}
        path = self.path.split("?")[0]
        e = ENGINE
        routes = {
            "/api/calibrate": e.start_calibration,
            "/api/calibrate/cancel": lambda: (e.cancel_calibration(), {"ok": True})[1],
            "/api/calibrate/dismiss": lambda: (e.dismiss_calibration(), {"ok": True})[1],
            "/api/server/start": e.start_server,
            "/api/server/stop": e.stop_server,
            "/api/sleep": e.sleep,
            "/api/wake": e.wake,
        }
        try:
            if path in routes:
                self._json(routes[path]())
            elif path == "/api/settings":
                self._json(e.update_settings(body))
            elif path == "/api/thermal":
                e.thermal = int(body.get("state", -1))
                self._json({"ok": True})
            else:
                self._send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--data", default="data")
    ap.add_argument("--settings", help="settings file (default data/app_settings.json)")
    ap.add_argument("--cameras", help="comma-separated camera list for this run (e.g. replay:... for tests)")
    ap.add_argument("--browser", action="store_true", help="open the UI in the default browser")
    ap.add_argument("--parent-watch", action="store_true", help="exit when the launching app is gone")
    a = ap.parse_args()
    try:     # frames without a person make the CoreML EP log an error per frame: keep app.log readable
        import onnxruntime
        onnxruntime.set_default_logger_severity(4)
    except Exception:
        pass
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    if a.cameras and not a.settings:      # a test run must not overwrite the real settings
        a.settings = os.path.join(a.data, "app_settings_override.json")
    ENGINE = Engine(a.data, settings_path=a.settings, cameras=a.cameras.split(",") if a.cameras else None)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True

    def bye(*_):
        try:
            ENGINE.shutdown()
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    if a.parent_watch:
        ppid = os.getppid()

        def watch():
            while os.getppid() == ppid:
                time.sleep(1)
            bye()
        threading.Thread(target=watch, daemon=True).start()
    print(f"FBT UI on http://127.0.0.1:{a.port}", flush=True)
    if a.browser:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{a.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
