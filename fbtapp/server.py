"""Local web server for the FBT app: the UI page, live MJPEG of every camera
with its 2D skeleton, a Server-Sent-Events state stream and a small JSON API.
The native window (macapp/, FBT.app) is a WKWebView on http://127.0.0.1:8765; on Windows
windows/FBT.cmd runs this with --browser, which opens an Edge app window.

  .venv/bin/python -m fbtapp --browser            # without the app: opens the default browser (Edge app window on Windows)
  .venv/bin/python -m fbtapp --cameras replay:data/rec_sit_lowres/cam0,replay:data/rec_sit_lowres/cam1,replay:data/rec_sit_lowres/cam2
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
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
PAGES, PAGE_T, PAGES_LOCK = 0, 0.0, threading.Lock()    # open UI pages (their /events streams), when the last one went


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
        except ConnectionError:         # the window closed mid-reply (Windows: ConnectionAbortedError)
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
        global PAGES, PAGE_T
        with PAGES_LOCK:
            PAGES += 1
        try:
            while True:
                setup = ENGINE.mode == "setup"
                msg = json.dumps(ENGINE.snapshot(with_skeleton=setup), ensure_ascii=False)
                self.wfile.write(f"data: {msg}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.1 if setup else 0.3)
        finally:
            with PAGES_LOCK:
                PAGES -= 1
                PAGE_T = time.perf_counter()

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
        except ConnectionError:         # the window closed mid-reply (Windows: ConnectionAbortedError)
            pass


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], ConnectionError):
            return                      # a closed window, not an error worth a traceback
        super().handle_error(request, client_address)


def already_running(port: int) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
            return r.read() == b"ok"
    except OSError:
        return False


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
    url = f"http://127.0.0.1:{a.port}"
    if already_running(a.port):          # started again (the shortcut clicked twice): just show the window
        if a.browser:
            open_window(url, a.data)
        return
    if sys.stdout is None:               # pythonw (the Windows shortcut, no console): output goes to app.log, as on a Mac
        os.makedirs(os.path.join(a.data, "logs"), exist_ok=True)
        sys.stdout = sys.stderr = open(os.path.join(a.data, "logs", "app.log"), "w", encoding="utf-8", buffering=1)
    if a.cameras and not a.settings:      # a test run must not overwrite the real settings
        a.settings = os.path.join(a.data, "app_settings_override.json")
    ENGINE = Engine(a.data, settings_path=a.settings, cameras=a.cameras.split(",") if a.cameras else None)
    srv = Server(("127.0.0.1", a.port), Handler)

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
    print(f"FBT UI on {url}", flush=True)
    if a.browser:
        win = open_window(url, a.data)
        if win is not None:              # FBT's own window: closing it quits FBT, like FBT.app on a Mac
            def window_watch():
                # Quit when the Edge we started has ended AND no FBT page has been open for 5 s.
                # The process alone is not enough: Edge sometimes hands its window to another Edge
                # process of the same profile and exits at once (seen: an FBT left running with no
                # window). The pages alone are not either: a page could drop its stream while its
                # window stays open.
                while True:
                    time.sleep(1.0)
                    with PAGES_LOCK:
                        gone = PAGES == 0 and PAGE_T > 0 and time.perf_counter() - PAGE_T > 5
                    if gone and win.poll() is not None:
                        print("FBT window closed: quitting", flush=True)
                        bye()
            threading.Thread(target=window_watch, daemon=True).start()
    srv.serve_forever()


def open_window(url: str, data: str = "data"):
    """The UI in its own window: on Windows Edge's app mode (no tabs, no address bar; Edge comes
    with Windows 10/11), otherwise the default browser. On a Mac FBT.app is the window.
    On Windows the window gets its own Edge profile (data/edge_window), so it is a separate Edge
    process that ends when the window is closed (measured: within 1 s): that process is returned,
    and the caller quits FBT with it. In the user's own profile Edge would hand the window to an
    already running Edge and exit at once."""
    if sys.platform == "win32":
        import subprocess
        prof = os.path.abspath(os.path.join(data, "edge_window"))
        prefs = os.path.join(prof, "Default", "Preferences")
        if not os.path.exists(prefs):    # a fresh profile: no offer to translate the Ukrainian page
            os.makedirs(os.path.dirname(prefs), exist_ok=True)
            with open(prefs, "w", encoding="utf-8") as fh:
                json.dump({"translate": {"enabled": False}, "signin": {"allowed": False}}, fh)
        for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
            edge = os.path.join(base or "", "Microsoft", "Edge", "Application", "msedge.exe")
            if base and os.path.exists(edge):
                # No sign-in, sync or extensions: a fresh Edge profile signs in to the Windows
                # Microsoft account by itself and pulls its extensions, whose pages kept the
                # process alive after the FBT window was closed (seen with a VPN extension).
                return subprocess.Popen([edge, f"--app={url}", "--window-size=1400,900", f"--user-data-dir={prof}",
                                         "--no-first-run", "--no-default-browser-check", "--disable-extensions",
                                         "--disable-sync", "--disable-background-mode",
                                         "--disable-features=msImplicitSignin,msEdgeAutoSignIn"])
    import webbrowser
    webbrowser.open(url)
    return None


if __name__ == "__main__":
    main()
