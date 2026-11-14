"""Fake VRChat OSC receiver: listens like VRChat does and reports which
trackers arrive, at what rate, and their last values. Lets us test the hub
without the headset.

  .venv/bin/python -m tools.fake_vrchat --port 9000
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import defaultdict

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer


class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.counts = defaultdict(int)
        self.last = {}
        self.bad = []

    def handle(self, address, *args):
        with self.lock:
            parts = address.split("/")
            ok = (len(parts) == 5 and parts[1:3] == ["tracking", "trackers"]
                  and parts[4] in ("position", "rotation") and len(args) == 3
                  and all(isinstance(a, float) for a in args))
            if not ok:
                self.bad.append((address, args))
                return
            self.counts[address] += 1
            self.last[address] = args


def serve(port: int, rec: Recorder) -> ThreadingOSCUDPServer:
    d = Dispatcher()
    d.set_default_handler(rec.handle)
    server = ThreadingOSCUDPServer(("0.0.0.0", port), d)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    a = ap.parse_args()
    rec = Recorder()
    serve(a.port, rec)
    print(f"fake VRChat listening on :{a.port}")
    prev = {}
    while True:
        time.sleep(1)
        with rec.lock:
            lines = []
            for addr in sorted(rec.counts):
                rate = rec.counts[addr] - prev.get(addr, 0)
                prev[addr] = rec.counts[addr]
                x, y, z = rec.last[addr]
                lines.append(f"  {addr:<38} {rate:3d} Hz  ({x:+.2f}, {y:+.2f}, {z:+.2f})")
            if rec.bad:
                lines.append(f"  !! {len(rec.bad)} malformed messages, last: {rec.bad[-1]}")
        print("\033[2J\033[H" + "\n".join(lines or ["  (no data yet)"]))


if __name__ == "__main__":
    main()
