"""Connect to FBT Cam phones and report fps, real capture->Mac latency
(from sensor timestamps + clock sync), clock-sync RTT and a snapshot.

  .venv/bin/python -m tools.fbtcam_check 192.168.31.28 192.168.31.81
  .venv/bin/python -m tools.fbtcam_check 127.0.0.1:5000        # USB via adb forward
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time

import cv2

from fbt.fbtcam import FBTCamSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hosts", nargs="+")
    ap.add_argument("--seconds", type=float, default=10)
    a = ap.parse_args()
    srcs = []
    for i, h in enumerate(a.hosts):
        host, _, port = h.partition(":")
        srcs.append(FBTCamSource(f"cam{i}", host, int(port or 5000)).start())
    time.sleep(1.5)  # let clock sync settle
    stats = {}

    def work(s):
        last, n, t0, img = -1, 0, time.perf_counter(), None
        s.latency_ms.clear()
        while time.perf_counter() < t0 + a.seconds:
            f = s.latest(last)
            if f is None:
                time.sleep(0.001)
                continue
            last, n, img = f.index, n + 1, f.image
        stats[s.camera_id] = (n / a.seconds, list(s.latency_ms), img)

    th = [threading.Thread(target=work, args=(s,)) for s in srcs]
    [t.start() for t in th]
    [t.join() for t in th]
    for s in srcs:
        fps, lat, img = stats[s.camera_id]
        s.stop()
        info = s.info
        print(f"\n== {s.camera_id} {s.host}: {info.get('device')} cam {info.get('camera_id')} "
              f"{info.get('width')}x{info.get('height')}@{info.get('fps')} "
              f"{'high-speed' if info.get('high_speed') else ''}")
        print(f"  received {fps:.1f} fps, dropped in hub {s.dropped}")
        if lat:
            lat.sort()
            print(f"  capture->decoded latency ms: p50 {statistics.median(lat):.0f}  "
                  f"p95 {lat[int(len(lat) * .95)]:.0f}  min {lat[0]:.0f}  (clock rtt {s.clock.rtt * 1000:.1f} ms)")
        if img is not None:
            path = f"data/fbtcam_{s.camera_id}.jpg"
            h, w = img.shape[:2]
            cv2.imwrite(path, cv2.resize(img, (w * 640 // h, 640)))
            print(f"  frame {w}x{h} saved to {path}")


if __name__ == "__main__":
    main()
