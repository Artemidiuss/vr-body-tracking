"""Measure phone -> Mac video latency automatically.

The Mac screen blinks black/white once per second; point the phone camera at
the screen so it fills most of the frame. Each blink is timed when drawn and
again when the received frame's brightness crosses the midpoint. The
difference is the real end-to-end latency: screen + camera exposure +
encode + network + decode.

  .venv/bin/python -m tools.latency_test http://192.168.1.50:8080/video
  .venv/bin/python -m tools.latency_test http://127.0.0.1:8080/video   # USB via adb forward

Press q to stop early.
"""

from __future__ import annotations

import argparse
import statistics
import time

import cv2
import numpy as np

from fbt.sources import OpenCVSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--blinks", type=int, default=12)
    ap.add_argument("--period", type=float, default=1.0, help="seconds per blink")
    a = ap.parse_args()

    src = OpenCVSource("phone", a.url).start()
    win = "latency-test"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    black = np.zeros((400, 400, 3), np.uint8)
    white = np.full((400, 400, 3), 255, np.uint8)

    # Calibrate brightness levels for black and white screens.
    levels = {}
    for name, img in (("black", black), ("white", white)):
        cv2.imshow(win, img)
        t_end = time.perf_counter() + 1.5
        vals, last = [], -1
        while time.perf_counter() < t_end:
            cv2.waitKey(1)
            f = src.latest(last)
            if f is not None:
                last = f.index
                vals.append(float(cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY).mean()))
        if not vals:
            print("no frames from the phone — check the URL")
            return
        levels[name] = statistics.median(vals[len(vals) // 2:])
    mid = (levels["black"] + levels["white"]) / 2
    if levels["white"] - levels["black"] < 30:
        print(f"contrast too low ({levels}) — point the camera at the screen, closer")
        return

    lat, fps_frames, last, state = [], 0, -1, False
    t_start = time.perf_counter()
    t_toggle = None
    for i in range(a.blinks * 2):
        state = not state
        cv2.imshow(win, white if state else black)
        cv2.waitKey(1)
        t_toggle = time.perf_counter()
        detected = False
        while time.perf_counter() < t_toggle + a.period:
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            f = src.latest(last)
            if f is None:
                continue
            last = f.index
            fps_frames += 1
            b = float(cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY).mean())
            if not detected and ((b > mid) == state):
                lat.append((f.received_at - t_toggle) * 1000)
                detected = True
            if detected and time.perf_counter() > t_toggle + a.period / 2:
                break
    src.stop()
    cv2.destroyAllWindows()
    wall = time.perf_counter() - t_start
    if not lat:
        print("no transitions detected")
        return
    print(f"received fps: {fps_frames / wall:.1f}, dropped by hub: {src.dropped}")
    print(f"latency ms over {len(lat)} blinks: median {statistics.median(lat):.0f}, "
          f"min {min(lat):.0f}, max {max(lat):.0f}")


if __name__ == "__main__":
    main()
