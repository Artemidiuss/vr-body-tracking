"""Check the MacBook camera from Terminal.app (which can show the macOS camera
permission prompt): resolution, fps, frame interval jitter. Writes
data/mac_probe.txt and data/mac_cam.jpg.

  .venv/bin/python -m tools.mac_cam_probe
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np


def main():
    out = open("data/mac_probe.txt", "w")

    def log(*a):
        print(*a, flush=True)
        print(*a, file=out, flush=True)

    t_end = time.time() + 120
    cap = None
    while time.time() < t_end:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            break
        log("waiting for camera permission (click Allow)…")
        time.sleep(2)
    if cap is None or not cap.isOpened():
        log("FAILED: camera not available")
        return 1
    for w, h in ((1280, 720), (1920, 1080)):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        T, P = [], []
        img = None
        for i in range(75):
            ok, img = cap.read()
            T.append(time.perf_counter())
            P.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        d = np.diff(T[15:]) * 1000
        log(f"asked {w}x{h}: got {img.shape[1]}x{img.shape[0]}, fps {1000 / d.mean():.1f}, "
            f"interval ms median {np.median(d):.1f} p5 {np.percentile(d, 5):.1f} p95 {np.percentile(d, 95):.1f}, "
            f"pos_msec {P[-3:]}")
        cv2.imwrite(f"data/mac_cam_{img.shape[1]}.jpg", img)
    log("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
