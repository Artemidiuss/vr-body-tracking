"""2D keypoint jitter per model/setting on a recording, split into STILL
frames (person barely moving: pure model noise) and MOVING frames.

  .venv/bin/python -m tools.jitter_models --cam cam1
"""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from fbt.pose import RTMPoseBackend, make_backend
from fbt.skeleton import IDX

LEG = [IDX[n] for n in ("left_ankle", "right_ankle", "left_knee", "right_knee", "left_heel",
                        "right_heel", "left_big_toe", "right_big_toe", "left_hip", "right_hip")]


def run(backend, frames):
    K, T = [], []
    for img in frames:
        t = time.perf_counter()
        p = backend(img)
        T.append((time.perf_counter() - t) * 1000)
        K.append(p.keypoints if p is not None else np.full((26, 2), np.nan))
    return np.array(K), np.median(T[5:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("--cam", default="cam1")
    ap.add_argument("--max", type=int, default=450)
    a = ap.parse_args()
    rec = json.load(open(a.calib))["meta"]["recording"]
    cap = cv2.VideoCapture(f"{rec}/{a.cam}.mp4")
    frames = []
    while len(frames) < a.max:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    # Still/moving split from a reference run (torso+legs speed).
    ref, _ = run(make_backend("rtmpose-x:gpu"), frames)
    speed = np.nanmedian(np.linalg.norm(np.diff(ref[:, LEG], axis=0), axis=-1), axis=1)
    still = np.where(speed < np.nanpercentile(speed, 25))[0]
    moving = np.where(speed > np.nanpercentile(speed, 60))[0]
    print(f"{a.cam}: {len(frames)} frames, still {len(still)}, moving {len(moving)}")
    configs = [
        ("rtmpose-m, det every 10 (current)", lambda: make_backend("rtmpose-m:gpu")),
        ("rtmpose-m, det every frame", lambda: RTMPoseBackend("m", "gpu", det_frequency=1)),
        ("rtmpose-x, det every 10", lambda: make_backend("rtmpose-x:gpu")),
        ("rtmpose-x, det every frame", lambda: RTMPoseBackend("x", "gpu", det_frequency=1)),
        ("mediapipe-full", lambda: make_backend("mediapipe-full")),
        ("mediapipe-heavy", lambda: make_backend("mediapipe-heavy")),
    ]
    print(f"{'config':<36} {'ms':>5} | still accel px (median/p95) | moving accel px | spikes>25px still")
    for name, mk in configs:
        K, ms = run(mk(), frames)
        acc = np.linalg.norm(np.diff(K[:, LEG], 2, axis=0), axis=-1)  # (T-2, J)
        s = acc[np.clip(still - 1, 0, len(acc) - 1)]
        m = acc[np.clip(moving - 1, 0, len(acc) - 1)]
        jump = np.linalg.norm(np.diff(K[:, LEG], axis=0), axis=-1)[np.clip(still, 0, len(K) - 2)]
        print(f"{name:<36} {ms:5.0f} | {np.nanmedian(s):6.2f} / {np.nanpercentile(s, 95):6.2f}"
              f"           | {np.nanmedian(m):6.2f}          | {np.nanmean(jump > 25):.2%}")


if __name__ == "__main__":
    main()
