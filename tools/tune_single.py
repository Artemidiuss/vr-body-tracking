"""Tune the single-camera pipeline on a recording (one camera of it):
jitter when standing still vs lag when moving, for several variants.

  .venv/bin/python -m tools.tune_single --cam cam0
"""

from __future__ import annotations

import argparse
import csv
import json

import cv2
import numpy as np

from fbt.calibrate import load_calibration
from fbt.filters import OneEuro, SkeletonStabilizer
from fbt.mono3d import MediaPipeMono3D
from fbt.skeleton import IDX
from fbt.trackers import solve
from tools.run_hub_single import FEET, SEG_RATIO, segsum
from tools.tune_filter import lag_ms

TR = ("hip", "left_foot", "right_foot", "left_knee", "right_knee")


def mono_frames(rec, cam, variant):
    times = [float(r["t_capture"]) for r in csv.DictReader(open(f"{rec}/{cam}.csv"))]
    cap = cv2.VideoCapture(f"{rec}/{cam}.mp4")
    model = MediaPipeMono3D(variant)
    out, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok or i >= len(times):
            break
        r = model(img)
        out.append((times[i], r))
        i += 1
    return out


def run(frames, R, height, floor_mode, stab_kw, ema=None):
    hs = [segsum(r[0] @ R) for _, r in frames[:60] if r is not None]
    scale = SEG_RATIO * height / np.median(hs)
    floor, fixed, stab = None, None, SkeletonStabilizer(**stab_kw)
    ema_state = None
    pos = []
    for t, r in frames:
        if r is None:
            continue
        P = (r[0] @ R) * scale
        low = np.min(P[FEET, 1])
        if floor_mode == "per_frame":
            floor = low if floor is None else 0.9 * floor + 0.1 * low
        else:  # fixed after the first second
            fixed = low if fixed is None else (0.95 * fixed + 0.05 * low if len(pos) < 30 else fixed)
            floor = fixed
        P[:, 1] -= floor
        S = stab(P, r[1] > 0.3, t)
        if ema is not None:
            ema_state = S if ema_state is None else ema * ema_state + (1 - ema) * S
            S = ema_state
        trs = {x.id: x.position for x in solve(S, enabled=TR)}
        pos.append([trs[k] for k in sorted(trs)])
    return np.array(pos)  # (T, trackers, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("--cam", default="cam0")
    ap.add_argument("--height", type=float, default=1.73)
    a = ap.parse_args()
    cams, _ = load_calibration(a.calib)
    R = cams[int(a.cam[-1])].R
    rec = json.load(open(a.calib))["meta"]["recording"]
    frames = {m: mono_frames(rec, a.cam, m) for m in ("full", "heavy")}
    ts = np.array([t for t, r in frames["full"] if r is not None])
    variants = [
        ("current: floor per frame, euro 2.5/0.4", "full", "per_frame", dict(), None),
        ("ju1ce: euro 2.5 + EMA 0.7", "full", "fixed", dict(), 0.7),
        ("euro 1.0/0.4 linear", "full", "fixed", dict(min_cutoff=1.0, beta=0.4), None),
        ("v^2: 0.7 + 20*v^2", "full", "fixed", dict(min_cutoff=0.7, beta=20, power=2), None),
        ("v^2: 0.5 + 10*v^2", "full", "fixed", dict(min_cutoff=0.5, beta=10, power=2), None),
        ("v^2: 0.5 + 30*v^2", "full", "fixed", dict(min_cutoff=0.5, beta=30, power=2), None),
        ("v^2: 0.5 + 10*v^2 + foot lock", "full", "fixed", dict(min_cutoff=0.5, beta=10, power=2, foot_lock=True), None),
    ]
    ref = run(frames["full"], R, a.height, "fixed", dict(min_cutoff=1e6, beta=0, gate=False))
    speed = np.linalg.norm(np.diff(ref, axis=0), axis=-1).mean(-1)
    still = np.where(speed < np.percentile(speed, 30))[0]
    print(f"{'variant':<44} | still jitter mm (accel med/p95) | lag ms (fast) | lag ms (slow)")
    for name, model, floor_mode, kw, ema in variants:
        X = run(frames[model], R, a.height, floor_mode, kw, ema)
        n = min(len(X), len(ref))
        acc = np.linalg.norm(np.diff(X[:n], 2, axis=0), axis=-1).mean(-1) * 1000
        sacc = acc[np.clip(still - 1, 0, len(acc) - 1)]
        print(f"{name:<44} | {np.median(sacc):8.2f} / {np.percentile(sacc, 95):6.2f}"
              f"            | {lag_ms(ref[60:n], X[60:n], ts[60:n]):9.0f}     | "
              f"{lag_ms(ref[60:n], X[60:n], ts[60:n], band=(20, 60)):9.0f}")


if __name__ == "__main__":
    main()
