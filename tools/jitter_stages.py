"""Where does the shaking come from? Measures noise at every stage of the
pipeline on a calibrated recording.

  .venv/bin/python -m tools.jitter_stages

Stages: 2D keypoints per camera (px) -> camera sync -> triangulated 3D ->
bone-length fit -> One-Euro filter -> tracker positions/rotations.
Metrics per stage:
  accel   — mean |second difference| per frame (jitter that looks like shaking)
  spikes  — frames where a point jumps > 8 cm (or > 25 px in 2D) in one frame
  causes of 3D spikes are attributed: L/R swap flip, point newly invalid/filled,
  large 2D jump in one camera, or sync gap.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from fbt.calibrate import load_calibration, pair_tracks, poses_from_recording
from fbt.filters import SkeletonFit, SkeletonStabilizer
from fbt.pose import Pose2D, make_backend
from fbt.skeleton import IDX
from fbt.trackers import RotationSmoother, solve
from fbt.triangulate import triangulate_with_swap_check

LEG = [IDX[n] for n in ("left_ankle", "right_ankle", "left_knee", "right_knee", "left_heel",
                        "right_heel", "left_big_toe", "right_big_toe", "hip")]
NAMES = {i: n for n, i in IDX.items()}


def accel(x):  # x: (T, K, D)
    return np.linalg.norm(np.diff(x, 2, axis=0), axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("--rec")
    ap.add_argument("-b", "--backend", default="rtmpose-m:gpu")
    a = ap.parse_args()
    cams, _ = load_calibration(a.calib)
    rec = a.rec or json.load(open(a.calib))["meta"]["recording"]
    be = make_backend(a.backend)
    ta, tb = poses_from_recording(rec, "cam0", be), poses_from_recording(rec, "cam1", be)

    print(f"== {a.backend} on {rec}")
    for name, tr in (("cam0", ta), ("cam1", tb)):
        acc = accel(tr.kp[:, LEG])
        dt = np.diff(tr.t)
        print(f"2D {name}: accel median {np.median(acc):.1f}px p95 {np.percentile(acc, 95):.1f}px, "
              f"jumps>25px {np.mean(np.linalg.norm(np.diff(tr.kp[:, LEG], axis=0), axis=-1) > 25):.1%}, "
              f"frame gaps >50ms {np.mean(dt > 0.05):.1%}, low-conf legs {np.mean(tr.sc[:, LEG] < 0.3):.1%}")

    pa, pb, sa, sb, ts = pair_tracks(ta, tb)
    raw, valid, swapped, prev = [], [], [], None
    for i in range(len(pa)):
        r = triangulate_with_swap_check(cams, [Pose2D(pa[i], sa[i]), Pose2D(pb[i], sb[i])], prev=prev)
        prev = r
        raw.append(r.points.copy()); valid.append(r.valid.copy()); swapped.append(tuple(r.swapped))
    raw, valid = np.array(raw), np.array(valid)
    fit = SkeletonFit(60)
    fitted = np.array([fit(raw[i], valid[i]) for i in range(len(raw))])
    st = SkeletonStabilizer(min_cutoff=0.7, beta=0.5, predict_s=0.02, learn_frames=60, gate=False)
    old = np.array([st(raw[i], valid[i], ts[i]) for i in range(len(raw))])
    st = SkeletonStabilizer(learn_frames=60)  # current defaults: gate + gentler One-Euro
    filt = np.array([st(raw[i], valid[i], ts[i]) for i in range(len(raw))])
    print(f"outlier gate rejected {st.gate.rejected} point-frames")

    sl = slice(60, None)  # after bone learning
    for name, X in (("3D raw", raw), ("3D fitted", fitted), ("3D old filter", old), ("3D NEW", filt)):
        acc = accel(X[sl][:, LEG]) * 100
        jump = np.linalg.norm(np.diff(X[sl][:, LEG], axis=0), axis=-1) > 0.08
        print(f"{name:12s}: accel median {np.median(acc):.2f}cm p95 {np.percentile(acc, 95):.2f}cm, "
              f"jumps>8cm {jump.mean():.2%} of point-frames")

    # Attribute raw 3D spikes.
    d = np.linalg.norm(np.diff(raw[:, LEG], axis=0), axis=-1)
    spikes = np.argwhere(d > 0.08)
    causes = {"swap flip": 0, "invalid/refilled": 0, "2D jump": 0, "sync gap": 0, "other": 0}
    for f, j in spikes:
        k = LEG[j]
        if swapped[f + 1] != swapped[f]:
            causes["swap flip"] += 1
        elif valid[f + 1, k] != valid[f, k] or not valid[f + 1, k]:
            causes["invalid/refilled"] += 1
        elif max(np.linalg.norm(pa[f + 1, k] - pa[f, k]), np.linalg.norm(pb[f + 1, k] - pb[f, k])) > 25:
            causes["2D jump"] += 1
        elif ts[f + 1] - ts[f] > 0.05:
            causes["sync gap"] += 1
        else:
            causes["other"] += 1
    print(f"raw 3D spikes: {len(spikes)} -> {causes}")
    worst = np.argsort(-d.max(1))[:5]
    for f in worst:
        j = int(np.argmax(d[f]))
        print(f"   frame {f}: {NAMES[LEG[j]]} jumped {d[f, j] * 100:.0f} cm, swap {swapped[f]}->{swapped[f + 1]}, "
              f"valid {valid[f, LEG[j]]}->{valid[f + 1, LEG[j]]}, scores cam0 {sa[f + 1, LEG[j]]:.2f} cam1 {sb[f + 1, LEG[j]]:.2f}")

    # Tracker rotations jitter (what VRChat gets).
    for label, X, smooth in (("old", old, None), ("NEW", filt, RotationSmoother())):
        rots = {}
        for i in range(60, len(X)):
            trs = solve(X[i], enabled=("hip", "left_foot", "right_foot", "left_knee", "right_knee", "chest"))
            if smooth:
                trs = smooth(trs)
            for t in trs:
                rots.setdefault(t.id, []).append(t.rotation)
        line = []
        for tid, r in sorted(rots.items()):
            r = np.unwrap(np.radians(np.array(r)), axis=0)
            acc = np.degrees(np.abs(np.diff(r, 2, axis=0))).max(1)
            line.append(f"{tid}: {np.median(acc):.1f}/{np.percentile(acc, 95):.1f}°")
        print(f"rotation accel median/p95 [{label}] " + "  ".join(line))
    # lag of NEW vs fitted (no temporal filter)
    from tools.tune_filter import lag_ms
    print(f"lag vs unfiltered: old {lag_ms(fitted[sl][:, LEG], old[sl][:, LEG], ts[sl]):.0f} ms, "
          f"NEW {lag_ms(fitted[sl][:, LEG], filt[sl][:, LEG], ts[sl]):.0f} ms")


if __name__ == "__main__":
    main()
