"""Triangulate a calibrated recording and report 3D quality: valid points,
reprojection error, bone-length stability, height.

  .venv/bin/python -m tools.eval_recording                 # recording from data/calibration.json
  .venv/bin/python -m tools.eval_recording --rec data/rec_X --calib data/calibration.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from fbt.calibrate import load_calibration, pair_tracks, poses_from_recording
from fbt.pose import Pose2D, make_backend
from fbt.skeleton import IDX
from fbt.triangulate import triangulate_with_swap_check

SEGS = {"thigh L": ("left_hip", "left_knee"), "thigh R": ("right_hip", "right_knee"),
        "shin L": ("left_knee", "left_ankle"), "shin R": ("right_knee", "right_ankle"),
        "hip width": ("left_hip", "right_hip"), "shoulders": ("left_shoulder", "right_shoulder"),
        "upper arm L": ("left_shoulder", "left_elbow"), "foot L": ("left_heel", "left_big_toe")}


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
    pa, pb, sa, sb, _ = pair_tracks(ta, tb)
    L = {k: [] for k in SEGS}
    heights, valid, errs, swaps, prev = [], [], [], 0, None
    for i in range(len(pa)):
        res = triangulate_with_swap_check(cams, [Pose2D(pa[i], sa[i]), Pose2D(pb[i], sb[i])], prev=prev)
        prev = res
        swaps += any(res.swapped)
        valid.append(res.valid.mean())
        errs.extend(res.reproj_err[res.valid])
        P = res.points
        for k, (x, y) in SEGS.items():
            if res.valid[IDX[x]] and res.valid[IDX[y]]:
                L[k].append(np.linalg.norm(P[IDX[x]] - P[IDX[y]]) * 100)
        feet = [IDX[n] for n in ("left_heel", "right_heel", "left_big_toe", "right_big_toe") if res.valid[IDX[n]]]
        if res.valid[IDX["head"]] and feet:
            heights.append((P[IDX["head"], 1] - min(P[f, 1] for f in feet)) * 100)
    print(f"frames {len(pa)}, valid points {np.mean(valid):.0%}, reprojection median "
          f"{np.median(errs):.1f}px, frames with a camera L/R swap fixed: {swaps}")
    for k, v in L.items():
        v = np.array(v)
        print(f"  {k:12s} median {np.median(v):5.1f} cm  spread (std) {v.std():4.1f} cm")
    print(f"  head-top to floor p90: {np.percentile(heights, 90):.0f} cm")


if __name__ == "__main__":
    main()
