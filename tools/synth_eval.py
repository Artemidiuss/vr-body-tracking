"""Evaluate triangulation on the synthetic rig: how accurate is each camera
placement, per motion, with a realistic noise model?

  .venv/bin/python -m tools.synth_eval
  .venv/bin/python -m tools.synth_eval --layout=-45,45 --layout=-45,45,180 --sigma 6

Error = distance between triangulated and ground-truth joints (cm), over
the leg joints that matter for VRChat trackers. "lost" = share of
leg-joint samples that could not be triangulated (too few cameras / failed
check) and would have to be filled by the skeleton model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from fbt.camera import ring_of_cameras
from fbt.skeleton import IDX
from fbt.synthetic import MOTIONS, NoiseModel, motion_sequence, render_2d
from fbt.triangulate import triangulate_with_swap_check

LEGS = [IDX[n] for n in ("hip", "left_knee", "right_knee", "left_ankle",
                         "right_ankle", "left_heel", "right_heel",
                         "left_big_toe", "right_big_toe")]


def evaluate(angles, noise_kw, seed=0):
    cams = ring_of_cameras(angles)
    noise = NoiseModel(seed=seed, **noise_kw)
    per_motion = defaultdict(lambda: {"err": [], "lost": [], "swap_wrong": 0, "n": 0})
    prev = None
    for _, J, name in motion_sequence(MOTIONS):
        poses = [render_2d(c, J, noise) for c in cams]
        res = triangulate_with_swap_check(cams, poses, prev=prev)
        prev = res
        m = per_motion[name]
        v = res.valid[LEGS]
        d = np.linalg.norm(res.points[LEGS] - J[LEGS], axis=1)
        m["err"].extend(d[v] * 100)
        m["lost"].append(1 - v.mean())
        lk, rk = IDX["left_knee"], IDX["right_knee"]
        if res.valid[lk] and np.linalg.norm(res.points[lk] - J[rk]) < np.linalg.norm(res.points[lk] - J[lk]):
            m["swap_wrong"] += 1
        m["n"] += 1
    return per_motion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", action="append", dest="layouts",
                    help="camera angles, e.g. --layout=-45,45 (repeatable)")
    ap.add_argument("--sigma", type=float, default=3.0, help="2D jitter px (front view)")
    a = ap.parse_args()
    a.layouts = a.layouts or ["-15,15", "-30,30", "-45,45", "-60,60", "-90,90", "0,180", "-45,45,180"]
    noise_kw = {"pixel_sigma": a.sigma}
    print(f"noise: {noise_kw}; cells = mean leg error cm / lost % / wrong-side %\n")
    header = f"{'layout (deg)':<14}" + "".join(f"{m:>18}" for m in MOTIONS) + f"{'ALL':>18}"
    print(header)
    for lay in a.layouts:
        angles = [float(x) for x in lay.split(",")]
        res = evaluate(angles, noise_kw)
        cells, allerr, alllost, allwrong, alln = [], [], [], 0, 0
        for mname in MOTIONS:
            m = res[mname]
            e = np.mean(m["err"]) if m["err"] else float("nan")
            cells.append(f"{e:5.1f}/{100 * np.mean(m['lost']):3.0f}/{100 * m['swap_wrong'] / m['n']:3.0f}")
            allerr += m["err"]; alllost += m["lost"]; allwrong += m["swap_wrong"]; alln += m["n"]
        cells.append(f"{np.mean(allerr):5.1f}/{100 * np.mean(alllost):3.0f}/{100 * allwrong / alln:3.0f}")
        print(f"{lay:<14}" + "".join(f"{c:>18}" for c in cells))


if __name__ == "__main__":
    main()
