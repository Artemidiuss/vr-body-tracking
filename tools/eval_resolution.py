"""Does the phones' resolution matter? One high-resolution recording (e.g. phones at
1920x1080, calibrated at that size) replayed at full size and downscaled to 720p and
to the 480p mode's pixel size (long side 640): the same motion and calibration, so
any difference comes from resolution alone. Each size is processed as the app would
see it: pose on the smaller frames, cameras with intrinsics scaled to that size,
the hub's per-camera thresholds, the Mac at 10 fps, the hub's stabiliser.

  .venv/bin/python -m tools.eval_resolution data/rec_res_XXXX --calib data/calibration.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import cv2
import numpy as np
from scipy.signal import savgol_filter

from fbt.calibrate import PoseTrack, load_calibration
from fbt.filters import SkeletonStabilizer
from fbt.pose import Pose2D, make_backend
from fbt.triangulate import PRIOR_SEGMENTS, view_thresholds, view_weights
from tools.eval_three_cams import LOWER, interp_track, run

LEG = [13, 14, 15, 16]


def poses(rec, cam, be, scale, info):
    """2D poses of one camera on frames downscaled by `scale` (keypoints in that size)."""
    cache = f"{rec}/poses_res{scale:.3f}_{cam}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        return PoseTrack(z["t"], z["kp"], z["sc"], int(z["wh"][0]), int(z["wh"][1]), info), float(z["ms"])
    import csv
    times = [float(r["t_capture"]) for r in csv.DictReader(open(f"{rec}/{cam}.csv"))]
    cap = cv2.VideoCapture(f"{rec}/{cam}.mp4")
    T, K, S, ms, i, w, h = [], [], [], [], 0, 0, 0
    while True:
        ok, img = cap.read()
        if not ok or i >= len(times):
            break
        if scale != 1.0:
            img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        p = be(img)
        ms.append((time.perf_counter() - t0) * 1000)
        if p is not None:
            T.append(times[i]); K.append(p.keypoints); S.append(p.scores)
        i += 1
    m = float(np.median(ms[20:])) if len(ms) > 20 else float("nan")
    np.savez(cache, t=T, kp=K, sc=S, wh=[w, h], ms=m)
    return PoseTrack(np.array(T), np.array(K), np.array(S), w, h, info), m


def scaled(cam, s):
    K = cam.K.copy()
    K[:2] *= s
    return dataclasses.replace(cam, K=K, width=round(cam.width * s), height=round(cam.height * s))


def evaluate(cams, tr, label, ms):
    t = tr[0].t[(tr[0].t > tr[0].t[0] + 1) & (tr[0].t < tr[0].t[-1] - 1)]
    k0, s0 = interp_track(tr[0], t)
    k1, s1 = interp_track(tr[1], t)
    frames = [[Pose2D(k0[i], s0[i]), Pose2D(k1[i], s1[i])] for i in range(len(t))]
    if len(tr) > 2:                                    # the Mac as in the app: 10 fps, 2 % border
        m = tr[2]
        sel, last = np.zeros(len(m.t), bool), -1e9
        for i, x in enumerate(m.t):
            if x - last >= 0.099:
                sel[i], last = True, x
        m10 = PoseTrack(m.t[sel], m.kp[sel], m.sc[sel], m.width, m.height, m.info)
        k2, s2 = interp_track(m10, t - cams[2].time_offset, max_gap=0.25)
        out = (k2[..., 0] < .02 * m.width) | (k2[..., 0] > .98 * m.width) | (k2[..., 1] < .02 * m.height) | (k2[..., 1] > .98 * m.height)
        s2 = np.where(out, 0, s2)
        for i, f in enumerate(frames):
            f.append(Pose2D(k2[i], s2[i]) if s2[i].max() > 0 else None)
    res = run(cams, frames, view_thresholds(cams), view_weights(cams))
    P = np.array([r.points for r in res])
    V = np.array([r.valid for r in res])
    L = np.array([[np.linalg.norm(r.points[a] - r.points[b]) if r.valid[a] and r.valid[b] else np.nan
                   for a, b in PRIOR_SEGMENTS] for r in res])
    spread = (np.nanpercentile(L, 75, 0) - np.nanpercentile(L, 25, 0)) / 1.349 * 100
    st = SkeletonStabilizer(min_cutoff=2.5, beta=0.4, floor_clip=True)
    S = np.array([st(p, v, x) for p, v, x in zip(P, V, t)])
    g = np.arange(t[0], t[-1], 1 / 30)
    Y = np.stack([np.interp(g, t, S[:, k, d]) for k in LEG for d in range(3)], 1).reshape(len(g), 4, 3)
    dev = np.linalg.norm(Y - savgol_filter(Y, 15, 2, axis=0), axis=-1) * 100
    leg_sc = np.mean([np.mean(x.sc[:, LEG]) for x in tr[:2]])
    print(f"{label:22} pose {ms:4.1f} ms | leg confidence {leg_sc:.3f} | leg points lost {np.mean(~V[:, LEG]):5.1%} | "
          f"leg bones spread {np.nanmean(spread[LOWER]):.2f} cm | after stabiliser: shake {np.median(dev):.2f} cm, "
          f"frames > 3 cm {np.mean((dev > 3).any(1)):5.1%}, > 5 cm {np.mean((dev > 5).any(1)):4.1%}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("--backend", default="rtmpose-m:gpu")
    a = ap.parse_args()
    cams, _ = load_calibration(a.calib)
    info = json.load(open(f"{a.rec}/info.json"))
    be = make_backend(a.backend)
    long_side = max(cams[0].width, cams[0].height)
    mac = None
    if len(cams) > 2:
        mac, _ = poses(a.rec, "cam2", be, 1.0, info.get("cam2", {}))
    sizes = [(1.0, f"{min(cams[0].width, cams[0].height)}p (as recorded)")]
    for target, name in ((1280, "720p"), (640, "480p mode")):
        if target < long_side:
            sizes.append((target / long_side, name))
    print(f"{a.rec}: phones at {cams[0].width}x{cams[0].height}")
    for s, name in sizes:
        tr, ms = [], []
        for i in range(2):
            p, m = poses(a.rec, f"cam{i}", be, s, info.get(f"cam{i}", {}))
            tr.append(p); ms.append(m)
        cs = [scaled(c, s) for c in cams[:2]] + list(cams[2:])
        evaluate(cs, tr + ([mac] if mac is not None else []), name, float(np.mean(ms)))


if __name__ == "__main__":
    main()
