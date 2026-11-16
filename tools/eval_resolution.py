"""Does resolution / frame rate matter? One full-size recording sitting (phones e.g. at
1920x1080, the Mac at 1920x1080 30 fps) replayed at smaller sizes and lower Mac rates: same
motion, same calibration, so differences come from the camera settings alone. Each variant
is processed as the app would see it: pose on the smaller frames, intrinsics scaled to that
size, the hub's per-camera thresholds and weights, the hub's stabiliser.

  A. phones at full size / 720p / the 480p mode's pixel size, Mac 720p at 10 fps (the app)
  B. phones as in the app (480p mode), the Mac: off, 1080p / 720p / 480p at 10 fps,
     720p at 30 / 15 / 5 fps

  .venv/bin/python -m tools.eval_resolution data/rec_res_X_sit --calib-rec data/rec_res_X_stand
  (calibrates all cameras at full size on the standing part first; or pass --calib)
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


def evaluate(cams, tr, label, ms, mac_fps=10.0, base_px=25.0):
    t = tr[0].t[(tr[0].t > tr[0].t[0] + 1) & (tr[0].t < tr[0].t[-1] - 1)]
    k0, s0 = interp_track(tr[0], t)
    k1, s1 = interp_track(tr[1], t)
    frames = [[Pose2D(k0[i], s0[i]), Pose2D(k1[i], s1[i])] for i in range(len(t))]
    if len(tr) > 2:                                    # the Mac as in the app: 10 fps, 2 % border
        m = tr[2]
        sel, last = np.zeros(len(m.t), bool), -1e9
        for i, x in enumerate(m.t):
            if x - last >= 1.0 / mac_fps - 0.001:
                sel[i], last = True, x
        m10 = PoseTrack(m.t[sel], m.kp[sel], m.sc[sel], m.width, m.height, m.info)
        k2, s2 = interp_track(m10, t - cams[2].time_offset, max_gap=2.5 / mac_fps)
        out = (k2[..., 0] < .02 * m.width) | (k2[..., 0] > .98 * m.width) | (k2[..., 1] < .02 * m.height) | (k2[..., 1] > .98 * m.height)
        s2 = np.where(out, 0, s2)
        for i, f in enumerate(frames):
            f.append(Pose2D(k2[i], s2[i]) if s2[i].max() > 0 else None)
    cams = cams[:len(frames[0])]
    res = run(cams, frames, view_thresholds(cams, base_px) if len(cams) > 2 else base_px, view_weights(cams) if len(cams) > 2 else 1.0)
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
    ap.add_argument("rec", help="the sitting recording")
    ap.add_argument("--calib", help="calibration made at the recording's size")
    ap.add_argument("--calib-rec", help="standing recording to calibrate on (all cameras, full size)")
    ap.add_argument("--height", type=float, default=1.73)
    ap.add_argument("--backend", default="rtmpose-m:gpu")
    a = ap.parse_args()
    calib = a.calib
    if calib is None:
        calib = f"{a.calib_rec}/calibration.json"
        if not os.path.exists(calib):
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "tools.calibrate", "--rec", a.calib_rec, "--height",
                            str(a.height), "--out", calib], check=True)
    cams, _ = load_calibration(calib)
    info = json.load(open(f"{a.rec}/info.json"))
    be = make_backend(a.backend)
    short = lambda c: min(c.width, c.height)
    def phones(s):
        tr, ms = [], []
        for i in range(2):
            p, m = poses(a.rec, f"cam{i}", be, s, info.get(f"cam{i}", {}))
            tr.append(p); ms.append(m)
        return tr, [scaled(c, s) for c in cams[:2]], float(np.mean(ms))
    def mac(s):
        p, m = poses(a.rec, "cam2", be, s, info.get("cam2", {}))
        return p, scaled(cams[2], s), m
    ph_long = max(cams[0].width, cams[0].height)
    has_mac = len(cams) > 2
    mac_s = {"1080p": 1080 / short(cams[2]), "720p": 720 / short(cams[2]), "480p": 480 / short(cams[2])} if has_mac else {}
    mac_s = {k: v for k, v in mac_s.items() if v <= 1.0001}
    print(f"{a.rec}: phones {cams[0].width}x{cams[0].height}" + (f", Mac {cams[2].width}x{cams[2].height}" if has_mac else ""))

    print("\nA. phone resolution (Mac 720p at 10 fps, as in the app)")
    for target, name in ((ph_long, f"{short(cams[0])}p (as recorded)"), (1280, "720p"), (640, "480p mode")):
        if target > ph_long:
            continue
        tr, cs, ms = phones(target / ph_long)
        if has_mac and "720p" in mac_s:
            m, mc, _ = mac(mac_s["720p"])
            evaluate(cs + [mc], tr + [m], "phones " + name, ms)
            if target != 640:     # the 25 px threshold was tuned at 480p: same angle at this size
                evaluate(cs + [mc], tr + [m], "  + threshold scaled", ms, base_px=25.0 * target / 640)
        else:
            evaluate(cs, tr, "phones " + name, ms)

    if not has_mac:
        return
    print("\nB. the MacBook camera (phones in the 480p mode, as in the app)")
    tr, cs, _ = phones(min(1.0, 640 / ph_long))
    evaluate(cs, tr, "no Mac", 0.0)
    for name, s in mac_s.items():
        m, mc, ms = mac(s)
        evaluate(cs + [mc], tr + [m], f"Mac {name} 10 fps", ms, mac_fps=10)
    if "720p" in mac_s:
        m, mc, ms = mac(mac_s["720p"])
        for fps in (30, 15, 5):
            evaluate(cs + [mc], tr + [m], f"Mac 720p {fps} fps", ms, mac_fps=fps)


if __name__ == "__main__":
    main()
