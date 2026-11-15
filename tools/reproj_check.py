"""Visual proof that two-camera fusion is right: triangulated 3D points are
projected back into both camera images. Green = what the model detected in
that camera, red = where the fused 3D point lands. They should coincide.

  .venv/bin/python -m tools.reproj_check --frames 100 300 500
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from fbt.calibrate import load_calibration, pair_tracks, poses_from_recording
from fbt.pose import Pose2D, make_backend
from fbt.skeleton import BONE_IDX
from fbt.triangulate import triangulate_with_swap_check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("--frames", type=int, nargs="+", default=[100, 300, 500])
    a = ap.parse_args()
    cams, _ = load_calibration(a.calib)
    rec = json.load(open(a.calib))["meta"]["recording"]
    be = make_backend("rtmpose-m:gpu")
    ta, tb = poses_from_recording(rec, "cam0", be), poses_from_recording(rec, "cam1", be)
    pa, pb, sa, sb, ts = pair_tracks(ta, tb)
    caps = [cv2.VideoCapture(f"{rec}/cam{i}.mp4") for i in (0, 1)]
    times = [ta.t, tb.t]
    rows, errs, prev = [], [], None
    for i in range(len(pa)):
        res = triangulate_with_swap_check(cams, [Pose2D(pa[i], sa[i]), Pose2D(pb[i], sb[i])], prev=prev)
        prev = res
        if i not in a.frames:
            continue
        tiles = []
        for ci, (cam, kp, sc) in enumerate(((cams[0], pa[i], sa[i]), (cams[1], pb[i], sb[i]))):
            fidx = int(np.argmin(np.abs(times[ci] - ts[i])))
            caps[ci].set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, img = caps[ci].read()
            uv, _ = cam.project(res.points)
            for a_, b_ in BONE_IDX:
                if sc[a_] > 0.3 and sc[b_] > 0.3:
                    cv2.line(img, tuple(map(int, kp[a_])), tuple(map(int, kp[b_])), (0, 200, 0), 2)
                if res.valid[a_] and res.valid[b_]:
                    cv2.line(img, tuple(map(int, uv[a_])), tuple(map(int, uv[b_])), (0, 0, 255), 1)
            for k in np.where(res.valid & (sc > 0.3))[0]:
                errs.append(np.linalg.norm(uv[k] - kp[k]))
                cv2.circle(img, tuple(map(int, kp[k])), 5, (0, 200, 0), 2)
                cv2.circle(img, tuple(map(int, uv[k])), 3, (0, 0, 255), -1)
            h, w = img.shape[:2]
            y0 = int(max(0, np.nanmin(kp[:, 1]) - 60)); y1 = int(min(h, np.nanmax(kp[:, 1]) + 60))
            x0 = int(max(0, np.nanmin(kp[:, 0]) - 80)); x1 = int(min(w, np.nanmax(kp[:, 0]) + 80))
            crop = img[y0:y1, x0:x1]
            tiles.append(cv2.resize(crop, (int(crop.shape[1] * 500 / crop.shape[0]), 500)))
        rows.append(np.hstack(tiles))
    width = max(r.shape[1] for r in rows)
    out = np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in rows])
    cv2.imwrite("data/reproj_check.jpg", out)
    e = np.array(errs)
    print(f"reprojection error on shown frames: median {np.median(e):.1f}px, p90 {np.percentile(e, 90):.1f}px, max {e.max():.1f}px")
    print("saved data/reproj_check.jpg")


if __name__ == "__main__":
    main()
