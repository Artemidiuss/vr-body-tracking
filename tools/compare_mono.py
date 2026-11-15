"""One camera vs two: compare single-camera 3D models against two-camera
triangulation on a calibrated recording.

The two-camera skeleton is the reference (≈2 cm on the synthetic rig). For
each single-camera model and each camera we report:
  shape err   — per-frame best-fit alignment (rotation+scale+translation):
                how wrong the body *shape* is (PA-MPJPE), legs + torso, cm
  fixed err   — one rotation+scale for the whole recording, translation per
                frame: what a tracker would actually see (depth mistakes of
                the knees/feet relative to the hips show up here), cm
  depth err   — the part of "fixed err" along the camera's viewing axis, cm
  bone spread — std of thigh/shin lengths over time, cm (real bones don't change)
  jitter      — mean frame-to-frame acceleration of the feet, cm/frame²

  .venv/bin/python -m tools.compare_mono
  .venv/bin/python -m tools.compare_mono --models mediapipe-full rtmw3d --every 2
"""

from __future__ import annotations

import argparse
import csv
import json

import cv2
import numpy as np

from fbt.calibrate import load_calibration, pair_tracks, poses_from_recording
from fbt.mono3d import make_mono
from fbt.pose import Pose2D, make_backend
from fbt.skeleton import IDX
from fbt.triangulate import triangulate_with_swap_check

JOINTS = [IDX[n] for n in ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
                           "right_ankle", "left_heel", "right_heel", "left_big_toe",
                           "right_big_toe", "left_shoulder", "right_shoulder")]
BONES = [(IDX[a], IDX[b]) for a, b in (("left_hip", "left_knee"), ("right_hip", "right_knee"),
                                       ("left_knee", "left_ankle"), ("right_knee", "right_ankle"))]
FEET = [IDX[n] for n in ("left_ankle", "right_ankle", "left_heel", "right_heel")]


def similarity(A, B):
    """s, R, t minimising |s R A + t - B| (Umeyama)."""
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (A0 ** 2).sum()
    return s, R, mb - s * R @ ma


def stereo_reference(rec, cams, backend):
    ta, tb = poses_from_recording(rec, "cam0", backend), poses_from_recording(rec, "cam1", backend)
    pa, pb, sa, sb, ts = pair_tracks(ta, tb)
    out, prev = {}, None
    for i in range(len(pa)):
        r = triangulate_with_swap_check(cams, [Pose2D(pa[i], sa[i]), Pose2D(pb[i], sb[i])], prev=prev)
        prev = r
        out[round(ts[i], 4)] = (r.points.copy(), r.valid.copy())
    return out, ts


def mono_track(rec, cam, model, every):
    times = [float(r["t_capture"]) for r in csv.DictReader(open(f"{rec}/{cam}.csv"))]
    cap = cv2.VideoCapture(f"{rec}/{cam}.mp4")
    res, i = {}, 0
    while True:
        ok, img = cap.read()
        if not ok or i >= len(times):
            break
        if i % every == 0:
            r = model(img)
            if r is not None:
                res[times[i]] = r
        i += 1
    return res


def evaluate(ref, ref_times, mono, cam_R, label):
    """mono: {t: (P, S)} camera axes; ref: {t: (P, valid)} world."""
    pairs = []
    for t, (P, S) in mono.items():
        j = np.argmin(np.abs(ref_times - t))
        if abs(ref_times[j] - t) > 0.02:
            continue
        RP, RV = ref[round(ref_times[j], 4)]
        m = np.array([RV[k] and S[k] > 0.3 for k in JOINTS])
        if m.sum() < 8:
            continue
        pairs.append((np.array(JOINTS)[m], P, RP))
    if not pairs:
        print(f"{label}: no overlapping frames")
        return None
    # Per-frame alignment (shape error).
    shape = []
    for js, P, RP in pairs:
        s, R, t = similarity(P[js], RP[js])
        shape.append(np.linalg.norm((s * P[js] @ R.T + t) - RP[js], axis=1).mean())
    # One global rotation+scale for the recording (hip-centred), translation per frame.
    B = np.concatenate([RP[js] - RP[js].mean(0) for js, P, RP in pairs])
    A = np.concatenate([(P[js] - P[js].mean(0)) for js, P, RP in pairs])
    s, R, _ = similarity(A, B)
    view = cam_R[2]  # camera viewing axis in world
    fixed, depth, bone, feet_traj = [], [], [], []
    for js, P, RP in pairs:
        Q = s * (P[js] - P[js].mean(0)) @ R.T + RP[js].mean(0)
        d = Q - RP[js]
        fixed.append(np.linalg.norm(d, axis=1).mean())
        depth.append(np.abs(d @ view).mean())
        bone.append([np.linalg.norm(s * (P[a] - P[b])) for a, b in BONES])
        feet_traj.append(s * P[FEET] @ R.T)
    bone = np.array(bone)
    ft = np.array(feet_traj)
    jit = np.linalg.norm(np.diff(ft, 2, axis=0), axis=2).mean() * 100 if len(ft) > 3 else np.nan
    row = (label, np.mean(shape) * 100, np.mean(fixed) * 100, np.mean(depth) * 100,
           bone.std(0).mean() * 100, jit, len(pairs))
    print(f"{label:<28} {row[1]:9.1f} {row[2]:9.1f} {row[3]:9.1f} {row[4]:11.1f} {row[5]:8.2f} {row[6]:7d}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("--rec")
    ap.add_argument("--models", nargs="+", default=["mediapipe-full", "rtmw3d"])
    ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    cams, _ = load_calibration(a.calib)
    rec = a.rec or json.load(open(a.calib))["meta"]["recording"]
    print("two-camera reference…")
    ref, ref_t = stereo_reference(rec, cams, make_backend("rtmpose-m:gpu"))
    # Stereo self-check: its own stats with the same metric code.
    stereo = {}
    for t in ref_t[::a.every]:
        P, V = ref[round(t, 4)]
        Pc = (P - P[IDX["hip"]]) @ cams[0].R.T   # into camera-0 axes
        stereo[t] = (Pc, V.astype(float))
    print(f"\n{'model / camera':<28} {'shape cm':>9} {'fixed cm':>9} {'depth cm':>9} {'bone spread':>11} {'jitter':>8} {'frames':>7}")
    evaluate(ref, ref_t, stereo, cams[0].R, "2 cameras (reference)")
    for spec in a.models:
        model = make_mono(spec)
        for ci, cam in enumerate(("cam0", "cam1")):
            mono = mono_track(rec, cam, model, a.every)
            evaluate(ref, ref_t, mono, cams[ci].R, f"{model.name} / {cams[ci].name.split()[0]}")
    print("\nshape/fixed/depth = mean joint error vs two-camera skeleton; "
          "bone spread = thigh/shin length std; jitter = feet acceleration")


if __name__ == "__main__":
    main()
