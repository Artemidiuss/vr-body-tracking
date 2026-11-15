"""Camera calibration from the user's own body — no checkerboard.

1. Intrinsics: initial guess from the phone (focal length mm, sensor size mm
   in the FBT Cam HELLO), refined later by bundle adjustment.
2. Relative pose of camera B to camera A: essential matrix from body
   keypoints seen by both cameras over many frames (the person moving gives
   enough 3D variety), RANSAC, then bundle adjustment of rotation,
   translation, focal lengths and all 3D points with a robust loss.
3. Metric scale: from the user's height.
4. World frame for VRChat: Y up (median torso direction while standing),
   floor at Y=0 (lowest foot points), origin under the user, Z towards the
   cameras.

Assumes negligible lens distortion: phones apply distortion correction to
ultra-wide video by default (straight walls look straight).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .camera import Camera
from .skeleton import IDX, NUM_KEYPOINTS
from .triangulate import triangulate_point

# Body points only: face points are unreliable from behind and nearly coplanar.
BODY = [IDX[n] for n in (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_big_toe", "right_big_toe", "neck", "head")]
FEET = [IDX[n] for n in ("left_heel", "right_heel", "left_big_toe", "right_big_toe",
                         "left_small_toe", "right_small_toe")]


def intrinsics_from_info(info: dict, w: int, h: int) -> np.ndarray:
    """K for a (possibly rotated) w x h image. The video's long side spans
    the sensor's long side (16:9 is cropped from 4:3 on the short side)."""
    f = info["focal_mm"] * max(w, h) / max(info["sensor_mm"])
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


@dataclass
class PoseTrack:
    """2D poses of one camera over a recording."""
    t: np.ndarray        # (N,) capture time, Mac clock
    kp: np.ndarray       # (N,26,2)
    sc: np.ndarray       # (N,26)
    width: int
    height: int
    info: dict


def poses_from_recording(rec_dir: str, cam: str, backend, every: int = 1) -> PoseTrack:
    info = json.load(open(f"{rec_dir}/info.json"))[cam]
    times = [float(r["t_capture"]) for r in csv.DictReader(open(f"{rec_dir}/{cam}.csv"))]
    cap = cv2.VideoCapture(f"{rec_dir}/{cam}.mp4")
    T, K, S = [], [], []
    i, w, h = 0, 0, 0
    while True:
        ok, img = cap.read()
        if not ok or i >= len(times):
            break
        h, w = img.shape[:2]
        if i % every == 0:
            p = backend(img)
            if p is not None:
                T.append(times[i]); K.append(p.keypoints); S.append(p.scores)
        i += 1
    return PoseTrack(np.array(T), np.array(K), np.array(S), w, h, info)


def pair_tracks(a: PoseTrack, b: PoseTrack, max_dt: float = 0.04):
    """For each frame of A, B's keypoints linearly interpolated to A's
    capture time (true sync via sensor timestamps + clock sync)."""
    pa, pb, sa, sb, ts = [], [], [], [], []
    for i, t in enumerate(a.t):
        j = np.searchsorted(b.t, t)
        if j == 0 or j >= len(b.t):
            continue
        t0, t1 = b.t[j - 1], b.t[j]
        if t1 - t0 > max_dt * 2 or min(t - t0, t1 - t) > max_dt:
            continue
        w = (t - t0) / (t1 - t0)
        pa.append(a.kp[i]); sa.append(a.sc[i])
        pb.append((1 - w) * b.kp[j - 1] + w * b.kp[j])
        sb.append(np.minimum(b.sc[j - 1], b.sc[j]))
        ts.append(t)
    return np.array(pa), np.array(pb), np.array(sa), np.array(sb), np.array(ts)


def _proj(K, R, t, X):
    Xc = X @ R.T + t
    return (Xc[:, :2] / Xc[:, 2:3]) * K[0, 0] + K[:2, 2]


def calibrate_pair(a: PoseTrack, b: PoseTrack, height_m: float, min_score: float = 0.6,
                   max_points: int = 2500, seed: int = 0, log=print):
    pa, pb, sa, sb, ts = pair_tracks(a, b)
    if len(pa) < 30:
        raise RuntimeError(f"only {len(pa)} synchronised frames — record longer / check both cameras see you")
    KA = intrinsics_from_info(a.info, a.width, a.height)
    KB = intrinsics_from_info(b.info, b.width, b.height)

    # Observations: (frame, joint) seen confidently by both cameras.
    obs = [(f, k) for f in range(len(pa)) for k in BODY
           if sa[f, k] >= min_score and sb[f, k] >= min_score]
    rng = np.random.default_rng(seed)
    if len(obs) > max_points:
        obs = [obs[i] for i in rng.choice(len(obs), max_points, replace=False)]
    ua = np.array([pa[f, k] for f, k in obs])
    ub = np.array([pb[f, k] for f, k in obs])
    log(f"  synced frames: {len(pa)}, correspondences used: {len(obs)}")

    # Essential matrix on normalised coordinates (different K per camera).
    na = cv2.undistortPoints(ua.reshape(-1, 1, 2), KA, None).reshape(-1, 2)
    nb = cv2.undistortPoints(ub.reshape(-1, 1, 2), KB, None).reshape(-1, 2)
    thr = 4.0 / KA[0, 0]
    E, mask = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=thr)
    _, R, t, mask2 = cv2.recoverPose(E, na, nb, np.eye(3), mask=mask)
    inl = mask2.ravel() > 0
    log(f"  essential matrix inliers: {inl.mean():.0%}")
    ua, ub = ua[inl], ub[inl]
    obs = [o for o, m in zip(obs, inl) if m]
    t = t.ravel()

    # Initial 3D points.
    PA = KA @ np.hstack([np.eye(3), np.zeros((3, 1))])
    PB = KB @ np.hstack([R, t[:, None]])
    X0 = np.array([triangulate_point([PA, PB], np.stack([x, y]), np.ones(2)) for x, y in zip(ua, ub)])

    # Bundle adjustment: rvec(3), t direction (2 angles), fA, fB, points.
    def t_from(ang):
        th, ph = ang
        return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])

    th0 = np.arccos(np.clip(t[2], -1, 1))
    ph0 = np.arctan2(t[1], t[0])
    rv0 = cv2.Rodrigues(R)[0].ravel()
    n = len(X0)
    x0 = np.concatenate([rv0, [th0, ph0, KA[0, 0], KB[0, 0]], X0.ravel()])

    def residuals(x):
        rv, ang, fa, fb = x[:3], x[3:5], x[5], x[6]
        X = x[7:].reshape(-1, 3)
        Rb = cv2.Rodrigues(rv)[0]
        Ka = KA.copy(); Ka[0, 0] = Ka[1, 1] = fa
        Kb = KB.copy(); Kb[0, 0] = Kb[1, 1] = fb
        ra = _proj(Ka, np.eye(3), np.zeros(3), X) - ua
        rb = _proj(Kb, Rb, t_from(ang), X) - ub
        return np.concatenate([ra.ravel(), rb.ravel()])

    sp = lil_matrix((4 * n, 7 + 3 * n), dtype=int)
    for i in range(n):
        for r in (2 * i, 2 * i + 1):            # camera A rows
            sp[r, 5] = 1
            sp[r, 7 + 3 * i: 10 + 3 * i] = 1
        for r in (2 * n + 2 * i, 2 * n + 2 * i + 1):  # camera B rows
            sp[r, 0:5] = 1
            sp[r, 6] = 1
            sp[r, 7 + 3 * i: 10 + 3 * i] = 1
    r0 = residuals(x0)
    res = least_squares(residuals, x0, jac_sparsity=sp, loss="huber", f_scale=3.0,
                        x_scale="jac", max_nfev=200)
    r1 = res.fun.reshape(-1, 2)
    err0 = np.linalg.norm(r0.reshape(-1, 2), axis=1)
    err1 = np.linalg.norm(r1, axis=1)
    log(f"  reprojection error px: before BA median {np.median(err0):.1f}, after {np.median(err1):.1f} "
        f"(p90 {np.percentile(err1, 90):.1f})")
    rv, ang, fa, fb = res.x[:3], res.x[3:5], res.x[5], res.x[6]
    R = cv2.Rodrigues(rv)[0]
    t = t_from(ang)
    KA[0, 0] = KA[1, 1] = fa
    KB[0, 0] = KB[1, 1] = fb
    log(f"  focal px: A {fa:.0f} (phone said {intrinsics_from_info(a.info, a.width, a.height)[0, 0]:.0f}), "
        f"B {fb:.0f} (phone said {intrinsics_from_info(b.info, b.width, b.height)[0, 0]:.0f})")

    # Triangulate whole skeletons (unit-baseline frame A coordinates).
    PA = KA @ np.hstack([np.eye(3), np.zeros((3, 1))])
    PB = KB @ np.hstack([R, t[:, None]])
    skel = np.full((len(pa), NUM_KEYPOINTS, 3), np.nan)
    for f in range(len(pa)):
        for k in range(NUM_KEYPOINTS):
            if sa[f, k] >= 0.4 and sb[f, k] >= 0.4:
                skel[f, k] = triangulate_point([PA, PB], np.stack([pa[f, k], pb[f, k]]),
                                               np.array([sa[f, k], sb[f, k]]))

    # Up direction: median ankle-midpoint -> neck vector.
    ank = np.nanmean(skel[:, [IDX["left_ankle"], IDX["right_ankle"]]], axis=1)
    torso = skel[:, IDX["neck"]] - ank
    torso = torso[np.isfinite(torso).all(1)]
    up = np.median(torso / np.linalg.norm(torso, axis=1, keepdims=True), axis=0)
    up /= np.linalg.norm(up)
    # Scale from height: head top to floor along "up".
    hgt = (skel[:, IDX["head"]] @ up) - np.nanmin(skel[:, FEET] @ up, axis=1)
    hgt = hgt[np.isfinite(hgt)]
    if len(hgt) < 10:
        raise RuntimeError("not enough full-body frames to estimate scale")
    measured = np.percentile(hgt, 90)              # standing tall frames
    head_to_top = 0.03                             # 'head' keypoint sits a bit below the crown
    scale = (height_m - head_to_top) / measured
    log(f"  baseline between cameras: {scale:.2f} m")

    # World frame.
    floor = np.nanpercentile(skel[:, FEET] @ up, 2)
    hip = skel[:, IDX["hip"]]
    hip = hip[np.isfinite(hip).all(1)]
    centre = np.median(hip, axis=0)
    centre = centre - (centre @ up - floor) * up       # on the floor under the user
    cams_c = np.stack([np.zeros(3), -R.T @ t])           # centres in frame-A coords
    fwd = cams_c.mean(0) - centre
    fwd -= (fwd @ up) * up
    fwd /= np.linalg.norm(fwd)
    xaxis = np.cross(up, fwd)
    Rw = np.stack([xaxis, up, fwd])                      # world <- A-frame rotation (rows)
    # A-frame point p -> world: Rw @ (p - centre) * scale
    cams = []
    for name, K, Rc, tc, tr in (("A", KA, np.eye(3), np.zeros(3), a), ("B", KB, R, t, b)):
        # world -> cam:  x_c = Rc (p) + tc, p = Rw^T (X/scale) + centre
        R_wc = Rc @ Rw.T
        t_wc = (Rc @ centre + tc) * scale
        cams.append(Camera(tr.info.get("device", name), K, R_wc, t_wc, tr.width, tr.height))
    for c in cams:
        log(f"  {c.name}: position {np.round(c.center, 2)} m (height above floor {c.center[1]:.2f} m)")
    a_dir = cams[0].R[2]; b_dir = cams[1].R[2]
    ang = np.degrees(np.arccos(np.clip(np.dot(a_dir, b_dir), -1, 1)))
    log(f"  angle between camera directions: {ang:.0f}°")
    return cams, {"reproj_median_px": float(np.median(err1)), "baseline_m": float(scale),
                  "angle_deg": float(ang), "frames": int(len(pa)), "points": int(n)}


def save_calibration(path: str, cams: list[Camera], hosts: list[str], meta: dict):
    data = {"meta": meta, "cameras": [
        {"name": c.name, "host": h, "K": c.K.tolist(), "R": c.R.tolist(), "t": c.t.tolist(),
         "width": c.width, "height": c.height} for c, h in zip(cams, hosts)]}
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def load_calibration(path: str) -> tuple[list[Camera], list[str]]:
    d = json.load(open(path))
    cams = [Camera(c["name"], np.array(c["K"]), np.array(c["R"]), np.array(c["t"]),
                   c["width"], c["height"]) for c in d["cameras"]]
    return cams, [c["host"] for c in d["cameras"]]
