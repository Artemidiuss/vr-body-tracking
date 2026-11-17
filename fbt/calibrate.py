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
import os
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .camera import Camera
from .skeleton import IDX, NUM_KEYPOINTS
from .triangulate import triangulate_point, triangulate_pose

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


def poses_from_recording(rec_dir: str, cam: str, backend, every: int = 1,
                         cache: bool = True) -> PoseTrack:
    """2D poses of every frame of a recorded camera. Cached next to the video
    (poses_<cam>.npz) so calibration and offline evaluations share one pass."""
    info = json.load(open(f"{rec_dir}/info.json"))[cam]
    cpath = f"{rec_dir}/poses_{cam}.npz"
    if cache and every == 1 and os.path.exists(cpath):
        z = np.load(cpath)
        return PoseTrack(z["t"], z["kp"], z["sc"], int(z["wh"][0]), int(z["wh"][1]), info)
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
    tr = PoseTrack(np.array(T), np.array(K), np.array(S), w, h, info)
    if cache and every == 1 and len(T):
        np.savez(cpath, t=tr.t, kp=tr.kp, sc=tr.sc, wh=np.array([w, h]))
    return tr


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


def upright_direction(skel: np.ndarray, cone_deg: float = 12.0) -> np.ndarray:
    """'Up' for the world frame: the ankle-midpoint -> neck direction while standing straight.

    The median over the whole recording mixes in crouches, leans and reaches: one calibration
    came out 3.5 deg off the upright frames that way, so in VRChat the feet sat ~8 cm to one
    side of a player standing straight (ARCHITECTURE.md §26). Starting from that median, keep
    only the frames within a narrow cone of it and take their median, a few times over."""
    ank = np.nanmean(skel[:, [IDX["left_ankle"], IDX["right_ankle"]]], axis=1)
    torso = skel[:, IDX["neck"]] - ank
    torso = torso[np.isfinite(torso).all(1)]
    u = torso / np.linalg.norm(torso, axis=1, keepdims=True)
    up = np.median(u, axis=0)
    up /= np.linalg.norm(up)
    for _ in range(4):
        sel = u @ up > np.cos(np.radians(cone_deg))
        if sel.sum() < max(30, 0.2 * len(u)):
            break
        up = np.median(u[sel], axis=0)
        up /= np.linalg.norm(up)
    return up


def gravity_up_in_camera(track: PoseTrack) -> np.ndarray | None:
    """'Up' in this camera's frame (x right, y down, z forward in the image as calibrated),
    from the phone's accelerometer (FBT Cam STATUS 'gravity', stored in info['gravity']).

    Android reports gravity as an upward vector in device axes (X right, Y up, Z out of
    the screen, portrait). The Mac rotates a back camera's frames by sensor_orientation,
    which puts them in that same portrait orientation: image x = X, image y = -Y, and the
    lens looks along -Z. None when there is no reading, the phone was not still, or the
    frames are not in that orientation."""
    g = track.info.get("gravity") or {}
    v = g.get("v")
    rot = track.info.get("sensor_orientation")
    w0, h0 = track.info.get("width"), track.info.get("height")
    if v is None or g.get("sd_deg", 99) > 0.5 or rot not in (90, 270):
        return None
    if not (w0 and h0 and (track.width, track.height) == (h0, w0)):      # not the rotated frames
        return None
    u = np.array([v[0], -v[1], -v[2]], float)
    n = np.linalg.norm(u)
    return u / n if n > 1e-6 else None


def merge_gravity(readings: list) -> dict:
    """One gravity for the whole calibration from the phone's per-second readings (each a
    2 s average): the mean direction, and the worst spread, so a phone that was bumped
    (or moved) during the recording is not used."""
    V = np.array([r["v"] for r in readings], float)
    U = V / np.linalg.norm(V, axis=1, keepdims=True)
    m = U.mean(0)
    m /= np.linalg.norm(m)
    drift = max(_angle(u, m) for u in U)
    sd = max(float(r.get("sd_deg", 0)) for r in readings)
    return {"v": (m * np.linalg.norm(V, axis=1).mean()).tolist(), "sd_deg": round(max(sd, drift), 3),
            "n": len(readings)}


def _angle(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def choose_up(posture: np.ndarray, phones: list, log=print) -> tuple:
    """The vertical of the world frame: the standing posture (upright_direction).
    The phones' accelerometers are only logged next to it. Measured against the headset, their
    camera modules sit skewed to the accelerometer (Galaxy 1.2 deg, Redmi 4.4 deg), and on
    17.11 two phones agreeing within 1.9 deg were still 2.4 deg off together: the feet stood
    5-6 cm to one side of the head. With the posture's vertical: 1-2 cm (ARCHITECTURE.md §31).
    phones: up vectors in the same frame as posture, None where unknown."""
    g = [p for p in phones if p is not None]
    info = {"posture_vs_phones_deg": None, "phones_deg": None}
    if len(g) == 2:
        info["phones_deg"] = round(_angle(g[0], g[1]), 2)
    if g:
        mean = np.sum(g, axis=0)
        info["posture_vs_phones_deg"] = round(_angle(mean / np.linalg.norm(mean), posture), 2)
        log(f"  vertical: standing posture (phones' accelerometers {info['posture_vs_phones_deg']:.1f}° off it, for reference)")
    else:
        log("  vertical: standing posture")
    return posture, {"source": "posture", **info}


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

    # Vertical: the phones' accelerometers (frame A coordinates: x_B = R x_A + t), checked
    # against the standing posture.
    gA, gB = gravity_up_in_camera(a), gravity_up_in_camera(b)
    up, up_info = choose_up(upright_direction(skel), [gA, None if gB is None else R.T @ gB], log=log)
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
                  "angle_deg": float(ang), "frames": int(len(pa)), "points": int(n), "up": up_info}


def triangulate_tracks(cams: list[Camera], a: PoseTrack, b: PoseTrack, min_score: float = 0.4):
    """World skeletons from two calibrated cameras at A's frame times.
    Returns (times (N,), points (N,26,3) with NaN where not triangulated)."""
    from .pose import Pose2D
    pa, pb, sa, sb, ts = pair_tracks(a, b)
    skel = np.full((len(ts), NUM_KEYPOINTS, 3), np.nan)
    for f in range(len(ts)):
        r = triangulate_pose(cams, [Pose2D(pa[f], sa[f]), Pose2D(pb[f], sb[f])], min_score=min_score)
        skel[f, r.valid] = r.points[r.valid]
    return ts, skel


def _interp_skeleton(ts, skel, t_query, max_gap: float = 0.1):
    """Linear interpolation of (N,26,3) skeletons at t_query (M,); NaN outside / over gaps."""
    j = np.searchsorted(ts, t_query)
    ok = (j > 0) & (j < len(ts))
    j = np.clip(j, 1, len(ts) - 1)
    t0, t1 = ts[j - 1], ts[j]
    ok &= (t1 - t0) <= max_gap
    w = ((t_query - t0) / np.maximum(t1 - t0, 1e-9))[:, None, None]
    out = (1 - w) * skel[j - 1] + w * skel[j]       # NaN propagates
    out[~ok] = np.nan
    return out


# Points used to place an extra camera: body + face (a camera in front sees
# the face well; from behind the face points get low scores and drop out).
EXTRA_PTS = sorted(set(BODY) | {IDX[n] for n in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")})


def calibrate_extra(track: PoseTrack, ts: np.ndarray, skel: np.ndarray, name: str = "extra",
                    min_score: float = 0.5, border: float = 0.02, max_dt: float = 0.15,
                    focal: float | None = None, focal_range: tuple[float, float] | None = None,
                    log=print) -> tuple[Camera, dict]:
    """Place one more camera (e.g. the MacBook's) in an existing calibration.

    The phones already give the world 3D skeleton over the recording; this
    camera gives 2D keypoints of the same moments, so its pose is a standard
    PnP problem (no new essential-matrix step). Unknowns solved: rotation,
    translation, focal length (the Mac reports none: grid of field-of-view
    guesses + refinement) and the clock offset between this camera's
    timestamps and the phones' (grid search, re-solving each step).
    Robust (Huber) least squares on the reprojection error.

    focal / focal_range: the body seen from one distance barely separates
    focal length from distance (measured on the MacBook: one half of a
    recording gave 28° at 3.4 m, the other 60° at 1.5 m, same 20 px error).
    A known focal, or a plausible range, removes that degeneracy.
    """
    w, h = track.width, track.height
    cx, cy = w / 2, h / 2

    def corr(dt):
        X = _interp_skeleton(ts, skel, track.t + dt)
        m = (track.sc >= min_score) & np.isfinite(X).all(-1)
        u, v = track.kp[..., 0], track.kp[..., 1]
        m &= (u > border * w) & (u < (1 - border) * w) & (v > border * h) & (v < (1 - border) * h)
        keep = np.zeros(NUM_KEYPOINTS, bool); keep[EXTRA_PTS] = True
        m &= keep[None, :]
        return X[m], track.kp[m].astype(np.float64)

    def K_of(f):
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])

    def refine(X, u, rv, tv, f, max_nfev=100):
        if focal is not None:            # known intrinsics: pose only (6 unknowns)
            def res6(x):
                return (_proj(K_of(focal), cv2.Rodrigues(x[:3])[0], x[3:6], X) - u).ravel()
            r = least_squares(res6, np.concatenate([rv, tv]), loss="huber", f_scale=4.0,
                              x_scale="jac", max_nfev=max_nfev)
            return r.x[:3], r.x[3:6], focal, np.linalg.norm(r.fun.reshape(-1, 2), axis=1)

        def res(x):
            return (_proj(K_of(x[6]), cv2.Rodrigues(x[:3])[0], x[3:6], X) - u).ravel()
        lo, hi = focal_range if focal_range is not None else (1.0, np.inf)
        bounds = ([-np.inf] * 6 + [lo], [np.inf] * 6 + [hi])
        r = least_squares(res, np.concatenate([rv, tv, [np.clip(f, lo * 1.0001, hi * 0.9999)]]),
                          loss="huber", f_scale=4.0, x_scale="jac", max_nfev=max_nfev, bounds=bounds)
        e = np.linalg.norm(r.fun.reshape(-1, 2), axis=1)
        return r.x[:3], r.x[3:6], r.x[6], e

    X, u = corr(0.0)
    if len(X) < 50:
        raise RuntimeError(f"{name}: only {len(X)} usable points — is the person visible to this camera?")
    # Initial pose: RANSAC PnP over a grid of horizontal fields of view.
    best = None
    if focal is not None:
        fovs = [np.degrees(2 * np.arctan(w / 2 / focal))]
    elif focal_range is not None:
        fovs = np.degrees(2 * np.arctan(w / 2 / np.linspace(focal_range[1], focal_range[0], 6)))
    else:
        fovs = range(45, 101, 5)
    for hfov in fovs:
        f = (w / 2) / np.tan(np.radians(hfov) / 2)
        ok, rv, tv, inl = cv2.solvePnPRansac(X.astype(np.float32), u.astype(np.float32), K_of(f), None,
                                             reprojectionError=0.02 * w, iterationsCount=300,
                                             flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None:
            continue
        rv, tv, f2, e = refine(X, u, rv.ravel(), tv.ravel(), f, max_nfev=50)
        if f2 <= 0 or np.median((X @ cv2.Rodrigues(rv)[0].T + tv)[:, 2]) <= 0:
            continue                                  # degenerate / points behind the camera
        score = np.median(e)
        if best is None or score < best[0]:
            best = (score, rv, tv, f2)
    if best is None:
        raise RuntimeError(f"{name}: PnP failed")
    _, rv, tv, f = best
    # Clock offset: re-solve on a grid, keep the lowest median error.
    grid = []
    for dt in np.arange(-max_dt, max_dt + 1e-9, 0.01):
        Xd, ud = corr(dt)
        if len(Xd) < 50:
            continue
        rv_d, tv_d, f_d, e = refine(Xd, ud, rv, tv, f, max_nfev=40)
        grid.append((float(np.median(e)), float(dt), rv_d, tv_d, f_d))
    _, dt, rv, tv, f = min(grid, key=lambda g: g[0])
    X, u = corr(dt)
    rv, tv, f, e = refine(X, u, rv, tv, f, max_nfev=200)
    R = cv2.Rodrigues(rv)[0]
    cam = Camera(track.info.get("device", name), K_of(f), R, tv, w, h, time_offset=dt)
    hfov = np.degrees(2 * np.arctan(w / 2 / f))
    log(f"  {name}: {len(X)} points, reprojection median {np.median(e):.1f} px (p90 {np.percentile(e, 90):.1f}), "
        f"focal {f:.0f} px (horizontal FOV {hfov:.0f}°), clock offset {dt * 1000:+.0f} ms")
    log(f"  {name}: position {np.round(cam.center, 2)} m (height above floor {cam.center[1]:.2f} m)")
    return cam, {"reproj_median_px": float(np.median(e)), "focal_px": float(f), "hfov_deg": float(hfov),
                 "time_offset_s": dt, "points": int(len(X))}


def save_calibration(path: str, cams: list[Camera], hosts: list[str], meta: dict):
    data = {"meta": meta, "cameras": [
        {"name": c.name, "host": h, "K": c.K.tolist(), "R": c.R.tolist(), "t": c.t.tolist(),
         "width": c.width, "height": c.height, "time_offset": c.time_offset}
        for c, h in zip(cams, hosts)]}
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def load_calibration(path: str) -> tuple[list[Camera], list[str]]:
    d = json.load(open(path))
    cams = [Camera(c["name"], np.array(c["K"]), np.array(c["R"]), np.array(c["t"]),
                   c["width"], c["height"], time_offset=c.get("time_offset", 0.0))
            for c in d["cameras"]]
    return cams, [c["host"] for c in d["cameras"]]
