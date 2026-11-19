"""Multi-view triangulation of Halpe26 skeletons.

Per keypoint: confidence-weighted DLT over all cameras that see it, then
reprojection error check. Per frame: left/right label swaps in one camera
are detected by trying the swapped labelling and keeping the one with lower
reprojection error (and continuity with the previous frame).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .pose import Pose2D
from .skeleton import HALPE26, IDX, NUM_KEYPOINTS

# Index permutation that swaps every left_* with right_*.
LR_SWAP = np.array([
    IDX[n.replace("left_", "right_")] if n.startswith("left_")
    else IDX[n.replace("right_", "left_")] if n.startswith("right_")
    else i
    for i, n in enumerate(HALPE26)
])


@dataclass
class Pose3D:
    points: np.ndarray       # (26,3) metres, world frame
    valid: np.ndarray        # (26,) bool
    reproj_err: np.ndarray   # (26,) mean reprojection error, px
    n_views: np.ndarray      # (26,) cameras used per point
    swapped: list[bool]      # per camera: were its L/R labels swapped


def triangulate_point(Ps, uvs, weights) -> np.ndarray:
    """Weighted DLT. Ps: list of 3x4, uvs: (V,2), weights: (V,)."""
    A = []
    for P, (u, v), w in zip(Ps, uvs, weights):
        A.append(w * (u * P[2] - P[0]))
        A.append(w * (v * P[2] - P[1]))
    _, _, vt = np.linalg.svd(np.asarray(A))
    X = vt[-1]
    return X[:3] / X[3]


def _reproj_err(cams, X, uvs) -> np.ndarray:
    return np.array([np.linalg.norm(c.project(X)[0][0] - uv) for c, uv in zip(cams, uvs)])


def triangulate_pose(cams: list[Camera], poses: list[Pose2D | None],
                     min_score: float = 0.3, max_err_px=25.0, view_weight=1.0) -> Pose3D:
    """Triangulate one frame. Points seen by < 2 cameras, or whose
    reprojection error stays above max_err_px after dropping the worst view,
    are marked invalid (the solver later fills them from the skeleton model).
    max_err_px: one value, or one per camera (a close camera sees the same
    metric error as more pixels); views are compared by error / threshold.
    view_weight: per-camera multiplier of the DLT weight. A DLT row's error is
    focal x metric error, so a long-focal (or close) camera outweighs the
    others for no reason; f_ref / f_camera puts all views on equal terms.
    """
    thr = np.broadcast_to(np.asarray(max_err_px, float), (len(cams),))
    vw = np.broadcast_to(np.asarray(view_weight, float), (len(cams),))
    pts = np.zeros((NUM_KEYPOINTS, 3))
    valid = np.zeros(NUM_KEYPOINTS, bool)
    err = np.full(NUM_KEYPOINTS, np.inf)
    nv = np.zeros(NUM_KEYPOINTS, int)
    for k in range(NUM_KEYPOINTS):
        views = [(c, p.keypoints[k], p.scores[k] * vw[i], thr[i]) for i, (c, p) in enumerate(zip(cams, poses))
                 if p is not None and p.scores[k] >= min_score]
        while len(views) >= 2:
            X = triangulate_point([v[0].P for v in views],
                                  np.array([v[1] for v in views]),
                                  np.array([v[2] for v in views]))
            e = _reproj_err([v[0] for v in views], X, [v[1] for v in views])
            r = e / np.array([v[3] for v in views])
            if r.max() <= 1.0 or len(views) == 2:
                ok = r.max() <= 1.0
                pts[k], valid[k], err[k], nv[k] = X, ok, e.mean(), len(views)
                break
            views.pop(int(np.argmax(r)))  # 3+ views: drop the outlier view
    return Pose3D(pts, valid, err, nv, [False] * len(cams))


EXTRA_MAX_WEIGHT = 0.27


def view_weights(cams: list[Camera], n_ref: int = 2) -> np.ndarray:
    """DLT weight multipliers: 1 for the first n_ref cameras (phones), f_ref / f
    for the others, so a long-focal camera does not outweigh the phones.
    Offline (held-out half of a recording): upper-body bone spread 1.68 -> 1.36 cm.

    And at most EXTRA_MAX_WEIGHT: an extra camera runs its pose at a lower rate and
    arrives later, so it joins only part of the 3D steps (~58% for a phone at 15 fps)
    and every join or leave moves the point. With a third phone at its focal weight
    (0.95) knees and hips shook 3x (standing 12.9 / 10.6 mm per frame vs 4.0 / 3.5
    from the phones alone); at 0.27, the MacBook's own weight, 4.3 / 3.5."""
    w = np.ones(len(cams))
    if len(cams) > n_ref:
        fref = np.mean([c.K[0, 0] for c in cams[:n_ref]])
        w[n_ref:] = [min(fref / c.K[0, 0], EXTRA_MAX_WEIGHT) for c in cams[n_ref:]]
    return w


def view_thresholds(cams: list[Camera], base_px: float = 25.0, n_ref: int = 2,
                    subject=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Per-camera reprojection thresholds: base_px for the first n_ref
    cameras (the phones, unchanged), scaled for the others by their pixels
    per metre at the user (focal / distance) relative to the phones'."""
    ppm = np.array([c.K[0, 0] / np.linalg.norm(c.center - np.asarray(subject)) for c in cams])
    thr = np.full(len(cams), base_px)
    if len(cams) > n_ref:
        thr[n_ref:] = base_px * ppm[n_ref:] / ppm[:n_ref].mean()
    return thr


def _swap(p: Pose2D) -> Pose2D:
    return Pose2D(p.keypoints[LR_SWAP], p.scores[LR_SWAP])


# Segments whose length is fixed for a person. Used to reject mis-paired
# skeletons: pairing camera A's left leg with camera B's right leg can be
# geometrically consistent (legs at the same height, near-horizontal
# epipolar lines) but collapses hip/shoulder width and distorts limbs.
PRIOR_SEGMENTS = [(IDX[a], IDX[b]) for a, b in [
    ("left_hip", "right_hip"), ("left_shoulder", "right_shoulder"),
    ("left_hip", "left_knee"), ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
]]
SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (8, 9)]  # indices into PRIOR_SEGMENTS


def segment_lengths(p: Pose3D) -> np.ndarray:
    """(len(PRIOR_SEGMENTS),) lengths, NaN where an end point is invalid."""
    out = np.full(len(PRIOR_SEGMENTS), np.nan)
    for i, (a, b) in enumerate(PRIOR_SEGMENTS):
        if p.valid[a] and p.valid[b]:
            out[i] = np.linalg.norm(p.points[a] - p.points[b])
    return out


def body_cost(p: Pose3D, ref_lengths: np.ndarray | None) -> float:
    """Implausibility of a skeleton, in cm. With calibrated reference
    lengths: mean deviation from them. Without: left/right asymmetry plus
    a penalty for hip/shoulder width collapsing below typical values."""
    L = segment_lengths(p)
    if ref_lengths is not None:
        d = np.abs(L - ref_lengths)
        return float(np.nanmean(d) * 100) if np.isfinite(d).any() else 0.0
    cost = [abs(L[a] - L[b]) for a, b in SYMMETRIC_PAIRS
            if np.isfinite(L[a]) and np.isfinite(L[b])]
    for i, typical in ((0, 0.18), (1, 0.36)):  # hip, shoulder width
        if np.isfinite(L[i]):
            cost.append(max(0.0, typical * 0.7 - L[i]) * 3)
    return float(np.mean(cost) * 100) if cost else 0.0


def _left_side_score(p: Pose3D) -> float:
    """> 0 when labelled left points are on the body's left, judged by the
    face direction (nose ahead of the neck)."""
    n, nk = IDX["nose"], IDX["neck"]
    if not (p.valid[n] and p.valid[nk]):
        return 0.0
    fwd = p.points[n] - p.points[nk]
    fwd[1] = 0
    left_dir = np.cross([0, 1.0, 0], fwd)
    s = 0.0
    for side in ("hip", "shoulder", "knee"):
        a, b = IDX[f"left_{side}"], IDX[f"right_{side}"]
        if p.valid[a] and p.valid[b]:
            s += float(np.dot(p.points[a] - p.points[b], left_dir))
    return s


def triangulate_with_swap_check(cams, poses, prev: Pose3D | None = None,
                                ref_lengths: np.ndarray | None = None,
                                w_body: float = 1.0, w_motion: float = 0.3,
                                hysteresis: float = 3.0, **kw) -> Pose3D:
    """Like triangulate_pose, but also tries flipping L/R labels per camera.

    Candidate cost = reprojection error (px) + w_body * body implausibility
    (cm) + w_motion * jump from the previous frame (cm).
    "Swap camera A" vs "swap camera B" are mirror labellings of the same
    geometry; continuity decides, else the face direction.
    """
    n = len(cams)
    candidates = [[False] * n]
    for j in range(n):
        if poses[j] is not None:
            c = [False] * n
            c[j] = True
            candidates.append(c)

    def cost(r: Pose3D) -> float:
        c = float(np.where(r.valid, r.reproj_err, 100.0).mean())
        c += w_body * body_cost(r, ref_lengths)
        if prev is not None:
            m = r.valid & prev.valid
            if m.any():
                c += w_motion * 100 * float(np.linalg.norm(r.points[m] - prev.points[m], axis=1).mean())
            # Hysteresis: only change the relabelling decision for a clear win
            # (measured: one-frame swap flip-flops caused ~12% of 3D spikes).
            if list(r.swapped) != list(prev.swapped):
                c += hysteresis
        return c

    results = []
    for cand in candidates:
        ps = [(_swap(p) if s and p is not None else p) for p, s in zip(poses, cand)]
        r = triangulate_pose(cams, ps, **kw)
        r.swapped = cand
        results.append(r)

    costs = [cost(r) for r in results]
    best = min(costs)
    good = [r for r, c in zip(results, costs) if c <= best + 0.5]
    if len(good) == 1 or prev is not None:
        return results[int(np.argmin(costs))]
    return max(good, key=_left_side_score)
