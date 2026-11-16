"""Offline: does the MacBook camera make the 3D skeleton better?

On a recording of the two phones + the Mac (tools.calibrate ... mac), the
same frames are triangulated with the phones only and with all three
cameras, then compared:
  * bone length spread (cm): a person's bones do not change length, so a
    smaller spread over time = more accurate 3D (upper / lower body)
  * head-root jitter (mm/frame^2): the point VRChat anchors all trackers to
  * how often each point gets 3 views, and how often the Mac view is dropped
Honest split: with --holdout the Mac camera is placed (PnP) on the first half
of the recording and everything is measured on the second half.

  .venv/bin/python -m tools.eval_three_cams data/rec_XXXX --holdout
"""

from __future__ import annotations

import argparse

import numpy as np

from fbt.calibrate import (PoseTrack, calibrate_extra, load_calibration, poses_from_recording,
                           triangulate_tracks)
from fbt.pose import Pose2D, make_backend
from fbt.skeleton import IDX
from fbt.triangulate import PRIOR_SEGMENTS, triangulate_with_swap_check, view_thresholds, view_weights

_LEG = {IDX[n] for n in ("left_knee", "right_knee", "left_ankle", "right_ankle")}
LOWER = [i for i, (a, b) in enumerate(PRIOR_SEGMENTS) if a in _LEG or b in _LEG]
UPPER = [i for i in range(len(PRIOR_SEGMENTS)) if i not in LOWER]


def interp_track(tr: PoseTrack, tq: np.ndarray, max_gap: float = 0.1):
    j = np.clip(np.searchsorted(tr.t, tq), 1, len(tr.t) - 1)
    t0, t1 = tr.t[j - 1], tr.t[j]
    ok = (tq >= t0) & (tq <= t1) & (t1 - t0 <= max_gap)
    w = ((tq - t0) / np.maximum(t1 - t0, 1e-9))[:, None, None]
    kp = (1 - w) * tr.kp[j - 1] + w * tr.kp[j]
    sc = np.minimum(tr.sc[j - 1], tr.sc[j])
    sc[~ok] = 0
    return kp, sc


def head_root(P, valid):
    if not (valid[IDX["left_ear"]] and valid[IDX["right_ear"]] and valid[IDX["neck"]] and valid[IDX["hip"]]):
        return None
    sp = P[IDX["neck"]] - P[IDX["hip"]]
    return (P[IDX["left_ear"]] + P[IDX["right_ear"]]) / 2 - 0.04 * sp / np.linalg.norm(sp)


def run(cams, poses_per_frame, thr, vw=1.0):
    prev, out = None, []
    for ps in poses_per_frame:
        r = triangulate_with_swap_check(cams, ps, prev=prev, max_err_px=thr, view_weight=vw)
        prev = r
        out.append(r)
    return out


def metrics(name, res, border_mask=None):
    L = []
    for r in res:
        row = []
        for a, b in PRIOR_SEGMENTS:
            row.append(np.linalg.norm(r.points[a] - r.points[b]) if r.valid[a] and r.valid[b] else np.nan)
        L.append(row)
    L = np.array(L)
    # spread = robust std (IQR / 1.349) per segment, cm
    q75, q25 = np.nanpercentile(L, 75, axis=0), np.nanpercentile(L, 25, axis=0)
    spread = (q75 - q25) / 1.349 * 100
    roots = [head_root(r.points, r.valid) for r in res]
    ok = np.array([x is not None for x in roots])
    R = np.array([x if x is not None else np.full(3, np.nan) for x in roots])
    d2 = np.linalg.norm(R[2:] - 2 * R[1:-1] + R[:-2], axis=1)
    jit = np.nanmedian(d2) * 1000
    v3 = np.mean([(r.valid & (r.n_views >= 3)).sum() for r in res])
    val = np.mean([r.valid.mean() for r in res])
    print(f"{name:<34} bones upper {np.nanmean(spread[UPPER]):5.2f} lower {np.nanmean(spread[LOWER]):5.2f} cm | "
          f"head root jitter {jit:5.2f} mm, valid {ok.mean():4.0%} | valid pts {val:4.0%}, 3-view pts {v3:4.1f}")
    return spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("--holdout", action="store_true", help="place the Mac camera on the 1st half, test on the 2nd")
    ap.add_argument("--backend", default="rtmpose-m:gpu")
    a = ap.parse_args()
    cams, hosts = load_calibration(a.calib)
    if len(cams) < 3:
        raise SystemExit("calibration has no third camera")
    be = make_backend(a.backend)
    tr = [poses_from_recording(a.rec, f"cam{i}", be) for i in range(3)]
    if a.holdout:
        ts, skel = triangulate_tracks(cams[:2], tr[0], tr[1])
        mid = ts[len(ts) // 2]
        first = tr[2].t < mid
        half = PoseTrack(tr[2].t[first], tr[2].kp[first], tr[2].sc[first], tr[2].width, tr[2].height, tr[2].info)
        from fbt.maccam import mac_focal_range
        cams[2], _ = calibrate_extra(half, ts, skel, name="mac (1st half)",
                                     focal_range=tuple(mac_focal_range(half.width)))
        t_eval = tr[0].t[tr[0].t >= mid]
    else:
        t_eval = tr[0].t
    # Frames at camera 0's capture times; the others interpolated (as in the hub).
    k1, s1 = interp_track(tr[1], t_eval)
    k2, s2 = interp_track(tr[2], t_eval - cams[2].time_offset)
    k0 = tr[0].kp[np.isin(tr[0].t, t_eval)]
    s0 = tr[0].sc[np.isin(tr[0].t, t_eval)]
    # Same border rule as the hub for the Mac view.
    w, h, b = tr[2].width, tr[2].height, 0.02
    out = (k2[..., 0] < b * w) | (k2[..., 0] > (1 - b) * w) | (k2[..., 1] < b * h) | (k2[..., 1] > (1 - b) * h)
    s2 = np.where(out, 0, s2)
    frames2 = [[Pose2D(k0[i], s0[i]), Pose2D(k1[i], s1[i])] for i in range(len(t_eval))]
    frames3 = [f + [Pose2D(k2[i], s2[i]) if s2[i].max() > 0 else None] for i, f in enumerate(frames2)]
    thr = view_thresholds(cams)
    print(f"{len(t_eval)} frames; Mac view has a pose in {np.mean(s2.max(1) > 0):.0%}; "
          f"thresholds px {np.round(thr).astype(int).tolist()}")
    r2 = run(cams[:2], frames2, 25.0)
    r3 = run(cams, frames3, thr, view_weights(cams))
    r3f = run(cams, frames3, thr, 1.0)
    metrics("phones only (now)", r2)
    metrics("phones + Mac (hub: equal weights)", r3)
    metrics("phones + Mac, Mac weight x f", r3f)
    # Mac calibration check: phones-only 3D projected into the Mac view.
    errs = []
    for r, kp, sc in zip(r2, k2, s2):
        m = r.valid & (sc >= 0.5)
        if m.any():
            uv, _ = cams[2].project(r.points[m])
            errs.extend(np.linalg.norm(uv - kp[m], axis=1))
    errs = np.array(errs)
    print(f"phones-only 3D -> Mac view: median {np.median(errs):.1f} px, p90 {np.percentile(errs, 90):.1f} px "
          f"(threshold {thr[2]:.0f} px)")
    for name, rr in (("equal weights", r3), ("weight x f", r3f)):
        used = np.mean([(r.n_views >= 3).sum() for r in rr])
        print(f"  {name}: points using all 3 views per frame {used:.1f}")


if __name__ == "__main__":
    main()
