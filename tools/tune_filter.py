"""Tune the One-Euro smoothing on a real recording: for each parameter set,
measure jitter (while still) and lag (while moving) of the feet/knees,
relative to the unfiltered triangulated skeleton.

  .venv/bin/python -m tools.tune_filter            # recording from data/calibration.json

jitter = mean frame-to-frame acceleration magnitude (mm/frame²) — what you
         see as shaking
lag    = time shift (ms) that best aligns the filtered trajectory with the
         raw one during movement — what you feel as delay
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from fbt.calibrate import load_calibration, pair_tracks, poses_from_recording
from fbt.filters import SkeletonStabilizer
from fbt.pose import Pose2D, make_backend
from fbt.skeleton import IDX
from fbt.triangulate import triangulate_with_swap_check

PTS = [IDX[n] for n in ("left_ankle", "right_ankle", "left_knee", "right_knee", "hip")]


def lag_ms(raw, filt, ts, max_shift=15, band=(60, 100)):
    """Shift (frames) that minimises |filt(t) - raw(t - s)|, on frames whose
    speed lies in the given percentile band (default: the faster 40%)."""
    speed = np.linalg.norm(np.diff(raw, axis=0), axis=-1).mean(-1)
    lo, hi = np.percentile(speed, band[0]), np.percentile(speed, band[1])
    moving = np.where((speed >= lo) & (speed <= hi))[0] + 1
    moving = moving[(moving > max_shift) & (moving < len(raw) - 1)]
    best, best_e = 0, 1e9
    for s in range(0, max_shift + 1):
        e = np.linalg.norm(filt[moving] - raw[moving - s], axis=-1).mean()
        if e < best_e:
            best, best_e = s, e
    dt = np.median(np.diff(ts))
    return best * dt * 1000


def jitter(x):
    return np.linalg.norm(np.diff(x, 2, axis=0), axis=-1).mean() * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("--rec")
    a = ap.parse_args()
    cams, _ = load_calibration(a.calib)
    rec = a.rec or json.load(open(a.calib))["meta"]["recording"]
    be = make_backend("rtmpose-m:gpu")
    ta, tb = poses_from_recording(rec, "cam0", be), poses_from_recording(rec, "cam1", be)
    pa, pb, sa, sb, ts = pair_tracks(ta, tb)
    raw, valids, prev = [], [], None
    for i in range(len(pa)):
        r = triangulate_with_swap_check(cams, [Pose2D(pa[i], sa[i]), Pose2D(pb[i], sb[i])], prev=prev)
        prev = r
        raw.append(r.points.copy()); valids.append(r.valid.copy())
    raw, valids = np.array(raw), np.array(valids)

    # Reference: skeleton fit only (bone lengths), no temporal filtering.
    from fbt.filters import SkeletonFit
    fit = SkeletonFit(60)
    ref = np.array([fit(raw[i], valids[i]) for i in range(len(raw))])[:, PTS]
    print(f"{'min_cutoff':>10} {'beta':>6} {'predict ms':>10} | {'jitter mm':>9} {'lag ms':>7}")
    print(f"{'raw':>10} {'':>6} {'':>10} | {jitter(ref[60:]):9.2f} {0:7.0f}")
    def spikes(x):
        return (np.linalg.norm(np.diff(x, axis=0), axis=-1) > 0.08).mean() * 100

    print(f"{'':>34} slow-lag ms  spikes>8cm % | gate")
    for mc, beta, pred, gate in [(0.7, 0.5, 20, False), (1.5, 0.8, 0, True), (2.5, 0.8, 0, True),
                                 (3.5, 0.8, 0, True), (2.5, 0.4, 0, True), (5.0, 0.5, 0, True)]:
        st = SkeletonStabilizer(min_cutoff=mc, beta=beta, predict_s=pred / 1000, learn_frames=60, gate=gate)
        out = np.array([st(raw[i], valids[i], ts[i]) for i in range(len(raw))])[:, PTS]
        print(f"{mc:10.1f} {beta:6.1f} {pred:10d} | {jitter(out[60:]):9.2f} {lag_ms(ref[60:], out[60:], ts[60:]):7.0f}"
              f"   {lag_ms(ref[60:], out[60:], ts[60:], band=(20, 60)):9.0f}   {spikes(out[60:]):9.2f}   {gate}")


if __name__ == "__main__":
    main()
