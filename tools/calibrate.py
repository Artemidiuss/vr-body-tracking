"""Calibrate two FBT Cam phones (+ optionally the MacBook camera) from the user's body.

  # record 20 s and calibrate (move around: step, squat, raise arms, turn a bit)
  .venv/bin/python -m tools.calibrate 192.168.31.81 192.168.31.28 --height 1.80
  # + MacBook camera as a third view (run from Terminal.app: camera permission)
  .venv/bin/python -m tools.calibrate 192.168.31.81 192.168.31.28 mac --height 1.73 --voice

  # re-run on an existing recording
  .venv/bin/python -m tools.calibrate --rec data/rec_20261114_201500 --height 1.80

Saves data/calibration.json.
"""

from __future__ import annotations

import argparse
import json

from fbt.calibrate import (calibrate_extra, calibrate_pair, poses_from_recording, save_calibration,
                           triangulate_tracks)
from fbt.pose import make_backend
from tools.record import record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hosts", nargs="*")
    ap.add_argument("--rec", help="existing recording dir instead of recording now")
    ap.add_argument("--height", type=float, required=True, help="your height, metres")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--backend", default="rtmpose-m:gpu")
    ap.add_argument("--out", default="data/calibration.json")
    ap.add_argument("--voice", action="store_true", help="spoken start/stop cues")
    a = ap.parse_args()

    rec = a.rec
    if rec is None:
        if len(a.hosts) < 2:
            ap.error("give two phone addresses (+ optional 'mac'), or --rec")
        print("Move in front of the cameras: walk in place, squat, step left/right, raise arms.")
        rec = record(a.hosts, a.seconds, voice=a.voice)
    info = json.load(open(f"{rec}/info.json"))
    names = sorted(info, key=lambda c: int(c[3:]))
    hosts = [info[c]["host"] for c in names]

    print("running pose on the recording…")
    be = make_backend(a.backend)
    ta = poses_from_recording(rec, "cam0", be)
    tb = poses_from_recording(rec, "cam1", be)
    print(f"  poses: cam0 {len(ta.t)} frames, cam1 {len(tb.t)} frames")
    print("calibrating…")
    cams, meta = calibrate_pair(ta, tb, a.height)
    if len(names) > 2:
        # Extra cameras (MacBook): PnP against the phones' 3D skeleton.
        ts, skel = triangulate_tracks(cams, ta, tb)
        meta["extra"] = {}
        for i, c in enumerate(names[2:], start=2):
            tr = poses_from_recording(rec, c, be)
            print(f"  poses: {c} {len(tr.t)} frames")
            try:
                rng = tr.info.get("focal_range_px")
                if rng is None and hosts[i] == "mac":
                    from fbt.maccam import mac_focal_range
                    rng = mac_focal_range(tr.width)
                cam, m = calibrate_extra(tr, ts, skel, name=c, focal_range=tuple(rng) if rng else None)
            except RuntimeError as e:
                print(f"  {c} skipped: {e}")
                hosts[i] = None
                continue
            cams.append(cam)
            meta["extra"][c] = m
        hosts = [h for h in hosts if h is not None]
    meta["recording"] = rec
    meta["height_m"] = a.height
    save_calibration(a.out, cams, hosts, meta)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
