"""Calibrate two FBT Cam phones from the user's body.

  # record 20 s and calibrate (move around: step, squat, raise arms, turn a bit)
  .venv/bin/python -m tools.calibrate 192.168.31.81 192.168.31.28 --height 1.80

  # re-run on an existing recording
  .venv/bin/python -m tools.calibrate --rec data/rec_20261114_201500 --height 1.80

Saves data/calibration.json.
"""

from __future__ import annotations

import argparse
import json

from fbt.calibrate import calibrate_pair, poses_from_recording, save_calibration
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
    a = ap.parse_args()

    rec = a.rec
    if rec is None:
        if len(a.hosts) != 2:
            ap.error("give two phone addresses, or --rec")
        print("Move in front of the cameras: walk in place, squat, step left/right, raise arms.")
        rec = record(a.hosts, a.seconds)
    info = json.load(open(f"{rec}/info.json"))
    hosts = [info[c]["host"] for c in ("cam0", "cam1")]

    print("running pose on the recording…")
    be = make_backend(a.backend)
    ta = poses_from_recording(rec, "cam0", be)
    tb = poses_from_recording(rec, "cam1", be)
    print(f"  poses: cam0 {len(ta.t)} frames, cam1 {len(tb.t)} frames")
    print("calibrating…")
    cams, meta = calibrate_pair(ta, tb, a.height)
    meta["recording"] = rec
    meta["height_m"] = a.height
    save_calibration(a.out, cams, hosts, meta)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
