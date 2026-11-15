"""Record synchronised video from FBT Cam phones for offline work
(calibration, model comparison, replay tests — ARCHITECTURE.md §9.1 level D).

  .venv/bin/python -m tools.record 192.168.31.81 192.168.31.28 --seconds 15

Writes data/rec_<time>/camN.mp4, camN.csv (frame index, capture time on the
Mac clock, latency ms) and info.json (HELLO metadata per camera).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time

import cv2

from fbt.fbtcam import FBTCamSource


def record(hosts, seconds, out_dir=None, countdown=3):
    out_dir = out_dir or time.strftime("data/rec_%Y%m%d_%H%M%S")
    os.makedirs(out_dir, exist_ok=True)
    srcs = []
    for i, h in enumerate(hosts):
        host, _, port = h.partition(":")
        srcs.append(FBTCamSource(f"cam{i}", host, int(port or 5000)).start())
    time.sleep(1.5)  # clock sync settles
    for c in range(countdown, 0, -1):
        print(f"  start in {c}…", flush=True)
        time.sleep(1)
    print("  RECORDING", flush=True)

    def work(i, s):
        writer, rows, last = None, [], -1
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            f = s.latest(last)
            if f is None:
                time.sleep(0.001)
                continue
            last = f.index
            if writer is None:
                h, w = f.image.shape[:2]
                writer = cv2.VideoWriter(f"{out_dir}/cam{i}.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                         s.info.get("fps", 30), (w, h))
            writer.write(f.image)
            rows.append((len(rows), f"{f.timestamp:.6f}", f"{(f.received_at - f.timestamp) * 1000:.1f}"))
        if writer:
            writer.release()
        with open(f"{out_dir}/cam{i}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "t_capture", "latency_ms"])
            w.writerows(rows)

    th = [threading.Thread(target=work, args=(i, s)) for i, s in enumerate(srcs)]
    [t.start() for t in th]
    [t.join() for t in th]
    info = {s.camera_id: {**s.info, "host": s.host} for s in srcs}
    for s in srcs:
        s.stop()
    with open(f"{out_dir}/info.json", "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"  saved to {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hosts", nargs="+")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--out")
    a = ap.parse_args()
    record(a.hosts, a.seconds, a.out)


if __name__ == "__main__":
    main()
