"""Live preview of one or more phone cameras with the pose overlay, FPS and
inference time per camera. First check that the phones are set up right.

  .venv/bin/python -m tools.live_view http://192.168.1.50:8080/video http://192.168.1.51:8080/video
  .venv/bin/python -m tools.live_view 0                # Mac webcam, for a quick test
  .venv/bin/python -m tools.live_view URL -b mediapipe-full

Press q to quit.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from fbt.pose import make_backend
from fbt.sources import OpenCVSource
from tools.bench_pose import draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("uris", nargs="+")
    ap.add_argument("-b", "--backend", default="rtmpose-m:gpu")
    ap.add_argument("--height", type=int, default=640, help="preview tile height")
    a = ap.parse_args()

    srcs = [OpenCVSource(f"cam{i}", int(u) if u.isdigit() else u).start()
            for i, u in enumerate(a.uris)]
    backends = [make_backend(a.backend) for _ in srcs]
    last = [-1] * len(srcs)
    tiles = [np.zeros((a.height, a.height * 9 // 16, 3), np.uint8) for _ in srcs]
    stats = [{"n": 0, "t0": time.perf_counter(), "fps": 0.0, "ms": 0.0} for _ in srcs]

    while True:
        for i, (src, be) in enumerate(zip(srcs, backends)):
            f = src.latest(last[i])
            if f is None:
                continue
            last[i] = f.index
            t = time.perf_counter()
            pose = be(f.image)
            st = stats[i]
            st["ms"] = 0.9 * st["ms"] + 0.1 * (time.perf_counter() - t) * 1000
            st["n"] += 1
            if time.perf_counter() - st["t0"] > 1:
                st["fps"] = st["n"] / (time.perf_counter() - st["t0"])
                st["n"], st["t0"] = 0, time.perf_counter()
            img = f.image.copy()
            if pose is not None:
                draw(img, pose)
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * a.height / h), a.height))
            cv2.putText(img, f"{src.camera_id} {w}x{h} {st['fps']:.0f}fps pose {st['ms']:.0f}ms "
                        f"drop {src.dropped}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            tiles[i] = img
        cv2.imshow("live", np.hstack([cv2.resize(t, (tiles[0].shape[1], a.height)) if t.shape[0] != a.height else t
                                      for t in tiles]))
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    for s in srcs:
        s.stop()


if __name__ == "__main__":
    main()
