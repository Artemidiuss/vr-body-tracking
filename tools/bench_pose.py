"""Benchmark pose backends on a video (or a still image looped).

Measures per-frame inference time and how often each foot keypoint is
detected confidently, and can write an overlay video for visual comparison.

  .venv/bin/python -m tools.bench_pose data/clip.mp4 -b rtmpose-m mediapipe-full
  .venv/bin/python -m tools.bench_pose image.jpg --frames 200 -b rtmpose-m:gpu rtmpose-m:cpu
"""

from __future__ import annotations

import argparse
import statistics
import time

import cv2
import numpy as np

from fbt.pose import make_backend
from fbt.skeleton import BONE_IDX, IDX

FOOT = ["left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe"]


def frames_from(path: str, n: int, width: int | None):
    img = cv2.imread(path)
    if img is not None:  # still image: repeat
        for _ in range(n):
            yield img
        return
    cap = cv2.VideoCapture(path)
    i = 0
    while i < n:
        ok, f = cap.read()
        if not ok:
            break
        if width and f.shape[1] > width:
            f = cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1])))
        yield f
        i += 1


def draw(img, pose, thr=0.3):
    for a, b in BONE_IDX:
        if pose.scores[a] > thr and pose.scores[b] > thr:
            cv2.line(img, tuple(map(int, pose.keypoints[a])),
                     tuple(map(int, pose.keypoints[b])), (0, 255, 0), 2)
    for (x, y), s in zip(pose.keypoints, pose.scores):
        if s > thr:
            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)


def run(backend_spec, path, n, width, out_path, thr):
    backend = make_backend(backend_spec)
    times, foot_hits, frames_with_pose, total = [], {k: 0 for k in FOOT}, 0, 0
    writer = None
    for i, img in enumerate(frames_from(path, n, width)):
        t0 = time.perf_counter()
        pose = backend(img)
        dt = (time.perf_counter() - t0) * 1000
        if i >= 5:  # skip warm-up
            times.append(dt)
        total += 1
        if pose is not None:
            frames_with_pose += 1
            for k in FOOT:
                foot_hits[k] += pose.scores[IDX[k]] > thr
        if out_path:
            vis = img.copy()
            if pose is not None:
                draw(vis, pose, thr)
            cv2.putText(vis, f"{backend.name} {dt:.1f}ms", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            writer.write(vis)
    if writer:
        writer.release()
    if not times:
        print(f"{backend.name}: not enough frames")
        return
    p = np.percentile(times, [50, 95])
    print(f"\n== {backend.name}")
    print(f"  frames: {total}, pose found: {frames_with_pose / total:.0%}")
    print(f"  latency ms: mean {statistics.mean(times):.1f}  p50 {p[0]:.1f}  p95 {p[1]:.1f}"
          f"  -> ~{1000 / statistics.mean(times):.0f} fps per camera")
    if frames_with_pose:
        feet = "  ".join(f"{k}:{foot_hits[k] / frames_with_pose:.0%}" for k in FOOT)
        print(f"  foot keypoints > {thr}: {feet}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-b", "--backends", nargs="+", default=["rtmpose-m", "mediapipe-full"])
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--width", type=int, default=1280, help="downscale wider frames")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--overlay", action="store_true", help="write <backend>.mp4 overlays to data/")
    a = ap.parse_args()
    for b in a.backends:
        out = f"data/overlay_{b.replace(':', '_')}.mp4" if a.overlay else None
        run(b, a.path, a.frames, a.width, out, a.thr)


if __name__ == "__main__":
    main()
