"""Live 3D skeleton from calibrated FBT Cam phones.

  .venv/bin/python -m tools.live3d                     # uses data/calibration.json
  .venv/bin/python -m tools.live3d --calib my.json

Window: both camera views with 2D skeletons, plus front / side / top views of
the triangulated 3D skeleton, and live end-to-end latency (sensor capture ->
3D ready on the Mac). Press q to quit.
"""

from __future__ import annotations

import argparse
import collections
import threading
import time

import cv2
import numpy as np

from fbt.calibrate import load_calibration
from fbt.fbtcam import FBTCamSource
from fbt.pose import Pose2D, make_backend
from fbt.skeleton import BONE_IDX
from fbt.triangulate import triangulate_with_swap_check
from tools.bench_pose import draw


class CamWorker:
    """Runs pose on the newest frame of one camera; keeps recent results."""

    def __init__(self, src, backend_spec: str, t_offset: float = 0.0, border: float = 0.0,
                 max_rate: float = 0.0):
        """t_offset: s added to frame timestamps (calibrated clock offset).
        border: keypoints within this fraction of the image edge (or outside
        it) get score 0 — for a camera that sees only part of the body, where
        the model still guesses the cut-off joints."""
        self.src = src
        self.t_offset, self.border = t_offset, border
        # max_rate (Hz, 0 = every frame): an auxiliary camera computes fewer poses
        # so the GPU keeps up with the phones on a hot, fanless Air.
        self.min_interval = 1.0 / max_rate if max_rate else 0.0
        self._last_t = -1e9
        self.backend = make_backend(backend_spec)
        self.poses = collections.deque(maxlen=8)   # (t_capture, Pose2D)
        self.last_img = None
        self.last_pose = None
        self.pose_ms = 0.0
        self.n_pose = 0           # poses computed (for rate stats)
        self.lock = threading.Lock()
        self.new = threading.Event()
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        last = -1
        while True:
            f = self.src.latest(last)
            if f is None:
                if self.src.finished.is_set():
                    return
                time.sleep(0.001)
                continue
            last = f.index
            if self.min_interval and f.timestamp - self._last_t < self.min_interval - 0.005:
                continue
            self._last_t = f.timestamp
            t0 = time.perf_counter()
            try:
                p = self.backend(f.image)
            except Exception as e:   # never let one bad frame kill this camera for the session
                self.errors = getattr(self, "errors", 0) + 1
                if self.errors <= 3:
                    print(f"\npose error on {self.src.camera_id}: {e!r} — skipping frame", flush=True)
                p = None
            if p is not None and self.border > 0:
                h, w = f.image.shape[:2]
                u, v = p.keypoints[:, 0], p.keypoints[:, 1]
                b = self.border
                out = (u < b * w) | (u > (1 - b) * w) | (v < b * h) | (v > (1 - b) * h)
                if out.any():
                    p = Pose2D(p.keypoints, np.where(out, 0.0, p.scores).astype(np.float32))
            self.pose_ms = 0.9 * self.pose_ms + 0.1 * (time.perf_counter() - t0) * 1000
            self.n_pose += 1
            with self.lock:
                self.last_img, self.last_pose = f.image, p
                if p is not None:
                    self.poses.append((f.timestamp + self.t_offset, p))
            self.new.set()

    def pose_at(self, t: float, max_dt: float = 0.05, max_gap: float = 0.1) -> Pose2D | None:
        """Pose interpolated to capture time t (or nearest within max_dt)."""
        with self.lock:
            ps = list(self.poses)
        if not ps:
            return None
        for (t0, p0), (t1, p1) in zip(ps, ps[1:]):
            if t0 <= t <= t1 and t1 - t0 < max_gap:
                w = (t - t0) / (t1 - t0) if t1 > t0 else 0
                return Pose2D(((1 - w) * p0.keypoints + w * p1.keypoints).astype(np.float32),
                              np.minimum(p0.scores, p1.scores))
        tn, pn = min(ps, key=lambda tp: abs(tp[0] - t))
        return pn if abs(tn - t) <= max_dt else None


def draw_views(P, valid, cams, size=(360, 640), scale=230):
    """Front (X-Y), side (Z-Y) and top (X-Z) orthographic views, metres."""
    w, h = size
    panels = []
    for name, (ax, ay, flip_y) in {"front": (0, 1, True), "side": (2, 1, True),
                                   "top": (0, 2, False)}.items():
        img = np.full((h, w, 3), 30, np.uint8)
        if flip_y:
            oy = h - 40
            cv2.line(img, (0, oy), (w, oy), (90, 90, 90), 1)  # floor
            to = lambda p: (int(w / 2 + p[ax] * scale), int(oy - p[ay] * scale))
        else:
            to = lambda p: (int(w / 2 + p[ax] * scale), int(h / 2 - p[ay] * scale))
            for c in cams:  # camera positions in top view
                cv2.circle(img, to(c.center), 6, (0, 200, 255), -1)
        cv2.putText(img, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if P is not None:
            for a, b in BONE_IDX:
                if valid[a] and valid[b]:
                    cv2.line(img, to(P[a]), to(P[b]), (0, 255, 120), 2)
            for k in np.where(valid)[0]:
                cv2.circle(img, to(P[k]), 3, (0, 80, 255), -1)
        panels.append(img)
    return np.hstack(panels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("-b", "--backend", default="rtmpose-m:gpu")
    a = ap.parse_args()
    cams, hosts = load_calibration(a.calib)
    workers = []
    for i, h in enumerate(hosts):
        host, _, port = h.partition(":")
        workers.append(CamWorker(FBTCamSource(f"cam{i}", host, int(port or 5000)).start(), a.backend))
    prev, lat3d = None, collections.deque(maxlen=60)
    fps_n, fps_t, fps = 0, time.perf_counter(), 0.0
    while True:
        workers[0].new.wait(0.5)
        workers[0].new.clear()
        with workers[0].lock:
            if not workers[0].poses:
                ref = None
            else:
                ref = workers[0].poses[-1]
        P = valid = None
        if ref is not None:
            t_ref, pose_a = ref
            pose_b = workers[1].pose_at(t_ref)
            if pose_b is not None:
                res = triangulate_with_swap_check(cams, [pose_a, pose_b], prev=prev)
                prev = res
                P, valid = res.points, res.valid
                lat3d.append((time.perf_counter() - t_ref) * 1000)
                fps_n += 1
        if time.perf_counter() - fps_t > 1:
            fps, fps_n, fps_t = fps_n / (time.perf_counter() - fps_t), 0, time.perf_counter()

        tiles = []
        for wk in workers:
            with wk.lock:
                img, pose = wk.last_img, wk.last_pose
            if img is None:
                tiles.append(np.zeros((640, 360, 3), np.uint8))
                continue
            img = img.copy()
            if pose is not None:
                draw(img, pose)
            tiles.append(cv2.resize(img, (360, 640)))
        view = np.hstack(tiles + [draw_views(P, valid, cams)])
        lat = f"{np.median(lat3d):.0f}" if lat3d else "-"
        cam_lat = " ".join(f"{np.median(w.src.latency_ms[-30:]):.0f}" if w.src.latency_ms else "-" for w in workers)
        cv2.putText(view, f"3D {fps:.0f} fps | capture->3D {lat} ms | camera->Mac {cam_lat} ms | "
                    f"pose {' '.join(f'{w.pose_ms:.0f}' for w in workers)} ms",
                    (10, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("live3d", view)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    for w in workers:
        w.src.stop()


if __name__ == "__main__":
    main()
