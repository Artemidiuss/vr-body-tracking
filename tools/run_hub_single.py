"""Single-camera pipeline (fallback when only one phone is available):
phone -> MediaPipe single-image 3D -> world axes -> stabilise -> VRChat OSC.

  .venv/bin/python -m tools.run_hub_single --quest 192.168.31.150 --cam 0

Depth is guessed by the model (~8.5 cm leg error vs two cameras on our
recordings), but VRChat anchors the trackers to the head, so only the body
shape relative to the head matters. The camera's orientation comes from the
two-camera calibration (data/calibration.json, camera index --cam); feet are
kept on the floor. Head: position every frame (shoulder midpoint), yaw only
once when a T-pose is held; left/right mix-ups fixed by frame continuity.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from fbt.calibrate import load_calibration
from fbt.fbtcam import FBTCamSource
from fbt.filters import SkeletonStabilizer
from fbt.mono3d import MediaPipeMono3D
from fbt.osc_out import VRChatSender
from fbt.skeleton import BONE_IDX, IDX
from fbt.trackers import head_tracker, is_tpose, solve
from fbt.triangulate import LR_SWAP
from tools.live3d import draw_views

FEET = [IDX[n] for n in ("left_heel", "right_heel", "left_big_toe", "right_big_toe")]
SEG_RATIO = 0.766  # (shin + thigh + hip->shoulder) / height, on our synthetic body


def segsum(P):
    g = lambda n: P[IDX[n]]
    mid = lambda a, b: (g(a) + g(b)) / 2
    shin = (np.linalg.norm(g("left_knee") - g("left_ankle")) + np.linalg.norm(g("right_knee") - g("right_ankle"))) / 2
    thigh = (np.linalg.norm(g("left_hip") - g("left_knee")) + np.linalg.norm(g("right_hip") - g("right_knee"))) / 2
    torso = np.linalg.norm(mid("left_shoulder", "right_shoulder") - mid("left_hip", "right_hip"))
    return shin + thigh + torso


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quest", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("--cam", type=int, default=0, help="which calibrated camera to use")
    ap.add_argument("--model", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--trackers", default="hip,chest,left_foot,right_foot,left_knee,right_knee")
    ap.add_argument("--height", type=float, default=1.73, help="your height, metres")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--seconds", type=float, default=0)
    a = ap.parse_args()

    cams, hosts = load_calibration(a.calib)
    cam = cams[a.cam]
    host, _, port = hosts[a.cam].partition(":")
    src = FBTCamSource("cam", host, int(port or 5000)).start()
    model = MediaPipeMono3D(a.model)
    sender = VRChatSender(a.quest, a.port)
    stab = SkeletonStabilizer(min_cutoff=1.0, beta=0.4)  # single cam is noisier; measured best balance (tools.tune_single)
    enabled = tuple(a.trackers.split(","))
    floor = None
    scale, heights = None, []   # MediaPipe skeletons are "average person" sized
    prev_P, swaps = None, 0
    tpose_since, last_yaw = None, -1e9
    last, n, t_fps, fps = -1, 0, time.perf_counter(), 0.0
    t_stop = time.perf_counter() + a.seconds if a.seconds else float("inf")
    print(f"single camera {cam.name} ({host}) -> {a.quest}:{a.port}; stay in view ~3 s (sitting is fine for scale)", flush=True)
    while time.perf_counter() < t_stop:
        f = src.latest(last)
        if f is None:
            time.sleep(0.002)
            continue
        last = f.index
        r = model(f.image)
        S = None
        if r is not None:
            Pc, sc = r                       # camera axes, metres, hip-centred
            P = Pc @ cam.R                   # camera -> world rotation (R is world->camera)
            valid = sc > 0.3
            if scale is None:
                # Scale to the user's height from bone lengths (shin + thigh +
                # torso), which do not depend on posture — the earlier
                # head-height method blew the skeleton up when the user sat.
                heights.append(segsum(P))
                if len(heights) >= 60:
                    scale = float(np.clip(SEG_RATIO * a.height / np.median(heights), 0.6, 1.6))
                    print(f"scale to your height: x{scale:.2f}", flush=True)
                continue
            P = P * scale
            # Single-image models often swap left/right when the user is seen
            # from behind. Keep whichever labelling is closer to the last frame.
            if prev_P is not None:
                legs = [IDX[n] for n in ("left_knee", "right_knee", "left_ankle", "right_ankle",
                                         "left_hip", "right_hip", "left_shoulder", "right_shoulder")]
                Ps = P[LR_SWAP]
                if (np.linalg.norm(Ps[legs] - prev_P[legs], axis=1).mean() <
                        np.linalg.norm(P[legs] - prev_P[legs], axis=1).mean() - 0.02):
                    P, valid = Ps, valid[LR_SWAP]
                    swaps += 1
            prev_P = P.copy()
            # Keep the feet on the floor (smoothed), since depth/height of the
            # root is unknown from one image.
            low = np.min(P[FEET, 1])
            floor = low if floor is None else 0.9 * floor + 0.1 * low
            P[:, 1] -= floor
            S = stab(P, valid, f.timestamp)
            if stab.fit.ready:
                trs = solve(S, enabled=enabled)
                # Head POSITION every frame from the shoulder midpoint (symmetric:
                # immune to left/right mix-ups). Head ROTATION is NOT streamed:
                # VRChat would turn the whole tracker space after our camera-based
                # yaw (which flips 180 deg on a left/right mix-up), dragging the
                # feet cubes away. Yaw is sent ONCE when a T-pose is held 1 s.
                h = head_tracker(S, source="torso")
                now = time.perf_counter()
                tpose_since = (tpose_since or now) if is_tpose(S) else None
                send_rot = tpose_since is not None and now - tpose_since > 1.0 and now - last_yaw > 5.0
                if send_rot:
                    last_yaw = now
                    print("T-pose held: yaw aligned once", flush=True)
                trs.append(type(h)("head", h.position, h.rotation if send_rot else None))
                sender.send(trs)
            n += 1
        if time.perf_counter() - t_fps > 1:
            fps, n, t_fps = n / (time.perf_counter() - t_fps), 0, time.perf_counter()
            print(f"{fps:.0f} fps, L/R fixes {swaps}, camera->Mac {np.median(src.latency_ms[-30:]) if src.latency_ms else 0:.0f} ms, "
                  f"{'LIVE' if stab.fit.ready else 'learning bones'}", flush=True)
        if a.no_view:
            continue
        img = cv2.resize(f.image, (360, 640))
        view = np.hstack([img, draw_views(S, np.ones(26, bool) if S is not None else None, [cam])])
        cv2.putText(view, f"single cam | {fps:.0f} fps | {'LIVE' if stab.fit.ready else 'learning bones'}",
                    (10, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("fbt hub (single camera)", view)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    src.stop()


if __name__ == "__main__":
    main()
