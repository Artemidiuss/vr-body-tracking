"""The full pipeline: phones -> 2D pose -> 3D -> stabilise -> VRChat trackers -> OSC.

  # test locally with the fake VRChat (run tools.fake_vrchat in another terminal)
  .venv/bin/python -m tools.run_hub --quest 127.0.0.1

  # real headset (VRChat: Action menu -> Options -> OSC -> Enabled)
  .venv/bin/python -m tools.run_hub --quest 192.168.31.XX
  then in VRChat: Quick Menu -> gear -> Tracking & IK -> "Auto-center OSC Trackers"
  (press again if the body faces backwards), then Calibrate FBT.

Options: --no-view (no window), --predict 0.02 (seconds of extrapolation),
--head once|slow|stream|none (how VRChat's head alignment is fed), --smooth 0.7,
--trackers hip,chest,left_foot,right_foot,left_knee,right_knee[,left_elbow,right_elbow]
Stand still for ~3 s at start: bone lengths are learned first.
"""

from __future__ import annotations

import argparse
import os
import collections
import time

import cv2
import numpy as np

from fbt.calibrate import load_calibration
from fbt.fbtcam import FBTCamSource
from fbt.filters import SkeletonStabilizer
from fbt.osc_out import VRChatSender
from fbt.skeleton import IDX
from fbt.trackers import head_tracker, is_hands_up, is_tpose, solve
from fbt.triangulate import triangulate_with_swap_check
from tools.bench_pose import draw
from tools.live3d import CamWorker, draw_views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quest", default="127.0.0.1", help="headset IP (or 127.0.0.1 for fake VRChat)")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("-b", "--backend", default="rtmpose-m:gpu")
    ap.add_argument("--trackers", default="hip,chest,left_foot,right_foot,left_knee,right_knee")
    ap.add_argument("--predict", type=float, default=0.0)
    ap.add_argument("--head", default="tpose", choices=["pos", "tpose", "slimevr", "once", "slow", "stream", "none"],
                    help="VRChat aligns the whole tracker space to our head EVERY FRAME. "
                         "pos (default) = like Fever/PinoFBT and ju1ce: head POSITION every frame from a "
                         "steady point above the shoulders, NEVER a rotation (VRChat turns the whole "
                         "tracker space after any head rotation we send); "
                         "tpose = like ju1ce: head POSITION every frame from a steady point "
                         "above the shoulders, NO head rotation except ONE yaw message when you hold a "
                         "T-pose ~1 s (i.e. while doing Calibrate FBT); "
                         "slimevr = like SlimeVR: head POSITION every frame from a steady "
                         "neck-based point, head ROTATION (yaw) as ONE message at start / on [a]; "
                         "none = never; "
                         "once = send an averaged head for --align-secs then stop; "
                         "slow = keep sending a heavily smoothed head; stream = raw")
    ap.add_argument("--align-secs", type=float, default=3.0)
    ap.add_argument("--smooth", type=float, default=2.5, help="One-Euro min cutoff Hz (lower = smoother)")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run until q)")
    ap.add_argument("--log", default="", help="write every sent frame as JSON lines (for analysis)")
    a = ap.parse_args()
    t_stop = time.perf_counter() + a.seconds if a.seconds else float("inf")
    logf = open(a.log, "w") if a.log else None

    cams, hosts = load_calibration(a.calib)
    workers = []
    for i, h in enumerate(hosts):
        host, _, port = h.partition(":")
        workers.append(CamWorker(FBTCamSource(f"cam{i}", host, int(port or 5000)).start(), a.backend))
    sender = VRChatSender(a.quest, a.port)
    stab = SkeletonStabilizer(min_cutoff=a.smooth, beta=0.4, predict_s=a.predict)
    align_until = None      # head is sent until this time (mode "once")
    headset_down = False
    tpose_since, last_yaw = None, -1e9
    force_yaw = False
    tpose_since, last_yaw = None, -1e9
    head_avg = None
    enabled = tuple(a.trackers.split(","))
    prev, lat = None, collections.deque(maxlen=90)
    n, t_fps, fps, last_ref = 0, time.perf_counter(), 0.0, None
    print(f"sending {len(enabled)} trackers to {a.quest}:{a.port} — stand still ~3 s to learn bone lengths")
    learned_msg = False
    import collections as _c
    skips = _c.Counter()
    last_S, last_S_t = None, 0.0
    while time.perf_counter() < t_stop:
        # Wake on a new pose from EITHER camera. Triangulate at the capture time
        # of the camera that is behind (its newest pose), interpolating the
        # camera that is ahead — so a synchronised pair always exists and no
        # extra waiting is added. (Triggering only on camera 0 skipped frames
        # whenever camera 1's pose finished later.)
        if not workers[0].new.wait(0.01):
            workers[1].new.wait(0.01)
        for w in workers:
            w.new.clear()
        latest = []
        for w in workers:
            with w.lock:
                latest.append(w.poses[-1] if w.poses else None)
        S = None
        t_ref = None
        if latest[0] is not None and latest[1] is not None:
            behind = 0 if latest[0][0] <= latest[1][0] else 1
            t_ref = latest[behind][0]
        if t_ref is not None and (last_ref is None or t_ref > last_ref):
            last_ref = t_ref
            if behind == 0:
                pose_a, pose_b = latest[0][1], workers[1].pose_at(t_ref)
            else:
                pose_a, pose_b = workers[0].pose_at(t_ref), latest[1][1]
            if pose_a is None or pose_b is None:
                skips["no synced pair"] += 1
                pose_b = None
            if pose_b is not None:
                res = triangulate_with_swap_check(cams, [pose_a, pose_b], prev=prev)
                core = [IDX[k] for k in ("hip", "left_hip", "right_hip", "neck")]
                if not res.valid[core].all():
                    skips["torso not triangulated"] += 1
                if res.valid[core].all():
                    prev = res
                    S = stab(res.points, res.valid, t_ref)
                    if stab.fit.ready:
                        trs = solve(S, enabled=enabled)
                        h = head_tracker(S)
                        now = time.perf_counter()
                        if a.head == "pos":
                            hp = head_tracker(S, source="torso")
                            trs.append(type(hp)("head", hp.position, None))
                        if a.head == "tpose":
                            # VRChat turns the whole OSC space to match any head rotation we
                            # stream; camera head yaw is too imprecise for that. So: position
                            # every frame (steady torso-based point), rotation only once, when
                            # the user holds a T-pose (body and head face the same way).
                            hp = head_tracker(S, source="torso")
                            # Trigger: hands above head (or a T-pose) held 1 s, or the
                            # file data/align_now (lets the operator trigger it remotely).
                            tp = is_hands_up(S) or is_tpose(S)
                            if os.path.exists("data/align_now"):
                                os.remove("data/align_now")
                                force_yaw = True
                            tpose_since = (tpose_since or now) if tp else None
                            send_rot = force_yaw or (tpose_since is not None and now - tpose_since > 1.0
                                                     and now - last_yaw > 5.0)
                            force_yaw = False
                            if send_rot:
                                last_yaw = now
                                print("yaw aligned (one message sent)", flush=True)
                            trs.append(type(hp)("head", hp.position, hp.rotation if send_rot else None))
                        if a.head == "slimevr":
                            # Position every frame. Rotation exactly once: VRChat treats a lone
                            # rotation message (none other within 300 ms) as a one-time yaw snap.
                            send_rot = align_until is None
                            if send_rot:
                                align_until = now
                                print("yaw aligned to your current facing (press [a] to redo)")
                            trs.append(type(h)("head", h.position, h.rotation if send_rot else None))
                        if a.head == "once" and align_until is None:
                            align_until, head_avg = now + a.align_secs, None
                            print("aligning to the headset: stand straight, look forward…")
                        if a.head == "stream":
                            trs.append(h)
                        elif a.head == "slow" or (a.head == "once" and now < align_until):
                            k = 0.02 if a.head == "slow" else 0.15
                            p, r = np.array(h.position), np.array(h.rotation)
                            head_avg = (p, r) if head_avg is None else (head_avg[0] * (1 - k) + p * k, r)
                            trs.append(type(h)("head", tuple(head_avg[0]), tuple(head_avg[1])))
                        errs_before = sender.errors
                        sender.send(trs)
                        if logf:
                            import json as _json
                            logf.write(_json.dumps({
                                "t": t_ref, "swapped": list(res.swapped),
                                "valid": float(res.valid.mean()),
                                "raw": res.points.round(4).tolist(),
                                "stab": S.round(4).tolist(),
                                "sent": {x.id: [x.position, x.rotation] for x in trs},
                            }) + "\n")
                        if sender.errors > errs_before:
                            headset_down = True
                        elif headset_down:
                            # Headset is reachable again: the one-shot yaw may have been lost.
                            headset_down = False
                            align_until = None
                            last_yaw = -1e9  # re-send the T-pose yaw next time
                            print("headset reachable again — will re-send yaw alignment")
                        if not learned_msg:
                            print("bone lengths learned — trackers are live")
                            learned_msg = True
                    lat.append((time.perf_counter() - t_ref) * 1000)
                    n += 1
        if time.perf_counter() - t_fps > 1:
            fps, n, t_fps = n / (time.perf_counter() - t_fps), 0, time.perf_counter()
            if skips:
                print(f"3D fps {fps:.1f}, skipped so far: {dict(skips)}", flush=True)
            if a.no_view:
                print(f"\r{fps:4.1f} fps  capture->OSC {np.median(lat) if lat else 0:4.0f} ms", end="", flush=True)
        if a.no_view:
            continue
        tiles = []
        for wk in workers:
            with wk.lock:
                img, pose = wk.last_img, wk.last_pose
            img = np.zeros((640, 360, 3), np.uint8) if img is None else img.copy()
            if pose is not None:
                draw(img, pose)
            tiles.append(cv2.resize(img, (360, 640)))
        if S is not None:
            last_S, last_S_t = S, time.perf_counter()
        # Draw the last good skeleton instead of an empty panel (no blinking).
        valid = np.ones(26, bool) if last_S is not None else None
        view = np.hstack(tiles + [draw_views(last_S, valid, cams)])
        stale_ms = (time.perf_counter() - last_S_t) * 1000 if last_S is not None else 0
        tot = n + sum(skips.values()) or 1
        cv2.putText(view, f"skipped: {dict(skips)} | stale {stale_ms:.0f} ms",
                    (740, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        if not stab.fit.ready:
            status = f"learning bones {len(stab.fit.samples)}/{stab.fit.learn_frames}"
        elif a.head == "once" and align_until and time.perf_counter() < align_until:
            status = "ALIGNING head - stand straight"
        else:
            status = f"LIVE (head: {a.head}) [a]=realign [r]=relearn"
            if a.head == "tpose":
                status += " | hands up 1s -> yaw" + (" SENT" if time.perf_counter() - last_yaw < 3 else "")
        cam_lat = "/".join(f"{np.median(w.src.latency_ms[-30:]):.0f}" if w.src.latency_ms else "-" for w in workers)
        cv2.putText(view, f"{status} | {fps:.0f} fps -> {a.quest}:{a.port}",
                    (10, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(view, f"latency ms: camera->Mac {cam_lat} | pose {'/'.join(f'{w.pose_ms:.0f}' for w in workers)}"
                    f" | capture->OSC {np.median(lat) if lat else 0:.0f}",
                    (10, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("fbt hub", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a") and a.head == "tpose":
            force_yaw = True  # one yaw message on the next frame (look straight, body square)
        if key == ord("a") and a.head in ("once", "slimevr"):
            align_until = None  # re-align on the next frame
        if key == ord("r"):
            stab = SkeletonStabilizer(min_cutoff=a.smooth, beta=0.4, predict_s=a.predict)
            align_until, learned_msg = None, False
            print("re-learning bone lengths")
    for w in workers:
        w.src.stop()


if __name__ == "__main__":
    main()
