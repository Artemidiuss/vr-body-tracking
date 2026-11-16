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
from fbt.maccam import open_source
from fbt.filters import SkeletonStabilizer
from fbt.osc_out import VRChatSender
from fbt.skeleton import IDX
from fbt.trackers import RotationSmoother, head_tracker, is_hands_up, is_tpose, solve
from fbt.triangulate import triangulate_with_swap_check, view_thresholds, view_weights
from tools.bench_pose import draw
from tools.live3d import CamWorker, draw_views


def _fit(img, w, h):
    """Letterbox any image (portrait phone or landscape Mac) into a w x h tile."""
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quest", default="127.0.0.1", help="headset IP (or 127.0.0.1 for fake VRChat)")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("-b", "--backend", default="rtmpose-m:gpu")
    ap.add_argument("--trackers", default="hip,chest,left_foot,right_foot,left_knee,right_knee")
    ap.add_argument("--predict", type=float, default=0.08,
                    help="latency hiding horizon, s (speed-gated, clamped; 0 = off). Live: data/predict_ms")
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
    ap.add_argument("--no-mac", action="store_true", help="ignore extra cameras in the calibration")
    ap.add_argument("--extra-fps", type=float, default=10,
                    help="pose rate of extra cameras (the Mac): 3 models at 30 fps overheat a fanless "
                         "Air (~70 poses/s sustained); offline the Mac helps as much at 10 fps as at 30")
    a = ap.parse_args()
    t_stop = time.perf_counter() + a.seconds if a.seconds else float("inf")
    logf = open(a.log, "w") if a.log else None

    cams, hosts = load_calibration(a.calib)
    # Cameras 0 and 1 are the phones (required, they set the frame times);
    # any further camera (the MacBook's) is optional: used when it has a pose
    # at that moment, skipped if it fails to open or loses the person.
    workers, use_cams = [], []
    for i, (c, h) in enumerate(zip(cams, hosts)):
        if i >= 2 and a.no_mac:
            continue
        try:
            src = open_source(f"cam{i}", h).start()
        except Exception as e:
            if i < 2:
                raise
            print(f"extra camera {h} unavailable ({e}) — continuing with {len(workers)} cameras", flush=True)
            continue
        workers.append(CamWorker(src, a.backend, t_offset=c.time_offset, border=0.02 if i >= 2 else 0.0,
                                 max_rate=a.extra_fps if i >= 2 else 0.0))
        use_cams.append(c)
    cams = use_cams
    view_thr, view_w = view_thresholds(cams), view_weights(cams)
    if len(cams) > 2:
        print(f"reprojection thresholds px: {np.round(view_thr).astype(int).tolist()}", flush=True)
    use_extra, use_extra_mtime, n3 = True, None, 0
    print(f"cameras: {', '.join(h for h, c in zip(hosts, cams))}" if len(cams) == len(hosts) else
          f"cameras in use: {len(cams)} of {len(hosts)}", flush=True)
    sender = VRChatSender(a.quest, a.port)
    stab = SkeletonStabilizer(min_cutoff=a.smooth, beta=0.4, floor_clip=True, predict_gated=a.predict)
    align_until = None      # head is sent until this time (mode "once")
    headset_down = False
    tpose_since, last_yaw = None, -1e9
    force_yaw = False
    foot_state, rot_smooth = {}, RotationSmoother()
    head_fwd, head_fwd_mtime = 0.0, None
    head_dir, facing_samples = None, []
    from fbt.filters import MotionPredictor, OneEuro as _OE
    head_euro = _OE(2.5, 0.4)
    head_pred = MotionPredictor(0.15, lo=0.05, hi=0.3, clamp=0.15)
    head_knob_mtime = {}
    foot_mode, foot_mode_mtime = "exact", None
    predict_mtime = None
    yaw_trim, yaw_trim_mtime, yaw_trim_changed, aligned_yaw = 0.0, None, False, None
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
    stat_n, stat_prev, stat_t = 0, None, time.perf_counter()
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
            # Extra cameras: interpolated to t_ref, or a pose within 20 ms, else not used.
            extra = [w.pose_at(t_ref, max_dt=0.02, max_gap=2.5 / max(a.extra_fps, 1)) if use_extra else None
                     for w in workers[2:]]
            if pose_a is None or pose_b is None:
                skips["no synced pair"] += 1
                pose_b = None
            if pose_b is not None:
                res = triangulate_with_swap_check(cams, [pose_a, pose_b] + extra, prev=prev,
                                                  max_err_px=view_thr, view_weight=view_w)
                n3 = int((res.valid & (res.n_views >= 3)).sum())
                core = [IDX[k] for k in ("hip", "left_hip", "right_hip", "neck")]
                if not res.valid[core].all():
                    skips["torso not triangulated"] += 1
                if res.valid[core].all():
                    prev = res
                    S = stab(res.points, res.valid, t_ref)
                    if stab.fit.ready:
                        trs = solve(S, enabled=enabled, foot_mode=foot_mode)
                        h = head_tracker(S)
                        now = time.perf_counter()
                        if a.head == "pos":
                            hp = head_tracker(S, source="torso", forward=head_fwd)
                            trs.append(type(hp)("head", hp.position, None))
                        if a.head == "tpose":
                            # VRChat turns the whole OSC space to match any head rotation we
                            # stream; camera head yaw is too imprecise for that. So: position
                            # every frame (steady torso-based point), rotation only once, when
                            # the user holds a T-pose (body and head face the same way).
                            hp = head_tracker(S, source="torso")   # used for the yaw only
                            # Anchor = the real head root (ear midpoint, 4 cm down the spine),
                            # smoothed and predicted ~150 ms: VRChat re-anchors ALL trackers to
                            # it every frame against the instant headset, so a lagging anchor or
                            # one that does not follow the head when bending drags the feet.
                            # On our logs: feet slide 5.9 -> 2.8 cm (sway), 4.4 -> 1.0 cm (bend).
                            from fbt.trackers import to_unity as _tu0
                            Su = _tu0(S)
                            sp_ = Su[IDX["neck"]] - Su[IDX["hip"]]
                            root = (Su[IDX["left_ear"]] + Su[IDX["right_ear"]]) / 2 - 0.04 * sp_ / np.linalg.norm(sp_)
                            root = head_euro(root, t_ref)
                            root = head_pred(root[None, :], t_ref)[0]
                            pos = root
                            # Head-turn compensation: VRChat derives the avatar head bone from
                            # the headset with the avatar's viewpoint offset; a mismatch makes
                            # the head bone circle a few cm when the head turns, and VRChat drags
                            # all trackers along. Shifting our anchor along the HEAD's facing
                            # (smoothed ear line) by data/head_offset metres cancels it; the same
                            # knob also moves all cubes forward/back.
                            from fbt.trackers import to_unity as _tu
                            U = _tu(S)
                            ear = U[IDX["right_ear"]] - U[IDX["left_ear"]]
                            hd = np.cross(ear, [0.0, 1.0, 0.0])
                            hd[1] = 0
                            if np.linalg.norm(hd) > 1e-6:
                                hd /= np.linalg.norm(hd)
                                head_dir = hd if head_dir is None else head_dir * 0.8 + hd * 0.2
                            if head_dir is not None and np.linalg.norm(head_dir) > 1e-6:
                                pos = pos + head_fwd * head_dir / np.linalg.norm(head_dir)
                            # Body facing samples while the alignment pose is held (averaged).
                            tp = is_hands_up(S) or is_tpose(S)
                            if os.path.exists("data/align_now"):
                                os.remove("data/align_now")
                                force_yaw = True
                            tpose_since = (tpose_since or now) if tp else None
                            if tp:
                                facing_samples.append(hp.rotation[1])
                            elif tpose_since is None:
                                facing_samples.clear()
                            send_rot = force_yaw or (tpose_since is not None and now - tpose_since > 1.0
                                                     and now - last_yaw > 5.0)
                            force_yaw = False
                            rot = None
                            if send_rot:
                                if facing_samples:
                                    a_ = np.radians(facing_samples)
                                    aligned_yaw = float(np.degrees(np.arctan2(np.sin(a_).mean(), np.cos(a_).mean())))
                                else:
                                    aligned_yaw = hp.rotation[1]
                                facing_samples.clear()
                            if send_rot or yaw_trim_changed:
                                last_yaw = now
                                yaw_trim_changed = False
                                rot = (0.0, (aligned_yaw if aligned_yaw is not None else hp.rotation[1]) + yaw_trim, 0.0)
                                print(f"yaw aligned: {rot[1]:+.1f} deg (trim {yaw_trim:+.1f}), one message", flush=True)
                            hp = type(hp)("head", tuple(map(float, pos)), rot)
                            trs.append(hp)
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
                                "views": res.n_views.tolist(),
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
        # Live A/B of the extra camera: data/use_mac = 1 | 0
        try:
            m = os.path.getmtime("data/use_mac")
            if m != use_extra_mtime:
                use_extra_mtime = m
                use_extra = open("data/use_mac").read().strip() != "0"
                print(f"extra camera: {'ON' if use_extra else 'OFF'}", flush=True)
        except OSError:
            pass
        # Live-tunable head anchor shift (metres, + = forward): data/head_offset
        try:
            m = os.path.getmtime("data/head_offset")
            if m != head_fwd_mtime:
                head_fwd_mtime = m
                head_fwd = float(open("data/head_offset").read().strip() or 0)
                print(f"head anchor shift: {head_fwd * 100:+.0f} cm", flush=True)
        except (OSError, ValueError):
            pass
        # Live head-anchor tuning: data/head_predict_ms, data/head_smooth_hz
        for fname, attr in (("data/head_predict_ms", "h"), ("data/head_smooth_hz", "hz")):
            try:
                m = os.path.getmtime(fname)
                if m != head_knob_mtime.get(fname):
                    head_knob_mtime[fname] = m
                    val = float(open(fname).read().strip())
                    if attr == "h":
                        head_pred.h = val / 1000
                        print(f"head prediction: {val:.0f} ms", flush=True)
                    else:
                        head_euro.min_cutoff = val
                        print(f"head smoothing: {val:.1f} Hz", flush=True)
            except (OSError, ValueError):
                pass
        # Live prediction horizon: data/predict_ms (milliseconds)
        try:
            m = os.path.getmtime("data/predict_ms")
            if m != predict_mtime:
                predict_mtime = m
                ms = float(open("data/predict_ms").read().strip() or 0)
                if stab.pred is not None:
                    stab.pred.h = ms / 1000
                print(f"prediction: {ms:.0f} ms", flush=True)
        except (OSError, ValueError):
            pass
        # Live foot mode: data/foot_mode = "flat" | "exact"
        try:
            m = os.path.getmtime("data/foot_mode")
            if m != foot_mode_mtime:
                foot_mode_mtime = m
                v = open("data/foot_mode").read().strip()
                foot_mode = v if v in ("flat", "exact") else "exact"
                print(f"foot mode: {foot_mode}", flush=True)
        except OSError:
            pass
        # Live yaw trim (degrees) like ju1ce's rotation slider: data/yaw_trim.
        # Changing it re-sends one yaw message with the trim applied.
        try:
            m = os.path.getmtime("data/yaw_trim")
            if m != yaw_trim_mtime:
                first = yaw_trim_mtime is None
                yaw_trim_mtime = m
                yaw_trim = float(open("data/yaw_trim").read().strip() or 0)
                yaw_trim_changed = not first and aligned_yaw is not None
                print(f"yaw trim: {yaw_trim:+.1f} deg", flush=True)
        except (OSError, ValueError):
            pass
        if time.perf_counter() - t_fps > 1:
            fps, n, t_fps = n / (time.perf_counter() - t_fps), 0, time.perf_counter()
            if skips:
                print(f"3D fps {fps:.1f}, skipped so far: {dict(skips)}", flush=True)
            if a.no_view:
                print(f"\r{fps:4.1f} fps  capture->OSC {np.median(lat) if lat else 0:4.0f} ms", end="", flush=True)
            # Per-camera rates every 5 s: frames arriving, poses computed, pose time, camera latency.
            stat_n += 1
            if stat_n % 5 == 0:
                cur = [(getattr(w.src, "_i", 0), w.n_pose) for w in workers]
                if stat_prev is not None:
                    dt_s = time.perf_counter() - stat_t
                    cams_txt = " | ".join(
                        f"cam{i}: in {(c[0] - p_[0]) / dt_s:4.1f} pose {(c[1] - p_[1]) / dt_s:4.1f}/s "
                        f"{w.pose_ms:3.0f} ms lat {np.median(w.src.latency_ms[-30:]) if w.src.latency_ms else 0:3.0f}"
                        for i, (c, p_, w) in enumerate(zip(cur, stat_prev, workers)))
                    print(f"\nstats 3D {fps:4.1f} fps | {cams_txt}", flush=True)
                stat_prev, stat_t = cur, time.perf_counter()
        if a.no_view:
            continue
        tiles = []
        for wk in workers:
            with wk.lock:
                img, pose = wk.last_img, wk.last_pose
            img = np.zeros((640, 360, 3), np.uint8) if img is None else img.copy()
            if pose is not None:
                draw(img, pose)
            tiles.append(_fit(img, 360, 640))
        if S is not None:
            last_S, last_S_t = S, time.perf_counter()
        # Draw the last good skeleton instead of an empty panel (no blinking).
        valid = np.ones(26, bool) if last_S is not None else None
        view = np.hstack(tiles + [draw_views(last_S, valid, cams)])
        if view.shape[1] > 1440:   # 3 cameras + 3 views do not fit a laptop screen
            sc = 1440 / view.shape[1]
            view = cv2.resize(view, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        view = np.vstack([view, np.zeros((90, view.shape[1], 3), np.uint8)])
        H = view.shape[0]
        stale_ms = (time.perf_counter() - last_S_t) * 1000 if last_S is not None else 0
        extra_txt = (f" | Mac camera {'ON' if use_extra else 'OFF'}: {n3} points from 3 views"
                     if len(workers) > 2 else "")
        cv2.putText(view, f"skipped: {dict(skips)} | stale {stale_ms:.0f} ms{extra_txt}",
                    (10, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
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
                    (10, H - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(view, f"latency ms: camera->Mac {cam_lat} | pose {'/'.join(f'{w.pose_ms:.0f}' for w in workers)}"
                    f" | capture->OSC {np.median(lat) if lat else 0:.0f}",
                    (10, H - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("fbt hub", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a") and a.head == "tpose":
            force_yaw = True  # one yaw message on the next frame (look straight, body square)
        if key == ord("a") and a.head in ("once", "slimevr"):
            align_until = None  # re-align on the next frame
        if key == ord("r"):
            stab = SkeletonStabilizer(min_cutoff=a.smooth, beta=0.4, floor_clip=True, predict_gated=a.predict)
            align_until, learned_msg = None, False
            print("re-learning bone lengths")
    for w in workers:
        w.src.stop()


if __name__ == "__main__":
    main()
