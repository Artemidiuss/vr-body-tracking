"""Core of the FBT app (fbtapp): cameras that reconnect on their own, live 3D for
the preview, calibration from the poses computed live, and the VRChat tracker
pipeline.

The tracker pipeline is the production path of tools/run_hub.py (--head tpose,
its live knob files in data/), which stays as the CLI for experiments. The app keeps
those values in data/app_settings.json (imported from the knob files once) and the
settings window changes them live.
"""

from __future__ import annotations

import collections
import csv
import glob
import json
import os
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np

from .calibrate import (PoseTrack, calibrate_extra, calibrate_pair, load_calibration, save_calibration,
                        triangulate_tracks)
from .filters import MotionPredictor, OneEuro, SkeletonFit, SkeletonStabilizer
from .maccam import mac_focal_range, open_source
from .osc_out import VRChatSender
from .pose import Pose2D, make_backend
from .skeleton import IDX
from .trackers import head_tracker, is_hands_up, is_tpose, solve, to_unity
from .triangulate import triangulate_with_swap_check, view_thresholds, view_weights

CORE = [IDX[k] for k in ("hip", "left_hip", "right_hip", "neck")]
DEFAULTS = {
    "height_m": 1.73,
    "quest_ip": "192.168.31.150",
    "cameras": ["192.168.31.81", "192.168.31.28", "mac"],
    "names": {},                 # host -> friendly name, learned on first connect
    "use_mac": True,
    "voice_alerts": True,
    "calib_seconds": 30,
    "keep_logs": 5,
    # Trackers (hip and feet are always on)
    "hmd_anchor": True,          # with the headset pose: trackers in VRChat space, head = real headset
    "track_chest": True,
    "track_knees": True,
    "track_elbows": False,
    "foot_flat": False,          # feet perpendicular to the shin (steadier, no ankle flex)
    # Tuning, applied live
    "smooth_hz": 2.5,            # body One-Euro min cutoff: lower = smoother, more lag
    "predict_ms": 0,             # body motion prediction (speed-gated)
    "head_offset_cm": -5,        # head anchor along the head's facing: moves all trackers fwd/back in VRChat
    "head_predict_ms": 200,
    "head_smooth_hz": 2.5,
    "yaw_trim": 0,               # degrees added to the one-shot yaw alignment
    "extra_fps": 10,             # pose rate of the Mac camera (heat: see ARCHITECTURE.md §19)
}
TUNABLE = {"hmd_anchor", "smooth_hz", "predict_ms", "head_offset_cm", "head_predict_ms", "head_smooth_hz", "yaw_trim",
           "extra_fps", "track_chest", "track_knees", "track_elbows", "foot_flat"}
# Old live-knob files in data/ (tools/run_hub.py) -> settings, imported once.
KNOB_FILES = {"head_offset": ("head_offset_cm", lambda v: round(float(v) * 100)),
              "head_predict_ms": ("head_predict_ms", float), "head_smooth_hz": ("head_smooth_hz", float),
              "predict_ms": ("predict_ms", float), "yaw_trim": ("yaw_trim", float),
              "foot_mode": ("foot_flat", lambda v: v.strip() == "flat"),
              "use_mac": ("use_mac", lambda v: v.strip() != "0")}


def enabled_trackers(st: dict) -> tuple:
    t = ["hip", "left_foot", "right_foot"]
    if st.get("track_chest"):
        t.append("chest")
    if st.get("track_knees"):
        t += ["left_knee", "right_knee"]
    if st.get("track_elbows"):
        t += ["left_elbow", "right_elbow"]
    return tuple(t)


def friendly_name(host: str, info: dict | None) -> str | None:
    if host == "mac" or host.startswith("mac:"):
        return "MacBook"
    d = (info or {}).get("device", "")
    if not d:
        return None
    if "23053RN02Y" in d or "Redmi" in d:
        return "Redmi"
    if "SM-A32" in d:
        return "Samsung"
    if d.startswith("MacBook"):
        return "MacBook"
    return d.split()[0].capitalize()


def say(text: str):
    try:
        subprocess.Popen(["say", "-v", "Lesya", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


class Cancelled(Exception):
    pass


class HeadsetPose:
    """Headset and wrist poses that VRChat sends over OSCQuery (/tracking/vrsystem/...).

    VRChat on the Quest sends them only to a service it discovers AFTER "Allow
    Sending Head and Wrist VR Tracking OSC Data" is switched on; a service found
    earlier only gets queried. So while VRChat queries us (or the headset is up)
    and nothing arrives, the service is re-advertised under a new name and ports
    (ARCHITECTURE.md §22)."""

    def __init__(self, quest_ip_fn, headset_up_fn=lambda: True):
        self.quest_ip, self.headset_up = quest_ip_fn, headset_up_fn
        self.rx = None
        self.lock = threading.Lock()
        self.head = collections.deque(maxlen=400)          # (t_mac, x, y, z, ex, ey, ez)
        self.last = {}                                     # address -> (t, args)
        self.n = 0
        self.rate = 0.0
        self.readvertised = 0
        self.queries = self.datagrams = 0
        self._queries_before = self._datagrams_before = 0
        self._n_start = 0
        self.logf = None
        self._t_rx = 0.0
        self._quit = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="hmd").start()

    @property
    def fresh(self) -> bool:
        with self.lock:
            return bool(self.head) and time.perf_counter() - self.head[-1][0] < 1.0

    def pose_at(self, t: float):
        """Headset (position, euler) at Mac time t, interpolated; None if not covered."""
        with self.lock:
            if len(self.head) < 2 or not (self.head[0][0] <= t <= self.head[-1][0]):
                return None
            h = list(self.head)
        ts = [x[0] for x in h]
        import bisect
        j = bisect.bisect_left(ts, t)
        a, b = h[j - 1], h[j]
        if b[0] - a[0] > 0.1:
            return None
        w = (t - a[0]) / max(b[0] - a[0], 1e-9)
        p = (1 - w) * np.array(a[1:4]) + w * np.array(b[1:4])
        return p, np.array(b[4:7])

    def latest(self):
        with self.lock:
            return self.head[-1] if self.head else None

    def yaw_rate(self, window: float = 0.3) -> float:
        """|deg/s| of the headset yaw over the last `window` seconds."""
        with self.lock:
            h = [x for x in self.head if self.head[-1][0] - x[0] <= window]
        if len(h) < 2:
            return 999.0
        dy = (h[-1][5] - h[0][5] + 180) % 360 - 180
        return abs(dy) / max(h[-1][0] - h[0][0], 1e-3)

    def _on(self, addr, args, t):
        if not addr.startswith("/tracking/vrsystem/") or len(args) < 6:
            return
        with self.lock:
            self.last[addr] = (t, args)
            if addr.endswith("head/pose"):
                self.head.append((t, *map(float, args[:6])))
                self.n += 1
            f = self.logf
        if f is not None:
            try:
                f.write(json.dumps({"t": t, "a": addr.split("/")[3], "v": [round(float(x), 5) for x in args[:6]]}) + "\n")
            except (OSError, ValueError):
                pass

    def _start(self):
        from .oscquery import OSCQueryReceiver
        if self.rx is not None:
            try:
                self.rx.close()
            except Exception:
                pass
        if self.rx is not None:
            self._queries_before = self.queries
            self._datagrams_before = self.datagrams
        if self.rx is not None:
            time.sleep(0.3)                 # let the old sockets go before binding the same ports
        self.rx = OSCQueryReceiver(name="FBT", on_message=self._on, extra_ports=(9001,),
                                   osc_port=9101, http_port=9102)
        self._t_rx = time.perf_counter()
        self._n_start = self.n

    def _try_start(self):
        try:
            self._start()
            print(f"hmd: advertising OSCQuery on {self.rx.ip} udp {self.rx.osc_port} http {self.rx.http_port}", flush=True)
        except Exception as e:
            self.rx = None
            print(f"hmd: cannot advertise: {e!r}", flush=True)

    def _run(self):
        self._try_start()
        n_prev, t_prev, attempts, seen_hits = 0, time.perf_counter(), [], 0
        while not self._quit.wait(1.0):
            now = time.perf_counter()
            self.rate, n_prev, t_prev = (self.n - n_prev) / (now - t_prev), self.n, now
            rx = self.rx
            if rx is None:
                if now - self._t_rx > 10:
                    self._t_rx = now
                    self._try_start()
                continue
            hits = [h for h in rx.http_hits if h[1] == self.quest_ip()]
            for h in hits[seen_hits:]:
                print(f"hmd: VRChat queried {h[2]}", flush=True)
            seen_hits = len(hits)
            self.queries = self._queries_before + len(hits)
            self.datagrams = self._datagrams_before + sum(rx.raw.values())
            if self.fresh:
                attempts.clear()
                continue
            # Advertise anew (at most 3 times in 5 minutes: every new service makes VRChat show a
            # notice in the headset) when
            # VRChat looked at this service 10 s ago and sends nothing: it found us before
            # "Allow Sending Head and Wrist..." was on. A pause in a flowing stream is NOT a
            # reason: with the headset taken off VRChat stops sending and later resumes to the
            # same service by itself (seen live), so re-advertising then would lose it.
            queried_silent = hits and time.time() - hits[-1][0] > 10 and self.n == self._n_start
            if queried_silent:
                attempts = [a for a in attempts if now - a < 300]
                if len(attempts) < 3:
                    attempts.append(now)
                    self.readvertised += 1
                    print(f"hmd: queried but no data, advertising anew ({len(attempts)}/3)", flush=True)
                    self._try_start()
                    seen_hits = 0

    def close(self):
        self._quit.set()
        if self.rx is not None:
            try:
                self.rx.close()
            except Exception:
                pass


# ---------------------------------------------------------------- cameras
class CamSlot:
    """One camera: connects (and reconnects) in the background and runs pose on
    its newest frames. The Mac camera computes fewer poses (max_rate) unless a
    calibration is being recorded."""

    def __init__(self, index: int, host: str, backend_spec: str, border: float = 0.0,
                 max_rate: float = 0.0, t_offset: float = 0.0):
        self.index, self.host = index, host
        self.backend_spec = backend_spec
        self.border, self.max_rate, self.t_offset = border, max_rate, t_offset
        self.state, self.error = "connecting", ""
        self.src = None
        self.poses = collections.deque(maxlen=8)        # (t_capture + offset, Pose2D)
        self.lock = threading.Lock()
        self.new = threading.Event()
        self.last_img = self.last_pose = None
        self.person_t = 0.0
        self.pose_ms, self.n_pose, self.n_frames, self.errors = 0.0, 0, 0, 0
        self.rec: list | None = None                    # (t_capture, kp, sc) while calibrating
        self.rec_size = None
        self.rate_pose = self.rate_in = 0.0
        self.offline_since: float | None = time.perf_counter()
        self._last_t = -1e9
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name=f"cam{index}").start()

    @property
    def online(self) -> bool:
        return self.state == "online"

    @property
    def info(self) -> dict:
        return getattr(self.src, "info", None) or {}

    @property
    def is_mac(self) -> bool:
        return self.host.startswith("mac") or self.info.get("device", "").startswith("MacBook")

    def status(self) -> dict:
        s = self.src
        st, t = getattr(s, "status", None), getattr(s, "status_t", None)
        if not st or t is None or time.perf_counter() - t > 20:
            return {}
        return st

    def stop(self):
        self._stop.set()
        s = self.src
        if s is not None:
            s.stop()

    def _run(self):
        try:
            self.backend = make_backend(self.backend_spec)
        except Exception as e:
            self.state, self.error = "error", f"модель: {e}"
            return
        while not self._stop.is_set():
            try:
                src = open_source(f"cam{self.index}", self.host).start()
            except Exception as e:
                self.state, self.error = "offline", str(e)
                self._stop.wait(2.0)
                continue
            self.src = src
            why = self._pose_loop(src)
            src.stop()
            with self.lock:
                self.poses.clear()
                self.last_pose = None
            if self.state == "online":
                self.offline_since = time.perf_counter()
            self.state, self.error = "offline", why
            self._stop.wait(1.0)

    def _pose_loop(self, src) -> str:
        last, last_new = -1, time.perf_counter()
        while not self._stop.is_set():
            f = src.latest(last)
            if f is None:
                if src.finished.is_set():
                    return "зв'язок закрито"
                if time.perf_counter() - last_new > 5.0:
                    return "немає кадрів"
                time.sleep(0.002)
                continue
            last, last_new = f.index, time.perf_counter()
            self.n_frames += 1
            if self.state != "online":
                self.state, self.error, self.offline_since = "online", "", None
            rec = self.rec
            if (self.max_rate and rec is None
                    and f.timestamp - self._last_t < 1.0 / self.max_rate - 0.005):
                with self.lock:
                    self.last_img = f.image     # preview stays smooth between poses
                continue
            self._last_t = f.timestamp
            t0 = time.perf_counter()
            try:
                p = self.backend(f.image)
            except Exception:                     # one bad frame never kills the camera
                self.errors += 1
                p = None
            if rec is not None and p is not None:
                rec.append((f.timestamp, p.keypoints.copy(), p.scores.copy()))
                self.rec_size = (f.image.shape[1], f.image.shape[0])
            if p is not None and self.border > 0:
                h, w = f.image.shape[:2]
                u, v, b = p.keypoints[:, 0], p.keypoints[:, 1], self.border
                out = (u < b * w) | (u > (1 - b) * w) | (v < b * h) | (v > (1 - b) * h)
                if out.any():
                    p = Pose2D(p.keypoints, np.where(out, 0.0, p.scores).astype(np.float32))
            self.pose_ms = 0.9 * self.pose_ms + 0.1 * (time.perf_counter() - t0) * 1000
            self.n_pose += 1
            if p is not None and int((p.scores > 0.5).sum()) >= 6:
                self.person_t = time.perf_counter()
            with self.lock:
                self.last_img, self.last_pose = f.image, p
                if p is not None:
                    self.poses.append((f.timestamp + self.t_offset, p))
            self.new.set()
        return "зупинено"

    def latest_pose(self):
        with self.lock:
            return self.poses[-1] if self.poses else None

    def pose_at(self, t: float, max_dt: float = 0.05, max_gap: float = 0.1) -> Pose2D | None:
        """Pose interpolated to capture time t (or the nearest within max_dt)."""
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


class VideoRecorder(threading.Thread):
    """Writes one camera's frames during a calibration (same files as
    tools.record, so the offline tools can re-run on them)."""

    def __init__(self, slot: CamSlot, out_dir: str):
        super().__init__(daemon=True)
        self.slot, self.out_dir = slot, out_dir
        self.halt = threading.Event()
        self.info: dict = {}
        self.start()

    def run(self):
        writer, rows, last, src = None, [], -1, None
        name = f"cam{self.slot.index}"
        while not self.halt.is_set():
            s = self.slot.src
            if s is not src:
                src, last = s, -1
            f = src.latest(last) if src is not None else None
            if f is None:
                time.sleep(0.002)
                continue
            last = f.index
            if writer is None:
                h, w = f.image.shape[:2]
                self.info = {**src.info, "host": self.slot.host}
                writer = cv2.VideoWriter(f"{self.out_dir}/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                         src.info.get("fps", 30), (w, h))
            writer.write(f.image)
            rows.append((len(rows), f"{f.timestamp:.6f}", f"{(f.received_at - f.timestamp) * 1000:.1f}"))
        if writer is not None:
            writer.release()
        with open(f"{self.out_dir}/{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "t_capture", "latency_ms"])
            w.writerows(rows)


class TrackerOutput:
    """Skeleton -> VRChat trackers + the head anchor (tools/run_hub.py, --head tpose)."""

    def __init__(self, sender: VRChatSender, settings: dict, log_path: str | None, align=None, hmd=None):
        self.sender, self.st = sender, settings
        self.align, self.hmd = align, hmd
        self.anchor = "camera"             # "camera" | "hmd": where the head anchor comes from
        self.align_yaw_sent = None         # alignment yaw (deg) when the last head yaw went out
        self.trim_sent = settings["yaw_trim"]
        self.head_euro = OneEuro(2.5, 0.4)
        self.head_pred = MotionPredictor(0.15, lo=0.05, hi=0.3, clamp=0.15)
        self.head_dir, self.facing = None, []
        self.tpose_since, self.last_yaw, self.aligned_yaw = None, -1e9, None
        self.yaw_sent_t: float | None = None
        self.headset_down = False
        self.logf = open(log_path, "w") if log_path else None

    def close(self):
        if self.logf:
            self.logf.close()
            self.logf = None

    def step(self, S, res, t_ref):
        st = self.st
        self.head_pred.h = float(st["head_predict_ms"]) / 1000
        self.head_euro.min_cutoff = max(0.05, float(st["head_smooth_hz"]))
        trs = solve(S, enabled=enabled_trackers(st), foot_mode="flat" if st["foot_flat"] else "exact")
        now = time.perf_counter()
        hp = head_tracker(S, source="torso")                  # used for the yaw only
        # Anchor = the real head root (ear midpoint, 4 cm down the spine), smoothed and
        # predicted: VRChat re-anchors all trackers to it every frame (see run_hub.py).
        Su = to_unity(S)
        sp = Su[IDX["neck"]] - Su[IDX["hip"]]
        root = (Su[IDX["left_ear"]] + Su[IDX["right_ear"]]) / 2 - 0.04 * sp / np.linalg.norm(sp)
        al, hmd = self.align, self.hmd
        self._q_root = root.copy()
        if al is not None and hmd is not None and hmd.fresh:
            al.add(t_ref, root.copy(), hmd.pose_at)
        root = self.head_euro(root, t_ref)
        pos = self.head_pred(root[None, :], t_ref)[0]
        ear = Su[IDX["right_ear"]] - Su[IDX["left_ear"]]
        hd = np.cross(ear, [0.0, 1.0, 0.0])
        hd[1] = 0
        if np.linalg.norm(hd) > 1e-6:
            hd /= np.linalg.norm(hd)
            self.head_dir = hd if self.head_dir is None else self.head_dir * 0.8 + hd * 0.2
        if self.head_dir is not None and np.linalg.norm(self.head_dir) > 1e-6:
            pos = pos + float(st["head_offset_cm"]) / 100 * self.head_dir / np.linalg.norm(self.head_dir)
        # One yaw message after hands up / T-pose held 1 s (averaged body facing).
        tp = is_hands_up(S) or is_tpose(S)
        self.tpose_since = (self.tpose_since or now) if tp else None
        if tp:
            self.facing.append(hp.rotation[1])
        elif self.tpose_since is None:
            self.facing.clear()
        send_rot = self.tpose_since is not None and now - self.tpose_since > 1.0 and now - self.last_yaw > 5.0
        if send_rot:
            if self.facing:
                a = np.radians(self.facing)
                self.aligned_yaw = float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))
            else:
                self.aligned_yaw = hp.rotation[1]
            self.facing.clear()
        trim_changed = st["yaw_trim"] != self.trim_sent and self.aligned_yaw is not None
        rot = None
        if send_rot or trim_changed:
            self.last_yaw = self.yaw_sent_t = now
            self.trim_sent = st["yaw_trim"]
            rot = (0.0, (self.aligned_yaw if self.aligned_yaw is not None else hp.rotation[1])
                   + float(st["yaw_trim"]), 0.0)
        last = hmd.latest() if hmd is not None else None
        if (st.get("hmd_anchor", True) and al is not None and hmd is not None and hmd.fresh and last is not None
                and self.tpose_since is not None and now - self.tpose_since > 1.0
                and now - getattr(self, "_reset_t", -1e9) > 5.0):
            facing = float(np.degrees(np.arctan2(np.sin(np.radians(self.facing)).mean(),
                                                 np.cos(np.radians(self.facing)).mean()))) if self.facing else hp.rotation[1]
            al.reset_from_pose(self._q_root, facing, last[1:4], last[4:7])
            self._reset_t = now
            self.align_yaw_sent = None              # -> a fresh yaw message goes out below
        aligned = al is not None and al.ready and st.get("hmd_anchor", True)
        if aligned:
            trs, head = self._to_vrchat(trs, pos, hp, tp, now)
            trs.append(head)
        else:
            self.anchor = "camera"
            if self.align_yaw_sent is not None:   # back from the headset mode: the old yaw is stale
                self.align_yaw_sent = None
                self.last_yaw = -1e9
            trs.append(type(hp)("head", tuple(map(float, pos)), rot))
        errs = self.sender.errors
        self.sender.send(trs)
        if self.sender.errors > errs:
            self.headset_down = True
        elif self.headset_down:                 # headset back: the one-shot yaw may be lost
            self.headset_down = False
            self.last_yaw = -1e9
            self.yaw_sent_t = None
        if self.logf:
            self.logf.write(json.dumps({
                "t": t_ref, "anchor": self.anchor,
                "align": None if not aligned else [round(float(v), 5) for v in al.x],
                "swapped": list(res.swapped), "valid": float(res.valid.mean()),
                "views": res.n_views.tolist(), "raw": res.points.round(4).tolist(),
                "stab": S.round(4).tolist(), "sent": {x.id: [x.position, x.rotation] for x in trs},
            }) + "\n")


    def _to_vrchat(self, trs, cam_head, hp, tpose, now):
        """Trackers already in VRChat space; head = the real headset root when it is worn
        (else our camera anchor moved into VRChat space, so nothing jumps). VRChat then
        needs only a yaw that matches the headset's: sent once, while the head is still."""
        from .hmd_align import Ry, unity_R
        from .trackers import unity_euler
        al, hmd = self.align, self.hmd
        al.tick(now)
        R = Ry(np.radians(al.yaw_deg()))
        out = []
        for tr in trs:
            p = al.to_vrc(np.array(tr.position, float))
            r = None if tr.rotation is None else unity_euler(R @ unity_R(tr.rotation))
            out.append(type(tr)(tr.id, tuple(map(float, p)), r))
        last = hmd.latest()
        worn = hmd.fresh and al.worn and last is not None
        if worn:
            head_pos = al.head_root(last[1:4], last[4:7])
            # Where the avatar's head root sits relative to its eyes depends on the avatar, so the
            # same "trackers forward / back" knob as in the camera mode (along the headset's facing).
            yaw = np.radians(float(last[5]))
            head_pos = head_pos + float(self.st["head_offset_cm"]) / 100 * np.array([np.sin(yaw), 0.0, np.cos(yaw)])
            yaw_now = float(last[5])
            self.anchor = "hmd"
        else:
            head_pos = al.to_vrc(np.array(cam_head, float))
            yaw_now = (hp.rotation[1] + al.yaw_deg()) if hp.rotation is not None else None
            self.anchor = "camera"
        rot = None
        need = (self.align_yaw_sent is None or abs((al.yaw_deg() - self.align_yaw_sent + 180) % 360 - 180) > 3
                or (tpose and now - self.last_yaw > 5.0))
        if need and yaw_now is not None and (not worn or hmd.yaw_rate() < 20.0):
            rot = (0.0, yaw_now, 0.0)
            self.align_yaw_sent = al.yaw_deg()
            self.last_yaw = self.yaw_sent_t = now
        return out, type(hp)("head", tuple(map(float, head_pos)), rot)


# ---------------------------------------------------------------- engine
class Engine:
    def __init__(self, data_dir: str = "data", backend: str = "rtmpose-m:gpu", settings_path: str | None = None,
                 cameras: list[str] | None = None):
        self.d = data_dir
        self.backend = backend
        self.calib_path = f"{data_dir}/calibration.json"
        self.settings_path = settings_path or f"{data_dir}/app_settings.json"
        os.makedirs(f"{data_dir}/logs", exist_ok=True)
        self.lock = threading.RLock()
        self.cams, self.calib_hosts, self.calib_meta, self.calib_time = [], [], {}, None
        self._load_calibration()
        self.settings = self._load_settings()
        if cameras:
            self.settings["cameras"] = list(cameras)
        self.mode = "setup"
        self.calib: dict = {"state": "idle"}
        self._cancel = threading.Event()
        self.out: TrackerOutput | None = None
        self.stab: SkeletonStabilizer | None = None
        self.skeleton = self.skeleton_valid = None
        self.visible_t = 0.0
        self.fps3d, self.skips = 0.0, collections.Counter()
        self._agree = collections.deque(maxlen=150)
        self.headset_online: bool | None = None
        self.thermal: int | None = None
        self.mac_battery: dict | None = None
        self.server_t0: float | None = None
        self._caffeinate = None
        self._statsf = None
        self._quit = threading.Event()
        self.slots: list[CamSlot] = []
        self._make_slots()
        self.hmd = HeadsetPose(lambda: self.settings.get("quest_ip"), lambda: bool(self.headset_online))
        from .hmd_align import HmdAlign
        self.align = HmdAlign()
        threading.Thread(target=self._align_loop, daemon=True, name="align").start()
        for fn in (self._track_loop, self._monitor_loop, self._ping_loop):
            threading.Thread(target=fn, daemon=True, name=fn.__name__).start()

    # ---------------- settings / calibration files
    def _load_settings(self) -> dict:
        s = dict(DEFAULTS, names={})
        if self.calib_hosts:
            s["cameras"] = list(self.calib_hosts)
            for c, h in zip(self.cams, self.calib_hosts):      # names before the first connect
                n = friendly_name(h, {"device": c.name})
                if n:
                    s["names"][h] = n
        saved = {}
        try:
            saved = json.load(open(self.settings_path))
        except (OSError, ValueError):
            pass
        for f, (k, conv) in KNOB_FILES.items():
            if k not in saved:
                try:
                    s[k] = conv(open(f"{self.d}/{f}").read().strip())
                except (OSError, ValueError):
                    pass
        names = {**s["names"], **saved.get("names", {})}
        s.update(saved)
        s["names"] = names
        return s

    def save_settings(self):
        with open(self.settings_path, "w") as fh:
            json.dump(self.settings, fh, indent=2, ensure_ascii=False)

    def _load_calibration(self):
        if not os.path.exists(self.calib_path):
            return
        try:
            cams, hosts = load_calibration(self.calib_path)
            meta = json.load(open(self.calib_path)).get("meta", {})
        except (OSError, ValueError, KeyError):
            return
        self._apply_calibration(cams, hosts, meta, os.path.getmtime(self.calib_path))

    def _apply_calibration(self, cams, hosts, meta, when):
        with self.lock:
            self.cams, self.calib_hosts, self.calib_meta, self.calib_time = cams, hosts, meta, when
            self.view_thr = view_thresholds(cams) if len(cams) >= 2 else None
            self.view_w = view_weights(cams) if len(cams) >= 2 else None
            for s in getattr(self, "slots", []):
                s.t_offset = cams[s.index].time_offset if s.index < len(cams) else 0.0
            self._reset_3d = True
            self._agree = collections.deque(maxlen=150)
            if getattr(self, "align", None) is not None:      # new camera space: learn the link again
                from .hmd_align import HmdAlign
                self.align = HmdAlign()
                if self.out is not None:
                    self.out.align = self.align

    def _make_slots(self):
        for s in self.slots:
            s.stop()
        cams = self.settings["cameras"]
        self.slots = [CamSlot(i, h, self.backend, border=0.02 if i >= 2 else 0.0,
                              max_rate=self.settings["extra_fps"] if i >= 2 else 0.0,
                              t_offset=self.cams[i].time_offset if i < len(self.cams) else 0.0)
                      for i, h in enumerate(cams)]
        self._reset_3d = True

    def update_settings(self, new: dict) -> dict:
        allowed = {"height_m", "quest_ip", "cameras", "use_mac", "voice_alerts", "calib_seconds"} | TUNABLE
        old_cams = list(self.settings["cameras"])
        for k, v in new.items():
            if k in allowed:
                self.settings[k] = v
        self.save_settings()
        if list(self.settings["cameras"]) != old_cams:
            self._make_slots()
        for sl in self.slots[2:]:
            sl.max_rate = float(self.settings["extra_fps"])
        if self.out is not None and "quest_ip" in new:
            self.out.sender = VRChatSender(self.settings["quest_ip"], 9000)
        return self.settings

    def cam_name(self, s: CamSlot) -> str:
        n = friendly_name(s.host, s.info)
        if n and self.settings["names"].get(s.host) != n:
            self.settings["names"][s.host] = n
            self.save_settings()
        return n or self.settings["names"].get(s.host) or s.host

    # ---------------- 3D / tracking loop
    def _track_loop(self):
        prev, last_ref, n, t_fps = None, None, 0, time.perf_counter()
        while not self._quit.is_set():
            slots = self.slots
            if len(slots) < 2:
                time.sleep(0.05)
                continue
            if not slots[0].new.wait(0.01):
                slots[1].new.wait(0.01)
            for s in slots:
                s.new.clear()
            stab = self.stab
            if stab is not None:     # live tuning from the settings window
                stab.euro.min_cutoff = max(0.05, float(self.settings["smooth_hz"]))
                pm = float(self.settings["predict_ms"])
                if pm > 0 and stab.pred is None:
                    stab.pred = MotionPredictor(pm / 1000)
                elif pm <= 0:
                    stab.pred = None
                elif stab.pred is not None:
                    stab.pred.h = pm / 1000
            if time.perf_counter() - t_fps > 1:
                self.fps3d, n, t_fps = n / (time.perf_counter() - t_fps), 0, time.perf_counter()
            with self.lock:
                cams, thr, vw = self.cams, self.view_thr, self.view_w
                if self._reset_3d:
                    prev, last_ref, self._reset_3d = None, None, False
                    if self.stab is not None:
                        self.stab = self._new_stab()
            if len(cams) < 2 or not (slots[0].online and slots[1].online):
                continue
            la, lb = slots[0].latest_pose(), slots[1].latest_pose()
            if la is None or lb is None:
                continue
            behind = 0 if la[0] <= lb[0] else 1
            t_ref = (la if behind == 0 else lb)[0]
            if last_ref is not None and t_ref <= last_ref:
                continue
            last_ref = t_ref
            pa = la[1] if behind == 0 else slots[0].pose_at(t_ref)
            pb = slots[1].pose_at(t_ref) if behind == 0 else lb[1]
            if pa is None or pb is None:
                self.skips["no synced pair"] += 1
                continue
            extra = []
            gap = 2.5 / max(self.settings["extra_fps"], 1)
            for i in range(2, len(cams)):
                s = slots[i] if i < len(slots) else None
                ok = s is not None and s.online and self.settings["use_mac"]
                extra.append(s.pose_at(t_ref, max_dt=0.02, max_gap=gap) if ok else None)
            res = triangulate_with_swap_check(cams, [pa, pb] + extra, prev=prev,
                                              max_err_px=thr, view_weight=vw)
            # Both phones clearly see a person but the torso does not triangulate:
            # the cameras were probably moved since the calibration.
            if (pa.scores > 0.5).sum() >= 10 and (pb.scores > 0.5).sum() >= 10:
                self._agree.append(bool(res.valid[CORE].all()))
            if not res.valid[CORE].all():
                self.skips["torso not triangulated"] += 1
                continue
            prev = res
            n += 1
            now = time.perf_counter()
            stab, out = self.stab, self.out
            S = stab(res.points, res.valid, t_ref) if stab is not None else None
            with self.lock:
                self.skeleton = (S if S is not None else res.points).copy()
                self.skeleton_valid = np.ones(26, bool) if S is not None else res.valid.copy()
                self.visible_t = now
            if out is not None and S is not None and stab.fit.ready:
                try:
                    out.step(S, res, t_ref)
                except Exception as e:   # never stop tracking for one bad frame
                    self.skips[f"send error {type(e).__name__}"] += 1

    def _new_stab(self):
        st = SkeletonStabilizer(min_cutoff=float(self.settings["smooth_hz"]), beta=0.4, floor_clip=True,
                                predict_gated=float(self.settings["predict_ms"]) / 1000)
        bones = self.calib_meta.get("bones") if self.calib_meta else None
        if bones and len(bones) == 26:
            st.fit.lengths = np.array([np.nan if b is None else b for b in bones], float)
        return st

    # ---------------- server mode
    def start_server(self) -> dict:
        if len(self.cams) < 2:
            return {"error": "Спочатку відкалібруй камери"}
        if self.calib["state"] in ("countdown", "recording", "computing"):
            return {"error": "Зачекай, йде калібрування"}
        with self.lock:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self._rotate_logs()
            self.out = TrackerOutput(VRChatSender(self.settings["quest_ip"], 9000), self.settings,
                                     f"{self.d}/logs/session_{stamp}.jsonl", align=self.align, hmd=self.hmd)
            self._statsf = open(f"{self.d}/logs/session_{stamp}.log", "w")
            self.hmd.logf = open(f"{self.d}/logs/session_{stamp}_hmd.jsonl", "w")
            self.stab = self._new_stab()      # learn bone lengths fresh, standing in the play area
            self.mode, self.server_t0 = "server", time.perf_counter()
        try:   # keep the display awake while the server runs
            self._caffeinate = subprocess.Popen(["caffeinate", "-d", "-i", "-w", str(os.getpid())])
        except OSError:
            self._caffeinate = None
        return {"ok": True}

    def stop_server(self) -> dict:
        with self.lock:
            self.mode = "setup"
            if self.out:
                self.out.close()
            self.out, self.stab, self.server_t0 = None, None, None
            if self._statsf:
                self._statsf.close()
                self._statsf = None
            f, self.hmd.logf = self.hmd.logf, None
            if f:
                f.close()
        if self._caffeinate:
            self._caffeinate.terminate()
            self._caffeinate = None
        return {"ok": True}

    def _rotate_logs(self):
        files = sorted(p for p in glob.glob(f"{self.d}/logs/session_*.jsonl") if not p.endswith("_hmd.jsonl"))
        for p in files[:max(0, len(files) - int(self.settings["keep_logs"]) + 1)]:
            for q in (p, p[:-6] + ".log", p[:-6] + "_hmd.jsonl"):
                try:
                    os.remove(q)
                except OSError:
                    pass

    # ---------------- background: rates, voice alerts, stats, headset ping
    def _monitor_loop(self):
        prev = {}
        was_online: dict = {}
        low_said: set = set()
        t_stats = time.perf_counter()
        while not self._quit.is_set():
            time.sleep(1.0)
            now = time.perf_counter()
            for s in self.slots:
                p = prev.get(id(s))
                if p is not None:
                    dt = now - p[0]
                    s.rate_pose, s.rate_in = (s.n_pose - p[1]) / dt, (s.n_frames - p[2]) / dt
                prev[id(s)] = (now, s.n_pose, s.n_frames)
                # Voice alerts while playing (the screen is not visible from inside the headset).
                key = s.host
                if self.mode == "server" and self.settings["voice_alerts"]:
                    name = self.cam_name(s)
                    if was_online.get(key) and not s.online and s.offline_since and now - s.offline_since > 3:
                        say(f"{name} відключився")
                        was_online[key] = False
                    elif was_online.get(key) is False and s.online:
                        say(f"{name} знову на зв'язку")
                        was_online[key] = True
                    elif key not in was_online:
                        was_online[key] = s.online
                    st = s.status()
                    if st.get("battery", 100) >= 0 and st.get("battery", 100) <= 15 and not st.get("charging"):
                        if key not in low_said:
                            low_said.add(key)
                            say(f"{name}: низький заряд")
                    elif st.get("battery", 0) > 20:
                        low_said.discard(key)
                else:
                    was_online[key] = s.online
            mb = self.mac_battery
            if self.mode == "server" and self.settings["voice_alerts"] and mb and not mb["charging"]:
                if mb["percent"] <= 15 and not getattr(self, "_mac_low_said", False):
                    self._mac_low_said = True
                    say("MacBook майже розряджений, підключи зарядку")
            elif mb and mb["charging"]:
                self._mac_low_said = False
            if self._statsf and now - t_stats >= 5:
                t_stats = now
                cams = " | ".join(f"cam{s.index}: {s.state} in {s.rate_in:4.1f} pose {s.rate_pose:4.1f}/s "
                                  f"{s.pose_ms:3.0f} ms lat "
                                  f"{np.median(s.src.latency_ms[-30:]) if s.src and s.src.latency_ms else 0:3.0f}"
                                  for s in self.slots)
                try:
                    self._statsf.write(f"{time.strftime('%H:%M:%S')} 3D {self.fps3d:4.1f} fps | {cams} | "
                                       f"thermal {self.thermal} | headset {self.headset_online} | "
                                       f"hmd pose {self.hmd.rate:4.1f}/s (queries {self.hmd.queries}, datagrams {self.hmd.datagrams}, re-adv {self.hmd.readvertised}) | "
                                       f"mac battery {self.mac_battery} | "
                                       f"skips {dict(self.skips)}\n")
                    self._statsf.flush()
                except (OSError, ValueError):
                    pass

    def _mac_battery(self):
        """{'percent': 57, 'charging': False} from pmset, or None on a desktop Mac."""
        try:
            out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=3).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
        import re
        m = re.search(r"(\d+)%;\s*([a-zA-Z ]+);", out)
        if not m:
            return None
        return {"percent": int(m.group(1)), "charging": "AC Power" in out}

    def _align_loop(self):
        while not self._quit.wait(0.5):
            try:
                self.align.maybe_fit()
            except Exception as e:          # a failed fit must not stop the loop
                print(f"align: fit failed: {e!r}", flush=True)

    def _ping_loop(self):
        n = 0
        while not self._quit.is_set():
            if n % 10 == 0:
                self.mac_battery = self._mac_battery()
            n += 1
            ip = self.settings.get("quest_ip")
            if ip:
                try:
                    r = subprocess.run(["ping", "-c", "1", "-W", "1000", ip], capture_output=True, timeout=3)
                    self.headset_online = r.returncode == 0
                except (OSError, subprocess.TimeoutExpired):
                    self.headset_online = False
            self._quit.wait(3.0)

    # ---------------- calibration
    def start_calibration(self) -> dict:
        if self.calib["state"] in ("countdown", "recording", "computing"):
            return {"error": "Калібрування вже йде"}
        if self.mode != "setup":
            return {"error": "Спочатку зупини гру"}
        off = [self.cam_name(s) for s in self.slots[:2] if not s.online]
        if off:
            return {"error": f"Не в мережі: {', '.join(off)}"}
        self._cancel.clear()
        secs = float(self.settings["calib_seconds"])
        self.calib = {"state": "countdown", "t_start": time.time() + 3, "seconds": secs}
        threading.Thread(target=self._calibrate, args=(secs,), daemon=True).start()
        return {"ok": True}

    def cancel_calibration(self):
        self._cancel.set()

    def dismiss_calibration(self):
        if self.calib["state"] in ("done", "failed"):
            self.calib = {"state": "idle"}

    def _wait_until(self, t_wall: float, check=None):
        while time.time() < t_wall:
            if self._cancel.is_set():
                raise Cancelled()
            if check:
                check()
            time.sleep(0.05)

    def _calibrate(self, seconds: float):
        slots = [s for s in self.slots if s.online]
        recorders = []
        try:
            self._wait_until(self.calib["t_start"])
            out_dir = time.strftime(f"{self.d}/rec_%Y%m%d_%H%M%S")
            os.makedirs(out_dir, exist_ok=True)
            for s in slots:
                s.rec = []
            recorders = [VideoRecorder(s, out_dir) for s in slots]
            t_end = time.time() + seconds
            self.calib = {"state": "recording", "t_end": t_end, "seconds": seconds}

            def phones_ok():
                for s in self.slots[:2]:
                    if not s.online:
                        raise RuntimeError(f"{self.cam_name(s)} відключився під час запису")
            self._wait_until(t_end, phones_ok)
            data = {s.index: (s.rec, s.rec_size) for s in slots}
            for s in slots:
                s.rec = None
            for r in recorders:
                r.halt.set()
            for r in recorders:
                r.join(timeout=5)
            self.calib = {"state": "computing"}
            info = {f"cam{r.slot.index}": r.info for r in recorders}
            with open(f"{out_dir}/info.json", "w") as fh:
                json.dump(info, fh, indent=2)
            tracks = {}
            for i, (rec, size) in data.items():
                if len(rec) < 30 or size is None:
                    continue
                t = np.array([r[0] for r in rec])
                kp = np.array([r[1] for r in rec])
                sc = np.array([r[2] for r in rec])
                tracks[i] = PoseTrack(t, kp, sc, size[0], size[1], info.get(f"cam{i}", {}))
                np.savez(f"{out_dir}/poses_live_cam{i}.npz", t=t, kp=kp, sc=sc, wh=np.array(size))
            if 0 not in tracks or 1 not in tracks:
                raise RuntimeError("only 0 synchronised frames")
            lines = []
            lg = lambda m: lines.append(m)
            cams, meta = calibrate_pair(tracks[0], tracks[1], float(self.settings["height_m"]), log=lg)
            hosts = [self.slots[0].host, self.slots[1].host]
            result = {"phones_px": meta["reproj_median_px"], "baseline_m": meta["baseline_m"],
                      "angle_deg": meta["angle_deg"], "mac": None, "mac_error": None}
            if meta["reproj_median_px"] > 3.0:
                raise RuntimeError(f"bad phones {meta['reproj_median_px']:.1f}")
            # Bone lengths from these 30 s (whole body in view, moving): the game starts with
            # them instead of learning from whatever the cameras see right after "start".
            ts_all, skel_all = triangulate_tracks(cams, tracks[0], tracks[1])
            bones = SkeletonFit.lengths_from(skel_all)
            meta["bones"] = [None if not np.isfinite(b) else round(float(b), 4) for b in bones]
            meta["extra"] = {}
            for i in sorted(tracks):
                if i < 2:
                    continue
                ts, skel = triangulate_tracks(cams, tracks[0], tracks[1])
                tr = tracks[i]
                rng = tr.info.get("focal_range_px") or (mac_focal_range(tr.width) if self.slots[i].is_mac else None)
                try:
                    cam, m = calibrate_extra(tr, ts, skel, name=f"cam{i}", focal_range=tuple(rng) if rng else None, log=lg)
                except RuntimeError as e:
                    result["mac_error"] = "Mac майже не бачив тебе" if "usable points" in str(e) else str(e)
                    continue
                if m["reproj_median_px"] > 30:
                    result["mac_error"] = f"Mac розташовано неточно ({m['reproj_median_px']:.0f} px)"
                    continue
                cams.append(cam)
                hosts.append(self.slots[i].host)
                meta["extra"][f"cam{i}"] = m
                result["mac"] = {"px": m["reproj_median_px"]}
            meta.update({"recording": out_dir, "height_m": float(self.settings["height_m"]), "source": "app (live poses)"})
            if os.path.exists(self.calib_path):
                shutil.copy(self.calib_path, f"{self.d}/calibration_prev.json")
            save_calibration(self.calib_path, cams, hosts, meta)
            with open(f"{out_dir}/calibration.log", "w") as fh:
                fh.write("\n".join(lines) + "\n")
            self._apply_calibration(cams, hosts, meta, time.time())
            self.calib = {"state": "done", "result": {**result, "quality": quality(meta)}}
        except Cancelled:
            self.calib = {"state": "idle"}
        except Exception as e:
            self.calib = {"state": "failed", "message": human_error(e)}
        finally:
            for s in slots:
                s.rec = None
            for r in recorders:
                r.halt.set()

    # ---------------- snapshot for the UI
    def snapshot(self, with_skeleton: bool = True) -> dict:
        now = time.perf_counter()
        cams = []
        for s in self.slots:
            st = s.status()
            cams.append({
                "name": self.cam_name(s), "host": s.host, "state": s.state, "error": s.error,
                "mac": s.is_mac or (s.index >= 2 and not s.online),
                "battery": st.get("battery"), "charging": st.get("charging"),
                "hot": bool(st and (st.get("thermal", -1) >= 2 or st.get("temp_c", 0) >= 42)),
                "person": now - s.person_t < 1.0, "fps": round(s.rate_pose, 1),
                "calibrated": s.index < len(self.cams),
            })
        out = self.out
        snap = {
            "mode": self.mode, "cams": cams, "headset": self.headset_online, "thermal": self.thermal,
            "hmd": {"ok": self.hmd.fresh, "rate": round(self.hmd.rate), "queries": self.hmd.queries,
                    "datagrams": self.hmd.datagrams},
            "mac_battery": self.mac_battery,
            "calibrated": len(self.cams) >= 2,
            "calib_info": {"time": self.calib_time, "quality": quality(self.calib_meta) if self.cams else None,
                           "n": len(self.cams)},
            "calib": self._calib_view(),
            "tracking": {
                "fps": round(self.fps3d, 1), "visible": now - self.visible_t < 1.0,
                "bones_ready": bool(self.stab is not None and self.stab.fit.ready),
                "learn": (len(self.stab.fit.samples) / self.stab.fit.learn_frames) if self.stab is not None else 0,
                "yaw_ago": (now - out.yaw_sent_t) if out is not None and out.yaw_sent_t else None,
                "uptime": (now - self.server_t0) if self.server_t0 else 0,
                "mismatch": len(self._agree) >= 90 and sum(self._agree) < 0.5 * len(self._agree),
                "anchor": out.anchor if out is not None else None,
                "align_ready": bool(self.align.ready),
                "align_lost": bool(self.align.lost),
            },
            "settings": {k: self.settings[k] for k in ("height_m", "quest_ip", "cameras", "use_mac", "voice_alerts", *TUNABLE)},
            "defaults": {k: DEFAULTS[k] for k in TUNABLE},
        }
        if with_skeleton:
            with self.lock:
                sk, va = self.skeleton, self.skeleton_valid
                fresh = now - self.visible_t < 1.0
            snap["skeleton"] = sk.round(3).tolist() if sk is not None and fresh else None
            snap["valid"] = va.astype(int).tolist() if va is not None and fresh else None
            snap["cam_poses"] = [{"c": c.center.round(3).tolist(), "f": c.R[2].round(3).tolist(),
                                  "x": c.R[0].round(3).tolist()} for c in self.cams]
        return snap

    def _calib_view(self) -> dict:
        c = dict(self.calib)
        if "t_start" in c:
            c["left"] = max(0.0, c["t_start"] - time.time())
        if "t_end" in c:
            c["left"] = max(0.0, c["t_end"] - time.time())
        return c

    def shutdown(self):
        self.stop_server()
        self._quit.set()
        self.hmd.close()
        for s in self.slots:
            s.stop()


def quality(meta: dict) -> str | None:
    px = meta.get("reproj_median_px") if meta else None
    if px is None:
        return None
    # Pixel error relative to the image: the low-res phone modes have half the pixels.
    return "відмінна" if px <= 1.5 else "добра" if px <= 3.0 else "погана"


def human_error(e: Exception) -> str:
    m = str(e)
    if "synchronised frames" in m:
        return "Телефони майже не бачили тебе одночасно. Стань так, щоб тебе бачили обидва, і спробуй ще."
    if "full-body" in m:
        return "Потрібно, щоб тебе було видно повністю — від голови до ніг."
    if "bad phones" in m:
        return "Калібрування вийшло неточним. Рухайся в центрі, де тебе бачать обидва телефони, і спробуй ще."
    if "відключився" in m:
        return m
    return f"Не вдалося: {m}"
