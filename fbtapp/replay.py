"""Replays of past games for the FBT app's "Записи" page.

Every game writes data/logs/session_<stamp>.jsonl: one line per 3D frame (~30/s) with the
stabilised skeleton (`stab`, our world: metres, y up). This turns one into a compact binary
the page plays back (float32 times + int16 millimetres, ~25 MB for 80 min instead of
~400 MB of JSON), cached next to the log as session_<stamp>.replay and built once
(a few seconds for a long game). The timeline marks come from the same log:

  posture per second  standing / sitting or crouching / lying (torso tilt, head height)
  activity per second how fast the feet moved
  realign             the trackers were re-aligned to the headset (a T-pose)
  freeze              no 3D frame for over 0.4 s (a camera or Wi-Fi hiccup)
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re
import struct
import threading

import numpy as np

from fbt.skeleton import BONE_IDX, HALPE26, IDX

MAGIC = b"FBTR1\n"
_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


def _stamp(name: str) -> dt.datetime | None:
    m = re.match(r"session_(\d{8}_\d{6})$", name)
    return dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S") if m else None


def _first_last_t(path: str) -> tuple[float, float] | None:
    try:
        with open(path, "rb") as f:
            first = f.readline()
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 200_000))
            tail = f.read().splitlines()
        t0 = json.loads(first)["t"]
        for line in reversed(tail):
            try:
                return t0, json.loads(line)["t"]
            except ValueError:
                continue
    except (OSError, ValueError, KeyError):
        pass
    return None


def list_sessions(data_dir: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(f"{data_dir}/logs/session_*.jsonl"), reverse=True):
        name = os.path.basename(p)[:-6]
        st = _stamp(name)
        if st is None:
            continue
        ft = _first_last_t(p)
        if ft is None or ft[1] - ft[0] < 10:
            continue                                   # empty or a few seconds: nothing to watch
        out.append({"name": name, "start": st.isoformat(), "duration_s": round(ft[1] - ft[0], 1),
                    "size_mb": round(os.path.getsize(p) / 1e6, 1),
                    "ready": os.path.exists(f"{data_dir}/logs/{name}.replay")})
    return out


def _cameras(data_dir: str, name: str) -> list[dict]:
    """Camera positions for the game: its own calibration copy if the app saved one, else the
    newest calibration whose recording was made before the game started."""
    own = f"{data_dir}/logs/{name}_calib.json"
    cands = [own] if os.path.exists(own) else []
    if not cands:
        st = _stamp(name)
        best = None
        for p in glob.glob(f"{data_dir}/calibration*.json"):
            try:
                rec = json.load(open(p))["meta"].get("recording", "")
            except (OSError, ValueError, KeyError):
                continue
            m = re.search(r"rec_(\d{8}_\d{6})", rec)
            if not m:
                continue
            rt = dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
            if st and rt <= st and (st - rt).total_seconds() < 3600 and (best is None or rt > best[0]):
                best = (rt, p)
        cands = [best[1]] if best else []
    out = []
    for p in cands:
        try:
            cams = json.load(open(p))["cameras"]
        except (OSError, ValueError, KeyError):
            continue
        for c in cams:
            R, t = np.array(c["R"], float), np.array(c["t"], float).reshape(3)
            out.append({"name": c.get("name", ""), "c": (-R.T @ t).round(3).tolist(),
                        "f": (R.T @ [0, 0, 1.0]).round(3).tolist()})
    return out


def build(data_dir: str, name: str) -> str:
    """Make (or reuse) the .replay file of one game; returns its path."""
    if not re.match(r"session_\d{8}_\d{6}$", name):
        raise ValueError("bad session name")
    src = f"{data_dir}/logs/{name}.jsonl"
    dst = f"{data_dir}/logs/{name}.replay"
    with _guard:
        lock = _locks.setdefault(name, threading.Lock())
    with lock:
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            return dst
        T, P = [], []
        realign, prev_al = [], None
        with open(src) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    S = np.asarray(d["stab"], np.float32)
                except (ValueError, KeyError):
                    continue
                if S.shape != (len(HALPE26), 3) or not np.isfinite(S).all():
                    continue
                T.append(d["t"]); P.append(S)
                al = d.get("align")
                if al is not None and prev_al is not None:
                    dyaw = abs((np.degrees(al[0] - prev_al[0]) + 180) % 360 - 180)
                    dpos = float(np.linalg.norm(np.subtract(al[1:4], prev_al[1:4])))
                    if dyaw > 3 or dpos > 0.08:
                        realign.append(len(T) - 1)
                prev_al = al
        if not T:
            raise ValueError("no frames")
        t = np.asarray(T, np.float64)
        t -= t[0]
        P = np.stack(P)
        n = len(t)

        # per-second posture and activity
        secs = int(np.ceil(t[-1])) + 1
        sec = np.minimum(t.astype(int), secs - 1)
        spine = P[:, IDX["neck"]] - P[:, IDX["hip"]]
        tilt = np.degrees(np.arccos(np.clip(spine[:, 1] / (np.linalg.norm(spine, axis=1) + 1e-9), -1, 1)))
        head_y = P[:, IDX["head"]][:, 1]
        stand_h = float(np.percentile(head_y, 95))
        feet = P[:, [IDX["left_ankle"], IDX["right_ankle"]]]
        speed = np.zeros(n)
        speed[1:] = np.linalg.norm(np.diff(feet, axis=0), axis=2).mean(1) / np.maximum(np.diff(t), 1e-3)
        speed = np.minimum(speed, 5.0)
        posture = np.zeros(secs, np.uint8)       # 0 none, 1 stand, 2 sit/crouch, 3 lying
        activity = np.zeros(secs, np.float32)
        cnt = np.bincount(sec, minlength=secs)
        tl = np.bincount(sec, tilt, secs) / np.maximum(cnt, 1)
        hy = np.bincount(sec, head_y, secs) / np.maximum(cnt, 1)
        activity[:] = np.bincount(sec, speed, secs) / np.maximum(cnt, 1)
        has = cnt > 0
        posture[has] = np.where(tl[has] > 55, 3, np.where(hy[has] < 0.82 * stand_h, 2, 1))
        dts = np.diff(t)
        freezes = [[round(float(t[i]), 2), round(float(dts[i]), 2)] for i in np.where(dts > 0.4)[0]]

        st = _stamp(name)
        header = {
            "name": name, "start": st.isoformat() if st else None, "n": n, "duration_s": float(t[-1]),
            "joints": HALPE26, "bones": BONE_IDX, "cameras": _cameras(data_dir, name),
            "posture": posture.tolist(), "activity": np.round(activity, 3).tolist(),
            "realign": [round(float(t[i]), 2) for i in realign], "freezes": freezes,
            "stand_head_m": round(stand_h, 3),
        }
        hb = json.dumps(header, ensure_ascii=False).encode()
        pts = np.clip(np.round(P * 1000), -32767, 32767).astype("<i2")
        tmp = dst + ".tmp"
        with open(tmp, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<I", len(hb)))
            f.write(hb)
            f.write(b"\0" * ((-len(MAGIC) - 4 - len(hb)) % 4))      # align the float32 block
            f.write(t.astype("<f4").tobytes())
            f.write(pts.tobytes())
        os.replace(tmp, dst)
        return dst
