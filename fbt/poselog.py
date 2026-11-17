"""What every camera saw during a game: its 2D pose at each 3D step, including the steps that
gave no 3D frame (no synced pair, torso not triangulated). With it a game can be replayed
offline from the cameras to VRChat, the way the headset link was (ARCHITECTURE.md §29, §32).

File: MAGIC, u32 header length, JSON header, then fixed-size records (RECORD dtype).
The record's t is the same t_ref as the "t" of the session .jsonl line, when there is one.
Writing costs a few microseconds per step and never raises into the tracking loop.
"""

from __future__ import annotations

import json
import struct

import numpy as np

MAGIC = b"FBTP1\n"
FRAME3D, NO_TORSO, NO_PAIR = 0, 1, 2       # what came of the step
OUTCOMES = {"3d frame": FRAME3D, "torso not triangulated": NO_TORSO, "no synced pair": NO_PAIR}
J = 26


def record_dtype(n_cams: int) -> np.dtype:
    return np.dtype([("t", "<f8"), ("outcome", "u1"), ("present", "u1", (n_cams,)),
                     ("kp", "<f4", (n_cams, J, 2)), ("sc", "<f4", (n_cams, J))])


class PoseLog:
    def __init__(self, path: str, n_cams: int, cams: list[str]):
        self.n = n_cams
        self.dtype = record_dtype(n_cams)
        self._rec = np.zeros(1, self.dtype)
        self.f = open(path, "wb")
        head = json.dumps({"joints": J, "cams": cams, "outcomes": OUTCOMES}).encode()
        self.f.write(MAGIC + struct.pack("<I", len(head)) + head)

    def write(self, t: float, outcome: int, poses: list) -> None:
        """poses: one Pose2D or None per calibrated camera, as passed to the triangulation."""
        try:
            r = self._rec
            r["t"], r["outcome"] = t, outcome
            r["present"] = 0
            r["kp"] = 0
            r["sc"] = 0
            for i, p in enumerate(poses[:self.n]):
                if p is not None:
                    r["present"][0, i] = 1
                    r["kp"][0, i] = p.keypoints
                    r["sc"][0, i] = p.scores
            self.f.write(r.tobytes())
        except (ValueError, OSError, TypeError):     # closed meanwhile (game stopped): drop it
            pass

    def close(self) -> None:
        try:
            self.f.close()
        except OSError:
            pass


def read(path: str) -> tuple[dict, np.ndarray]:
    """(header, records); records is a structured array (fields t, outcome, present, kp, sc).
    A record cut short at the end (the app was closed mid-write) is left out."""
    with open(path, "rb") as fh:
        data = fh.read()
    if not data.startswith(MAGIC):
        raise ValueError("not an FBT pose log")
    (n,) = struct.unpack_from("<I", data, len(MAGIC))
    start = len(MAGIC) + 4
    head = json.loads(data[start:start + n])
    dt = record_dtype(len(head["cams"]))
    body = data[start + n:]
    body = body[:len(body) - len(body) % dt.itemsize]
    return head, np.frombuffer(body, dt)
