"""Align our camera world frame with the Quest playspace, by the user.

VRChat on Quest does not apply OSC head alignment for us, and the
Auto-center OSC Trackers button is not available, so trackers end up in our
raw coordinates (origin under the user at calibration time, arbitrary yaw).

Quest's own recenter (hold the Meta button) moves the playspace origin under
the headset and points "forward" where the user looks. If at that same moment
we take the user's head position and body facing as OUR origin and forward,
both frames coincide. The result is persisted, because a Quest recenter also
persists until the next one.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .skeleton import IDX


def body_facing(P: np.ndarray) -> np.ndarray:
    """Horizontal forward of the torso in our right-handed, Y-up world."""
    left = (P[IDX["left_hip"]] - P[IDX["right_hip"]]) + (P[IDX["left_shoulder"]] - P[IDX["right_shoulder"]])
    f = np.cross(left, [0.0, 1.0, 0.0])   # left = up x fwd  =>  fwd = left x up
    f[1] = 0
    return f / (np.linalg.norm(f) + 1e-9)


def head_root(P: np.ndarray) -> np.ndarray:
    spine = P[IDX["neck"]] - P[IDX["hip"]]
    return P[IDX["neck"]] + 0.10 * spine / (np.linalg.norm(spine) + 1e-9)


class SpaceAligner:
    def __init__(self, path: str = "data/space_align.json"):
        self.path = path
        self.origin = np.zeros(3)
        self.R = np.eye(3)
        self.active = False
        if os.path.exists(path):
            d = json.load(open(path))
            self.origin, self.R, self.active = np.array(d["origin"]), np.array(d["R"]), True

    def zero(self, frames: list[np.ndarray]):
        """frames: recent full skeletons (user standing still, looking forward)."""
        P = np.mean(frames, axis=0)
        f = body_facing(P)
        o = head_root(P)
        self.origin = np.array([o[0], 0.0, o[2]])  # floor stays at y=0
        # Rotate about Y so our facing maps to (0,0,-1), which to_unity() turns
        # into Unity's forward (0,0,+1) = the direction Quest calls forward.
        phi = np.pi - np.arctan2(f[0], f[2])
        c, s = np.cos(phi), np.sin(phi)
        self.R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        self.active = True
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        json.dump({"origin": self.origin.tolist(), "R": self.R.tolist()}, open(self.path, "w"), indent=2)

    def __call__(self, P: np.ndarray) -> np.ndarray:
        if not self.active:
            return P
        return (P - self.origin) @ self.R.T
