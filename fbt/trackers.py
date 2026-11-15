"""3D skeleton -> VRChat OSC trackers.

Our world frame is right-handed (Y up). Unity/VRChat is left-handed (Y up),
so points are mirrored by negating Z; a mirror swaps handedness without
swapping the user's left and right.

Rotations are built in Unity space as [right, up, forward] frames per body
part and converted to Unity Euler angles (Z, then X, then Y), which is what
VRChat's /tracking/trackers/*/rotation expects.
"""

from __future__ import annotations

import numpy as np

from .osc_out import Tracker
from .skeleton import IDX

# Index -> body part. This is the order that worked in VRChat on the very
# first test (1 hip, 2 chest, 3/4 feet, 5/6 knees, 7/8 elbows — the same
# "VRChat indexed schema" used by vr-body-tracker). Switching to SlimeVR's
# order (2/3 feet, 4/5 thighs, 6 chest) broke the legs and lifted the body
# (camera inside the torso), consistent with VRChat assigning roles by index.
TRACKER_IDS = {"hip": "1", "chest": "2", "left_foot": "3", "right_foot": "4",
               "left_knee": "5", "right_knee": "6", "left_elbow": "7", "right_elbow": "8"}


def to_unity(P: np.ndarray) -> np.ndarray:
    Q = P.copy()
    Q[..., 2] *= -1
    return Q


def _n(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def frame(up, forward) -> np.ndarray:
    """Rotation matrix (columns right, up, forward), up kept exact."""
    u = _n(up)
    f = _n(forward - np.dot(forward, u) * u)
    r = np.cross(u, f)  # Unity: Cross(up, forward) = right
    return np.stack([r, u, f], axis=1)


def unity_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Unity eulerAngles (degrees) for R = Ry(y) Rx(x) Rz(z)."""
    x = np.degrees(np.arcsin(np.clip(-R[1, 2], -1, 1)))
    y = np.degrees(np.arctan2(R[0, 2], R[2, 2]))
    z = np.degrees(np.arctan2(R[1, 0], R[1, 1]))
    return float(x), float(y), float(z)


def solve(P_world: np.ndarray, enabled=("hip", "chest", "left_foot", "right_foot",
                                        "left_knee", "right_knee"),
          send_head: bool = False) -> list[Tracker]:
    P = to_unity(P_world)
    g = lambda n: P[IDX[n]]
    up_world = np.array([0, 1.0, 0])
    right = g("right_hip") - g("left_hip")
    spine = g("neck") - g("hip")
    fwd = np.cross(right, spine)  # Unity: Cross(right, up) = forward
    torso = frame(spine, fwd)
    facing = frame(up_world, fwd)

    out = []

    def add(name, pos, R):
        if name in enabled:
            out.append(Tracker(TRACKER_IDS[name], tuple(map(float, pos)), unity_euler(R)))

    add("hip", g("hip"), torso)
    chest_pos = g("hip") + 0.8 * (g("neck") - g("hip"))  # upper chest
    sh_right = g("right_shoulder") - g("left_shoulder")
    add("chest", chest_pos, frame(spine, np.cross(sh_right, spine)))
    for side in ("left", "right"):
        toe = (g(f"{side}_big_toe") + g(f"{side}_small_toe")) / 2
        foot_fwd = toe - g(f"{side}_heel")
        shin_up = g(f"{side}_knee") - g(f"{side}_ankle")
        add(f"{side}_foot", g(f"{side}_ankle"), frame(shin_up, foot_fwd))
        thigh = g(f"{side}_hip") - g(f"{side}_knee")
        # Knee forward: the bend direction when clearly bent, else the body's
        # facing (a straight leg has no stable bend direction).
        bis = (g(f"{side}_knee") - g(f"{side}_hip")) + (g(f"{side}_knee") - g(f"{side}_ankle"))
        bend = np.linalg.norm(bis)
        knee_fwd = _n(fwd) * max(0.0, 1 - bend / 0.15) + _n(bis) * min(1.0, bend / 0.15)
        add(f"{side}_knee", g(f"{side}_knee"), frame(thigh, knee_fwd))
        upper = g(f"{side}_shoulder") - g(f"{side}_elbow")
        bis = (g(f"{side}_elbow") - g(f"{side}_shoulder")) + (g(f"{side}_elbow") - g(f"{side}_wrist"))
        add(f"{side}_elbow", g(f"{side}_elbow"), frame(upper, bis if np.linalg.norm(bis) > 0.05 else -fwd))
    if send_head:
        out.append(head_tracker(P_world))
    return out


def is_tpose(P_world: np.ndarray) -> bool:
    """Arms stretched sideways at shoulder height (VRChat's calibration pose)."""
    g = lambda n: P_world[IDX[n]]
    sh = np.linalg.norm(g("left_shoulder") - g("right_shoulder"))
    span = np.linalg.norm(g("left_wrist") - g("right_wrist"))
    level = max(abs(g("left_wrist")[1] - g("left_shoulder")[1]),
                abs(g("right_wrist")[1] - g("right_shoulder")[1]))
    return span > 3.5 * sh and level < 0.15


def is_hands_up(P_world: np.ndarray) -> bool:
    """Both wrists clearly above the head: an easy, in-frame alignment gesture."""
    g = lambda n: P_world[IDX[n]]
    top = g("head")[1]
    return g("left_wrist")[1] > top + 0.05 and g("right_wrist")[1] > top + 0.05


def head_tracker(P_world: np.ndarray, source: str = "ears") -> Tracker:
    """The head as VRChat needs it for aligning our space to the headset.

    VRChat (docs: osc-trackers) uses it in two ways:
      position — every frame the whole OSC space is shifted so this point sits
                 on the avatar's head bone (root of the head, skull base);
      rotation — the whole OSC space yaw is slowly turned until THIS yaw
                 matches the headset's yaw.
    So the rotation must be the HEAD's yaw, not the body's: sending the torso
    facing made VRChat swing all trackers around the head whenever the user
    turned the head (observed in VRChat). We take the head direction from the
    ear-to-ear line (visible from behind too) and the head root just below
    the ears' midpoint. VRChat's slow yaw lerp absorbs the camera noise.
    """
    P = to_unity(P_world)
    g = lambda n: P[IDX[n]]
    spine = g("neck") - g("hip")
    up = np.array([0, 1.0, 0])
    if source == "ears":
        right = g("right_ear") - g("left_ear")
        root = (g("left_ear") + g("right_ear")) / 2 - 0.04 * _n(spine)
    else:  # torso (old behaviour, kept for comparison)
        right = g("right_hip") - g("left_hip")
        root = g("neck") + 0.10 * _n(spine)
    fwd = np.cross(right, up)  # Unity: Cross(right, up) = forward
    return Tracker("head", tuple(map(float, root)), unity_euler(frame(up, fwd)))


class RotationSmoother:
    """Per-tracker rotation smoothing. Foot/knee rotations come from short
    vectors (heel->toe) and are the noisiest output; positions are already
    smoothed upstream. Small changes are smoothed strongly, large ones
    (a real turn) pass quickly — the same idea as One-Euro, on angles."""

    def __init__(self, slow: float = 0.25, fast_deg: float = 30.0):
        self.slow, self.fast = slow, np.radians(fast_deg)
        self.R: dict[str, np.ndarray] = {}

    @staticmethod
    def _euler_to_R(e):
        x, y, z = np.radians(e)
        Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        return Ry @ Rx @ Rz

    def __call__(self, trackers: list[Tracker]) -> list[Tracker]:
        out = []
        for t in trackers:
            R = self._euler_to_R(t.rotation)
            prev = self.R.get(t.id)
            if prev is not None:
                ang = np.arccos(np.clip((np.trace(prev.T @ R) - 1) / 2, -1, 1))
                a = float(np.clip(self.slow + (1 - self.slow) * ang / self.fast, self.slow, 1.0))
                M = prev * (1 - a) + R * a
                U, _, Vt = np.linalg.svd(M)
                R = U @ Vt
            self.R[t.id] = R
            out.append(Tracker(t.id, t.position, unity_euler(R)))
        return out
