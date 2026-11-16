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


def frame_fwd(forward, up_hint) -> np.ndarray:
    """Like frame(), but keeps FORWARD exact (up is orthogonalised to it)."""
    f = _n(forward)
    u = _n(up_hint - np.dot(up_hint, f) * f)
    r = np.cross(u, f)
    return np.stack([r, u, f], axis=1)


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


def _yaw_of(v):
    return np.arctan2(v[0], v[2])


def _ang(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def foot_frame(heel, toe, body_fwd, prev_yaw=None, max_pitch_deg=15.0, st=None):
    """Robust foot orientation (Unity space), built from measurements on our
    session logs (heel/toe keypoints sometimes make the foot near-vertical,
    pitch up to +-88 deg, so yaw becomes arbitrary -> visible 360 deg spins):
      1. yaw candidates from the HORIZONTAL heel->toe line: a and a+180
         (heel/toe can be swapped by the model); pick the one closer to the
         previous frame. Near-vertical foot (<6 cm horizontal) -> keep previous.
      2. slow sanity check vs the pelvis facing: only after ~1 s of the foot
         pointing backwards relative to the hips do we switch candidates
         (the pelvis direction itself glitches for single frames).
      3. rate limit 20 deg/frame (~600 deg/s): glitches become small blips.
      4. pitch measured along the FINAL forward direction, clamped; roll = 0.
    st: per-foot dict (state for step 2). Returns (rotation, yaw)."""
    f = toe - heel
    horiz = np.array([f[0], 0.0, f[2]])
    hl = np.linalg.norm(horiz)
    body_yaw = _yaw_of(body_fwd)
    ref = prev_yaw if prev_yaw is not None else body_yaw
    if hl < 0.06:
        yaw = ref
    else:
        a = _yaw_of(horiz)
        cands = (a, _ang(a + np.pi))
        yaw = min(cands, key=lambda c: abs(_ang(c - ref)))
        if prev_yaw is not None and abs(_ang(yaw - prev_yaw)) > np.radians(60):
            # Both readings are ~sideways to the last known direction: heel and
            # toe were detected side by side (seen on the left foot in our logs).
            # Unreliable this frame; real turns still get through gradually via
            # the slow check below and later frames.
            st_skip = st.get("skip", 0) + 1 if st is not None else 1
            if st is not None:
                st["skip"] = st_skip
            if st_skip <= 15:
                yaw = prev_yaw
            elif st is not None:
                st["skip"] = 0
        elif st is not None:
            st["skip"] = 0
        if st is not None:
            back = abs(_ang(yaw - body_yaw)) > np.radians(120)
            st["back"] = st.get("back", 0) + 1 if back else 0
            if st["back"] > 30:
                yaw = cands[1] if yaw == cands[0] else cands[0]
                st["back"] = 0
        if prev_yaw is not None:
            # Adaptive smoothing (One-Euro idea on the angle): the heel->toe line
            # is short, so a few cm of keypoint noise = 15-20 deg of yaw noise.
            # Small changes are mostly kept out, big real turns pass quickly.
            d = _ang(yaw - prev_yaw)
            k = min(1.0, 0.12 + abs(d) / np.radians(60))
            d *= k
            lim = np.radians(20)
            if abs(d) > lim:
                d = np.sign(d) * lim
            yaw = _ang(prev_yaw + d)
    u = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
    if hl >= 0.06:
        sgn = 1.0 if np.dot(horiz, u) >= 0 else -1.0
        pitch = float(np.clip(np.arctan2(sgn * f[1], hl), -np.radians(max_pitch_deg),
                              np.radians(max_pitch_deg)))
    else:
        pitch = 0.0
    fwd = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    right = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])
    up = np.cross(fwd, right)                    # left-handed: up = fwd x right
    return np.stack([right, up, fwd], axis=1), yaw


def solve(P_world: np.ndarray, enabled=("hip", "chest", "left_foot", "right_foot",
                                        "left_knee", "right_knee"),
          send_head: bool = False, state: dict | None = None,
          foot_mode: str = "exact") -> list[Tracker]:
    """state: optional dict kept between frames (foot yaw continuity).
    foot_mode: "exact" (default, chosen in VRChat: livelier) keeps heel->toe exactly; "flat" = foot forced perpendicular to the shin (stable, the
    tested default; ankle flex/extension is lost); "exact" = heel->toe kept
    exactly, so pointed toes (ballet) and flexed feet reach VRChat, at the
    price of more jitter from noisy heel/toe keypoints."""
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
        # Foot rotation: the version that tested best in VRChat (shin as "up",
        # heel->toe as "forward"). foot_frame() is kept but not used: on real
        # play it made the feet worse.
        toe = (g(f"{side}_big_toe") + g(f"{side}_small_toe")) / 2
        foot_fwd = toe - g(f"{side}_heel")
        shin_up = g(f"{side}_knee") - g(f"{side}_ankle")
        Rf = frame_fwd(foot_fwd, shin_up) if foot_mode == "exact" else frame(shin_up, foot_fwd)
        add(f"{side}_foot", g(f"{side}_ankle"), Rf)
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


def is_tpose(P_world: np.ndarray, max_tilt_deg: float = 30.0) -> bool:
    """Arms stretched sideways at shoulder height (VRChat's calibration pose), standing
    or sitting upright: wrists at least 85% of the full arm span apart (the span measured
    on this same skeleton, so it does not depend on body proportions), level with the
    shoulders, and the torso within 30 deg of vertical. Lying on the floor with the arms
    spread is not one: there the torso is ~90 deg off (a T-pose standing: 3-8 deg), and
    taking it re-aligned the trackers to a wrong facing (ARCHITECTURE.md §27)."""
    g = lambda n: P_world[IDX[n]]
    spine = g("neck") - g("hip")
    if spine[1] < np.cos(np.radians(max_tilt_deg)) * np.linalg.norm(spine):
        return False
    sh = np.linalg.norm(g("left_shoulder") - g("right_shoulder"))
    arms = sum(np.linalg.norm(g(f"{s}_shoulder") - g(f"{s}_elbow")) + np.linalg.norm(g(f"{s}_elbow") - g(f"{s}_wrist"))
               for s in ("left", "right"))
    span = np.linalg.norm(g("left_wrist") - g("right_wrist"))
    level = max(abs(g("left_wrist")[1] - g("left_shoulder")[1]),
                abs(g("right_wrist")[1] - g("right_shoulder")[1]))
    return span > 0.85 * (sh + arms) and level < 0.15


def head_tracker(P_world: np.ndarray, source: str = "ears", forward: float = 0.0) -> Tracker:
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
    # Tunable horizontal shift of the anchor along the facing direction. Moving
    # the head point BACK makes VRChat place all trackers further FORWARD.
    root = root + forward * _n(np.array([fwd[0], 0.0, fwd[2]]))
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
