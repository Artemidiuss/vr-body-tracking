"""Synthetic test rig (ARCHITECTURE.md §9.1 level A).

A procedural Halpe26 skeleton performs scripted motions (idle, walking,
squats, a full turn, kicks, side steps). Its 3D joints are the ground truth;
virtual cameras project them to 2D, and a noise model imitates a pose
network: pixel jitter, low confidence and bigger errors on joints the camera
sees from behind, random dropouts and occasional left/right label swaps.

No downloads, fully deterministic with a seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .pose import Pose2D
from .skeleton import IDX, NUM_KEYPOINTS

# ---------------------------------------------------------------- skeleton


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


@dataclass
class BodyState:
    """Joint angles in radians. Body frame: +X = user's left, +Y up,
    +Z = forward."""
    root_xz: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0
    hip: tuple[float, float] = (0.0, 0.0)     # flexion (forward +), L, R
    knee: tuple[float, float] = (0.0, 0.0)    # flexion (bend +), L, R
    hip_abd: tuple[float, float] = (0.0, 0.0)  # abduction (outwards +), L, R
    ankle: tuple[float, float] = (0.0, 0.0)   # dorsiflexion (+ toes up), L, R
    shoulder: tuple[float, float] = (0.0, 0.0)  # forward swing, L, R
    elbow: tuple[float, float] = (0.2, 0.2)
    lean: float = 0.0                          # torso forward lean


def skeleton_from_state(s: BodyState, height: float = 1.75) -> np.ndarray:
    """Forward kinematics -> (26,3) world joints, feet placed on the floor."""
    k = height / 1.75
    J = np.zeros((NUM_KEYPOINTS, 3))
    pelvis = np.array([0, 0.95 * k, 0])
    J[IDX["hip"]] = pelvis
    torso = _rot_x(s.lean)
    for i, side in enumerate(("left", "right")):
        sgn = 1 if side == "left" else -1
        hip = pelvis + np.array([sgn * 0.09 * k, 0, 0])
        R_thigh = _rot_x(-s.hip[i]) @ np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        abd = np.array([[np.cos(s.hip_abd[i]), -sgn * np.sin(s.hip_abd[i]), 0],
                        [sgn * np.sin(s.hip_abd[i]), np.cos(s.hip_abd[i]), 0],
                        [0, 0, 1]])
        down = np.array([0, -1.0, 0])
        thigh_dir = abd @ _rot_x(-s.hip[i]) @ down
        knee = hip + 0.43 * k * thigh_dir
        shin_dir = abd @ _rot_x(-(s.hip[i] - s.knee[i])) @ down
        ankle = knee + 0.42 * k * shin_dir
        pitch = s.hip[i] - s.knee[i] + s.ankle[i]  # 0 = foot flat
        foot_R = abd @ _rot_x(-pitch)
        f = foot_R @ np.array([0, 0, 1.0])
        d = foot_R @ np.array([0, -1.0, 0])
        lat = abd @ np.array([sgn * 1.0, 0, 0])
        J[IDX[f"{side}_hip"]] = hip
        J[IDX[f"{side}_knee"]] = knee
        J[IDX[f"{side}_ankle"]] = ankle
        J[IDX[f"{side}_heel"]] = ankle + 0.06 * k * d - 0.05 * k * f
        J[IDX[f"{side}_big_toe"]] = ankle + 0.07 * k * d + 0.15 * k * f + 0.01 * k * lat
        J[IDX[f"{side}_small_toe"]] = ankle + 0.07 * k * d + 0.12 * k * f + 0.05 * k * lat

    neck = pelvis + torso @ np.array([0, 0.52 * k, 0])
    J[IDX["neck"]] = neck
    up = torso @ np.array([0, 1.0, 0])
    fwd = torso @ np.array([0, 0, 1.0])
    J[IDX["head"]] = neck + 0.22 * k * up
    J[IDX["nose"]] = neck + 0.13 * k * up + 0.10 * k * fwd
    for side, sgn in (("left", 1), ("right", -1)):
        lat = np.array([sgn * 1.0, 0, 0])
        J[IDX[f"{side}_eye"]] = neck + 0.16 * k * up + 0.08 * k * fwd + 0.03 * k * lat
        J[IDX[f"{side}_ear"]] = neck + 0.14 * k * up + 0.075 * k * lat
        i = 0 if side == "left" else 1
        sh = neck - 0.03 * k * up + 0.19 * k * lat
        J[IDX[f"{side}_shoulder"]] = sh
        arm = torso @ _rot_x(-s.shoulder[i]) @ np.array([sgn * 0.12, -1.0, 0])
        arm /= np.linalg.norm(arm)
        el = sh + 0.29 * k * arm
        fore = torso @ _rot_x(-(s.shoulder[i] + s.elbow[i])) @ np.array([sgn * 0.05, -1.0, 0])
        fore /= np.linalg.norm(fore)
        J[IDX[f"{side}_elbow"]] = el
        J[IDX[f"{side}_wrist"]] = el + 0.26 * k * fore

    feet = [IDX[n] for n in ("left_heel", "right_heel", "left_big_toe",
                             "right_big_toe", "left_small_toe", "right_small_toe")]
    J[:, 1] -= J[feet, 1].min()  # stand on the floor
    J = J @ _rot_y(s.yaw).T
    J[:, 0] += s.root_xz[0]
    J[:, 2] += s.root_xz[1]
    return J


# ----------------------------------------------------------------- motions

def _motion_state(name: str, t: float) -> BodyState:
    w = 2 * np.pi
    if name == "idle":
        sway = 0.02 * np.sin(w * 0.3 * t)
        return BodyState(root_xz=(sway, 0), shoulder=(0.05, 0.05))
    if name == "walk":  # walking in place, 1 step/s per leg
        ph = w * 1.0 * t
        a = 0.45 * np.sin(ph)
        return BodyState(
            hip=(max(a, -0.2), max(-a, -0.2)),
            knee=(0.9 * max(0, np.sin(ph + 0.6)), 0.9 * max(0, np.sin(ph + np.pi + 0.6))),
            shoulder=(-0.4 * np.sin(ph), 0.4 * np.sin(ph)))
    if name == "squat":
        d = 0.5 * (1 - np.cos(w * 0.4 * t))  # 0..1
        th = 1.2 * d
        return BodyState(hip=(th, th), knee=(2 * th, 2 * th), ankle=(th, th),
                         lean=0.5 * d, shoulder=(1.2 * d, 1.2 * d))
    if name == "turn":  # full 360 in 6 s while idling
        return BodyState(yaw=w * t / 6.0, shoulder=(0.1, 0.1))
    if name == "kick":
        k = max(0.0, np.sin(w * 0.5 * t)) ** 2
        return BodyState(hip=(1.3 * k, 0), knee=(1.2 * k * (1 - k) + 0.1, 0.05),
                         shoulder=(-0.3 * k, 0.3 * k))
    if name == "side_step":
        x = 0.4 * np.sin(w * 0.25 * t)
        abd = 0.25 * abs(np.cos(w * 0.25 * t))
        return BodyState(root_xz=(x, 0), hip_abd=(abd, abd * 0.5))
    raise ValueError(name)


MOTIONS = ["idle", "walk", "squat", "turn", "kick", "side_step"]


def motion_sequence(names=MOTIONS, seconds_each=6.0, fps=30, height=1.75):
    """Yields (t, (26,3) ground-truth joints, motion name)."""
    i = 0
    for name in names:
        for f in range(int(seconds_each * fps)):
            t = f / fps
            yield i / fps, skeleton_from_state(_motion_state(name, t), height), name
            i += 1


# ------------------------------------------------------- 2D noise model

@dataclass
class NoiseModel:
    pixel_sigma: float = 3.0        # px jitter on a 720x1280 frame, front view
    back_view_factor: float = 3.0   # jitter multiplier when seen from behind
    dropout: float = 0.02           # chance a keypoint is missed
    lr_swap_front: float = 0.002    # per-frame chance labels flip, front view
    lr_swap_back: float = 0.10      # ... when the camera sees the back
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)


def facing(J: np.ndarray) -> np.ndarray:
    """Unit forward direction of the body (horizontal)."""
    left = J[IDX["left_hip"]] - J[IDX["right_hip"]]
    fwd = np.cross(left, [0, 1.0, 0])
    fwd[1] = 0
    return fwd / (np.linalg.norm(fwd) + 1e-9)


def render_2d(cam: Camera, J: np.ndarray, noise: NoiseModel | None) -> Pose2D | None:
    uv, z = cam.project(J)
    inside = cam.in_frame(uv, z)
    if inside.sum() < 6:
        return None
    to_cam = cam.center - J[IDX["hip"]]
    to_cam[1] = 0
    to_cam /= np.linalg.norm(to_cam) + 1e-9
    front = float(np.dot(facing(J), to_cam))   # 1 = sees face, -1 = sees back
    back = (1 - front) / 2                       # 0..1
    scores = np.where(inside, 0.95 - 0.35 * back, 0.0)
    if noise is None:
        return Pose2D(uv.astype(np.float32), scores.astype(np.float32))
    rng = noise.rng
    sigma = noise.pixel_sigma * (1 + (noise.back_view_factor - 1) * back)
    uv = uv + rng.normal(0, sigma, uv.shape)
    scores = np.clip(scores + rng.normal(0, 0.05, scores.shape), 0, 1)
    scores[rng.random(NUM_KEYPOINTS) < noise.dropout] = 0.0
    p = Pose2D(uv.astype(np.float32), scores.astype(np.float32))
    swap_p = noise.lr_swap_front + (noise.lr_swap_back - noise.lr_swap_front) * back
    if rng.random() < swap_p:
        from .triangulate import LR_SWAP
        p = Pose2D(p.keypoints[LR_SWAP], p.scores[LR_SWAP])
    return p
