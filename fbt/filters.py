"""Temporal stabilisation of the 3D skeleton.

- OneEuro: adaptive low-pass per coordinate. Strong smoothing when a joint
  is still (kills jitter), weak when it moves fast (little lag).
- SkeletonFit: learns bone lengths during the first seconds, then forces
  every frame to them (no rubber legs) and fills joints that could not be
  triangulated from the previous frame's pose.
- Prediction: extrapolates a few tens of ms ahead to hide part of the
  pipeline latency.
"""

from __future__ import annotations

import math

import numpy as np

from .skeleton import IDX, NUM_KEYPOINTS


class OneEuro:
    """One-Euro filter. power=1 is the textbook form (cutoff = min + beta*speed).
    power=2 is what PinoFBT 2.0 / Fever use for webcam FBT (min 1 Hz,
    beta 400): heavy smoothing while standing still, opening up very fast
    once the joint actually moves."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.8, d_cutoff: float = 1.0,
                 power: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff, self.power = min_cutoff, beta, d_cutoff, power
        self.x = None
        self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t, 1e-3)
        self.t = t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        speed = np.linalg.norm(self.dx.reshape(-1, 3), axis=1).repeat(3).reshape(x.shape)
        cutoff = self.min_cutoff + self.beta * speed ** self.power
        a = 1.0 / (1.0 + (1.0 / (2 * np.pi * cutoff)) / dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()

    @property
    def velocity(self):
        return self.dx


# Kinematic tree used to enforce bone lengths: child -> parent (from the hip out).
PARENT = {
    "left_hip": "hip", "right_hip": "hip", "neck": "hip",
    "left_knee": "left_hip", "right_knee": "right_hip",
    "left_ankle": "left_knee", "right_ankle": "right_knee",
    "left_heel": "left_ankle", "right_heel": "right_ankle",
    "left_big_toe": "left_ankle", "right_big_toe": "right_ankle",
    "left_small_toe": "left_ankle", "right_small_toe": "right_ankle",
    "left_shoulder": "neck", "right_shoulder": "neck", "head": "neck", "nose": "neck",
    "left_elbow": "left_shoulder", "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow", "right_wrist": "right_elbow",
    "left_eye": "nose", "right_eye": "nose", "left_ear": "head", "right_ear": "head",
}
ORDER = ["hip"] + [n for n in PARENT]  # parents always before children
_CHAIN = [(IDX[c], IDX[PARENT[c]]) for c in ORDER[1:]]


class SkeletonFit:
    def __init__(self, learn_frames: int = 90, symmetric: bool = True):
        self.learn_frames = learn_frames
        self.symmetric = symmetric
        self.samples: list[np.ndarray] = []
        self.lengths: np.ndarray | None = None   # per child index
        self.last: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self.lengths is not None

    def _learn(self, P, valid):
        L = np.full(NUM_KEYPOINTS, np.nan)
        for c, p in _CHAIN:
            if valid[c] and valid[p]:
                L[c] = np.linalg.norm(P[c] - P[p])
        self.samples.append(L)
        if len(self.samples) >= self.learn_frames:
            with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(np.array(self.samples), axis=0)
            if self.symmetric:  # average left/right pairs
                for n in list(IDX):
                    if n.startswith("left_"):
                        r = IDX[n.replace("left_", "right_")]
                        m = np.nanmean([med[IDX[n]], med[r]])
                        med[IDX[n]] = med[r] = m
            self.lengths = med

    def __call__(self, P: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Returns a complete (26,3) skeleton."""
        P = P.copy()
        if self.last is not None:
            # Fill joints we could not triangulate: keep their offset from the parent.
            for c, p in _CHAIN:
                if not valid[c]:
                    P[c] = P[p] + (self.last[c] - self.last[p])
        if not self.ready:
            self._learn(P, valid)
        else:
            out = P.copy()
            for c, p in _CHAIN:
                L = self.lengths[c]
                if not np.isfinite(L):
                    out[c] = out[p] + (P[c] - P[p])
                    continue
                d = P[c] - P[p]
                n = np.linalg.norm(d)
                out[c] = out[p] + (d / n * L if n > 1e-6 else d)
            P = out
        self.last = P
        return P


class OutlierGate:
    """Rejects physically impossible jumps BEFORE smoothing.

    Measured on real recordings: ~4% of leg point-frames jump > 8 cm in one
    frame (mostly single-frame 2D keypoint glitches on toes/heels, also
    left/right relabelling and points re-appearing after being lost).
    One-Euro is speed-adaptive, so it lets such spikes straight through.
    A joint may move at most max_speed (m/s) * dt + margin per frame; a
    larger jump is treated as missing for up to max_hold frames. If it
    persists it is real: we accept it but blend in over a few frames, so
    there is no visible snap.
    """

    def __init__(self, max_speed: float = 3.0, margin: float = 0.04, max_hold: int = 4,
                 blend: float = 0.6):
        self.max_speed, self.margin, self.max_hold, self.blend = max_speed, margin, max_hold, blend
        self.last = None       # last accepted raw position per joint
        self.t = None
        self.hold = np.zeros(NUM_KEYPOINTS, int)
        self.offset = np.zeros((NUM_KEYPOINTS, 3))
        self.rejected = 0

    def __call__(self, P: np.ndarray, valid: np.ndarray, t: float):
        P, valid = P.copy(), valid.copy()
        if self.last is None:
            self.last, self.t = P.copy(), t
            return P, valid
        dt = max(t - self.t, 1e-3)
        self.t = t
        allowed = self.margin + self.max_speed * dt * (1 + self.hold)
        jump = np.linalg.norm(P - self.last, axis=1)
        for k in np.where(valid)[0]:
            if jump[k] > allowed[k] and self.hold[k] < self.max_hold:
                valid[k] = False
                self.hold[k] += 1
                self.rejected += 1
                continue
            if self.hold[k] > 0:  # accepted after rejections: start a soft blend
                self.offset[k] = (self.last[k] + self.offset[k]) - P[k]
            self.hold[k] = 0
            self.last[k] = P[k]
        P = P + self.offset
        self.offset *= self.blend
        return P, valid


class FootLock:
    """Keeps a standing foot from sliding/jittering (idea from SlimeVR's
    LegTweaks skating correction). While a foot is near the floor and slow,
    its horizontal position is held; it is released when it lifts, moves
    fast, or the raw position drifts away, then blends back over a few
    frames. Applied to all points of a foot together."""

    SIDES = {s: [IDX[f"{s}_{p}"] for p in ("ankle", "heel", "big_toe", "small_toe")]
             for s in ("left", "right")}

    def __init__(self, floor_band=0.065, speed=0.25, release=0.05, blend=0.6):
        self.floor_band, self.speed, self.release, self.blend = floor_band, speed, release, blend
        self.anchor = {s: None for s in self.SIDES}
        self.offset = {s: np.zeros(2) for s in self.SIDES}
        self.prev = None
        self.t = None

    def __call__(self, P: np.ndarray, t: float) -> np.ndarray:
        P = P.copy()
        dt = max(t - self.t, 1e-3) if self.t is not None else None
        for side, idx in self.SIDES.items():
            xz = P[idx][:, [0, 2]].mean(0)
            low = P[idx, 1].min()
            v = (np.linalg.norm(xz - self.prev[side]) / dt) if (self.prev is not None and dt) else 0.0
            a = self.anchor[side]
            if a is None:
                if low < self.floor_band and v < self.speed:
                    self.anchor[side] = xz.copy()
            elif low > self.floor_band * 1.5 or v > self.speed * 2 or np.linalg.norm(xz - a) > self.release:
                self.offset[side] = (a - xz)  # release softly from where it was held
                self.anchor[side] = None
            if self.anchor[side] is not None:
                d = self.anchor[side] - xz
            else:
                d = self.offset[side]
                self.offset[side] = self.offset[side] * self.blend
            P[idx, 0] += d[0]
            P[idx, 2] += d[1]
            if self.prev is None:
                self.prev = {}
            self.prev[side] = xz
        self.t = t
        return P


class SkeletonStabilizer:
    """outlier gate -> bone-length fit -> One-Euro -> foot lock -> (optional) prediction."""

    def __init__(self, min_cutoff=2.5, beta=0.4, predict_s=0.0, learn_frames=90, gate=True,
                 power=1.0, foot_lock=False):
        self.gate = OutlierGate() if gate else None
        self.fit = SkeletonFit(learn_frames)
        self.euro = OneEuro(min_cutoff, beta, power=power)
        self.lock = FootLock() if foot_lock else None
        self.predict_s = predict_s

    def __call__(self, P, valid, t):
        if self.gate is not None:
            P, valid = self.gate(P, valid, t)
        full = self.fit(P, valid)
        sm = self.euro(full.ravel(), t).reshape(-1, 3)
        if self.lock is not None:
            sm = self.lock(sm, t)
        if self.predict_s > 0:
            sm = sm + self.euro.velocity.reshape(-1, 3) * self.predict_s
        return sm
