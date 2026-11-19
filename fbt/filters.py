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

# Links that are not one rigid bone, as (shortest, longest) share of the learned length.
# hip->neck is the whole spine: sitting on the floor it measured 0.41-0.44 m against 0.60 m
# standing, and holding it at 0.60 pushed the neck and head 12-15 cm up, so the link to the
# headset learned a wrong scale (ARCHITECTURE.md §29). The neck bends as well.
FLEXIBLE = {IDX["neck"]: (0.6, 1.05), IDX["head"]: (0.7, 1.3), IDX["nose"]: (0.7, 1.3)}

# The legs hang from where their ankles are, not from the pelvis keypoint: the model puts that
# keypoint 2-4.5 cm high in a T-pose (the thigh 41 cm instead of 38.6) and too far from the knee
# lying with knees bent (43 instead of 38), while the ankles stay put. Hanging the legs from it
# raised the feet in the T-pose VRChat calibrates in, so they sank afterwards (1.3-3.8 cm in our
# games), and lowered them lying (up to 4 cm). Now the pelvis moves by what the two legs say
# (their bones, as now, ending at the ankles), smoothed over FEET_TAU: the error follows the pose,
# keypoint noise is fast, and unsmoothed the hip tracker shook 17-33% more. An ankle glitch now
# moves the pelvis by half of it, smoothed: at 0.5 s that doubled the hip's vertical jerks > 5 mm
# (2-5 -> 7-11 per minute standing), at 1 s 6 / 6 / 0 vs 5 / 2 / 0 with 86% of the correction in
# place 2 s into a T-pose (ARCHITECTURE §40-41).
LEGS = [(IDX[f"{s}_hip"], IDX[f"{s}_knee"], IDX[f"{s}_ankle"]) for s in ("left", "right")]
FEET_TAU = 1.0


class SkeletonFit:
    def __init__(self, learn_frames: int = 90, symmetric: bool = True):
        self.learn_frames = learn_frames
        self.symmetric = symmetric
        self.samples: list[np.ndarray] = []
        self.lengths: np.ndarray | None = None   # per child index
        self.last: np.ndarray | None = None
        self.pelvis_shift: np.ndarray | None = None   # smoothed pelvis correction from the legs
        self._shift_t = None

    @property
    def ready(self) -> bool:
        return self.lengths is not None

    @staticmethod
    def lengths_from(skel: np.ndarray, symmetric: bool = True) -> np.ndarray:
        """Median bone lengths (per child joint) of a triangulated recording (N,26,3)."""
        L = np.full((len(skel), NUM_KEYPOINTS), np.nan)
        for c, p in _CHAIN:
            L[:, c] = np.linalg.norm(skel[:, c] - skel[:, p], axis=1)
        full = np.all(np.isfinite(skel[:, SkeletonFit.FULL_BODY, 0]), axis=1)
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(L[full] if full.sum() >= 30 else L, axis=0)
        if symmetric:
            for n in list(IDX):
                if n.startswith("left_"):
                    r = IDX[n.replace("left_", "right_")]
                    m = np.nanmean([med[IDX[n]], med[r]])
                    med[IDX[n]] = med[r] = m
        return med

    # Only frames where the whole body is triangulated count: right after "start"
    # the user is often still at the laptop, half out of view, and learning from
    # those frames gave 17 cm forearms (so the T-pose was never recognised).
    FULL_BODY = [IDX[n] for n in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                                  "left_wrist", "right_wrist", "left_hip", "right_hip",
                                  "left_knee", "right_knee", "left_ankle", "right_ankle")]

    def _learn(self, P, valid):
        if not np.all(valid[self.FULL_BODY]):
            return
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

    def _shift(self, P: np.ndarray, t: float | None) -> np.ndarray:
        """How far to move the pelvis so the legs, at their learned lengths and measured
        directions, end at the ankles (mean of both legs), smoothed over FEET_TAU."""
        L = self.lengths
        hip = IDX["hip"]
        if not all(np.isfinite(L[[hl, kn, an]]).all() for hl, kn, an in LEGS):
            return np.zeros(3)
        unit = lambda v: v / max(np.linalg.norm(v), 1e-6)
        d = np.mean([P[an] + unit(P[kn] - P[an]) * L[an] + unit(P[hl] - P[kn]) * L[kn]
                     - (P[hip] + unit(P[hl] - P[hip]) * L[hl]) for hl, kn, an in LEGS], axis=0)
        if t is None:
            return d
        if self.pelvis_shift is None or self._shift_t is None or t < self._shift_t:
            self.pelvis_shift = d                       # first frame
        else:
            # after a gap the step is simply longer (and converges faster); starting afresh there
            # made the pelvis jump whenever the torso was lost for a moment
            self.pelvis_shift = self.pelvis_shift + (1 - np.exp(-(t - self._shift_t) / FEET_TAU)) * (d - self.pelvis_shift)
        self._shift_t = t
        return self.pelvis_shift

    def __call__(self, P: np.ndarray, valid: np.ndarray, t: float | None = None) -> np.ndarray:
        """Returns a complete (26,3) skeleton. t (seconds) smooths the pelvis correction."""
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
            out[IDX["hip"]] = P[IDX["hip"]] + self._shift(P, t)
            for c, p in _CHAIN:
                L = self.lengths[c]
                # the spine reaches the neck where it was measured, from the moved pelvis
                d = P[c] - out[p] if c == IDX["neck"] else P[c] - P[p]
                if not np.isfinite(L):
                    out[c] = out[p] + d
                    continue
                n = np.linalg.norm(d)
                if c in FLEXIBLE:
                    lo, hi = FLEXIBLE[c]
                    L = min(max(n, lo * L), hi * L)
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


class MotionPredictor:
    """Latency hiding (Fever/PinoFBT-style velocity extrapolation), gated by
    speed so nothing is added while a joint is still, and clamped like
    KinectToVR's maxDeviation. Chosen on our session logs
    (tools/eval_prediction.py): at 80 ms it removes ~24% of the lag error in
    motion, adds no error at rest, +10% jitter."""

    def __init__(self, horizon_s=0.08, lo=0.15, hi=0.5, v_smooth=0.4, clamp=0.05):
        self.h, self.lo, self.hi, self.vs, self.clamp = horizon_s, lo, hi, v_smooth, clamp
        self.prev = None
        self.t = None
        self.v = None

    def __call__(self, P: np.ndarray, t: float) -> np.ndarray:
        if self.prev is None or self.h <= 0:
            self.prev, self.t, self.v = P.copy(), t, np.zeros_like(P)
            return P
        dt = max(t - self.t, 1e-3)
        self.v = (1 - self.vs) * self.v + self.vs * (P - self.prev) / dt
        self.prev, self.t = P.copy(), t
        sp = np.linalg.norm(self.v, axis=1, keepdims=True)
        w = np.clip((sp - self.lo) / (self.hi - self.lo), 0, 1)
        d = self.v * self.h * w
        n = np.linalg.norm(d, axis=1, keepdims=True)
        d = np.where(n > self.clamp, d * (self.clamp / np.maximum(n, 1e-9)), d)
        return P + d


class FloorClip:
    """SlimeVR LegTweaks "floor clip": a foot can never be below the floor.
    If the lowest heel/toe point of a foot is under the floor (y=0 after
    calibration), the whole foot (ankle, heel, toes) is lifted together, so its
    rotation is unchanged. Only physically impossible frames are touched
    (measured: ~15% of frames had a foot point below the floor, mostly < 3 cm,
    i.e. keypoint noise of a foot standing on the floor). Knees likewise."""

    FEET = {s: [IDX[f"{s}_{p}"] for p in ("ankle", "heel", "big_toe", "small_toe")]
            for s in ("left", "right")}
    KNEES = [IDX["left_knee"], IDX["right_knee"]]

    def __init__(self, floor: float = 0.0):
        self.floor = floor

    def __call__(self, P: np.ndarray) -> np.ndarray:
        P = P.copy()
        for idx in self.FEET.values():
            low = P[idx, 1].min()
            if low < self.floor:
                P[idx, 1] += self.floor - low
        for k in self.KNEES:
            P[k, 1] = max(P[k, 1], self.floor + 0.05)
        return P


class SkeletonStabilizer:
    """outlier gate -> bone-length fit -> One-Euro -> foot lock -> (optional) prediction."""

    def __init__(self, min_cutoff=2.5, beta=0.4, predict_s=0.0, learn_frames=90, gate=True,
                 power=1.0, foot_lock=False, floor_clip=False, predict_gated=0.0):
        self.gate = OutlierGate() if gate else None
        self.fit = SkeletonFit(learn_frames)
        self.euro = OneEuro(min_cutoff, beta, power=power)
        self.lock = FootLock() if foot_lock else None
        self.clip = FloorClip() if floor_clip else None
        self.pred = MotionPredictor(predict_gated) if predict_gated > 0 else None
        self.predict_s = predict_s

    def __call__(self, P, valid, t):
        if self.gate is not None:
            P, valid = self.gate(P, valid, t)
        full = self.fit(P, valid, t)
        sm = self.euro(full.ravel(), t).reshape(-1, 3)
        if self.pred is not None:
            sm = self.pred(sm, t)
        if self.lock is not None:
            sm = self.lock(sm, t)
        if self.clip is not None:
            sm = self.clip(sm)
        if self.predict_s > 0:
            sm = sm + self.euro.velocity.reshape(-1, 3) * self.predict_s
        return sm
