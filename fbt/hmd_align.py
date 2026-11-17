"""Our camera space -> VRChat's tracking space, from the headset pose VRChat sends.

Model (both spaces in Unity coordinates: y up, left-handed):
    p_hmd(t) = Ry(yaw) * q_root(t) + t0 + R_hmd(t) * d
q_root: our head root from the cameras (ear midpoint, 4 cm down the spine);
d: the headset point relative to that root, in the headset's own frame (eyes are
~6 cm above and ~8 cm in front of it, measured). No scale: our metres come from the
calibration (the height setting). Learned during play, the scale wandered 0.93-1.10 and
every 1% of it moved the feet ~1.5 cm against the head: bent legs (ARCHITECTURE.md §29).

A T-pose sets it at once (reset_from_pose); a background fit over
the last ~40 s of pairs, taken only while the headset moves with our head,
refines it slowly, and corrects a link that is clearly wrong (a Quest recenter, a T-pose
taken with the head turned) once the data show it. With it the trackers are sent already
in VRChat space and the head anchor is the real headset: on our recordings the feet
standing on the floor slid about half as much in VRChat as with the camera head (§23).
"""

from __future__ import annotations

import collections
import threading
import time

import numpy as np
from scipy.optimize import least_squares


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def unity_R(euler_deg) -> np.ndarray:
    """Unity eulerAngles (x, y, z degrees) -> rotation matrix (Ry Rx Rz)."""
    x, y, z = np.radians(euler_deg)
    return Ry(y) @ Rx(x) @ Rz(z)


class HmdAlign:
    LAG = -0.06          # our stabilised head trails the headset by ~60 ms (filter), measured
    WINDOW = 40.0        # s of pairs used for a fit
    WORN_ERR = 0.10      # m: pairs further than this from the link are not learned from

    def __init__(self):
        self.pairs = collections.deque()      # (t, q(3), p(3), R(3,3))
        self.x = None                         # yaw, tx, ty, tz, log s (always 0), dx, dy, dz  (in use)
        self.target = None                    # latest fit; x glides to it (tick)
        self._t_tick = None
        self.err = None                       # median residual of the last fit, m
        self.ready = False
        self.worn = False
        self._last_fit = 0.0
        self._bad_since = None
        self.lost = False                     # worn but off the link for 5 s: ask for a T-pose
        self._recent = collections.deque()
        self._resets = 0                      # T-pose resets so far (a fit started before one is dropped)
        self._on_since = self._off_since = None
        self.lock = threading.Lock()

    # ---------------- data
    D0 = np.array([0.0, 0.06, 0.09])   # headset vs head root before the first fit (measured)

    def _motion(self, t_cap, q, root):
        """How far our head root and the headset's head root moved in the last 1.5 s.
        Head roots on both sides: the headset itself sits ~10 cm in front of the ears and
        swings on every nod, which made a worn headset look like it moved on its own."""
        self._recent.append((t_cap, q, root))
        while self._recent and t_cap - self._recent[0][0] > 1.5:
            self._recent.popleft()
        if len(self._recent) < 10:
            return 0.0, 0.0
        Q = np.array([r[1] for r in self._recent])
        P = np.array([r[2] for r in self._recent])
        return float(np.ptp(Q, axis=0).max()), float(np.ptp(P, axis=0).max())

    def add(self, t_cap: float, q_root: np.ndarray, hmd_at) -> None:
        """q_root: our head root (Unity frame) at capture time t_cap.
        hmd_at(t) -> (position(3), euler(3)) or None."""
        h = hmd_at(t_cap + self.LAG)
        if h is None:
            return
        q = np.asarray(q_root, float)
        p, e = np.asarray(h[0], float), np.asarray(h[1], float)
        R = unity_R(e)
        with self.lock:
            root = p - R @ (self.x[5:8] if self.x is not None else self.D0)
            mq, mp = self._motion(t_cap, q, root)
            r = None if self.x is None else float(np.linalg.norm(self.predict_hmd(q, R) - p))
            # Worn? Only clear evidence of a headset off the head, held 2 s, says no: it lies
            # still while we move, it is carried around, or it is far from the link.
            off = (mq > 0.10 and mp < 0.02) or (mp - mq > 0.30) or (r is not None and r > 0.25)
            if off:
                self._on_since = None
                self._off_since = self._off_since or t_cap
                if t_cap - self._off_since >= 2.0:
                    self.worn = False
            else:
                self._off_since = None
                self._on_since = self._on_since or t_cap
                if t_cap - self._on_since >= 0.5:
                    self.worn = True
            # Learn only from clean pairs: both head roots moving alike, close to the link.
            if mq == 0.0 or abs(mp - mq) >= max(0.04, 0.5 * mq):
                return
            if r is not None:
                if r >= self.WORN_ERR:
                    # On the head but off the link (the headset space was recentred, or the
                    # link is wrong): after 5 s the app asks for a T-pose. The pair is still
                    # learned from, so the fit can find the new link by itself (maybe_fit).
                    self._bad_since = self._bad_since or t_cap
                    self.lost = t_cap - self._bad_since > 5.0
                else:
                    self._bad_since, self.lost = None, False
            self.pairs.append((t_cap, q, p, R))
            while self.pairs and t_cap - self.pairs[0][0] > self.WINDOW:
                self.pairs.popleft()

    # ---------------- model
    def _unpack(self, x):
        return x[0], x[1:4], np.exp(x[4]), x[5:8]

    def predict_hmd(self, q, R_hmd):
        yaw, t0, s, d = self._unpack(self.x)
        return s * (Ry(yaw) @ q) + t0 + R_hmd @ d

    def to_vrc(self, P: np.ndarray) -> np.ndarray:
        """Points (...,3) from our Unity frame to VRChat space."""
        yaw, t0, s, _ = self._unpack(self.x)
        return s * (P @ Ry(yaw).T) + t0

    def yaw_deg(self) -> float:
        return float(np.degrees(self.x[0]))

    def head_root(self, p_hmd, euler) -> np.ndarray:
        """The real head root in VRChat space from a headset pose."""
        _, _, _, d = self._unpack(self.x)
        return np.asarray(p_hmd, float) - unity_R(euler) @ d

    # ---------------- reset from a T-pose
    def reset_from_pose(self, q_root, facing_yaw_deg: float, p_hmd, euler_hmd) -> None:
        """The player holds a T-pose, looking straight ahead: the body facing from
        our cameras and the headset yaw are the same direction. That fixes the yaw at once;
        the offset follows from the head. The background fit then only refines it. A manual
        reset for whatever the automatic fit got wrong (it used to be the yaw alignment)."""
        R_h = unity_R(euler_hmd)
        with self.lock:
            yaw = np.radians(float(euler_hmd[1]) - facing_yaw_deg)
            d = self.x[5:8].copy() if self.x is not None else self.D0.copy()
            t0 = np.asarray(p_hmd, float) - R_h @ d - Ry(yaw) @ np.asarray(q_root, float)
            self.x = np.r_[yaw, t0, 0.0, d]
            self._resets += 1
            self.target = self.x.copy()
            self.pairs.clear()
            self.ready = self.worn = True
            self._bad_since, self.lost = None, False
            self.err = None

    # ---------------- fit
    MOVE_BIG = 0.4       # m walked in the window before a big correction is trusted

    def maybe_fit(self, now: float | None = None, every: float = 2.0) -> bool:
        now = time.perf_counter() if now is None else now
        if now - self._last_fit < every:
            return False
        self._last_fit = now
        with self.lock:
            pairs = list(self.pairs)
            x_in_use = None if self.x is None else self.x.copy()
            resets = self._resets
        if len(pairs) < 150:
            return False
        q = np.array([p[1] for p in pairs])
        P = np.array([p[2] for p in pairs])
        Rh = np.array([p[3] for p in pairs])
        # the fit needs movement: a still player cannot tell yaw from offset
        move = max(np.ptp(P[:, [0, 2]], axis=0).max(), np.ptp(q[:, [0, 2]], axis=0).max())
        if move < 0.15:
            return False

        # fitted: yaw, t0 (3), d (3); the scale stays 1 (x[4] = log s = 0)
        def full(y):
            return np.r_[y[:4], 0.0, y[4:]]

        def pred(y):
            return q @ Ry(y[0]).T + y[1:4] + np.einsum("nij,j->ni", Rh, y[4:])

        def res(y):
            return (pred(y) - P).ravel()

        # plausible people: the headset sits in front of and above the ear midpoint, roughly centred
        lo = [-np.inf, -np.inf, -np.inf, -np.inf, -0.02, -0.03, -0.02]
        hi = [np.inf, np.inf, np.inf, np.inf, 0.02, 0.16, 0.18]
        cur = None if x_in_use is None else float(np.median(np.linalg.norm(
            pred(np.r_[x_in_use[:4], x_in_use[5:]]) - P, axis=1)))
        starts = [] if x_in_use is None else [np.r_[x_in_use[:4], x_in_use[5:]]]
        if cur is None or cur > 0.06:            # no link, or the one in use does not explain the data
            for a0 in np.linspace(-np.pi, np.pi, 8, endpoint=False):
                starts.append(np.r_[a0, P.mean(0) - Ry(a0) @ q.mean(0), self.D0])
        best = None
        for y0 in starts:
            y0 = np.clip(y0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
            r = least_squares(res, y0, loss="huber", f_scale=0.03, bounds=(lo, hi), max_nfev=200)
            if best is None or r.cost < best.cost:
                best = r
        err = float(np.median(np.linalg.norm(best.fun.reshape(-1, 3), axis=1)))
        if err > 0.06 or (x_in_use is not None and err >= 0.04):
            return False                         # not a usable fit (headset off most of the window?)
        new = full(best.x)
        with self.lock:
            if self._resets != resets:           # a T-pose reset the link meanwhile: its data are newer
                return False
            if self.x is not None:
                big = (abs(np.angle(np.exp(1j * (new[0] - self.x[0])))) > np.radians(5)
                       or np.linalg.norm(new[1:4] - self.x[1:4]) > 0.1)
                # A big change only when the last 40 s clearly show the link in use is wrong
                # (a Quest recenter, a T-pose taken with the head turned): the new link explains
                # them within 3 cm, the old one at least twice worse, and the player walked
                # enough for the yaw to be certain. tick() glides it in over a few seconds.
                if big and not (err < 0.03 and cur > max(0.04, 2 * err) and move >= self.MOVE_BIG):
                    return False
            self.target = new
            if self.x is None:
                self.x = new.copy()
            self.err = err
            self.ready = True
        return True

    def tick(self, now: float, tau: float = 4.0) -> None:
        """Glide the transform in use towards the latest fit. Applied in jumps every 2 s,
        a few cm / degrees of refinement made planted feet slide in VRChat (seen on the
        replay); spread over seconds it is well under 1 cm/s."""
        with self.lock:
            if self.x is None or self.target is None:
                self._t_tick = now
                return
            dt = 0.0 if self._t_tick is None else max(0.0, now - self._t_tick)
            self._t_tick = now
            a = min(1.0, dt / tau)
            dyaw = np.angle(np.exp(1j * (self.target[0] - self.x[0])))
            nx = self.x + a * (self.target - self.x)
            nx[0] = self.x[0] + a * dyaw
            self.x = nx
