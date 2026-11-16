"""One-to-one ports of the temporal filters/predictors of three open-source
FBT projects, compared offline on our hub logs (no effect on the live hub).

Sources (read verbatim, see ARCHITECTURE.md):
  KinectToVR  KinectV2Process/KinectJointFilter.cpp + SmoothingParameters.cpp
              Holt double exponential, per frame, params (smoothing .25,
              correction .25, prediction .25, jitterRadius .03 m, maxDev .05 m)
  April-Tag   Simple-OpenVR-Bridge-Driver TrackerDevice.cpp
              per-channel OLS over samples younger than max_time, evaluated at
              the latency-compensated "now" (cap 0.2 s ahead), 0.5 m outlier drop,
              then Update() blend with `smoothing` (client default 0 -> snap)
  Fever       Stabilizer/TwoEuroJointSmoother.swift + Pipeline/PredictiveUpsampler.swift
              One-Euro (min 1 Hz, beta 400, speed^2, d 1 Hz) -> target = joints +
              velocity*lead (clamped 0.18 m) -> Unity SmoothDamp follower,
              smoothTime = clamp(0.7/fps, .03, .10)
Ours          gate+fit -> One-Euro (2.5 Hz, beta .4) [+ speed-gated velocity 80 ms]

All pipelines get the same input: our outlier gate + bone-length fit output
(the "fitted" skeleton), since those stages are not prediction.
Truth = zero-phase (non-causal) Savitzky-Golay smoothing of that input.
Metrics on legs: error vs truth(t) and vs truth(t + L) (L = what the user feels:
our ~150 ms capture->VRChat delay), on moving / still frames, and output jitter.

  .venv/bin/python -m tools.eval_prediction_exact data/hub_log5.jsonl data/hub_log6.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.signal import savgol_filter

from fbt.filters import MotionPredictor, OneEuro, OutlierGate, SkeletonFit
from fbt.skeleton import IDX

LEGS = [IDX[n] for n in ("left_ankle", "right_ankle", "left_knee", "right_knee", "hip")]


# ---------------------------------------------------------------- KinectToVR
class KinectHolt:
    def __init__(self, smoothing=.25, correction=.25, prediction=.25, jitter=.03, maxdev=.05):
        self.s, self.c, self.p, self.J, self.M = smoothing, correction, prediction, max(jitter, 1e-4), maxdev
        self.n = 0
        self.r = self.f = self.t = None

    def __call__(self, raw):  # raw (K,3); one call per body frame
        if self.n == 0:
            f, t = raw.copy(), np.zeros_like(raw)
            self.n = 1
        elif self.n == 1:
            f = (raw + self.r) * 0.5
            t = (f - self.f) * self.c + self.t * (1 - self.c)
            self.n = 2
        else:
            d = np.linalg.norm(raw - self.f, axis=1, keepdims=True)
            f = np.where(d <= self.J, raw * (d / self.J) + self.f * (1 - d / self.J), raw)
            f = f * (1 - self.s) + (self.f + self.t) * self.s
            t = (f - self.f) * self.c + self.t * (1 - self.c)
        p = f + t * self.p
        d = np.linalg.norm(p - raw, axis=1, keepdims=True)
        k = np.where(d > self.M, self.M / np.maximum(d, 1e-12), 1.0)
        p = p * k + raw * (1 - k)
        self.r, self.f, self.t = raw.copy(), f, t
        return p


# ----------------------------------------------------------------- April-Tag
class AprilTagDriver:
    """Buffer of (age, pos); ages advance with wall time between saves."""

    def __init__(self, max_time=0.5, max_saved=120, latency=0.15, smoothing=0.0):
        self.max_time, self.N, self.lat, self.sm = max_time, max_saved, latency, smoothing
        self.buf = []       # list of [age, pos(K,3)], youngest first
        self.last_update = None
        self.out = None

    def _predict(self, x_star):
        if x_star < -0.2:
            x_star = -0.2
        if not self.buf:
            return None
        if len(self.buf) < 4:
            return self.buf[0][1].copy()
        ages = np.array([a for a, _ in self.buf])
        vals = np.stack([v for _, v in self.buf])            # (n,K,3)
        mt = ages.mean()
        vt = ((ages - mt) ** 2).mean()
        if vt < 1e-12:
            return vals.mean(0)
        mv = vals.mean(0)
        cov = ((ages - mt)[:, None, None] * (vals - mv)).mean(0)
        b = cov / vt
        a = mv - b * mt
        return a + b * x_star

    def save(self, pos, wall_t, frame_age):
        time_offset = frame_age + self.lat
        if self.last_update is not None and self.buf:
            # sample age is relative to last_update; predict at the new sample's age
            pred = self._predict(time_offset - (wall_t - self.last_update))
        else:
            pred = None
        if self.last_update is not None:
            dt = wall_t - self.last_update
            self.buf = [[a + dt, v] for a, v in self.buf if a + dt <= self.max_time]
        self.last_update = wall_t
        if pred is not None and np.linalg.norm(pred - pos, axis=1).max() > 0.5:
            return
        if time_offset > self.max_time:
            return
        self.buf.append([time_offset, pos.copy()])
        self.buf.sort(key=lambda r: r[0])
        self.buf = self.buf[: self.N]

    def update(self):
        p = self._predict(0.0)     # latency-compensated "now"
        if p is None:
            return self.out
        self.out = p if (self.sm == 0 or self.out is None) else p * (1 - self.sm) + self.out * self.sm
        return self.out


# --------------------------------------------------------------------- Fever
class FeverPipeline:
    def __init__(self, lead=0.0, fps=30.0, max_step=0.18):
        self.lead, self.dt, self.max_step = lead, 1.0 / fps, max_step
        self.st = min(0.10, max(0.03, 0.7 / fps))
        self.x = self.dx = self.last = None
        self.lt = None
        self.dj = self.dv = None

    @staticmethod
    def _alpha(c, dt):
        tau = 1.0 / (2 * np.pi * max(c, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def _smooth(self, raw, t):
        if self.lt is None:
            self.x, self.last, self.lt, self.dx = raw.copy(), raw.copy(), t, np.zeros_like(raw)
            return raw
        dt = max(t - self.lt, 1e-4)
        self.lt = t
        deriv = (raw - self.last) / dt
        self.dx = self.dx + (deriv - self.dx) * self._alpha(1.0, dt)
        speed = np.linalg.norm(self.dx, axis=1, keepdims=True)
        cut = 1.0 + 400.0 * speed ** 2
        a = 1.0 / (1.0 + (1.0 / (2 * np.pi * cut)) / dt)
        self.x = self.x + (raw - self.x) * a
        self.last = raw.copy()
        return self.x

    @staticmethod
    def _smooth_damp(cur, tgt, vel, st, dt):
        s = max(1e-4, st)
        om = 2 / s
        x = om * dt
        e = 1 / (1 + x + 0.48 * x * x + 0.235 * x ** 3)
        ch = cur - tgt
        tmp = (vel + om * ch) * dt
        vel = (vel - om * tmp) * e
        return tgt + (ch + tmp) * e, vel

    def __call__(self, raw, t):
        j = self._smooth(raw, t)
        target = j.copy()
        if self.lead > 0:
            d = self.dx * self.lead
            m = np.linalg.norm(d, axis=1, keepdims=True)
            d = np.where(m > self.max_step, d * (self.max_step / np.maximum(m, 1e-12)), d)
            target = j + d
        if self.dj is None:
            self.dj, self.dv = target.copy(), np.zeros_like(target)
        else:
            self.dj, self.dv = self._smooth_damp(self.dj, target, self.dv, self.st, self.dt)
        return self.dj


# ---------------------------------------------------------------- evaluation
def load(paths):
    T, R, V = [], [], []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            T.append(r["t"]); R.append(np.array(r["raw"])); V.append(np.array(r.get("valid_mask", [1] * 26), bool))
    o = np.argsort(T)
    return np.array(T)[o], np.array(R)[o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--latency", type=float, default=0.15, help="felt delay to compensate, s")
    a = ap.parse_args()
    T, R = load(a.logs)
    # Shared pre-processing (not prediction): our gate + bone-length fit.
    gate, fit, F = OutlierGate(), SkeletonFit(60), []
    for t, P in zip(T, R):
        Pg, v = gate(P, np.ones(26, bool), t)
        F.append(fit(Pg, v))
    F = np.array(F)[:, LEGS]
    dt = float(np.median(np.diff(T)))
    truth = savgol_filter(F, 9, 2, axis=0)
    L = int(round(a.latency / dt))
    sp = np.linalg.norm(np.diff(truth, axis=0), axis=-1).mean(-1) / dt
    idx = np.arange(80, len(F) - L - 5)
    mov = idx[sp[idx] > np.percentile(sp, 70)]
    sti = idx[sp[idx] < np.percentile(sp, 30)]

    def run(name, fn):
        out = np.array([fn(i) for i in range(len(F))])
        e0 = np.linalg.norm(out[idx] - truth[idx], axis=-1).mean(-1)
        eL = np.linalg.norm(out[idx] - truth[idx + L], axis=-1).mean(-1)
        jit = np.linalg.norm(np.diff(out[80:], 2, axis=0), axis=-1).mean() * 1000
        m, s = np.isin(idx, mov), np.isin(idx, sti)
        print(f"{name:<40} {e0[m].mean()*100:6.2f} {e0[s].mean()*100:6.2f} | {eL[m].mean()*100:6.2f} {eL[s].mean()*100:6.2f} | {jit:6.2f}")

    print(f"{len(F)} frames, dt {dt*1000:.0f} ms, felt latency {a.latency*1000:.0f} ms ({L} frames)")
    print(f"{'pipeline':<40} vs truth(t) mov still | vs truth(t+L) mov still | jitter mm")
    run("raw input (no temporal filter)", lambda i: F[i])

    e = OneEuro(2.5, 0.4)
    run("OURS: One-Euro 2.5/0.4", lambda i: e(F[i].ravel(), T[i]).reshape(-1, 3))
    e2, pr = OneEuro(2.5, 0.4), MotionPredictor(0.08)
    run("OURS + gated velocity 80 ms", lambda i: pr(e2(F[i].ravel(), T[i]).reshape(-1, 3), T[i]))
    e3, pr3 = OneEuro(2.5, 0.4), MotionPredictor(0.15)
    run("OURS + gated velocity 150 ms", lambda i: pr3(e3(F[i].ravel(), T[i]).reshape(-1, 3), T[i]))

    k = KinectHolt()
    run("KinectToVR Holt (exact defaults)", lambda i: k(F[i]))

    for lat in (0.08, a.latency):
        at = AprilTagDriver(max_time=0.5, latency=lat)

        def f_at(i, at=at):
            at.save(F[i], T[i], 0.0)
            return at.update()
        run(f"April-Tag OLS 0.5 s window, latency {lat*1000:.0f} ms", f_at)

    for lead in (0.0, 0.05, 0.15):
        fv = FeverPipeline(lead=lead, fps=1 / dt)
        run(f"Fever exact, lead {lead*1000:.0f} ms", lambda i, fv=fv: fv(F[i], T[i]))


if __name__ == "__main__":
    main()
