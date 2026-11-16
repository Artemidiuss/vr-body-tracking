"""Offline evaluation of motion prediction (latency hiding) on hub logs.

For each frame t we predict the position at t+h from data up to t, and
compare with where the joint really was at t+h (from the same log). Baseline
(no prediction) = S(t). Metrics on legs (ankles, knees, hip):
  err   — mean |pred - S(t+h)| in cm, on moving frames (what latency hiding buys)
  still — mean |pred - S(t+h)| on still frames (prediction must not add error)
  jit   — accel of the predicted output, mm/frame^2 (noise amplification)

Methods (from studied projects):
  none                 — no prediction
  vel (Fever)          — S + v*h, v = One-Euro-style smoothed velocity
  reg N (April-Tag)    — linear regression over last N samples, evaluated at t+h
  holt (KinectToVR)    — double exponential trend, clamped to max deviation
All predictions are clamped to `clamp` metres from S(t) (KinectToVR maxDev).

  .venv/bin/python -m tools.eval_prediction data/hub_log6.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from fbt.skeleton import IDX

J = [IDX[n] for n in ("left_ankle", "right_ankle", "left_knee", "right_knee", "hip")]


def load(paths):
    T, S = [], []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            T.append(r["t"]); S.append(np.array(r["stab"])[J])
    T, S = np.array(T), np.array(S)
    o = np.argsort(T)
    return T[o], S[o]


def clampd(P, S, c):
    d = P - S
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return np.where(n > c, S + d * (c / np.maximum(n, 1e-9)), P)


def predict(T, S, h, method, N=6, clamp=0.05, alpha=0.5, beta=0.3, vs=0.4, lo=0.15, hi=0.5):
    P = S.copy()
    if method == "none":
        return P
    if method == "vel":
        v = np.zeros_like(S[0])
        for i in range(1, len(S)):
            dt = max(T[i] - T[i - 1], 1e-3)
            v = (1 - vs) * v + vs * (S[i] - S[i - 1]) / dt
            P[i] = S[i] + v * h
    elif method == "velgate":
        # Fever-style velocity extrapolation, but only for joints that really
        # move: weight ramps 0->1 between lo and hi m/s (no prediction at rest).
        v = np.zeros_like(S[0])
        for i in range(1, len(S)):
            dt = max(T[i] - T[i - 1], 1e-3)
            v = (1 - vs) * v + vs * (S[i] - S[i - 1]) / dt
            sp = np.linalg.norm(v, axis=-1, keepdims=True)
            w = np.clip((sp - lo) / (hi - lo), 0, 1)
            P[i] = S[i] + v * h * w
    elif method.startswith("reg"):
        for i in range(N, len(S)):
            t = T[i - N + 1:i + 1] - T[i]
            X = S[i - N + 1:i + 1]
            tm = t.mean()
            slope = ((t - tm)[:, None, None] * (X - X.mean(0))).sum(0) / max(((t - tm) ** 2).sum(), 1e-9)
            P[i] = X.mean(0) + slope * (0 - tm) + slope * h
    elif method == "holt":
        lvl, tr = S[0].copy(), np.zeros_like(S[0])
        for i in range(1, len(S)):
            dt = max(T[i] - T[i - 1], 1e-3)
            prev = lvl.copy()
            lvl = alpha * S[i] + (1 - alpha) * (lvl + tr * dt)
            tr = beta * (lvl - prev) / dt + (1 - beta) * tr
            P[i] = lvl + tr * h
    return clampd(P, S, clamp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    T, S = load(a.logs)
    dt = np.median(np.diff(T))
    speed = np.linalg.norm(np.diff(S, axis=0), axis=-1).mean(-1) / dt
    print(f"{len(T)} frames, dt {dt * 1000:.0f} ms")
    for h_ms in (50, 80, 120):
        k = int(round(h_ms / 1000 / dt))
        idx = np.arange(10, len(S) - k)
        mov = idx[speed[idx] > np.percentile(speed, 70)]
        sti = idx[speed[idx] < np.percentile(speed, 30)]
        print(f"\nhorizon {h_ms} ms ({k} frames)          err moving cm | err still cm | jitter mm")
        for name, kw in [("none", {}), ("vel (Fever)", dict(method="vel")),
                         ("vel smoother v", dict(method="vel", vs=0.25)),
                         ("velgate 0.15-0.5", dict(method="velgate")),
                         ("velgate 0.3-0.8", dict(method="velgate", lo=0.3, hi=0.8)),
                         ("velgate 0.3-0.8 smooth v", dict(method="velgate", lo=0.3, hi=0.8, vs=0.25)),
                         ("velgate 0.5-1.2", dict(method="velgate", lo=0.5, hi=1.2))]:
            m = kw.pop("method", "none")
            P = predict(T, S, h_ms / 1000, m, **kw)
            e = np.linalg.norm(P[idx] - S[idx + k], axis=-1).mean(-1)
            em = e[np.isin(idx, mov)].mean() * 100
            es = e[np.isin(idx, sti)].mean() * 100
            jit = np.linalg.norm(np.diff(P, 2, axis=0), axis=-1).mean() * 1000
            print(f"  {name:<22} {em:14.2f} | {es:11.2f} | {jit:8.2f}")


if __name__ == "__main__":
    main()
