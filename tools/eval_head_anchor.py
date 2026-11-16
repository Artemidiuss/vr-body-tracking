"""How much would the feet slide in VRChat for different head-anchor choices?

VRChat shifts the whole OSC tracker space every frame so that our head point
sits on the avatar head bone, which follows the headset instantly. Our point
arrives ~L later. So the feet cubes are displaced by
    err(t) = sent_head(t) - real_head(t + L)
(real head = the head root, approximated by the ear midpoint smoothed
zero-phase, i.e. non-causally, from our own logs — not true mocap).
We report err on frames where the torso moves while the feet stay put
(sway/lean/bend), horizontal and vertical separately.

  .venv/bin/python -m tools.eval_head_anchor data/hub_log4.jsonl data/hub_log7.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.signal import savgol_filter

from fbt.filters import MotionPredictor, OneEuro
from fbt.skeleton import IDX


def load(paths):
    T, S = [], []
    for p in paths:
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            T.append(r["t"]); S.append(np.array(r["stab"]))
    o = np.argsort(T)
    return np.array(T)[o], np.array(S)[o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--latency", type=float, default=0.15)
    a = ap.parse_args()
    T, S = load(a.logs)
    dt = float(np.median(np.diff(T)))
    L = int(round(a.latency / dt))
    g = lambda n: S[:, IDX[n]]
    spine = g("neck") - g("hip")
    spine_n = spine / np.linalg.norm(spine, axis=1, keepdims=True)
    ears = (g("left_ear") + g("right_ear")) / 2
    real = savgol_filter(ears - 0.04 * spine_n, 9, 2, axis=0)   # head root, non-causal
    feet = (g("left_ankle") + g("right_ankle")) / 2
    fsp = np.linalg.norm(np.diff(savgol_filter(feet, 9, 2, axis=0), axis=0), axis=1) / dt
    hsp = np.linalg.norm(np.diff(real, axis=0), axis=1) / dt
    idx = np.arange(20, len(S) - L - 5)
    torso_only = idx[(fsp[idx] < 0.15) & (hsp[idx] > 0.15)]       # feet planted, upper body moving
    print(f"{len(S)} frames, latency {a.latency*1000:.0f} ms; frames with feet planted + torso moving: {len(torso_only)}")

    def report(name, sent):
        e = sent[torso_only] - real[torso_only + L]
        h = np.linalg.norm(e[:, [0, 2]], axis=1)
        v = e[:, 1]
        jit = np.linalg.norm(np.diff(sent[20:], 2, axis=0), axis=1).mean() * 1000
        print(f"  {name:<44} horiz {h.mean()*100:5.2f} cm | vert {np.abs(v).mean()*100:5.2f} cm (bias {v.mean()*100:+5.2f}) | jitter {jit:5.2f} mm")

    torso_pt = g("neck") + 0.10 * spine_n
    report("current: neck + 10 cm along spine", torso_pt)
    report("ears midpoint - 4 cm", ears - 0.04 * spine_n)
    e1 = OneEuro(2.5, 0.4)
    ears_f = np.array([e1(x, t) for x, t in zip(ears - 0.04 * spine_n, T)])
    report("ears, One-Euro 2.5", ears_f)
    for h_ms in (80, 150, 200):
        p = MotionPredictor(h_ms / 1000, lo=0.05, hi=0.3, clamp=0.15)
        e2 = OneEuro(2.5, 0.4)
        out = np.array([p(e2(x, t)[None, :], t)[0] for x, t in zip(ears - 0.04 * spine_n, T)])
        report(f"ears, One-Euro 2.5 + predict {h_ms} ms", out)
        p = MotionPredictor(h_ms / 1000, lo=0.05, hi=0.3, clamp=0.15)
        out = np.array([p(x[None, :], t)[0] for x, t in zip(torso_pt, T)])
        report(f"neck point + predict {h_ms} ms", out)


if __name__ == "__main__":
    main()
