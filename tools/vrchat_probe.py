"""Controlled VRChat test WITHOUT cameras: send a perfect synthetic skeleton,
to separate "how VRChat treats our data" from "camera tracking noise".

  .venv/bin/python -m tools.vrchat_probe --quest 192.168.31.150 --scenario stand
  .venv/bin/python -m tools.vrchat_probe --quest 192.168.31.150 --scenario lift

Scenarios:
  stand — person standing still (arms down). Expectation: the tracker cubes
          sit under your avatar (feet on the floor, hip at hip height) and
          follow you when you walk, because the head position anchors them.
  lift  — same, but the LEFT leg lifts for 1.5 s every 8 s (knee up).
          Calibrate while the leg is DOWN; afterwards your avatar's left
          leg should lift by itself.
Head handling (--head):
  pos   — head position every frame, no rotation (default)
  none  — no head at all
The synthetic person is built with the user's height and faces Unity +Z.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from fbt.osc_out import VRChatSender
from fbt.synthetic import BodyState, skeleton_from_state
from fbt.trackers import head_tracker, solve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quest", required=True)
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--scenario", choices=["stand", "lift"], default="stand")
    ap.add_argument("--head", choices=["posrot", "pos", "none"], default="posrot",
                    help="posrot = head position+rotation every frame, like VRChat's official sample")
    ap.add_argument("--height", type=float, default=1.73)
    ap.add_argument("--seconds", type=float, default=300)
    a = ap.parse_args()
    sender = VRChatSender(a.quest, a.port)
    # skeleton_from_state faces our +Z; to_unity() mirrors Z, so rotate 180°
    # to make the person face Unity +Z (VRChat's forward).
    yaw = np.pi
    t0 = time.perf_counter()
    n = 0
    print(f"sending synthetic '{a.scenario}' skeleton to {a.quest}:{a.port}, head={a.head}")
    while time.perf_counter() - t0 < a.seconds:
        t = time.perf_counter() - t0
        lift = 0.0
        if a.scenario == "lift":
            # 6.5 s resting (calibrate during this), then a 1.5 s knee lift.
            ph = t % 8.0
            lift = np.sin(np.pi * (ph - 6.5) / 1.5) if ph > 6.5 else 0.0
        st = BodyState(yaw=yaw, hip=(1.2 * lift, 0.0), knee=(1.4 * lift, 0.0))
        J = skeleton_from_state(st, height=a.height)
        trs = solve(J)
        if a.head in ("pos", "posrot"):
            h = head_tracker(J, source="torso")
            trs.append(type(h)("head", h.position, h.rotation if a.head == "posrot" else None))
        sender.send(trs)
        n += 1
        if n % 150 == 0:
            hip = next(x for x in trs if x.id == "1")
            print(f"  t={t:5.1f}s sent {n} frames; hip {np.round(hip.position, 2)}, "
                  f"head {np.round(trs[-1].position, 2) if a.head != 'none' else '-'}, send errors {sender.errors}",
                  flush=True)
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
