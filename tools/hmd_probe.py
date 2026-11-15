"""Check that VRChat sends us the headset pose via OSCQuery, and dump the raw
packets so we can see the format.

  .venv/bin/python -m tools.hmd_probe --seconds 60

In VRChat: OSC enabled, and Main Menu -> Tracking & IK ->
"Allow Sending Head and Wrist VR Tracking OSC Data". VRChat shows a HUD
notification when it starts sending to "FBT-Hub".
"""

from __future__ import annotations

import argparse
import json
import time

from fbt.oscquery import OSCQueryReceiver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--dump", default="data/hmd_probe.jsonl")
    a = ap.parse_args()
    log = open(a.dump, "w")

    def on_msg(addr, args, t):
        log.write(json.dumps({"t": t, "addr": addr, "args": list(args)}) + "\n")

    rx = OSCQueryReceiver(on_message=on_msg)
    print(flush=True)
    print(f"advertising FBT-Hub on {rx.ip}: OSC udp {rx.osc_port}, OSCQuery http {rx.http_port}")
    t_end = time.time() + a.seconds
    shown = set()
    while time.time() < t_end:
        time.sleep(1)
        for hit in rx.http_hits[len(shown):]:
            print("  http query:", hit, flush=True)
            shown.add(hit)
        if rx.raw:
            print(f"  raw datagrams {rx.raw}", flush=True)
        if rx.parse_errors:
            print(f"  parse errors: {rx.parse_errors[:3]}", flush=True)
        if rx.counts:
            summary = ", ".join(f"{k}: {v}" for k, v in sorted(rx.counts.items()))
            print(f"  from {sorted(rx.senders)}: {summary}", flush=True)
            for k, v in sorted(rx.last.items()):
                print(f"     {k} = {tuple(round(x, 4) if isinstance(x, float) else x for x in v)}")
    log.close()
    print(f"raw packets: {a.dump}", flush=True)
    import os
    os._exit(0)  # zeroconf shutdown can hang for seconds; nothing to clean up


if __name__ == "__main__":
    main()
