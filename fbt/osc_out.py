"""Sending trackers to VRChat (OSC Trackers protocol).

VRChat expects, per tracker 1..8 and `head`:
  /tracking/trackers/<id>/position  float x, y, z   (metres, Unity space, Y up)
  /tracking/trackers/<id>/rotation  float x, y, z   (Euler degrees, ZXY order)
Messages are sent in one OSC bundle per frame.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

from pythonosc.osc_bundle_builder import IMMEDIATELY, OscBundleBuilder
from pythonosc.osc_message_builder import OscMessageBuilder

VRCHAT_PORT = 9000


@dataclass
class Tracker:
    id: str                                   # "1".."8" or "head"
    position: tuple[float, float, float] | None
    rotation: tuple[float, float, float] | None  # None = do not send


def build_bundle(trackers: list[Tracker]) -> bytes:
    bundle = OscBundleBuilder(IMMEDIATELY)
    for t in trackers:
        for kind, vals in (("position", t.position), ("rotation", t.rotation)):
            if vals is None:
                continue
            msg = OscMessageBuilder(address=f"/tracking/trackers/{t.id}/{kind}")
            for v in vals:
                msg.add_arg(float(v))
            bundle.add_content(msg.build())
    return bundle.build().dgram


class VRChatSender:
    def __init__(self, host: str, port: int = VRCHAT_PORT):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.errors = 0

    def send(self, trackers: list[Tracker]) -> None:
        """Never raises: a sleeping headset drops off Wi-Fi ("no route to
        host") and must not kill the hub; sending resumes when it is back."""
        if not trackers:
            return
        try:
            self.sock.sendto(build_bundle(trackers), self.addr)
        except OSError:
            self.errors += 1
