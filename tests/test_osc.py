import time

import pytest

from fbt.osc_out import Tracker, VRChatSender
from tools.fake_vrchat import Recorder, serve


def test_trackers_reach_fake_vrchat():
    rec = Recorder()
    server = serve(0, rec)
    port = server.server_address[1]
    try:
        sender = VRChatSender("127.0.0.1", port)
        sender.send([
            Tracker("1", (0.0, 1.0, 0.1), (0.0, 90.0, 0.0)),
            Tracker("head", (0.0, 1.7, 0.0), (0.0, 0.0, 0.0)),
        ])
        deadline = time.time() + 2
        while time.time() < deadline and len(rec.counts) < 4:
            time.sleep(0.01)
        assert rec.bad == []
        assert rec.last["/tracking/trackers/1/position"] == pytest.approx((0.0, 1.0, 0.1))
        assert rec.last["/tracking/trackers/1/rotation"] == (0.0, 90.0, 0.0)
        assert "/tracking/trackers/head/position" in rec.last
    finally:
        server.shutdown()
