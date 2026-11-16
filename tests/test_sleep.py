import threading
import types

from fbt import engine as E


class FakeSlot:
    def __init__(self, person_t=0.0):
        self.person_t, self.stopped = person_t, False

    def stop(self):
        self.stopped = True


def _engine(mode="setup", calib="idle", slots=None, setup_since=0.0):
    e = E.Engine.__new__(E.Engine)          # no cameras, no models: only the sleep logic
    e.lock = threading.RLock()
    e.mode, e.calib, e.slept_auto = mode, {"state": calib}, False
    e.slots = slots if slots is not None else [FakeSlot(), FakeSlot()]
    e.settings = {"auto_sleep": True}
    e.setup_since = setup_since
    e.skeleton = e.skeleton_valid = None
    e.made = 0

    def make():
        e.made += 1
        e.slots = [FakeSlot(), FakeSlot()]
    e._make_slots = make
    return e


def test_sleep_stops_every_camera_and_wake_brings_them_back():
    e = _engine()
    old = e.slots
    assert e.sleep() == {"ok": True}
    assert e.mode == "sleep" and e.slots == [] and all(s.stopped for s in old)
    e.wake()
    assert e.mode == "setup" and e.made == 1 and len(e.slots) == 2


def test_no_sleep_during_a_game_or_a_calibration():
    assert "error" in _engine(mode="server").sleep()
    assert "error" in _engine(calib="recording").sleep()


def test_auto_sleep_only_after_five_minutes_without_anyone():
    e = _engine(slots=[FakeSlot(person_t=100.0), FakeSlot()])
    e._maybe_auto_sleep(100.0 + E.AUTO_SLEEP_S - 1)
    assert e.mode == "setup"
    e._maybe_auto_sleep(100.0 + E.AUTO_SLEEP_S + 1)
    assert e.mode == "sleep" and e.slept_auto
    e = _engine(slots=[FakeSlot(), FakeSlot()])
    e.settings["auto_sleep"] = False
    e._maybe_auto_sleep(1e6)
    assert e.mode == "setup"
