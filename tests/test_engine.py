"""App engine helpers: tracker toggles, calibration verdicts, settings import."""
import json

from fbt.engine import DEFAULTS, Engine, enabled_trackers, friendly_name, human_error, quality


def test_tracker_toggles():
    t = enabled_trackers({"track_chest": False, "track_knees": True, "track_elbows": True})
    assert t[:3] == ("hip", "left_foot", "right_foot")
    assert "chest" not in t and {"left_knee", "right_elbow"} <= set(t)


def test_quality_and_errors():
    assert quality({"reproj_median_px": 0.9}) == "відмінна"
    assert quality({"reproj_median_px": 2.5}) == "добра"
    assert quality({"reproj_median_px": 4}) == "погана"
    assert quality({}) is None
    assert "обидва" in human_error(RuntimeError("only 3 synchronised frames"))


def test_names():
    assert friendly_name("mac", None) == "MacBook"
    assert friendly_name("1.2.3.4", {"device": "samsung SM-A325F"}) == "Samsung"
    assert friendly_name("1.2.3.4", {"device": "Xiaomi 23053RN02Y"}) == "Redmi"


def test_settings_file_wins_over_defaults(tmp_path):
    e = Engine.__new__(Engine)          # settings logic only, no cameras / threads
    e.d, e.settings_path = str(tmp_path), str(tmp_path / "app_settings.json")
    e.cams, e.calib_hosts = [], []
    assert e._load_settings()["head_offset_cm"] == DEFAULTS["head_offset_cm"]
    (tmp_path / "app_settings.json").write_text(json.dumps({"head_offset_cm": 3}))
    assert e._load_settings()["head_offset_cm"] == 3
