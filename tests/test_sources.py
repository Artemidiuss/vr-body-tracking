import time

import cv2
import numpy as np
import pytest

from fbt.sources import NetworkEmulator, VideoFileSource


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "clip.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(30):
        w.write(np.full((48, 64, 3), i * 8, np.uint8))
    w.release()
    return path


def test_batch_reads_every_frame(video):
    src = VideoFileSource("a", video, realtime=False)
    frames = list(src.frames())
    assert len(frames) == 30
    assert frames[1].timestamp == pytest.approx(1 / 30)


def test_realtime_plays_at_file_fps(video):
    src = VideoFileSource("a", video, realtime=True).start()
    t0 = time.perf_counter()
    src.finished.wait(5)
    assert time.perf_counter() - t0 == pytest.approx(1.0, abs=0.25)


def test_latest_drops_stale_frames(video):
    src = VideoFileSource("a", video, realtime=True).start()
    time.sleep(0.5)  # slow consumer
    f = src.latest()
    assert f is not None and f.index >= 10
    assert src.latest(newer_than=f.index) is None or src.latest().index > f.index
    src.finished.wait(5)
    assert src.dropped > 0


def test_network_emulator_drops_frames(video):
    inner = VideoFileSource("a", video, realtime=True)
    net = NetworkEmulator(inner, latency_ms=5, jitter_ms=0, drop_rate=0.5, seed=1).start()
    seen, last = 0, -1
    while not net.finished.is_set():
        f = net.latest(last)
        if f:
            seen, last = seen + 1, f.index
        time.sleep(0.005)
    assert 5 < seen < 25
