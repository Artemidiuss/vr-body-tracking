"""The game's 2D pose log: exact round trip, safe to write from the tracking loop (§32)."""
import time

import numpy as np

from fbt.pose import Pose2D
from fbt.poselog import FRAME3D, NO_PAIR, NO_TORSO, PoseLog, read


def _pose(seed):
    rng = np.random.default_rng(seed)
    return Pose2D(rng.uniform(0, 640, (26, 2)).astype(np.float32), rng.uniform(0, 1, 26).astype(np.float32))


def test_round_trip_is_exact(tmp_path):
    p = tmp_path / "s_poses2d.bin"
    log = PoseLog(str(p), 3, ["192.168.31.81", "192.168.31.28", "mac"])
    a, b, c = _pose(1), _pose(2), _pose(3)
    log.write(1.5, FRAME3D, [a, b, c])
    log.write(1.6, NO_TORSO, [a, b, None])                # the Mac had no pose this step
    log.write(1.7, NO_PAIR, [a, None, None])
    log.close()
    head, r = read(str(p))
    assert head["cams"] == ["192.168.31.81", "192.168.31.28", "mac"]
    assert list(r["t"]) == [1.5, 1.6, 1.7] and list(r["outcome"]) == [FRAME3D, NO_TORSO, NO_PAIR]
    assert r["present"].tolist() == [[1, 1, 1], [1, 1, 0], [1, 0, 0]]
    assert np.array_equal(r["kp"][0, 2], c.keypoints) and np.array_equal(r["sc"][1, 1], b.scores)
    assert not r["kp"][1, 2].any()


def test_cut_record_at_the_end_is_left_out(tmp_path):
    p = tmp_path / "s_poses2d.bin"
    log = PoseLog(str(p), 2, ["a", "b"])
    for i in range(3):
        log.write(float(i), FRAME3D, [_pose(i), _pose(i + 9)])
    log.close()
    data = p.read_bytes()
    p.write_bytes(data[:-10])                             # the app closed mid-write
    _, r = read(str(p))
    assert len(r) == 2


def test_writing_after_close_never_raises_and_is_fast(tmp_path):
    log = PoseLog(str(tmp_path / "s.bin"), 3, ["a", "b", "c"])
    poses = [_pose(1), _pose(2), _pose(3)]
    t0 = time.perf_counter()
    for i in range(2000):
        log.write(float(i), FRAME3D, poses)
    per_write = (time.perf_counter() - t0) / 2000
    assert per_write < 1e-4                               # well under 0.1 ms per 3D step
    log.close()
    log.write(0.0, FRAME3D, poses)                         # the game stopped meanwhile: dropped
