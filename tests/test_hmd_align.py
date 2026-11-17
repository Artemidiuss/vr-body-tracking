"""Camera space -> VRChat space from headset poses (synthetic)."""
import numpy as np

from fbt.hmd_align import HmdAlign, Ry, unity_R


def test_recovers_transform_and_head_offset():
    rng = np.random.default_rng(0)
    yaw, t0, s, d = np.radians(201.0), np.array([0.02, -0.1, -0.2]), 1.0, np.array([-0.01, 0.06, 0.09])
    al = HmdAlign()
    poses = {}
    for i in range(1200):                                   # 40 s of walking around at 30 fps
        t = i / 30
        q = np.array([0.8 * np.sin(t / 3), 1.5 + 0.2 * np.sin(t / 2), 0.6 * np.cos(t / 4)])
        e = np.array([10 * np.sin(t), np.degrees(t / 2) % 360, 3 * np.cos(t)])
        p = s * (Ry(yaw) @ q) + t0 + unity_R(e) @ d
        poses[round(t + al.LAG, 6)] = (p, e)
        al.add(t, q + rng.normal(0, 0.01, 3), lambda tq: poses.get(round(tq, 6)))
    al.maybe_fit(now=100.0)
    assert al.ready and al.err < 0.02
    assert al.x[4] == 0.0                                    # the scale is our calibration's (§29)
    assert abs((al.yaw_deg() - 201.0 + 180) % 360 - 180) < 1.0
    # the head root it reports is the real one (headset pose minus the offset)
    p, e = poses[max(poses)]
    root = al.head_root(p, e)
    assert np.linalg.norm(root - (p - unity_R(e) @ d)) < 0.03


def test_glides_instead_of_jumping():
    al = HmdAlign()
    al.x = np.zeros(8)
    al.target = np.zeros(8)
    al.target[1] = 0.04                                      # a 4 cm refinement
    al.tick(0.0)
    al.tick(0.5)
    assert 0 < al.x[1] < 0.01                                # moved only a bit in 0.5 s


def _walk(al, poses, link, t_from, seconds, rng):
    """Walk around for `seconds`, the headset seen through `link` = (yaw deg, t0, d)."""
    yaw, t0, d = link
    for i in range(int(seconds * 30)):
        t = t_from + i / 30
        q = np.array([0.8 * np.sin(t / 3), 1.5 + 0.05 * np.sin(t / 2), 0.6 * np.cos(t / 4)])
        e = np.array([10 * np.sin(t), np.degrees(t / 2) % 360, 3 * np.cos(t)])
        p = Ry(np.radians(yaw)) @ q + t0 + unity_R(e) @ d
        poses[round(t + al.LAG, 6)] = (p, e)
        al.add(t, q + rng.normal(0, 0.01, 3), lambda tq: poses.get(round(tq, 6)))


def test_finds_the_link_again_after_a_recenter():
    """A Quest recenter turns and shifts the headset space: the fit takes the new link
    by itself (before, only a T-pose could), and glides to it."""
    rng = np.random.default_rng(1)
    d = np.array([0.0, 0.06, 0.09])
    al, poses = HmdAlign(), {}
    _walk(al, poses, (180.0, np.array([0.1, 0.0, 0.2]), d), 0.0, 40, rng)
    al.maybe_fit(now=40.0)
    assert abs((al.yaw_deg() - 180 + 180) % 360 - 180) < 1.0
    _walk(al, poses, (215.0, np.array([-0.4, 0.0, 0.5]), d), 40.0, 45, rng)     # recentred
    assert al.lost                                           # far off the old link
    al.maybe_fit(now=85.0)
    assert abs((al.target[0] - np.radians(215) + np.pi) % (2 * np.pi) - np.pi) < np.radians(1.5)
    for k in range(200):                                     # ~90% in 9 s, settled in 20 s
        al.tick(85.0 + k * 0.1)
    assert abs((al.yaw_deg() - 215 + 180) % 360 - 180) < 2.0


def test_no_big_correction_without_walking():
    """Standing almost still, the yaw is not certain: a big change is not taken."""
    rng = np.random.default_rng(2)
    al, poses = HmdAlign(), {}
    d = np.array([0.0, 0.06, 0.09])
    _walk(al, poses, (180.0, np.array([0.1, 0.0, 0.2]), d), 0.0, 40, rng)
    al.maybe_fit(now=40.0)
    x = al.x.copy()
    for i in range(1350):                                   # 45 s swaying ~20 cm on the spot (> the window)
        t = 40 + i / 30
        q = np.array([0.08 * np.sin(t), 1.5, 0.04 * np.cos(t)])
        e = np.array([0.0, 10 * np.sin(t / 5), 0.0])
        p = Ry(np.radians(215)) @ q + np.array([-0.4, 0.0, 0.5]) + unity_R(e) @ d
        poses[round(t + al.LAG, 6)] = (p, e)
        al.add(t, q + rng.normal(0, 0.005, 3), lambda tq: poses.get(round(tq, 6)))
    assert al.maybe_fit(now=85.0) is False
    assert np.allclose(al.target, x)


def test_a_fit_started_before_a_tpose_is_dropped():
    """The fit runs in its own thread: one that started before a T-pose reset must not
    overwrite the fresh link with a result from the older data."""
    rng = np.random.default_rng(3)
    al, poses = HmdAlign(), {}
    d = np.array([0.0, 0.06, 0.09])
    _walk(al, poses, (180.0, np.array([0.1, 0.0, 0.2]), d), 0.0, 40, rng)
    real_lsq = __import__("fbt.hmd_align", fromlist=["least_squares"]).least_squares

    def lsq_with_tpose(*a, **k):                             # a T-pose lands during the fit
        if al._resets == 0:
            al.reset_from_pose(np.array([0.0, 1.5, 0.0]), 0.0, np.array([0.0, 1.6, 0.0]), np.zeros(3))
        return real_lsq(*a, **k)
    import fbt.hmd_align as mod
    mod.least_squares = lsq_with_tpose
    try:
        x_reset = None
        assert al.maybe_fit(now=40.0) is False
        x_reset = al.x.copy()
    finally:
        mod.least_squares = real_lsq
    assert np.allclose(al.target, x_reset)
