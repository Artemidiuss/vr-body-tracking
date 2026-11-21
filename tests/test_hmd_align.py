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


def _sway(al, poses, yaw, t0, d, t_from, seconds, rng):
    """Swaying ~16 cm on the spot, turning the head: the headset seen through (yaw, t0, d)."""
    for i in range(int(seconds * 30)):
        t = t_from + i / 30
        q = np.array([0.08 * np.sin(t), 1.5, 0.04 * np.cos(t)])
        e = np.array([0.0, 10 * np.sin(t / 5), 0.0])
        p = Ry(np.radians(yaw)) @ q + t0 + unity_R(e) @ d
        poses[round(t + al.LAG, 6)] = (p, e)
        al.add(t, q + rng.normal(0, 0.005, 3), lambda tq: poses.get(round(tq, 6)))


def test_no_big_correction_without_walking():
    """Standing almost still, the yaw is not certain: a big change (8 deg here, the head ~7 cm off
    the link, so not lost) is not taken."""
    rng = np.random.default_rng(2)
    al, poses = HmdAlign(), {}
    d = np.array([0.0, 0.06, 0.09])
    t_old = np.array([0.1, 0.0, 0.2])
    _walk(al, poses, (180.0, t_old, d), 0.0, 40, rng)
    al.maybe_fit(now=40.0)
    x = al.x.copy()
    qc = np.array([0.0, 1.5, 0.0])                           # same place 7 cm off, turned 8 deg
    t_new = t_old + Ry(np.radians(180)) @ qc - Ry(np.radians(188)) @ qc + np.array([0.07, 0.0, 0.0])
    _sway(al, poses, 188.0, t_new, d, 40.0, 160, rng)
    assert not al.lost
    al.maybe_fit(now=200.0)
    assert abs(np.degrees(np.angle(np.exp(1j * (al.target[0] - x[0]))))) < 5.0     # no big change
    assert np.linalg.norm(al.target[1:4] - x[1:4]) < 0.1


def test_lost_link_waits_for_walking():
    """A recenter the jump check missed, then only swaying on the spot: the link is lost, but a
    yaw from 15 cm of travel is not certain either (15 deg off on a game, §46). It waits for
    walking or a T-pose, which the app asks for."""
    rng = np.random.default_rng(2)
    al, poses = HmdAlign(), {}
    d = np.array([0.0, 0.06, 0.09])
    _walk(al, poses, (180.0, np.array([0.1, 0.0, 0.2]), d), 0.0, 40, rng)
    al.maybe_fit(now=40.0)
    x = al.x.copy()
    _sway(al, poses, 215.0, np.array([-0.4, 0.0, 0.5]), d, 40.0, 45, rng)
    assert al.lost
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


def _still(al, link, t_from, seconds, rng, hmd_rate=25, x0=0.3, move=0.0):
    """Sitting still (or one fast move of `move` m), the headset seen through `link`; headset
    samples go to headset_sample as they arrive, camera frames to add."""
    yaw, t0, d = link
    poses = {}
    for i in range(int(seconds * hmd_rate)):
        t = t_from + i / hmd_rate
        q = np.array([x0 + move * min(1.0, max(0.0, (t - t_from - 1.0) * 10)), 0.6, 0.2])
        e = np.array([20.0, 40.0 + (yaw - 170.0) + 5 * np.sin(t), 0.0])   # the space turns the headset too
        p = Ry(np.radians(yaw)) @ q + t0 + unity_R(e) @ d
        poses[t] = (p, e)
        al.headset_sample(t, p, e)
        tc = t - al.LAG - 0.2                                 # the camera frame of that moment, 0.2 s late
        if tc >= t_from:
            al.add(tc, q + rng.normal(0, 0.005, 3), lambda tq: min(poses.items(), key=lambda kv: abs(kv[0] - tq))[1])
    return poses


def test_follows_a_recenter_at_once():
    """Lying still, the Quest recentres by itself (game 19.11 23:23, three times): the head lands
    at the origin facing 0 in one sample. The link follows within a fraction of a second, with
    no walking and no T-pose; a fast real move of the head is not mistaken for one."""
    rng = np.random.default_rng(2)
    d = np.array([0.0, 0.06, 0.09])
    link = (170.0, np.array([0.2, 0.05, -0.05]), d)
    al = HmdAlign()
    al.x = np.r_[np.radians(link[0]), link[1], 0.0, d]
    al.target, al.ready, al.worn = al.x.copy(), True, True
    _still(al, link, 0.0, 3, rng, move=0.25)                  # 25 cm in 0.1 s: a head can do that
    assert al.space_jumps == 0
    q = np.array([0.55, 0.6, 0.2])
    yaw = np.radians(link[0])
    # recentre: the new space has the head root at the origin, yaw 0 -> rotate by -(170 + 40) deg
    a = -np.radians(170.0 + 40.0)
    root_old = Ry(yaw) @ q + link[1]
    new_t0 = Ry(a) @ link[1] - Ry(a) @ root_old * np.array([1, 0, 1])
    new = (np.degrees(yaw + a), new_t0, d)
    _still(al, new, 3.0, 2, rng, x0=0.55)
    assert al.space_jumps == 1
    assert abs((al.yaw_deg() - new[0] + 180) % 360 - 180) < 3.0
    assert np.linalg.norm(al.x[1:4] - new_t0) < 0.03


def test_lying_is_not_learned_from():
    """Lying, the cameras' head is a few cm off and the player barely moves: fits there put the
    body 11-17 cm off (§46). Pairs with the torso not upright are not learned from, so a link
    learned walking stays as it is."""
    rng = np.random.default_rng(4)
    d = np.array([0.0, 0.06, 0.09])
    al, poses = HmdAlign(), {}
    _walk(al, poses, (180.0, np.array([0.1, 0.0, 0.2]), d), 0.0, 40, rng)
    al.maybe_fit(now=40.0)
    x, n = al.x.copy(), len(al.pairs)
    bias = np.array([0.0, 0.05, 0.06])                       # the cameras' head lying: off by ~8 cm
    for i in range(1800):                                    # 60 s lying, moving the head a bit
        t = 40 + i / 30
        q = np.array([0.3 + 0.1 * np.sin(t / 3), 0.25, 0.4 + 0.08 * np.cos(t / 4)])
        e = np.array([-60.0, 30 * np.sin(t / 5), 0.0])
        p = Ry(np.radians(180.0)) @ q + np.array([0.1, 0.0, 0.2]) + unity_R(e) @ d
        poses[round(t + al.LAG, 6)] = (p, e)
        al.add(t, q + bias + rng.normal(0, 0.005, 3), lambda tq: poses.get(round(tq, 6)), upright=False)
    assert len(al.pairs) == n
    al.maybe_fit(now=100.0)
    assert np.allclose(al.target, x)


def test_lost_link_forgets_the_old_pairs():
    """A recenter the jump check missed: once the link is lost (off > 10 cm for 5 s), the pairs
    from before are of the old space and go, so the long window does not hold the fit back."""
    rng = np.random.default_rng(5)
    d = np.array([0.0, 0.06, 0.09])
    al, poses = HmdAlign(), {}
    _walk(al, poses, (180.0, np.array([0.1, 0.0, 0.2]), d), 0.0, 60, rng)
    al.maybe_fit(now=60.0)
    _walk(al, poses, (215.0, np.array([-0.4, 0.0, 0.5]), d), 60.0, 10, rng)     # recentred, not caught
    assert al.lost
    assert al.pairs[0][0] >= 60.0
