"""Camera space -> VRChat space from headset poses (synthetic)."""
import numpy as np

from fbt.hmd_align import HmdAlign, Ry, unity_R


def test_recovers_transform_and_head_offset():
    rng = np.random.default_rng(0)
    yaw, t0, s, d = np.radians(201.0), np.array([0.02, -0.1, -0.2]), 1.05, np.array([-0.01, 0.06, 0.09])
    al = HmdAlign()
    poses = {}
    for i in range(1200):                                   # 40 s of walking around at 30 fps
        t = i / 30
        q = np.array([0.8 * np.sin(t / 3), 1.5 + 0.2 * np.sin(t / 2), 0.6 * np.cos(t / 4)])
        e = np.array([10 * np.sin(t), np.degrees(t / 2) % 360, 3 * np.cos(t)])
        p = s * (Ry(yaw) @ q) + t0 + unity_R(e) @ d
        poses[round(t + al.LAG, 6)] = (p, e)
        al.add(t, q + rng.normal(0, 0.01, 3), lambda tq: poses.get(round(tq, 6)))
    for k in range(40):                                     # the scale converges slowly (by design)
        al.maybe_fit(now=100.0 + 2 * k)
    assert al.ready and al.err < 0.02
    assert abs(np.exp(al.log_s) - 1.05) < 0.02
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
