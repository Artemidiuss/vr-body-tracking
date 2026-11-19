import numpy as np

from fbt.camera import ring_of_cameras
from fbt.pose import Pose2D
from fbt.skeleton import IDX
from fbt.synthetic import BodyState, NoiseModel, render_2d, skeleton_from_state
from fbt.triangulate import LR_SWAP, triangulate_pose, triangulate_with_swap_check

CAMS = ring_of_cameras([-45, 45])
J = skeleton_from_state(BodyState(hip=(0.4, -0.1), knee=(0.6, 0.1)))


def test_exact_projection_triangulates_exactly():
    poses = [render_2d(c, J, None) for c in CAMS]
    res = triangulate_pose(CAMS, poses)
    assert res.valid.all()
    assert np.abs(res.points - J).max() < 1e-6


def test_pixel_noise_gives_centimetre_error():
    noise = NoiseModel(pixel_sigma=3.0, dropout=0, lr_swap_front=0, lr_swap_back=0)
    errs = []
    for _ in range(50):
        res = triangulate_pose(CAMS, [render_2d(c, J, noise) for c in CAMS])
        errs.append(np.linalg.norm(res.points - J, axis=1)[res.valid].mean())
    assert np.mean(errs) < 0.03  # < 3 cm average


def test_point_seen_by_one_camera_is_invalid():
    poses = [render_2d(c, J, None) for c in CAMS]
    poses[1].scores[IDX["left_knee"]] = 0.0
    res = triangulate_pose(CAMS, poses)
    assert not res.valid[IDX["left_knee"]]
    assert res.valid[IDX["right_knee"]]


def test_outlier_view_is_dropped_with_three_cameras():
    cams = ring_of_cameras([-45, 0, 45])
    poses = [render_2d(c, J, None) for c in cams]
    poses[1].keypoints[IDX["left_ankle"]] += (80, -60)  # wild detection
    res = triangulate_pose(cams, poses)
    k = IDX["left_ankle"]
    assert res.valid[k] and res.n_views[k] == 2
    assert np.linalg.norm(res.points[k] - J[k]) < 1e-3


def test_left_right_swap_in_one_camera_is_detected():
    poses = [render_2d(c, J, None) for c in CAMS]
    prev = triangulate_pose(CAMS, poses)
    poses[1] = Pose2D(poses[1].keypoints[LR_SWAP], poses[1].scores[LR_SWAP])
    naive = triangulate_pose(CAMS, poses)
    fixed = triangulate_with_swap_check(CAMS, poses, prev=prev)
    assert fixed.swapped == [False, True]
    assert np.abs(fixed.points - J).max() < 1e-6
    # The mis-paired skeleton can be geometrically consistent, but wrong.
    assert np.abs(naive.points - J).max() > 0.02


def test_swap_resolved_without_history_by_body_shape():
    poses = [render_2d(c, J, None) for c in CAMS]
    poses[0] = Pose2D(poses[0].keypoints[LR_SWAP], poses[0].scores[LR_SWAP])
    fixed = triangulate_with_swap_check(CAMS, poses)
    assert np.abs(fixed.points - J).max() < 1e-6


def test_extra_camera_weight_is_capped():
    """A third phone (same focal as the phones) gets the MacBook-level weight, not 1:
    it joins only part of the steps and each join/leave would move the points."""
    from fbt.camera import Camera
    from fbt.triangulate import EXTRA_MAX_WEIGHT, view_weights
    phones = ring_of_cameras([-40, 40], distance=1.6, vfov_deg=96)
    third = Camera.look_at("third", (0.0, 1.0, -2.0), (0.0, 0.9, 0.0), vfov_deg=96, width=phones[0].width, height=phones[0].height)
    mac = Camera.look_at("mac", (0.0, 0.9, -1.5), (0.0, 1.0, 0.0), vfov_deg=30, width=phones[0].width, height=phones[0].height)
    w = view_weights(phones + [third])
    assert w[0] == w[1] == 1 and abs(w[2] - EXTRA_MAX_WEIGHT) < 1e-9
    w_mac = view_weights(phones + [mac])                       # long focal: its own, smaller weight
    assert w_mac[2] < EXTRA_MAX_WEIGHT
    assert abs(w_mac[2] - phones[0].K[0, 0] / mac.K[0, 0]) < 1e-9


def test_extra_camera_at_30_fps_is_not_capped():
    """At 25+ fps a third phone weighs as a phone (its focal weight); the MacBook keeps its
    small focal weight; at 15 fps the cap stays."""
    from fbt.camera import Camera
    from fbt.triangulate import EXTRA_MAX_WEIGHT, extra_cap, view_weights
    phones = ring_of_cameras([-40, 40], distance=1.6, vfov_deg=96)
    third = Camera.look_at("third", (0.0, 1.0, -2.0), (0.0, 0.9, 0.0), vfov_deg=96, width=phones[0].width, height=phones[0].height)
    mac = Camera.look_at("mac", (0.0, 0.9, -1.5), (0.0, 1.0, 0.0), vfov_deg=30, width=phones[0].width, height=phones[0].height)
    assert abs(view_weights(phones + [third], cap=extra_cap(30))[2] - 1.0) < 1e-9
    assert abs(view_weights(phones + [third], cap=extra_cap(15))[2] - EXTRA_MAX_WEIGHT) < 1e-9
    assert abs(view_weights(phones + [mac], cap=extra_cap(30))[2] - phones[0].K[0, 0] / mac.K[0, 0]) < 1e-9
