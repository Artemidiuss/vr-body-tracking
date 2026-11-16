import numpy as np

from fbt.calibrate import PoseTrack, calibrate_pair
from fbt.camera import ring_of_cameras
from fbt.skeleton import IDX
from fbt.synthetic import MOTIONS, NoiseModel, motion_sequence, render_2d


def _track(cam, frames, noise, sensor_long_mm, focal_mm):
    T, K, S = [], [], []
    for t, J in frames:
        p = render_2d(cam, J, noise)
        if p is None:
            continue
        T.append(t); K.append(p.keypoints); S.append(p.scores)
    info = {"focal_mm": focal_mm, "sensor_mm": [sensor_long_mm, sensor_long_mm * 0.75]}
    return PoseTrack(np.array(T), np.array(K), np.array(S), cam.width, cam.height, info)


def test_body_calibration_recovers_cameras():
    cams = ring_of_cameras([-40, 40], distance=1.2, vfov_deg=96)
    # Phone metadata slightly off (5%) — BA must fix the focal length.
    f_px = cams[0].K[0, 0]
    sensor_long = 3.67
    focal_mm_true = f_px * sensor_long / 1280
    frames = [(t, J) for t, J, _ in motion_sequence(["walk", "squat", "side_step", "kick"], seconds_each=3)]
    noise = NoiseModel(pixel_sigma=2.0, lr_swap_front=0, lr_swap_back=0, dropout=0.02, seed=1)
    a = _track(cams[0], frames, noise, sensor_long, focal_mm_true * 1.05)
    b = _track(cams[1], frames, noise, sensor_long, focal_mm_true * 0.95)
    est, meta = calibrate_pair(a, b, height_m=1.75, log=lambda *_: None)
    true_base = np.linalg.norm(cams[0].center - cams[1].center)
    assert abs(meta["baseline_m"] - true_base) / true_base < 0.08
    assert abs(meta["angle_deg"] - 80) < 4
    # Focal length is barely observable from two views: it must at least not
    # drift further from the truth than the phone's own (5% off) value.
    assert abs(est[0].K[0, 0] - f_px) / f_px <= 0.055
    assert abs(est[0].center[1] - 0.9) < 0.1     # camera height above floor


def test_extra_camera_placed_from_phone_skeleton():
    """A third, close, narrow camera behind the user that sees only the upper
    body, with its own clock offset: PnP against the phones' 3D skeleton."""
    from fbt.calibrate import calibrate_extra, triangulate_tracks
    from fbt.camera import Camera

    phones = ring_of_cameras([-40, 40], distance=1.6, vfov_deg=96)
    mac = Camera.look_at("mac", (0.15, 0.85, -1.0), (0.0, 1.15, 0.0), vfov_deg=45, width=1280, height=720)
    frames = [(t, J) for t, J, _ in motion_sequence(["walk", "squat", "side_step", "turn"], seconds_each=3)]
    noise = NoiseModel(pixel_sigma=2.0, lr_swap_front=0, lr_swap_back=0, dropout=0.02, seed=2)
    info = {"focal_mm": 2.0, "sensor_mm": [3.67, 2.75]}
    a = _track(phones[0], frames, noise, 3.67, 2.0)
    b = _track(phones[1], frames, noise, 3.67, 2.0)
    m_noise = NoiseModel(pixel_sigma=3.0, lr_swap_front=0.0, lr_swap_back=0.05, dropout=0.02, seed=3)
    m = _track(mac, frames, m_noise, 3.67, 2.0)
    m.t = m.t - 0.04                    # Mac clock 40 ms early vs the phones
    m.info = {"device": "mac"}
    ts, skel = triangulate_tracks(phones, a, b)
    est, meta = calibrate_extra(m, ts, skel, log=lambda *_: None)
    assert np.linalg.norm(est.center - mac.center) < 0.05
    assert abs(est.K[0, 0] - mac.K[0, 0]) / mac.K[0, 0] < 0.05
    assert abs(meta["time_offset_s"] - 0.04) <= 0.011
    assert meta["reproj_median_px"] < 15   # noise model: ~9 px sigma from behind
