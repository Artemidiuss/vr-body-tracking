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


def test_upright_direction_ignores_leaning_frames():
    """Crouches and leans to one side must not tilt 'up': it is the standing-straight direction."""
    from fbt.calibrate import upright_direction
    from fbt.skeleton import IDX, NUM_KEYPOINTS
    rng = np.random.default_rng(0)
    n_up, n_lean = 600, 400
    tilt = np.r_[rng.normal(0, 2, n_up), rng.normal(25, 5, n_lean)]      # degrees, towards +x
    a = np.radians(tilt)
    skel = np.full((len(a), NUM_KEYPOINTS, 3), np.nan)
    skel[:, IDX["left_ankle"]] = [-0.1, 0, 0]
    skel[:, IDX["right_ankle"]] = [0.1, 0, 0]
    skel[:, IDX["neck"]] = np.c_[1.4 * np.sin(a), 1.4 * np.cos(a), np.zeros(len(a))]
    up = upright_direction(skel)
    assert np.degrees(np.arccos(up[1])) < 1.0


def _with_gravity(track, cam, world_up=(0.0, 1.0, 0.0)):
    """FBT Cam metadata for a portrait phone whose frames the Mac rotated: sensor size swapped,
    and the accelerometer reading (device axes) that this camera pose gives."""
    u = cam.R @ np.asarray(world_up, float)             # up in camera axes (x right, y down, z fwd)
    g = 9.81 * np.array([u[0], -u[1], -u[2]])           # device axes: X = x, Y = -y, Z = -z
    track.info = {**track.info, "sensor_orientation": 90, "width": track.height, "height": track.width,
                  "gravity": {"v": g.tolist(), "sd_deg": 0.1, "n": 30}}
    return track


def test_gravity_mapping_portrait_phone():
    from fbt.calibrate import gravity_up_in_camera
    t = PoseTrack(np.zeros(1), np.zeros((1, 26, 2)), np.zeros((1, 26)), 480, 640,
                  {"sensor_orientation": 90, "width": 640, "height": 480,
                   "gravity": {"v": [0.0, 9.81, 0.0], "sd_deg": 0.1}})
    assert np.allclose(gravity_up_in_camera(t), [0, -1, 0])          # upright: image up
    t.info["gravity"] = {"v": [0.0, 0.0, 9.81], "sd_deg": 0.1}       # lying face up: lens looks down
    assert np.allclose(gravity_up_in_camera(t), [0, 0, -1])
    t.info["gravity"]["sd_deg"] = 2.0                                # phone was moving
    assert gravity_up_in_camera(t) is None


def test_vertical_is_the_posture_not_the_phones():
    """Phones agreeing on a vertical 3 deg off (camera modules skewed to the accelerometer)
    do not tilt the world: the standing posture sets it, the phones are only logged (§31)."""
    cams = ring_of_cameras([-40, 40], distance=1.2, vfov_deg=96)
    f_px = cams[0].K[0, 0]
    frames = [(t, J) for t, J, _ in motion_sequence(["walk", "squat", "side_step", "kick"], seconds_each=3)]
    noise = NoiseModel(pixel_sigma=2.0, lr_swap_front=0, lr_swap_back=0, dropout=0.02, seed=1)
    fmm = f_px * 3.67 / 1280
    a_ = np.radians(3.0)
    skewed = (np.sin(a_), np.cos(a_), 0.0)
    a = _with_gravity(_track(cams[0], frames, noise, 3.67, fmm), cams[0], skewed)
    b = _with_gravity(_track(cams[1], frames, noise, 3.67, fmm), cams[1], skewed)
    est, meta = calibrate_pair(a, b, height_m=1.75, log=lambda *_: None)
    assert meta["up"]["source"] == "posture"
    assert abs(meta["up"]["posture_vs_phones_deg"] - 3.0) < 1.0
    for e, c in zip(est, cams):
        want = c.R @ np.array([0.0, 1.0, 0.0])          # the true vertical, in camera axes
        got = e.R @ np.array([0.0, 1.0, 0.0])
        assert np.degrees(np.arccos(np.clip(want @ got, -1, 1))) < 1.5   # the posture's own ~1 deg


def test_phone_as_third_camera_uses_its_own_focal():
    """A third phone (ultra-wide, portrait) in the middle, low, aimed at the hips and legs:
    placed by PnP with the focal it reports itself (phone_focal)."""
    from fbt.calibrate import calibrate_extra, phone_focal, triangulate_tracks
    from fbt.camera import Camera

    phones = ring_of_cameras([-40, 40], distance=1.6, vfov_deg=96)
    third = Camera.look_at("third", (0.0, 0.6, -1.8), (0.0, 0.6, 0.0), vfov_deg=96, width=480, height=640)
    frames = [(t, J) for t, J, _ in motion_sequence(["walk", "squat", "side_step", "turn"], seconds_each=3)]
    noise = NoiseModel(pixel_sigma=2.0, lr_swap_front=0, lr_swap_back=0, dropout=0.02, seed=4)
    a = _track(phones[0], frames, noise, 3.67, 2.0)
    b = _track(phones[1], frames, noise, 3.67, 2.0)
    f_true = third.K[0, 0]
    m = _track(third, frames, NoiseModel(pixel_sigma=2.0, lr_swap_front=0, lr_swap_back=0, dropout=0.02, seed=5),
               3.67, f_true * 3.67 / 640)
    f = phone_focal(m.info, m.width, m.height)
    assert abs(f / f_true - 1) < 1e-6
    ts, skel = triangulate_tracks(phones, a, b)
    est, meta = calibrate_extra(m, ts, skel, focal=f, log=lambda *_: None)
    assert np.linalg.norm(est.center - third.center) < 0.03
    # free focal drifts along the viewing axis at the same error: why the phone's value is used
    est_free, meta_free = calibrate_extra(m, ts, skel, log=lambda *_: None)
    assert np.linalg.norm(est_free.center - third.center) > np.linalg.norm(est.center - third.center)
    assert abs(meta_free["reproj_median_px"] - meta["reproj_median_px"]) < 0.5


def test_phone_focal_needs_the_phone_metadata():
    from fbt.calibrate import phone_focal
    assert phone_focal({"device": "MacBook"}, 1280, 720) is None
    assert abs(phone_focal({"focal_mm": 1.65, "sensor_mm": [3.67, 2.75]}, 480, 640) - 287.74) < 0.01
