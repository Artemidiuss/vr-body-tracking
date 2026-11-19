import numpy as np
import pytest

from fbt.filters import OneEuro, SkeletonFit
from fbt.skeleton import IDX
from fbt.synthetic import BodyState, skeleton_from_state
from fbt.trackers import frame, solve, to_unity, unity_euler


def _unity_rot(x, y, z):
    x, y, z = np.radians([x, y, z])
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


@pytest.mark.parametrize("e", [(10, 20, 30), (-40, 170, 5), (0, -90, 0), (60, 0, -20)])
def test_unity_euler_roundtrip(e):
    assert np.allclose(unity_euler(_unity_rot(*e)), e, atol=1e-6)


def test_unity_conventions():
    # Quaternion.Euler(0,90,0) * forward == right in Unity.
    assert np.allclose(_unity_rot(0, 90, 0) @ [0, 0, 1], [1, 0, 0], atol=1e-9)


def test_standing_user_facing_minus_z_world():
    # Synthetic body faces world +Z; in Unity space (Z mirrored) it faces -Z.
    J = skeleton_from_state(BodyState())
    tr = {t.id: t for t in solve(J, enabled=("hip", "left_foot", "right_foot"))}
    hip = tr["1"]
    assert hip.position[1] == pytest.approx(0.92, abs=0.02)
    # Facing -Z in Unity = yaw 180.
    assert abs(abs(hip.rotation[1]) - 180) < 2
    assert abs(hip.rotation[0]) < 3 and abs(hip.rotation[2]) < 3
    # The left foot stays the user's left: user faces -Z, so their left is +X in Unity.
    assert tr["3"].position[0] > tr["4"].position[0]


def test_feet_rotation_follows_turn():
    J = skeleton_from_state(BodyState(yaw=np.radians(90)))
    lf = next(t for t in solve(J) if t.id == "3")
    hip = next(t for t in solve(J) if t.id == "1")
    assert abs(((lf.rotation[1] - hip.rotation[1]) + 180) % 360 - 180) < 10


def test_skeleton_fit_restores_bone_lengths():
    rng = np.random.default_rng(0)
    J = skeleton_from_state(BodyState())
    fit = SkeletonFit(learn_frames=20)
    valid = np.ones(26, bool)
    for _ in range(20):
        fit(J + rng.normal(0, 0.005, J.shape), valid)
    stretched = J.copy()
    stretched[IDX["left_ankle"]] += np.array([0, -0.1, 0])  # rubber shin
    out = fit(stretched, valid)
    shin = np.linalg.norm(out[IDX["left_ankle"]] - out[IDX["left_knee"]])
    assert shin == pytest.approx(0.42, abs=0.02)


def test_skeleton_fit_lets_the_spine_bend():
    """Sitting on the floor the spine is a third shorter: kept as measured, not stretched
    back to the standing length (that pushed the head 12-15 cm up, §29). Legs stay rigid."""
    J = skeleton_from_state(BodyState())
    fit = SkeletonFit(learn_frames=1)
    valid = np.ones(26, bool)
    fit(J, valid)
    L = np.linalg.norm(J[IDX["neck"]] - J[IDX["hip"]])
    slouch = J.copy()
    upper = [IDX[n] for n in ("neck", "left_shoulder", "right_shoulder", "head", "nose", "left_eye", "right_eye",
                              "left_ear", "right_ear", "left_elbow", "right_elbow", "left_wrist", "right_wrist")]
    slouch[upper] -= np.array([0, 0.3 * L, 0])                # spine 30% shorter
    out = fit(slouch, valid)
    assert np.linalg.norm(out[IDX["neck"]] - out[IDX["hip"]]) == pytest.approx(0.7 * L, abs=0.005)
    assert np.allclose(out[IDX["head"]], slouch[IDX["head"]], atol=1e-6)
    squash = J.copy()
    squash[upper] -= np.array([0, 0.6 * L, 0])                # impossible: held at 60%
    out = fit(squash, valid)
    assert np.linalg.norm(out[IDX["neck"]] - out[IDX["hip"]]) == pytest.approx(0.6 * L, abs=0.005)


def test_skeleton_fit_fills_missing_joint():
    J = skeleton_from_state(BodyState())
    fit = SkeletonFit(learn_frames=1)
    valid = np.ones(26, bool)
    fit(J, valid)
    moved = J + np.array([0.3, 0, 0])
    v = valid.copy()
    v[IDX["right_knee"]] = False
    moved[IDX["right_knee"]] = 0  # garbage
    out = fit(moved, v)
    assert np.allclose(out[IDX["right_knee"]], J[IDX["right_knee"]] + [0.3, 0, 0], atol=1e-6)


def test_one_euro_smooths_still_and_follows_motion():
    f = OneEuro()
    rng = np.random.default_rng(1)
    still = [f(np.array([0.0, 0, 0]) + rng.normal(0, 0.01, 3), i / 30)[0] for i in range(60)]
    assert np.std(still[20:]) < 0.004
    f = OneEuro()
    pos = [f(np.array([i * 0.05, 0, 0]), i / 30)[0] for i in range(30)]  # 1.5 m/s
    assert 29 * 0.05 - pos[-1] < 0.08  # lag under 8 cm at walking speed


def test_head_yaw_follows_head_not_body():
    from fbt.trackers import head_tracker
    J = skeleton_from_state(BodyState())
    base = head_tracker(J).rotation[1]
    # Turn only the head 40 degrees: rotate ears/nose/eyes/head around the neck.
    a = np.radians(40)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    T = J.copy()
    for n in ("left_ear", "right_ear", "nose", "left_eye", "right_eye", "head"):
        T[IDX[n]] = (J[IDX[n]] - J[IDX["neck"]]) @ R.T + J[IDX["neck"]]
    turned = head_tracker(T).rotation[1]
    d = (turned - base + 180) % 360 - 180
    assert abs(abs(d) - 40) < 3
    # Body yaw (hip tracker) unchanged.
    assert solve(T)[0].rotation == solve(J)[0].rotation


def test_tpose_detector():
    from fbt.trackers import is_tpose
    J = skeleton_from_state(BodyState())          # arms down
    assert not is_tpose(J)
    T = J.copy()
    for side, sgn in (("left", 1), ("right", -1)):
        sh = J[IDX[f"{side}_shoulder"]]
        T[IDX[f"{side}_elbow"]] = sh + [sgn * 0.29, 0, 0]
        T[IDX[f"{side}_wrist"]] = sh + [sgn * 0.55, 0, 0]
    assert is_tpose(T)


def test_tpose_not_taken_lying_down():
    """Lying on the floor with the arms spread: wrists wide apart and level with the
    shoulders, but it must not re-align the trackers."""
    from fbt.trackers import is_tpose
    J = skeleton_from_state(BodyState())
    T = J.copy()
    for side, sgn in (("left", 1), ("right", -1)):
        sh = J[IDX[f"{side}_shoulder"]]
        T[IDX[f"{side}_elbow"]] = sh + [sgn * 0.29, 0, 0]
        T[IDX[f"{side}_wrist"]] = sh + [sgn * 0.55, 0, 0]
    c, s_ = 0.0, 1.0                                   # rotate 90 deg about x: lying on the back
    R = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
    L = (T - T[IDX["hip"]]) @ R.T + [0, 0.15, 0]
    assert is_tpose(T) and not is_tpose(L)
    a = np.radians(20)                                 # leaning back a little, sitting: still counts
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    assert is_tpose((T - T[IDX["hip"]]) @ R.T + T[IDX["hip"]])


def test_exact_foot_mode_transmits_pointed_toes():
    from fbt.trackers import solve
    J = skeleton_from_state(BodyState(ankle=(-1.0, 0.0)))   # left toes pointed down
    flat = next(t for t in solve(J, foot_mode="flat") if t.id == "3").rotation
    exact = next(t for t in solve(J, foot_mode="exact") if t.id == "3").rotation
    # Unity pitch (x) positive = nose down; exact mode must show the extension.
    assert exact[0] - flat[0] > 30


def test_floor_clip_lifts_whole_foot_only_when_below_floor():
    from fbt.filters import FloorClip
    J = skeleton_from_state(BodyState())
    clip = FloorClip()
    assert np.allclose(clip(J), J)                      # standing on the floor: untouched
    sunk = J.copy()
    for n in ("left_ankle", "left_heel", "left_big_toe", "left_small_toe"):
        sunk[IDX[n], 1] -= 0.05
    out = clip(sunk)
    lf = [IDX[n] for n in ("left_heel", "left_big_toe", "left_small_toe")]
    assert abs(out[lf, 1].min()) < 1e-9                 # lifted back to the floor
    assert np.allclose(out[IDX["right_ankle"]], J[IDX["right_ankle"]])


def test_motion_predictor_quiet_at_rest_and_leads_in_motion():
    from fbt.filters import MotionPredictor
    p = MotionPredictor(0.08)
    rng = np.random.default_rng(0)
    base = np.zeros((26, 3))
    rest = [p(base + rng.normal(0, 0.002, base.shape), i / 30) for i in range(30)]
    assert np.abs(rest[-1]).max() < 0.01                       # no extrapolated noise at rest
    p = MotionPredictor(0.08)
    for i in range(30):
        out = p(base + [i * 0.03, 0, 0], i / 30)               # 0.9 m/s
    assert out[0, 0] - 29 * 0.03 > 0.04                        # leads by ~7 cm (clamped at 5)


def test_knee_of_a_straight_leg_stretched_forward_does_not_spin():
    """Sitting with a straight leg stretched forward (or lying, leg raised): the thigh points
    along the body's facing. The knee tracker must stay put (kneecap up) under 3 mm of noise;
    standing, the knee still faces the body's forward."""
    from fbt.hmd_align import unity_R
    J = skeleton_from_state(BodyState())
    knee = lambda P: unity_R({t.id: t for t in solve(P, enabled=("left_knee",))}["5"].rotation)
    fwd_stand = knee(J)[:, 2]
    hip_rot = unity_R({t.id: t for t in solve(J, enabled=("hip",))}["1"].rotation)
    assert np.dot(fwd_stand, hip_rot[:, 2]) > 0.98                    # standing: unchanged
    S = J.copy()
    h = S[IDX["left_hip"]]
    S[IDX["left_knee"]] = h + [0, 0, 0.42]                          # body faces world +Z
    S[IDX["left_ankle"]] = h + [0, 0, 0.84]
    rng = np.random.default_rng(0)
    fw = []
    for _ in range(50):
        P = S.copy()
        P[[IDX["left_hip"], IDX["left_knee"], IDX["left_ankle"]]] += rng.normal(0, 0.003, (3, 3))
        fw.append(knee(P)[:, 2])
    fw = np.array(fw)
    assert np.all(fw[:, 1] > 0.95)                                   # kneecap up (Unity y)
    assert np.degrees(np.arccos(np.clip(fw @ fw.mean(0) / np.linalg.norm(fw.mean(0)), -1, 1))).max() < 15
