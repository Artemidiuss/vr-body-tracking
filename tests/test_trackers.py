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


def test_hands_up_detector():
    from fbt.trackers import is_hands_up
    J = skeleton_from_state(BodyState())
    assert not is_hands_up(J)
    U = J.copy()
    for side in ("left", "right"):
        U[IDX[f"{side}_wrist"]] = J[IDX["head"]] + [0.1 if side == "left" else -0.1, 0.25, 0]
    assert is_hands_up(U)
