import numpy as np

from fbt.align import SpaceAligner, head_root
from fbt.synthetic import BodyState, skeleton_from_state
from fbt.trackers import solve


def test_zero_puts_head_at_origin_and_faces_unity_forward(tmp_path):
    # User stands at (1.2, 0, -0.7), turned 70 degrees, in our camera world.
    J = skeleton_from_state(BodyState(yaw=np.radians(70), root_xz=(1.2, -0.7)))
    al = SpaceAligner(str(tmp_path / "a.json"))
    al.zero([J] * 5)
    Q = al(J)
    h = head_root(Q)
    assert abs(h[0]) < 1e-6 and abs(h[2]) < 1e-6           # head above origin
    hip = next(t for t in solve(Q) if t.id == "1")
    assert abs(hip.rotation[1]) < 2                          # Unity yaw 0 = forward
    # Persisted and reloaded.
    al2 = SpaceAligner(str(tmp_path / "a.json"))
    assert np.allclose(al2(J), Q)
