"""Parts that differ between macOS and Windows: the detector cut for CUDA, the device choice,
and the OS helpers. All run on the CPU, on either system."""
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from fbt.pose import resolve_device, split_at_nms


def _tiny_detector():
    """boxes(1,N,4) and scores(1,1,N) -> a scaling op -> NonMaxSuppression -> gather the boxes."""
    n = 6
    nodes = [
        helper.make_node("Mul", ["boxes_in", "two"], ["boxes"]),
        helper.make_node("Sigmoid", ["scores_in"], ["scores"]),
        helper.make_node("NonMaxSuppression", ["boxes", "scores", "maxk", "iou", "thr"], ["sel"]),
        helper.make_node("Gather", ["sel", "two_i"], ["box_idx"], axis=1),
        helper.make_node("Gather", ["boxes", "box_idx"], ["picked"], axis=1),
    ]
    inits = [
        helper.make_tensor("two", TensorProto.FLOAT, [], [2.0]),
        helper.make_tensor("two_i", TensorProto.INT64, [], [2]),
        helper.make_tensor("maxk", TensorProto.INT64, [1], [10]),
        helper.make_tensor("iou", TensorProto.FLOAT, [1], [0.5]),
        helper.make_tensor("thr", TensorProto.FLOAT, [1], [0.3]),
    ]
    g = helper.make_graph(nodes, "det",
                          [helper.make_tensor_value_info("boxes_in", TensorProto.FLOAT, [1, n, 4]),
                           helper.make_tensor_value_info("scores_in", TensorProto.FLOAT, [1, 1, n])],
                          [helper.make_tensor_value_info("picked", TensorProto.FLOAT, None)], initializer=inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)


def _run(model, feed):
    s = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [i.name for i in s.get_inputs()]
    return s.run(None, {k: feed[k] for k in names}), [o.name for o in s.get_outputs()]


def test_detector_cut_at_nms_gives_the_same_boxes():
    m = _tiny_detector()
    rng = np.random.default_rng(0)
    xy = rng.random((1, 6, 2), dtype=np.float32) * 50
    feed = {"boxes_in": np.concatenate([xy, xy + 10], axis=2), "scores_in": rng.normal(0, 2, (1, 1, 6)).astype(np.float32)}
    (whole,), _ = _run(m, feed)
    a, b = split_at_nms(m)
    assert not any(n.op_type == "NonMaxSuppression" for n in a.graph.node)       # the GPU part
    assert any(n.op_type == "NonMaxSuppression" for n in b.graph.node)           # the CPU part
    mid, mid_names = _run(a, feed)
    (parts,), _ = _run(b, dict(zip(mid_names, mid)))
    assert whole.shape == parts.shape and np.allclose(whole, parts)


def test_gpu_means_coreml_on_a_mac_and_cuda_elsewhere():
    assert resolve_device("gpu") == ("gpu" if sys.platform == "darwin" else "cuda")
    assert resolve_device("cpu") == "cpu" and resolve_device("cuda") == "cuda"


def test_os_helpers_do_not_fail():
    from fbt.system import KeepAwake, battery, ping
    b = battery()
    assert b is None or (0 <= b["percent"] <= 100 and isinstance(b["charging"], bool))
    assert ping("127.0.0.1")
    k = KeepAwake()
    k.start()
    k.stop()
