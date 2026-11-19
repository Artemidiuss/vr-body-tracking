"""2D pose backends. Each returns keypoints in Halpe26 order (see skeleton.py)
so models can be swapped and compared without touching the rest of the hub.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
import threading
import urllib.request
from dataclasses import dataclass

import numpy as np

from .skeleton import IDX, MEDIAPIPE_TO_HALPE, NUM_KEYPOINTS

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


COREML_UNITS = {"gpu": "CPUAndGPU", "ane": "CPUAndNeuralEngine", "all": "ALL"}
_PREP_LOCK = threading.Lock()      # the camera threads build their backends at the same time


def resolve_device(device: str) -> str:
    """'gpu' is the Apple GPU (CoreML) on a Mac and an NVIDIA GPU (CUDA) elsewhere."""
    if device == "gpu" and sys.platform != "darwin":
        return "cuda"
    return device


def _save_atomic(model, path: str) -> None:
    import onnx
    tmp = f"{path}.{os.getpid()}.tmp"
    onnx.save(model, tmp)
    os.replace(tmp, path)


def static_model(onnx_path: str) -> str:
    """rtmlib's ONNX files have a dynamic batch dimension. Pinned to 1 and with the shape ops
    folded, CoreML takes the whole graph (else 2 of 247 nodes, the rest silently on the CPU)
    and CUDA gets fixed shapes. Cached in models/ as <name>.static.onnx."""
    import onnx
    import onnxslim

    base = os.path.splitext(os.path.basename(onnx_path))[0]
    fixed = os.path.join(MODELS_DIR, f"{base}.static.onnx")
    with _PREP_LOCK:
        if not os.path.exists(fixed):
            m = onnx.load(onnx_path)
            # Only inputs: outputs may be genuinely dynamic (detector box count).
            for t in m.graph.input:
                for d in t.type.tensor_type.shape.dim:
                    if d.dim_param:
                        d.Clear()
                        d.dim_value = 1
            os.makedirs(MODELS_DIR, exist_ok=True)
            _save_atomic(onnxslim.slim(m), fixed)
    return fixed


def coreml_session(onnx_path: str, device: str = "gpu"):
    """ONNX Runtime session on Apple GPU / Neural Engine via CoreML.
    RTMPose-m goes from ~17 ms (CPU) to ~5 ms (GPU) on an M3 (static_model is what makes it work).
    """
    import onnxruntime as ort

    fixed = static_model(onnx_path)
    so = ort.SessionOptions()
    # 4 = fatal only: frames without a person make the CoreML EP log an error per frame.
    so.log_severity_level = 4
    return ort.InferenceSession(fixed, sess_options=so, providers=[
        ("CoreMLExecutionProvider", {
            "ModelFormat": "MLProgram",
            "MLComputeUnits": COREML_UNITS[device],
            "ModelCacheDirectory": os.path.join(MODELS_DIR, "coreml_cache"),
        }),
        "CPUExecutionProvider",
    ])


# ---------------------------------------------------------------- NVIDIA (CUDA), e.g. on Windows
_cuda_ready = False


def _cuda_libraries() -> None:
    """onnxruntime-gpu with CUDA and cuDNN from pip (the nvidia-* packages, no CUDA install).
    cuDNN loads its sub-libraries by name while running, so their folders go on PATH too:
    otherwise the first convolution fails and everything quietly falls back to the CPU."""
    global _cuda_ready
    if _cuda_ready:
        return
    spec = importlib.util.find_spec("nvidia")
    for root in (spec.submodule_search_locations or []) if spec else []:
        for d in glob.glob(os.path.join(root, "*", "bin")):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(d)
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    _cuda_ready = True


def _session_options():
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    # One session per model per camera. With ORT's default intra-op pool (all cores, spinning)
    # in each of them, the models run side by side crawled: RTMPose-m 384x288 for 3 cameras
    # gave 5 poses/s instead of 30 on a GTX 1060 (the GPU work does not need these threads).
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return so


def cuda_session(onnx_path: str):
    import onnxruntime as ort
    _cuda_libraries()
    s = ort.InferenceSession(onnx_path, sess_options=_session_options(),
                             providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    if s.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA недоступна: перевір драйвер NVIDIA і пакет onnxruntime-gpu "
                           "(windows/setup.ps1)")
    return s


def split_at_nms(m):
    """(part before the NonMaxSuppression, the NMS and what follows) of an ONNX model; the
    tensors that cross the cut are part A's outputs and part B's inputs, in the same order."""
    import onnx
    m = onnx.shape_inference.infer_shapes(m)      # the cut tensors need shapes (Extractor)
    nodes = list(m.graph.node)
    k = next(i for i, n in enumerate(nodes) if n.op_type == "NonMaxSuppression")
    before = {o for n in nodes[:k] for o in n.output} | {i.name for i in m.graph.input}
    inits = {i.name for i in m.graph.initializer}
    cut = []          # tensors made before the NMS and used by it or after it
    for n in nodes[k:]:
        for i in n.input:
            if i in before and i not in inits and i not in cut:
                cut.append(i)
    ex = onnx.utils.Extractor(m)
    return (ex.extract_model([i.name for i in m.graph.input], cut),
            ex.extract_model(cut, [o.name for o in m.graph.output]))


class SplitDetector:
    """The person detector (YOLOX) on the GPU, but its NonMaxSuppression step and what follows
    it on the CPU: ORT's CUDA NonMaxSuppression took 9 of yolox-tiny's 16 ms on a GTX 1060
    (7 ms without it, the CPU part 0.4 ms). The same model cut in two, so the same boxes
    (300 recorded frames: max difference 1e-4). On a Mac CoreML has no NMS and ORT already
    ran that step on the CPU. Looks like an ONNX Runtime session to rtmlib."""

    def __init__(self, onnx_path: str):
        import onnx
        import onnxruntime as ort

        static = static_model(onnx_path)
        a_path, b_path = static.replace(".onnx", ".gpu.onnx"), static.replace(".onnx", ".nms.onnx")
        with _PREP_LOCK:
            if not (os.path.exists(a_path) and os.path.exists(b_path)):
                a, b = split_at_nms(onnx.load(static))
                _save_atomic(a, a_path)
                _save_atomic(b, b_path)
        self.gpu = cuda_session(a_path)
        self.cpu = ort.InferenceSession(b_path, sess_options=_session_options(),
                                        providers=["CPUExecutionProvider"])
        self.cut = [i.name for i in self.cpu.get_inputs()]

    def get_inputs(self):
        return self.gpu.get_inputs()

    def get_outputs(self):
        return self.cpu.get_outputs()

    def run(self, output_names, feed):
        mid = self.gpu.run(self.cut, feed)
        return self.cpu.run(output_names, dict(zip(self.cut, mid)))


@dataclass
class Pose2D:
    keypoints: np.ndarray  # (26, 2) pixel coords
    scores: np.ndarray     # (26,) confidence 0..1

    @staticmethod
    def empty() -> "Pose2D":
        return Pose2D(np.zeros((NUM_KEYPOINTS, 2), np.float32),
                      np.zeros(NUM_KEYPOINTS, np.float32))


class PoseBackend:
    name = "base"

    def __call__(self, image: np.ndarray) -> Pose2D | None:
        raise NotImplementedError


class RTMPoseBackend(PoseBackend):
    """RTMPose Halpe26 via rtmlib (ONNX Runtime).

    The person detector runs only every `det_frequency` frames; in between the
    box is taken from the previous skeleton, so most frames cost only the
    pose model.
    """

    _URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    # Halpe26 (body7) sizes, input w x h. s, m and x are the files of rtmlib's lightweight /
    # balanced / performance modes. Accuracy and cost (mmpose model zoo, Body8 AUC, GFLOPs):
    # m 71.9 / 1.95, l 73.2 / 4.19, m384 73.6 / 4.37, l384 74.4 / 9.40, x 74.8 / 17.3.
    VARIANTS = {
        "s": ("rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip", (192, 256)),
        "m": ("rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip", (192, 256)),
        "l": ("rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.zip", (192, 256)),
        "m384": ("rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.zip", (288, 384)),
        "l384": ("rtmpose-l_simcc-body7_pt-body7-halpe26_700e-384x288-734182ce_20230605.zip", (288, 384)),
        "x": ("rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip", (288, 384)),
    }
    # yolox-m costs ~200 ms per call on M3 CPU and causes latency spikes;
    # yolox-tiny is enough to find a single person.
    DETECTORS = {
        "tiny": (_URL + "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip", (416, 416)),
        "m": (_URL + "yolox_m_8xb8-300e_humanart-c2c7a14a.zip", (640, 640)),
    }

    def __init__(self, variant: str = "m", device: str = "cpu",
                 det_frequency: int = 1, detector: str = "tiny",
                 threads: int | None = None):
        from functools import partial

        from rtmlib import BodyWithFeet, PoseTracker

        device = resolve_device(device)
        self.name = f"rtmpose-{variant}-halpe26-det{detector}-{device}"
        det_url, det_size = self.DETECTORS[detector]
        pose_zip, pose_size = self.VARIANTS[variant]
        solution = partial(BodyWithFeet, pose=self._URL + pose_zip, pose_input_size=pose_size,
                           det=det_url, det_input_size=det_size)
        self.tracker = PoseTracker(
            solution, det_frequency=det_frequency,
            tracking=False, backend="onnxruntime", device="cpu",
        )
        if device in COREML_UNITS:
            for model in (self.tracker.det_model, self.tracker.pose_model):
                model.session = coreml_session(model.session._model_path, device)
        elif device == "cuda":
            det, pose = self.tracker.det_model, self.tracker.pose_model
            det.session = SplitDetector(det.session._model_path)
            pose.session = cuda_session(static_model(pose.session._model_path))
        elif threads and device == "cpu":
            # By default every ONNX session grabs all cores; with one session
            # per camera they fight each other. Cap threads per session.
            import onnxruntime as ort

            for model in (self.tracker.det_model, self.tracker.pose_model):
                so = ort.SessionOptions()
                so.intra_op_num_threads = threads
                old = model.session
                model.session = ort.InferenceSession(
                    old._model_path, sess_options=so, providers=["CPUExecutionProvider"])

    def __call__(self, image):
        try:
            kpts, scores = self.tracker(image)
        except IndexError:
            # rtmlib bug: with det_frequency=1 PoseTracker still runs its IoU
            # tracker, whose id list goes out of sync when the person leaves
            # and re-enters the frame ("pop from empty list"). Its state is
            # unused with detection on every frame, so reset and retry once.
            self.tracker.reset()
            kpts, scores = self.tracker(image)
        if kpts is None or len(kpts) == 0:
            return None
        # Single-user app: take the most confident person.
        best = int(np.argmax(np.asarray(scores).mean(axis=1)))
        return Pose2D(np.asarray(kpts[best], np.float32),
                      np.clip(np.asarray(scores[best], np.float32), 0, 1))


class MediaPipeBackend(PoseBackend):
    """MediaPipe Pose Landmarker (Tasks API), mapped to Halpe26."""

    URLS = {
        v: f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
           f"pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
        for v in ("lite", "full", "heavy")
    }

    def __init__(self, variant: str = "full"):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.name = f"mediapipe-{variant}"
        path = os.path.join(MODELS_DIR, f"pose_landmarker_{variant}.task")
        if not os.path.exists(path):
            os.makedirs(MODELS_DIR, exist_ok=True)
            urllib.request.urlretrieve(self.URLS[variant], path)
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._mp = mp
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)
        self._ts = 0

    def __call__(self, image):
        h, w = image.shape[:2]
        rgb = image[:, :, ::-1].copy()
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._ts += 33  # VIDEO mode needs monotonically increasing ms
        res = self.landmarker.detect_for_video(mp_img, self._ts)
        if not res.pose_landmarks:
            return None
        lms = res.pose_landmarks[0]
        out = Pose2D.empty()
        for mp_i, name in MEDIAPIPE_TO_HALPE.items():
            lm = lms[mp_i]
            out.keypoints[IDX[name]] = (lm.x * w, lm.y * h)
            vis = lm.visibility if lm.visibility is not None else 1.0
            out.scores[IDX[name]] = vis

        def mid(a, b, dst):
            ia, ib = IDX[a], IDX[b]
            out.keypoints[IDX[dst]] = (out.keypoints[ia] + out.keypoints[ib]) / 2
            out.scores[IDX[dst]] = min(out.scores[ia], out.scores[ib])

        mid("left_hip", "right_hip", "hip")
        mid("left_shoulder", "right_shoulder", "neck")
        mid("left_ear", "right_ear", "head")
        # MediaPipe has one toe point per foot; reuse it as the small toe
        # with reduced confidence so it is not over-trusted.
        for side in ("left", "right"):
            out.keypoints[IDX[f"{side}_small_toe"]] = out.keypoints[IDX[f"{side}_big_toe"]]
            out.scores[IDX[f"{side}_small_toe"]] = out.scores[IDX[f"{side}_big_toe"]] * 0.5
        return out


def make_backend(spec: str) -> PoseBackend:
    """'rtmpose-m:gpu', 'rtmpose-l384:cuda', 'rtmpose-x+detm:cpu', 'mediapipe-full', ...
    Devices: cpu, gpu (Apple GPU on a Mac, NVIDIA elsewhere), cuda, ane, all."""
    name, _, device = spec.partition(":")
    name, _, det = name.partition("+det")
    family, _, variant = name.partition("-")
    if family == "rtmpose":
        return RTMPoseBackend(variant or "m", device or "cpu", detector=det or "tiny")
    if family == "mediapipe":
        return MediaPipeBackend(variant or "full")
    raise ValueError(f"unknown backend {spec!r}")
