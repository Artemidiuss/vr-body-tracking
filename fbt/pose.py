"""2D pose backends. Each returns keypoints in Halpe26 order (see skeleton.py)
so models can be swapped and compared without touching the rest of the hub.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np

from .skeleton import IDX, MEDIAPIPE_TO_HALPE, NUM_KEYPOINTS

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


COREML_UNITS = {"gpu": "CPUAndGPU", "ane": "CPUAndNeuralEngine", "all": "ALL"}


def coreml_session(onnx_path: str, device: str = "gpu"):
    """ONNX Runtime session on Apple GPU / Neural Engine via CoreML.

    rtmlib's ONNX files have a dynamic batch dimension; CoreML then rejects
    almost the whole graph (2 of 247 nodes) and everything silently runs on
    CPU. Pinning batch=1 and folding the shape ops lets CoreML take all
    nodes: RTMPose-m goes from ~17 ms (CPU) to ~5 ms (GPU) on an M3.
    The fixed model is cached next to the other models.
    """
    import onnx
    import onnxruntime as ort
    import onnxslim

    base = os.path.splitext(os.path.basename(onnx_path))[0]
    fixed = os.path.join(MODELS_DIR, f"{base}.static.onnx")
    if not os.path.exists(fixed):
        m = onnx.load(onnx_path)
        # Only inputs: outputs may be genuinely dynamic (detector box count).
        for t in m.graph.input:
            for d in t.type.tensor_type.shape.dim:
                if d.dim_param:
                    d.Clear()
                    d.dim_value = 1
        os.makedirs(MODELS_DIR, exist_ok=True)
        onnx.save(onnxslim.slim(m), fixed)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(fixed, sess_options=so, providers=[
        ("CoreMLExecutionProvider", {
            "ModelFormat": "MLProgram",
            "MLComputeUnits": COREML_UNITS[device],
            "ModelCacheDirectory": os.path.join(MODELS_DIR, "coreml_cache"),
        }),
        "CPUExecutionProvider",
    ])


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

    VARIANTS = {"s": "lightweight", "m": "balanced", "x": "performance"}
    # yolox-m costs ~200 ms per call on M3 CPU and causes latency spikes;
    # yolox-tiny is enough to find a single person.
    DETECTORS = {
        "tiny": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                 "onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip", (416, 416)),
        "m": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
              "onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip", (640, 640)),
    }

    def __init__(self, variant: str = "m", device: str = "cpu",
                 det_frequency: int = 1, detector: str = "tiny",
                 threads: int | None = None):
        from functools import partial

        from rtmlib import BodyWithFeet, PoseTracker

        self.name = f"rtmpose-{variant}-halpe26-det{detector}-{device}"
        det_url, det_size = self.DETECTORS[detector]
        solution = partial(BodyWithFeet, mode=self.VARIANTS[variant],
                           det=det_url, det_input_size=det_size)
        self.tracker = PoseTracker(
            solution, det_frequency=det_frequency,
            tracking=False, backend="onnxruntime", device="cpu",
        )
        if device in COREML_UNITS:
            for model in (self.tracker.det_model, self.tracker.pose_model):
                model.session = coreml_session(model.session._model_path, device)
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
    """'rtmpose-m:gpu', 'rtmpose-x+detm:cpu', 'mediapipe-full', ...  (devices: cpu, gpu, ane, all)"""
    name, _, device = spec.partition(":")
    name, _, det = name.partition("+det")
    family, _, variant = name.partition("-")
    if family == "rtmpose":
        return RTMPoseBackend(variant or "m", device or "cpu", detector=det or "tiny")
    if family == "mediapipe":
        return MediaPipeBackend(variant or "full")
    raise ValueError(f"unknown backend {spec!r}")
