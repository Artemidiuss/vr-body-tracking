"""Single-camera 3D pose ("lifting"): the model guesses depth from one image.
Used to compare against two-camera triangulation, and as a future fallback
when only one camera sees the user.

Outputs Halpe26-ordered (26,3) points in metres-ish, root-relative, in the
camera's axes (x right, y down, z away from camera), plus a confidence.
"""

from __future__ import annotations

import os

import numpy as np

from .pose import MODELS_DIR, MediaPipeBackend
from .skeleton import IDX, MEDIAPIPE_TO_HALPE, NUM_KEYPOINTS

# COCO-WholeBody (133) body+feet -> Halpe26.
_COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
           "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
           "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
WHOLEBODY_TO_HALPE = {i: n for i, n in enumerate(_COCO17)}
WHOLEBODY_TO_HALPE.update({17: "left_big_toe", 18: "left_small_toe", 19: "left_heel",
                           20: "right_big_toe", 21: "right_small_toe", 22: "right_heel"})


def _fill_mids(P, S):
    for dst, a, b in (("hip", "left_hip", "right_hip"), ("neck", "left_shoulder", "right_shoulder"),
                      ("head", "left_ear", "right_ear")):
        P[IDX[dst]] = (P[IDX[a]] + P[IDX[b]]) / 2
        S[IDX[dst]] = min(S[IDX[a]], S[IDX[b]])


class MediaPipeMono3D:
    """MediaPipe Pose world landmarks: metric, hip-centred, camera axes."""

    name = "mediapipe-full-3d"

    def __init__(self, variant: str = "full"):
        self.mp = MediaPipeBackend(variant)  # reuse model download/setup
        self.name = f"mediapipe-{variant}-3d"

    def __call__(self, image):
        import mediapipe as mp
        rgb = image[:, :, ::-1].copy()
        self.mp._ts += 33
        res = self.mp.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.mp._ts)
        if not res.pose_world_landmarks:
            return None
        lms = res.pose_world_landmarks[0]
        P = np.zeros((NUM_KEYPOINTS, 3))
        S = np.zeros(NUM_KEYPOINTS)
        for i, n in MEDIAPIPE_TO_HALPE.items():
            P[IDX[n]] = (lms[i].x, lms[i].y, lms[i].z)
            S[IDX[n]] = lms[i].visibility or 0.0
        for side in ("left", "right"):
            P[IDX[f"{side}_small_toe"]] = P[IDX[f"{side}_big_toe"]]
            S[IDX[f"{side}_small_toe"]] = S[IDX[f"{side}_big_toe"]] * 0.5
        _fill_mids(P, S)
        return P - P[IDX["hip"]], S


class RTMW3DMono3D:
    """RTMW3D-x (rtmlib Wholebody3d): whole-body 3D from one image."""

    name = "rtmw3d-x"

    def __init__(self):
        from rtmlib import Wholebody3d
        self.model = Wholebody3d(backend="onnxruntime", device="cpu")

    def __call__(self, image):
        kp3d, scores, _, _ = self.model(image)
        if kp3d is None or len(kp3d) == 0:
            return None
        best = int(np.argmax(np.asarray(scores).mean(axis=1)))
        k, s = np.asarray(kp3d[best], float), np.asarray(scores[best], float)
        P = np.zeros((NUM_KEYPOINTS, 3))
        S = np.zeros(NUM_KEYPOINTS)
        for i, n in WHOLEBODY_TO_HALPE.items():
            P[IDX[n]] = k[i]
            S[IDX[n]] = s[i]
        _fill_mids(P, S)
        return P - P[IDX["hip"]], np.clip(S, 0, 1)


def make_mono(spec: str):
    if spec.startswith("mediapipe"):
        return MediaPipeMono3D(spec.split("-")[1] if "-" in spec else "full")
    if spec == "rtmw3d":
        return RTMW3DMono3D()
    raise ValueError(spec)
