"""Halpe26 keypoint layout — the common skeleton format used across the hub.

Every pose backend converts its output to Halpe26 order, so the rest of the
pipeline (triangulation, tracker solver) never depends on a specific model.
"""

from __future__ import annotations

HALPE26 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck", "hip",
    "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel",
]
NUM_KEYPOINTS = len(HALPE26)
IDX = {name: i for i, name in enumerate(HALPE26)}

# Bones used for drawing and for bone-length constraints later.
BONES = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_ankle", "left_big_toe"),
    ("left_big_toe", "left_small_toe"),
    ("right_ankle", "right_heel"), ("right_ankle", "right_big_toe"),
    ("right_big_toe", "right_small_toe"),
    ("head", "neck"), ("neck", "hip"),
]
BONE_IDX = [(IDX[a], IDX[b]) for a, b in BONES]

# MediaPipe BlazePose (33 points) -> Halpe26. Points MediaPipe lacks
# (head, neck, hip, small toes) are synthesised in the backend.
MEDIAPIPE_TO_HALPE = {
    0: "nose", 2: "left_eye", 5: "right_eye", 7: "left_ear", 8: "right_ear",
    11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow",
    14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
    31: "left_big_toe", 32: "right_big_toe",
}
