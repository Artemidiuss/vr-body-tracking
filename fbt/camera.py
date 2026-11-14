"""Pinhole camera model shared by calibration, triangulation and the
synthetic test rig.

World frame: metres, Y up (same as VRChat/Unity, but right-handed here;
the OSC layer converts handedness). Camera frame: OpenCV convention
(x right, y down, z forward).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    name: str
    K: np.ndarray            # 3x3 intrinsics
    R: np.ndarray            # 3x3 world -> camera rotation
    t: np.ndarray            # (3,) world -> camera translation
    width: int
    height: int
    dist: np.ndarray | None = None  # OpenCV distortion coeffs (unused for now)

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix."""
        return self.K @ np.hstack([self.R, self.t[:, None]])

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def to_camera(self, X: np.ndarray) -> np.ndarray:
        return X @ self.R.T + self.t

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,3) world points -> (N,2) pixels and (N,) depth."""
        Xc = self.to_camera(np.atleast_2d(X))
        z = Xc[:, 2]
        uv = (Xc @ self.K.T)[:, :2] / np.maximum(z[:, None], 1e-9)
        return uv, z

    def in_frame(self, uv: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < self.width) & \
               (uv[:, 1] >= 0) & (uv[:, 1] < self.height)

    @staticmethod
    def look_at(name: str, position, target, vfov_deg: float = 70.0,
                width: int = 720, height: int = 1280) -> "Camera":
        """Camera at `position` looking at `target`, world up = +Y.
        Default size = phone held in portrait at 720p."""
        pos = np.asarray(position, float)
        fwd = np.asarray(target, float) - pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd])  # rows = camera axes in world
        f = (height / 2) / np.tan(np.radians(vfov_deg) / 2)
        K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
        return Camera(name, K, R, -R @ pos, width, height)


def ring_of_cameras(angles_deg, distance=2.5, height=0.9, target_height=0.9,
                    **kw) -> list[Camera]:
    """Cameras on a circle around the origin. Angle 0 = straight in front of
    a user facing +Z; positive angles go towards the user's left (+X)."""
    cams = []
    for a in angles_deg:
        r = np.radians(a)
        pos = (distance * np.sin(r), height, distance * np.cos(r))
        cams.append(Camera.look_at(f"cam{a:+.0f}", pos, (0, target_height, 0), **kw))
    return cams
