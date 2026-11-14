"""Camera sources. The hub only sees `Frame`s; it never knows whether they
come from a phone, a recorded file, or a synthetic render.

All sources expose `latest()` semantics: a consumer gets the newest frame and
older ones are dropped, so a slow consumer never builds up a queue (the main
source of growing latency).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Frame:
    camera_id: str
    index: int
    timestamp: float          # capture time, seconds (source clock)
    image: np.ndarray         # BGR
    received_at: float = field(default_factory=time.perf_counter)


class CameraSource:
    """Base class: background reader keeping only the newest frame."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._consumed_index = -1
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.finished = threading.Event()
        self.dropped = 0

    # -- subclass API -------------------------------------------------------
    def _read(self) -> Frame | None:
        """Blocking read of the next frame; None means end of stream."""
        raise NotImplementedError

    def _close(self) -> None:
        pass

    # -- public API ---------------------------------------------------------
    def start(self) -> "CameraSource":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self._read()
                if frame is None:
                    break
                with self._lock:
                    if self._latest is not None and self._latest.index > self._consumed_index:
                        self.dropped += 1
                    self._latest = frame
        finally:
            self._close()
            self.finished.set()

    def latest(self, newer_than: int = -1) -> Frame | None:
        """Newest frame with index > newer_than, or None if nothing new."""
        with self._lock:
            f = self._latest
            if f is None or f.index <= newer_than:
                return None
            self._consumed_index = f.index
            return f

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


class OpenCVSource(CameraSource):
    """Any OpenCV-readable stream: webcam index, RTSP/HTTP URL (IP Webcam,
    DroidCam), or a video file played as fast as it decodes."""

    def __init__(self, camera_id: str, uri: str | int, width: int | None = None,
                 height: int | None = None):
        super().__init__(camera_id)
        self.cap = cv2.VideoCapture(uri)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {uri!r}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._i = 0

    def _read(self) -> Frame | None:
        ok, img = self.cap.read()
        if not ok:
            return None
        f = Frame(self.camera_id, self._i, time.perf_counter(), img)
        self._i += 1
        return f

    def _close(self) -> None:
        self.cap.release()


class VideoFileSource(CameraSource):
    """Recorded video replayed like a live camera.

    realtime=True  -> frames are released at the file's FPS (emulates a phone)
    realtime=False -> every frame is delivered, no dropping (batch metrics)
    start_offset   -> seconds to skip, used to align recordings from
                      different phones (clap/flash sync)
    """

    def __init__(self, camera_id: str, path: str, realtime: bool = True,
                 start_offset: float = 0.0, loop: bool = False):
        super().__init__(camera_id)
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path!r}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.realtime = realtime
        self.loop = loop
        if start_offset > 0:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start_offset * 1000)
        self._i = 0
        self._t0: float | None = None

    def _read(self) -> Frame | None:
        ok, img = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = self.cap.read()
        if not ok:
            return None
        media_t = self._i / self.fps
        if self.realtime:
            if self._t0 is None:
                self._t0 = time.perf_counter()
            delay = self._t0 + media_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        f = Frame(self.camera_id, self._i, media_t, img)
        self._i += 1
        return f

    def _close(self) -> None:
        self.cap.release()

    # Non-threaded sequential access for batch processing.
    def frames(self):
        while True:
            f = self._read()
            if f is None:
                return
            yield f


class NetworkEmulator(CameraSource):
    """Wraps a source and injects latency, jitter and frame loss, to test the
    hub under bad Wi-Fi without bad Wi-Fi."""

    def __init__(self, inner: CameraSource, latency_ms: float = 30,
                 jitter_ms: float = 10, drop_rate: float = 0.0, seed: int = 0):
        super().__init__(inner.camera_id)
        self.inner = inner
        self.latency = latency_ms / 1000
        self.jitter = jitter_ms / 1000
        self.drop_rate = drop_rate
        self.rng = np.random.default_rng(seed)
        self._last = -1

    def start(self):
        self.inner.start()
        return super().start()

    def _read(self) -> Frame | None:
        while True:
            f = self.inner.latest(self._last)
            if f is None:
                if self.inner.finished.is_set():
                    return None
                time.sleep(0.001)
                continue
            self._last = f.index
            if self.rng.random() < self.drop_rate:
                continue
            d = max(0.0, self.latency + self.rng.normal(0, self.jitter))
            time.sleep(d)
            return f

    def stop(self):
        self.inner.stop()
        super().stop()
