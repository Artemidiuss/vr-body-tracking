"""The MacBook's built-in camera as a third view (AVFoundation via PyAV).

Frame timestamps are the capture presentation times AVFoundation puts on
every sample buffer. They are on the host clock, the same clock as
time.perf_counter, so they line up with the phones' sensor timestamps (which
FBTCamSource converts to perf_counter by clock sync). Measured on the M3 Air:
pts interval exactly 33.3 ms, capture -> Python ~52 ms (phones: 86 / 128 ms).

macOS only grants camera access to an app that can show the permission
prompt: FBT.app asks for itself; the CLI tools must run from Terminal.app
(tools/mac_cam_probe.py asks once).
"""

from __future__ import annotations

import time

import av

from .sources import CameraSource, Frame


def mac_focal_range(width: int) -> list[float]:
    """Horizontal FOV 70°..50° -> focal px for this image width."""
    import math
    return [width / 2 / math.tan(math.radians(70) / 2), width / 2 / math.tan(math.radians(50) / 2)]


class MacCamSource(CameraSource):
    def __init__(self, camera_id: str = "mac", device: str = "0", size: str = "1280x720",
                 fps: int = 30):
        super().__init__(camera_id)
        self.host = "mac"
        self.container = av.open(f"{device}:none", format="avfoundation",
                                 options={"framerate": str(fps), "video_size": size,
                                          "pixel_format": "nv12"})
        self.stream = self.container.streams.video[0]
        w, h = self.stream.codec_context.width, self.stream.codec_context.height
        # Focal range for calibration (macOS reports no field of view): 1080p
        # webcam class, horizontal FOV 50-70°. Fits on our recording: 1115-1216 px.
        self.info = {"device": "MacBook camera", "width": w, "height": h, "fps": fps,
                     "timestamp_source": "host", "focal_range_px": mac_focal_range(w)}
        self.latency_ms: list[float] = []
        self._i = 0

    def _packet(self):
        while not self._stop.is_set():
            try:
                return next(self.container.demux(self.stream))
            except BlockingIOError:      # AVFoundation has no new frame yet
                time.sleep(0.002)
            except StopIteration:
                return None
        return None

    def _read(self) -> Frame | None:
        while True:
            try:
                pkt = self._packet()
            except (av.error.FFmpegError, OSError):
                return None
            if pkt is None:
                return None
            frames = pkt.decode()
            if not frames:
                continue
            fr = frames[-1]
            img = fr.to_ndarray(format="bgr24")
            now = time.perf_counter()
            t = float(fr.pts * fr.time_base) if fr.pts is not None else now
            if not 0 <= now - t < 1.0:   # not on the host clock: fall back to arrival time
                t = now - 0.05
            self.latency_ms.append((now - t) * 1000)
            self.latency_ms = self.latency_ms[-300:]
            f = Frame(self.camera_id, self._i, t, img, received_at=now)
            self._i += 1
            return f

    def _close(self) -> None:
        try:
            self.container.close()
        except Exception:
            pass


def open_source(camera_id: str, host: str):
    """'mac' -> MacBook camera ('mac@1920x1080', 'mac:1' for another device),
    'replay:<rec_dir>/camN' -> a recording played live,
    anything else -> FBT Cam phone 'ip[:port]'."""
    if host.startswith("replay:"):
        import os
        from .sources import ReplaySource
        path = host[len("replay:"):]
        return ReplaySource(camera_id, os.path.dirname(path), os.path.basename(path))
    if host == "mac" or host.startswith(("mac:", "mac@")):
        spec, _, size = host.partition("@")
        dev = spec.partition(":")[2] or "0"
        return MacCamSource(camera_id, device=dev, size=size or "1280x720")
    from .fbtcam import FBTCamSource
    ip, _, port = host.partition(":")
    return FBTCamSource(camera_id, ip, int(port or 5000))
