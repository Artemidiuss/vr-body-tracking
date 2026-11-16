"""Receiver for our own Android camera app (android/fbtcam, PROTOCOL.md).

Connects to the phone over TCP, decodes H.264 with low delay, rotates to
portrait, and converts every frame's sensor capture time to the Mac clock
(time.perf_counter) using PING/PONG clock sync. So `Frame.timestamp` is the
real moment the photo was taken, and `received_at - timestamp` is the live
end-to-end latency (capture -> decoded on the Mac).
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time

import av
import cv2
import numpy as np

from .sources import CameraSource, Frame

MAGIC = b"FBT1"
HDR = struct.Struct(">4sBBHIqI")  # magic type flags rsv seq ts len  (24 bytes)
HELLO, CONFIG, FRAME, PING, PONG, KEYFRAME_REQ, STATUS, EXPOSURE = range(8)


class ClockSync:
    """offset such that mac_time = phone_time + offset (seconds). Uses the
    sample with the smallest round trip among recent ones."""

    def __init__(self, window: int = 20):
        self.samples: list[tuple[float, float]] = []  # (rtt, offset)
        self.window = window
        self.offset: float | None = None
        self.rtt: float | None = None

    def add(self, mac_sent: float, phone_t: float, mac_recv: float):
        rtt = mac_recv - mac_sent
        off = (mac_sent + mac_recv) / 2 - phone_t
        self.samples.append((rtt, off))
        self.samples = self.samples[-self.window:]
        self.rtt, self.offset = min(self.samples)


class FBTCamSource(CameraSource):
    def __init__(self, camera_id: str, host: str, port: int = 5000,
                 rotate: bool = True, connect_timeout: float = 5.0):
        super().__init__(camera_id)
        self.host, self.port, self.rotate = host, port, rotate
        self.sock = socket.create_connection((host, port), timeout=connect_timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(None)
        self.rf = self.sock.makefile("rb", buffering=0)
        self._send_lock = threading.Lock()
        self.info: dict = {}
        self.clock = ClockSync()
        self.config = b""
        self.codec = av.CodecContext.create("h264", "r")
        self.codec.flags |= av.codec.context.Flags.low_delay
        self.codec.thread_type = "NONE"   # frame threading adds a frame of delay
        self._i = 0
        self.latency_ms: list[float] = []
        # Phone battery / heat (STATUS packet, newer app versions): dict + Mac time received.
        self.status: dict = {}
        self.status_t: float | None = None
        self._pinger = threading.Thread(target=self._ping_loop, daemon=True)

    # -- protocol --------------------------------------------------------
    def _send(self, ptype: int, ts: int = 0):
        with self._send_lock:
            self.sock.sendall(HDR.pack(MAGIC, ptype, 0, 0, 0, ts, 0))

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.rf.read(n - len(buf))
            if not chunk:
                raise ConnectionError("phone closed the stream")
            buf += chunk
        return bytes(buf)

    def _ping_loop(self):
        # A burst at start for a quick first estimate, then once a second.
        n = 0
        while not self._stop.is_set():
            try:
                self._send(PING, time.perf_counter_ns())
            except OSError:
                return
            n += 1
            time.sleep(0.05 if n < 10 else 1.0)

    def request_keyframe(self):
        self._send(KEYFRAME_REQ)

    def remeter_exposure(self):
        """Ask the phone to meter the light again (~3 s) and then hold that exposure.
        Older FBT Cam versions ignore it."""
        try:
            self._send(EXPOSURE)
        except OSError:
            pass

    def start(self):
        super().start()
        self._pinger.start()
        return self

    # -- CameraSource ----------------------------------------------------
    def _read(self) -> Frame | None:
        while True:
            try:
                magic, ptype, flags, _, seq, ts, ln = HDR.unpack(self._recv_exact(HDR.size))
                if magic != MAGIC:
                    raise ConnectionError("bad magic — not an FBT Cam stream")
                payload = self._recv_exact(ln) if ln else b""
            except (ConnectionError, OSError):
                return None
            if ptype == HELLO:
                self.info = json.loads(payload)
            elif ptype == CONFIG:
                self.config = payload
            elif ptype == PONG:
                (phone_ns,) = struct.unpack(">q", payload)
                self.clock.add(ts / 1e9, phone_ns / 1e9, time.perf_counter())
            elif ptype == STATUS:
                try:
                    self.status, self.status_t = json.loads(payload), time.perf_counter()
                except ValueError:
                    pass
            elif ptype == FRAME:
                frame = self._decode(payload, bool(flags & 1), ts)
                if frame is not None:
                    return frame

    def _decode(self, data: bytes, key: bool, ts_us: int) -> Frame | None:
        if key and self.config:
            data = self.config + data
        try:
            frames = self.codec.decode(av.Packet(data))
        except av.error.InvalidDataError:
            self.request_keyframe()
            return None
        if not frames:
            return None
        img = frames[-1].to_ndarray(format="bgr24")
        if self.rotate:
            rot = self.info.get("sensor_orientation", 90)
            if rot == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        now = time.perf_counter()
        if self.clock.offset is not None:
            t = ts_us / 1e6 + self.clock.offset
            self.latency_ms.append((now - t) * 1000)
            self.latency_ms = self.latency_ms[-300:]
        else:
            t = now
        f = Frame(self.camera_id, self._i, t, np.ascontiguousarray(img), received_at=now)
        self._i += 1
        return f

    def _close(self):
        try:
            self.sock.close()
        except OSError:
            pass
