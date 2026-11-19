"""What the hub needs from the operating system, for macOS and Windows: a spoken alert, keeping
the computer awake during a game, the laptop battery, and pinging the headset."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading

MAC, WINDOWS = sys.platform == "darwin", sys.platform == "win32"
# console tools started from a windowless app must not flash a console window on Windows
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if WINDOWS else {}

# Windows: speak with an installed Ukrainian voice, if there is one (the text comes in through
# the environment, so quotes and apostrophes in it need no escaping).
_SAY_PS = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
           "$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'uk*' } | Select-Object -First 1; "
           "if ($v) { $s.SelectVoice($v.VoiceInfo.Name); $s.Speak($env:FBT_SAY) }")


def say(text: str) -> None:
    """Spoken alert (Ukrainian voice): macOS `say`; Windows only if a Ukrainian voice is installed."""
    try:
        if MAC:
            subprocess.Popen(["say", "-v", "Lesya", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif WINDOWS:
            subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", _SAY_PS],
                             env={**os.environ, "FBT_SAY": text},
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **_NO_WINDOW)
    except OSError:
        pass


class KeepAwake:
    """Keeps the display and the computer awake while a game runs."""

    def __init__(self):
        self._proc = None
        self._stop = None

    def start(self) -> None:
        self.stop()
        if MAC:
            try:
                self._proc = subprocess.Popen(["caffeinate", "-d", "-i", "-w", str(os.getpid())])
            except OSError:
                self._proc = None
        elif WINDOWS:
            # SetThreadExecutionState holds only while the calling thread lives: a thread of its own
            stop = self._stop = threading.Event()

            def hold():
                import ctypes
                k = ctypes.windll.kernel32
                ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x1, 0x2
                k.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
                stop.wait()
                k.SetThreadExecutionState(ES_CONTINUOUS)
            threading.Thread(target=hold, daemon=True, name="keep-awake").start()

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc = None
        if self._stop:
            self._stop.set()
            self._stop = None


def battery() -> dict | None:
    """{'percent': 57, 'charging': False} of a laptop, or None (desktop, unknown)."""
    if MAC:
        try:
            out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=3).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
        m = re.search(r"(\d+)%;\s*([a-zA-Z ]+);", out)
        if not m:
            return None
        return {"percent": int(m.group(1)), "charging": "AC Power" in out}
    if WINDOWS:
        import ctypes

        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        st = SYSTEM_POWER_STATUS()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
            return None
        if st.BatteryFlag & 128 or st.BatteryLifePercent > 100:     # no system battery / unknown
            return None
        return {"percent": int(st.BatteryLifePercent), "charging": st.ACLineStatus == 1}
    return None


def ping(ip: str) -> bool:
    """One ping, 1 s timeout. On Windows a router's "host unreachable" also returns 0, so the
    reply itself (TTL=) decides."""
    try:
        if WINDOWS:
            r = subprocess.run(["ping", "-n", "1", "-w", "1000", ip], capture_output=True, timeout=3, **_NO_WINDOW)
            return r.returncode == 0 and b"TTL=" in r.stdout.upper()
        r = subprocess.run(["ping", "-c", "1", "-W", "1000", ip], capture_output=True, timeout=3)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False
