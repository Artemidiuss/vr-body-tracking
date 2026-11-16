"""One frame from every camera -> data/snap_camN.jpg (run from Terminal.app if 'mac' is included).

  .venv/bin/python -m tools.snap_all 192.168.31.81 192.168.31.28 mac
"""
import sys
import time

import cv2

from fbt.maccam import open_source

srcs = [open_source(f"cam{i}", h).start() for i, h in enumerate(sys.argv[1:])]
time.sleep(3)
for i, s in enumerate(srcs):
    f = s.latest()
    print(f"cam{i} {sys.argv[1 + i]}: {'no frame' if f is None else f.image.shape}", flush=True)
    if f is not None:
        cv2.imwrite(f"data/snap_cam{i}.jpg", f.image)
    s.stop()
