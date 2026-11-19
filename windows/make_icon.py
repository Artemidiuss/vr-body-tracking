"""windows/FBT.ico: the same icon as FBT.app (drawn by macapp/make_icon.py), packed as a Windows
icon (an .ico can hold PNG images directly, so no image library is needed)."""
import os
import struct
import subprocess
import sys
import tempfile

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "windows", "FBT.ico")
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run([sys.executable, os.path.join(root, "macapp", "make_icon.py"), tmp], check=True)
    sizes = [16, 32, 64, 128, 256]
    pngs = [open(os.path.join(tmp, f"icon_{s}x{s}.png"), "rb").read() for s in sizes]
head = struct.pack("<HHH", 0, 1, len(sizes))
offset = 6 + 16 * len(sizes)
entries = b""
for s, data in zip(sizes, pngs):
    entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), offset)   # 256 is written as 0
    offset += len(data)
with open(out, "wb") as f:
    f.write(head + entries + b"".join(pngs))
print(out)
