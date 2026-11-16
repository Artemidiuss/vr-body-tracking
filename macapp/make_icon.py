"""Draws the FBT app icon (a tracked figure: green bones, white joints) into an .iconset."""
import os
import sys

import cv2
import numpy as np

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
S = 4096                                   # supersampled, then downscaled (anti-aliasing)
img = np.zeros((S, S, 4), np.uint8)
pad, r = int(S * 0.09), int(S * 0.2)
mask = np.zeros((S, S), np.uint8)
cv2.rectangle(mask, (pad + r, pad), (S - pad - r, S - pad), 255, -1)
cv2.rectangle(mask, (pad, pad + r), (S - pad, S - pad - r), 255, -1)
for cx, cy in ((pad + r, pad + r), (S - pad - r, pad + r), (pad + r, S - pad - r), (S - pad - r, S - pad - r)):
    cv2.circle(mask, (cx, cy), r, 255, -1)
yy = np.linspace(0, 1, S)[:, None]
bg = np.dstack([(30 - 12 * yy) * np.ones((1, S)), (26 - 10 * yy) * np.ones((1, S)), (22 - 9 * yy) * np.ones((1, S))]).astype(np.uint8)
img[..., :3] = bg
img[..., 3] = mask
u = S / 1024
P = {k: (int(x * u), int(y * u)) for k, (x, y) in {
    "head": (512, 300), "neck": (512, 390), "hip": (512, 600),
    "ls": (420, 410), "rs": (604, 410), "le": (330, 330), "re": (694, 330), "lw": (290, 230), "rw": (734, 230),
    "lh": (450, 610), "rh": (574, 610), "lk": (420, 735), "rk": (604, 735), "la": (400, 850), "ra": (624, 850)}.items()}
green, white = (132, 220, 61), (241, 236, 233)
for a, b in (("neck", "hip"), ("ls", "rs"), ("lh", "rh"), ("ls", "le"), ("le", "lw"), ("rs", "re"), ("re", "rw"),
             ("lh", "lk"), ("lk", "la"), ("rh", "rk"), ("rk", "ra"), ("ls", "lh"), ("rs", "rh")):
    cv2.line(img, P[a], P[b], (*green, 255), int(34 * u), cv2.LINE_AA)
for k, p in P.items():
    if k != "head":
        cv2.circle(img, p, int(24 * u), (*white, 255), -1, cv2.LINE_AA)
cv2.circle(img, P["head"], int(62 * u), (*green, 255), int(30 * u), cv2.LINE_AA)
img[..., 3] = np.minimum(img[..., 3], mask)
for size in (16, 32, 64, 128, 256, 512, 1024):
    im = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if size <= 512:
        cv2.imwrite(f"{out}/icon_{size}x{size}.png", im)
    if size >= 32:
        cv2.imwrite(f"{out}/icon_{size // 2}x{size // 2}@2x.png", im)
