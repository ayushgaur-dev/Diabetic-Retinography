"""Generate verified synthetic demo fixtures (Phase 14). Non-PHI by
construction. Each fixture is CHECKED through the real quality gate and
only kept if it lands in its intended state; the script prints the
verified mapping (fixture -> status)."""
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, r"C:\Users\Lenovo\Documents\Default Project")
from src.quality.quality_pipeline import assess_image

OUT = r"C:\Users\Lenovo\Documents\Default Project\demo\fixtures"


def fundus(seed=3, size=512, blur=0, dark=1.0, low_contrast=False):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    c = size // 2
    r = size // 2 - 20
    img[(xx - c) ** 2 + (yy - c) ** 2 <= r ** 2] = [110, 70, 40]
    img[(xx - c - 70) ** 2 + (yy - c) ** 2 <= 30 ** 2] = [225, 195, 135]
    for _ in range(150):
        x0, y0 = rng.integers(c - r, c + r, 2)
        cv2.line(img, (int(x0), int(y0)),
                 (int(x0 + rng.integers(-50, 50)), int(y0 + rng.integers(-50, 50))),
                 (int(rng.integers(35, 85)), 25, 15), 2)
    img = img.astype(float) + rng.normal(0, 6, img.shape)
    img = np.clip(img, 0, 255)
    if low_contrast:
        img = (img - img.mean()) * 0.35 + 110
    img = np.clip(img * dark, 0, 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return np.clip(img, 0, 255).astype(np.uint8)


def check(name, img, want):
    small = np.array(Image.fromarray(img).convert("RGB").resize((224, 224)))
    got, _ = assess_image(small)
    flag = "OK " if got.status == want else "FAIL"
    print(f"{flag} {name}: want={want} got={got.status}")
    if got.status == want:
        Image.fromarray(img).save(f"{OUT}/{name}.png")
    return got.status == want


import os

os.makedirs(OUT, exist_ok=True)
ok = True
ok &= check("demo_good", fundus(), "GOOD")
ok &= check("demo_borderline", fundus(low_contrast=True), "BORDERLINE")
ok &= check("demo_ungradable", np.zeros((512, 512, 3), np.uint8), "UNGRADABLE")
print("ALL FIXTURES VERIFIED" if ok else "FIXTURE MISMATCH — investigate, do not ship")
