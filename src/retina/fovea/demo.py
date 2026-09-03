"""Single-image fovea demo: python -m src.retina.fovea.demo --image IMG [--fov FOV] [--out FIG]"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.retina.fovea.pipeline import localize_fovea  # noqa: E402
from src.retina.fovea.visualization import save_fovea_figure  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 4C fovea demo")
    ap.add_argument("--image", required=True)
    ap.add_argument("--fov", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rgb = np.array(Image.open(args.image).convert("RGB"))
    fov = None
    if args.fov:
        import cv2

        m = np.array(Image.open(args.fov).convert("L"))
        if m.shape != rgb.shape[:2]:
            m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        fov = ((m > 0).astype(np.uint8)) * 255
    res = localize_fovea(rgb, fov_mask=fov)
    d = res.to_dict()
    print("FOVEA LOCALIZATION")
    print(f"Detected: {d['detected']} ({d['status']})")
    print(f"Center (x=column, y=row): {d['center_x_y']}")
    print(f"Confidence: {d['confidence']} (heuristic candidate score, NOT calibrated)")
    print(f"Candidates: {d['candidate_count']} | laterality: {d['laterality']} "
          f"| disc used: {d['disc_used']}")
    for w in d["warnings"]:
        print(f"Note: {w}")
    if args.out:
        save_fovea_figure(rgb, d, args.out)
        print(f"Figure saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
