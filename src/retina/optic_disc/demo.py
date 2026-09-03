"""Single-image disc demo: python -m src.retina.optic_disc.demo --image IMG [--fov FOV] [--vessels V] [--out FIG]"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.retina.optic_disc.pipeline import localize_optic_disc  # noqa: E402
from src.retina.optic_disc.visualization import save_disc_figure  # noqa: E402


def _gray(path, shape):
    m = np.array(Image.open(path).convert("L"))
    if m.shape != shape:
        import cv2

        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return ((m > 0).astype(np.uint8)) * 255


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 4B disc demo")
    ap.add_argument("--image", required=True)
    ap.add_argument("--fov", default=None)
    ap.add_argument("--vessels", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rgb = np.array(Image.open(args.image).convert("RGB"))
    fov = _gray(args.fov, rgb.shape[:2]) if args.fov else None
    ves = _gray(args.vessels, rgb.shape[:2]) if args.vessels else None
    res = localize_optic_disc(rgb, fov_mask=fov, vessel_mask=ves)
    d = res.to_dict()
    print("OPTIC DISC LOCALIZATION")
    print(f"Detected: {d['detected']} ({d['status']})")
    print(f"Center: {d['center']}")
    print(f"Radius: {d['radius']} px (normalized {d['radius_normalized']})")
    print(f"Confidence: {d['confidence']} (algorithmic candidate score, NOT calibrated)")
    print(f"Candidates: {d['candidate_count']}")
    for w in d["warnings"]:
        print(f"Note: {w}")
    if args.out:
        save_disc_figure(rgb, d, args.out, fov_mask=fov)
        print(f"Figure saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
