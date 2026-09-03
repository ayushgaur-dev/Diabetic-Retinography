"""Lesion demo: python -m src.retina.lesions.demo --image IMG [--out FIG]"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.retina.lesions.pipeline import detect_lesions  # noqa: E402
from src.retina.lesions.visualization import save_lesion_figure  # noqa: E402
from src.retina.lesions.types import LESION_TYPES  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 4D lesion demo")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rgb = np.array(Image.open(args.image).convert("RGB"))
    t0 = time.perf_counter()
    res = detect_lesions(rgb)
    info = res.pop("_info")
    print(f"Image: {rgb.shape[1]}x{rgb.shape[0]} "
          f"({(time.perf_counter()-t0)*1000:.0f} ms total)")
    for lt in LESION_TYPES:
        d = res[lt].to_dict()
        print(f"{lt.replace('_', ' ').title()} evidence:")
        print(f"    status: {d['status']}")
        print(f"    candidates: {d['candidate_count']}")
        print(f"    area: {d['total_evidence_area']}")
        print(f"    confidence: {d['confidence']} (heuristic, NOT probability)")
    if args.out:
        save_lesion_figure(rgb, {lt: res[lt].to_dict() for lt in LESION_TYPES},
                           args.out)
        print(f"Figure saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
