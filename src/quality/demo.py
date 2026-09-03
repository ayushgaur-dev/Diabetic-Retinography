"""Deterministic quality demo: python -m src.quality.demo <image> [--out viz.png]"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.quality.quality_pipeline import assess_image  # noqa: E402
from src.quality.visualization import save_quality_figure  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 2 quality demo")
    ap.add_argument("image", help="Fundus image file (png/jpg)")
    ap.add_argument("--out", default=None, help="Optional path to save quality figure")
    args = ap.parse_args(argv)
    rgb = np.array(Image.open(args.image).convert("RGB"))
    t0 = time.perf_counter()
    result, info = assess_image(rgb)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"QUALITY STATUS: {result.status}")
    print(f"Overall score: {result.overall_score:.3f} (advisory only)")
    print(f"Focus: {result.focus.status} (var {result.focus.measurement:.1f})")
    print(f"Illumination: {result.illumination.status} (median {result.illumination.measurement:.1f})")
    print(f"Contrast: {result.contrast.status} (std {result.contrast.measurement:.2f})")
    print(f"Exposure: {result.exposure.status} (clipped {result.exposure.measurement:.4f})")
    print(f"Field of view: {result.field_of_view.status} (fraction {result.field_of_view.measurement:.3f})")
    print(f"Retinal coverage: {result.retinal_coverage.status} (completeness {result.retinal_coverage.measurement:.3f})")
    print(f"Enhancement required: {result.enhancement_required} | applied: {result.enhancement_applied}")
    for r in result.reasons:
        print(f"Reason: {r}")
    for m in result.recapture_feedback:
        print(f"Recapture: {m}")
    print(f"Pipeline time: {elapsed_ms:.1f} ms (assess {info['elapsed_ms']:.1f} ms)")
    if args.out:
        save_quality_figure(rgb, result, info["mask"], args.out)
        print(f"Figure saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
