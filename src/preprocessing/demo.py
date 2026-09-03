"""Deterministic enhancement demo: python -m src.preprocessing.demo <image> [--out viz.png]"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.enhancement_pipeline import enhance_image  # noqa: E402
from src.preprocessing.visualization import save_enhancement_figure  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 3 enhancement demo")
    ap.add_argument("image", help="Fundus image file (png/jpg)")
    ap.add_argument("--out", default=None, help="Optional path to save before/after figure")
    args = ap.parse_args(argv)
    rgb = np.array(Image.open(args.image).convert("RGB"))
    t0 = time.perf_counter()
    result, _ = enhance_image(rgb)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    bq, aq = result.before_quality, result.after_quality
    print(f"QUALITY BEFORE: {bq.get('status')} (score {bq.get('overall_score')})")
    print(f"ENHANCEMENT: bypassed={result.bypassed} rejected={result.rejected} "
          f"successful={result.enhancement_successful} comparison={result.comparison}")
    print(f"OPERATIONS: {result.operations_applied or 'none'}")
    if aq:
        print(f"QUALITY AFTER: {aq.get('status')} (score {aq.get('overall_score')})")
    for d in result.changed_quality_dimensions:
        print(f"Improved dimension: {d}")
    for w in result.warnings:
        print(f"Note: {w}")
    print(f"FINAL STATE: {'enhanced image accepted' if result.enhancement_successful else ('no enhancement (GOOD)' if result.bypassed else 'original retained / recapture')}")
    print(f"Total time: {elapsed_ms:.1f} ms")
    if args.out:
        save_enhancement_figure(rgb, result, args.out)
        print(f"Figure saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
