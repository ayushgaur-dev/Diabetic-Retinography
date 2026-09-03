"""Single-image vessel demo: python -m src.retina.vessels.demo --image IMG [--gt GT] [--fov FOV] [--out FIG]"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.retina.vessels.pipeline import load_fov_mask, segment_vessels  # noqa: E402
from src.retina.vessels.preprocessing import extract_green  # noqa: E402
from src.retina.vessels.metrics import evaluate_mask  # noqa: E402
from src.retina.vessels.visualization import save_vessel_figure  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 4A vessel demo")
    ap.add_argument("--image", required=True)
    ap.add_argument("--gt", default=None, help="optional ground-truth mask")
    ap.add_argument("--fov", default=None, help="optional FOV mask")
    ap.add_argument("--out", default=None, help="optional figure path")
    args = ap.parse_args(argv)
    rgb = np.array(Image.open(args.image).convert("RGB"))
    fov = load_fov_mask(args.fov, rgb.shape[:2]) if args.fov else None
    res = segment_vessels(rgb, fov)
    print("VESSEL SEGMENTATION")
    print(f"Method: {res.method} (fov_source={res.fov_source})")
    print(f"Image size: {rgb.shape[1]}x{rgb.shape[0]}")
    print(f"Processing time: {res.processing_time_ms:.1f} ms")
    print(f"Vessel density: {res.vessel_density:.4f} (image statistic, not a biomarker)")
    for w in res.warnings:
        print(f"Note: {w}")
    gt = np.array(Image.open(args.gt).convert("L")) if args.gt else None
    if gt is not None:
        m = evaluate_mask(res.vessel_mask, gt, res.fov_mask)
        print(f"F1={m['f1']:.4f} sens={m['sensitivity']:.4f} spec={m['specificity']:.4f} "
              f"acc={m['accuracy']:.4f} (FOV-restricted)")
    if args.out:
        save_vessel_figure(rgb, extract_green(rgb), res.vessel_response,
                           res.vessel_mask, args.out, gt_mask=gt)
        print(f"Figure saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
