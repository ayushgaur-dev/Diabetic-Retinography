"""Calibration demo: python -m src.calibration.demo --image IMG [--temperature T]"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.inference import get_model  # noqa: E402
from src.calibration.config import load_calibration_config  # noqa: E402
from src.calibration.pipeline import calibrate_record  # noqa: E402
from src.evaluation.inference import preprocess_image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
GRADE_LABELS = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR",
                3: "Severe NPDR", 4: "Proliferative DR"}


def _default_temperature():
    import json

    for name in ("metrics.json", "config.json"):
        p = REPO_ROOT / "reports" / "calibration" / name
        if p.exists():
            try:
                d = json.load(open(p))
                t = (d.get("temperature") or d.get("fitted_temperature")
                     or (d.get("fit") or {}).get("temperature"))
                if t:
                    return float(t), str(p)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 7 calibration demo")
    ap.add_argument("--image", required=True)
    ap.add_argument("--temperature", type=float, default=None)
    args = ap.parse_args(argv)
    cfg = load_calibration_config()
    if args.temperature is not None:
        temp = args.temperature
    else:
        temp, src = _default_temperature()
        if temp is None:
            print("No fitted temperature found. Run "
                  "`python -m src.calibration.evaluate` first, or pass --temperature.")
            return 1
        print(f"Temperature {temp} (from {src})")
    model = get_model(str(REPO_ROOT / cfg["model_artifact"]))
    # 224 = frozen EfficientNet preprocessing (Phase 1/5 procedure).
    batch = np.expand_dims(preprocess_image(args.image, 224), 0)
    probs = np.asarray(model.predict(batch, verbose=0)[0], dtype=float)
    rec = calibrate_record(probs, int(np.argmax(probs)), temp)
    d = rec.to_dict()
    pgm = cfg["referable_grade_min"]
    print(f"Raw prediction: Grade {d['raw_prediction']} "
          f"({GRADE_LABELS[d['raw_prediction']]})")
    print(f"Raw confidence: {d['raw_confidence']:.4f} (UNCALIBRATED)")
    print(f"Calibrated prediction: Grade {d['calibrated_prediction']} "
          f"({GRADE_LABELS[d['calibrated_prediction']]})")
    print(f"Calibrated confidence: {d['calibrated_confidence']:.4f} "
          f"(temperature-scaled, not clinical)")
    print(f"Temperature: {d['temperature']}")
    print(f"Referable raw score P2+P3+P4: {probs[pgm:].sum():.4f}")
    print(f"Referable calibrated score: {np.array(d['calibrated_probabilities'])[pgm:].sum():.4f}")
    for w in d["warnings"]:
        print(f"WARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
