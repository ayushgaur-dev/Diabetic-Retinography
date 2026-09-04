"""Grading evaluation demo: single image through quality gate + frozen model."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.config import load_eval_config  # noqa: E402
from src.evaluation.inference import get_model, preprocess_image  # noqa: E402
from src.evaluation.quality_analysis import select_grading_image  # noqa: E402
from src.quality.quality_pipeline import assess_image  # noqa: E402
from PIL import Image  # noqa: E402

GRADE_LABELS = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR",
                3: "Severe NPDR", 4: "Proliferative DR"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 5 grading demo")
    ap.add_argument("--image", required=True)
    args = ap.parse_args(argv)
    cfg = load_eval_config()
    arr = np.array(Image.open(args.image).convert("RGB").resize((224, 224)))
    qres, _ = assess_image(arr)
    print(f"QUALITY: {qres.status}")
    if qres.status == "UNGRADABLE":
        print("Blocked: ungradable image is never graded.")
        return 0
    sel = select_grading_image({"image_path": args.image,
                                "quality_status": qres.status})
    if sel["array"] is None:
        print("Blocked: enhancement discarded.")
        return 0
    from tensorflow.keras.applications.efficientnet import preprocess_input

    model = get_model(str(Path(__file__).resolve().parents[2] / cfg["model_artifact"]))
    probs = model.predict(np.expand_dims(
        preprocess_input(sel["array"].astype("float32")), 0), verbose=0)[0]
    pred = int(np.argmax(probs))
    print(f"GRADE: {pred} ({GRADE_LABELS[pred]}) via {sel['used_image']} image")
    print(f"RAW softmax confidence: {float(np.max(probs)):.4f} (UNCALIBRATED)")
    print(f"Referable (grade>=2): {pred >= 2} | P2+P3+P4={float(probs[2:].sum()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
