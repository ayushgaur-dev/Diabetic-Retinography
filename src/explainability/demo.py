"""Explainability demo: python -m src.explainability.demo --image IMG [--out FIG]"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.inference import get_model  # noqa: E402
from src.explainability.config import load_explainability_config  # noqa: E402
from src.explainability.pipeline import explain_image, faithfulness_deletion  # noqa: E402
from src.explainability.visualization import save_explainability_figure  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 6 explainability demo")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=str(REPO_ROOT / "models" /
                                           "efficientnetb0_finetuned_patched.keras"))
    ap.add_argument("--no-faithfulness", action="store_true")
    args = ap.parse_args(argv)
    rgb = np.array(Image.open(args.image).convert("RGB"))
    model = get_model(args.model)
    t0 = time.perf_counter()
    res = explain_image(rgb, model=model)
    d = res.to_dict()
    print(f"Grade {d['predicted_grade']} | raw conf {d['raw_confidence']} (UNCALIBRATED)")
    print(f"Explained class {d['explained_class']} via "
          f"{'original' if d['gradcam']['via_original_path'] else 'targeted'} Grad-CAM path")
    print(f"Consistency: {d['consistency']['category']} — {d['consistency']['reason']}")
    print(f"Evidence regions: {len(d['evidence_regions'])}")
    print(f"Explain time: {(time.perf_counter()-t0)*1000:.0f} ms")
    if not args.no_faithfulness:
        f = faithfulness_deletion(rgb, model, d["predicted_grade"], d["probabilities"])
        print(f"Faithfulness: P={f['original_probability']} -> {f['masked_probability']} "
              f"(drop {f['probability_drop']}; masked pred {f['masked_prediction']})")
    print("--- summary ---")
    print(d["summary"])
    if args.out:
        d["_maps"] = {"gradcam": res.maps["gradcam"],
                      "vessels": res.maps["vessel"]
                      if res.maps["vessel"] is not None else res.maps["blank"],
                      "blank": res.maps["blank"]}
        save_explainability_figure(rgb, d, args.out)
        print(f"Figure saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
