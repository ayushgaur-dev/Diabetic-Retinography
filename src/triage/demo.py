"""Triage demo: python -m src.triage.demo --image IMG [--fast] [--out FIG]

Full chain (slow): quality -> enhancement -> model -> calibration ->
vessels -> disc -> fovea -> lesions -> explainability-consistency -> triage.
Fast mode: quality + model + calibration + triage only (optional evidence
marked missing with warnings). Screening language only, never diagnosis.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.calibration.temperature_scaling import apply_temperature  # noqa: E402
from src.evaluation.inference import get_model  # noqa: E402
from src.explainability.consistency import classify as consistency_of  # noqa: E402
from src.explainability.evidence_fusion import lesion_overlap_stats  # noqa: E402
from src.explainability.evidence_map import hot_mask, upsample_heatmap  # noqa: E402
from src.explainability.gradcam_adapter import explain_class  # noqa: E402
from src.explainability.config import load_explainability_config  # noqa: E402
from src.preprocessing.enhancement_pipeline import enhance_image  # noqa: E402
from src.quality.quality_pipeline import assess_image  # noqa: E402
from src.triage.config import load_triage_config  # noqa: E402
from src.triage.pipeline import triage_from_phases  # noqa: E402
from src.triage.visualization import save_triage_card  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
GRADE_LABELS = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR",
                3: "Severe NPDR", 4: "Proliferative DR"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 8 triage demo")
    ap.add_argument("--image", required=True)
    ap.add_argument("--fast", action="store_true",
                    help="skip vessels/disc/fovea/lesions/Grad-CAM (marked missing)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    tcfg = load_triage_config()
    rgb = np.array(Image.open(args.image).convert("RGB"))
    t0 = time.perf_counter()

    small = np.array(Image.fromarray(rgb).convert("RGB").resize((224, 224)))
    qres, _ = assess_image(small)
    print(f"Quality: {qres.status}")
    if qres.status == "UNGRADABLE":
        res = triage_from_phases(
            config=tcfg, quality=qres.to_dict(), grade=0,
            raw_probs=[0.2] * 5, temperature=tcfg["temperature"],
            enhancement_status="none")
        print(res.explanation)
        return 0
    work = small
    enh_status = "none"
    if qres.status == "BORDERLINE" and not args.fast:
        eres, _ = enhance_image(small)
        if eres.enhancement_successful:
            work, enh_status = eres.enhanced_image, "success"
            print(f"Enhanced: {eres.before_quality['status']} -> "
                  f"{eres.after_quality['status']}")
        else:
            res = triage_from_phases(
                config=tcfg, quality=qres.to_dict(), grade=0,
                raw_probs=[0.2] * 5, temperature=tcfg["temperature"],
                enhancement_status="failed")
            print(res.explanation)
            return 0

    model = get_model(str(REPO_ROOT / "models" / "efficientnetb0_finetuned_patched.keras"))
    from tensorflow.keras.applications.efficientnet import preprocess_input

    batch = np.expand_dims(preprocess_input(work.astype("float32")), 0)
    probs = [float(v) for v in model.predict(batch, verbose=0)[0]]
    grade = int(np.argmax(probs))

    lesions, vessel, disc, fovea, consistency = None, None, None, None, {"category": "UNKNOWN"}
    if not args.fast:
        from src.retina.lesions.pipeline import detect_lesions
        from src.retina.vessels.pipeline import segment_vessels
        from src.retina.optic_disc.pipeline import localize_optic_disc
        from src.retina.fovea.pipeline import localize_fovea

        fov = None
        try:
            from src.quality.field_of_view import detect_retinal_field
            from src.quality.config import load_config as load_qconfig

            fov = detect_retinal_field(rgb, load_qconfig())["mask"]
            vessel = segment_vessels(rgb, fov).vessel_mask
        except Exception as e:
            print(f"vessels unavailable: {e}")
        try:
            disc = localize_optic_disc(rgb, fov_mask=fov).to_dict()
        except Exception as e:
            print(f"disc unavailable: {e}")
        try:
            fovea = localize_fovea(rgb, fov_mask=fov, disc=disc).to_dict()
        except Exception as e:
            print(f"fovea unavailable: {e}")
        try:
            lesions = detect_lesions(rgb)
            lesions.pop("_info", None)
        except Exception as e:
            print(f"lesions unavailable: {e}")
        try:
            g = explain_class(batch, model, grade, load_explainability_config())
            heat = upsample_heatmap(g["heatmap"], rgb.shape[:2])
            hot = hot_mask(heat, 0.75)
            import cv2

            lmasks = {}
            for lt, res in (lesions or {}).items():
                lmasks[lt] = cv2.resize(res.evidence_mask,
                                        (rgb.shape[1], rgb.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
            fov_full = fov if fov is not None else np.full(rgb.shape[:2], 255, np.uint8)
            overlap = lesion_overlap_stats(lmasks, hot, fov_full)
            cat, _ = consistency_of(overlap, float(heat.max()),
                                    load_explainability_config())
            consistency = {"category": cat}
        except Exception as e:
            print(f"consistency unavailable: {e}")

    res = triage_from_phases(
        config=tcfg, quality=qres.to_dict(), grade=grade, raw_probs=probs,
        temperature=tcfg["temperature"], lesion_results=lesions, vessel=vessel,
        disc=disc, fovea=fovea, consistency=consistency,
        enhancement_status=enh_status)
    d = res.to_dict()
    print(f"DR prediction: Grade {grade} ({GRADE_LABELS[grade]})")
    print(f"Calibrated confidence: {d['calibrated_confidence']:.4f}")
    print(f"Referable probability: {d['referable_score']:.4f} (threshold 0.70)")
    print(f"Triage: {d['decision']}")
    print(f"Priority: {d['priority']}")
    print("Reasons:")
    for r in d["reason_codes"]:
        print(f"  {r}")
    print("Warnings:")
    for w in d["warnings"] or ["none"]:
        print(f"  {w}")
    print(f"Total: {(time.perf_counter()-t0):.0f} s")
    if args.out:
        save_triage_card(rgb if max(rgb.shape[:2]) <= 1600 else small, d,
                         args.out, qres.status)
        print(f"Card saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
