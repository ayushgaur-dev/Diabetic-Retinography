"""SIH26038 clinician-facing reference UI (Phase 10A).

Integrates Phases 2-9 through app/screening_pipeline.py only. No ML logic
lives here: quality/enhancement/grading/calibration/anatomy/lesions/
explainability/triage/reporting all come from their frozen modules.
Research/decision-support prototype — not a diagnostic device.
"""

import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.screening_pipeline import (  # noqa: E402
    content_hash,
    downscale_for_display,
    run_screening,
    set_model_for_gradcam,
    should_reuse,
)

GRADE_LABELS = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR",
                3: "Severe NPDR", 4: "Proliferative DR"}

SAFETY_BANNER = (
    "Research and decision-support prototype — not a diagnostic device. "
    "Qualified human review is required. Lesion evidence is not confirmation. "
    "Calibrated confidence is not clinical certainty. Dataset performance "
    "does not guarantee deployment performance. Ungradable images require "
    "recapture/review."
)

MODEL_PATH = REPO_ROOT / "models" / "efficientnetb0_finetuned_patched.keras"


@st.cache_resource(show_spinner=False)
def load_vision_model():
    try:
        from tensorflow.keras.models import load_model as _load

        return _load(str(MODEL_PATH)), None
    except Exception as e:
        return None, str(e)


def default_grade_fn_factory(model):
    from tensorflow.keras.applications.efficientnet import preprocess_input

    def grade_fn(arr224):
        batch = np.expand_dims(preprocess_input(
            np.asarray(arr224).astype("float32")), 0)
        return [float(v) for v in model.predict(batch, verbose=0)[0]]

    return grade_fn


st.set_page_config(page_title="SIH26038 DR Screening Reference", layout="wide")
st.title("Explainable DR Screening — Reference UI")
st.warning(SAFETY_BANNER)

st.header("1. Image upload")
uploaded = st.file_uploader("Fundus photograph", type=["png", "jpg", "jpeg"])
full_evidence = st.checkbox("Full evidence workup (vessels, landmarks, lesions, "
                            "explainability — slower)", value=True)
run = st.button("Run screening", type="primary")

if uploaded is not None:
    raw = uploaded.getvalue()
    ihash = content_hash(raw)
    st.caption(f"Image SHA-256: `{ihash[:16]}…` ({len(raw) // 1024} KB). "
               "No patient metadata collected; nothing is persisted.")
    rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    st.image(downscale_for_display(rgb), caption="Uploaded fundus photograph",
             width=420)
    if st.session_state.get("case_hash") != ihash:
        st.session_state.pop("case", None)  # new image invalidates stale results
        st.session_state.pop("report", None)
        st.session_state["case_hash"] = ihash
    if run or should_reuse(st.session_state.get("case_hash"), ihash,
                           "case" in st.session_state):
        if "case" not in st.session_state:
            model, merr = load_vision_model()
            if model is None:
                st.error(f"Model unavailable: {merr}")
                st.stop()
            set_model_for_gradcam(model)
            with st.spinner("Running screening pipeline (this can take a few minutes "
                            "with full evidence)…"):
                st.session_state["case"] = run_screening(
                    rgb, grade_fn=default_grade_fn_factory(model),
                    stages={"full_evidence": full_evidence})
                st.session_state["case_rgb"] = downscale_for_display(rgb)
        case = st.session_state["case"]
        disp = st.session_state.get("case_rgb", downscale_for_display(rgb))

        st.header("2. Quality gate")
        q = case.get("quality", {})
        st.subheader(f"Quality: {q.get('status', '?')}")
        comps = {k: (v or {}).get("status", "?") for k, v in q.items()
                 if isinstance(v, dict) and "status" in v}
        if comps:
            st.table([{"component": k, "status": v} for k, v in comps.items()])
        for r in q.get("reasons", []) or []:
            st.write(f"- {r}")
        for m in q.get("recapture_feedback", []) or []:
            st.warning(m)
        if case.get("blocked"):
            st.error(f"Screening stopped: {case.get('blocked_reason')}. "
                     "No DR grade was produced.")
            tri = case.get("triage", {})
            if tri:
                st.write(f"Routing: **{tri.get('decision')}**")
        else:
            if case.get("enhancement_status") == "success":
                st.header("3. Enhancement (BORDERLINE rescued)")
                enh = case.get("enhancement") or {}
                bq, aq = (enh.get("before_quality") or {}), (enh.get("after_quality") or {})
                st.write(f"Before: **{bq.get('status')}** → After: **{aq.get('status')}**")
                st.write(f"Operations: {', '.join(enh.get('operations_applied', []))}")
                st.caption("Enhancement adjusts image quality; it does not recover "
                           "fundamentally unusable images.")
            elif q.get("status") == "BORDERLINE":
                st.caption("BORDERLINE image graded without enhancement.")

            st.header("4. Screening result")
            st.subheader(f"Grade {case['grade']} — {GRADE_LABELS[case['grade']]}")
            probs = case["raw_probabilities"]
            cals = case["calibrated_probabilities"]
            st.table([{"grade": f"{i} ({GRADE_LABELS[i]})",
                       "raw": round(probs[i], 4),
                       "calibrated": round(cals[i], 4)} for i in range(5)])
            st.metric("Calibrated confidence", f"{case['calibrated_confidence']:.4f}")
            st.caption("Calibrated confidence is not clinical certainty.")

            st.header("5. Referable DR")
            st.write("Definition: predicted Grade ≥ 2; frozen threshold 0.7 on "
                     "calibrated P(Grade ≥ 2).")
            ref = case["referable_score"]
            st.subheader("REFERABLE" if ref >= 0.7 else "NON_REFERABLE")
            st.metric("Referable score", f"{ref:.4f}")

            if full_evidence:
                st.header("6. Retinal anatomy")
                ana_cols = st.columns(3)
                with ana_cols[0]:
                    st.write("**Vessels**")
                    v = case.get("vessel")
                    if v is not None:
                        st.image(downscale_for_display(
                            (np.asarray(v) > 0).astype(np.uint8) * 255),
                            caption="Vessel evidence map", width=260)
                    else:
                        st.caption("Vessel evidence unavailable.")
                with ana_cols[1]:
                    st.write("**Optic disc**")
                    dd = case.get("disc") or {}
                    st.write(f"Status: {dd.get('status', 'UNKNOWN')}")
                    if dd.get("center") is not None:
                        st.write(f"Center: {dd['center']}, radius {dd.get('radius')}")
                with ana_cols[2]:
                    st.write("**Fovea**")
                    fd = case.get("fovea") or {}
                    st.write(f"Status: {fd.get('status', 'UNKNOWN')}")
                    if fd.get("center_x_y") is not None:
                        st.write(f"Center: {fd['center_x_y']}")

                st.header("7. Lesion evidence")
                st.caption("Candidates are heuristic evidence, never confirmed disease. "
                           "Neovascularization detection is not implemented.")
                for lt in ("microaneurysm", "hemorrhage", "hard_exudate", "soft_exudate"):
                    d = (case.get("lesions") or {}).get(lt, {})
                    with st.expander(f"{lt.replace('_', ' ').title()}: "
                                     f"{d.get('status', '?')} "
                                     f"({d.get('candidate_count', 0)} candidates)",
                                     expanded=False):
                        for c in (d.get("candidates") or [])[:10]:
                            st.write(f"- {c.get('candidate_id')}: "
                                     f"score {c.get('score')}, bbox {c.get('bbox')}")

                st.header("8. Explainability")
                ex = case.get("explainability") or {}
                cons = ex.get("consistency", {}) or {}
                st.write(f"Consistency: **{cons.get('category', '?')}** — "
                         f"{cons.get('reason', '')}")
                for lt, s in (ex.get("lesion_overlap") or {}).items():
                    st.write(f"- {lt}: spatial agreement "
                             f"{s.get('lesion_inside_gradcam_fraction')}")
                st.caption("Spatial agreement is not causal proof.")

            st.header("9. Triage (Phase 8, displayed verbatim)")
            tri = case.get("triage", {})
            st.subheader(f"{tri.get('decision', '?')} [{tri.get('priority', '?')}]")
            st.write("Reasons:")
            for r in tri.get("reason_codes", []) or []:
                st.write(f"- {r}")
            flags = tri.get("safety_flags", {}) or {}
            raised = [k for k, v in flags.items() if v]
            st.write(f"Safety flags: {', '.join(raised) if raised else 'none'}")
            with st.expander("Deterministic explanation", expanded=False):
                st.write(tri.get("explanation", ""))
            for w in tri.get("warnings", []) or []:
                st.caption(f"Warning: {w}")

        st.header("10. Report download (Phase 9)")
        st.caption("Phase 9 ungradable pathway applies automatically when grading "
                   "was blocked.")
        if st.button("Generate screening report"):
            from src.reporting.config import load_reporting_config
            from src.reporting.deterministic_generator import generate
            from src.reporting.html_renderer import render_html
            from src.reporting.json_renderer import render_json
            from src.reporting.markdown_renderer import render_markdown
            from src.reporting.pdf_renderer import render_pdf

            rep_in = _report_input(case)
            rep = generate(rep_in, load_reporting_config()).to_dict()
            st.session_state["report"] = rep
        if "report" in st.session_state:
            rep = st.session_state["report"]
            st.download_button("Download JSON (authoritative)",
                               render_json(rep), file_name="report.json",
                               mime="application/json")
            st.download_button("Download Markdown",
                               render_markdown(rep), file_name="report.md",
                               mime="text/markdown")
            st.download_button("Download HTML",
                               render_html(rep), file_name="report.html",
                               mime="text/html")
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                render_pdf(rep, tf.name)
                tf.flush()
                with open(tf.name, "rb") as f:
                    pdf_bytes = f.read()
            st.download_button("Download PDF", pdf_bytes,
                               file_name="report.pdf",
                               mime="application/pdf")

        st.header("11. Limitations / safety")
        st.write(SAFETY_BANNER)
        for err_stage, err in (case.get("errors") or {}).items():
            st.error(f"{err_stage} failed: {err}")
        st.caption(f"Timings (ms): {case.get('timings_ms', {})}")


def _report_input(case):
    """Map orchestrator output onto the Phase 9 input contract (no recompute)."""
    tri = case.get("triage", {}) or {}
    lesions = {}
    for lt, d in (case.get("lesions") or {}).items():
        lesions[lt] = {"status": d.get("status", "UNKNOWN"),
                       "candidate_count": d.get("candidate_count", 0),
                       "confidence": d.get("confidence", 0.0),
                       "total_evidence_area": 0,
                       "candidates": d.get("candidates", [])}
    disc = case.get("disc") or {}
    fovea = case.get("fovea") or {}
    ex = case.get("explainability") or {}
    return {
        "image_id": "streamlit-upload",
        "quality": case.get("quality", {}),
        "enhancement": case.get("enhancement") or {},
        "grading": {"predicted_grade": case.get("grade"),
                    "raw_probabilities": case.get("raw_probabilities", []),
                    "calibrated_probabilities": case.get("calibrated_probabilities", []),
                    "calibrated_confidence": case.get("calibrated_confidence")},
        "referable": {"score": case.get("referable_score"), "threshold": 0.7},
        "lesions": lesions,
        "vessels": {"available": case.get("vessel") is not None},
        "optic_disc": {"status": disc.get("status", "UNKNOWN"),
                       "center": disc.get("center"), "radius": disc.get("radius"),
                       "confidence": disc.get("confidence")},
        "fovea": {"status": fovea.get("status", "UNKNOWN"),
                  "center_x_y": fovea.get("center_x_y"),
                  "confidence": fovea.get("confidence")},
        "explainability": {"consistency": ex.get("consistency", {}),
                           "lesion_overlap": ex.get("lesion_overlap", {}),
                           "gradcam": ex.get("gradcam", {}),
                           "figure": None},
        "triage": tri,
        "warnings": list(case.get("warnings", [])),
    }
