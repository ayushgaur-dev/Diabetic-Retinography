"""Streamlit demo app — Retinal Screening Triage System (Block 18).

Two tabs:
  Screening — upload a fundus image + editable (synthetic-prefilled)
              intake form, wired through: model -> triage rules -> RAG
              retrieval -> LLM report generation.
  Insights  — static model/RAG performance figures, a live summary of
              cases processed in the current browser session, a link to
              the full Tableau dashboard, and a minimal EDA sanity view
              over data/processed/eda_summary.csv.

Not a diagnostic device. Every screen in the Screening tab carries that
notice — see the disclaimer constant below.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model as keras_load_model

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.gradcam import make_gradcam_heatmap, overlay_gradcam  # noqa: E402
from src.rag.generator import generate_report  # noqa: E402
from src.rag.indexer import load_index, retrieve  # noqa: E402
from src.rules.triage import triage_decision  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "efficientnetb0_finetuned_patched.keras"
SYNTHETIC_INTAKE_PATH = REPO_ROOT / "data" / "synthetic" / "synthetic_intake_demo.csv"
EDA_SUMMARY_PATH = REPO_ROOT / "data" / "processed" / "eda_summary.csv"

# Static model/RAG performance figures for the Insights tab. Hardcoded,
# not recomputed live — these are the fine-tuned model's validation-set
# results (docs/experiments.md, data/processed/confusion_matrix.csv) and
# the RAG retrieval eval (src/rag/eval_rag.py), not something this app
# session can regenerate on its own.
MODEL_QWK = 0.7987
PER_GRADE_RECALL = {0: 0.952, 1: 0.309, 2: 0.353, 3: 0.690, 4: 0.455}
RAG_HIT_RATE_AT_3 = 1.00
RAG_HIT_RATE_AT_1 = 0.778
TABLEAU_DASHBOARD_URL = (
    "https://public.tableau.com/app/profile/manu.daza/viz/"
    "RetinalScreeningTriageSystemFigures/"
    "RetinalScreeningTriageSystemModelEvaluationDashboard"
)

IMG_SIZE = 224
GRADE_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

DEVICE_NOTICE = (
    "⚠️ This is a triage assistant, not a diagnostic device. "
    "Every output requires clinician review before being acted on."
)

FALLBACK_INTAKE_DEFAULTS = {
    "age": 60.0,
    "diabetes_years": 10,
    "hba1c": 7.8,
    "visual_acuity": 0.75,
    "spherical_refraction": 0.0,
}


# ---------------------------------------------------------------------------
# Cached resource loaders — each returns (resource, error_message); error is
# None on success. Never raises, so the caller always has a graceful path.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_vision_model():
    try:
        return keras_load_model(MODEL_PATH), None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_guideline_index():
    try:
        return load_index(), None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preprocess_uploaded_image(pil_image):
    """Resize to 224x224 and apply EfficientNet's preprocess_input — the
    same preprocessing used in notebooks/03_transfer_learning.ipynb.
    NOT /255.0 normalisation; using generic normalisation here would
    reproduce the near-random-performance bug documented in
    docs/experiments.md.
    """
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def pick_synthetic_intake_defaults():
    """One random row from the synthetic intake CSV, as a plain dict.
    Falls back to fixed values (with an on-screen note) if the CSV is
    missing rather than crashing the app.
    """
    try:
        df = pd.read_csv(SYNTHETIC_INTAKE_PATH)
        row = df.sample(1).iloc[0]
        return {
            "age": float(row["age"]),
            "diabetes_years": int(row["diabetes_years"]),
            "hba1c": float(row["hba1c"]),
            "visual_acuity": float(row["visual_acuity"]),
            "spherical_refraction": float(row["spherical_refraction"]),
        }, None
    except Exception as e:
        return dict(FALLBACK_INTAKE_DEFAULTS), str(e)


def build_retrieval_query(predicted_grade, action):
    """No free-text question exists in this flow (unlike test_retrieval.py
    / eval_rag.py's hand-written queries) — this builds one from the
    triage outcome itself, so retrieval still has something concrete to
    search against. A design choice, not something the spec dictated.
    """
    return (
        f"ICDRSS grade {predicted_grade} diabetic retinopathy: guideline "
        f"criteria, recommended follow-up interval, and referral criteria "
        f"relevant to a {action} decision."
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Retinal Screening Triage", layout="wide")
st.title("Retinal Screening Triage System")

tab_screening, tab_insights = st.tabs(["Screening", "Insights"])

with tab_screening:
    st.warning(DEVICE_NOTICE)

    st.subheader("Patient intake")
    st.caption(
        "SYNTHETIC DATA — DEMONSTRATION ONLY. Pre-filled from a random row of "
        "data/synthetic/synthetic_intake_demo.csv (not a real patient). "
        "Edit any field freely."
    )

    if "intake_defaults" not in st.session_state:
        defaults, load_error = pick_synthetic_intake_defaults()
        st.session_state.intake_defaults = defaults
        st.session_state.intake_defaults_error = load_error
    defaults = st.session_state.intake_defaults

    if st.session_state.intake_defaults_error:
        st.info(
            f"Could not load synthetic_intake_demo.csv "
            f"({st.session_state.intake_defaults_error}); using fixed "
            f"fallback defaults instead."
        )

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=90, value=int(round(defaults["age"])))
        diabetes_years = st.number_input(
            "Years since diabetes diagnosis", min_value=0, max_value=80,
            value=int(defaults["diabetes_years"]),
        )
        hba1c = st.number_input(
            "HbA1c (%)", min_value=4.0, max_value=15.0,
            value=float(defaults["hba1c"]), step=0.1,
        )
    with col2:
        visual_acuity = st.number_input(
            "Visual acuity (decimal, 1.0 = normal)", min_value=0.0, max_value=1.0,
            value=float(defaults["visual_acuity"]), step=0.01,
        )
        spherical_refraction = st.number_input(
            "Spherical refraction (D)", min_value=-20.0, max_value=10.0,
            value=float(defaults["spherical_refraction"]), step=0.25,
        )

    st.caption(
        "Note: only HbA1c, diabetes duration, and spherical refraction "
        "currently feed the rule engine's patient modifiers "
        "(src/rules/triage.py). Age and visual acuity are captured for "
        "clinical context but not yet modelled there."
    )

    uploaded_file = st.file_uploader("Upload fundus photograph", type=["png", "jpg", "jpeg"])

    if st.button("Generate Report", type="primary"):
        if uploaded_file is None:
            st.warning("Please upload a fundus image before generating a report.")
        else:
            image = Image.open(uploaded_file)

            model, model_error = load_vision_model()
            if model is None:
                st.error(
                    f"Could not load the vision model from `{MODEL_PATH}`. "
                    f"Details: {model_error}"
                )
                st.stop()

            try:
                preprocessed = preprocess_uploaded_image(image)
                probs = model.predict(preprocessed, verbose=0)[0]
                predicted_grade = int(np.argmax(probs))
                confidence = float(np.max(probs))
            except Exception as e:
                st.error(f"Model inference failed: {e}")
                st.stop()

            # Grad-CAM: overlay on the ORIGINAL (non-preprocessed) image
            # resized to IMG_SIZE, not the EfficientNet-preprocessed
            # tensor — see src/models/gradcam.py for why that gives a
            # visually accurate overlay. Failure here degrades to a
            # warning, not a crash — the report can still proceed without it.
            gradcam_overlay, gradcam_error = None, None
            try:
                heatmap, _ = make_gradcam_heatmap(preprocessed, model)
                original_resized = np.array(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
                gradcam_overlay = overlay_gradcam(original_resized, heatmap, img_size=IMG_SIZE)
            except Exception as e:
                gradcam_error = str(e)

            img_col, gradcam_col, metric_col = st.columns([1, 1, 1])
            with img_col:
                st.image(image, caption="Uploaded fundus photograph", width=260)
            with gradcam_col:
                if gradcam_overlay is not None:
                    st.image(gradcam_overlay, caption="Grad-CAM overlay", width=260)
                else:
                    st.warning(f"Grad-CAM unavailable: {gradcam_error}")
            with metric_col:
                st.metric("Predicted grade", f"{predicted_grade} — {GRADE_LABELS.get(predicted_grade, '?')}")
                st.metric("Confidence", f"{confidence:.1%}")

            patient_params = {
                "age": age,
                "hba1c": hba1c,
                "diabetes_years": diabetes_years,
                "visual_acuity": visual_acuity,
                "spherical_refraction": spherical_refraction,
            }
            decision = triage_decision(predicted_grade, confidence, patient_params)

            # Session-only case tracking for the Insights tab (Section 2).
            # Recorded here — right after the rule engine succeeds — rather
            # than after report generation, since automation rate is a
            # rule-engine property, independent of whether the downstream
            # LLM report succeeds. Resets on page reload (session_state is
            # per-browser-session, not persisted anywhere).
            if "processed_cases" not in st.session_state:
                st.session_state.processed_cases = []
            st.session_state.processed_cases.append({
                "predicted_grade": predicted_grade,
                "confidence": confidence,
                "action": decision["action"],
                "requires_human_review": decision["requires_human_review"],
            })

            index_data, index_error = load_guideline_index()
            if index_data is None:
                st.error(
                    f"Could not load the guideline FAISS index. "
                    f"Details: {index_error}. Run `python -m src.rag.indexer` "
                    f"to build it first."
                )
                st.stop()
            index, chunk_mapping = index_data

            try:
                query = build_retrieval_query(predicted_grade, decision["action"])
                retrieved_chunks = retrieve(query, index, chunk_mapping, top_k=3)
            except Exception as e:
                st.error(f"Guideline retrieval failed: {e}")
                st.stop()

            report = generate_report(decision, retrieved_chunks)

            st.subheader("Structured pre-report")

            if report.get("error"):
                st.error(
                    f"Report generation failed ({report['error']}) — "
                    f"showing the rule-engine decision only, findings/"
                    f"guideline_applied are unavailable this run."
                )

            if report.get("requires_human_review"):
                st.error("⚠️ REQUIRES HUMAN REVIEW")
            else:
                st.success("No forced review flag on this decision.")

            st.write(f"**Action:** {report.get('action')}")
            interval = report.get("interval_months")
            st.write(f"**Follow-up interval:** {interval} months" if interval is not None else "**Follow-up interval:** N/A")

            st.write("**Findings:**")
            st.write(report.get("findings") or "_Not available_")

            st.write("**Guideline applied:**")
            st.write(report.get("guideline_applied") or "_Not available_")

            st.caption(f"Source chunks: {report.get('source_chunks')}")
            st.caption(
                "macular_status_assessed: false — this system does not "
                "assess diabetic macular oedema from a single 2D fundus "
                "image; see the guideline corpus for why."
            )

            st.warning(DEVICE_NOTICE)

with tab_insights:
    # --- 1. Model performance (static) ---
    st.subheader("Model performance")
    st.caption(
        "Fine-tuned EfficientNetB0, validation set. Figures are fixed "
        "results from training/evaluation (docs/experiments.md), not "
        "recomputed live by this app."
    )

    st.metric("Quadratic Weighted Kappa (QWK)", f"{MODEL_QWK:.4f}")

    recall_df = pd.DataFrame({
        "Grade": [f"{g} — {GRADE_LABELS[g]}" for g in sorted(PER_GRADE_RECALL)],
        "Recall": [PER_GRADE_RECALL[g] for g in sorted(PER_GRADE_RECALL)],
    }).set_index("Grade")
    st.write("Per-grade recall")
    st.dataframe(recall_df.style.format({"Recall": "{:.3f}"}), use_container_width=True)

    rag_col1, rag_col2 = st.columns(2)
    rag_col1.metric("RAG hit rate @3", f"{RAG_HIT_RATE_AT_3:.0%}")
    rag_col2.metric("RAG hit rate @1", f"{RAG_HIT_RATE_AT_1:.0%}")

    st.divider()

    # --- 2. This session (dynamic) ---
    st.subheader("This session")
    processed_cases = st.session_state.get("processed_cases", [])

    if not processed_cases:
        st.info(
            "No cases processed yet this session. Upload an image and "
            "click \"Generate Report\" in the Screening tab to populate "
            "this section."
        )
    else:
        cases_df = pd.DataFrame(processed_cases)

        st.metric("Cases processed this session", len(cases_df))

        grade_counts = cases_df["predicted_grade"].value_counts().sort_index()
        grade_counts.index = [f"Grade {g}" for g in grade_counts.index]
        st.write("Grade distribution (this session)")
        st.bar_chart(grade_counts)

        automation_rate = (~cases_df["requires_human_review"]).mean()
        st.metric("Automation rate (this session)", f"{automation_rate:.0%}")
        st.caption(
            f"{(~cases_df['requires_human_review']).sum()} of {len(cases_df)} "
            f"cases did not require forced human review."
        )

    st.divider()

    # --- 3. Full dashboard ---
    st.subheader("Full dashboard")
    st.markdown(
        f"For the complete picture — class distribution, image quality by "
        f"class, confusion matrix, and the automation-vs-recall trade-off "
        f"curve — see the full interactive dashboard:\n\n"
        f"**[View the Tableau dashboard →]({TABLEAU_DASHBOARD_URL})**"
    )

    st.divider()

    # --- 4. EDA (dataset-level, kept from the original implementation) ---
    st.subheader("EDA insights")
    st.caption(
        "Deliberately minimal — superseded by the Tableau dashboard above; "
        "shown here only as a quick sanity view over the training dataset."
    )

    try:
        eda_df = pd.read_csv(EDA_SUMMARY_PATH)
    except FileNotFoundError:
        st.info(
            f"`{EDA_SUMMARY_PATH}` not found. Generate it from "
            f"`notebooks/01_data_download_eda.ipynb` (Block 4) to populate "
            f"this tab."
        )
    except Exception as e:
        st.error(f"Could not load EDA summary: {e}")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.write("Class distribution")
            if "diagnosis" in eda_df.columns:
                st.bar_chart(eda_df["diagnosis"].value_counts().sort_index())
            else:
                st.warning("No 'diagnosis' column found in eda_summary.csv")

        with col2:
            st.write("Luminance distribution")
            if "mean_luminance" in eda_df.columns:
                counts, bin_edges = np.histogram(eda_df["mean_luminance"].dropna(), bins=20)
                bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}" for i in range(len(counts))]
                st.bar_chart(pd.DataFrame({"count": counts}, index=bin_labels))
            else:
                st.warning("No 'mean_luminance' column found in eda_summary.csv")
