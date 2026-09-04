"""Screening service (Phase 10B). Thin adapter over app/screening_pipeline.
No pipeline logic duplicated: run_screening() does the work; this module
adds job bookkeeping, display-layer rendering (PNG data URIs), report
rendering from stored results, and the synthetic demo image."""

import base64
import hashlib
import io
import threading
import time
import uuid

import numpy as np
from PIL import Image

GRADE_LABELS = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR",
                3: "Severe NPDR", 4: "Proliferative DR"}

_JOBS = {}
_JOBS_LOCK = threading.Lock()
_STORE = {}
_MODEL = None
_GRADE_FN = None


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]


def get_model():
    global _MODEL, _GRADE_FN
    if _MODEL is None:
        import sys

        sys.path.insert(0, str(_repo_root()))
        from src.evaluation.inference import get_model as _gm

        from app.screening_pipeline import set_model_for_gradcam

        _MODEL = _gm(str(_repo_root() / "models" /
                         "efficientnetb0_finetuned_patched.keras"))
        set_model_for_gradcam(_MODEL)

        from tensorflow.keras.applications.efficientnet import preprocess_input

        def _grade(arr224):
            batch = np.expand_dims(preprocess_input(
                np.asarray(arr224).astype("float32")), 0)
            return [float(v) for v in _MODEL.predict(batch, verbose=0)[0]]

        _GRADE_FN = _grade
    return _MODEL, _GRADE_FN


def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _png_uri(arr, max_width=900):
    img = Image.fromarray(np.asarray(arr).astype(np.uint8))
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _overlays(rgb, result):
    """Display layers from stored result (no recompute). Missing layers omitted."""
    import cv2

    layers = {"original": _png_uri(rgb)}
    H, W = rgb.shape[:2]
    try:
        from src.explainability.localization_overlay import (
            draw_disc, draw_fovea, draw_lesions, draw_vessels, heat_overlay)
        from src.explainability.evidence_map import upsample_heatmap

        g = (result.get("gradcam") or {})
        heat = None
        if g.get("heatmap_small") is not None:
            heat = cv2.resize(np.asarray(g["heatmap_small"], dtype=float), (W, H),
                              interpolation=cv2.INTER_LINEAR)
            layers["gradcam"] = _png_uri(heat_overlay(rgb, heat))
        ves = result.get("vessel")
        if ves is not None:
            vm = cv2.resize((np.asarray(ves) > 0).astype(np.uint8) * 255, (W, H),
                            interpolation=cv2.INTER_NEAREST)
            layers["vessels"] = _png_uri(draw_vessels(rgb, vm))
        disc = result.get("disc") or {}
        fovea = result.get("fovea") or {}
        ana = draw_disc(rgb, disc.get("center"), disc.get("radius") or 0)
        ana = draw_fovea(ana, fovea.get("center_x_y"))
        layers["anatomy"] = _png_uri(ana)
        les = result.get("lesions") or {}
        layers["lesions"] = _png_uri(draw_lesions(rgb, les))
        combo = heat_overlay(rgb, heat) if heat is not None else np.asarray(rgb)
        if ves is not None:
            combo = draw_vessels(combo, cv2.resize(
                (np.asarray(ves) > 0).astype(np.uint8) * 255, (W, H),
                interpolation=cv2.INTER_NEAREST))
        combo = draw_disc(combo, disc.get("center"), disc.get("radius") or 0)
        combo = draw_fovea(combo, fovea.get("center_x_y"))
        layers["combined"] = _png_uri(draw_lesions(combo, les))
    except Exception:
        pass
    return layers


def _summary(ihash, result):
    tri = result.get("triage") or {}
    lesions = result.get("lesions") or {}
    disc = result.get("disc") or {}
    fovea = result.get("fovea") or {}
    ex = result.get("explainability") or {}
    grade = result.get("grade")
    return {
        "image_hash": ihash,
        "quality_status": (result.get("quality") or {}).get("status", "?"),
        "blocked": bool(result.get("blocked")),
        "blocked_reason": result.get("blocked_reason"),
        "grade": grade,
        "grade_label": GRADE_LABELS.get(grade) if grade is not None else None,
        "raw_probabilities": result.get("raw_probabilities", []),
        "calibrated_probabilities": result.get("calibrated_probabilities", []),
        "calibrated_confidence": result.get("calibrated_confidence"),
        "referable_score": result.get("referable_score"),
        "referable": (result.get("referable_score") or 0) >= 0.7
        if result.get("referable_score") is not None else None,
        "triage_decision": tri.get("decision"),
        "triage_priority": tri.get("priority"),
        "reason_codes": tri.get("reason_codes", []),
        "safety_flags": tri.get("safety_flags", {}),
        "lesion_counts": {k: (v or {}).get("candidate_count", 0)
                          for k, v in lesions.items()},
        "anatomy": {
            "vessel_available": result.get("vessel") is not None,
            "disc": {"status": disc.get("status"),
                     "center": disc.get("center"),
                     "confidence": disc.get("confidence")},
            "fovea": {"status": fovea.get("status"),
                      "center": fovea.get("center_x_y"),
                      "confidence": fovea.get("confidence")},
        },
        "consistency": (ex.get("consistency") or {}).get("category"),
        "warnings": result.get("warnings", []),
        "errors": result.get("errors", {}),
        "timings_ms": result.get("timings_ms", {}),
    }


def create_job(data: bytes, full_evidence: bool = True):
    import sys

    sys.path.insert(0, str(_repo_root()))
    from app.screening_pipeline import content_hash as _h
    from app.screening_pipeline import run_screening

    ihash = _h(data)
    job_id = uuid.uuid4().hex[:12]
    rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"))

    def stages_state():
        with _JOBS_LOCK:
            return dict(_JOBS.get(job_id, {}))

    def on_stage(name):
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["stage"] = name

    def work():
        try:
            with _JOBS_LOCK:
                _JOBS[job_id] = {"state": "running", "stage": "starting"}
            _, grade_fn = get_model()
            t0 = time.perf_counter()
            result = run_screening(rgb, grade_fn=grade_fn,
                                   stages={"full_evidence": full_evidence},
                                   on_stage=on_stage)
            result["_elapsed_s"] = round(time.perf_counter() - t0, 1)
            layers = _overlays(rgb, result)
            with _JOBS_LOCK:
                _STORE[ihash] = {"result": result}
                _JOBS[job_id] = {"state": "done", "stage": "done",
                                 "result": _summary(ihash, result),
                                 "layers": layers}
        except Exception as e:
            with _JOBS_LOCK:
                _JOBS[job_id] = {"state": "error", "stage": None,
                                 "error": f"{type(e).__name__}: {str(e)[:300]}"}

    with _JOBS_LOCK:
        _JOBS[job_id] = {"state": "queued", "stage": "queued"}
    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id, "image_hash": ihash, "stages_state": stages_state}


def get_job(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def mark_synthetic(ihash: str):
    with _JOBS_LOCK:
        if ihash in _STORE:
            _STORE[ihash]["synthetic"] = True


def get_stored(ihash: str):
    with _JOBS_LOCK:
        return _STORE.get(ihash)


def synthetic_demo_image(seed: int = 7, size: int = 1024):
    """Deterministic synthetic fundus (test-fixture family). Non-PHI by
    construction; always labelled synthetic end-to-end. Textured to pass
    the quality gate (vessel detail + speckle), like the unit-test
    fixtures — demo curation, not metric tuning."""
    import cv2

    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    c = size // 2
    r = size // 2 - 20
    img[(xx - c) ** 2 + (yy - c) ** 2 <= r ** 2] = [110, 70, 40]
    img[(xx - c - 150) ** 2 + (yy - c) ** 2 <= 60 ** 2] = [225, 195, 135]
    for _ in range(300):
        x0, y0 = rng.integers(c - r, c + r, 2)
        cv2.line(img, (int(x0), int(y0)),
                 (int(x0 + rng.integers(-80, 80)), int(y0 + rng.integers(-80, 80))),
                 (int(rng.integers(35, 85)), 25, 15), rng.integers(1, 3))
    speck = rng.integers(0, size, (400, 2))
    for x, y in speck:
        cv2.circle(img, (int(x), int(y)), 1, (200, 170, 120), -1)
    img = img.astype(float) + rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def render_report(ihash: str, fmt: str):
    """Phase 9 rendering from the stored screening result (no recompute)."""
    stored = get_stored(ihash)
    if stored is None:
        return None, None
    import sys

    sys.path.insert(0, str(_repo_root()))
    from src.reporting.config import load_reporting_config
    from src.reporting.deterministic_generator import generate
    from src.reporting.html_renderer import render_html
    from src.reporting.json_renderer import render_json
    from src.reporting.markdown_renderer import render_markdown
    from src.reporting.pdf_renderer import render_pdf

    result = stored["result"]
    tri = result.get("triage") or {}
    lesions = {}
    for lt, d in (result.get("lesions") or {}).items():
        lesions[lt] = {"status": d.get("status", "UNKNOWN"),
                       "candidate_count": d.get("candidate_count", 0),
                       "confidence": d.get("confidence", 0.0),
                       "total_evidence_area": 0, "candidates": []}
    rep_in = {
        "image_id": ("synthetic-demo" if stored.get("synthetic") else ihash[:12]),
        "quality": result.get("quality", {}),
        "enhancement": result.get("enhancement") or {},
        "grading": {"predicted_grade": result.get("grade"),
                    "raw_probabilities": result.get("raw_probabilities", []),
                    "calibrated_probabilities": result.get(
                        "calibrated_probabilities", []),
                    "calibrated_confidence": result.get("calibrated_confidence")},
        "referable": {"score": result.get("referable_score"), "threshold": 0.7},
        "lesions": lesions,
        "vessels": {"available": result.get("vessel") is not None},
        "optic_disc": {"status": (result.get("disc") or {}).get("status", "UNKNOWN")},
        "fovea": {"status": (result.get("fovea") or {}).get("status", "UNKNOWN")},
        "explainability": {"consistency": ((result.get("explainability") or {})
                                           .get("consistency", {}))},
        "triage": tri, "warnings": result.get("warnings", []),
    }
    rep = generate(rep_in, load_reporting_config()).to_dict()
    media = {"json": "application/json", "md": "text/markdown",
             "html": "text/html", "pdf": "application/pdf"}
    if fmt == "json":
        return render_json(rep).encode(), media["json"]
    if fmt == "md":
        return render_markdown(rep).encode(), media["md"]
    if fmt == "html":
        return render_html(rep).encode(), media["html"]
    if fmt == "pdf":
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            render_pdf(rep, tf.name)
            tf.flush()
            with open(tf.name, "rb") as f:
                return f.read(), media["pdf"]
    return None, None
