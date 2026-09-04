"""FastAPI adapter (Phase 10B). Thin HTTP layer over backend.service, which
itself only calls app/screening_pipeline.run_screening(). No pipeline
logic here or in the frontend."""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from backend import service
from backend.schemas import DemoInfo, JobCreated, JobStatus

app = FastAPI(title="SIH26038 Screening API",
              description="Thin adapter over the frozen Python screening pipeline.",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_BYTES = 15 * 1024 * 1024
ALLOWED = {"image/png", "image/jpeg", "image/jpg"}


@app.get("/api/health")
def health():
    return {"status": "ok", "pipeline": "frozen", "llm": "none"}


@app.post("/api/screen", response_model=JobCreated)
async def screen(file: UploadFile = File(...),
                 full_evidence: bool = True):
    data = await file.read()
    if file.content_type not in ALLOWED:
        raise HTTPException(400, f"Unsupported type {file.content_type}. Use PNG/JPEG.")
    if len(data) > MAX_BYTES:
        raise HTTPException(400, "Image exceeds 15 MB.")
    try:
        from PIL import Image

        Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Unreadable image file.")
    job = service.create_job(data, full_evidence=full_evidence)
    return {"job_id": job["job_id"], "image_hash": job["image_hash"]}


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str):
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    out = {"job_id": job_id, "state": job.get("state"),
           "stage": job.get("stage"), "result": job.get("result"),
           "layers": job.get("layers"), "error": job.get("error")}
    return out


@app.post("/api/demo", response_model=JobCreated)
def demo():
    """Deterministic synthetic fundus (non-PHI) through the real pipeline."""
    import numpy as np
    from PIL import Image

    arr = service.synthetic_demo_image()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    data = buf.getvalue()
    job = service.create_job(data, full_evidence=True)
    service.mark_synthetic(job["image_hash"])
    info = DemoInfo(synthetic=True,
                    note="Synthetic fixture; not a patient image.").model_dump()
    return {"job_id": job["job_id"], "image_hash": job["image_hash"], **info}


@app.get("/api/report/{image_hash}/{fmt}")
def report(image_hash: str, fmt: str):
    if fmt not in ("json", "md", "html", "pdf"):
        raise HTTPException(400, "Format must be json|md|html|pdf.")
    body, media = service.render_report(image_hash, fmt)
    if body is None:
        raise HTTPException(404, "No screening result for this hash.")
    ext = {"json": "json", "md": "md", "html": "html", "pdf": "pdf"}[fmt]
    return Response(content=body, media_type=media,
                    headers={"Content-Disposition":
                             f"attachment; filename=report.{ext}"})


@app.get("/api/demo-info", response_model=DemoInfo)
def demo_info():
    return DemoInfo(synthetic=True, note="Synthetic fixture; not a patient image.")
