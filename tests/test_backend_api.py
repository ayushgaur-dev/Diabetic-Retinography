"""Backend API tests — Phase 10B. FastAPI TestClient with stubbed service
(no model weights, no datasets). Real-pipeline smoke is done separately."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import service
from backend.api import app

client = TestClient(app)

CANNED = {
    "quality": {"status": "GOOD", "overall_score": 0.9},
    "blocked": False,
    "grade": 2,
    "raw_probabilities": [0.05, 0.12, 0.74, 0.07, 0.02],
    "calibrated_probabilities": [0.05, 0.12, 0.74, 0.07, 0.02],
    "calibrated_confidence": 0.74,
    "referable_score": 0.83,
    "temperature": 0.9542,
    "lesions": {"hemorrhage": {"status": "DETECTED", "candidate_count": 4,
                               "confidence": 0.8, "candidates": []}},
    "vessel": None, "disc": None, "fovea": None,
    "explainability": {"consistency": {"category": "SUPPORTIVE", "reason": "x"},
                       "lesion_overlap": {}},
    "triage": {"decision": "REFER", "priority": "HIGH", "referable": True,
               "reason_codes": ["REFERABLE_DR_PREDICTION"],
               "evidence_summary": {"lesion_flags": ["HEMORRHAGE_EVIDENCE"]},
               "safety_flags": {}, "confidence": 0.74, "predicted_grade": 2,
               "calibrated_confidence": 0.74, "referable_score": 0.83,
               "warnings": [], "method": "evidence_triage_v1", "explanation": "x"},
    "warnings": [], "errors": {}, "timings_ms": {},
}

def _png_bytes(size=64, color=(120, 70, 40)):
    import io

    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.full((size, size, 3), color, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


FAKE_PNG = _png_bytes()


@pytest.fixture(autouse=True)
def stub_service(monkeypatch):
    def fake_create(data, full_evidence=True):
        ihash = service.image_hash(data)
        service._STORE[ihash] = {"result": dict(CANNED)}
        service._JOBS["job1"] = {"state": "done", "stage": "done",
                                 "result": service._summary(ihash, dict(CANNED)),
                                 "layers": {"original": "data:image/png;base64,AAA"}}
        return {"job_id": "job1", "image_hash": ihash}

    monkeypatch.setattr(service, "create_job", fake_create)
    yield
    service._JOBS.clear()
    service._STORE.clear()


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_screen_success():
    r = client.post("/api/screen", files={"file": ("f.png", FAKE_PNG, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == "job1" and len(body["image_hash"]) == 64


def test_screen_rejects_type():
    r = client.post("/api/screen", files={"file": ("f.txt", b"hi", "text/plain")})
    assert r.status_code == 400


def test_screen_rejects_unreadable():
    r = client.post("/api/screen", files={"file": ("f.png", b"not-an-image", "image/png")})
    assert r.status_code == 400


def test_job_poll_delivers_result():
    r = client.post("/api/screen", files={"file": ("f.png", FAKE_PNG, "image/png")})
    job_id = r.json()["job_id"]
    j = client.get(f"/api/jobs/{job_id}").json()
    assert j["state"] == "done"
    assert j["result"]["grade"] == 2
    assert j["result"]["triage_decision"] == "REFER"  # triage integrity
    assert j["result"]["raw_probabilities"] == CANNED["raw_probabilities"]
    assert j["layers"]["original"].startswith("data:image/png;base64,")


def test_unknown_job_404():
    assert client.get("/api/jobs/nope").status_code == 404


def test_ungradable_state(monkeypatch):
    import copy

    blocked = copy.deepcopy(CANNED)
    blocked["blocked"] = True
    blocked["blocked_reason"] = "UNGRADABLE"
    blocked.pop("grade", None)

    def fake_create(data, full_evidence=True):
        ihash = service.image_hash(data)
        service._STORE[ihash] = {"result": dict(blocked)}
        service._JOBS["job9"] = {"state": "done", "stage": "done",
                                 "result": service._summary(ihash, dict(blocked)),
                                 "layers": {"original": "data:image/png;base64,AAA"}}
        return {"job_id": "job9", "image_hash": ihash}

    monkeypatch.setattr(service, "create_job", fake_create)
    jid = client.post("/api/screen",
                      files={"file": ("f.png", FAKE_PNG, "image/png")}).json()["job_id"]
    j = client.get(f"/api/jobs/{jid}").json()
    assert j["result"]["blocked"] is True
    assert j["result"]["grade"] is None  # no fabricated grade


def test_report_downloads_match_stored_triage():
    ihash = client.post("/api/screen",
                        files={"file": ("f.png", FAKE_PNG, "image/png")}).json()["image_hash"]
    for fmt, needle in (("json", b'"decision": "REFER"'), ("md", b"REFER"),
                        ("html", b"REFER"), ("pdf", b"%PDF")):
        r = client.get(f"/api/report/{ihash}/{fmt}")
        assert r.status_code == 200, fmt
        assert needle in r.content, fmt
    assert client.get("/api/report/deadbeef/json").status_code == 404
    assert client.get(f"/api/report/{ihash}/xml").status_code == 400


def test_missing_evidence_tolerated():
    assert service._summary("h", {})["quality_status"] == "?"
    assert service._summary("h", {})["lesion_counts"] == {}


def test_demo_endpoint(monkeypatch):
    import numpy as np

    monkeypatch.setattr(service, "synthetic_demo_image",
                        lambda seed=7, size=1024: np.zeros((64, 64, 3), np.uint8))
    r = client.post("/api/demo")
    assert r.status_code == 200
    assert len(r.json()["image_hash"]) == 64
