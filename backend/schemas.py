"""API schemas (Phase 10B). Pydantic v2. The Python pipeline is
authoritative; these schemas only VALIDATE and TRANSPORT its outputs."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobCreated(BaseModel):
    job_id: str
    image_hash: str


class ComponentStatus(BaseModel):
    status: str = "UNKNOWN"
    candidate_count: int = 0


class ScreeningSummary(BaseModel):
    image_hash: str
    quality_status: str
    blocked: bool
    blocked_reason: Optional[str] = None
    grade: Optional[int] = None
    grade_label: Optional[str] = None
    raw_probabilities: List[float] = Field(default_factory=list)
    calibrated_probabilities: List[float] = Field(default_factory=list)
    calibrated_confidence: Optional[float] = None
    referable_score: Optional[float] = None
    referable: Optional[bool] = None
    triage_decision: Optional[str] = None
    triage_priority: Optional[str] = None
    reason_codes: List[str] = Field(default_factory=list)
    safety_flags: Dict[str, bool] = Field(default_factory=dict)
    lesion_counts: Dict[str, int] = Field(default_factory=dict)
    anatomy: Dict[str, Any] = Field(default_factory=dict)
    consistency: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    errors: Dict[str, str] = Field(default_factory=dict)
    timings_ms: Dict[str, float] = Field(default_factory=dict)


class JobStatus(BaseModel):
    job_id: str
    state: str  # queued|running|done|error
    stage: Optional[str] = None
    result: Optional[ScreeningSummary] = None
    layers: Optional[Dict[str, str]] = None  # name -> PNG data URI
    error: Optional[str] = None


class DemoInfo(BaseModel):
    synthetic: bool = True
    note: str = ""
