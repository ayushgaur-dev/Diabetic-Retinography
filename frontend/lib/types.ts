export interface ScreeningSummary {
  image_hash: string;
  quality_status: string;
  blocked: boolean;
  blocked_reason: string | null;
  grade: number | null;
  grade_label: string | null;
  raw_probabilities: number[];
  calibrated_probabilities: number[];
  calibrated_confidence: number | null;
  referable_score: number | null;
  referable: boolean | null;
  triage_decision: string | null;
  triage_priority: string | null;
  reason_codes: string[];
  safety_flags: Record<string, boolean>;
  lesion_counts: Record<string, number>;
  anatomy: {
    vessel_available: boolean;
    disc: { status: string; center: [number, number] | null; confidence: number | null };
    fovea: { status: string; center: [number, number] | null; confidence: number | null };
  };
  consistency: string | null;
  warnings: string[];
  errors: Record<string, string>;
  timings_ms: Record<string, number>;
}

export interface JobState {
  job_id: string;
  state: "queued" | "running" | "done" | "error";
  stage: string | null;
  result: ScreeningSummary | null;
  layers: Record<string, string> | null;
  error: string | null;
}

export const STAGES = [
  "quality",
  "enhancement",
  "grading",
  "evidence",
  "triage",
] as const;

export const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  starting: "Starting",
  quality: "01 · Quality",
  enhancement: "02 · Enhancement",
  grading: "03 · Grading",
  evidence: "04–06 · Anatomy · Evidence · Explanation",
  triage: "07 · Triage",
  done: "Complete",
};
