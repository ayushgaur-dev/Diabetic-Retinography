import type { JobState } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

export async function health(): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/api/health`);
  return json(res);
}

export async function startScreen(file: File, fullEvidence = true): Promise<{ job_id: string; image_hash: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/screen?full_evidence=${fullEvidence}`, {
    method: "POST",
    body: form,
  });
  return json(res);
}

export async function startDemo(): Promise<{ job_id: string; image_hash: string }> {
  const res = await fetch(`${BASE}/api/demo`, { method: "POST" });
  return json(res);
}

export async function pollJob(jobId: string): Promise<JobState> {
  const res = await fetch(`${BASE}/api/jobs/${jobId}`);
  return json(res);
}

export function reportUrl(imageHash: string, fmt: "json" | "md" | "html" | "pdf"): string {
  return `${BASE}/api/report/${imageHash}/${fmt}`;
}

export async function waitForJob(
  jobId: string,
  onStage: (stage: string | null) => void,
  timeoutMs = 600000,
  intervalMs = 1200
): Promise<JobState> {
  const t0 = Date.now();
  for (;;) {
    const job = await pollJob(jobId);
    onStage(job.stage);
    if (job.state === "done" || job.state === "error") return job;
    if (Date.now() - t0 > timeoutMs) throw new Error("Screening timed out.");
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}
