"use client";

import Link from "next/link";
import { useCallback, useRef, useState } from "react";
import Timeline from "../../components/Timeline";
import Viewer from "../../components/Viewer";
import { SafetyStrip, TechLabel, useReveal } from "../../components/chrome";
import { reportUrl, startDemo, startScreen, waitForJob } from "../../lib/api";
import type { JobState } from "../../lib/types";

const GRADES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"];

type Phase = "upload" | "working" | "review";

export default function Screen() {
  useReveal();
  const [phase, setPhase] = useState<Phase>("upload");
  const [preview, setPreview] = useState<string | null>(null);
  const [stage, setStage] = useState<string | null>(null);
  const [job, setJob] = useState<JobState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fullEvidence, setFullEvidence] = useState(true);
  const fileRef = useRef<File | null>(null);

  const onFile = useCallback((f: File | undefined) => {
    if (!f) return;
    fileRef.current = f;
    setJob(null);
    setError(null);
    setPhase("upload");
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const run = useCallback(
    async (jobId: string) => {
      setPhase("working");
      setError(null);
      try {
        const done = await waitForJob(jobId, (s) => setStage(s));
        setJob(done);
        if (done.state === "error") setError(done.error);
        else setPhase("review");
      } catch (e) {
        setError(e instanceof Error ? e.message : "Screening failed.");
        setPhase("upload");
      }
    },
    []
  );

  const submit = useCallback(async () => {
    const f = fileRef.current;
    if (!f) return;
    try {
      const { job_id } = await startScreen(f, fullEvidence);
      await run(job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    }
  }, [fullEvidence, run]);

  const demo = useCallback(async () => {
    try {
      const { job_id } = await startDemo();
      setPreview(null);
      await run(job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Demo failed.");
    }
  }, [run]);

  const r = job?.result ?? null;
  const blocked = r?.blocked ?? false;

  return (
    <main className="bg-paper text-ink">
      <header className="flex items-center justify-between px-6 py-5 md:px-16">
        <Link href="/" className="font-display text-2xl tracking-tight">RetinaLens</Link>
        <TechLabel>Screening · Upload → Analyze → Review</TechLabel>
      </header>

      {phase === "upload" && (
        <section className="px-6 pb-24 pt-10 md:px-16">
          <TechLabel>01 · Upload</TechLabel>
          <h1 className="mt-4 font-display text-6xl leading-[0.95] md:text-8xl">
            Drop fundus<br />image.
          </h1>
          <div
            className="mt-10 border-2 border-dashed border-ink/30 px-6 py-20 text-center"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              onFile(e.dataTransfer.files?.[0]);
            }}
          >
            <p className="font-display text-3xl">DROP FUNDUS IMAGE</p>
            <p className="mt-3 text-sm text-ink/60">or</p>
            <label className="mt-4 inline-block cursor-pointer bg-ink px-8 py-3 text-sm font-semibold uppercase tracking-widest text-paper">
              Choose image
              <input type="file" accept="image/png,image/jpeg" className="hidden"
                onChange={(e) => onFile(e.target.files?.[0])} />
            </label>
            <p className="mt-4 text-xs uppercase tracking-[0.2em] text-ink/50">PNG · JPEG · WEBP · up to 15 MB · no patient data stored</p>
          </div>
          {preview && (
            <div className="mt-8 grid gap-8 md:grid-cols-2">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={preview} alt="Uploaded fundus preview" className="w-full rounded-sm" />
              <div>
                <label className="flex cursor-pointer items-center gap-3 text-sm">
                  <input type="checkbox" checked={fullEvidence} onChange={(e) => setFullEvidence(e.target.checked)} />
                  Full evidence workup (slower, complete)
                </label>
                <button onClick={submit} className="mt-6 bg-arterial px-10 py-4 text-sm font-semibold uppercase tracking-widest text-paper hover:bg-ink">
                  Analyze
                </button>
              </div>
            </div>
          )}
          <div className="mt-10">
            <button onClick={demo} className="border border-ink px-8 py-3 text-sm uppercase tracking-widest hover:bg-ink hover:text-paper">
              Try synthetic demo
            </button>
            <p className="mt-2 text-xs text-ink/55">Deterministic non-PHI fixture through the real pipeline.</p>
          </div>
          {error && <p className="mt-6 border border-arterial p-4 text-sm text-arterial">{error}</p>}
        </section>
      )}

      {phase === "working" && (
        <section className="px-6 py-20 md:px-16">
          <TechLabel>02 · Analyze</TechLabel>
          <h2 className="mt-4 font-display text-5xl">Reading the retina…</h2>
          <div className="mt-10 max-w-md"><Timeline stage={stage} done={false} /></div>
          <p className="mt-8 text-sm text-ink/60">Stages reflect actual backend progress — nothing is fabricated.</p>
          {error && <p className="mt-6 border border-arterial p-4 text-sm text-arterial">{error}</p>}
        </section>
      )}

      {phase === "review" && r && (
        <section className="px-6 pb-24 pt-10 md:px-16">
          <TechLabel>03 · Review</TechLabel>
          {blocked ? (
            <div className="mt-6 border-2 border-arterial p-8 md:p-12">
              <h2 className="font-display text-5xl leading-tight md:text-7xl">IMAGE NOT<br />SUITABLE FOR<br />AUTOMATED GRADING</h2>
              <p className="mt-6 max-w-2xl text-lg">Reason: {r.blocked_reason}. No DR grade was produced.</p>
              <div className="mt-4 flex flex-wrap gap-3">
                {(["json", "md", "html", "pdf"] as const).map((f) => (
                  <a key={f} href={reportUrl(r.image_hash, f)} className="border border-ink px-5 py-2 text-xs uppercase tracking-widest hover:bg-ink hover:text-paper">
                    Download {f}
                  </a>
                ))}
              </div>
            </div>
          ) : (
            <>
              <p className="mt-6 font-display text-2xl">Screening result</p>
              <div className="grid gap-10 lg:grid-cols-12">
                <div className="lg:col-span-7">
                  <Viewer layers={job?.layers ?? null} />
                </div>
                <div className="lg:col-span-5">
                  <p className="font-display text-[10rem] leading-none">{r.grade}</p>
                  <p className="font-display text-4xl">{r.grade != null ? ["No DR", "Mild", "Moderate", "Severe", "Proliferative"][r.grade] : ""}</p>
                  <p className={`mt-2 inline-block px-4 py-1 text-sm font-bold uppercase tracking-widest ${r.referable ? "bg-arterial text-paper" : "bg-ink/10"}`}>
                    Referable: {r.referable == null ? "—" : r.referable ? "YES" : "NO"}
                  </p>
                  <dl className="mt-8 space-y-3 border-t border-ink/15 pt-6">
                    <div className="flex justify-between"><dt className="text-xs uppercase tracking-widest text-ink/60">Calibrated confidence</dt><dd className="font-display text-2xl">{r.calibrated_confidence?.toFixed(4)}</dd></div>
                    <div className="flex justify-between"><dt className="text-xs uppercase tracking-widest text-ink/60">Referable score</dt><dd className="font-display text-2xl">{r.referable_score?.toFixed(4)}</dd></div>
                  </dl>
                  <p className="mt-2 text-xs text-ink/55">Calibrated confidence is not clinical certainty.</p>
                  <details className="mt-4 text-sm">
                    <summary className="cursor-pointer uppercase tracking-widest text-xs">Probability distribution</summary>
                    <table className="mt-2 w-full text-sm">
                      <thead><tr className="text-left text-xs uppercase text-ink/55"><th>Grade</th><th>Raw</th><th>Calibrated</th></tr></thead>
                      <tbody>
                        {(r.raw_probabilities ?? []).map((p, i) => (
                          <tr key={i} className="border-t border-ink/10">
                            <td>{i} · {GRADES[i]}</td><td>{p.toFixed(4)}</td>
                            <td>{(r.calibrated_probabilities ?? [])[i]?.toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </details>
                </div>
              </div>

              <div className="mt-16 grid gap-10 md:grid-cols-3">
                <div>
                  <TechLabel>Quality · {r.quality_status}</TechLabel>
                  <p className="mt-2 text-sm text-ink/70">Enhancement: see report for before/after status.</p>
                </div>
                <div>
                  <TechLabel>Lesion evidence</TechLabel>
                  <ul className="mt-2 space-y-1 text-sm">
                    {Object.entries(r.lesion_counts ?? {}).map(([k, v]) => (
                      <li key={k} className="flex justify-between border-b border-ink/10 py-1">
                        <span>{k.replace("_", " ")}</span><span className="font-semibold">{v} candidates</span>
                      </li>
                    ))}
                  </ul>
                  <p className="mt-2 text-xs text-ink/55">Candidates, never confirmed pathology. Neovascularization detection is not implemented.</p>
                </div>
                <div>
                  <TechLabel>Anatomy</TechLabel>
                  <ul className="mt-2 space-y-1 text-sm">
                    <li className="flex justify-between border-b border-ink/10 py-1"><span>Vessels</span><span>{r.anatomy?.vessel_available ? "mapped" : "unavailable"}</span></li>
                    <li className="flex justify-between border-b border-ink/10 py-1"><span>Optic disc</span><span>{r.anatomy?.disc?.status}</span></li>
                    <li className="flex justify-between border-b border-ink/10 py-1"><span>Fovea</span><span>{r.anatomy?.fovea?.status}</span></li>
                    <li className="flex justify-between py-1"><span>Consistency</span><span>{r.consistency}</span></li>
                  </ul>
                </div>
              </div>

              <div className="mt-16 bg-coal p-8 text-paper md:p-12">
                <TechLabel dark>Triage · Phase 8 output, rendered verbatim</TechLabel>
                <p className="mt-4 font-display text-6xl">{r.triage_decision}</p>
                <p className="mt-2 text-sm uppercase tracking-[0.25em] text-paper/70">Priority: {r.triage_priority}</p>
                <div className="mt-6 flex flex-wrap gap-2">
                  {(r.reason_codes ?? []).map((c) => (
                    <span key={c} className="border border-paper/30 px-3 py-1 text-[11px] uppercase tracking-widest">{c}</span>
                  ))}
                </div>
                <p className="mt-6 text-sm text-paper/70">Safety flags: {Object.entries(r.safety_flags ?? {}).filter(([, v]) => v).map(([k]) => k).join(", ") || "none"}</p>
              </div>

              <div className="mt-10 flex flex-wrap gap-3">
                {(["json", "md", "html", "pdf"] as const).map((f) => (
                  <a key={f} href={reportUrl(r.image_hash, f)} className="bg-ink px-6 py-3 text-xs font-semibold uppercase tracking-widest text-paper hover:bg-arterial">
                    Download {f}
                  </a>
                ))}
              </div>
              {(r.warnings ?? []).length > 0 && (
                <details className="mt-6 text-sm">
                  <summary className="cursor-pointer text-xs uppercase tracking-widest">Warnings ({r.warnings.length})</summary>
                  <ul className="mt-2 space-y-1 text-ink/70">{r.warnings.map((w, i) => <li key={i}>— {w}</li>)}</ul>
                </details>
              )}
            </>
          )}
          <div className="mt-16">
            <button onClick={() => { setPhase("upload"); setJob(null); }} className="border border-ink px-8 py-3 text-sm uppercase tracking-widest hover:bg-ink hover:text-paper">
              Screen another image
            </button>
          </div>
        </section>
      )}

      <div className="px-6 md:px-16"><SafetyStrip /></div>
    </main>
  );
}
