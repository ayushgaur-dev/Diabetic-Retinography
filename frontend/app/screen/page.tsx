"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import Timeline from "../../components/Timeline";
import Viewer from "../../components/Viewer";
import { SafetyStrip, TechLabel, useReveal } from "../../components/chrome";
import { reportUrl, startDemo, startScreen, waitForJob } from "../../lib/api";
import type { JobState } from "../../lib/types";

const GRADES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"];

type Phase = "upload" | "working" | "review";

function useElapsed(running: boolean) {
  const [s, setS] = useState(0);
  useEffect(() => {
    if (!running) return;
    setS(0);
    const t0 = Date.now();
    const id = setInterval(() => setS(Math.floor((Date.now() - t0) / 1000)), 500);
    return () => clearInterval(id);
  }, [running]);
  return s;
}

export default function Screen() {
  useReveal();
  const [phase, setPhase] = useState<Phase>("upload");
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [job, setJob] = useState<JobState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fullEvidence, setFullEvidence] = useState(true);
  const fileRef = useRef<File | null>(null);
  const elapsed = useElapsed(phase === "working");

  const onFile = useCallback((f: File | undefined) => {
    if (!f) return;
    fileRef.current = f;
    setJob(null);
    setError(null);
    setPhase("upload");
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(f);
    });
  }, []);

  const run = useCallback(async (jobId: string) => {
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
  }, []);

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
    <main className="relative min-h-screen overflow-hidden bg-[#08080c] text-[#f2ede4]">
      <div className="orb orb-red left-[-180px] top-[10%] h-[420px] w-[420px]" />
      <header className="relative z-10 flex items-center justify-between px-6 py-5 md:px-16">
        <Link href="/" className="font-display text-2xl tracking-tight">RetinaLens</Link>
        <TechLabel>Screening · Upload → Analyze → Review</TechLabel>
      </header>

      {phase === "upload" && (
        <section className="relative z-10 px-6 pb-24 pt-10 md:px-16">
          <TechLabel>01 · Upload</TechLabel>
          <h1 className="mt-4 font-display text-6xl leading-[0.95] md:text-8xl">
            Drop fundus<br />image.
          </h1>
          <div
            className={`mt-10 rounded-3xl border-2 border-dashed px-6 py-20 text-center backdrop-blur-xl transition-all duration-300 ${
              dragging ? "border-[#ff5a4e] bg-[#ff5a4e]/10 shadow-[0_0_60px_rgb(255,90,78,0.25)]" : "border-white/20 bg-white/[0.03]"
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); onFile(e.dataTransfer.files?.[0]); }}
          >
            <p className="font-display text-3xl">DROP FUNDUS IMAGE</p>
            <p className="mt-3 text-sm text-white/50">or</p>
            <label className="mt-4 inline-block cursor-pointer rounded-full bg-white px-8 py-3 text-sm font-semibold uppercase tracking-widest text-black transition-all hover:bg-[#ff5a4e] hover:text-white">
              Choose image
              <input type="file" accept="image/png,image/jpeg,image/webp" className="hidden"
                onChange={(e) => onFile(e.target.files?.[0])} />
            </label>
            <p className="mt-4 text-xs uppercase tracking-[0.2em] text-white/40">PNG · JPEG · WEBP · up to 15 MB · no patient data stored</p>
          </div>
          {preview && (
            <div className="rise-in mt-8 grid gap-8 md:grid-cols-2">
              <div className="relative overflow-hidden rounded-2xl border border-white/10">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={preview} alt="Uploaded fundus preview" className="w-full" />
                <div className="scanline" />
              </div>
              <div className="glass rounded-2xl p-8">
                <TechLabel>Ready to analyze</TechLabel>
                <label className="mt-4 flex cursor-pointer items-center gap-3 text-sm text-white/80">
                  <input type="checkbox" checked={fullEvidence} onChange={(e) => setFullEvidence(e.target.checked)} className="h-4 w-4 accent-[#ff5a4e]" />
                  Full evidence workup (slower, complete)
                </label>
                <button onClick={submit} className="mt-6 w-full bg-[#ff5a4e] px-10 py-4 text-sm font-semibold uppercase tracking-widest text-white shadow-[0_0_40px_rgb(255,90,78,0.35)] transition-all hover:bg-white hover:text-black">
                  Analyze retina
                </button>
              </div>
            </div>
          )}
          <div className="mt-10">
            <button onClick={demo} className="glass rounded-full px-8 py-3 text-sm uppercase tracking-widest transition-all hover:border-white/40">
              Try synthetic demo
            </button>
            <p className="mt-2 text-xs text-white/45">Deterministic non-PHI fixture through the real pipeline.</p>
          </div>
          {error && <p className="glass mt-6 rounded-2xl border-[#ff5a4e]/50 p-4 text-sm text-[#ff8a80]">{error}</p>}
        </section>
      )}

      {phase === "working" && (
        <section className="relative z-10 px-6 py-16 md:px-16">
          <div className="grid gap-12 lg:grid-cols-2">
            <div>
              <TechLabel>02 · Analyze</TechLabel>
              <h2 className="mt-4 font-display text-5xl md:text-6xl">Reading<br />the retina…</h2>
              <p className="mt-4 font-display text-6xl text-[#ff5a4e] tabular-nums">{elapsed}s</p>
              <div className="glass mt-8 max-w-md rounded-2xl p-8"><Timeline stage={stage} done={false} /></div>
              <p className="mt-6 text-sm text-white/50">Stages reflect actual backend progress — nothing is fabricated.</p>
              {error && <p className="mt-6 rounded-2xl border border-[#ff5a4e]/50 p-4 text-sm text-[#ff8a80]">{error}</p>}
            </div>
            <div className="relative self-start overflow-hidden rounded-3xl border border-white/10">
              {preview ? (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={preview} alt="Image under analysis" className="w-full opacity-80" />
                  <div className="scanline" />
                  <div className="absolute inset-0 shimmer" />
                </>
              ) : (
                <div className="shimmer aspect-square w-full" />
              )}
              <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
                <div className="pulse-ring absolute inset-0 rounded-full border-2 border-[#ff5a4e]" />
                <div className="glass flex h-24 w-24 items-center justify-center rounded-full text-[11px] uppercase tracking-widest">
                  {stage?.toUpperCase() ?? "…"}
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {phase === "review" && r && (
        <section className="rise-in relative z-10 px-6 pb-24 pt-10 md:px-16">
          <TechLabel>03 · Review</TechLabel>
          {blocked ? (
            <div className="glass mt-6 rounded-3xl border-[#ff5a4e]/40 p-8 md:p-12">
              <h2 className="font-display text-5xl leading-tight md:text-7xl">IMAGE NOT<br />SUITABLE FOR<br />AUTOMATED GRADING</h2>
              <p className="mt-6 max-w-2xl text-lg text-white/75">Reason: {r.blocked_reason}. No DR grade was produced.</p>
              <div className="mt-6 flex flex-wrap gap-3">
                {(["json", "md", "html", "pdf"] as const).map((f) => (
                  <a key={f} href={reportUrl(r.image_hash, f)} className="glass rounded-full px-5 py-2 text-xs uppercase tracking-widest transition-all hover:border-white/40">
                    Download {f}
                  </a>
                ))}
              </div>
            </div>
          ) : (
            <>
              <p className="mt-6 font-display text-2xl text-white/70">Screening result</p>
              <div className="grid gap-10 lg:grid-cols-12">
                <div className="lg:col-span-7">
                  <Viewer layers={job?.layers ?? null} />
                </div>
                <div className="lg:col-span-5">
                  <div className="glass rounded-3xl p-8">
                    <p className="grade-glow font-display text-[9rem] leading-none text-[#ff5a4e]">{r.grade}</p>
                    <p className="font-display text-4xl">{r.grade != null ? ["No DR", "Mild", "Moderate", "Severe", "Proliferative"][r.grade] : ""}</p>
                    <p className={`mt-3 inline-block rounded-full px-4 py-1 text-sm font-bold uppercase tracking-widest ${r.referable ? "bg-[#ff5a4e] text-white" : "bg-white/10"}`}>
                      Referable: {r.referable == null ? "—" : r.referable ? "YES" : "NO"}
                    </p>
                    <dl className="mt-8 space-y-3 border-t border-white/10 pt-6">
                      <div className="flex justify-between"><dt className="text-xs uppercase tracking-widest text-white/50">Calibrated confidence</dt><dd className="font-display text-2xl">{r.calibrated_confidence?.toFixed(4)}</dd></div>
                      <div className="flex justify-between"><dt className="text-xs uppercase tracking-widest text-white/50">Referable score</dt><dd className="font-display text-2xl">{r.referable_score?.toFixed(4)}</dd></div>
                    </dl>
                    <p className="mt-2 text-xs text-white/45">Calibrated confidence is not clinical certainty.</p>
                    <details className="mt-4 text-sm">
                      <summary className="cursor-pointer text-xs uppercase tracking-widest text-white/60">Probability distribution</summary>
                      <table className="mt-2 w-full text-sm">
                        <thead><tr className="text-left text-xs uppercase text-white/45"><th>Grade</th><th>Raw</th><th>Calibrated</th></tr></thead>
                        <tbody>
                          {(r.raw_probabilities ?? []).map((p, i) => (
                            <tr key={i} className="border-t border-white/10">
                              <td>{i} · {GRADES[i]}</td><td>{p.toFixed(4)}</td>
                              <td>{(r.calibrated_probabilities ?? [])[i]?.toFixed(4)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </details>
                  </div>
                </div>
              </div>

              <div className="mt-12 grid gap-6 md:grid-cols-3">
                <div className="glass rounded-2xl p-6">
                  <TechLabel>Quality · {r.quality_status}</TechLabel>
                  <p className="mt-2 text-sm text-white/60">Enhancement: see report for before/after status.</p>
                </div>
                <div className="glass rounded-2xl p-6">
                  <TechLabel>Lesion evidence</TechLabel>
                  <ul className="mt-2 space-y-1 text-sm">
                    {Object.entries(r.lesion_counts ?? {}).map(([k, v]) => (
                      <li key={k} className="flex justify-between border-b border-white/10 py-1">
                        <span className="text-white/75">{k.replace("_", " ")}</span><span className="font-semibold">{v} candidates</span>
                      </li>
                    ))}
                  </ul>
                  <p className="mt-2 text-xs text-white/45">Candidates, never confirmed pathology. Neovascularization detection is not implemented.</p>
                </div>
                <div className="glass rounded-2xl p-6">
                  <TechLabel>Anatomy</TechLabel>
                  <ul className="mt-2 space-y-1 text-sm">
                    <li className="flex justify-between border-b border-white/10 py-1"><span className="text-white/75">Vessels</span><span>{r.anatomy?.vessel_available ? "mapped" : "unavailable"}</span></li>
                    <li className="flex justify-between border-b border-white/10 py-1"><span className="text-white/75">Optic disc</span><span>{r.anatomy?.disc?.status}</span></li>
                    <li className="flex justify-between border-b border-white/10 py-1"><span className="text-white/75">Fovea</span><span>{r.anatomy?.fovea?.status}</span></li>
                    <li className="flex justify-between py-1"><span className="text-white/75">Consistency</span><span>{r.consistency}</span></li>
                  </ul>
                </div>
              </div>

              <div className="mt-10 rounded-3xl border border-[#ff5a4e]/30 bg-gradient-to-br from-[#ff5a4e]/15 to-transparent p-8 backdrop-blur-xl md:p-12">
                <TechLabel>Triage · Phase 8 output, rendered verbatim</TechLabel>
                <p className="mt-4 font-display text-6xl">{r.triage_decision}</p>
                <p className="mt-2 text-sm uppercase tracking-[0.25em] text-white/60">Priority: {r.triage_priority}</p>
                <div className="mt-6 flex flex-wrap gap-2">
                  {(r.reason_codes ?? []).map((c) => (
                    <span key={c} className="rounded-full border border-white/20 bg-white/5 px-3 py-1 text-[11px] uppercase tracking-widest">{c}</span>
                  ))}
                </div>
                <p className="mt-6 text-sm text-white/60">Safety flags: {Object.entries(r.safety_flags ?? {}).filter(([, v]) => v).map(([k]) => k).join(", ") || "none"}</p>
              </div>

              <div className="mt-10 flex flex-wrap gap-3">
                {(["json", "md", "html", "pdf"] as const).map((f) => (
                  <a key={f} href={reportUrl(r.image_hash, f)} className="glass rounded-full px-6 py-3 text-xs font-semibold uppercase tracking-widest transition-all hover:border-[#ff5a4e]/60">
                    Download {f}
                  </a>
                ))}
              </div>
              {(r.warnings ?? []).length > 0 && (
                <details className="mt-6 text-sm">
                  <summary className="cursor-pointer text-xs uppercase tracking-widest text-white/60">Warnings ({r.warnings.length})</summary>
                  <ul className="mt-2 space-y-1 text-white/60">{r.warnings.map((w, i) => <li key={i}>— {w}</li>)}</ul>
                </details>
              )}
            </>
          )}
          <div className="mt-16">
            <button onClick={() => { setPhase("upload"); setJob(null); }} className="glass rounded-full px-8 py-3 text-sm uppercase tracking-widest transition-all hover:border-white/40">
              Screen another image
            </button>
          </div>
        </section>
      )}

      <div className="relative z-10 px-6 md:px-16"><SafetyStrip /></div>
    </main>
  );
}
