"use client";

import Link from "next/link";
import { SafetyStrip, TechLabel, useReveal } from "../components/chrome";

function StoryStep({
  n,
  title,
  body,
  points,
  dark = false,
  children,
}: {
  n: string;
  title: string;
  body: string;
  points: string[];
  dark?: boolean;
  children?: React.ReactNode;
}) {
  return (
    <section className={`reveal px-6 py-24 md:px-16 lg:px-28 ${dark ? "bg-coal text-paper" : ""}`}>
      <div className="grid gap-10 md:grid-cols-12">
        <div className="md:col-span-2">
          <p className="font-display text-7xl leading-none md:text-8xl">{n}</p>
        </div>
        <div className="md:col-span-4">
          <TechLabel dark={dark}>Phase {n}</TechLabel>
          <h2 className="mt-4 font-display text-5xl leading-[1.02] md:text-6xl">{title}</h2>
          <p className={`mt-6 max-w-md text-lg leading-relaxed ${dark ? "text-paper/75" : "text-ink/75"}`}>{body}</p>
          <ul className="mt-6 space-y-2 text-sm">
            {points.map((p) => (
              <li key={p} className={`border-l-2 pl-3 ${dark ? "border-arterial" : "border-arterial"}`}>
                {p}
              </li>
            ))}
          </ul>
        </div>
        <div className="md:col-span-6">{children}</div>
      </div>
    </section>
  );
}

export default function Landing() {
  useReveal();
  return (
    <main className="bg-paper text-ink">
      <header className="flex items-center justify-between px-6 py-5 md:px-16">
        <p className="font-display text-2xl tracking-tight">RetinaLens</p>
        <nav className="flex items-center gap-6 text-sm">
          <Link href="#story" className="hidden sm:inline hover:underline">How it works</Link>
          <Link href="#evidence" className="hidden sm:inline hover:underline">Evidence</Link>
          <Link href="/screen" className="border border-ink px-5 py-2 text-sm font-semibold uppercase tracking-widest hover:bg-ink hover:text-paper">
            Start screening
          </Link>
        </nav>
      </header>

      {/* HERO */}
      <section className="tech-grid relative overflow-hidden px-6 pb-16 pt-10 md:px-16 md:pt-16">
        <TechLabel>Explainable retinal screening · SIH26038</TechLabel>
        <div className="grid items-end gap-8 md:grid-cols-12">
          <h1 className="font-display leading-[0.9] md:col-span-7">
            <span className="block text-[19vw] md:text-[11rem]">SEE</span>
            <span className="block text-[19vw] md:text-[11rem]">THE</span>
            <span className="block text-[19vw] text-arterial md:text-[11rem]">RETINA.</span>
          </h1>
          <div className="md:col-span-5 md:pb-6">
            <p className="max-w-md text-xl leading-relaxed text-ink/80">
              Explainable AI-assisted diabetic retinopathy screening for
              resource-constrained and telemedicine workflows.
            </p>
            <div className="mt-8 flex flex-wrap gap-4">
              <Link href="/screen" className="bg-ink px-8 py-4 text-sm font-semibold uppercase tracking-widest text-paper hover:bg-arterial">
                Start screening
              </Link>
              <Link href="#story" className="border border-ink px-8 py-4 text-sm font-semibold uppercase tracking-widest hover:bg-ink hover:text-paper">
                How it works
              </Link>
            </div>
          </div>
        </div>
        <figure className="relative mt-12 md:ml-[38%]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/hero-retina.png" alt="Retinal fundus photograph used for screening" className="w-full rounded-sm" />
          <figcaption className="mt-2 flex justify-between text-[11px] uppercase tracking-[0.25em] text-ink/55">
            <span>Fig. 01 — Fundus, 45° field</span>
            <span>Ø optic disc · temporal raphe · fovea</span>
          </figcaption>
          <div className="pointer-events-none absolute left-[8%] top-[30%] hidden h-24 w-px bg-arterial md:block" />
          <div className="pointer-events-none absolute left-[8%] top-[30%] hidden pl-3 text-[11px] uppercase tracking-[0.25em] text-arterial md:block">
            Optic disc
          </div>
        </figure>
      </section>

      <div id="story">
        <StoryStep n="01" title="Capture" body="A retinal image enters the system. The pipeline first asks whether the photograph itself is usable — before any model is allowed to speak." points={["Fundus upload", "Format + integrity checks", "No patient metadata stored"]}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/retina-detail.png" alt="Detail of retinal vasculature" className="w-full rounded-sm" />
        </StoryStep>

        <StoryStep n="02" title="Understand" dark body="Focus, illumination, contrast, exposure and field of view are measured with interpretable image processing. Ungradable images are stopped here — never silently graded." points={["GOOD / BORDERLINE / UNGRADABLE", "Focus · illumination · contrast", "Recapture guidance when blocked"]}>
          <div className="tech-grid-dark rounded-sm bg-coal p-8 text-paper">
            <TechLabel dark>Quality gate · live values</TechLabel>
            <dl className="mt-6 space-y-4 font-display text-3xl">
              {[["FOCUS", "measured"], ["ILLUMINATION", "measured"], ["CONTRAST", "measured"], ["FIELD OF VIEW", "measured"]].map(([k, v]) => (
                <div key={k} className="flex items-baseline justify-between border-b border-paper/15 pb-3">
                  <dt className="text-sm uppercase tracking-[0.25em] text-paper/60">{k}</dt>
                  <dd>{v}</dd>
                </div>
              ))}
            </dl>
          </div>
        </StoryStep>

        <StoryStep n="03" title="Map" body="The retina is mapped into meaningful structures: the vascular tree, the optic disc and the fovea. Each is localized by its own interpretable module." points={["Vessel segmentation", "Optic disc localization", "Fovea localization"]}>
          <div className="grid grid-cols-3 gap-3 text-center">
            {["VESSELS", "OPTIC DISC", "FOVEA"].map((t) => (
              <div key={t} className="rounded-sm border border-ink/20 px-2 py-10">
                <p className="text-[11px] uppercase tracking-[0.25em] text-ink/60">{t}</p>
                <p className="mt-3 font-display text-2xl">mapped</p>
              </div>
            ))}
          </div>
        </StoryStep>

        <StoryStep n="04" title="Find evidence" body="Candidate lesion regions are surfaced as evidence — microaneurysms, hemorrhages, exudates. Candidates are never presented as confirmed disease." points={["4 lesion evidence modules", "Candidate boxes + scores", "Optic-disc exclusion for bright lesions"]}>
          <div className="rounded-sm bg-ink p-8 text-paper">
            <TechLabel dark>Evidence, not diagnosis</TechLabel>
            <p className="mt-4 font-display text-4xl leading-tight">Candidates,<br />not confirmations.</p>
          </div>
        </StoryStep>

        <StoryStep n="05" title="Explain" dark body="Grad-CAM attribution is fused with explicit evidence into one map. Spatial agreement is reported honestly — overlap is not proof of causation." points={["Grad-CAM attribution", "Evidence overlays", "Consistency: supportive → conflicting"]}>
          <div className="tech-grid-dark rounded-sm bg-coal p-8 text-paper">
            <TechLabel dark>Attribution aid</TechLabel>
            <p className="mt-4 max-w-md text-lg text-paper/80">
              “The heatmap indicates image regions associated with the model prediction.
              It is an attribution aid, not proof of disease.”
            </p>
          </div>
        </StoryStep>

        <StoryStep n="06" title="Act" body="A deterministic rule engine converts grades, calibration, quality and evidence into a screening workflow recommendation — routine, refer, urgent, or technical review." points={["UNGRADABLE · ROUTINE · REFER · URGENT", "Reason codes + safety flags", "Calibrated, not clinical, confidence"]}>
          <div id="evidence" className="rounded-sm border border-ink/20 p-8">
            <TechLabel>Held-out engineering results</TechLabel>
            <dl className="mt-4 space-y-3 font-display text-4xl">
              <div className="flex justify-between border-b border-ink/10 pb-2"><dt className="text-sm uppercase tracking-widest text-ink/60">Referable sensitivity</dt><dd>96.0%</dd></div>
              <div className="flex justify-between border-b border-ink/10 pb-2"><dt className="text-sm uppercase tracking-widest text-ink/60">Referable specificity</dt><dd>87.2%</dd></div>
              <div className="flex justify-between"><dt className="text-sm uppercase tracking-widest text-ink/60">ROC-AUC</dt><dd>0.974</dd></div>
            </dl>
            <p className="mt-4 text-xs text-ink/60">Engineering evaluation on held-out APTOS data — not clinical validation.</p>
            <Link href="/screen" className="mt-6 inline-block bg-arterial px-8 py-4 text-sm font-semibold uppercase tracking-widest text-paper hover:bg-ink">
              Start screening
            </Link>
          </div>
        </StoryStep>
      </div>

      <footer className="bg-coal px-6 py-14 text-paper md:px-16">
        <p className="font-display text-3xl">RetinaLens</p>
        <p className="mt-3 max-w-2xl text-sm text-paper/65">
          SIH26038 — Explainable AI for Diabetic Retinopathy Screening in Rural India.
          Research prototype. Results require qualified human review.
        </p>
      </footer>
      <div className="px-6 md:px-16"><SafetyStrip /></div>
    </main>
  );
}
