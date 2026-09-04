"use client";

import Link from "next/link";
import { Counter, SafetyStrip, TechLabel, useReveal } from "../components/chrome";

function StoryStep({
  n,
  title,
  body,
  points,
  children,
}: {
  n: string;
  title: string;
  body: string;
  points: string[];
  children?: React.ReactNode;
}) {
  return (
    <section className="reveal relative px-6 py-24 md:px-16 lg:px-28">
      <div className="grid gap-10 md:grid-cols-12">
        <div className="md:col-span-2">
          <p className="font-display text-7xl leading-none text-white/15 md:text-8xl">{n}</p>
        </div>
        <div className="md:col-span-4">
          <TechLabel>Phase {n}</TechLabel>
          <h2 className="mt-4 font-display text-5xl leading-[1.02] md:text-6xl">{title}</h2>
          <p className="mt-6 max-w-md text-lg leading-relaxed text-white/70">{body}</p>
          <ul className="mt-6 space-y-2 text-sm text-white/80">
            {points.map((p) => (
              <li key={p} className="border-l-2 border-[#ff5a4e] pl-3">
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
    <main className="relative overflow-hidden bg-[#08080c] text-[#f2ede4]">
      <div className="orb orb-red left-[-200px] top-[-100px] h-[500px] w-[500px]" />
      <div className="orb orb-amber right-[-150px] top-[40%] h-[420px] w-[420px]" />

      <header className="relative z-10 flex items-center justify-between px-6 py-5 md:px-16">
        <p className="font-display text-2xl tracking-tight">RetinaLens</p>
        <nav className="flex items-center gap-6 text-sm text-white/70">
          <Link href="#story" className="hidden transition-colors hover:text-white sm:inline">How it works</Link>
          <Link href="#evidence" className="hidden transition-colors hover:text-white sm:inline">Evidence</Link>
          <Link href="/screen" className="glass rounded-full px-5 py-2 text-sm font-semibold uppercase tracking-widest transition-all hover:border-[#ff5a4e]/60">
            Start screening
          </Link>
        </nav>
      </header>

      {/* HERO */}
      <section className="tech-grid-dark relative px-6 pb-16 pt-10 md:px-16 md:pt-16">
        <TechLabel>Explainable retinal screening · SIH26038</TechLabel>
        <div className="grid items-end gap-8 md:grid-cols-12">
          <h1 className="font-display leading-[0.9] md:col-span-7">
            <span className="block text-[19vw] md:text-[11rem]">SEE</span>
            <span className="block text-[19vw] md:text-[11rem]">THE</span>
            <span className="grade-glow block text-[19vw] text-[#ff5a4e] md:text-[11rem]">RETINA.</span>
          </h1>
          <div className="md:col-span-5 md:pb-6">
            <p className="max-w-md text-xl leading-relaxed text-white/75">
              Explainable AI-assisted diabetic retinopathy screening for
              resource-constrained and telemedicine workflows.
            </p>
            <div className="mt-8 flex flex-wrap gap-4">
              <Link href="/screen" className="bg-[#ff5a4e] px-8 py-4 text-sm font-semibold uppercase tracking-widest text-white shadow-[0_0_40px_rgb(255,90,78,0.35)] transition-all hover:bg-white hover:text-black">
                Start screening
              </Link>
              <Link href="#story" className="glass rounded-sm px-8 py-4 text-sm font-semibold uppercase tracking-widest transition-all hover:border-white/40">
                How it works
              </Link>
            </div>
            <div className="glass mt-8 flex gap-8 rounded-2xl p-5">
              <div><p className="font-display text-3xl"><Counter value={96} suffix="%" /></p><p className="text-[11px] uppercase tracking-widest text-white/50">Referable sensitivity</p></div>
              <div><p className="font-display text-3xl"><Counter value={0.974} decimals={3} /></p><p className="text-[11px] uppercase tracking-widest text-white/50">ROC-AUC</p></div>
              <div><p className="font-display text-3xl"><Counter value={9} /></p><p className="text-[11px] uppercase tracking-widest text-white/50">Pipeline stages</p></div>
            </div>
          </div>
        </div>
        <figure className="relative mt-12 md:ml-[38%]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/hero-retina.png" alt="Retinal fundus photograph used for screening" className="w-full rounded-2xl shadow-[0_30px_80px_rgb(0,0,0,0.6)]" />
          <div className="glass pointer-events-none absolute left-[6%] top-[8%] hidden rounded-full px-4 py-2 text-[11px] uppercase tracking-[0.25em] md:block">
            Ø optic disc
          </div>
          <div className="glass pointer-events-none absolute bottom-[10%] right-[6%] hidden rounded-full px-4 py-2 text-[11px] uppercase tracking-[0.25em] md:block">
            Fovea
          </div>
          <figcaption className="mt-2 flex justify-between text-[11px] uppercase tracking-[0.25em] text-white/45">
            <span>Fig. 01 — Fundus, 45° field</span>
            <span>Engineering demo imagery</span>
          </figcaption>
        </figure>
      </section>

      <div id="story">
        <StoryStep n="01" title="Capture" body="A retinal image enters the system. The pipeline first asks whether the photograph itself is usable — before any model is allowed to speak." points={["Fundus upload", "Format + integrity checks", "No patient metadata stored"]}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/retina-detail.png" alt="Detail of retinal vasculature" className="w-full rounded-2xl shadow-[0_20px_60px_rgb(0,0,0,0.5)]" />
        </StoryStep>

        <StoryStep n="02" title="Understand" body="Focus, illumination, contrast, exposure and field of view are measured with interpretable image processing. Ungradable images are stopped here — never silently graded." points={["GOOD / BORDERLINE / UNGRADABLE", "Focus · illumination · contrast", "Recapture guidance when blocked"]}>
          <div className="glass rounded-2xl p-8">
            <TechLabel>Quality gate · live values</TechLabel>
            <dl className="mt-6 space-y-4 font-display text-3xl">
              {[["Focus", "measured"], ["Illumination", "measured"], ["Contrast", "measured"], ["Field of view", "measured"]].map(([k, v]) => (
                <div key={k} className="flex items-baseline justify-between border-b border-white/10 pb-3">
                  <dt className="text-sm uppercase tracking-[0.25em] text-white/50">{k}</dt>
                  <dd>{v}</dd>
                </div>
              ))}
            </dl>
          </div>
        </StoryStep>

        <StoryStep n="03" title="Map" body="The retina is mapped into meaningful structures: the vascular tree, the optic disc and the fovea. Each is localized by its own interpretable module." points={["Vessel segmentation", "Optic disc localization", "Fovea localization"]}>
          <div className="grid grid-cols-3 gap-3 text-center">
            {["Vessels", "Optic disc", "Fovea"].map((t) => (
              <div key={t} className="glass rounded-2xl px-2 py-10 transition-transform hover:scale-[1.03]">
                <p className="text-[11px] uppercase tracking-[0.25em] text-white/50">{t}</p>
                <p className="mt-3 font-display text-2xl">mapped</p>
              </div>
            ))}
          </div>
        </StoryStep>

        <StoryStep n="04" title="Find evidence" body="Candidate lesion regions are surfaced as evidence — microaneurysms, hemorrhages, exudates. Candidates are never presented as confirmed disease." points={["4 lesion evidence modules", "Candidate boxes + scores", "Optic-disc exclusion for bright lesions"]}>
          <div className="rounded-2xl bg-gradient-to-br from-[#ff5a4e]/25 to-transparent p-8 backdrop-blur-xl">
            <TechLabel>Evidence, not diagnosis</TechLabel>
            <p className="mt-4 font-display text-4xl leading-tight md:text-5xl">Candidates,<br />not confirmations.</p>
          </div>
        </StoryStep>

        <StoryStep n="05" title="Explain" body="Grad-CAM attribution is fused with explicit evidence into one map. Spatial agreement is reported honestly — overlap is not proof of causation." points={["Grad-CAM attribution", "Evidence overlays", "Consistency: supportive → conflicting"]}>
          <div className="glass rounded-2xl p-8">
            <TechLabel>Attribution aid</TechLabel>
            <p className="mt-4 max-w-md text-lg text-white/75">
              “The heatmap indicates image regions associated with the model prediction.
              It is an attribution aid, not proof of disease.”
            </p>
          </div>
        </StoryStep>

        <StoryStep n="06" title="Act" body="A deterministic rule engine converts grades, calibration, quality and evidence into a screening workflow recommendation — routine, refer, urgent, or technical review." points={["UNGRADABLE · ROUTINE · REFER · URGENT", "Reason codes + safety flags", "Calibrated, not clinical, confidence"]}>
          <div id="evidence" className="glass rounded-2xl p-8">
            <TechLabel>Held-out engineering results</TechLabel>
            <dl className="mt-4 space-y-3 font-display text-4xl">
              <div className="flex justify-between border-b border-white/10 pb-2"><dt className="text-sm uppercase tracking-widest text-white/50">Referable sensitivity</dt><dd><Counter value={96.0} decimals={1} suffix="%" /></dd></div>
              <div className="flex justify-between border-b border-white/10 pb-2"><dt className="text-sm uppercase tracking-widest text-white/50">Referable specificity</dt><dd><Counter value={87.2} decimals={1} suffix="%" /></dd></div>
              <div className="flex justify-between"><dt className="text-sm uppercase tracking-widest text-white/50">ROC-AUC</dt><dd><Counter value={0.974} decimals={3} /></dd></div>
            </dl>
            <p className="mt-4 text-xs text-white/50">Engineering evaluation on held-out APTOS data — not clinical validation.</p>
            <Link href="/screen" className="mt-6 inline-block bg-[#ff5a4e] px-8 py-4 text-sm font-semibold uppercase tracking-widest text-white shadow-[0_0_40px_rgb(255,90,78,0.35)] transition-all hover:bg-white hover:text-black">
              Start screening
            </Link>
          </div>
        </StoryStep>
      </div>

      <footer className="border-t border-white/10 px-6 py-14 md:px-16">
        <p className="font-display text-3xl">RetinaLens</p>
        <p className="mt-3 max-w-2xl text-sm text-white/55">
          SIH26038 — Explainable AI for Diabetic Retinopathy Screening in Rural India.
          Research prototype. Results require qualified human review.
        </p>
      </footer>
      <div className="px-6 md:px-16"><SafetyStrip /></div>
    </main>
  );
}
