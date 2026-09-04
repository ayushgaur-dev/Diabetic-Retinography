"use client";

import { useEffect } from "react";

export function useReveal() {
  useEffect(() => {
    const els = Array.from(document.querySelectorAll(".reveal"));
    if (!("IntersectionObserver" in window)) {
      els.forEach((el) => el.classList.add("is-visible"));
      return;
    }
    const io = new IntersectionObserver(
      (entries) =>
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("is-visible");
            io.unobserve(e.target);
          }
        }),
      { threshold: 0.12 }
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);
}

export function TechLabel({ children, dark = false }: { children: React.ReactNode; dark?: boolean }) {
  return (
    <p className={`text-[11px] font-semibold uppercase tracking-[0.3em] ${dark ? "text-paper/60" : "text-ink/60"}`}>
      {children}
    </p>
  );
}

export function SafetyStrip() {
  return (
    <div className="border-t border-ink/15 py-10 text-sm leading-relaxed text-ink/70">
      <TechLabel>Research prototype</TechLabel>
      <p className="mt-3 max-w-3xl">
        AI-assisted decision support only — not a diagnostic device. Results require qualified
        human review. Lesion outputs are evidence candidates, not confirmed pathology.
        Calibrated confidence is not clinical certainty. Ungradable images should be
        recaptured or reviewed. Dataset performance does not guarantee deployment performance.
      </p>
    </div>
  );
}
