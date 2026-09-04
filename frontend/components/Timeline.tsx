"use client";

import { STAGES, STAGE_LABELS } from "../lib/types";

const ORDER = ["queued", "starting", "quality", "enhancement", "grading", "evidence", "triage", "done"];

export default function Timeline({ stage, done }: { stage: string | null; done: boolean }) {
  const idx = ORDER.indexOf(stage ?? "queued");
  return (
    <ol className="space-y-3">
      {STAGES.map((s) => {
        const si = ORDER.indexOf(s);
        const isActive = stage === s;
        const passed = done || (idx >= 0 && si < idx);
        return (
          <li key={s} className="flex items-center gap-4">
            <span
              className={`stage-dot h-3 w-3 rounded-full border ${
                passed ? "border-arterial bg-arterial" : isActive ? "border-arterial" : "border-ink/30"
              }`}
            />
            <span className={`text-sm uppercase tracking-[0.2em] ${passed || isActive ? "" : "text-ink/40"}`}>
              {STAGE_LABELS[s]}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
