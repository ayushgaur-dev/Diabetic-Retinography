"use client";

import { useState } from "react";

const LAYERS = ["original", "gradcam", "vessels", "anatomy", "lesions", "combined"] as const;
type Layer = (typeof LAYERS)[number];

const LABELS: Record<Layer, string> = {
  original: "Original",
  gradcam: "Grad-CAM",
  vessels: "Vessels",
  anatomy: "Disc + Fovea",
  lesions: "Lesions",
  combined: "Combined",
};

export default function Viewer({ layers }: { layers: Record<string, string> | null }) {
  const available = LAYERS.filter((l) => layers?.[l]);
  const [active, setActive] = useState<Layer>("original");
  const shown: Layer = layers?.[active] ? active : "original";
  if (!layers?.original) return <p className="text-sm text-ink/60">No imagery available.</p>;
  return (
    <div>
      <div className="flex flex-wrap gap-2">
        {LAYERS.map((l) => (
          <button
            key={l}
            disabled={!layers?.[l]}
            onClick={() => setActive(l)}
            className={`border px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.2em] disabled:cursor-not-allowed disabled:opacity-30 ${
              shown === l ? "border-ink bg-ink text-paper" : "border-ink/30 hover:border-ink"
            }`}
          >
            {LABELS[l]}
          </button>
        ))}
      </div>
      <figure className="relative mt-4 overflow-hidden rounded-sm bg-coal">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          key={shown}
          src={layers[shown]}
          alt={`Retinal view: ${LABELS[shown]}`}
          className="viewer-fade w-full"
        />
        <figcaption className="flex justify-between px-3 py-2 text-[11px] uppercase tracking-[0.25em] text-paper/60">
          <span>{LABELS[shown]}</span>
          <span>Attribution aid — not proof of disease</span>
        </figcaption>
      </figure>
      <div className="mt-3 flex items-center justify-between text-xs text-ink/60">
        <span>Legend: heat = model attention · cyan = vessels · green = disc · magenta = fovea · yellow = lesion candidates</span>
        <button onClick={() => setActive("original")} className="uppercase tracking-widest underline">
          Reset view
        </button>
      </div>
    </div>
  );
}
