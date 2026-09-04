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
  if (!layers?.original) return <p className="text-sm text-white/50">No imagery available.</p>;
  return (
    <div>
      <div className="flex flex-wrap gap-2">
        {LAYERS.map((l) => (
          <button
            key={l}
            disabled={!layers?.[l]}
            onClick={() => setActive(l)}
            className={`rounded-full border px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.2em] backdrop-blur-md transition-all disabled:cursor-not-allowed disabled:opacity-30 ${
              shown === l
                ? "border-[#ff5a4e]/70 bg-[#ff5a4e]/20 text-white shadow-[0_0_20px_rgb(255,90,78,0.25)]"
                : "border-white/15 bg-white/5 hover:border-white/40"
            }`}
          >
            {LABELS[l]}
          </button>
        ))}
      </div>
      <figure className="relative mt-4 overflow-hidden rounded-2xl border border-white/10 bg-black shadow-[0_20px_60px_rgb(0,0,0,0.5)]">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          key={shown}
          src={layers[shown]}
          alt={`Retinal view: ${LABELS[shown]}`}
          className="viewer-fade w-full"
        />
        <figcaption className="flex justify-between px-4 py-2 text-[11px] uppercase tracking-[0.25em] text-white/50">
          <span>{LABELS[shown]}</span>
          <span>Attribution aid — not proof of disease</span>
        </figcaption>
      </figure>
      <div className="mt-3 flex items-center justify-between text-xs text-white/50">
        <span>Legend: heat = model attention · cyan = vessels · green = disc · magenta = fovea · yellow = lesion candidates</span>
        <button onClick={() => setActive("original")} className="uppercase tracking-widest underline hover:text-white">
          Reset view
        </button>
      </div>
      <p className="mt-1 text-xs text-white/40">{available.length} of {LAYERS.length} layers available.</p>
    </div>
  );
}
