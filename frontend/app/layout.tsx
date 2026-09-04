import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RetinaLens — Explainable DR Screening",
  description:
    "Explainable AI-assisted diabetic retinopathy screening for resource-constrained and telemedicine workflows. Research prototype, not a diagnostic device.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-paper text-ink font-body antialiased">{children}</body>
    </html>
  );
}
