import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RetinaLens — Explainable DR Screening",
  description:
    "Explainable AI-assisted diabetic retinopathy screening for resource-constrained and telemedicine workflows. Research prototype, not a diagnostic device.",
  themeColor: "#08080c",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#08080c] font-body text-[#f2ede4] antialiased">{children}</body>
    </html>
  );
}
