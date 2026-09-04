"""PDF renderer (SIH26038 Phase 9). Dependency-free via matplotlib PDF
backend: title, screening summary, key sections, limitations, plus the
explainability figure when available. JSON/Markdown/HTML remain primary;
this PDF is a convenience rendering of the same authoritative dict."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def render_pdf(report, path, figure_path=None):
    s = report["sections"]
    sm = s.get("summary", {})
    tr = s.get("triage", {})
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        lines = ["AI-Assisted Retinal Screening Report", "",
                 f"Image: {s.get('image_id')} | Status: {report.get('status')}", "",
                 "SCREENING SUMMARY",
                 f"Quality: {sm.get('image_quality')}",
                 f"Prediction: {sm.get('model_prediction')}",
                 f"Calibrated confidence: {sm.get('calibrated_confidence')}",
                 f"Referable probability: {sm.get('referable_probability')}",
                 f"Workflow: {sm.get('workflow_recommendation')} "
                 f"[{sm.get('priority')}]", "",
                 "WORKFLOW",
                 f"{tr.get('decision')} ({tr.get('priority')})",
                 *[f"- {c}" for c in (tr.get("reason_codes") or [])], "",
                 "LIMITATIONS",
                 *[f"- {x}" for x in (s.get("limitations", []) or [])], "",
                 ("Screening support only — not a diagnosis. Lesion evidence "
                  "is heuristic; confidence is not clinical certainty.")]
        ax.text(0.05, 0.97, "\n".join(lines), fontsize=9, family="monospace",
                verticalalignment="top", transform=ax.transAxes)
        pdf.savefig(fig)
        plt.close(fig)
        if figure_path:
            try:
                import matplotlib.image as mpimg

                img = mpimg.imread(figure_path)
                fig2, ax2 = plt.subplots(figsize=(8.27, 6))
                ax2.imshow(img)
                ax2.axis("off")
                ax2.set_title("Explainability figure")
                pdf.savefig(fig2)
                plt.close(fig2)
            except (FileNotFoundError, OSError):
                pass
    return str(path)
