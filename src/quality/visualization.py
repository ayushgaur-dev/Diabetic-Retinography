"""Quality visualisation (SIH26038 Phase 2).

One honest figure: original image + retinal-field contour overlay +
per-component status table + overall verdict. No decorative AI visuals.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_quality_figure(rgb, result, mask, path):
    """Save a quality-assessment PNG. Returns the path as a string."""
    import cv2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title("Original fundus image")
    axes[0].axis("off")
    overlay = rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    axes[1].imshow(overlay)
    axes[1].set_title("Detected retinal field")
    axes[1].axis("off")
    lines = [
        f"QUALITY: {result.status}  (score {result.overall_score:.2f})",
        "",
        f"Focus: {result.focus.status} ({result.focus.measurement:.0f})",
        f"Illumination: {result.illumination.status} ({result.illumination.measurement:.0f})",
        f"Contrast: {result.contrast.status} ({result.contrast.measurement:.1f})",
        f"Exposure: {result.exposure.status} (clipped {result.exposure.measurement:.2%})",
        f"Field of view: {result.field_of_view.status} ({result.field_of_view.measurement:.0%})",
        f"Retinal coverage: {result.retinal_coverage.status} ({result.retinal_coverage.measurement:.0%})",
    ]
    if result.reasons:
        lines += ["", "Reasons:"] + [f"- {r}" for r in result.reasons]
    if result.recapture_feedback:
        lines += ["", "Recapture:"] + [f"- {m}" for m in result.recapture_feedback]
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=8, family="monospace",
             verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    fig.tight_layout(rect=[0, 0.25, 1, 1])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
