"""Lesion visualisation: multi-panel diagnostic figure (SIH26038 Phase 4D).

Labels say 'evidence', never 'confirmed'. Panels: original | FOV+vessels+
disc+fovea context | 4 evidence overlays with candidate boxes+scores.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def save_lesion_figure(rgb, results, path, context=None):
    """results: dict lesion_type -> LesionResult.to_dict(). context: optional
    dict with fov/vessel/disc_xy/fovea_xy (working or original frame note)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original fundus")
    axes[0, 1].imshow(rgb)
    if context:
        if context.get("disc_xy") is not None:
            x, y = context["disc_xy"]
            axes[0, 1].plot(x, y, "co", ms=8, label="disc")
        if context.get("fovea_xy") is not None:
            x, y = context["fovea_xy"]
            axes[0, 1].plot(x, y, "m+", ms=10, mew=1.5, label="fovea")
        axes[0, 1].legend(fontsize=7)
    axes[0, 1].set_title("Anatomical context")
    axes[0, 2].axis("off")
    order = ["microaneurysm", "hemorrhage", "hard_exudate", "soft_exudate"]
    titles = {"microaneurysm": "Microaneurysm evidence",
              "hemorrhage": "Hemorrhage evidence",
              "hard_exudate": "Hard-exudate evidence",
              "soft_exudate": "Soft-exudate evidence"}
    positions = [(1, 0), (1, 1), (1, 2), (0, 2)]
    for (r, c), lt in zip(positions, order):
        ax = axes[r, c]
        ax.imshow(rgb)
        d = results.get(lt, {})
        for cand in d.get("candidates", [])[:30]:
            x0, y0, x1, y1 = cand["bbox"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, color="yellow", lw=0.8))
        ax.set_title(f"{titles[lt]}: {d.get('status', '?')} "
                     f"(n={d.get('candidate_count', 0)})", fontsize=8)
        ax.axis("off")
    for ax in axes.flat:
        ax.axis("off")
    lines = []
    for lt in order:
        d = results.get(lt, {})
        lines.append(f"{lt}: {d.get('status')} n={d.get('candidate_count')} "
                     f"conf={d.get('confidence')}")
    fig.text(0.55, 0.30, "\n".join(lines), fontsize=8, family="monospace",
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)
