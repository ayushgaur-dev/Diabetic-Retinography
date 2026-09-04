"""Unified 6-panel visualization (SIH26038 Phase 6)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.explainability.localization_overlay import (draw_disc, draw_fovea,
                                                    draw_lesions,
                                                    draw_vessels, heat_overlay)


def save_explainability_figure(rgb, ev, path):
    """ev: ExplainabilityResult.to_dict() + '_maps' with aligned arrays."""
    maps = ev.get("_maps", {})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("1. Original fundus")
    axes[0, 1].imshow(heat_overlay(rgb, maps.get("gradcam",
                                                maps.get("blank"))))
    axes[0, 1].set_title(f"2. Grad-CAM (class {ev.get('explained_class')})")
    fv = draw_vessels(rgb, maps.get("vessels", maps.get("blank")))
    axes[0, 2].imshow(fv)
    axes[0, 2].set_title("3. FOV + vessels")
    da = draw_disc(rgb, (ev.get("optic_disc") or {}).get("center"),
                   ((ev.get("optic_disc") or {}).get("radius") or 0))
    da = draw_fovea(da, (ev.get("fovea") or {}).get("center_x_y"))
    axes[1, 0].imshow(da)
    axes[1, 0].set_title("4. Optic disc + fovea")
    axes[1, 1].imshow(draw_lesions(rgb, ev.get("lesions", {})))
    axes[1, 1].set_title("5. Lesion evidence")
    from src.explainability.localization_overlay import unified_overlay

    axes[1, 2].imshow(unified_overlay(
        rgb, maps.get("gradcam", maps.get("blank")),
        maps.get("vessels", maps.get("blank")),
        ev.get("optic_disc"), ev.get("fovea"), ev.get("lesions", {})))
    axes[1, 2].set_title("6. Unified explanation")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"Grade {ev.get('predicted_grade')} | raw conf "
                 f"{ev.get('raw_confidence')} (uncalibrated) | "
                 f"{ev.get('consistency', {}).get('category')}", fontsize=10)
    fig.text(0.01, 0.01, (ev.get("summary") or "")[:900], fontsize=6,
             family="monospace", verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    fig.tight_layout(rect=[0, 0.18, 1, 0.94])
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)
