"""Fovea visualisation: fundus + FOV + disc + candidates + selection (4C)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def save_fovea_figure(rgb, result, path, disc_xy=None, search_corners=None):
    d = result if isinstance(result, dict) else result.to_dict()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Original fundus")
    axes[1].imshow(rgb)
    if disc_xy is not None:
        axes[1].plot(disc_xy[0], disc_xy[1], "co", ms=8, label="optic disc")
    for c in d.get("candidates", [])[:60]:
        x, y = c["x_y"]
        axes[1].plot(x, y, "y.", ms=3)
    axes[1].set_title(f"Disc + {d.get('candidate_count', 0)} candidates")
    axes[2].imshow(rgb)
    if disc_xy is not None:
        axes[2].plot(disc_xy[0], disc_xy[1], "co", ms=8)
    if d.get("center_x_y") is not None:
        fx, fy = d["center_x_y"]
        axes[2].plot(fx, fy, "r+", ms=14, mew=2)
        if d.get("estimated_region") is not None:
            x0, y0, x1, y1 = d["estimated_region"]
            axes[2].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                        fill=False, color="red", lw=1.2))
    axes[2].set_title(f"FOVEA: {d.get('status')} (conf {d.get('confidence', 0):.2f}, "
                      f"{d.get('laterality', '?')} eye)")
    for ax in axes:
        ax.axis("off")
    lines = [f"status={d.get('status')} laterality={d.get('laterality')} "
             f"disc_used={d.get('disc_used')}"]
    for c in d.get("candidates", [])[:4]:
        lines.append(f"- cand score={c['score']:.2f} dark={c['local_darkness_score']:.2f} "
                     f"spars={c['vessel_sparsity_score']:.2f} geo={c['temporal_geometry_score']:.2f}")
    for w in d.get("warnings", [])[:3]:
        lines.append(f"! {w[:100]}")
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=7, family="monospace",
             verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    fig.tight_layout(rect=[0, 0.28, 1, 1])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
