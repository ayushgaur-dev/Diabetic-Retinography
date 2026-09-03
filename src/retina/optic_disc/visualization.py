"""Disc visualisation: original | FOV+candidates | selected region (SIH26038 4B)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def save_disc_figure(rgb, result, path, fov_mask=None):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    axes[0].imshow(rgb)
    axes[0].set_title("Original fundus")
    axes[1].imshow(rgb)
    if fov_mask is not None:
        axes[1].contour(fov_mask, colors="cyan", linewidths=0.8)
    for c in result.get("candidates", []):
        x, y = c["centroid"]
        axes[1].plot(x, y, "yo", ms=4)
    axes[1].set_title(f"FOV + {result.get('candidate_count', 0)} candidates")
    axes[2].imshow(rgb)
    if result.get("center") is not None:
        (cx, cy), r = result["center"], result["radius"]
        x0, y0, x1, y1 = result["bounding_box"]
        axes[2].add_patch(Circle((cx, cy), r, fill=False, color="lime", lw=1.5))
        axes[2].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    fill=False, color="lime", lw=1.0, ls="--"))
        axes[2].plot(cx, cy, "r+", ms=10, mew=1.5)
    axes[2].set_title(f"OPTIC DISC: {result.get('status')} "
                      f"(conf {result.get('confidence', 0):.2f})")
    for ax in axes:
        ax.axis("off")
    lines = [f"status={result.get('status')} detected={result.get('detected')}"]
    for c in result.get("candidates", [])[:5]:
        lines.append(f"- cand score={c['score']:.2f} "
                     f"circ={c['circularity']:.2f} conv={c['convergence']:.2f}")
    for w in result.get("warnings", [])[:3]:
        lines.append(f"! {w[:100]}")
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=7, family="monospace",
             verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    fig.tight_layout(rect=[0, 0.25, 1, 1])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
