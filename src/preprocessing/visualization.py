"""Before/after enhancement visualisation (SIH26038 Phase 3).

Honest figure only: original + enhanced side by side, quality verdicts,
operations applied. Wording rule: 'improved according to configured
engineering quality metrics' — never 'image fixed'.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_enhancement_figure(original, result, path):
    """Save a before/after PNG. Returns the path as a string."""
    enhanced = result.enhanced_image if result.enhanced_image is not None else original
    bq, aq = result.before_quality, result.after_quality
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(original)
    axes[0].set_title(f"ORIGINAL — {bq.get('status', '?')} "
                      f"({bq.get('overall_score', 0):.2f})")
    axes[0].axis("off")
    title = "ENHANCED" if result.enhancement_successful else "ENHANCED (discarded)"
    if aq:
        title += f" — {aq.get('status', '?')} ({aq.get('overall_score', 0):.2f})"
    axes[1].imshow(enhanced)
    axes[1].set_title(title)
    axes[1].axis("off")
    lines = ["OPERATIONS APPLIED:"] + [f"- {op}" for op in result.operations_applied]
    if result.changed_quality_dimensions:
        lines += ["", "Improved dimensions:"] + [f"- {d}" for d in result.changed_quality_dimensions]
    if result.warnings:
        lines += ["", "Notes:"] + [f"- {w}" for w in result.warnings]
    lines += ["", "Image quality improved according to configured engineering",
              "quality metrics." if result.enhancement_successful else
              "quality metrics — original retained."]
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=8, family="monospace",
             verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    fig.tight_layout(rect=[0, 0.32, 1, 1])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
