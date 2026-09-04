"""Offline triage card (SIH26038 Phase 8). Debugging visualization, not UI."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_triage_card(rgb, triage_dict, path, quality_status="?"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title("Fundus image")
    axes[0].axis("off")
    axes[1].axis("off")
    lines = [
        f"Grade: {triage_dict.get('predicted_grade')} | "
        f"calibrated confidence {triage_dict.get('calibrated_confidence')}",
        f"TRIAGE: {triage_dict.get('decision')} [{triage_dict.get('priority')}]",
        f"Referable: {triage_dict.get('referable')}",
        f"Quality: {quality_status}",
        "Reasons:",
    ]
    lines += [f"- {r}" for r in triage_dict.get("reason_codes", [])]
    flags = triage_dict.get("evidence_summary", {}).get("lesion_flags", [])
    if flags:
        lines += ["Evidence:"] + [f"- {f}" for f in flags]
    lines += [""]
    lines += (triage_dict.get("explanation") or "").split("\n")[:14]
    axes[1].text(0.02, 0.98, "\n".join(lines), fontsize=7, family="monospace",
                 verticalalignment="top", transform=axes[1].transAxes,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.95))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)
