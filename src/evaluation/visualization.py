"""Evaluation figures (SIH26038 Phase 5). Each plot answers one question."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_confusion(cm, path, title, labels=("0", "1", "2", "3", "4")):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(np.asarray(cm), cmap="Blues")
    ax.set_xticks(range(5), labels)
    ax.set_yticks(range(5), labels)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_title(title)
    for i in range(5):
        for j in range(5):
            v = np.asarray(cm)[i, j]
            ax.text(j, i, f"{v:.2f}" if isinstance(v, float) else f"{v}",
                    ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def save_roc(fpr, tpr, auroc, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR (1-specificity)")
    ax.set_ylabel("TPR (sensitivity)")
    ax.set_title("Referable-DR ROC (score P2+P3+P4)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def save_pr(recall, precision, ap, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(recall, precision, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Referable-DR precision-recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def save_confidence_hist(hist, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(10), hist)
    ax.set_xlabel("Raw max-softmax decile")
    ax.set_ylabel("Images")
    ax.set_title("Raw confidence distribution (UNCALIBRATED)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def save_threshold_curve(grid_rows, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    t = [r["threshold"] for r in grid_rows]
    ax.plot(t, [r["sensitivity"] for r in grid_rows], "o-", label="sensitivity")
    ax.plot(t, [r["specificity"] for r in grid_rows], "s-", label="specificity")
    ax.plot(t, [r["f1"] for r in grid_rows], "^-", label="F1")
    ax.set_xlabel("Referable threshold on P2+P3+P4")
    ax.set_ylabel("Metric")
    ax.set_title("Threshold trade-off (validation split)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
