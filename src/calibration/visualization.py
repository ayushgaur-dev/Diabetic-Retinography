"""Reliability diagrams (SIH26038 Phase 7). Same bins raw vs calibrated,
perfect-calibration diagonal, bin counts annotated."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_reliability(bins, title, path, xlabel="Mean confidence"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6),
                                   gridspec_kw={"height_ratios": [3, 1]})
    xs = [(b["bin"][0] + b["bin"][1]) / 2 for b in bins]
    acc = [b["accuracy"] if b["accuracy"] is not None else float("nan") for b in bins]
    conf = [b["mean_confidence"] if b["mean_confidence"] is not None else float("nan")
            for b in bins]
    counts = [b["count"] for b in bins]
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax1.plot(conf, acc, "o-", label="observed")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Empirical accuracy")
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax2.bar(xs, counts, width=0.08)
    ax2.set_xlabel("Confidence bin")
    ax2.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def save_confidence_shift(raw_conf, cal_conf, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(raw_conf, bins=20, range=(0, 1), alpha=0.6, label="raw")
    ax.hist(cal_conf, bins=20, range=(0, 1), alpha=0.6, label="calibrated")
    ax.set_xlabel("Confidence (max probability)")
    ax.set_ylabel("Images")
    ax.set_title("Confidence distribution shift (descriptive)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
