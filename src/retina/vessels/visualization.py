"""Vessel visualisation (SIH26038 Phase 4A). Debugging figure, not decoration:
original | green | response | predicted mask | ground truth (if given) |
FP/FN error map (if given)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_vessel_figure(rgb, green, response, pred_mask, path, gt_mask=None,
                       fov_mask=None, title=""):
    ncols = 6 if gt_mask is not None else 4
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.6))
    axes[0].imshow(rgb)
    axes[0].set_title("Original fundus")
    axes[1].imshow(green, cmap="gray")
    axes[1].set_title("Green channel")
    axes[2].imshow(response, cmap="hot")
    axes[2].set_title("Vessel response")
    axes[3].imshow(pred_mask, cmap="gray")
    axes[3].set_title("Predicted vessels")
    if gt_mask is not None:
        axes[4].imshow(gt_mask, cmap="gray")
        axes[4].set_title("Ground truth (1st_manual)")
        p = np.asarray(pred_mask) > 0
        g = np.asarray(gt_mask) > 0
        err = np.zeros((*p.shape, 3), dtype=np.uint8)
        err[p & g] = [255, 255, 255]  # TP white
        err[p & ~g] = [255, 0, 0]      # FP red
        err[~p & g] = [0, 0, 255]      # FN blue
        axes[5].imshow(err)
        axes[5].set_title("TP white / FP red / FN blue")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
