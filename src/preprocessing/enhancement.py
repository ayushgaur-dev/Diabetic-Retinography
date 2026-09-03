"""Adaptive operation selection + application (SIH26038 Phase 3).

The Phase 2 QualityResult decides WHAT runs — never a fixed chain.
Mapping (documented, configurable via selection.*):
  contrast BORDERLINE     -> clahe + denoise (denoise counters CLAHE noise gain)
  illumination BORDERLINE -> illumination_normalization + color_normalization
  exposure BORDERLINE     -> illumination_normalization if dark-dominated;
                             nothing if saturation-dominated (clipped pixels
                             hold no recoverable signal -> warning instead)
  focus BORDERLINE        -> NOTHING sharpened: blur destroys information that
                             enhancement cannot recover (warning recorded).
                             allow_unsharp exists but defaults to False.
  color cast              -> color_normalization when inter-channel spread
                             exceeds trigger (image-based, never label-based).
"""

from src.preprocessing import operations as ops
from src.quality.types import BORDERLINE


def select_operations(quality_result, rgb, mask, cfg):
    """Return (ordered_unique_op_names, warnings). Pure function of the
    quality verdict + image statistics. No DR-grade input exists here."""
    sel = cfg["selection"]
    chosen, warnings = [], []

    q = quality_result.to_dict() if hasattr(quality_result, "to_dict") else quality_result

    if q["contrast"]["status"] == BORDERLINE:
        chosen += list(sel["contrast_operations"])
    if q["illumination"]["status"] == BORDERLINE:
        chosen += list(sel["illumination_operations"])
    if q["exposure"]["status"] == BORDERLINE:
        det = q["exposure"].get("details", {})
        if det.get("black_fraction", 0) >= det.get("white_fraction", 0):
            chosen += list(sel["exposure_dark_operations"])
        else:
            chosen += list(sel["exposure_bright_operations"])
            warnings.append("Bright saturation holds no recoverable signal; "
                            "no exposure operation applied.")
    if q["focus"]["status"] == BORDERLINE:
        if sel.get("allow_unsharp"):
            warnings.append("allow_unsharp is enabled — unsharp path not "
                            "implemented; focus left untouched.")
        else:
            warnings.append("Focus is marginal but blur cannot be reversed by "
                            "enhancement; no sharpening applied.")
    spread = ops.channel_spread(rgb, mask)
    if spread > cfg["color"]["trigger_spread"] and "color_normalization" not in chosen:
        chosen.append("color_normalization")

    seen, unique = set(), []
    for op in chosen:
        if op not in seen:
            seen.add(op)
            unique.append(op)
    return unique, warnings


def apply_operations(rgb, mask, op_names, cfg):
    """Apply ops in order. Returns (enhanced_rgb, actually_applied)."""
    img, applied = rgb, []
    for op in op_names:
        if op == "clahe":
            c = cfg["clahe"]
            img = ops.apply_clahe(img, mask, c["clip_limit"], tuple(c["tile_grid"]))
        elif op == "illumination_normalization":
            c = cfg["illumination"]
            img = ops.normalize_illumination(img, mask, c["median_kernel"],
                                             c["gain_cap_lo"], c["gain_cap_hi"])
        elif op == "denoise":
            c = cfg["denoise"]
            img = ops.denoise(img, mask, c["diameter"], c["sigma_color"],
                              c["sigma_space"])
        elif op == "color_normalization":
            img = ops.normalize_color(img, mask, cfg["color"]["max_gain"])
        else:
            continue
        applied.append(op)
    return img, applied
