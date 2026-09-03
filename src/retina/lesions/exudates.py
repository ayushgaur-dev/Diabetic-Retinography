"""Bright-lesion evidence: hard + soft exudates (SIH26038 Phase 4D).

Hard exudates: bright YELLOW compact lesions — top luminance percentile +
high saturation + bounded red-blue gap + shape filter. Soft exudates
(cotton-wool): bright PALE lesions — top luminance + LOW saturation +
larger, fluffy (low circularity tolerated) + local texture.

Both detectors take a disc_exclusion_mask (dilated 4B circle, None-safe):
the optic disc is bright and would otherwise dominate. Fovea/disc
distances are recorded as context features, never hard rules.
"""

import cv2
import numpy as np

from src.retina.lesions.microaneurysm import _contour
from src.retina.lesions.postprocessing import circularity, components


def _hsv(work_rgb):
    return cv2.cvtColor(work_rgb, cv2.COLOR_RGB2HSV)


def _luminance(work_rgb):
    f = work_rgb.astype(np.float32)
    return 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]


def _base_bright(work_rgb, fov, percentile):
    lum = _luminance(work_rgb)
    inside = lum[fov > 0]
    if inside.size == 0:
        return np.zeros(fov.shape, np.uint8), lum
    thresh = float(np.percentile(inside, percentile))
    return (((lum >= thresh) & (fov > 0)).astype(np.uint8)) * 255, lum


def detect_hard_exudates(work_rgb, fov, disc_excl, disc_xy, fovea_xy,
                         cfg, common, width):
    from src.retina.lesions.config import scaled_area

    c = cfg["hard_exudate"]
    f = work_rgb.astype(np.float32)
    lum = _luminance(work_rgb)
    # Measured on tuning data: saturation does NOT separate yellow exudates
    # from reddish background (both highly saturated); yellowness
    # min(R,G)-B does. The old red-blue-gap upper bound was backwards and
    # is removed (kept as a reported feature only).
    yellow = np.minimum(f[:, :, 0], f[:, :, 1]) - f[:, :, 2]
    inside = lum[fov > 0]
    if inside.size == 0:
        return np.zeros(fov.shape, np.uint8), []
    lthresh = float(np.percentile(inside, c["brightness_percentile"]))
    keep = ((lum >= lthresh) & (yellow >= c["yellow_threshold"])
            & (fov > 0)).astype(np.uint8) * 255
    if disc_excl is not None:
        keep[disc_excl > 0] = 0
    hsv = _hsv(work_rgb)
    sat = hsv[:, :, 1].astype(np.float32)
    return _score_bright(keep, lum, yellow, work_rgb, fov, disc_xy, fovea_xy,
                         c, common, width, "hard_exudate")


def detect_soft_exudates(work_rgb, fov, disc_excl, disc_xy, fovea_xy,
                         cfg, common, width):
    from src.retina.lesions.config import scaled_area  # noqa: F401 (symmetry)

    c = cfg["soft_exudate"]
    raw, lum = _base_bright(work_rgb, fov, c["brightness_percentile"])
    hsv = _hsv(work_rgb)
    sat = hsv[:, :, 1].astype(np.float32)
    gray_u8 = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray_u8, cv2.CV_32F)  # uint8 in (cv2 5 requires it)
    tex = cv2.GaussianBlur(lap * lap, (5, 5), 0)
    tex_in = tex[fov > 0]
    # 'Fluffy' cotton-wool has higher local texture than flat pale
    # background (e.g. peripapillary sheen): gate at FOV median.
    tex_thresh = float(np.median(tex_in)) if tex_in.size else 0.0
    keep = ((sat <= c["max_saturation"]) & (raw > 0) & (tex >= tex_thresh)
            ).astype(np.uint8) * 255
    if disc_excl is not None:
        keep[disc_excl > 0] = 0
    paleness = 255.0 - sat  # high values = pale/white regions
    return _score_bright(keep, lum, paleness, work_rgb, fov, disc_xy, fovea_xy,
                         c, common, width, "soft_exudate", color_high=True,
                         texture=tex)


def _score_bright(keep, lum, color_map, work_rgb, fov, disc_xy, fovea_xy,
                  c, common, width, ltype, color_high=True, texture=None):
    from src.retina.lesions.config import scaled_area

    ref = common["reference_width"]
    lo, hi = scaled_area(c["min_area"], width, ref), scaled_area(c["max_area"], width, ref)
    lum_in = lum[fov > 0]
    lmax = float(lum_in.max()) if lum_in.size else 1.0
    col_in = color_map[fov > 0]
    cmax = float(col_in.max()) if col_in.size else 1.0
    out = []
    evidence = np.zeros(fov.shape, np.uint8)
    for comp in components(keep):
        area = comp["area"]
        if not (lo <= area <= hi):
            continue
        cnt = _contour(comp["mask"])
        circ = circularity(area, cv2.arcLength(cnt, True)) if cnt is not None else 0.0
        if circ < c["min_circularity"]:
            continue
        m = comp["mask"]
        bright = float(lum[m].mean()) / max(lmax, 1e-6)
        col = float(color_map[m].mean()) / max(cmax, 1e-6)
        color = col if color_high else 1.0 - col
        hsv = _hsv(work_rgb)
        smean = float(hsv[:, :, 1][m].mean()) / 255.0
        w = c["weights"]
        score = w["brightness"] * bright + w["color"] * color + w["shape"] * circ
        cx, cy = comp["centroid"]
        feats = {"circularity": float(circ), "mean_saturation": float(smean),
                 "mean_luminance": float(lum[m].mean()),
                 "mean_yellowness": float(color_map[m].mean()),
                 "distance_to_disc": _dist((cx, cy), disc_xy, width),
                 "distance_to_fovea": _dist((cx, cy), fovea_xy, width)}
        if texture is not None:
            feats["mean_texture"] = float(texture[m].mean())
        out.append({"bbox": comp["bbox"], "centroid": comp["centroid"],
                    "area": area, "score": float(score), "features": feats})
        evidence[m] = 255
    out.sort(key=lambda k: k["score"], reverse=True)
    return evidence, out


def _dist(p, q, width):
    if q is None:
        return -1.0
    return float(np.hypot(p[0] - q[0], p[1] - q[1])) / max(width, 1)
