"""Microaneurysm evidence: small dark round blobs (SIH26038 Phase 4D).

Pipeline: normalized green -> invert -> multi-scale white top-hat (small
scales) -> FOV percentile threshold -> size/circularity filter ->
vessel-overlap veto -> scored candidates. Vessels and hemorrhages mimic
MA responses, so shape + isolation features carry real weight.
"""

import cv2
import numpy as np

from src.retina.lesions.postprocessing import circularity, components


def detect_microaneurysms(work_rgb, fov, vessel, dist_vessel, green_norm,
                          cfg, common, width):
    c = cfg["microaneurysm"]
    inv = (1.0 - green_norm).astype(np.float32)
    resp = np.zeros_like(inv)
    for s in c["tophat_scales"]:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(s), int(s)))
        np.maximum(resp, cv2.morphologyEx(inv, cv2.MORPH_TOPHAT, k), out=resp)
    resp[fov == 0] = 0.0
    inside = resp[fov > 0]
    if inside.size == 0 or inside.max() <= 0:
        return np.zeros(fov.shape, np.uint8), []
    thresh = float(np.percentile(inside, c["response_percentile"]))
    raw = (((resp > thresh) & (fov > 0)).astype(np.uint8)) * 255
    return _score_components(raw, resp, work_rgb, fov, vessel, dist_vessel,
                             c, common, width, "microaneurysm")


def _score_components(raw, resp, work_rgb, fov, vessel, dist_vessel,
                      c, common, width, ltype):
    from src.retina.lesions.config import scaled_area

    ref = common["reference_width"]
    lo, hi = scaled_area(c["min_area"], width, ref), scaled_area(c["max_area"], width, ref)
    rmax = float(resp.max())
    green = work_rgb[:, :, 1].astype(np.float32)
    out_cands = []
    kept = np.zeros(fov.shape, np.uint8)
    for comp in components(raw):
        area = comp["area"]
        if not (lo <= area <= hi):
            continue
        cnt = _contour(comp["mask"])
        circ = circularity(area, cv2.arcLength(cnt, True)) if cnt is not None else 0.0
        if circ < c["min_circularity"]:
            continue
        m = comp["mask"]
        overlap = (float((m & (vessel > 0)).sum()) / area) if vessel is not None else 0.0
        if overlap > c["max_vessel_overlap"]:
            continue
        contrast = float(resp[m].mean()) / rmax if rmax > 0 else 0.0
        isolation = 1.0 - overlap
        w = c["weights"]
        score = w["contrast"] * contrast + w["shape"] * circ + w["isolation"] * isolation
        kept[m] = 255
        out_cands.append({
            "bbox": comp["bbox"], "centroid": comp["centroid"], "area": area,
            "score": float(score),
            "features": {
                "equivalent_diameter": float(2 * np.sqrt(area / np.pi)),
                "circularity": float(circ),
                "mean_green": float(green[m].mean()),
                "response_contrast": float(contrast),
                "vessel_overlap": float(overlap),
                "distance_to_vessel": (float(dist_vessel[int(comp["centroid"][1]),
                                                           int(comp["centroid"][0])])
                                       if dist_vessel is not None else -1.0),
            },
        })
    out_cands.sort(key=lambda k: k["score"], reverse=True)
    return kept, out_cands


def _contour(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None
