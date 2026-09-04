"""Evidence fusion: regions + spatial statistics (SIH26038 Phase 6).

DESCRIPTIVE ONLY — fusion never alters predicted_grade/probabilities
(asserted by test). Lesion candidates become EvidenceRegions; anatomy
(disc/fovea/vessels) becomes context regions; Grad-CAM hot components
become attention regions. Overlaps are 'spatial agreement', never causal.
"""

import cv2
import numpy as np

from src.explainability.evidence_map import hot_mask
from src.explainability.types import EvidenceRegion


def _components(mask):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ((np.asarray(mask) > 0).astype(np.uint8)) * 255, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(stats[i, c]) for c in
                            (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                             cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA))
        out.append({"bbox": [x, y, x + w, y + h],
                    "centroid": (float(centroids[i][0]), float(centroids[i][1])),
                    "area": float(area), "mask": (labels == i)})
    return out


def _overlap_frac(region_mask, hot):
    region = region_mask > 0
    area = int(region.sum())
    if area == 0:
        return 0.0
    return float(((region) & (hot > 0)).sum()) / area


def build_regions(lesion_dicts, width, hot, disc_xy=None, fovea_xy=None):
    """lesion_dicts: lesion_type -> LesionResult.to_dict() with ORIGINAL-frame
    bboxes. Returns ranked-ready list[EvidenceRegion] (ranking in summary)."""
    regions = []
    for ltype, d in (lesion_dicts or {}).items():
        for c in d.get("candidates", [])[:50]:
            x0, y0, x1, y1 = (int(v) for v in c["bbox"])
            cx, cy = float(c["centroid_x_y"][0]), float(c["centroid_x_y"][1])
            tmp = np.zeros(hot.shape, np.uint8)
            tmp[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 255
            regions.append(EvidenceRegion(
                source=ltype, evidence_type="lesion_candidate",
                bbox=[x0, y0, x1, y1], centroid=[cx, cy],
                area=c["area"], score=c["score"],
                overlap_with_gradcam=_overlap_frac(tmp, hot),
                distance_to_fovea=(float(np.hypot(cx - fovea_xy[0], cy - fovea_xy[1])) / width
                                   if fovea_xy else -1.0),
                distance_to_optic_disc=(float(np.hypot(cx - disc_xy[0], cy - disc_xy[1])) / width
                                        if disc_xy else -1.0),
                features={"candidate_id": c["candidate_id"]}))
    return regions


def anatomy_regions(disc_d, fovea_d, vessel_mask, hot, width):
    regions = []
    if disc_d and disc_d.get("center") is not None:
        (cx, cy), r = disc_d["center"], disc_d["radius"]
        tmp = np.zeros(hot.shape, np.uint8)
        yy, xx = np.mgrid[0:hot.shape[0], 0:hot.shape[1]]
        tmp[((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2] = 255
        regions.append(EvidenceRegion(
            source="optic_disc", evidence_type="anatomy",
            bbox=[cx - r, cy - r, cx + r, cy + r], centroid=[cx, cy],
            area=float(np.pi * r ** 2), score=disc_d.get("confidence", 0),
            overlap_with_gradcam=_overlap_frac(tmp, hot)))
    if fovea_d and fovea_d.get("center_x_y") is not None:
        cx, cy = fovea_d["center_x_y"]
        regions.append(EvidenceRegion(
            source="fovea", evidence_type="anatomy",
            bbox=[cx - 5, cy - 5, cx + 5, cy + 5], centroid=[cx, cy],
            area=100.0, score=fovea_d.get("confidence", 0),
            overlap_with_gradcam=_overlap_frac(
                _dot(hot.shape, (cx, cy)), hot)))
    if vessel_mask is not None:
        v = (np.asarray(vessel_mask) > 0)
        regions.append(EvidenceRegion(
            source="vessel", evidence_type="vessel_map",
            bbox=[0, 0, hot.shape[1], hot.shape[0]], centroid=[0, 0],
            area=float(v.sum()), score=1.0,
            overlap_with_gradcam=(float((v & (hot > 0)).sum()) / v.sum()
                                  if v.sum() else 0.0)))
    return regions


def _dot(shape, xy, r=5):
    tmp = np.zeros(shape, np.uint8)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    tmp[((xx - xy[0]) ** 2 + (yy - xy[1]) ** 2) <= r ** 2] = 255
    return tmp


def lesion_overlap_stats(lesion_masks, hot, fov):
    """Per-type: lesion_area, overlap area, lesion-inside-hot frac,
    hot-inside-lesion frac, IoU. All FOV-restricted."""
    stats = {}
    hot_f = ((np.asarray(hot) > 0) & (np.asarray(fov) > 0))
    for ltype, mask in (lesion_masks or {}).items():
        les = ((np.asarray(mask) > 0) & (np.asarray(fov) > 0))
        la, ha = int(les.sum()), int(hot_f.sum())
        inter = int((les & hot_f).sum())
        union = int((les | hot_f).sum())
        stats[ltype] = {
            "lesion_area": la,
            "gradcam_overlap_area": inter,
            "lesion_inside_gradcam_fraction": round(inter / la, 4) if la else None,
            "gradcam_inside_lesion_fraction": round(inter / ha, 4) if ha else None,
            "iou": round(inter / union, 4) if union else None,
        }
    return stats


def landmark_activation(heat, disc_mask, fovea_mask, fov):
    """Mean activation inside disc / fovea / elsewhere (FOV-restricted)."""
    h = np.asarray(heat, dtype=float)
    f = np.asarray(fov) > 0
    out = {}
    for name, m in (("optic_disc", disc_mask), ("fovea", fovea_mask)):
        if m is None:
            out[f"mean_inside_{name}"] = None
            continue
        sel = (np.asarray(m) > 0) & f
        out[f"mean_inside_{name}"] = round(float(h[sel].mean()), 4) if sel.any() else None
    rest = f.copy()
    if disc_mask is not None:
        rest = rest & ~(np.asarray(disc_mask) > 0)
    if fovea_mask is not None:
        rest = rest & ~(np.asarray(fovea_mask) > 0)
    out["mean_outside_landmarks"] = round(float(h[rest].mean()), 4) if rest.any() else None
    out["fraction_outside_fov"] = None
    return out


def vessel_activation(heat, vessel_mask, fov):
    h = np.asarray(heat, dtype=float)
    f = np.asarray(fov) > 0
    v = (np.asarray(vessel_mask) > 0) & f if vessel_mask is not None else None
    if v is None or not v.any():
        return {"mean_on_vessels": None, "mean_off_vessels": None}
    return {"mean_on_vessels": round(float(h[v].mean()), 4),
            "mean_off_vessels": round(float(h[f & ~v].mean()), 4) if (f & ~v).any() else None}


def rank_regions(regions, cfg):
    w = cfg["ranking"]["weights"]
    areas = [r.area for r in regions] or [1.0]
    amax = max(areas)
    scored = []
    for r in regions:
        s = (w["gradcam_overlap"] * r.overlap_with_gradcam
             + w["candidate_score"] * r.score
             + w["area"] * (r.area / amax if amax else 0))
        scored.append((s, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in scored]
