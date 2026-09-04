"""Overlay primitives: one visual channel per evidence kind (SIH26038 Phase 6).
Grad-CAM -> heat colormap | vessels -> cyan lines | disc -> green circle |
fovea -> magenta marker | lesions -> yellow boxes. Never fully obscures fundus.
"""

import cv2
import numpy as np


def heat_overlay(rgb, heat_hires, alpha=0.45):
    h = np.asarray(heat_hires, dtype=float)
    h = (h - h.min()) / (h.max() - h.min()) if h.max() > h.min() else h * 0
    colored = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(np.asarray(rgb), 1 - alpha, colored, alpha, 0)


def draw_vessels(base, vessel_mask, color=(0, 255, 255)):
    out = np.asarray(base).copy()
    out[np.asarray(vessel_mask) > 0] = color
    return out


def draw_disc(base, center_xy, radius, color=(0, 255, 0)):
    out = np.asarray(base).copy()
    if center_xy is None:
        return out
    cv2.circle(out, (int(center_xy[0]), int(center_xy[1])), int(radius), color, 2)
    cv2.drawMarker(out, (int(center_xy[0]), int(center_xy[1])), color,
                   cv2.MARKER_TILTED_CROSS, 16, 2)
    return out


def draw_fovea(base, center_xy, color=(255, 0, 255)):
    out = np.asarray(base).copy()
    if center_xy is None:
        return out
    cv2.drawMarker(out, (int(center_xy[0]), int(center_xy[1])), color,
                   cv2.MARKER_CROSS, 20, 2)
    return out


def draw_lesions(base, lesion_dicts, max_boxes=30):
    out = np.asarray(base).copy()
    for ltype, d in (lesion_dicts or {}).items():
        for c in d.get("candidates", [])[:max_boxes]:
            x0, y0, x1, y1 = (int(v) for v in c["bbox"])
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 0), 1)
    return out


def unified_overlay(rgb, heat, vessel_mask, disc, fovea_d, lesion_dicts):
    out = heat_overlay(rgb, heat, alpha=0.35)
    out = draw_vessels(out, vessel_mask)
    if disc and disc.get("center") is not None:
        out = draw_disc(out, disc["center"], disc["radius"])
    if fovea_d and fovea_d.get("center_x_y") is not None:
        out = draw_fovea(out, fovea_d["center_x_y"])
    return draw_lesions(out, lesion_dicts)
