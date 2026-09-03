"""Vessel pipeline: single entry point for future phases (SIH26038 Phase 4A).

segment_vessels(rgb, fov_mask=None, config=None) -> VesselSegmentationResult.
If no FOV mask is supplied, the Phase 2 field-of-view detector is used as a
CLEARLY LABELLED fallback (fov_source='phase2_fallback') — never silently.
"""

import time

import numpy as np
from PIL import Image

from src.retina.vessels import enhancement as enh
from src.retina.vessels import preprocessing as pre
from src.retina.vessels import segmentation as seg
from src.retina.vessels.config import load_vessel_config
from src.retina.vessels.types import VesselSegmentationResult


def load_fov_mask(path, shape):
    m = np.array(Image.open(path).convert("L"))
    if m.shape != shape:
        raise ValueError(f"FOV mask shape {m.shape} != image shape {shape}.")
    return ((m > 0).astype(np.uint8)) * 255


def _fallback_fov(rgb, qcfg):
    from src.quality.field_of_view import detect_retinal_field

    info = detect_retinal_field(rgb, qcfg)
    return info["mask"], info


def segment_vessels(rgb, fov_mask=None, config=None, quality_config=None):
    t0 = time.perf_counter()
    pre._check_rgb(rgb)
    warnings = []
    if fov_mask is None:
        from src.quality.config import load_config as load_qconfig

        qcfg = quality_config or load_qconfig()
        fov_mask, info = _fallback_fov(rgb, qcfg)
        warnings.append("No FOV mask supplied; used Phase 2 detector fallback.")
        if not info["plausible"]:
            warnings.append(f"Fallback FOV detector uncertain: {info['reason']}.")
        fov_source = "phase2_fallback"
    else:
        fov_mask = pre._check_mask(fov_mask, rgb.shape[:2])
        fov_source = "provided"

    cfg = config or load_vessel_config()
    p = cfg["preprocessing"]
    green = pre.preprocess_green(rgb, fov_mask, p["stretch_low_percentile"],
                                 p["stretch_high_percentile"])
    response = enh.enhance_vessels(green, fov_mask, tuple(cfg["enhancement"]["scales"]))
    s = cfg["segmentation"]
    mask = seg.segment_response(response, fov_mask, s["method"],
                                s.get("percentile", 89.0), s.get("fixed_threshold"))
    pp = cfg["postprocessing"]
    mask = seg.postprocess_mask(mask, fov_mask, pp["min_component_size"],
                                pp["closing_kernel"], pp["apply_closing"])
    fov_px = int((fov_mask > 0).sum())
    area = int((mask > 0).sum())
    density = area / fov_px if fov_px else 0.0
    return VesselSegmentationResult(
        vessel_mask=mask, vessel_response=response, fov_mask=fov_mask,
        vessel_density=density, vessel_area=area,
        processing_time_ms=(time.perf_counter() - t0) * 1000.0,
        method=cfg.get("method", "multiscale_tophat_classical"),
        fov_source=fov_source, warnings=warnings,
    )
