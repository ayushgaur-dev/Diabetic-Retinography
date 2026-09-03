"""Multi-scale top-hat vessel enhancement (SIH26038 Phase 4A).

Choice rationale: vessels are thin dark ridges on a slowly varying
background. White top-hat (image − opening) on the INVERTED green channel
extracts bright thin structures at each disk scale; the max-projection
over scales responds to both thin and wide vessels without any learned
weights. Interpretable, dependency-free (cv2+numpy only), reproducible.

Output is a float32 response map in 0..1 — called a RESPONSE, never a
probability.
"""

import cv2
import numpy as np


def enhance_vessels(green01, fov_mask, scales=(3, 5, 9, 15)):
    """green01: float32 0..1 (from preprocessing). Returns response 0..1."""
    if green01.dtype != np.float32 or green01.min() < 0 or green01.max() > 1:
        raise ValueError("enhance_vessels expects float32 green in 0..1.")
    inverted = (1.0 - green01).astype(np.float32)
    response = np.zeros_like(inverted)
    for s in scales:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(s), int(s)))
        tophat = cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, k)
        np.maximum(response, tophat, out=response)
    response[fov_mask == 0] = 0.0
    mx = float(response.max())
    if mx > 0:
        response /= mx
    return response.astype(np.float32)
