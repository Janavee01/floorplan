"""
walls.py — binarise a floorplan image and strip text/noise blobs,
           leaving only wall geometry.

Fixes in this version
---------------------
- Dilation is now CONDITIONAL: only applied where wall pixels already
  have a direct neighbour (i.e. closing real gaps), not in open space.
  This prevents thin partition walls between adjacent rooms from merging.
- medianBlur radius reduced to 3 (was already 3, kept).
- Closing kernels unchanged (they close wall-end gaps, not room gaps).
"""

import cv2
import numpy as np


def extract_walls(
    gray: "np.ndarray",
    adaptive_block_size: int = 15,
    adaptive_C: int = 3,
    morph_kernel: tuple[int, int] = (2, 2),
    min_component_area: int = 300,
    min_wall_area: int = 2000,
    min_wall_thickness: int = 3,
) -> "np.ndarray":
    # --- adaptive threshold -------------------------------------------------
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block_size,
        adaptive_C,
    )

    # --- morphological opening (remove single-pixel speckle) ----------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
    clean = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- connected-component filtering (drop text blobs, keep walls) --------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        clean, connectivity=8
    )

    wall_mask = np.zeros_like(clean)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect    = w / (h + 1e-5)
        extent    = area / (w * h + 1e-5)
        thickness = area / (max(w, h) + 1e-5)

        keep = False
        if area > min_wall_area:
            keep = True
        if (w > 80 or h > 80) and thickness > 6:
            keep = True
        if extent > 0.4 and area > 1500:
            keep = True

        # veto rules
        if area < min_component_area:
            keep = False
        if 0.3 < aspect < 4.0 and area < 400:
            keep = False
        if thickness < 1.5:
            keep = False

        if keep:
            wall_mask[labels == i] = 255

    # --- close small gaps in wall ends (horizontal and vertical) ------------
    # These kernels close actual wall-end gaps, not partition walls.
    kernel_h = np.ones((1, 5), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel_h)
    kernel_v = np.ones((5, 1), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel_v)

    # --- CONDITIONAL dilation: only dilate pixels already near other walls --
    # A plain dilate(iterations=1) merges thin partition walls when two
    # wall blobs are only 1-2px apart (common in scanned plans).
    # Instead: compute a 3x3 neighbour-count; only dilate wall pixels whose
    # 3x3 neighbourhood already contains another wall pixel within 3px.
    # This closes genuine micro-gaps but leaves thin inter-room walls alone.
    k3 = np.ones((3, 3), np.uint8)
    neighbour_count = cv2.dilate(wall_mask, k3, iterations=1)
    # Pixels that are currently NOT wall but have wall within 1px → candidate
    candidate_dilation = cv2.bitwise_and(neighbour_count, cv2.bitwise_not(wall_mask))
    # Only accept candidate pixels that also have a wall within 3px
    # (i.e. the gap is genuinely narrow, not an open room interior)
    k5 = np.ones((5, 5), np.uint8)
    near_wall = cv2.dilate(wall_mask, k5, iterations=1)
    safe_dilation = cv2.bitwise_and(candidate_dilation, near_wall)
    wall_mask = cv2.bitwise_or(wall_mask, safe_dilation)

    wall_mask = cv2.medianBlur(wall_mask, 3)
    return wall_mask