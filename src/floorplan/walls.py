"""
walls.py — binarise a floorplan image and strip text/noise blobs,
           leaving only wall geometry.
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
    """
    Convert a grayscale floorplan to a binary wall mask with text removed.

    Parameters
    ----------
    gray                 : grayscale image (uint8)
    adaptive_block_size  : neighbourhood size for adaptive threshold
    adaptive_C           : constant subtracted from the mean
    morph_kernel         : size of opening kernel for noise removal
    min_component_area   : blobs below this area are always dropped
    min_wall_area        : blobs above this area are always kept
    min_wall_thickness   : blobs with mean thickness below this are dropped

    Returns
    -------
    binary wall mask (uint8, 255 = wall, 0 = free)
    """
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
        if 0.3 < aspect < 4.0 and area < 1000:
            keep = False
        if thickness < min_wall_thickness:
            keep = False

        if keep:
            wall_mask[labels == i] = 255

    return wall_mask
