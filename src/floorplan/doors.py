"""
doors.py — detect and REMOVE door arcs from the wall mask before segmentation.

Architectural floorplans draw doors as:
  - a straight line (the door itself)
  - a quarter-circle arc (the swing radius)

Strategy
--------
1. Detect arc circles via Hough transform
2. Fill each detected arc region with wall pixels (closiīng the room boundary)
3. Return the patched wall mask for segmentation, plus a clearance map
   for furniture placement
"""

import cv2
import numpy as np


def detect_and_remove_doors(
    wall_mask: "np.ndarray",
    min_radius: int = None,
    max_radius: int = None,
    hough_param2: int = 18,
) -> tuple:
    """
    Detect door arcs in wall_mask and fix gaps.
    """

    h, w = wall_mask.shape

    # ✅ NEW: scale radii based on image size
    scale = min(h, w) / 800  # 800px reference

    if min_radius is None:
        min_radius = int(20 * scale)

    if max_radius is None:
        max_radius = int(120 * scale)  # bigger for hi-res

    door_mask = np.zeros_like(wall_mask)
    clean_walls = wall_mask.copy()

    circles = cv2.HoughCircles(
        wall_mask,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_radius * 2,
        param1=50,
        param2=hough_param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))

        for (cx, cy, r) in circles:
            # mark door area
            cv2.circle(door_mask, (cx, cy), r, 255, -1)

            # close wall gap (ONLY outline)
            cv2.circle(clean_walls, (cx, cy), r, 255, 3)

    return clean_walls, door_mask

def build_clearance_map(
    door_mask: "np.ndarray",
    clearance_px: int = 80,
) -> "np.ndarray":
    """
    Dilate the door mask to create a zone furniture must stay out of.

    Parameters
    ----------
    door_mask     : output of detect_and_remove_doors()
    clearance_px  : keep-clear radius around each door in pixels

    Returns
    -------
    clearance_map : uint8 binary mask (255 = no furniture here)
    """
    if not np.any(door_mask):
        return np.zeros_like(door_mask)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (clearance_px * 2, clearance_px * 2)
    )
    return cv2.dilate(door_mask, kernel, iterations=1)