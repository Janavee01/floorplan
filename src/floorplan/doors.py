"""
doors.py — detect and REMOVE door arcs from the wall mask before segmentation.

Architectural floorplans draw doors as:
  - a straight line (the door itself)
  - a quarter-circle arc (the swing radius)

Strategy
--------
1. Detect arc circles via Hough transform
2. Fill each detected arc region with wall pixels (closing the room boundary)
3. Return the patched wall mask for segmentation, plus a clearance map
   for furniture placement
"""

import cv2
import numpy as np


def detect_and_remove_doors(
    wall_mask: "np.ndarray",
    min_radius: int = 15,
    max_radius: int = 60,
    hough_param2: int = 18,
) -> tuple:
    """
    Detect door arcs in *wall_mask*, patch the gaps they create,
    and return both the cleaned wall mask and a door location mask.

    Parameters
    ----------
    wall_mask    : binary wall image (255 = wall, 0 = free)
    min_radius   : min door swing radius in pixels
    max_radius   : max door swing radius in pixels
    hough_param2 : Hough accumulator threshold — lower finds more circles

    Returns
    -------
    clean_walls : wall_mask with door gaps filled in (ready for segmentation)
    door_mask   : binary mask marking where doors were found (for clearance)
    """
    door_mask  = np.zeros_like(wall_mask)
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
            # Mark door location
            cv2.circle(door_mask, (cx, cy), r, 255, -1)

            # Close the gap: draw a filled arc region back as wall
            # This seals the room boundary so segmentation doesn't bleed through
            cv2.circle(clean_walls, (cx, cy), r, 255, 2)

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
