"""
segment.py — segment free space in a wall mask into individual rooms
             using distance-transform + connected components.
"""

import cv2
import numpy as np


def segment_rooms(
    wall_mask: "np.ndarray",
    corridor_thresh: int = 12,
    min_room_area: int = 1500,
    morph_kernel: tuple[int, int] = (5, 5),
) -> tuple[list, "np.ndarray", "np.ndarray"]:
    """
    Identify room regions from a binary wall mask.

    Strategy
    --------
    1. Invert the wall mask to get free space.
    2. Run a distance transform — pixels far from walls get high values.
    3. Threshold at *corridor_thresh* to cut narrow corridors / doorways.
    4. Connected components on the remaining blobs = individual rooms.

    Parameters
    ----------
    wall_mask        : binary image, walls = 255, free = 0
    corridor_thresh  : distance threshold (px) to split corridors from rooms
    min_room_area    : drop blobs smaller than this (px²)
    morph_kernel     : opening kernel applied after thresholding

    Returns
    -------
    rooms      : list of uint8 masks, one per room (255 inside, 0 outside)
    room_core  : debug — the thresholded core image
    dist       : debug — raw distance-transform output
    """
    _, bin_img = cv2.threshold(wall_mask, 127, 255, cv2.THRESH_BINARY)
    free = 255 - bin_img

    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)

    corridor_mask = dist < corridor_thresh
    room_core = free.copy()
    room_core[corridor_mask] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
    room_core = cv2.morphologyEx(room_core, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels = cv2.connectedComponents(room_core)

    rooms = []
    for i in range(1, num_labels):
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == i] = 255
        if cv2.countNonZero(mask) >= min_room_area:
            rooms.append(mask)

    return rooms, room_core, dist


def extract_room_features(room_masks: list) -> list[dict]:
    """
    Compute geometric features for each room mask.

    Returns
    -------
    List of dicts with: id, area, centroid, bbox, aspect_ratio,
                        compactness, rectangularity, contour, mask
    """
    rooms_data = []

    for i, mask in enumerate(room_masks):
        area = cv2.countNonZero(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(cnt)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        perimeter    = cv2.arcLength(cnt, True)
        compactness  = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        rectangularity = area / (w * h + 1e-6)

        rooms_data.append({
            "id":            i,
            "area":          area,
            "centroid":      (cx, cy),
            "bbox":          (x, y, w, h),
            "aspect_ratio":  w / (h + 1e-5),
            "compactness":   compactness,
            "rectangularity": rectangularity,
            "contour":       cnt,
            "mask":          mask,
        })

    return rooms_data
