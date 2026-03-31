"""
segment.py — segment free space in a wall mask into individual rooms.

Strategy
--------
1. Compute a distance transform on free space.
2. Derive a scale-aware corridor threshold from the image itself
   (median of local maxima) rather than a fixed pixel value.
3. Use marker-based watershed to split touching rooms that share a thin
   connection — pure connected-components merges them.
4. Return per-room masks plus debug images.
"""

from __future__ import annotations
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_rooms(
    wall_mask: np.ndarray,
    corridor_thresh: int | None = None,   # None = auto-derive from image
    min_room_area: int | None  = None,    # None = auto-derive from image
    morph_kernel: tuple[int, int] = (3, 3),
) -> tuple[list, np.ndarray, np.ndarray]:
    """
    Identify room regions from a binary wall mask.

    All thresholds default to *None* and are derived from the image when
    not supplied, so the same code works on any scan resolution.

    Parameters
    ----------
    wall_mask        : binary uint8 image, walls=255, free=0
    corridor_thresh  : distance-transform cutoff to split corridors.
                       Pass an explicit int to override auto-detection.
    min_room_area    : minimum blob area in px².
                       Pass an explicit int to override auto-detection.
    morph_kernel     : morphological opening kernel after thresholding.

    Returns
    -------
    rooms      : list of uint8 masks, one per room (255 inside, 0 outside)
    room_core  : debug — thresholded core image
    dist       : debug — raw distance-transform output
    """
    _, bin_img = cv2.threshold(wall_mask, 127, 255, cv2.THRESH_BINARY)
    free = 255 - bin_img

    # Distance transform — every free pixel = distance to nearest wall
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)

    # --- Auto-derive corridor_thresh from this image -------------------------
    if corridor_thresh is None:
        corridor_thresh = _auto_corridor_thresh(dist)

    # --- Auto-derive min_room_area from image size ---------------------------
    if min_room_area is None:
        min_room_area = _auto_min_room_area(wall_mask)

    # Threshold: keep only pixels comfortably inside rooms
    _, room_core = cv2.threshold(dist, corridor_thresh, 255, cv2.THRESH_BINARY)
    room_core = room_core.astype(np.uint8)

    # Morphological open — remove speckle from the core
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
    room_core = cv2.morphologyEx(room_core, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- Watershed to split touching rooms -----------------------------------
    rooms = _watershed_split(free, room_core, dist, min_room_area)

    return rooms, room_core, dist


def extract_room_features(room_masks: list) -> list[dict]:
    """
    Compute geometric features for each room mask.

    Returns list of dicts:
        id, area, centroid, bbox, aspect_ratio,
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

        perimeter      = cv2.arcLength(cnt, True)
        compactness    = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        rectangularity = area / (w * h + 1e-6)

        rooms_data.append({
            "id":             i,
            "area":           area,
            "centroid":       (cx, cy),
            "bbox":           (x, y, w, h),
            "aspect_ratio":   w / (h + 1e-5),
            "compactness":    compactness,
            "rectangularity": rectangularity,
            "contour":        cnt,
            "mask":           mask,
        })

    return rooms_data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auto_corridor_thresh(dist: np.ndarray) -> float:
    """
    Derive a corridor threshold purely from the distance map.

    Finds local maxima of the distance transform (room centre-points),
    then sets the threshold at 35% of their median value.
    This scales naturally with image resolution and wall thickness —
    no hardcoded pixel values.
    """
    smooth = cv2.GaussianBlur(dist, (5, 5), 0)

    # Dilate to find local maxima
    k = max(5, int(min(dist.shape) * 0.02) | 1)   # ~2% of image, always odd
    dilated = cv2.dilate(smooth, np.ones((k, k), np.uint8))
    local_max = (smooth >= dilated - 0.5) & (smooth > 0)

    peak_values = smooth[local_max]
    if len(peak_values) == 0:
        return 6.0

    median_peak = float(np.median(peak_values))
    thresh = median_peak * 0.35
    return max(thresh, 4.0)


def _auto_min_room_area(wall_mask: np.ndarray) -> int:
    """
    Minimum room area = 1.5% of total image area, floored at 800 px².
    Scales automatically with any resolution.
    """
    total = wall_mask.shape[0] * wall_mask.shape[1]
    return max(800, int(total * 0.015))


def _watershed_split(
    free: np.ndarray,
    room_core: np.ndarray,
    dist: np.ndarray,
    min_room_area: int,
) -> list[np.ndarray]:
    """
    Marker-based watershed to separate touching rooms.

    Each connected blob in room_core becomes a seed marker.
    Watershed expands those seeds through full free-space, using the
    inverted distance transform as the topographic surface.

    Correctly splits two rooms connected only by a doorway-width passage
    where pure connectedComponents would merge them into one blob.
    """
    num_seeds, seed_labels = cv2.connectedComponents(room_core, connectivity=8)

    if num_seeds <= 1:
        return _cc_fallback(free, min_room_area)

    markers = seed_labels.astype(np.int32)

    # Build 3-channel image for cv2.watershed (inverted dist = room centres are valleys)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inverted  = 255 - dist_norm
    vis       = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)

    cv2.watershed(vis, markers)

    rooms = []
    for label_id in range(1, num_seeds):
        mask = np.zeros(free.shape, dtype=np.uint8)
        mask[markers == label_id] = 255
        mask = cv2.bitwise_and(mask, free)   # constrain to free space only
        if cv2.countNonZero(mask) >= min_room_area:
            rooms.append(mask)

    return rooms


def _cc_fallback(free: np.ndarray, min_room_area: int) -> list[np.ndarray]:
    """Simple connected-components fallback when watershed finds no seeds."""
    num_labels, labels = cv2.connectedComponents(free, connectivity=8)
    rooms = []
    for i in range(1, num_labels):
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == i] = 255
        if cv2.countNonZero(mask) >= min_room_area:
            rooms.append(mask)
    return rooms