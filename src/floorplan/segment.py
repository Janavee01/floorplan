"""
segment.py — OCR-anchored flood-fill room segmentation.

Key fix: wall gaps are sealed with morphological closing (no Hough needed).
Closing kernel size = estimated door width, so all door-sized gaps get filled
regardless of whether they were drawn as arcs, lines, or nothing at all.
"""
from __future__ import annotations
import cv2
import numpy as np
from .classify import _normalise_label


def segment_rooms(
    wall_mask: np.ndarray,
    ocr_labels: list[dict],
) -> list[dict]:
    h, w = wall_mask.shape

    # Step 1 — seal ALL wall gaps (doors, windows, tiny breaks)
    sealed = _seal_all_gaps(wall_mask, h, w)

    # Step 2 — free space = everything that is not wall
    free = cv2.bitwise_not(sealed)

    rooms = []
    used_pixels = np.zeros((h, w), dtype=np.uint8)

    for label in ocr_labels:
        room_type = _normalise_label(label["text"])
        if room_type is None:
            continue
        # Skip corridor labels — they're connectors not rooms
        if room_type == "corridor":
            continue

        sx = label["x"] + label["w"] // 2
        sy = label["y"] + label["h"] // 2
        sx = max(1, min(w - 2, sx))
        sy = max(1, min(h - 2, sy))

        # If seed lands on wall, find nearest free pixel
        if free[sy, sx] == 0:
            sx, sy = _nearest_free(free, sx, sy)
            if sx is None:
                print(f"      [skip] '{label['text']}' — seed stuck in wall")
                continue

        # If already claimed by a previous room, skip
        if used_pixels[sy, sx] == 255:
            print(f"      [skip] '{label['text']}' — area already claimed")
            continue

        # Flood fill from seed
        fill_input = free.copy()
        fill_mask  = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(fill_input, fill_mask, (sx, sy), 128)
        room_mask = (fill_input == 128).astype(np.uint8) * 255

        area = cv2.countNonZero(room_mask)
        if area < 500:
            print(f"      [skip] '{label['text']}' — fill too small ({area}px)")
            continue

        # Mark as used
        used_pixels = cv2.bitwise_or(used_pixels, room_mask)

        contours, _ = cv2.findContours(
            room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(cnt)

        print(f"      [room] '{label['text']}' -> {room_type}  area={area}")

        rooms.append({
            "type":         room_type,
            "label_source": "ocr",
            "mask":         room_mask,
            "area":         area,
            "bbox":         (x, y, bw, bh),
            "centroid":     (sx, sy),
            "label_text":   label["text"],
        })

    return rooms


def _seal_all_gaps(wall_mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Close all wall gaps up to door-width using morphological closing.

    Door width on a typical floorplan ≈ 2.5–4% of the shorter image dimension.
    We close with that kernel size so every door gap gets sealed regardless
    of whether it was drawn as an arc, a line, or not at all.

    Then we also run horizontal and vertical 1D closing to catch
    wall-end gaps that the 2D kernel might miss.
    """
    # Estimate door gap size from image dimensions
    gap_px = max(5, int(min(h, w) * 0.02))

    sealed = wall_mask.copy()

    # 2D closing — seals gaps in any direction
    kernel_2d = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (gap_px, gap_px)
    )
    sealed = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, kernel_2d)

    # 1D horizontal closing — catches horizontal wall breaks
    kernel_h = np.ones((1, gap_px), np.uint8)
    sealed = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, kernel_h)

    # 1D vertical closing — catches vertical wall breaks
    kernel_v = np.ones((gap_px, 1), np.uint8)
    sealed = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, kernel_v)

    return sealed


def _nearest_free(free: np.ndarray, x: int, y: int, search_r: int = 30):
    h, w = free.shape
    for r in range(1, search_r):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and free[ny, nx] == 255:
                    return nx, ny
    return None, None