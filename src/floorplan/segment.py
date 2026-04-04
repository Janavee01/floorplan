"""
segment.py — OCR-anchored flood-fill room segmentation.

Strategy
--------
1. Seal wall gaps with morphological closing
2. Flood fill from each OCR seed in priority order
3. For open-plan regions where two seeds share connected space,
   use nearest-seed Voronoi to split the filled region
"""
from __future__ import annotations
import cv2
import numpy as np
from .classify import _normalise_label

SEED_PRIORITY = {
    "bathroom":    0,
    "utility":     1,
    "kitchen":     2,
    "dining_room": 3,
    "bedroom":     4,
    "corridor":    5,
    "living_room": 6,
}


def segment_rooms(
    wall_mask: np.ndarray,
    ocr_labels: list[dict],
) -> list[dict]:
    h, w = wall_mask.shape

    sealed = _seal_all_gaps(wall_mask, h, w)
    free   = cv2.bitwise_not(sealed)
    seeds  = _resolve_seeds(ocr_labels, free, w, h)
    seeds.sort(key=lambda s: SEED_PRIORITY.get(s["room_type"], 99))

    if not seeds:
        return []

    # ── Step 1: flood fill each seed, track which pixels each fills ──────────
    # We first do unconstrained flood fills to find which seeds share regions
    fill_results = []
    for seed in seeds:
        sx, sy = seed["sx"], seed["sy"]
        fi = free.copy()
        fm = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(fi, fm, (sx, sy), 128)
        filled = (fi == 128).astype(np.uint8) * 255
        fill_results.append(filled)

    # ── Step 2: build Voronoi label map over ALL free pixels ─────────────────
    # For each free pixel, find which seed is closest (Euclidean)
    # This splits open-plan regions correctly
    seed_points = np.array([[s["sx"], s["sy"]] for s in seeds], dtype=np.float32)

    # Create coordinate grids
    ys, xs = np.mgrid[0:h, 0:w]
    coords  = np.stack([xs, ys], axis=-1).astype(np.float32)  # (h, w, 2)

    # Compute distance from every pixel to every seed — shape (h, w, n)
    n = len(seeds)
    dist_stack = np.zeros((h, w, n), dtype=np.float32)
    for i, seed in enumerate(seeds):
        dx = coords[:, :, 0] - seed["sx"]
        dy = coords[:, :, 1] - seed["sy"]
        dist_stack[:, :, i] = np.sqrt(dx*dx + dy*dy)

    # Nearest seed index per pixel
    voronoi = np.argmin(dist_stack, axis=2)  # (h, w)

    # ── Step 3: assign free pixels to rooms via Voronoi ─────────────────────
    rooms = []
    used_pixels = np.zeros((h, w), dtype=np.uint8)

    for i, seed in enumerate(seeds):
        room_type = seed["room_type"]
        label     = seed["label"]
        sx, sy    = seed["sx"], seed["sy"]

        # Pixels: nearest to this seed AND free AND not yet used
        voronoi_free = (
            (voronoi == i) &
            (free == 255) &
            (used_pixels == 0)
        ).astype(np.uint8) * 255

        # Only keep the connected component containing the seed
        if voronoi_free[sy, sx] == 0:
            sx2, sy2 = _nearest_free_in_mask(voronoi_free, sx, sy)
            if sx2 is None:
                print(f"      [skip] '{label['text']}' — no reachable free pixel")
                continue
            sx, sy = sx2, sy2

        flood = voronoi_free.copy()
        fm    = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(flood, fm, (sx, sy), 128)
        room_mask = (flood == 128).astype(np.uint8) * 255

        area = cv2.countNonZero(room_mask)
        if area < 500:
            print(f"      [skip] '{label['text']}' — too small ({area}px)")
            continue

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
            "centroid":     (seed["sx"], seed["sy"]),
            "label_text":   label["text"],
        })

    return rooms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_seeds(ocr_labels, free, w, h):
    seeds = []
    seen  = set()

    for label in ocr_labels:
        room_type = _normalise_label(label["text"])
        if room_type is None or room_type == "corridor":
            continue

        sx = label["x"] + label["w"] // 2
        sy = label["y"] + label["h"] // 2
        sx = max(1, min(w - 2, sx))
        sy = max(1, min(h - 2, sy))

        if free[sy, sx] == 0:
            sx, sy = _nearest_free(free, sx, sy)
            if sx is None:
                print(f"      [skip] '{label['text']}' — seed stuck in wall")
                continue

        bucket = (room_type, sx // 80, sy // 80)
        if bucket in seen:
            continue
        seen.add(bucket)

        seeds.append({
            "sx": sx, "sy": sy,
            "room_type": room_type,
            "label": label,
        })

    return seeds


def _seal_all_gaps(wall_mask, h, w):
    gap_px = max(5, int(min(h, w) * 0.02))
    sealed = wall_mask.copy()
    k2d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_px, gap_px))
    sealed = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, k2d)
    sealed = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, np.ones((1, gap_px), np.uint8))
    sealed = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, np.ones((gap_px, 1), np.uint8))
    return sealed


def _nearest_free(free, x, y, search_r=30):
    h, w = free.shape
    for r in range(1, search_r):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and free[ny, nx] == 255:
                    return nx, ny
    return None, None


def _nearest_free_in_mask(mask, x, y, search_r=40):
    h, w = mask.shape
    for r in range(1, search_r):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] == 255:
                    return nx, ny
    return None, None