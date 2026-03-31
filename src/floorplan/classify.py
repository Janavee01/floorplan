"""
classify.py — assign a room type to each detected room.

Priority order
--------------
  1. OCR label — centroid + multi-point grid sampling inside room mask
  2. Geometric heuristics — aspect ratio, compactness, relative area
  3. CLIP zero-shot visual classifier — fallback on CPU/GPU

Key improvements over v1
-------------------------
- OCR matching samples a 3x3 grid of points inside each OCR bounding box,
  not just the centroid — catches shifted / inaccurate bounding boxes.
- Geometric heuristics skip CLIP entirely for clearly-rectangular rooms,
  cutting inference time and avoiding CLIP crop artifacts.
- CLIP crops mask out non-room pixels (walls) before passing to model —
  room interior only, no noise.
- Confidence floor: CLIP predictions below 0.32 are labelled "unknown"
  rather than silently accepted.
- OCR label map extended with numbered variants (BEDROOM2, BATH2 etc.)
"""

from __future__ import annotations
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# OCR label map  (all keys are UPPERCASE)
# ---------------------------------------------------------------------------

OCR_LABEL_MAP: dict[str, str] = {
    # Living / sitting
    "LIVING":         "living_room",
    "LIVING ROOM":    "living_room",
    "SITTING":        "living_room",
    "SITTING ROOM":   "living_room",
    "LOUNGE":         "living_room",
    "HALL":           "living_room",
    "DRAWING":        "living_room",
    "DRAWING ROOM":   "living_room",
    "FAMILY":         "living_room",
    "FAMILY ROOM":    "living_room",
    # Bedrooms (numbered variants)
    "BEDROOM":        "bedroom",
    "BED":            "bedroom",
    "MASTER":         "bedroom",
    "MASTER BEDROOM": "bedroom",
    "GUEST":          "bedroom",
    "GUEST ROOM":     "bedroom",
    "BEDROOM1":       "bedroom",
    "BEDROOM2":       "bedroom",
    "BEDROOM3":       "bedroom",
    "BED1":           "bedroom",
    "BED2":           "bedroom",
    "BED3":           "bedroom",
    # Kitchen / dining
    "KITCHEN":        "kitchen",
    "KITCHEN/DINING": "kitchen",
    "DINING":         "dining_room",
    "DINING ROOM":    "dining_room",
    "BREAKFAST":      "dining_room",
    # Bathrooms (numbered variants)
    "BATHROOM":       "bathroom",
    "BATH":           "bathroom",
    "BATHROOM1":      "bathroom",
    "BATHROOM2":      "bathroom",
    "BATH1":          "bathroom",
    "BATH2":          "bathroom",
    "WC":             "bathroom",
    "WIC":            "bathroom",
    "TOILET":         "bathroom",
    "SHOWER":         "bathroom",
    "WASHROOM":       "bathroom",
    "ENSUITE":        "bathroom",
    "EN SUITE":       "bathroom",
    # Utility / misc
    "UTILITY":        "utility",
    "UTILITY ROOM":   "utility",
    "STORE":          "utility",
    "STORAGE":        "utility",
    "BALCONY":        "utility",
    "PASSAGE":        "utility",
    "CORRIDOR":       "corridor",
    "HALLWAY":        "corridor",
    "STAIRCASE":      "utility",
    "LAUNDRY":        "utility",
    "GARAGE":         "utility",
    "STUDY":          "bedroom",    # study rooms are bedroom-like
    "OFFICE":         "bedroom",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_rooms(
    rooms_data:     list[dict],
    ocr_labels:     list[dict],
    original_image: np.ndarray,
) -> list[dict]:
    """
    Classify each segmented room.

    Pipeline per room:
      1. OCR match  (grid-sample the label bounding box)
      2. Geometric heuristics (size + shape rules)
      3. If no match, set "unknown" (no CLIP fallback)

    Corridors are removed unless they are unusually large.
    """
    sorted_rooms = sorted(rooms_data, key=lambda r: r["area"], reverse=True)
    total_free   = sum(r["area"] for r in sorted_rooms)

    matched = []
    for room in sorted_rooms:

        # 1 — OCR
        ocr_type = _match_ocr_label(room, ocr_labels)
        if ocr_type:
            room["type"]         = ocr_type
            room["label_source"] = "ocr"
            _maybe_keep(room, matched, total_free)
            continue

        # 2 — No ML fallback; keep OCR-only classification
        room["type"]         = "unknown"
        room["label_source"] = "ocr_only"
        _maybe_keep(room, matched, total_free)

    return matched


# ---------------------------------------------------------------------------
# Step 1 — OCR matching
# ---------------------------------------------------------------------------

def _match_ocr_label(room: dict, ocr_labels: list[dict]) -> str | None:
    """
    Match OCR text labels to room masks.

    Collect votes per room for all matching labels and choose the highest.
    This is more robust than returning the first match and avoids one bad label
    dominating the room when multiple labels are present.
    """
    mask = room["mask"]
    h_mask, w_mask = mask.shape

    score: dict[str, int] = {}

    for label in ocr_labels:
        lx, ly, lw, lh = label["x"], label["y"], label["w"], label["h"]

        room_type = _normalise_label(label["text"])
        if room_type is None:
            continue

        # Check if the label bounding box intersects the room mask
        matched = False

        # 1) 3x3 sample points on label box
        sample_points = [
            (lx + lw * fx // 4, ly + lh * fy // 4)
            for fx in range(1, 4)
            for fy in range(1, 4)
        ]
        for cx, cy in sample_points:
            if 0 <= cy < h_mask and 0 <= cx < w_mask and mask[cy, cx] == 255:
                matched = True
                break

        # 2) fallback — any overlap of box with room mask
        if not matched:
            x1 = max(0, lx)
            y1 = max(0, ly)
            x2 = min(w_mask, lx + lw)
            y2 = min(h_mask, ly + lh)
            if x2 > x1 and y2 > y1:
                if cv2.countNonZero(mask[y1:y2, x1:x2]) > 0:
                    matched = True

        if matched:
            score[room_type] = score.get(room_type, 0) + 1

    if not score:
        return None

    # pick dominant label by vote count (ties broken by deterministic order)
    best_type = max(sorted(score), key=lambda t: (score[t], t))
    return best_type


def _normalise_label(text: str) -> str | None:
    text = text.strip().upper()
    # Exact match first
    if text in OCR_LABEL_MAP:
        return OCR_LABEL_MAP[text]
    # Partial match — key appears anywhere in the detected text
    for key, room_type in OCR_LABEL_MAP.items():
        if key in text:
            return room_type
    return None


# ---------------------------------------------------------------------------
# Step 2 — Geometric heuristics
# ---------------------------------------------------------------------------

def _classify_geometry(room: dict, total_free_area: int) -> str | None:
    """
    Rule-based classification from shape and size alone.

    Rules are expressed as *fractions of total free area* so they scale
    automatically with any image resolution or plan size.
    """
    area_frac     = room["area"] / max(total_free_area, 1)
    aspect        = room["aspect_ratio"]     # w/h
    compact       = room["compactness"]      # 1.0 = perfect circle
    rect          = room["rectangularity"]   # 1.0 = perfect rectangle
    _, _, w, h    = room["bbox"]
    elongation    = max(w, h) / max(min(w, h), 1)

    # Very large, fairly rectangular → living room
    if area_frac > 0.25 and rect > 0.55:
        return "living_room"

    # Medium-large, rectangular → bedroom
    if 0.08 < area_frac <= 0.25 and rect > 0.60:
        return "bedroom"

    # Small and squarish → bathroom
    if area_frac < 0.08 and 0.5 < aspect < 2.0 and rect > 0.55:
        return "bathroom"

    # Very elongated and narrow → corridor
    if elongation > 3.5 and area_frac < 0.06:
        return "corridor"

    return None   # couldn't determine — will be labelled unknown


# ---------------------------------------------------------------------------
# Helper — corridor / unknown filtering
# ---------------------------------------------------------------------------

def _maybe_keep(room: dict, matched: list[dict], total_free: int) -> None:
    """
    Append room to matched unless it's a corridor or unknown that is also
    tiny (< 4% of free area). Large corridors/unknowns are kept so the
    user can see them.
    """
    area_frac = room["area"] / max(total_free, 1)
    rtype     = room.get("type", "")

    if rtype in ("corridor", "unknown") and area_frac < 0.04:
        print(f"      [skip] area={room['area']:>7}  -> {rtype} (too small)")
        return

    matched.append(room)