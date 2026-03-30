"""
classify.py — assign a room type to each detected room.

Priority order:
  1. OCR label matched to room centroid  (reads what's written on the plan)
  2. CLIP zero-shot fallback             (Phase 2 — slot is ready)
  3. Geometry heuristics                 (area + shape — last resort)
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Canonical label map — normalise whatever OCR reads to a room type
# ---------------------------------------------------------------------------

OCR_LABEL_MAP = {
    # Living areas
    "LIVING":        "living_room",
    "LIVING ROOM":   "living_room",
    "LOUNGE":        "living_room",
    "HALL":          "living_room",
    "DRAWING":       "living_room",
    "DRAWING ROOM":  "living_room",
    # Bedrooms
    "BEDROOM":       "bedroom",
    "BED":           "bedroom",
    "MASTER":        "bedroom",
    "GUEST":         "bedroom",
    # Kitchen
    "KITCHEN":       "kitchen",
    "KITCHEN/DINING":"kitchen",
    "DINING":        "kitchen",
    # Bathroom
    "BATHROOM":      "bathroom",
    "BATH":          "bathroom",
    "WC":            "bathroom",
    "TOILET":        "bathroom",
    "SHOWER":        "bathroom",
    "WASHROOM":      "bathroom",
    # Utility / other
    "UTILITY":       "utility",
    "STORE":         "utility",
    "STORAGE":       "utility",
    "BALCONY":       "utility",
    "PASSAGE":       "utility",
    "CORRIDOR":      "utility",
    "STAIRCASE":     "utility",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_rooms(rooms_data: list[dict], ocr_labels: list[dict]) -> list[dict]:
    """
    Label each room dict with a "type" key and a "label_source" key
    showing how the label was determined.

    Parameters
    ----------
    rooms_data  : output of segment.extract_room_features()
    ocr_labels  : output of ocr.extract_labels() —
                  list of dicts with keys: text, x, y, w, h

    Returns
    -------
    Same list sorted largest → smallest, each room gets:
      "type"         : room type string
      "label_source" : "ocr" | "heuristic"
    """
    sorted_rooms = sorted(rooms_data, key=lambda r: r["area"], reverse=True)

    for rank, room in enumerate(sorted_rooms):
        ocr_type = _match_ocr_label(room, ocr_labels)

        if ocr_type:
            room["type"]         = ocr_type
            room["label_source"] = "ocr"
        else:
            # Phase 2: insert CLIP call here before falling back to heuristics
            room["type"]         = _heuristic(room, rank)
            room["label_source"] = "heuristic"

    return sorted_rooms


# ---------------------------------------------------------------------------
# Step 1 — OCR label matching
# ---------------------------------------------------------------------------

def _match_ocr_label(room: dict, ocr_labels: list[dict]) -> str | None:
    """
    Check whether any OCR text centroid falls inside this room's mask.
    Return the normalised room type, or None if no match.

    How it works
    ------------
    Each OCR result has a bounding box (x, y, w, h).
    We compute the centre of that box and test whether that pixel
    is inside the room mask (mask[cy, cx] == 255).
    """
    mask = room["mask"]   # shape (H, W), 255 inside room

    for label in ocr_labels:
        # Centre of the OCR text bounding box
        cx = label["x"] + label["w"] // 2
        cy = label["y"] + label["h"] // 2

        # Bounds check
        if cy < 0 or cy >= mask.shape[0] or cx < 0 or cx >= mask.shape[1]:
            continue

        if mask[cy, cx] != 255:
            continue  # text centre is not inside this room

        # Normalise the text and look it up
        text = label["text"].strip().upper()
        room_type = _normalise_label(text)
        if room_type:
            return room_type

    return None


def _normalise_label(text: str) -> str | None:
    """
    Map raw OCR text to a room type using OCR_LABEL_MAP.
    Handles partial matches — e.g. "BED ROOM 1" still matches "bedroom".
    """
    # Exact match first
    if text in OCR_LABEL_MAP:
        return OCR_LABEL_MAP[text]

    # Partial match — check if any key is a substring of the OCR text
    for key, room_type in OCR_LABEL_MAP.items():
        if key in text:
            return room_type

    return None


# ---------------------------------------------------------------------------
# Step 2 — Geometry heuristic fallback
# ---------------------------------------------------------------------------

def _heuristic(room: dict, rank: int) -> str:
    """
    Last-resort classifier using area + shape.
    Used only when OCR found no label inside the room.
    """
    area    = room["area"]
    aspect  = room["aspect_ratio"]
    compact = room["compactness"]

    if rank == 0:
        return "living_room"

    if area < 45_000:
        return "bathroom" if compact > 0.5 else "utility"

    if area < 140_000:
        return "kitchen" if aspect > 1.5 else "bedroom"

    return "bedroom"
