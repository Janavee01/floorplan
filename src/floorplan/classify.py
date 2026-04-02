"""
classify.py — room type lookup and coloring.
"""
from __future__ import annotations
import numpy as np
import cv2
import re

OCR_LABEL_MAP: dict[str, str] = {
    "LIVING":               "living_room",
    "LIVING ROOM":          "living_room",
    "SITTING":              "living_room",
    "SITTING ROOM":         "living_room",
    "LOUNGE":               "living_room",
    "HALL":                 "living_room",
    "DRAWING ROOM":         "living_room",
    "FAMILY ROOM":          "living_room",
    "LIVING & DINING":      "living_room",
    "LIVING & DINING AREA": "living_room",
    "BEDROOM":              "bedroom",
    "MASTER BEDROOM":       "bedroom",
    "GUEST ROOM":           "bedroom",
    "BEDROOM 1":            "bedroom",
    "BEDROOM 2":            "bedroom",
    "BEDROOM 3":            "bedroom",
    "STUDY":                "bedroom",
    "OFFICE":               "bedroom",
    "KITCHEN":              "kitchen",
    "KITCHEN/DINING":       "kitchen",
    "DINING":               "dining_room",
    "DINING ROOM":          "dining_room",
    "BATHROOM":             "bathroom",
    "BATH":                 "bathroom",
    "WC":                   "bathroom",
    "W.C.":                 "bathroom",
    "TOILET":               "bathroom",
    "SHOWER":               "bathroom",
    "ENSUITE":              "bathroom",
    "EN SUITE":             "bathroom",
    "ENTRANCE":             "corridor",
    "ENTRY":                "corridor",
    "FOYER":                "corridor",
    "CORRIDOR":             "corridor",
    "HALLWAY":              "corridor",
    "UTILITY":              "utility",
    "UTILITY ROOM":         "utility",
    "STORAGE":              "utility",
    "LAUNDRY":              "utility",
    "GARAGE":               "utility",
    "CLOSET":               "utility",
    "BALCONY":              "utility",
}

ROOM_COLORS: dict[str, tuple] = {
    "bedroom":     (255, 180, 180),
    "living_room": (180, 255, 180),
    "kitchen":     (180, 220, 255),
    "bathroom":    (255, 255, 180),
    "dining_room": (220, 180, 255),
    "corridor":    (200, 200, 200),
    "utility":     (255, 220, 180),
    "unknown":     (230, 230, 230),
}


def _normalise_label(text: str) -> str | None:
    text = text.strip().upper()
    text = re.sub(r'^[^A-Z0-9]+', '', text)
    text = re.sub(r'[^A-Z0-9 /&.]+', '', text)
    if not text:
        return None
    if text in OCR_LABEL_MAP:
        return OCR_LABEL_MAP[text]
    for key, room_type in OCR_LABEL_MAP.items():
        if key in text:
            return room_type
    return None


def colorize_rooms(
    image_bgr: np.ndarray,
    rooms: list[dict],
) -> np.ndarray:
    result = image_bgr.copy()
    for room in rooms:
        color = ROOM_COLORS.get(room["type"], ROOM_COLORS["unknown"])
        mask = room["mask"]
        overlay = np.full_like(result, color[::-1])  # BGR
        result[mask == 255] = (
            result[mask == 255] * 0.4 + overlay[mask == 255] * 0.6
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, (30, 30, 30), 2)

        cx, cy = room["centroid"]
        label = room["type"].replace("_", " ").upper()
        cv2.putText(
            result, label, (cx - 30, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1
        )
    return result