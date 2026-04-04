"""
ocr.py — extract room-label text from a floorplan image using Tesseract.

Key improvements over v1
------------------------
- Deduplicates by (text, grid-position) not just text — so two rooms
  both labelled "BEDROOM" are each detected independently.
- Normalises OCR noise tokens (dashes, symbols) before dedup so they
  don't crowd out real labels.
- Runs multiple preprocessing passes with different scales and PSM modes
  so it handles any scan quality, font size or contrast.
"""
from __future__ import annotations
from .classify import OCR_LABEL_MAP
import re
import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def extract_labels(image_bgr: np.ndarray, min_length: int = 2) -> list[dict]:
    """
    Run multiple OCR passes on *image_bgr* and return all detected text regions.

    Returns
    -------
    list of dicts — keys: text, x, y, w, h  (original image coordinates)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    candidates = _make_candidates(gray)

    all_labels: list[dict] = []
    # Dedup key = (normalised_text, grid_col, grid_row)
    # Grid bucket size = 5% of image dimension so two rooms with the same
    # label in different areas of the plan are both kept.
    bucket_w = max(20, image_bgr.shape[1] // 20)
    bucket_h = max(20, image_bgr.shape[0] // 20)
    seen: set[tuple] = set()

    for img_variant, scale in candidates:
        labels = _run_tesseract(img_variant, scale, min_length)
        for label in labels:
            text = label["text"]
            gc = label["x"] // bucket_w
            gr = label["y"] // bucket_h
            key = (text, gc, gr)
            if key not in seen:
                seen.add(key)
                all_labels.append(label)

    # Merge adjacent tokens that form known multi-word labels (e.g. LIVING + ROOM)
    all_labels = _merge_adjacent_tokens(all_labels)

    # Spatial dedup — if two labels with same text overlap heavily, keep one
    all_labels = _spatial_dedup(all_labels)

    return all_labels


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_candidates(gray: np.ndarray) -> list[tuple]:
    """
    Return [(preprocessed_image, scale_factor), ...] to try OCR on.
    scale_factor maps detected coordinates back to original image space.
    """
    candidates = []

    # Pass 1 — original size, binary threshold
    _, t1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    candidates.append((t1, 1.0))

    # Pass 2 — 2x upscale + fixed binary threshold (catches small text)
    big2 = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, t2 = cv2.threshold(big2, 150, 255, cv2.THRESH_BINARY)
    candidates.append((t2, 2.0))

    # Pass 3 — 2x upscale + Otsu (adapts to varying contrast)
    _, t3 = cv2.threshold(big2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append((t3, 2.0))

    # Pass 4 — 3x upscale + Otsu (catches very small text)
    big3 = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, t4 = cv2.threshold(big3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append((t4, 3.0))

    # Pass 5 — CLAHE equalisation + 2x upscale (handles low contrast scans)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    big_eq = cv2.resize(eq, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, t5 = cv2.threshold(big_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append((t5, 2.0))

    return candidates


def _run_tesseract(img: np.ndarray, scale: float, min_length: int) -> list[dict]:
    """
    Run Tesseract with multiple PSM modes and return merged results.
    PSM 6  = assume uniform text block
    PSM 11 = sparse text — find as much as possible
    PSM 12 = sparse text + OSD
    """
    labels: list[dict] = []
    seen: set[str] = set()

    for psm in [6, 11, 12]:
        config = f"--psm {psm}"
        try:
            data = pytesseract.image_to_data(img, output_type=Output.DICT, config=config)
        except Exception:
            continue

        for i, raw_text in enumerate(data["text"]):
            text = raw_text.strip().upper()
            if len(text) <= min_length:
                continue
            if _is_noise(text):
                continue

            # Per-pass dedup (avoid duplicating within the same image variant)
            if text in seen:
                continue
            seen.add(text)

            labels.append({
                "text": text,
                "x":    int(data["left"][i]  / scale),
                "y":    int(data["top"][i]   / scale),
                "w":    int(data["width"][i] / scale),
                "h":    int(data["height"][i]/ scale),
            })

    return labels

def _is_noise(text: str) -> bool:
    # Leading punctuation = OCR artifact
    if text and text[0] in '-"\'|~`':
        return True
    if re.match(r"^[\d'\"X\.x!':,\s]+$", text):
        return True
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 3:
        return True
    if alpha_count / len(text) < 0.60:
        return True
    noise_chars = sum(1 for c in text if c in "[](){}|_&@#<>")
    if noise_chars >= 2:
        return True
    if len(text) <= 4 and not any(text in k for k in OCR_LABEL_MAP):
        return True
    return False

def _merge_adjacent_tokens(labels: list[dict]) -> list[dict]:
    MULTI_WORD = {
        "LIVING ROOM", "MASTER BEDROOM", "DINING ROOM",
        "SITTING ROOM", "FAMILY ROOM", "DRAWING ROOM",
        "GUEST ROOM", "UTILITY ROOM", "EN SUITE",
    }

    merged = list(labels)
    to_remove = set()          # ← track which originals got merged
    n = len(labels)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            combined = labels[i]["text"] + " " + labels[j]["text"]
            if combined not in MULTI_WORD:
                continue
            cy_i = labels[i]["y"] + labels[i]["h"] / 2
            cy_j = labels[j]["y"] + labels[j]["h"] / 2
            if abs(cy_i - cy_j) > max(labels[i]["h"], labels[j]["h"]) * 1.2:
                continue
            gap = abs(labels[j]["x"] - (labels[i]["x"] + labels[i]["w"]))
            if gap > labels[i]["w"] * 1.5:
                continue
            x1 = min(labels[i]["x"], labels[j]["x"])
            y1 = min(labels[i]["y"], labels[j]["y"])
            x2 = max(labels[i]["x"] + labels[i]["w"], labels[j]["x"] + labels[j]["w"])
            y2 = max(labels[i]["y"] + labels[i]["h"], labels[j]["y"] + labels[j]["h"])
            merged.append({"text": combined, "x": x1, "y": y1, "w": x2-x1, "h": y2-y1})
            to_remove.add(i)   # ← mark originals for removal
            to_remove.add(j)

    # Remove original tokens that were successfully merged
    merged = [l for idx, l in enumerate(labels) if idx not in to_remove] + \
             [l for l in merged if l not in labels]

    return merged

def _spatial_dedup(labels: list[dict]) -> list[dict]:
    """
    Remove duplicate labels that have the same text AND whose bounding
    boxes overlap by more than 50% (IoU > 0.3).  Keeps the larger box.
    """
    def iou(a, b):
        ax1, ay1 = a["x"], a["y"]
        ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
        bx1, by1 = b["x"], b["y"]
        bx2, by2 = bx1 + b["w"], by1 + b["h"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = a["w"] * a["h"] + b["w"] * b["h"] - inter
        return inter / max(union, 1)

    keep = [True] * len(labels)
    for i in range(len(labels)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(labels)):
            if not keep[j]:
                continue
            if labels[i]["text"] != labels[j]["text"]:
                continue
            if iou(labels[i], labels[j]) > 0.30:
                # Drop the smaller one
                area_i = labels[i]["w"] * labels[i]["h"]
                area_j = labels[j]["w"] * labels[j]["h"]
                if area_i >= area_j:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    return [l for l, k in zip(labels, keep) if k]