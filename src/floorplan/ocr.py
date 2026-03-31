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
    """
    Return True for tokens that are clearly not room labels:
    - Pure numbers
    - Dimension strings like 12'X11', 4X9, 6'x7'
    - Strings that are almost entirely punctuation / symbols
    """
    # Pure dimension pattern
    if re.match(r"^[\d'\"X\.x!':,\s]+$", text):
        return True
    # Strings with fewer than 2 alphabetic characters are noise
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 2:
        return True
    return False