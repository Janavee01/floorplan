"""
ocr.py — extract room-label text from a floorplan image using Tesseract.
"""

import cv2
import pytesseract
from pytesseract import Output
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_labels(image_bgr: "np.ndarray", min_length: int = 2) -> list[dict]:
    """
    Run Tesseract OCR on *image_bgr* and return detected text regions.

    Parameters
    ----------
    image_bgr   : colour image loaded with cv2.imread
    min_length  : ignore tokens shorter than this (filters single chars / noise)

    Returns
    -------
    list of dicts with keys: text, x, y, w, h
    """
    data = pytesseract.image_to_data(image_bgr, output_type=Output.DICT)

    labels = []
    for i, text in enumerate(data["text"]):
        text = text.strip().upper()
        if len(text) <= min_length:
            continue
        labels.append({
            "text": text,
            "x":    data["left"][i],
            "y":    data["top"][i],
            "w":    data["width"][i],
            "h":    data["height"][i],
        })

    return labels
