import cv2
import numpy as np

img = cv2.imread("fp.jpg")
h, w = img.shape[:2]

# Copy your exact OCR output from the terminal
seeds = [
    {"text": "BEDROOM",    "x": 178, "y": 194, "w": 86,  "h": 14},
    {"text": "LIVING",     "x": 418, "y": 194, "w": 60,  "h": 14},
    {"text": "KITCHEN",    "x": 548, "y": 344, "w": 86,  "h": 14},
    {"text": "-BATH",      "x": 358, "y": 416, "w": 50,  "h": 14},
    {"text": "BEDROOM2",   "x": 168, "y": 460, "w": 100, "h": 14},
]

for s in seeds:
    cx = s["x"] + s["w"] // 2
    cy = s["y"] + s["h"] // 2
    cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)
    cv2.putText(img, s["text"], (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

cv2.imwrite("seeds_debug.png", img)
print("saved seeds_debug.png")