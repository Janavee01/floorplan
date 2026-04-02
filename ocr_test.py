import cv2
img = cv2.imread("fp.jpg")

labels = [
    {"text": "BEDROOM",    "x": 442-30, "y": 199-5,  "w": 60, "h": 10},
]

# Just draw every OCR result directly
import pytesseract
from pytesseract import Output
import numpy as np

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
_, thresh = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
data = pytesseract.image_to_data(thresh, output_type=Output.DICT, config='--psm 11')

for i, text in enumerate(data["text"]):
    if text.strip() == "": continue
    x = data["left"][i] // 2
    y = data["top"][i] // 2
    w = data["width"][i] // 2
    h = data["height"][i] // 2
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
    cv2.putText(img, text.strip(), (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

cv2.imwrite("ocr_debug.png", img)
print("saved ocr_debug.png")

