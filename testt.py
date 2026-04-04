import cv2
import pytesseract
from pytesseract import Output

# If tesseract not in PATH, uncomment and set path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread("fp.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

_, thresh = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

data = pytesseract.image_to_data(thresh, output_type=Output.DICT, config='--psm 11')

for i, text in enumerate(data["text"]):
    if text.strip().lower() in ["living", "room"]:
        print(f"text='{text}' x={data['left'][i]//2} y={data['top'][i]//2} w={data['width'][i]//2} h={data['height'][i]//2}")