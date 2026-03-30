# Generated from: floorplan_mar_8.ipynb
# Converted at: 2026-03-29T11:46:41.925Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image

import torch

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
dtype = torch.float16 if use_cuda else torch.float32

!apt-get update
!apt-get install -y tesseract-ocr

!pip install pytesseract

import cv2
import numpy as np

img = cv2.imread("floorplan.jpg", cv2.IMREAD_GRAYSCALE)
assert img is not None, "Image not loaded"

import pytesseract
from pytesseract import Output

original_color = cv2.imread("floorplan.jpg")

ocr_data = pytesseract.image_to_data(
original_color,
output_type=Output.DICT
)

room_labels = []

for i in range(len(ocr_data["text"])):
    text = ocr_data["text"][i].strip().upper()


    if len(text) > 2:
        x = ocr_data["left"][i]
        y = ocr_data["top"][i]
        w = ocr_data["width"][i]
        h = ocr_data["height"][i]

        room_labels.append((text, x, y, w, h))

print("Detected room labels:", room_labels)

bin_img = cv2.adaptiveThreshold(
img, 255,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
cv2.THRESH_BINARY_INV,
15, 3
)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
clean = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)

text_removed = np.zeros_like(clean)

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]


    component = (labels == i).astype(np.uint8) * 255

    aspect_ratio = w / (h + 1e-5)
    extent = area / (w * h + 1e-5)
    thickness = area / (max(w, h) + 1e-5)

    keep = False

    if area > 2000:
        keep = True

    if (w > 80 or h > 80) and thickness > 6:
        keep = True

    if extent > 0.4 and area > 1500:
        keep = True

    if area < 300:
        keep = False

    if 0.3 < aspect_ratio < 4.0 and area < 1000:
        keep = False

    if thickness < 3:
        keep = False

    if keep:
        text_removed[labels == i] = 255

cv2.imwrite("walls_no_text.png", text_removed)

import cv2
import numpy as np

def extract_rooms(wall_img):
    """
    wall_img: binary image
              walls = white (255)
              background = black (0)
    returns: list of room masks
    """

    _, bin_img = cv2.threshold(wall_img, 127, 255, cv2.THRESH_BINARY)

    inv = cv2.bitwise_not(bin_img)

    h, w = inv.shape

    flood = inv.copy()
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, mask, (0,0), 0)

    rooms_only = cv2.bitwise_and(inv, cv2.bitwise_not(flood))

    num_labels, labels = cv2.connectedComponents(rooms_only)

    room_masks = []

    for i in range(1, num_labels):
        room_mask = np.zeros_like(labels, dtype=np.uint8)
        room_mask[labels == i] = 255
        area = cv2.countNonZero(room_mask)

        if area > 500:
            room_masks.append(room_mask)

    return room_masks

import cv2
import numpy as np

def segment_rooms_from_walls(wall_img, corridor_thresh=12, min_room_area=1500):
    """
    wall_img: binary image
              walls = 255
              background = 0

    corridor_thresh: width threshold to cut narrow connectors
    min_room_area: filter noise regions
    """

    _, bin_img = cv2.threshold(wall_img, 127, 255, cv2.THRESH_BINARY)

    free = 255 - bin_img

    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)

    corridor_mask = dist < corridor_thresh

    room_core = free.copy()
    room_core[corridor_mask] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    room_core = cv2.morphologyEx(room_core, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels = cv2.connectedComponents(room_core)

    rooms = []
    for i in range(1, num_labels):
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == i] = 255

        area = cv2.countNonZero(mask)
        if area > min_room_area:
            rooms.append(mask)

    return rooms, room_core, dist

img = cv2.imread("walls_no_text.png", cv2.IMREAD_GRAYSCALE)

rooms, room_core, dist = segment_rooms_from_walls(
    img,
    corridor_thresh=12,
    min_room_area=1500
)

print("Detected rooms:", len(rooms))

cv2.imwrite("debug_room_core.png", room_core)
cv2.imwrite("debug_dist.png", cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]

for i, room in enumerate(rooms):
    contours, _ = cv2.findContours(room, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, colors[i % len(colors)], 2)

cv2.imwrite("debug_rooms.png", vis)

import cv2
import numpy as np

def extract_room_features(room_masks):
    rooms_data = []

    for i, mask in enumerate(room_masks):
        area = cv2.countNonZero(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        x,y,w,h = cv2.boundingRect(cnt)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
        else:
            cx, cy = 0, 0

        aspect_ratio = w / h if h != 0 else 0

        perimeter = cv2.arcLength(cnt, True)

        compactness = (4*np.pi*area) / (perimeter**2 + 1e-6)

        rect_area = w*h
        rectangularity = area / rect_area if rect_area != 0 else 0

        rooms_data.append({
            "id": i,
            "area": area,
            "centroid": (cx, cy),
            "bbox": (x,y,w,h),
            "aspect_ratio": aspect_ratio,
            "compactness": compactness,
            "rectangularity": rectangularity,
            "contour": cnt,
            "mask": mask
        })

    return rooms_data

rooms_data = extract_room_features(rooms)

for r in rooms_data:
    print(f"Room {r['id']}: area={r['area']}, aspect={r['aspect_ratio']:.2f}, compact={r['compactness']:.3f}")

def classify_rooms(rooms_data):
    areas = [r["area"] for r in rooms_data]
    max_area = max(areas)

    sorted_rooms = sorted(rooms_data, key=lambda x: x["area"], reverse=True)

    for i, r in enumerate(sorted_rooms):
        area = r["area"]
        aspect = r["aspect_ratio"]
        compact = r["compactness"]

        if i == 0:
            r["type"] = "living_room"

        elif area < 45000:
            if compact > 0.5:
                r["type"] = "bathroom"
            else:
                r["type"] = "utility"

        elif area < 140000:
            if aspect > 1.5:
                r["type"] = "kitchen"
            else:
                r["type"] = "bedroom"

        else:
            r["type"] = "bedroom"

    return sorted_rooms

rooms_data = classify_rooms(rooms_data)

for r in rooms_data:
    print(f"Room {r['id']} → {r['type']}")

_, wall_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

import cv2
import numpy as np

def largest_inner_rect(mask):
    h, w = mask.shape
    hist = [0]*w
    best = (0,0,0,0)
    best_area = 0

    for y in range(h):
        for x in range(w):
            hist[x] = hist[x]+1 if mask[y,x] else 0

        stack=[]
        x=0
        while x<=w:
            cur = hist[x] if x<w else 0
            if not stack or cur>=hist[stack[-1]]:
                stack.append(x); x+=1
            else:
                top=stack.pop()
                width = x if not stack else x-stack[-1]-1
                area = hist[top]*width
                if area>best_area:
                    best_area=area
                    best=(x-width, y-hist[top]+1, width, hist[top])
    return best

def place_against_wall_constrained(room_mask, wall_mask,
                                   door_mask, clearance_map,
                                   size=(80,40)):
    ys, xs = np.where(room_mask > 0)
    candidates = list(zip(xs, ys))

    best = None
    best_score = -1

    step = max(5, int(np.sqrt(len(candidates))//20))
    for x,y in candidates[::step]:
        w,h = size
        x1,y1 = x-w//2, y-h//2
        x2,y2 = x1+w, y1+h

        if x1<0 or y1<0 or x2>=room_mask.shape[1] or y2>=room_mask.shape[0]:
            continue

        if np.any(room_mask[y1:y2, x1:x2] == 0):
            continue

        if np.any(door_mask[y1:y2, x1:x2] > 0):
            continue

        if np.any(clearance_map[y1:y2, x1:x2] > 0):
            continue

        edge = wall_mask[y1:y2, x1:x2]
        touch = (
            np.any(edge[0,:]) or
            np.any(edge[-1,:]) or
            np.any(edge[:,0]) or
            np.any(edge[:,-1])
        )

        score = 1000 if touch else 0

        if score > best_score:
            best_score = score
            best = (x1,y1,w,h)

    return best

def place_living(room_mask, wall_mask, door_mask, clearance_map):
    sofa = place_against_wall_constrained(
        room_mask, wall_mask, door_mask, clearance_map, size=(140,60)
    )
    table = largest_inner_rect(room_mask)
    return {"sofa": sofa, "table": table}

def place_bedroom(room_mask, wall_mask, door_mask, clearance_map):
    bed = place_against_wall_constrained(
        room_mask, wall_mask, door_mask, clearance_map, size=(140,80)
    )
    wardrobe = place_against_wall_constrained(
        room_mask, wall_mask, door_mask, clearance_map, size=(60,120)
    )
    return {"bed": bed, "wardrobe": wardrobe}

def place_bathroom(room_mask, wall_mask, door_mask, clearance_map):
    wc = place_against_wall_constrained(
        room_mask, wall_mask, door_mask, clearance_map, size=(40,40)
    )
    sink = place_against_wall_constrained(
        room_mask, wall_mask, door_mask, clearance_map, size=(40,30)
    )
    return {"wc": wc, "sink": sink}

def place_utility(room_mask, wall_mask, door_mask, clearance_map):
    wm = place_against_wall_constrained(
        room_mask, wall_mask, door_mask, clearance_map, size=(60,60)
    )
    return {"washing_machine": wm}

def draw_furniture(img, items, color):
    for name, rect in items.items():
        if rect is None:
            continue
        x,y,w,h = rect
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv2.putText(img, name, (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

img = cv2.imread("walls_no_text.png", cv2.IMREAD_GRAYSCALE)
_, wall_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for r in rooms_data:

    mask = r["mask"]
    t = r["type"]

    if t == "living_room":
        furn = place_living(mask, wall_mask, door_mask, clearance_map)
        draw_furniture(out, furn, (0,255,0))

    elif t == "bedroom":
        furn = place_bedroom(mask, wall_mask, door_mask, clearance_map)
        draw_furniture(out, furn, (255,0,0))

    elif t == "bathroom":
        furn = place_bathroom(mask, wall_mask, door_mask, clearance_map)
        draw_furniture(out, furn, (0,0,255))

    elif t == "utility":
        furn = place_utility(mask, wall_mask, door_mask, clearance_map)
        draw_furniture(out, furn, (255,255,0))

cv2.imwrite("furnished_plan.png", out)
print("Saved furnished_plan.png")

gemini_img = cv2.imread("furniture.png")
assert gemini_img is not None, "image not loaded"

gemini_img = cv2.resize(gemini_img, (512, 512))

gray = cv2.cvtColor(gemini_img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
mlsd_image = Image.fromarray(edges_rgb)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-mlsd",
    torch_dtype=dtype
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=dtype,
    safety_checker=None
).to(device)

prompt = """
top-down isometric 3D architectural floorplan render,
modern apartment interior visualization,
realistic materials,
volumetric soft daylight,
global illumination,
real wood flooring,
white matte walls,
subtle ambient occlusion,
cinematic lighting,
high detail,
ultra realistic
"""

negative_prompt = """
cartoon, sketch, flat 2d, diagram,
distorted layout, warped walls,
extra furniture, misplaced objects,
text, labels, watermark,
low quality, blurry
"""

generator = torch.Generator(device=device).manual_seed(42)

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=mlsd_image,
    num_inference_steps=40,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.4,
    generator=generator
).images[0]

result
