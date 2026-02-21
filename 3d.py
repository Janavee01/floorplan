!pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# HF STACK (COMPATIBLE)
!pip install diffusers==0.27.2
!pip install transformers==4.41.2
!pip install accelerate==0.31.0
!pip install peft==0.11.1
!pip install huggingface_hub==0.23.4

# CV LIBS
!pip install opencv-python pillow numpy

import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Using:", device)

img = cv2.imread("download.jpg", cv2.IMREAD_GRAYSCALE)

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

img = cv2.imread("walls_no_text.png", cv2.IMREAD_GRAYSCALE)

_, bin_img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
binary = bin_img.copy()

kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
walls_thick = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

# IMPORTANT: preserve inner walls
kernel_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
eroded = cv2.erode(walls_thick, kernel_erosion, iterations=1)

kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
dilated = cv2.dilate(eroded, kernel_dilate, iterations=2)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

final = np.zeros_like(dilated)

for i in range(1, num_labels):
    if stats[i][4] > 3000:
        final[labels == i] = 255

cv2.imwrite("final_no_furniture.png", final)

final_img = cv2.imread("final_no_furniture.png", cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(final_img, 50, 150)
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
edges_rgb = cv2.resize(edges_rgb, (512, 512))
mlsd_image = Image.fromarray(edges_rgb)

rooms = cv2.bitwise_not(final_img)

kernel = np.ones((5,5), np.uint8)
rooms = cv2.morphologyEx(rooms, cv2.MORPH_OPEN, kernel)

num_labels, labels = cv2.connectedComponents(rooms)

seg_map = np.zeros((rooms.shape[0], rooms.shape[1], 3), dtype=np.uint8)

np.random.seed(42)

for i in range(1, num_labels):
    seg_map[labels == i] = np.random.randint(0, 255, size=3)

seg_map = cv2.resize(seg_map, (512, 512))
seg_image = Image.fromarray(seg_map)

controlnet_mlsd = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-mlsd",
    torch_dtype=dtype
)

controlnet_seg = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-seg",
    torch_dtype=dtype
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=[controlnet_mlsd, controlnet_seg],
    torch_dtype=dtype,
    safety_checker=None
).to(device)

prompt = """
realistic top-down 3D floorplan render,
modern apartment interior,
correct furniture placement,
architectural visualization,
photorealistic, ultra detailed,
wooden flooring in all rooms,
beige painted walls
"""

negative_prompt = """
cartoon, sketch, distorted walls, text, watermark
"""

generator = torch.Generator(device=device).manual_seed(42)

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=[mlsd_image, seg_image],
    num_inference_steps=40,
    guidance_scale=7.0,
    controlnet_conditioning_scale=[0.9, 0.8],
    generator=generator
).images[0]

result.save("3d_floorplan.jpg")
result
