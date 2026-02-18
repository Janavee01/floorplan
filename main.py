!pip install torch torchvision torchaudio
!pip install diffusers transformers accelerate
!pip install opencv-python pillow numpy



!pip uninstall -y xformers


!pip uninstall -y xformers
!pip uninstall -y xformers   # run twice to ensure removal


!pip cache purge


!pip install torch torchvision torchaudio --upgrade
!pip install diffusers transformers accelerate
!pip install opencv-python pillow numpy


import torch
from PIL import Image
import cv2
import numpy as np

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# ----------------------------
# 1️⃣ Load floorplan image
# ----------------------------

floorplan_path = "fp.jpg"

floorplan_image = Image.open(floorplan_path).convert("RGB")

# Resize (important for stability)
floorplan_image = floorplan_image.resize((768,768))

# Convert to numpy
floorplan_np = np.array(floorplan_image)

# Convert to grayscale
gray = cv2.cvtColor(floorplan_np, cv2.COLOR_RGB2GRAY)

# Threshold to isolate lines
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Remove small components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    thresh, connectivity=8
)

min_area = 150
clean = np.zeros_like(thresh)

for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] > min_area:
        clean[labels == i] = 255

clean = cv2.bitwise_not(clean)

# Edge detection
floorplan_edges = cv2.Canny(clean, 50, 150)

# ControlNet expects RGB
control_image = Image.fromarray(floorplan_edges).convert("RGB")

control_image


# ----------------------------
# 2️⃣ Load ControlNet + SD
# ----------------------------

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(device)

# Optional optimization
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("xformers enabled")
except:
    print("xformers not available")


# ----------------------------
# 3️⃣ Prompt
# ----------------------------

prompt = (
    "high-quality architectural 3D floorplan, top-down axonometric view, "
    "sharp walls, realistic rooms, modern interior, clean visualization, "
    "no text, no labels, consistent proportions, minimal furniture"
)

negative_prompt = (
    "cartoon, sketch, painting, abstract, distorted perspective, warped rooms, messy lines"
)

generator = torch.Generator(device=device).manual_seed(42)


# ----------------------------
# 4️⃣ Generate
# ----------------------------

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=control_image,
    num_inference_steps=40,
    guidance_scale=7,
    controlnet_conditioning_scale=0.7,
    generator=generator
).images[0]

result


# ----------------------------
# 5️⃣ Save
# ----------------------------

result.save("floorplan_3d_render.png")

print("✅ Done")
