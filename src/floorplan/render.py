"""
render.py — isometric 3-D render via Stable Diffusion + ControlNet MLSD.
"""
from __future__ import annotations
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image

PROMPT = (
    "top-down isometric 3D architectural floorplan render, "
    "modern apartment interior visualization, realistic materials, "
    "volumetric soft daylight, global illumination, real wood flooring, "
    "white matte walls, subtle ambient occlusion, cinematic lighting, high detail, ultra realistic"
)
NEGATIVE_PROMPT = (
    "cartoon, sketch, flat 2d, diagram, distorted layout, warped walls, "
    "extra furniture, misplaced objects, text, labels, watermark, low quality, blurry"
)

def _get_device():
    use_cuda = torch.cuda.is_available()
    return ("cuda" if use_cuda else "cpu", torch.float16 if use_cuda else torch.float32)

def load_pipeline():
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    device, dtype = _get_device()
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd", torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet,
        torch_dtype=dtype, safety_checker=None,
    ).to(device)
    return pipe, device

def render_isometric(
    furnished_bgr, pipe, device,
    image_size=512, num_inference_steps=40,
    guidance_scale=7.5, controlnet_conditioning_scale=1.4, seed=42,
):
    img = cv2.resize(furnished_bgr, (image_size, image_size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    control_image = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
    generator = torch.Generator(device=device).manual_seed(seed)
    return pipe(
        prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT, image=control_image,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale, generator=generator,
    ).images[0]
