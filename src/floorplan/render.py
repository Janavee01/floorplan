"""
render.py — isometric 3D render via Stable Diffusion + ControlNet MLSD.

GTX 1650 (4GB VRAM) notes:
- float16 produces all-NaN latents — must use float32
- float32 runs out of VRAM at postprocess step — offload to CPU for decode
"""
from __future__ import annotations
import cv2
import numpy as np
import torch
from PIL import Image

PROMPT = (
    "top-down isometric 3D architectural floor plan render, "
    "modern apartment interior, realistic materials, "
    "soft daylight, wood flooring, white walls, "
    "furniture clearly visible, bed in bedroom, sofa in living room, "
    "high detail, ultra realistic, clean architectural visualization"
)
NEGATIVE_PROMPT = (
    "cartoon, sketch, flat 2d, blurry, distorted, "
    "text, labels, watermark, low quality, black image, "
    "dark, overexposed, duplicate rooms"
)


def _get_device():
    if torch.cuda.is_available():
        return "cuda", torch.float32
    return "cpu", torch.float32


def load_pipeline():
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    device, dtype = _get_device()
    print(f"      Device: {device} ({dtype})")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-mlsd",
        torch_dtype=dtype,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    # Enable attention slicing — reduces VRAM usage significantly
    pipe.enable_attention_slicing()

    return pipe, device


def render_isometric(
    pipe,
    device: str,
    line_drawing_rgb: np.ndarray,
    image_size: int = 512,
    num_inference_steps: int = 20,   # reduced from 40 to save VRAM + time
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.4,
    seed: int = 42,
) -> Image.Image:
    edge_pixels  = int(np.sum(line_drawing_rgb[:,:,0] < 128))
    total_pixels = image_size * image_size
    print(f"      Edge pixels: {edge_pixels} / {total_pixels} "
          f"({100*edge_pixels/total_pixels:.1f}%)")

    control_image = Image.fromarray(line_drawing_rgb)
    generator     = torch.Generator(device=device).manual_seed(seed)

    # Get latents — stop before VAE decode to avoid OOM
    output = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=control_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator,
        output_type="latent",
    ).images

    print(f"      Latents OK — NaN count: {torch.isnan(output).sum().item()}")

    # Free GPU memory before VAE decode
    torch.cuda.empty_cache()

    # Decode on CPU to avoid OOM
    with torch.no_grad():
        pipe.vae = pipe.vae.to("cpu")
        latents  = output.to("cpu", dtype=torch.float32)
        latents  = latents / pipe.vae.config.scaling_factor
        decoded  = pipe.vae.decode(latents).sample
        decoded  = (decoded / 2 + 0.5).clamp(0, 1)
        decoded  = decoded.permute(0, 2, 3, 1).numpy()
        decoded  = (decoded[0] * 255).astype(np.uint8)

    return Image.fromarray(decoded)