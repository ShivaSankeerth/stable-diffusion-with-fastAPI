from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image

path_token = Path("token.txt")
token = path_token.read_text().strip()
pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=token, 
    torch_dtype=torch.float16,
    revision="fp16")

pipeline.to("mps")
prompt = "A photo of a cat"

def generate_image_from_prompt(prompt: str,seed: int = 0, guidance_scale: float = 7.5, num_inference_steps: int = 40) -> Image:
    return pipeline(prompt,guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]

