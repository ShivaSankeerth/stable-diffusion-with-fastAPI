from fastapi import FastAPI
from fastapi.responses import FileResponse
import io
from fastapi.responses import StreamingResponse
from model import generate_image_from_prompt
app = FastAPI()


@app.get("/generate")
def generate_image(prompt: str, seed: int = 0, guidance_scale: float = 7.5, num_inference_steps: int = 40):
    stream_memory = io.BytesIO()
    image = generate_image_from_prompt(prompt,seed=seed, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
    image.save(stream_memory, format="PNG")
    stream_memory.seek(0)
    return StreamingResponse(stream_memory, media_type="image/png")


