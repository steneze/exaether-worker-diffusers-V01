"""
Qwen T2I pipeline module.

Pipelines:
  - t2i : Text-to-image (DiffusionPipeline with Qwen-Image)

Model loaded at module level from /runpod-volume/models/.
"""
import logging
import os

import torch
from diffusers import DiffusionPipeline

from utils import encode_image, resolve_seed

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "/runpod-volume/models/diffusers")
LORA_DIR = os.getenv("LORA_DIR", "/runpod-volume/models/loras")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

logger.info("Loading Qwen-Image T2I model...")
pipe = DiffusionPipeline.from_pretrained(
    os.path.join(MODEL_DIR, "Qwen/Qwen-Image"),
    torch_dtype=torch.bfloat16,
).to("cuda")
logger.info("Qwen T2I model loaded.")

_current_lora = None


# ---------------------------------------------------------------------------
# LoRA management
# ---------------------------------------------------------------------------

def _apply_lora(lora_name: str, lora_strength: float = 1.0):
    """Load and fuse a LoRA if different from the currently loaded one."""
    global _current_lora

    if not lora_name:
        if _current_lora is not None:
            pipe.unload_lora_weights()
            _current_lora = None
        return

    if _current_lora == lora_name:
        return

    if _current_lora is not None:
        pipe.unload_lora_weights()

    lora_path = os.path.join(LORA_DIR, lora_name)
    logger.info(f"Loading LoRA: {lora_name} (strength={lora_strength})")
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora(lora_scale=lora_strength)
    _current_lora = lora_name


# ---------------------------------------------------------------------------
# Pipeline: t2i
# ---------------------------------------------------------------------------

def run_t2i(params: dict, callback) -> dict:
    """Run Qwen T2I pipeline.

    Args:
        params: {prompt, width, height, steps, seed, lora?, lora_strength?}
        callback: Called at each diffusion step
    """
    prompt = params["prompt"]
    width = int(params.get("width", 1024))
    height = int(params.get("height", 1024))
    steps = int(params.get("steps", 8))
    seed = resolve_seed(params.get("seed", -1))
    lora = params.get("lora", "")
    lora_strength = float(params.get("lora_strength", 1.0))

    _apply_lora(lora, lora_strength)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=" ",
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
        callback_on_step_end=callback,
    )

    return {
        "image": encode_image(result.images[0]),
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Pipeline registry
# ---------------------------------------------------------------------------

PIPELINES = {
    "t2i": run_t2i,
}
