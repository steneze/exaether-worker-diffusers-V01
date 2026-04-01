"""
Qwen Edit pipeline module.

Pipelines:
  - edit         : Image editing without mask (QwenImageEditPlusPipeline)
  - inpaint_edit : Inpainting with mask (QwenImageEditInpaintPipeline)

Models loaded at module level from /runpod-volume/models/.
LoRA dynamically loaded per request, cached to avoid reloads.
"""
import logging
import os

import torch
from diffusers import QwenImageEditPlusPipeline, QwenImageEditInpaintPipeline

from utils import (
    decode_base64_image,
    encode_image,
    resize_to_mpixels,
    prepare_mask,
    resolve_seed,
)

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "/runpod-volume/models")
LORA_DIR = os.getenv("LORA_DIR", "/runpod-volume/loras")

# ---------------------------------------------------------------------------
# Model loading (at import time = RunPod container startup)
# ---------------------------------------------------------------------------

logger.info("Loading Qwen-Image-Edit model...")
edit_pipe = QwenImageEditPlusPipeline.from_pretrained(
    os.path.join(MODEL_DIR, "Qwen/Qwen-Image-Edit-2511"),
    torch_dtype=torch.bfloat16,
).to("cuda")

logger.info("Loading Qwen-Image-Edit Inpaint model...")
inpaint_pipe = QwenImageEditInpaintPipeline.from_pretrained(
    os.path.join(MODEL_DIR, "Qwen/Qwen-Image-Edit-2511"),
    torch_dtype=torch.bfloat16,
).to("cuda")

logger.info("Qwen Edit models loaded.")

# LoRA cache: track the currently loaded LoRA to avoid reloads
_current_lora = {"edit": None, "inpaint": None}


# ---------------------------------------------------------------------------
# LoRA management
# ---------------------------------------------------------------------------

def _apply_lora(pipe, pipe_key: str, lora_name: str, lora_strength: float = 1.0):
    """Load and fuse a LoRA if different from the currently loaded one."""
    if not lora_name:
        if _current_lora[pipe_key] is not None:
            pipe.unload_lora_weights()
            _current_lora[pipe_key] = None
        return

    if _current_lora[pipe_key] == lora_name:
        return  # Already loaded

    # Swap LoRA
    if _current_lora[pipe_key] is not None:
        pipe.unload_lora_weights()

    lora_path = os.path.join(LORA_DIR, lora_name)
    logger.info(f"Loading LoRA: {lora_name} (strength={lora_strength})")
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora(lora_scale=lora_strength)
    _current_lora[pipe_key] = lora_name


# ---------------------------------------------------------------------------
# Pipeline: edit (image editing without mask)
# ---------------------------------------------------------------------------

def run_edit(params: dict, callback) -> dict:
    """Run Qwen Edit pipeline (no mask).

    Args:
        params: {prompt, image1, image2?, image3?, steps, seed, mpixels,
                 cfg, shift, lora?, lora_strength?}
        callback: Called at each diffusion step with (pipe, step, timestep, kwargs)
    """
    prompt = params["prompt"]
    steps = int(params.get("steps", 8))
    seed = resolve_seed(params.get("seed", -1))
    mpixels = float(params.get("mpixels", 1.0))
    cfg = float(params.get("cfg", 1.0))
    shift = float(params.get("shift", 3.1))
    lora = params.get("lora", "")
    lora_strength = float(params.get("lora_strength", 1.0))

    # Decode and resize images
    images = []
    for key in ("image1", "image2", "image3"):
        img_data = params.get(key)
        if img_data and isinstance(img_data, str) and len(img_data) > 100:
            img = decode_base64_image(img_data)
            img = resize_to_mpixels(img, mpixels)
            images.append(img)

    if not images:
        raise ValueError("At least image1 is required for edit pipeline")

    # Apply LoRA
    _apply_lora(edit_pipe, "edit", lora, lora_strength)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = edit_pipe(
        prompt=prompt,
        negative_prompt=" ",
        image=images,
        num_inference_steps=steps,
        true_cfg_scale=cfg,
        generator=generator,
        callback_on_step_end=callback,
    )

    return {
        "image": encode_image(result.images[0]),
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Pipeline: inpaint_edit (mask-based inpainting)
# ---------------------------------------------------------------------------

def run_inpaint_edit(params: dict, callback) -> dict:
    """Run Qwen Edit Inpaint pipeline (with mask).

    Args:
        params: {prompt, image1, mask, image2?, image3?, steps, seed, mpixels,
                 cfg, shift, lora?, lora_strength?, mask_grow?, mask_blur?}
        callback: Called at each diffusion step
    """
    prompt = params["prompt"]
    steps = int(params.get("steps", 8))
    seed = resolve_seed(params.get("seed", -1))
    mpixels = float(params.get("mpixels", 1.0))
    cfg = float(params.get("cfg", 1.0))
    shift = float(params.get("shift", 3.1))
    denoise = float(params.get("denoise", 0.9))
    mask_grow = int(params.get("mask_grow", 10))
    mask_blur = int(params.get("mask_blur", 10))
    lora = params.get("lora", "")
    lora_strength = float(params.get("lora_strength", 1.0))

    # Decode source image
    img_data = params.get("image1")
    if not img_data or not isinstance(img_data, str) or len(img_data) < 100:
        raise ValueError("image1 is required for inpaint_edit pipeline")
    source_image = decode_base64_image(img_data)
    source_image = resize_to_mpixels(source_image, mpixels)

    # Decode and prepare mask
    mask_data = params.get("mask")
    if not mask_data or not isinstance(mask_data, str) or len(mask_data) < 100:
        raise ValueError("mask is required for inpaint_edit pipeline")
    mask_image = decode_base64_image(mask_data)
    mask_image = prepare_mask(mask_image, source_image.size, grow=mask_grow, blur=mask_blur)

    # Apply LoRA
    _apply_lora(inpaint_pipe, "inpaint", lora, lora_strength)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = inpaint_pipe(
        prompt=prompt,
        negative_prompt=" ",
        image=source_image,
        mask_image=mask_image,
        strength=denoise,
        num_inference_steps=steps,
        true_cfg_scale=cfg,
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
    "edit": run_edit,
    "inpaint_edit": run_inpaint_edit,
}
