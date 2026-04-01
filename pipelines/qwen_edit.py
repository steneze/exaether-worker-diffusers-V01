"""
Qwen Edit pipeline functions.

Pipelines:
  - edit         : Image editing without mask
  - inpaint_edit : Inpainting with mask

These functions receive a pre-loaded pipe from ModelManager.
They do NOT load models themselves.
"""
import logging

from utils import (
    decode_base64_image,
    encode_image,
    resize_to_mpixels,
    prepare_mask,
    resolve_seed,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline: edit (image editing without mask)
# ---------------------------------------------------------------------------

def run_edit(pipe, params: dict, callback) -> dict:
    """Run Qwen Edit pipeline (no mask).

    Args:
        pipe: Pre-loaded QwenImageEditPlusPipeline from ModelManager
        params: {prompt, image1, image2?, image3?, steps?, seed?, mpixels?,
                 cfg?, shift?}
        callback: Diffusers callback_on_step_end
    """
    import torch

    prompt = params["prompt"]
    steps = int(params.get("steps", 50))
    seed = resolve_seed(params.get("seed", -1))
    mpixels = float(params.get("mpixels", 1.0))
    cfg = float(params.get("cfg", 1.0))
    shift = float(params.get("shift", 3.1))

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

    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
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

def run_inpaint_edit(pipe, params: dict, callback) -> dict:
    """Run Qwen Edit Inpaint pipeline (with mask).

    Args:
        pipe: Pre-loaded QwenImageEditInpaintPipeline from ModelManager
        params: {prompt, image1, mask, image2?, image3?, steps?, seed?,
                 mpixels?, cfg?, shift?, denoise?, mask_grow?, mask_blur?}
        callback: Diffusers callback_on_step_end
    """
    import torch

    prompt = params["prompt"]
    steps = int(params.get("steps", 50))
    seed = resolve_seed(params.get("seed", -1))
    mpixels = float(params.get("mpixels", 1.0))
    cfg = float(params.get("cfg", 1.0))
    shift = float(params.get("shift", 3.1))
    denoise = float(params.get("denoise", 0.9))
    mask_grow = int(params.get("mask_grow", 10))
    mask_blur = int(params.get("mask_blur", 10))

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
    mask_image = prepare_mask(
        mask_image, source_image.size, grow=mask_grow, blur=mask_blur
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
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
