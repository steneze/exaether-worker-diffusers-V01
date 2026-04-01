"""
Qwen T2I pipeline function.

Pipelines:
  - t2i : Text-to-image

Receives a pre-loaded pipe from ModelManager.
Does NOT load models itself.
"""
import logging

from utils import encode_image, resolve_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline: t2i (text-to-image)
# ---------------------------------------------------------------------------

def run_t2i(pipe, params: dict, callback) -> dict:
    """Run Qwen T2I pipeline.

    Args:
        pipe: Pre-loaded DiffusionPipeline from ModelManager
        params: {prompt, width?, height?, steps?, seed?}
        callback: Diffusers callback_on_step_end
    """
    import torch

    prompt = params["prompt"]
    width = int(params.get("width", 1024))
    height = int(params.get("height", 1024))
    steps = int(params.get("steps", 50))
    seed = resolve_seed(params.get("seed", -1))

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
