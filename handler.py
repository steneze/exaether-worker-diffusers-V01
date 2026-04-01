"""
RunPod Serverless Handler for diffusers pipelines.

MODEL_FAMILY env var determines which pipelines are loaded at startup.
Uses thread+queue pattern to bridge synchronous diffusers callback
with async RunPod streaming yield.
"""
import logging
import os
import queue
import threading

import torch
import runpod

from pipelines import load_pipelines, get_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Load pipelines at startup (before any jobs)
load_pipelines()


def streaming_handler(job):
    """RunPod streaming handler. Yields progress messages, then final result."""
    job_input = job["input"]
    pipeline_name = job_input.get("pipeline", "edit")
    params = job_input.get("params", {})
    steps = int(params.get("steps", 8))

    logger.info(f"Job {job['id']}: pipeline={pipeline_name}, steps={steps}")

    # Get the pipeline function
    try:
        pipeline_fn = get_pipeline(pipeline_name)
    except ValueError as e:
        yield {"status": "error", "error": str(e)}
        return

    # Queue for bridging sync callback -> async yield
    progress_queue = queue.Queue()

    def diffusion_callback(pipe, step, timestep, kwargs):
        """Called by diffusers at each denoising step."""
        progress_queue.put({
            "status": "progress",
            "step": step + 1,
            "total": steps,
            "percent": round((step + 1) / steps * 100, 1),
        })
        return kwargs

    def run_pipeline():
        """Run the actual pipeline in a thread."""
        try:
            result = pipeline_fn(params, diffusion_callback)
            progress_queue.put({"status": "completed", **result})
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            progress_queue.put({"status": "error", "error": str(e)})

    # Start pipeline in thread
    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    # Yield initial status
    yield {"status": "started", "total_steps": steps}

    # Forward progress messages
    while True:
        try:
            msg = progress_queue.get(timeout=300)  # 5 min max per step
        except queue.Empty:
            yield {"status": "error", "error": "Pipeline timed out (no progress for 5 minutes)"}
            break

        yield msg

        if msg.get("status") in ("completed", "error"):
            break

    # Cleanup
    thread.join(timeout=10)
    torch.cuda.empty_cache()
    logger.info(f"Job {job['id']}: done, VRAM cleared")


runpod.serverless.start({
    "handler": streaming_handler,
    "return_aggregate_stream": True,
})
