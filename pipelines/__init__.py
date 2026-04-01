"""
Pipeline registry.

Loads the appropriate pipeline module based on MODEL_FAMILY env var.
Each module exports a PIPELINES dict: {pipeline_name: run_function}.
"""
import os
import logging

logger = logging.getLogger(__name__)

REGISTRY = {}


def load_pipelines() -> dict:
    """Load pipeline module for the configured MODEL_FAMILY."""
    family = os.getenv("MODEL_FAMILY", "qwen_edit")
    logger.info(f"Loading pipelines for MODEL_FAMILY={family}")

    if family == "qwen_edit":
        from .qwen_edit import PIPELINES
    elif family == "qwen_t2i":
        from .qwen_t2i import PIPELINES
    else:
        raise ValueError(f"Unknown MODEL_FAMILY: {family}")

    REGISTRY.update(PIPELINES)
    logger.info(f"Loaded {len(REGISTRY)} pipeline(s): {list(REGISTRY.keys())}")
    return REGISTRY


def get_pipeline(name: str):
    """Get a pipeline function by name."""
    if name not in REGISTRY:
        available = ", ".join(REGISTRY.keys()) or "(none loaded)"
        raise ValueError(f"Unknown pipeline '{name}'. Available: {available}")
    return REGISTRY[name]
