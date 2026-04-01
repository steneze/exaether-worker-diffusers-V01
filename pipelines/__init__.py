"""
Pipeline registry + Model Manager with LRU VRAM cache.

No MODEL_FAMILY env var — the worker accepts any pipeline name and
lazy-loads the corresponding model on demand. Models stay in VRAM
as long as there's space; when VRAM runs low, the least recently
used model is evicted.
"""
import importlib
import logging
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/models")
LORA_DIR = os.getenv("LORA_DIR", "/workspace/loras")

# Min free VRAM (GB) to keep as headroom for inference (activations, etc.)
VRAM_HEADROOM_GB = float(os.getenv("VRAM_HEADROOM_GB", "4.0"))


# =========================================================================
# Pipeline registry — maps pipeline name → module, function, model key
# =========================================================================

PIPELINE_REGISTRY: Dict[str, Dict[str, str]] = {
    "edit":         {"module": "pipelines.qwen_edit", "fn": "run_edit",         "model": "edit_causal"},
    "inpaint_edit": {"module": "pipelines.qwen_edit", "fn": "run_inpaint_edit", "model": "edit_causal"},
    "t2i":          {"module": "pipelines.qwen_t2i",  "fn": "run_t2i",          "model": "qwen_image"},
}


# =========================================================================
# Model configs — maps model key → HF path + pipeline classes
# =========================================================================

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "edit_causal": {
        "path": os.path.join(MODEL_DIR, "lightx2v/Qwen-Image-Edit-Causal"),
        "estimated_vram_gb": 42,
        "pipelines": {
            "edit": {
                "class": "QwenImageEditPlusPipeline",
                "from_module": "diffusers",
            },
            "inpaint_edit": {
                "class": "QwenImageEditInpaintPipeline",
                "from_module": "diffusers",
            },
        },
    },
    "qwen_image": {
        "path": os.path.join(MODEL_DIR, "Qwen/Qwen-Image"),
        "estimated_vram_gb": 42,
        "pipelines": {
            "t2i": {
                "class": "DiffusionPipeline",
                "from_module": "diffusers",
            },
        },
    },
}


# =========================================================================
# Model Manager — lazy load + LRU VRAM cache
# =========================================================================

class ModelManager:
    """Manages model loading with LRU eviction based on VRAM availability.

    - Lazy loads models on first request
    - Keeps all loaded models in VRAM as long as there's space
    - Evicts least recently used model when VRAM is insufficient
    - Caches LoRA per model to avoid redundant load/unload
    """

    def __init__(self):
        # OrderedDict: model_key → {"pipes": {pipeline_name: pipe}, "lora": str|None}
        # Most recently used is at the end
        self._loaded: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._pipeline_fns: Dict[str, Callable] = {}  # pipeline_name → function

    def get_pipe(self, pipeline_name: str):
        """Get a loaded pipe for the given pipeline name. Lazy loads if needed."""
        if pipeline_name not in PIPELINE_REGISTRY:
            available = ", ".join(PIPELINE_REGISTRY.keys())
            raise ValueError(f"Unknown pipeline '{pipeline_name}'. Available: {available}")

        reg = PIPELINE_REGISTRY[pipeline_name]
        model_key = reg["model"]

        # Load model if not in VRAM
        if model_key not in self._loaded:
            self._ensure_vram_for(model_key)
            self._load_model(model_key)

        # Mark as recently used
        self._loaded.move_to_end(model_key)

        return self._loaded[model_key]["pipes"][pipeline_name]

    def get_pipeline_fn(self, pipeline_name: str) -> Callable:
        """Get the pipeline run function (e.g., run_edit, run_t2i)."""
        if pipeline_name not in self._pipeline_fns:
            reg = PIPELINE_REGISTRY[pipeline_name]
            module = importlib.import_module(reg["module"])
            self._pipeline_fns[pipeline_name] = getattr(module, reg["fn"])
        return self._pipeline_fns[pipeline_name]

    def apply_lora(
        self, pipeline_name: str, lora_name: Optional[str], lora_strength: float = 1.0
    ):
        """Load/swap LoRA on the model used by this pipeline."""
        model_key = PIPELINE_REGISTRY[pipeline_name]["model"]
        if model_key not in self._loaded:
            return  # Model not loaded yet

        entry = self._loaded[model_key]
        current_lora = entry.get("lora")

        # No LoRA requested
        if not lora_name:
            if current_lora is not None:
                # Unload current LoRA from all pipes of this model
                for pipe in entry["pipes"].values():
                    try:
                        pipe.unload_lora_weights()
                    except Exception:
                        pass
                entry["lora"] = None
            return

        # Same LoRA already loaded
        if current_lora == lora_name:
            return

        # Swap LoRA
        if current_lora is not None:
            for pipe in entry["pipes"].values():
                try:
                    pipe.unload_lora_weights()
                except Exception:
                    pass

        lora_path = os.path.join(LORA_DIR, lora_name)
        logger.info(f"Loading LoRA: {lora_name} (strength={lora_strength})")

        # Load on the pipe that will be used (LoRA is shared across pipes of same model)
        pipe = entry["pipes"][pipeline_name]
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_strength)
        entry["lora"] = lora_name

    # -----------------------------------------------------------------
    # VRAM management
    # -----------------------------------------------------------------

    def _ensure_vram_for(self, model_key: str):
        """Evict LRU models until there's enough VRAM for the new model."""
        needed_gb = MODEL_CONFIGS[model_key]["estimated_vram_gb"]

        while self._loaded:
            free_gb = self._get_free_vram_gb()
            if free_gb >= needed_gb + VRAM_HEADROOM_GB:
                break

            # Evict oldest (LRU = first item in OrderedDict)
            oldest_key, oldest_entry = self._loaded.popitem(last=False)
            logger.info(
                f"Evicting model '{oldest_key}' to free VRAM "
                f"(free={free_gb:.1f}GB, need={needed_gb + VRAM_HEADROOM_GB:.1f}GB)"
            )
            self._unload_entry(oldest_entry)

        # Final check: even with nothing loaded, might not have enough
        free_gb = self._get_free_vram_gb()
        if free_gb < needed_gb:
            logger.warning(
                f"VRAM may be insufficient: {free_gb:.1f}GB free, "
                f"{needed_gb}GB needed for '{model_key}'. Loading anyway."
            )

    def _load_model(self, model_key: str):
        """Load all pipelines for a model into VRAM."""
        config = MODEL_CONFIGS[model_key]
        model_path = config["path"]
        pipes: Dict[str, Any] = {}

        logger.info(f"Loading model '{model_key}' from {model_path}")

        for pipeline_name, pipe_cfg in config["pipelines"].items():
            cls_name = pipe_cfg["class"]
            from_mod = pipe_cfg["from_module"]

            logger.info(f"  Loading pipeline '{pipeline_name}' ({cls_name})")
            mod = importlib.import_module(from_mod)
            pipeline_cls = getattr(mod, cls_name)

            pipe = pipeline_cls.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to("cuda")

            pipes[pipeline_name] = pipe

        self._loaded[model_key] = {"pipes": pipes, "lora": None}
        logger.info(f"Model '{model_key}' loaded ({len(pipes)} pipeline(s))")

    @staticmethod
    def _unload_entry(entry: Dict[str, Any]):
        """Unload all pipes in an entry and free VRAM."""
        for name, pipe in entry["pipes"].items():
            del pipe
        entry["pipes"].clear()
        torch.cuda.empty_cache()

    @staticmethod
    def _get_free_vram_gb() -> float:
        """Get free VRAM in GB."""
        if not torch.cuda.is_available():
            return 0.0
        free, _ = torch.cuda.mem_get_info()
        return free / (1024 ** 3)


# =========================================================================
# Singleton
# =========================================================================

model_manager = ModelManager()
