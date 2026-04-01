"""
Download HuggingFace models and LoRAs to the RunPod network volume.

Run this ONCE on a RunPod GPU pod with the volume mounted:
  python download_models.py

Volume layout:
  /runpod-volume/
  ├── models/
  │   ├── lightx2v/
  │   │   └── Qwen-Image-Edit-Causal/   (edit + inpaint, ~57GB)
  │   └── Qwen/
  │       └── Qwen-Image/               (T2I, ~30GB)
  └── loras/
      ├── Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors
      ├── Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors
      ├── Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors
      └── Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors

Each HuggingFace model is self-contained (VAE, CLIP, scheduler included).
"""
import os
import sys

MODEL_DIR = os.getenv("MODEL_DIR", "/runpod-volume/models")
LORA_DIR = os.getenv("LORA_DIR", "/runpod-volume/loras")

# HuggingFace models (repo_id -> local subdirectory under MODEL_DIR)
MODELS = {
    # Edit Causal (LightX2V) — edit + inpaint, block causal attention
    "lightx2v/Qwen-Image-Edit-Causal": "lightx2v/Qwen-Image-Edit-Causal",
    # Qwen T2I
    "Qwen/Qwen-Image": "Qwen/Qwen-Image",
}

# LoRAs — (repo_id, filename) -> local filename in LORA_DIR
LORAS = {
    # Edit Lightning LoRAs (fp32)
    ("lightx2v/Qwen-Image-Edit-2511-Lightning",
     "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors"):
        "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",

    ("lightx2v/Qwen-Image-Edit-2511-Lightning",
     "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors"):
        "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors",

    # T2I Lightning LoRAs (fp32)
    ("lightx2v/Qwen-Image-2512-Lightning",
     "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"):
        "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",

    ("lightx2v/Qwen-Image-2512-Lightning",
     "Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors"):
        "Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors",
}


def main():
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import snapshot_download, hf_hub_download

    # --- Download models ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"=== Downloading models to {MODEL_DIR} ===\n")

    for repo_id, local_name in MODELS.items():
        local_dir = os.path.join(MODEL_DIR, local_name)
        if os.path.exists(local_dir) and any(
            f.endswith(".json") for f in os.listdir(local_dir)
        ):
            print(f"SKIP: {repo_id} (already at {local_dir})")
            continue

        print(f"Downloading {repo_id} -> {local_dir}")
        print("This may take a while...\n")

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"DONE: {repo_id}\n")

    # --- Download LoRAs ---
    os.makedirs(LORA_DIR, exist_ok=True)
    print(f"\n=== Downloading LoRAs to {LORA_DIR} ===\n")

    for (repo_id, filename), local_name in LORAS.items():
        local_path = os.path.join(LORA_DIR, local_name)
        if os.path.exists(local_path):
            print(f"SKIP: {local_name} (already exists)")
            continue

        print(f"Downloading {repo_id}/{filename}")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=LORA_DIR,
        )
        print(f"DONE: {local_name}\n")

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"\nModels ({MODEL_DIR}):")
    for d in sorted(os.listdir(MODEL_DIR)):
        full = os.path.join(MODEL_DIR, d)
        if os.path.isdir(full):
            print(f"  {d}/")
            for sub in sorted(os.listdir(full)):
                subpath = os.path.join(full, sub)
                if os.path.isdir(subpath):
                    print(f"    {sub}/")

    print(f"\nLoRAs ({LORA_DIR}):")
    for f in sorted(os.listdir(LORA_DIR)):
        fpath = os.path.join(LORA_DIR, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / 1e6
            print(f"  {f} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
