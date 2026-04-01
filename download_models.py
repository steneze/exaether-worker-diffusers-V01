"""
Download HuggingFace models and LoRAs to the RunPod network volume.

Run this ONCE on a RunPod GPU pod with the volume mounted:
  python download_models.py

Volume layout (clean, no ComfyUI legacy):
  /runpod-volume/
  ├── models/
  │   └── Qwen/
  │       ├── Qwen-Image-Edit-2511/   (edit + inpaint, ~30GB)
  │       └── Qwen-Image/             (T2I, ~30GB — uncomment when needed)
  └── loras/
      ├── Qwen-Image-EDIT-2511-Lightning-4steps-V1.0-bf16.safetensors
      └── Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors

Each HuggingFace model is self-contained (includes VAE, CLIP, scheduler, etc.).
No separate vae/ or clip/ directories needed.
"""
import os
import sys

MODEL_DIR = os.getenv("MODEL_DIR", "/runpod-volume/models")
LORA_DIR = os.getenv("LORA_DIR", "/runpod-volume/loras")

# HuggingFace models to download (repo_id -> local subdirectory)
MODELS = {
    # Qwen Edit — used for edit (no mask) + inpaint_edit (with mask)
    "Qwen/Qwen-Image-Edit-2511": "Qwen/Qwen-Image-Edit-2511",

    # Qwen T2I — uncomment when ready to deploy T2I endpoint
    # "Qwen/Qwen-Image": "Qwen/Qwen-Image",
}

# LoRAs to download (repo_id, filename) -> local filename
# These are individual safetensors files, not full repos
LORAS = {
    # 4-step Lightning for Qwen Edit
    ("Qwen/Qwen-Image-Edit-2511", "Qwen-Image-EDIT-2511-Lightning-4steps-V1.0-bf16.safetensors"):
        "Qwen-Image-EDIT-2511-Lightning-4steps-V1.0-bf16.safetensors",

    # 8-step Lightning for Qwen Edit/T2I
    ("Qwen/Qwen-Image-Edit-2511", "Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors"):
        "Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors",
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
        print("This may take a while (~30GB)...\n")

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

        print(f"Downloading {repo_id}/{filename} -> {local_path}")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=LORA_DIR,
        )
        # hf_hub_download may nest the file; move if needed
        nested = os.path.join(LORA_DIR, filename)
        if nested != local_path and os.path.exists(nested):
            os.rename(nested, local_path)
        print(f"DONE: {local_name}\n")

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"Models: {MODEL_DIR}")
    for d in sorted(os.listdir(MODEL_DIR)):
        full = os.path.join(MODEL_DIR, d)
        if os.path.isdir(full):
            print(f"  {d}/")
            for sub in sorted(os.listdir(full)):
                print(f"    {sub}/")

    print(f"\nLoRAs: {LORA_DIR}")
    for f in sorted(os.listdir(LORA_DIR)):
        size_mb = os.path.getsize(os.path.join(LORA_DIR, f)) / 1e6
        print(f"  {f} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
