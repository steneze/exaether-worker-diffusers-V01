"""
Download HuggingFace models to the RunPod network volume.

Run this ONCE on a RunPod GPU pod with the volume mounted:
  python download_models.py

Downloads to /runpod-volume/models/diffusers/ (or MODEL_DIR env var).
LoRAs are expected in /runpod-volume/models/loras/ (existing ComfyUI arbo).
"""
import os
import sys


MODEL_DIR = os.getenv("MODEL_DIR", "/runpod-volume/models/diffusers")

MODELS = {
    # Qwen Edit (edit + inpaint) — ~30GB
    "Qwen/Qwen-Image-Edit-2511": "Qwen/Qwen-Image-Edit-2511",
    # Qwen T2I — ~30GB (download later when needed)
    # "Qwen/Qwen-Image": "Qwen/Qwen-Image",
}


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)

    for repo_id, local_name in MODELS.items():
        local_dir = os.path.join(MODEL_DIR, local_name)
        if os.path.exists(local_dir) and os.listdir(local_dir):
            print(f"SKIP: {repo_id} already exists at {local_dir}")
            continue

        print(f"\nDownloading {repo_id} -> {local_dir}")
        print("This may take a while (~30GB)...\n")

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"DONE: {repo_id}")

    print("\nAll models downloaded.")
    print(f"Models directory: {MODEL_DIR}")
    print(f"Contents: {os.listdir(MODEL_DIR)}")


if __name__ == "__main__":
    main()
