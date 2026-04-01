FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
COPY utils.py .
COPY pipelines/ pipelines/

# Volume paths
ENV MODEL_DIR=/workspace/models
ENV LORA_DIR=/workspace/loras
ENV VRAM_HEADROOM_GB=4.0

CMD ["python", "-u", "handler.py"]
