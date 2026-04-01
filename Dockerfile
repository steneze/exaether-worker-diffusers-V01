FROM runpod/pytorch:1.0.3-cu1290-torch280-ubuntu2204

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
