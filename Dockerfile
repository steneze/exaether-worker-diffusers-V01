FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
COPY utils.py .
COPY pipelines/ pipelines/

# Defaults — override per endpoint
ENV MODEL_FAMILY=qwen_edit
ENV MODEL_DIR=/runpod-volume/models
ENV LORA_DIR=/runpod-volume/loras

CMD ["python", "-u", "handler.py"]
