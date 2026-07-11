# Layered-PSD RunPod serverless worker.
# CUDA 12.1 + Python 3.10 + Node 20 (for ag-psd PSD assembly). Model weights are baked in.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git curl ca-certificates \
    libgl1 libglib2.0-0 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# --- Python deps (CUDA torch from the cu121 index) ---
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1 \
    && pip install -r requirements.txt

# --- Node deps (ag-psd) ---
COPY package.json .
RUN npm install --omit=dev

# --- App code ---
COPY pipeline.py rp_handler.py assemble_psd.mjs download_models.py ./

# --- Bake model weights into the image (no per-cold-start download) ---
RUN python download_models.py

CMD ["python", "-u", "rp_handler.py"]
