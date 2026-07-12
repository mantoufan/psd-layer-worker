# Layered-PSD RunPod serverless worker.
# Base = the OFFICIAL PyTorch image (torch+torchvision+CUDA are pre-installed and TESTED to run on
# RunPod's GPUs) — a hand-rolled nvidia/cuda + `pip install torch` combo produced "no kernel image
# available for execution on the device" on every card, so we defer to the known-good build.
# cuda12.4 base → keep the endpoint's allowedCudaVersions at >=12.4.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PIP_NO_CACHE_DIR=1

# System deps + Node 20 (for ag-psd PSD assembly).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates libgl1 libglib2.0-0 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps (torch/torchvision already in the base image; requirements.txt must NOT pin them) ---
COPY requirements.txt .
RUN pip install -r requirements.txt

# --- Node deps (ag-psd) ---
COPY package.json package-lock.json ./
RUN npm install --omit=dev

# --- Bake model weights (download-only; no model instantiation → safe on the build host's CPU). ---
# Ordered BEFORE the app-code COPY so editing pipeline/handler code does NOT re-download the ~4GB.
COPY download_models.py .
RUN python download_models.py

# --- App code (last, so code-only edits rebuild in seconds) ---
COPY pipeline.py rp_handler.py assemble_psd.mjs ./

CMD ["python", "-u", "rp_handler.py"]
