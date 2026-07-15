# Layered-PSD RunPod serverless worker.
# Base = OFFICIAL PyTorch image with cu128 → torch's arch_list includes sm_120 (Blackwell). RunPod's
# serverless pool is Blackwell-heavy (RTX PRO 6000 Blackwell MIG) and its gpuTypeIds filter does NOT
# reliably exclude them, so the worker kept landing on sm_120 cards that torch 2.4/cu124 (max sm_90)
# can't run → "no kernel image available for execution on the device". torch 2.8/cu128 supports every
# card RunPod assigns (Ampere sm_80/86, Ada sm_89, Hopper sm_90, Blackwell sm_120). Needs a >=12.8
# host driver → keep the endpoint's allowedCudaVersions at >=12.8.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

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

# --- Bake model weights FIRST (download-only; needs just hf hub + hf_transfer). Isolated from the
#     requirements layer so editing deps/app-code does NOT re-download the ~4GB of weights. ---
RUN pip install huggingface_hub==0.24.6 hf_transfer==0.1.8
COPY download_models.py .
RUN python download_models.py

# --- Python deps (torch/torchvision are in the base image; requirements.txt must NOT pin them) ---
COPY requirements.txt .
RUN pip install -r requirements.txt

# --- Node deps (ag-psd) ---
COPY package.json package-lock.json ./
RUN npm install --omit=dev

# --- App code (last, so code-only edits rebuild in seconds) ---
COPY pipeline.py rp_handler.py assemble_psd.mjs ./

CMD ["python", "-u", "rp_handler.py"]
