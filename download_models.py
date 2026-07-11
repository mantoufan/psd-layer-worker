"""Pre-download all model weights at image-build time so cold starts don't fetch them.

IMPORTANT: only DOWNLOAD files here — do NOT instantiate models (from_pretrained / SimpleLama()).
The build host (c.y1) has an older CPU whose instruction set crashes torch's CPU kernels with SIGILL
(exit 132) when a model is actually constructed. The models are instantiated at runtime on the RunPod
GPU host instead. snapshot_download pulls every repo file (weights + BiRefNet's trust_remote_code .py),
so runtime from_pretrained reads straight from the baked HF cache.
"""
import os
from huggingface_hub import snapshot_download

DINO_ID = os.environ.get("DINO_ID", "IDEA-Research/grounding-dino-base")
SAM_ID = os.environ.get("SAM_ID", "facebook/sam-vit-huge")
BIREFNET_ID = os.environ.get("BIREFNET_ID", "ZhengPeng7/BiRefNet")

for repo in (DINO_ID, SAM_ID, BIREFNET_ID):
    print(f"baking (download only): {repo}", flush=True)
    snapshot_download(repo_id=repo, ignore_patterns=["*.msgpack", "*.h5", "*.tflite"])

# LaMa (simple-lama-inpainting) fetches big-lama.pt from a GitHub release into the torch hub cache at
# runtime. Pre-fetch that single file to the path SimpleLama expects, without importing torch.
# BEST-EFFORT: it's only ~200MB and RunPod hosts have internet, so if this path/URL ever drifts the
# model just downloads once at the first cold start — never fail the build over the LaMa prefetch.
print("baking LaMa big-lama.pt (best-effort)", flush=True)
try:
    import urllib.request
    LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
    lama_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    os.makedirs(lama_dir, exist_ok=True)
    lama_path = os.path.join(lama_dir, "big-lama.pt")
    if not os.path.exists(lama_path):
        urllib.request.urlretrieve(LAMA_URL, lama_path)
    print("lama baked ->", os.path.getsize(lama_path), "bytes")
except Exception as e:
    print("lama prefetch skipped (will download at runtime):", e)
print("all weights baked")
