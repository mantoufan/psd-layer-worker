"""Pre-download all model weights at image-build time so cold starts don't fetch them."""
import os
from transformers import (
    AutoProcessor, AutoModelForZeroShotObjectDetection,
    SamProcessor, SamModel, AutoModelForImageSegmentation,
)

DINO_ID = os.environ.get("DINO_ID", "IDEA-Research/grounding-dino-base")
SAM_ID = os.environ.get("SAM_ID", "facebook/sam-vit-huge")
BIREFNET_ID = os.environ.get("BIREFNET_ID", "ZhengPeng7/BiRefNet")

print("baking GroundingDINO:", DINO_ID)
AutoProcessor.from_pretrained(DINO_ID)
AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID)

print("baking SAM:", SAM_ID)
SamProcessor.from_pretrained(SAM_ID)
SamModel.from_pretrained(SAM_ID)

print("baking BiRefNet:", BIREFNET_ID)
AutoModelForImageSegmentation.from_pretrained(BIREFNET_ID, trust_remote_code=True)

print("baking LaMa")
from simple_lama_inpainting import SimpleLama
try:
    SimpleLama(device="cpu")  # triggers the model download
except Exception as e:
    print("lama prefetch note:", e)

print("all weights baked")
