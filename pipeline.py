"""
Layered-PSD engine: image -> per-subject transparent layers + inpainted background.

Pipeline (all on GPU):
  1. GroundingDINO  — open-vocab detection of the requested subject names -> boxes
  2. SAM            — box-prompted instance masks
  3. BiRefNet       — high-quality global alpha matte; per-instance alpha = matte ∩ SAM mask
  4. LaMa           — inpaint the union of subject masks -> clean background layer

Returns full-canvas layers (background opaque + one RGBA per subject), pixel-aligned to the
input so stacking them reconstructs the original image. PSD assembly is done by assemble_psd.mjs
(ag-psd) — the same library cv.cm ships — so the .psd is identical to the photo-editor's format.
"""
import io, os, numpy as np, torch, requests
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

DINO_ID = os.environ.get("DINO_ID", "IDEA-Research/grounding-dino-base")
SAM_ID = os.environ.get("SAM_ID", "facebook/sam-vit-huge")
BIREFNET_ID = os.environ.get("BIREFNET_ID", "ZhengPeng7/BiRefNet")

_models = {}


def load_models():
    """Load once into module globals (RunPod keeps the worker warm between jobs)."""
    if _models:
        return _models
    from transformers import (
        AutoProcessor, AutoModelForZeroShotObjectDetection,
        SamProcessor, SamModel, AutoModelForImageSegmentation,
    )
    _models["dino_p"] = AutoProcessor.from_pretrained(DINO_ID)
    _models["dino_m"] = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(DEVICE).eval()
    _models["sam_p"] = SamProcessor.from_pretrained(SAM_ID)
    _models["sam_m"] = SamModel.from_pretrained(SAM_ID).to(DEVICE).eval()
    bire = AutoModelForImageSegmentation.from_pretrained(BIREFNET_ID, trust_remote_code=True)
    bire.to(DEVICE).eval()
    if DEVICE == "cuda":
        bire.half()
    _models["birefnet"] = bire
    from simple_lama_inpainting import SimpleLama
    _models["lama"] = SimpleLama(device=torch.device(DEVICE))
    return _models


def _download_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def _detect(img, subjects, box_threshold):
    m = _models
    text = ". ".join(s.strip().lower() for s in subjects if s.strip()) + "."
    inp = m["dino_p"](images=img, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = m["dino_m"](**inp)
    res = m["dino_p"].post_process_grounded_object_detection(
        out, inp.input_ids, box_threshold=box_threshold, text_threshold=0.25,
        target_sizes=[img.size[::-1]],
    )[0]
    dets = []
    for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
        dets.append({"label": label or "object", "score": float(score),
                     "box": [float(v) for v in box]})
    dets.sort(key=lambda d: -d["score"])
    return dets


def _sam_masks(img, boxes):
    m = _models
    si = m["sam_p"](img, input_boxes=[boxes], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        so = m["sam_m"](**si)
    masks = m["sam_p"].image_processor.post_process_masks(
        so.pred_masks.cpu(), si["original_sizes"].cpu(), si["reshaped_input_sizes"].cpu())[0]
    iou = so.iou_scores.cpu().numpy()[0]
    out = []
    for i in range(masks.shape[0]):
        out.append(masks[i, int(iou[i].argmax())].numpy().astype(bool))
    return out


def _matte(img):
    """BiRefNet global alpha (float 0..1) at original resolution."""
    from torchvision import transforms
    W, H = img.size
    tf = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(img).unsqueeze(0).to(DEVICE)
    if DEVICE == "cuda":
        x = x.half()
    with torch.no_grad():
        pred = _models["birefnet"](x)[-1].sigmoid().cpu()[0, 0]
    a = Image.fromarray((pred.float().numpy() * 255).astype(np.uint8)).resize((W, H))
    return np.asarray(a, dtype=np.float32) / 255.0


def process(image_url, subjects, max_subjects=6, box_threshold=0.30):
    load_models()
    img = _download_image(image_url)
    W, H = img.size
    rgb = np.asarray(img)

    dets = _detect(img, subjects, box_threshold)
    # keep best detection per label, cap at max_subjects
    seen, keep = set(), []
    for d in dets:
        k = d["label"]
        if k in seen:
            continue
        seen.add(k)
        keep.append(d)
        if len(keep) >= max_subjects:
            break
    if not keep:
        return {"error": "no_subjects_detected", "canvas": [W, H]}

    boxes = [d["box"] for d in keep]
    masks = _sam_masks(img, boxes)
    matte = _matte(img)

    layers, claimed = [], np.zeros((H, W), bool)
    union = np.zeros((H, W), bool)
    for d, mask in zip(keep, masks):
        inst = mask & ~claimed
        claimed |= inst
        union |= inst
        alpha = (matte * inst).clip(0, 1)
        if float(alpha.mean()) < 0.002:   # skip near-empty
            continue
        # Crop the subject layer to its content bounding box — a full-canvas RGBA where the subject
        # occupies a small region wastes most of the PSD on transparent pixels (huge file, slow
        # ag-psd write + upload). Store the tight crop + its (left, top) so it composites in place.
        a8 = (alpha * 255).astype(np.uint8)
        ys, xs = np.where(a8 > 0)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        rgba = np.dstack([rgb[y0:y1, x0:x1], a8[y0:y1, x0:x1]])
        layers.append({"name": d["label"], "score": d["score"], "rgba": rgba, "left": x0, "top": y0})

    # background: LaMa inpaint the dilated union of subject masks (kept full-canvas + opaque)
    from scipy.ndimage import binary_dilation
    hole = binary_dilation(union, iterations=max(3, (W + H) // 600))
    bg = _models["lama"](img, Image.fromarray((hole * 255).astype(np.uint8)))
    bg = bg.convert("RGB").resize((W, H))
    bg_rgba = np.dstack([np.asarray(bg), np.full((H, W), 255, np.uint8)])

    ordered = [{"name": "background", "rgba": bg_rgba, "left": 0, "top": 0}] + list(reversed(layers))
    return {"canvas": [W, H], "layers": ordered,
            "subjects": [{"name": l["name"], "score": round(l.get("score", 1.0), 3)} for l in layers]}
