"""
RunPod serverless handler for layered-PSD generation.

Input (job["input"]):
  image_url    (str, required)  — source image to layer
  subjects     (list[str])      — subject names to separate (from cv.cm's GLM vision step).
                                  If omitted, falls back to a generic object vocabulary.
  psd_put_url  (str, required)  — presigned S3 PUT URL; the assembled .psd is uploaded here
  max_subjects (int, default 6)
  box_threshold(float, default 0.30)

Output:
  { ok, canvas:[w,h], subjects:[{name,score}], layer_count, psd_bytes, timings:{...} }
The .psd itself is PUT to psd_put_url (kept out of the JSON response — it is large).
"""
import os, time, json, subprocess, tempfile, numpy as np, requests, runpod

GENERIC_VOCAB = ["person", "face", "product", "bottle", "box", "food", "car",
                 "animal", "cat", "dog", "plant", "furniture", "text", "logo"]


def _put(url, data, content_type):
    r = requests.put(url, data=data, headers={"Content-Type": content_type}, timeout=120)
    r.raise_for_status()


def handler(job):
    t0 = time.time()
    inp = job.get("input", {}) or {}

    # Diagnostic mode: report torch/GPU facts without loading models (for debugging arch/driver issues).
    if inp.get("diag"):
        import torch
        return {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        }
    image_url = inp.get("image_url")
    psd_put_url = inp.get("psd_put_url")
    if not image_url or not psd_put_url:
        return {"error": "image_url and psd_put_url are required"}
    subjects = inp.get("subjects") or GENERIC_VOCAB
    max_subjects = int(inp.get("max_subjects", 6))
    box_threshold = float(inp.get("box_threshold", 0.30))

    import pipeline
    t_load = time.time()
    result = pipeline.process(image_url, subjects, max_subjects, box_threshold)
    if "layers" not in result:
        return {"error": result.get("error", "processing_failed"), "canvas": result.get("canvas")}
    t_proc = time.time()

    W, H = result["canvas"]
    with tempfile.TemporaryDirectory() as td:
        manifest = {"width": W, "height": H, "layers": []}
        for i, lyr in enumerate(result["layers"]):
            fn = os.path.join(td, f"L{i}.rgba")
            lyr["rgba"].astype(np.uint8).tofile(fn)
            manifest["layers"].append({"name": lyr["name"], "w": W, "h": H, "file": fn})
        mpath = os.path.join(td, "manifest.json")
        json.dump(manifest, open(mpath, "w"))
        opath = os.path.join(td, "out.psd")
        subprocess.run(["node", "/app/assemble_psd.mjs", mpath, opath], check=True)
        psd_bytes = os.path.getsize(opath)
        with open(opath, "rb") as f:
            _put(psd_put_url, f.read(), "image/vnd.adobe.photoshop")
    t_done = time.time()

    return {
        "ok": True,
        "canvas": [W, H],
        "subjects": result["subjects"],
        "layer_count": len(result["layers"]),
        "psd_bytes": psd_bytes,
        "timings": {
            "model_load_s": round(t_load - t0, 2),
            "inference_s": round(t_proc - t_load, 2),
            "assemble_upload_s": round(t_done - t_proc, 2),
            "total_s": round(t_done - t0, 2),
        },
    }


runpod.serverless.start({"handler": handler})
