# psd-layer-worker

RunPod serverless worker that turns one image into a **layered PSD**: open-vocab detection
(GroundingDINO) → box-prompted instance masks (SAM) → high-quality alpha (BiRefNet) →
background inpaint (LaMa) → PSD assembly (ag-psd, same lib cv.cm ships). Layers are pixel-aligned
to the input, so stacking them reconstructs the original image.

Built for cv.cm's "分层 PSD（N 积分）" button. See memory `project_cvcm_layered_psd`.

## Job I/O

```jsonc
// input
{
  "image_url": "https://s3.cv.cm/....png",   // required
  "subjects": ["cat", "dog"],                 // from cv.cm GLM vision; omit -> generic vocab
  "psd_put_url": "https://s3...signed-PUT",   // required: assembled .psd is uploaded here
  "max_subjects": 6,
  "box_threshold": 0.30
}
// output
{ "ok": true, "canvas": [2496,1664], "subjects": [{"name":"cat","score":0.57}],
  "layer_count": 3, "psd_bytes": 12345678, "timings": {"inference_s": 6.2, "total_s": 8.1} }
```

## Deploy (RunPod serverless, GitHub-build)

Prereqs: **RunPod balance > $0** (no post-pay), GitHub App connected to RunPod (one-time OAuth).

1. Push this repo to `github.com/mantoufan/psd-layer-worker`.
2. RunPod console → Serverless → New Endpoint → **Custom / GitHub Repo** → pick this repo.
   Docker build context = repo root (uses `Dockerfile`).
3. GPU: **24GB** (RTX 3090 / A5000) is plenty; enable a 48GB fallback (A40) in the GPU list.
4. Workers: Active(min) = 0 (scale-to-zero → $0 idle), Max = 1–3. Idle timeout 5–10s.
   Execution timeout ≥ 120s. FlashBoot on.
5. After creation, note the endpoint id. Set it as cv.cm Pages secret `PSD_LAYER_ENDPOINT_ID`
   (and reuse `RUNPOD_API_KEY`).

Weights are **baked into the image** (~2GB), so no network volume is needed and cold start does
not re-download models.

## Call

```bash
curl -s https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"image_url":"https://...","subjects":["cat","dog"],"psd_put_url":"https://...signed"}}'
```

## Notes / gotchas
- MPS (Apple) dead-locks SAM box post-processing — this runs CUDA only.
- BiRefNet needs `trust_remote_code=True`.
- The `.psd` is PUT to the presigned URL, never returned inline (it is large).
