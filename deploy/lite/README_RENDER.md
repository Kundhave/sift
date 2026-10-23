# Sift (lite deployment)

Sift checks, hunk by hunk, whether an AI coding agent's diff matches what the developer asked for. This repo is the small
build served on Render's free web service (512 MB RAM, CPU only). It serves the website, all measured results and graphs,
curated test-set samples with ground truth, and live inference with the INT8 ONNX student (22 MB) and the baselines B4/B5.

Not served here: the main ModernBERT model (PyTorch, 570 MB checkpoint), k-means similar-hunk lookup and Integrated Gradients
explanations (all need PyTorch). They appear as unavailable; the full project runs them locally with `make demo`.
Latency numbers shown for models belong to the machine that ran them; the free instance has 0.1 CPU and sleeps when idle,
so the first request after a pause takes about a minute.

## Deploy
1. Push this folder to GitHub (`deploy/lite/push_github.sh` in the main project does it with `GITHUB_USER`/`GITHUB_TOKEN`).
2. Render: New -> Blueprint -> select the repo (reads `render.yaml`). Or New -> Web Service -> Docker, plan Free,
   health check path `/api/health`.

## Run locally
    docker build -t sift-lite . && docker run --rm -p 10000:10000 sift-lite
