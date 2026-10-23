# Deploying Sift (one container, CPU only)

The image serves the React app and the FastAPI backend on one port (`$PORT`, default 7860). Models run locally in the container; no external API is called.
Nothing has been deployed yet: no credentials exist on the build machine. Image size and Docker build were not measured (the Docker daemon is not accessible to this user); see "Verified locally" below.

## Option A: Hugging Face Spaces (free CPU, Docker SDK) - recommended

1. Create a free account at https://huggingface.co/join and note your username.
2. Create a token with **Write** access: https://huggingface.co/settings/tokens -> New token -> type "Write".
3. Log in on this machine (paste the token when asked):
   ```bash
   cd ~/Projects/Sift
   .venv/bin/pip install -q huggingface_hub
   .venv/bin/hf auth login          # older CLI: .venv/bin/huggingface-cli login
   ```
4. Deploy with one command (replace the username):
   ```bash
   HF_USERNAME=your-username ./deploy/deploy_hf.sh
   ```
   It creates the Space `your-username/sift` (Docker SDK, port 7860), uploads the needed files (about 0.8 GB; the 570 MB ModernBERT checkpoint goes through the Hub's large-file storage automatically) and writes the URL to `deploy/DEPLOY_URL.txt`.
5. Open https://huggingface.co/spaces/your-username/sift and watch the Build logs (first build roughly 10-20 min). The app is then at https://your-username-sift.hf.space.

Notes: free CPU Spaces have 16 GB RAM and 2 vCPU, which is enough for all models; they sleep when idle and wake on the next visit. The main model on 2 vCPU is slow (seconds per hunk); use the INT8 student in the model switcher for a fast demo. To restrict the server to light models set a Space variable `SIFT_MODELS=student_int8,b4,b5` (Settings -> Variables).

## Option B: Render free web service (from a GitHub repo, same Dockerfile)

1. Push the repo to GitHub, **including** `artifacts/` (use Git LFS for `artifacts/full/model.pt`, or drop the main model, see below), `results/`, `experiments/results/` (without `*.npz`) and the three `data/sift_ia_v1_*.csv` plus `data/dataset_statistics.json` (these are gitignored by default; force-add).
2. In Render: New -> Web Service -> connect the repo -> Runtime **Docker** -> Dockerfile path `deploy/Dockerfile` -> Instance type Free.
3. Environment variable `SIFT_MODELS=student_int8,b4,b5` (free tier RAM is 512 MB, too little for ModernBERT; the INT8 student is 22 MB). Render sets `$PORT` itself; the container honours it. To keep the image small also delete `artifacts/full` and `artifacts/student` before pushing.
4. Free instances sleep after idle time; first request after sleep is slow.

## Local container run (if Docker is available to you)
```bash
docker build -f deploy/Dockerfile -t sift .
docker run --rm -p 7860:7860 sift          # add -e SIFT_MODELS=student_int8,b4,b5 to restrict models
```

## Verified locally
See DECISIONS.md item "Single-container deployment" for what was actually run and measured.
