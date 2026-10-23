# Sift single-container image (backend + built SPA), CPU only. Build from the repo root:
#   docker build -f deploy/Dockerfile -t sift .
# Hugging Face Spaces (Docker SDK) expects a file named "Dockerfile" at the Space root: deploy_hf.py uploads this file there.
FROM node:22-slim AS web
WORKDIR /web
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY app/frontend/ ./
RUN npx vite build

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 CUDA_VISIBLE_DEVICES="" SIFT_DEVICE=cpu \
    HF_HOME=/opt/hf TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 PORT=7860
RUN useradd -m -u 1000 sift
WORKDIR /app
COPY deploy/requirements-deploy.txt /tmp/req.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.14.1+cpu" -r /tmp/req.txt \
    || pip install --extra-index-url https://download.pytorch.org/whl/cpu torch -r /tmp/req.txt
# Encoder configs/weights that SiftModel loads before the fine-tuned state dict is applied (baked in, so runtime needs no network).
RUN python -c "from huggingface_hub import snapshot_download as s; \
s('answerdotai/ModernBERT-base', allow_patterns=['*.json','*.safetensors']); \
s('sentence-transformers/all-MiniLM-L6-v2', allow_patterns=['*.json','*.txt','*.safetensors'])"
COPY src/ src/
COPY app/__init__.py app/
COPY app/backend/ app/backend/
COPY artifacts/ artifacts/
COPY results/ results/
COPY experiments/results/ experiments/results/
COPY data/dataset_statistics.json data/sift_ia_v1_train.csv data/sift_ia_v1_val.csv data/sift_ia_v1_test.csv data/
COPY --from=web /web/dist app/frontend/dist
RUN chown -R sift:sift /app /opt/hf
USER sift
ENV PYTHONPATH=/app/src HF_HUB_OFFLINE=1
EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
