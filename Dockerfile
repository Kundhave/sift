# Sift LITE image for Render's free web service (512 MB RAM, 0.1 CPU). No PyTorch.
# Served: React site, results, curated samples, live INT8 ONNX student + baselines B4/B5.
# Copied to the root of the standalone repo as "Dockerfile" by deploy/lite/make_render_repo.sh.
FROM node:22-slim AS web
WORKDIR /web
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY app/frontend/ ./
RUN npx vite build

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 CUDA_VISIBLE_DEVICES="" SIFT_DEVICE=cpu \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    SIFT_MODELS=student_int8,b4,b5 SIFT_LISTED_UNAVAILABLE=full,student,student_onnx_fp32 \
    SIFT_LOW_MEM=1 MALLOC_ARENA_MAX=2 SIFT_MAX_HUNKS=20 \
    SIFT_STATIC_DIR=/app/app/frontend/dist PYTHONPATH=/app/src PORT=10000
RUN useradd -m -u 1000 sift
WORKDIR /app
COPY requirements-lite.txt /tmp/req.txt
RUN pip install -r /tmp/req.txt
COPY src/ src/
COPY app/__init__.py app/
COPY app/backend/ app/backend/
COPY artifacts/ artifacts/
COPY results/ results/
COPY experiments/results/ experiments/results/
COPY data/ data/
COPY --from=web /web/dist app/frontend/dist
RUN chown -R sift:sift /app
USER sift
EXPOSE 10000
# one worker, one process: the 512 MB limit does not allow more
CMD ["sh", "-c", "exec uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
