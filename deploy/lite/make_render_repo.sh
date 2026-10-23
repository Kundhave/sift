#!/usr/bin/env bash
# Assemble the small standalone repo for Render at $1 (default /home/kundhave/Projects/sift-render). Run from anywhere.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-/home/kundhave/Projects/sift-render}"
PY="${PYTHON:-$SRC/.venv/bin/python}"
[ -d "$OUT/.git" ] && keep_git=1 || keep_git=0
mkdir -p "$OUT"
find "$OUT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cd "$SRC"
mkdir -p "$OUT/app" "$OUT/src/sift" "$OUT/artifacts/student_onnx_int8" "$OUT/results" "$OUT/experiments/results" "$OUT/data"
cp app/__init__.py "$OUT/app/"
rsync -a --exclude __pycache__ app/backend "$OUT/app/"
rsync -a --exclude node_modules --exclude dist --exclude '*.tsbuildinfo' app/frontend "$OUT/app/"
for m in __init__ serve infer data consolidate diffparse baselines rank; do cp src/sift/$m.py "$OUT/src/sift/"; done
cp artifacts/student_onnx_int8/model.onnx artifacts/student_onnx_int8/meta.json "$OUT/artifacts/student_onnx_int8/"
cp results/*.png results/*.json results/*.csv results/*.md "$OUT/results/" 2>/dev/null || true
cp experiments/results/*.json "$OUT/experiments/results/"
cp data/dataset_statistics.json "$OUT/data/"
PYTHONPATH=src "$PY" deploy/lite/build_lite_data.py "$OUT"      # baselines_lite.joblib, tokenizer.json, lite_samples.json
cp deploy/lite/Dockerfile.lite "$OUT/Dockerfile"
cp deploy/lite/requirements-lite.txt "$OUT/requirements-lite.txt"
cp deploy/lite/render.yaml "$OUT/render.yaml"
cp deploy/lite/README_RENDER.md "$OUT/README.md"
printf '__pycache__/\n*.pyc\nnode_modules/\napp/frontend/dist/\n.venv/\n' > "$OUT/.gitignore"
printf '.git\n**/node_modules\napp/frontend/dist\n__pycache__\n' > "$OUT/.dockerignore"
[ "$keep_git" = 1 ] || git -C "$OUT" init -q -b main
du -sh "$OUT" --exclude=.git | tail -1
