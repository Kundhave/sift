#!/usr/bin/env bash
# One-shot deploy to a Hugging Face Space. Requires: HF_USERNAME=<you> and a prior `hf auth login`.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_USERNAME:?set HF_USERNAME=<your Hugging Face username>}"
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
$PY -m pip install -q huggingface_hub
$PY deploy/deploy_hf.py --name "${SPACE_NAME:-sift}" "$@"
cat deploy/DEPLOY_URL.txt
