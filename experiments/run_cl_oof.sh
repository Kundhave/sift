#!/usr/bin/env bash
# 3-fold out-of-fold probabilities with the MiniLM cross-encoder (input to confident learning).
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=src
for f in 0 1 2; do
  .venv/bin/python -m sift.train --name cl_oof_fold$f --encoder sentence-transformers/all-MiniLM-L6-v2 \
     --max_len 512 --micro_bs 16 --accum 1 --epochs 8 --patience 3 --lr 5e-5 --nfolds 3 --fold $f --log_every 200 > experiments/runs/cl_oof_fold$f.log 2>&1
done
