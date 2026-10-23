#!/usr/bin/env bash
# Waits for the ablation queue (run_all_gpu.sh, pid 7351) to exit, then trains the DL baselines and the small Optuna study.
# Same 6 GB recipe as the main model: L=512, micro-batch 8 x accum 2, bf16, gradient checkpointing, denoised train rows.
set -u
cd "$(dirname "$0")/.."; export PYTHONPATH=src
while kill -0 ${WAIT_PID:-7351} 2>/dev/null; do sleep 60; done
PY=.venv/bin/python
C="--max_len 512 --micro_bs 8 --accum 2 --epochs 6 --patience 2 --grad_ckpt --log_every 200"
D="--rule_filter --drop_file experiments/results/cl_drop_ids.txt"
tr() { n=$1; shift; [ -f experiments/runs/$n/results.json ] && return; $PY -m sift.train --name $n $C $D "$@" > experiments/runs/$n.log 2>&1; }
[ -f experiments/runs/bilstm/results.json ] || $PY experiments/run_bilstm.py --name bilstm > experiments/runs/bilstm.log 2>&1
tr minilm_ce --encoder sentence-transformers/all-MiniLM-L6-v2 --lr 5e-5 --micro_bs 16 --accum 1
tr codebert --encoder microsoft/codebert-base --lr 2e-5
[ -f results/hpo.json ] || $PY experiments/hpo.py --trials 8 > experiments/runs/hpo.log 2>&1
