#!/usr/bin/env bash
# Reduced-size GPU plan (6 GB shared with abl_no_multitask). Lane A: small jobs now; ModernBERT/CodeBERT jobs after pid 75486 exits.
set -u
cd "$(dirname "$0")/.."; export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; R=experiments/runs
C="--max_len 256 --grad_ckpt --patience 2 --log_every 200"
D="--rule_filter --drop_file experiments/results/cl_drop_ids.txt"
ev() { [ -f $R/$1/results.json ] && $PY experiments/evaluate.py --run $1 > $R/$1.eval.log 2>&1; }
[ -f $R/bilstm/results.json ] || $PY experiments/run_bilstm.py --name bilstm > $R/bilstm.log 2>&1; ev bilstm
[ -f $R/minilm_ce/results.json ] || $PY -m sift.train --name minilm_ce $C $D --encoder sentence-transformers/all-MiniLM-L6-v2 --lr 5e-5 --micro_bs 16 --accum 1 --epochs 2 > $R/minilm_ce.log 2>&1; ev minilm_ce
while kill -0 75486 2>/dev/null; do sleep 30; done
[ -f $R/codebert/results.json ] || $PY -m sift.train --name codebert $C $D --encoder microsoft/codebert-base --lr 2e-5 --micro_bs 8 --accum 2 --epochs 1 --train_subset 4000 > $R/codebert.log 2>&1; ev codebert
[ -f results/hpo_small.json ] || $PY experiments/hpo.py --name hpo_small --trials 6 --n_train 1600 --n_val 800 --epochs 2 --lens 256,384 > $R/hpo_small.log 2>&1
[ -f results/noise_robustness.json ] || $PY experiments/noise_robustness.py > $R/noise.log 2>&1
echo PLAN_DONE > $R/plan_done.txt
