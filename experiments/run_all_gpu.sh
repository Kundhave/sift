#!/usr/bin/env bash
# Sequential GPU queue: full model + ablation variants (RTX 4050 6GB: L=512, micro-batch 8 x accum 2, bf16, grad-ckpt).
set -u
cd "$(dirname "$0")/.."; export PYTHONPATH=src
C="--max_len 512 --micro_bs 8 --accum 2 --epochs 6 --patience 2 --lr 2e-5 --grad_ckpt --log_every 200"
run() { n=$1; shift; [ -f experiments/runs/$n/results.json ] && return; .venv/bin/python -m sift.train --name $n $C "$@" > experiments/runs/$n.log 2>&1; }
D="--rule_filter --drop_file experiments/results/cl_drop_ids.txt"
run full $D --dump_train_logits
run abl_no_denoise
run abl_hunk_only $D --input_mode hunk_only
run abl_intent_only $D --input_mode intent_only
run abl_no_consolidation $D --intent_source raw
run abl_no_focal $D --ce
run abl_no_multitask $D --no_multitask
run abl_rules_only --rule_filter
