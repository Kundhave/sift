#!/usr/bin/env bash
# Distil the full model into the 22M MiniLM student (tau=4, alpha=0.5), then package + export ONNX/INT8.
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=src
D="--rule_filter --drop_file experiments/results/cl_drop_ids.txt"
[ -f experiments/runs/student/results.json ] || .venv/bin/python -m sift.train --name student --encoder sentence-transformers/all-MiniLM-L6-v2 \
   --max_len 512 --micro_bs 16 --accum 1 --epochs 8 --patience 3 --lr 5e-5 $D --teacher_logits experiments/runs/full/train_logits.npz --tau 4 --alpha 0.5 --log_every 200 > experiments/runs/student.log 2>&1
[ -f experiments/runs/student_nodistill/results.json ] || .venv/bin/python -m sift.train --name student_nodistill --encoder sentence-transformers/all-MiniLM-L6-v2 \
   --max_len 512 --micro_bs 16 --accum 1 --epochs 8 --patience 3 --lr 5e-5 $D --log_every 200 > experiments/runs/student_nodistill.log 2>&1
.venv/bin/python experiments/export_student.py
