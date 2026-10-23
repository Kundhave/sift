#!/usr/bin/env bash
# Regenerate every results JSON on a GPU machine, then rebuild the Lab Evaluation 2 PDFs.
# It only chains the EXISTING scripts (nothing here is new training code). It was syntax-checked with `bash -n` but NOT executed
# in the build session. Expect several hours on a 6 GB GPU (main model ~28 min, six ablations, baselines, student, XAI).
#   bash run_gpu_experiments.sh                 # everything
#   SKIP_TRAIN=1 bash run_gpu_experiments.sh    # reuse experiments/runs/* checkpoints, only re-evaluate and rebuild reports
# Each step is skipped by the underlying script when its results.json already exists (delete the run folder to retrain).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
PY=.venv/bin/python
mkdir -p experiments/runs results experiments/results reports/figures CSE23037
step() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

step "0. verify dataset counts and re-run the non-neural baselines B1-B5 (CPU)"
$PY experiments/verify_data.py
$PY experiments/run_baselines.py
$PY experiments/baseline_scores.py
$PY experiments/consolidation_eval.py

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  step "1. denoising inputs: 3-fold out-of-fold MiniLM probabilities, then rules + confident learning (writes cl_drop_ids.txt)"
  bash experiments/run_cl_oof.sh
  $PY experiments/run_denoise.py

  step "2. main model (full) and the ablation variants (run_all_gpu.sh: L=512, 8x2 batch, bf16, grad-ckpt, one seed)"
  bash experiments/run_all_gpu.sh

  step "3. distilled MiniLM student, no-distillation student, ONNX fp32 + INT8 export"
  bash experiments/run_student.sh

  step "4. comparison models (reduced protocol), small Optuna study, noise-robustness sweep (run_gpu_plan.sh)"
  bash experiments/run_gpu_plan.sh
fi

step "5. evaluation of every run on the test split (threshold from validation); fresh GPU inference for the main model"
$PY experiments/evaluate.py --run full --infer
for r in abl_hunk_only abl_intent_only abl_no_consolidation abl_no_focal abl_no_multitask abl_no_denoise abl_rules_only student student_nodistill bilstm minilm_ce codebert; do
  if [ -f experiments/runs/$r/results.json ]; then $PY experiments/evaluate.py --run $r; else echo "skip $r (no results.json)"; fi
done

step "6. calibration, INT8 benchmark, pooled embeddings + k-means, confident-learning and GAN plots"
$PY experiments/calibration.py
$PY experiments/int8_bench.py
$PY experiments/extract_embeddings.py
$PY experiments/kmeans.py

step "7. GAN-style baseline on frozen embeddings (CPU), then its comparison plots"
CUDA_VISIBLE_DEVICES='' $PY experiments/run_gan.py
$PY experiments/plots_denoise_gan.py

step "8. latency of the main model: CPU (idle machine) and GPU"
CUDA_VISIBLE_DEVICES='' $PY experiments/latency_full.py --device cpu
$PY experiments/latency_full.py --device cuda

step "9. XAI: Integrated Gradients, faithfulness test, worked cases, explanation latency"
$PY experiments/xai.py

step "10. master tables and graphs (results/*.csv, results/*.png, SUMMARY.md block)"
$PY experiments/make_tables.py

step "11. test suite and coverage (for the Testing deliverable; the e2e suite needs the built frontend, see Makefile 'e2e')"
CUDA_VISIBLE_DEVICES='' $PY -m pytest tests -q -p no:cacheprovider --junitxml=reports/junit_tests.xml || echo "unit tests reported failures (xfail defects are expected)"
# $PY experiments/build_test_summary.py reports/junit_tests.xml reports/junit_e2e.xml   # needs both XML files

step "12. build the Lab Evaluation 2 PDFs into CSE23037/ (all numbers come from the JSON above)"
$PY experiments/build_reports.py
echo; echo "done: CSE23037/*.pdf, reports/number_trace.json"
