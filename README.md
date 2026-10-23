# Sift: Real-Time Verification for AI Coding Agents

**Sift** is a local, real-time verification layer for AI coding agents (Claude Code, Cursor, Copilot). It reads the intent a developer expressed and classifies each hunk of a diff as **Aligned** or **Misaligned**, then ranks hunks by misalignment likelihood so a human reviews the riskiest ones first.

Misaligned hunks carry one of three drift types: **Collateral** (unrequested side effects), **Scope-Expansion** (unrequested features), or **Purpose-Drift** (serving a different purpose).

## Honest Novelty and Limitations

**What is novel:** the hunk-level, multi-turn intent task framing; the pipeline combining off-the-shelf models with this framing; and the **Sift-IA v1.0 dataset** (18,000 hunk-intent pairs, 6,111 sessions, 15 open-source repos). **Not novel:** the encoder (ModernBERT) and training methods are existing work.

**Dataset and modeling limitations** (to state in results):
- Misaligned hunks are *constructed* (donor hunks inserted into sessions), not collected from real agent errors. A model could learn construction artefacts; the hard tier mitigates but does not remove this.
- Seed commits may be tangled (multiple changes bundled).
- A donor hunk may coincidentally fit the intent.
- Languages: Python, JavaScript, TypeScript only.
- No human validation of labels yet (v1.0 uses construction-determined labels).
- Test set is repository-disjoint but still from open-source repositories.

**Hardware note:** Trained and tested on **NVIDIA RTX 4050 Laptop (6 GB VRAM)** with gradient checkpointing, bf16 autocast, and reduced max_length (512 tokens). Results are specific to this machine; latency and size are measured here.

## Quick Start

### Setup

```bash
# Create and activate virtual environment (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Unzip the dataset (if not already done)
unzip -q data/Sift-IA-v1_dataset.zip -d data/

# Verify you have the key data files
ls data/train.csv data/validation.csv data/test.csv
```

### Run the App

```bash
# One-command demo: builds frontend, starts API + UI (CPU only)
make demo

# Product site:       http://127.0.0.1:5173/      (workspace at /app)
# Developer console:  http://127.0.0.1:5173/dev   (results, experiments, dataset, testing, related work)
# API docs at http://127.0.0.1:8000/docs
# Stop with: make stop
```

### Run Backend Only

```bash
make backend
# Starts on http://127.0.0.1:8000
# Set SIFT_DEVICE=gpu if you have a GPU; defaults to CPU
```

### Run Frontend Only

```bash
make frontend
npm run dev  # or: npm run preview for production build
```

### Run Tests

```bash
# Unit tests + API tests (CPU only)
make e2e

# View coverage
cat .coverage | head -100  # or open htmlcov/index.html if generated
```

## Repository Layout

```
Sift/
  README.md              This file
  CLAUDE.md              Implementation brief and task definition
  PROGRESS.md            Phase completion status and owners
  DECISIONS.md           Key decisions, alternatives, and findings
  Makefile               Build and run targets (demo, backend, frontend, e2e, stop)
  
  data/                  Sift-IA v1.0 dataset (train/val/test CSVs, metadata)
                         See data/README.md for dataset documentation
  
  src/sift/              Core Python modules
    data.py              Data loaders, tokenisation, consolidation
    diffparse.py         Diff parser (handles renames, binary files, hunk segmentation)
    consolidate.py       Intent consolidation (drop chit-chat, join turns)
    model.py             ModernBERT cross-encoder with alignment, drift-type, calibration heads
    train.py             Training loop, early stopping, checkpointing
    losses.py            Focal loss, multi-task weighted loss
    calibrate.py         Temperature scaling, ECE computation
    denoise.py           Rule-based and confident-learning denoising
    distill.py           Knowledge distillation, INT8 ONNX export
    kmeans.py            K-Means clustering and index
    gan_baseline.py      Adversarially trained baseline
    explain.py           Integrated Gradients XAI
    rank.py              Hunk ranking and ranking metrics (NDCG, P@K, MRR)
    serve.py             Model serving logic (used by the backend)
    infer.py             Inference entry point
    
  app/
    backend/
      main.py            FastAPI server, model registry, endpoints
    frontend/
      src/               React components (Vite + TypeScript + Tailwind)
        pages/           Six routes (Review, Results, Experiments, Dataset, Testing, About)
        components/      DiffViewer, Charts, Tables
        api.ts           API client
    
  experiments/           Standalone scripts
    baselines.py         Non-neural baselines (TF-IDF, logistic regression)
    run_baselines_gpu.sh Launcher for GPU jobs
    hpo.py               Optuna hyperparameter optimization
    calibration.py       Temperature scaling
    kmeans.py            K-Means fitting
    xai.py               Integrated Gradients and faithfulness tests
    plots_denoise_gan.py Result plotting
    
  results/
    SUMMARY.md           Plain-language results and measurements
    eval_full.json       Main model test evaluation (NDCG, F1, AUC per tier)
    calibration.json     Temperature scaling (ECE before/after)
    xai.json             XAI faithfulness metrics
    latency_size_onnx.json  INT8 and ONNX benchmarks
    test_summary.json    Test suite results and coverage
    gan_vs_main.json     GAN baseline comparison
    kmeans.json          K-Means clustering metrics
    *.png                Charts (calibration, confusion, ROC, PR, latency, clusters, etc.)
    
  checkpoints/           Model checkpoints (best and final per run, git-ignored)
  
  artifacts/
    full/                Full model (PackedModel format, model+config+kmeans)
    student/             MiniLM student (knowledge distillation)
    student_int8/        INT8 ONNX student (CPU deployment)
    
  tests/                 pytest suite
    test_smoke.py        Basic sanity checks
    test_whitebox.py     Tensor shapes, head outputs, loss
    test_data_pipeline.py  Data loading, split leakage
    test_diffparse_consolidate.py  Diff parsing and intent consolidation
    test_blackbox.py     API response shapes and values
    test_metamorphic.py   Metamorphic relations (identifier rename, whitespace, etc.)
    test_redteam.py      Adversarial inputs (misleading comments, injection, unicode)
    test_api.py          Backend API endpoint tests
    test_backtoback.py    Full model vs student equivalence
    test_api_partitions.py  API response partitions and boundaries
    
  e2e/                   Playwright end-to-end tests (UI)
  docs/                  Documentation
    ARCHITECTURE.md      System pipeline and design
    screenshots/         Visual QA screenshots (light and dark theme)
    
  refs/                  Reference folder (read-only)
    - Sift-IA-v1_README.md  Dataset documentation
    - Sift-IA-v1_dataset.zip  Original dataset zip
    - Test_and_Evaluation_of_Artificial_Intelligence_Models_Framework_2.pdf
    - CSE23052/ (reference PDFs, layout only, read the DECISIONS.md for scope)
```

## How to Run

### Demo (one command)

```bash
make demo
```

This builds the frontend, starts the FastAPI backend on :8000 (CPU only), starts Vite on :5173, waits for the API to be ready, and prints both URLs: the product site at `/` (workspace at `/app`) and the developer console at `/dev`. Old routes (`/results`, `/review`, ...) redirect to the new ones.

### Backend Only

```bash
PYTHONPATH=src .venv/bin/uvicorn app.backend.main:app --port 8000
```

To force CPU (no GPU):
```bash
CUDA_VISIBLE_DEVICES='' SIFT_DEVICE=cpu PYTHONPATH=src .venv/bin/uvicorn app.backend.main:app --port 8000
```

To use a GPU (if available):
```bash
SIFT_DEVICE=gpu PYTHONPATH=src .venv/bin/uvicorn app.backend.main:app --port 8000
```

API docs: `http://127.0.0.1:8000/docs`

### Frontend Only

```bash
cd app/frontend
npm run dev    # Development server on :5173 (hot reload)
npm run build  # Production build
npm run preview  # Preview production build on :5173
```

The frontend proxies `/api` to `http://127.0.0.1:8000` by default. Override with the `SIFT_API` environment variable:

```bash
SIFT_API=http://your-api:8000 npm run dev
```

### Test Suite

Unit and integration tests:
```bash
make e2e
```

This runs CPU-only (CUDA not used), runs pytest on the `e2e` directory, and reports results.

To run a specific test:
```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_whitebox.py -v
```

With coverage:
```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ --cov=src/sift --cov=app.backend --cov-report=html
```

## Results and Metrics

All metrics are measured and logged to JSON files in `results/`. See:
- **`results/SUMMARY.md`** for a plain-language overview
- **`results/eval_full.json`** for the main model's test evaluation (macro-F1, NDCG@5, AUC by tier)
- **`results/test_summary.json`** for test suite pass/fail and coverage
- **`results/latency_size_onnx.json`** for INT8 and ONNX benchmarks
- **`results/xai.json`** for XAI faithfulness metrics
- **`results/gan_vs_main.json`** for GAN baseline comparison

### Known Data Discrepancies

The released dataset differs slightly from the CLAUDE.md brief. See `experiments/results/DISCREPANCIES.md`:

| Item | Brief | Actual |
|------|-------|--------|
| Hunks by language | Python 11,900 / JS 5,076 / TS 1,024 | Python 10,205 / JS 6,726 / TS 1,069 |
| TypeScript in test split | Implied present | **None** (only in validation) |
| Rankable test sessions | 447 | 456 |

## Key Decisions

- **Python 3.12.15 venv** (system Python 3.14.7 has unreliable ML wheels)
- **Max length 512 tokens** (dataset median ~132, p95 455; fits 6 GB VRAM with bf16, grad checkpointing)
- **No GPU in make demo** (assumes CPU; set `CUDA_VISIBLE_DEVICES=gpu` in the Makefile if you want GPU)
- `make reproduce` re-runs the CPU baselines and make_tables.py; heavy GPU runs are in experiments/run_all_gpu.sh (not re-run here)

For detailed decisions and findings, see `DECISIONS.md`.

## Hardware and Environment

- **CPU:** Intel i7-13650HX
- **GPU:** NVIDIA RTX 4050 Laptop (6 GB VRAM, shared during training)
- **RAM:** 15 GB
- **Python:** 3.12.15 (in `.venv`)
- **Framework:** PyTorch 2.14.1 (CUDA 12.6)

All numbers in this project are specific to this hardware. Latency, memory, and training time may vary significantly on other machines.

## Architecture

For a detailed explanation of the pipeline (session capture → intent consolidation → hunk segmentation → model → calibration/ranking → UI), see `docs/ARCHITECTURE.md`.

## Dataset

The **Sift-IA v1.0 dataset** includes:
- **18,000 hunk-intent pairs** from 6,111 sessions
- **15 open-source repositories** (MIT, BSD-3, Apache-2.0 licensed)
- **Languages:** Python, JavaScript, TypeScript
- **Splits:** repository-level (no repo shared across splits)
  - Train: 12,600 hunks (4,266 sessions)
  - Val: 2,700 hunks (921 sessions)
  - Test: 2,700 hunks (924 sessions)
- **Labels:** construction-determined (no human validation yet)
- **Noise:** 25.2% of rows include intent chit-chat, duplicates, non-semantic changes, or label flips

See `data/README.md` and `experiments/results/DISCREPANCIES.md` for details.

## Testing and Quality

- **178 pytest tests + 10 Playwright e2e tests** across smoke, whitebox, blackbox, data pipeline, metamorphic, red-team, and API partitioning
- **Line coverage:** 69.9% (src/sift + app/backend)
- **Test types:** smoke/readiness, structural, functional, metamorphic relations, adversarial inputs, back-to-back model comparison
- **Known defects:** recorded in DECISIONS.md with severity and mitigation

Run `make e2e` to execute the full suite. Results logged to `results/test_summary.json`.

---

**Student:** Kundhave S (CB.SC.U4CSE23037)  
**Course:** 23CSE473 Neural Networks and Deep Learning  
**University:** Amrita Vishwa Vidyapeetham, Amrita School of Computing
