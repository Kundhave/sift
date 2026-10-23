# Sift Architecture

**Sift** is a real-time verification layer that reads developer intent and ranks diff hunks by alignment. This document describes the pipeline, system components, and data flow.

## System Pipeline

```
Developer Session
        ↓
[Session Capture]
    (intent turns + diff)
        ↓
[Intent Consolidation]
    (drop chit-chat, join turns)
        ↓
[Hunk Segmentation]
    (parse diff, split into hunks)
        ↓
[Model Inference]
    (cross-encoder on [intent, hunk] pairs)
        ↓
[Calibration & Ranking]
    (temperature scaling, confidence, drift type)
        ↓
[Results Display]
    (sorted hunks with explanations & feedback)
        ↓
[User Review & Action]
    (accept / flag / needs-discussion)
```

## Core Components

### 1. Session Capture (`src/sift/data.py`)

**Input:** Developer conversation (chat turns) + a unified diff.

**Output:** Structured intent and hunk data.

- **`load_split(split: str)`** loads train/val/test from CSV
- **`split_turns(intent_turns: str)`** splits multi-turn text by separator
- **`hunk_input(row)`** formats the (intent, hunk) pair for the model

**Data structure:**
```python
{
    "intent_turns": "Turn 1. Turn 2. Turn 3.",  # Multi-turn chat
    "hunk_text": "[CLS] consolidated_intent [SEP] hunk_code",  # Tokenized later
    "label": True/False,  # Aligned or misaligned
    "drift_type": "Aligned" | "Collateral" | "Scope-Expansion" | "Purpose-Drift",
    "session_id": "...",
    "source_repository": "..."
}
```

### 2. Intent Consolidation (`src/sift/consolidate.py`)

**Goal:** Drop noise and join turns into a single consolidated intent.

**Noise types:**
- **Chit-chat:** greetings, off-topic remarks
- **Retracted requests:** "actually, never mind"
- **Repeated turns:** duplicates
- **Truncation:** turns exceeding max tokens

**Algorithm:**
1. Split multi-turn text by separator (TURN_SEP)
2. Filter turns using heuristic rules (regex on chit-chat patterns, length checks)
3. Join remaining turns with `.` if missing end punctuation

**Output:** Single consolidated-intent string, plus a `ConsolidationResult` with:
- `consolidated`: the joined text
- `turns_dropped`: count of filtered turns
- `noise_types`: list of detected noise categories

**Example:**
```python
input: "Hi there. Can you fix the bug in login.py? Actually, make it cleaner. OK thanks."
output: "Can you fix the bug in login.py. Make it cleaner."
```

**Configuration:** `CONSOLIDATION_RULES` in consolidate.py (extendable regex patterns).

### 3. Hunk Segmentation (`src/sift/diffparse.py`)

**Goal:** Parse a unified diff into individual hunks.

**Input:** Raw unified diff text (git diff format).

**Output:** List of hunks with metadata.

**Features:**
- Handles multi-file diffs
- Detects renames, mode changes, binary files
- Tracks old/new filenames and paths
- Robust to malformed `@@` headers (raises `DiffParseError`)

**Functions:**
- **`parse_diff(diff_text, strict=False)`** → list of hunk strings
- **`parse_diff_ex(diff_text, strict=False)`** → detailed hunk objects with filename, status, line ranges

**Hunk format:**
```
@@@ -old_start,old_count +new_start,new_count @@@
 context line
-removed line
+added line
 context line
```

### 4. Model Inference (`src/sift/model.py` + `src/sift/serve.py`)

**Goal:** Score each (intent, hunk) pair and predict alignment, drift type, and confidence.

**Architecture:**
- **Encoder:** ModernBERT-base (149M params, cross-encoder)
- **Input:** `[CLS] consolidated_intent [SEP] hunk_text` (max length 512 tokens)
- **Heads (multi-task):**
  1. **Alignment head:** Binary (aligned / misaligned)
  2. **Drift type head:** 3-way classifier on misaligned hunks (Collateral, Scope-Expansion, Purpose-Drift)
  3. **Calibration head:** Scalar confidence

**Loss function:**
```
L = 0.6 L_align + 0.3 L_drift + 0.1 L_cal

L_align = focal_loss(γ=2)  # For ~78:22 class imbalance
L_drift = cross_entropy  # Only on misaligned examples
L_cal = MSE(confidence_logit - 0|1)  # Calibration
```

**Training configuration (Phase 3, 6 GB VRAM):**
- Optimizer: AdamW (lr 2e-5, wd 0.01)
- Batch: 8 micro × 2 accumulation = 16 effective
- Dtype: bf16 autocast
- Checkpointing: gradient checkpointing enabled
- Epochs: up to 6, early stopping (patience 2) on val macro-F1
- Max length: 512 tokens

**Serving (`src/sift/serve.py`):**
- `SiftScorer` class: lazy-loads models, caches predictions
- Registry: full model, student, INT8, baselines (B4 TF-IDF, B5 logistic regression)
- Per-model inference with automatic device placement (CPU/GPU)

### 5. Calibration & Ranking (`src/sift/calibrate.py` + `src/sift/rank.py`)

#### Temperature Scaling

**Goal:** Make predicted probabilities match empirical frequencies.

**Process:**
1. Fit a single temperature parameter T on the validation set
2. Apply `P_calibrated = σ(logits / T)` at inference time
3. Report ECE (Expected Calibration Error) before and after

#### Denoising (`src/sift/denoise.py`)

Remove noisy training examples before training:
- **Rule-based:** filter non-semantic hunks (whitespace, lockfiles, version bumps) and exact duplicates
- **Confident learning** (`cleanlab`): detect and score label errors against the known 569 flips in the training set
- **Co-teaching:** (optional) train two networks on complementary subsets to handle label noise

#### K-Means Clustering (`src/sift/kmeans.py`)

**Goal:** Cluster (intent, hunk) pairs to find similar misaligned hunks.

**Process:**
1. Extract pooled [CLS] embeddings from the model
2. Run K-Means on L2-normalized cosine distance
3. Choose k by silhouette score
4. Build an index with cluster assignment and neighbor search

**Used by:**
- UI: show similar flagged hunks in the hunk details panel
- Hybrid ranking: cluster distance + prior features can improve calibration (small gain in practice)

#### Ranking Metrics

For each test set and tier (random/hard):

- **NDCG@K:** Normalized Discounted Cumulative Gain; measures ranking quality when you review the top k hunks
- **Precision@K:** fraction of top k hunks that are actually misaligned
- **MRR:** Mean Reciprocal Rank; average position of the first misaligned hunk
- **AUC (within-session pairwise):** fraction of (misaligned, aligned) hunk pairs correctly ranked (misaligned higher)

### 6. Post-Training: Distillation & Export

#### Knowledge Distillation

Train a smaller student model (MiniLM-L6, ~22M params) to mimic the full model:
- Temperature τ = 4, mixing α = 0.5
- Loss: `0.5 L_KL(student || teacher) + 0.5 L_CE(student, labels)`

**Targets:** 5–10x smaller, ~0.5–2x faster, <1% F1 drop.

#### INT8 ONNX Export

Convert the student to ONNX with INT8 quantization:
- Targets: <30 MB (achieved ~23 MB), p95 latency <100 ms on CPU
- Quantization: dynamic per-tensor
- Verification: compare INT8 vs fp32 on a small test set; re-fit temperature scaling

### 7. Explainability (`src/sift/explain.py`)

**Method:** Integrated Gradients (IG) with attribution aggregation.

**Process:**
1. Compute IG from a baseline (PAD token) to the input over k steps (24 steps)
2. Sum gradients × input for each token
3. Aggregate to words (by tokenizer decomposition) and diff lines
4. Normalize by layer activation magnitude

**Faithfulness test:** mask top-k attributed tokens (or replace with PAD) and measure the drop in P(misaligned) vs masking k random tokens.

**Performance:**
- Latency: ~500 ms p50, ~1.3 s p95 on a shared GPU (explanation is per-hunk, slow)
- Used in the UI: show on demand for the top 3 hunks per session

### 8. Backend API (`app/backend/main.py`)

**Framework:** FastAPI (Python).

**Key endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze` | POST | Analyze a session (turns + diff or sample ID) |
| `/api/samples` | GET | List curated test-set examples |
| `/api/samples/{id}` | GET | Get a sample with ground truth (shown after analysis) |
| `/api/models` | GET | List available models (full, student, baselines) |
| `/api/results/*` | GET | All JSON result files (evaluated lazily) |
| `/api/dataset/stats` | GET | Dataset split sizes and noise counts |
| `/api/health` | GET | Health check |
| `/api/tests` | GET | Latest test and coverage summary |
| `/docs` | GET | OpenAPI documentation (Swagger UI) |

**Model registry:**
- `full`: Full model (artifacts/full)
- `student`: Distilled MiniLM (artifacts/student)
- `student_onnx_fp32`, `student_int8`: Quantized exports
- `b4`, `b5`: TF-IDF and logistic regression baselines (fit on train on first use)
- Custom: any model with `artifacts/*/meta.json`

**Limits:**
- Max 50 conversation turns
- Max 5,000 chars per turn
- Max 1,000,000 diff chars
- Max 200 hunks per diff

### 9. Frontend (`app/frontend/`)

**Stack:** React 18 + Vite 5 + TypeScript + Tailwind 3 + Recharts 2.

**Six pages:**

1. **Review Workspace** (main screen)
   - Left: Intent panel (paste turns, show dropped noise)
   - Center: Ranked hunks as cards (misalignment risk, confidence, drift type, code)
   - Right: Selected hunk details (token attributions, similar flagged hunks, actions)
   - Controls: Sample loader, file upload, model switcher, threshold slider

2. **Results Dashboard**
   - Master comparison table (all models, sortable by metric)
   - Graphs: training curves, PR/ROC curves, NDCG, calibration, latency
   - Per-tier toggles, limitations box

3. **Experiments Page**
   - Training settings, ablation rows, noise-robustness curves
   - K-Means cluster visualization (t-SNE)
   - GAN baseline comparison

4. **Dataset Explorer**
   - Split sizes, noise type distribution
   - Sample browser with filtering

5. **Testing Page**
   - Test results by category
   - Code coverage
   - Defect register

6. **About**
   - Architecture diagram, problem statement
   - Honest novelty and limitations
   - SDG relevance

**Key features:**
- **No fake data:** all numbers come from real result files or live API calls
- **Theme support:** dark/light mode (CSS variables, localStorage)
- **Responsive:** desktop and mobile layouts
- **Interactivity:** threshold slider, sample picker, model switcher
- **Diff viewer:** syntax-highlighted hunks with line-level attribution

**Theme tokens:** CSS variables in `index.css` (swap for your brand).

## Data Flow Example

### Typical user workflow:

```
1. User logs in to Sift UI
   
2. Pastes intent turns and diff, or loads a test sample
   POST /api/analyze
   └─ Backend:
      ├─ consolidate_intent(turns) → consolidated text
      ├─ parse_diff_ex(diff) → hunks
      ├─ for each hunk:
      │  ├─ tokenize([consolidated, hunk])
      │  ├─ forward through SiftScorer (full model)
      │  ├─ get logits, apply temperature scaling
      │  ├─ lookup cluster_id (if k-means exists)
      │  └─ optionally run IG explanation
      └─ rank hunks by misalignment probability
   
3. Backend returns:
   {
     "session_id": "...",
     "consolidated_intent": "...",
     "hunks": [
       {
         "index": 0,
         "filename": "login.py",
         "old_range": "10-20",
         "new_range": "10-25",
         "text": "@@@ -10,10 +10,25 @@@\n ...",
         "aligned_probability": 0.85,
         "confidence": 0.92,
         "drift_type": "Aligned",
         "cluster_id": 2,
         "intent_words": [{"word": "fix", "attribution": 0.4}, ...],
         "hunk_lines": [{"line": "+  print(...)", "attribution": 0.6}, ...],
         "similar_ids": [123, 456]  # Other hunks in the same cluster
       },
       ...
     ]
   }
   
4. UI displays hunks sorted by alignment probability (ascending = most risky first)
   User reviews top hunks, marks as Accept/Flag/Discuss
   
5. (Optional) User exports a review summary or requests model/threshold change
```

## Technology Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Encoder | ModernBERT-base | 149M params, 8k context, modern (RoPE, GeGLU); fits 6 GB with optimizations |
| Cross-encoder | Yes (not siamese) | Hunks and intent are inherently paired; joint input is more expressive |
| Loss | Focal loss + multi-task | Focal addresses 78:22 imbalance; multi-task shares representations |
| Distillation | MiniLM-L6 student | Lightweight, ~22M params, >10x smaller than teacher |
| Export | INT8 ONNX | <30 MB, CPU inference, no runtime dependencies |
| Framework (backend) | FastAPI | Async, auto-docs, easy model registration |
| Framework (frontend) | React + Vite | Industry standard, fast builds, good tooling, TypeScript support |
| Chart library | Recharts | Stable, lightweight, built-in themes |
| Diff viewer | highlight.js | Syntax highlighting for 9 languages, no heavy dependencies |

## Extensibility

### Adding a New Model

1. Train or import a model in `experiments/` as a script
2. Export to `artifacts/model_name/` with:
   - `model.safetensors` (or `.bin` for torch)
   - `config.json` (training params, max_len, etc.)
   - `meta.json` (registry metadata)
   - Optional: `kmeans.pkl` (cluster index)
3. Register in `src/sift/serve.py` `MODEL_REGISTRY`
4. Backend will auto-load and list in `/api/models`
5. Frontend model switcher will include it

### Adding a New Ablation

1. Run `experiments/ablation.py <ablation_name>` (or write a script)
2. Output evaluation to `results/eval_abl_<name>.json`
3. Frontend Results page discovers and displays it automatically

### Adding Tests

1. Write pytest tests in `tests/` (following existing patterns)
2. Run `make e2e` to include in the suite
3. Coverage is automatically reported

## Glossary

| Term | Definition |
|------|-----------|
| **Intent** | The consolidated developer request (e.g., "fix the login bug") |
| **Hunk** | A contiguous code change (lines added/removed in one file) |
| **Session** | A development session: one intent + a diff with multiple hunks |
| **Aligned** | A hunk that matches the stated intent |
| **Misaligned** | A hunk that does not match the intent (3 types: Collateral, Scope-Expansion, Purpose-Drift) |
| **Drift** | Deviation from the stated intent (the type of misalignment) |
| **Calibration** | Matching predicted probabilities to real frequencies (e.g., 70% confident = 70% accurate) |
| **ECE** | Expected Calibration Error; how far calibrated probabilities are from ground truth |
| **NDCG@K** | Ranking metric: quality of the top-K hunks when reviewed in order |
| **Confidence** | Model's self-reported uncertainty in its prediction |
| **Attribution** | Which tokens/lines contributed most to a prediction (XAI) |

---

For implementation details, see the docstrings in `src/sift/*.py` and the test suite in `tests/`.
