# SIFT: Full Implementation Prompt for Claude Code

**How to use:** Put this file in `~/Projects/Sift/` as `CLAUDE.md`. The old `CLAUDE.md` (written for the Lab Evaluation 2 PDFs) is the wrong brief for this job, so rename it to `LAB_EVAL_BRIEF.md` and ignore it. Then send Claude Code this one message:

> Read `CLAUDE.md` completely. Implement the entire Sift project on your own, backend, model training, testing and a polished frontend, exactly as specified. Make your own decisions and log them in `DECISIONS.md`. Do not stop to ask me unless something truly blocks you. Keep working until everything in the "Definition of done" section is true, then give me the final summary described at the end.

---

## 0. The job in one paragraph

Build **Sift** end to end: a working, trained, tested system plus a **high-quality frontend**. That means: load and verify the dataset, train the deep-learning model and several baselines, run all experiments (comparison, ablation, noise, k-means, GAN baseline, explainability), calibrate and compress the model, serve it through a backend API, build a polished web UI, write a test suite, and make a results dashboard inside the app so that when you finish I can open one command, see what was trained, how well it works and every result. **I want minimal to no effort from my side.** You train, test, debug and verify everything yourself. My professor will judge the **frontend most heavily**, so it must look and work like a real product.

Not needed: any case-study paper, the Lab Evaluation 2 PDFs or slide decks. (A later task may reuse your outputs.) Do not touch the `CSE23052/` reference folder; it is another team's project and is irrelevant here.

## 1. Who and what

- Student: **Kundhave S, CB.SC.U4CSE23037**, **Team 24, single member**. Course 23CSE473 Neural Networks and Deep Learning, Dr. T. Senthil Kumar, Amrita Vishwa Vidyapeetham (CSE, Amrita School of Computing).
- Project folder: `~/Projects/Sift`. It already contains `Sift-IA-v1_dataset.zip` (unzipped as `Sift-IA-v1_dataset/`), `Sift-IA-v1_README.md`, the framework PDF `Test_and_Evaluation_of_Artificial_Intelligence_Models_Framework_2 ... .pdf` (use it as a testing reference), and the old brief.
- Machine: **RTX 4050 with 6 GB VRAM**, Python 3.14 on the system. **Use a separate venv with Python 3.11 or 3.12** (install with `uv` or `pyenv` if needed) because many ML packages have no 3.14 wheels. Confirm the hardware with `nvidia-smi` and record it.

## 2. What Sift is

A **local verification layer for AI coding agents** (Claude Code, Cursor, Copilot, …). A coding session produces a diff split into **hunks**. The developer's intent is spread over several chat turns, some of which are noise (chit-chat, retracted requests). Sift:

1. **Consolidates** the session's turns into one intent.
2. **Classifies every hunk** as **Aligned** or **Misaligned** with that intent.
3. For misaligned hunks, assigns a **drift type**: **Collateral** (unrequested change next to the requested one), **Scope-Expansion** (unrequested feature or behaviour), **Purpose-Drift** (serves a different purpose).
4. **Ranks hunks** inside the diff so a human reviews the likeliest misaligned ones first, with a **calibrated confidence** and an **explanation** (which words and lines drove the verdict).

Why deep learning: any diff-level scalar (lines changed, files touched) is identical for every hunk in one diff, so it cannot rank hunks within it. Measured: ordering by diff lines equals random (NDCG@5 0.7236). Judging whether one hunk serves a stated intent needs both texts read jointly, which a cross-encoder does. This is a message-code inconsistency style task, not defect prediction.

**Honest novelty:** the encoder and training methods are existing; the contribution is the hunk-level, multi-turn-intent task framing, the pipeline, and the dataset. Never claim a new architecture.

**SDG relevance (default, I can change it):** SDG 9 (Industry, Innovation and Infrastructure) as primary: trustworthy AI-assisted software engineering; SDG 8 (Decent Work and Economic Growth) as secondary: less reviewer burden. Present this on an app page with a plain-language justification.

## 3. The dataset: Sift-IA v1.0 (already built; verify before use)

**First read `Sift-IA-v1_README.md` and the CSV headers. Use the real column names. Do not assume a schema.**

- 18,000 hunk-intent pairs, 6,111 sessions, 8,016 source commits, 15 open-source repos (MIT/BSD-3/Apache-2.0). 11,900 Python, 5,076 JavaScript, 1,024 TypeScript hunks. Median 2 hunks per session (max 14); median ~132 tokens per pair, p95 455, max 4,031.
- **Repository-level splits:**

| Split | Repos | Sessions | Hunks | Aligned | Misaligned |
|---|---|---|---|---|---|
| Train | requests, flask, click, rich, black, express, fastify | 4,266 | 12,600 | 9,828 | 2,772 |
| Val | typer, sqlmodel, got, koa, chalk | 921 | 2,700 | 2,106 | 594 |
| Test | httpx, attrs, axios | 924 | 2,700 | 2,106 | 594 |

- Misaligned drift types: 1,782 Collateral, 1,307 Scope-Expansion, 871 Purpose-Drift. Difficulty tiers (misaligned): `random` 1,960, `hard` 2,000.
- **Labels are determined by construction:** aligned hunks are real hunks from the seed commit; a misaligned hunk is a donor hunk from a different commit inserted into the session. No human labels yet.
- **Noise, shipped with ground truth** (25.2% of rows have at least one): intent_turn 2,730 (`intent_noise_type`), duplicate_hunk 937 (`is_duplicate_hunk`), non_semantic_hunk 714 (`non_semantic_reason`), label_flip 569 train-only (`label_noise`, `label_clean`; P(flip|aligned)=0.03, P(flip|misaligned)=0.10), truncation 198 (`exceeds_max_len`). Train on `label`, evaluate on `label_clean`; val/test labels are clean.
- **Already-measured baselines (test split; reproduce them and report any mismatch):** NDCG@5 over 447 rankable sessions: B1 random 0.7236, B2 diff-lines 0.7236, B3 hunk size 0.7206, B4 TF-IDF dissimilarity 0.8295, B5 logistic regression 0.8159. Within-session pairwise AUC (random tier / hard tier): B3 0.479/0.493, B4 0.812/0.551, B5 0.777/0.551. B5 classification: macro-F1 0.698, precision 0.506, recall 0.574, specificity 0.842. NDCG@5 has a high floor (0.72) because sessions are short, so **always also report per-tier AUC**.
- Known limits to state in the app and README: misaligned hunks are constructed, seed commits may be tangled, a donor may coincidentally fit, only Python/JS/TS.

**"Compared across datasets":** there is one real dataset. Define the evaluation datasets honestly as: full test; Python-only; JS/TS-only; `hard` tier; `random` tier; and each test repository (httpx, attrs, axios). If you add any external dataset it must be real, licence-compatible and documented; never invent one.

## 4. Models and experiments to build

**Primary deep-learning model (the project's one DL model): ModernBERT-base cross-encoder** (149 M params, 22 layers, 768 hidden, 12 heads, 8k context). Input `[CLS] consolidated_intent [SEP] hunk_text`. Heads on the pooled representation (multi-task): (1) alignment, (2) drift type (3-way), (3) calibration/confidence. Loss `0.6·L_align + 0.3·L_drift + 0.1·L_cal`, alignment term = focal loss (γ=2) for the ~78:22 class ratio. Starting settings: AdamW, lr 2e-5, cosine schedule with 10% warm-up, up to 6 epochs, early stopping on validation macro-F1, bf16/fp16, gradient checkpointing, gradient accumulation. **On 6 GB use max length 512 and a small micro-batch (for example 4 with accumulation 4); lower it further if you hit out-of-memory.** If ModernBERT-base cannot train on this GPU, fall back (LoRA, or a smaller encoder) and document why. Tune a few key hyperparameters (learning rate, loss weights, max length) with a small Optuna study sized to the time available. Pick everything on validation only; touch the test set only for final reporting.

**Comparison models (all trained/evaluated on the same splits and metrics):**
- Non-neural: B1–B5 above.
- Deep-learning baselines: a small transformer cross-encoder (for example MiniLM), a code-aware encoder (for example CodeBERT or UniXcoder, only if it fits in memory), and a simple neural baseline (for example a siamese BiLSTM or CNN). Skip a baseline only if it cannot run, and say so.
- Optional: zero-shot LLM judge, only if an API key and network exist. Never block on it.

**Post-training:**
- **Temperature scaling** on validation; report ECE before and after.
- **Distillation** (τ=4, α=0.5) to a ~22 M MiniLM student, then **INT8 ONNX** export. Targets (not results): under 30 MB, p95 under 100 ms on CPU. Measure the real values on this machine and report them with the CPU model.

**Denoising:** rule-based removal of non-semantic and duplicate hunks; **confident learning** (`cleanlab`) to find label errors and score it against the **known 569 flips** (precision/recall of detection); co-teaching as a second method if time allows. Also run a **noise-robustness study**: train with increasing injected label noise and with/without denoising. (An autoencoder was rejected: reconstruction error means little for discrete text/code pairs.)

**K-Means integration (required by the course guideline):** run K-Means on the model's pooled embeddings of hunk-intent pairs. (a) Unsupervised analysis: choose k with elbow and silhouette, measure cluster purity and NMI against the drift-type and alignment labels, visualise with UMAP or t-SNE. (b) Hybrid: add cluster-distance/cluster-prior features to the ranking or calibration stage and compare with and without (also an ablation row). (c) In the UI, show each hunk's cluster and "similar flagged hunks". Report honestly if k-means adds nothing.

**GAN comparison (required):** Sift has no natural GAN, so implement an **adversarially trained baseline on frozen (intent, hunk) embeddings** (generator conditioned on the intent embedding produces aligned-hunk embeddings; the discriminator is trained to separate real aligned pairs from generated ones and is used as an alignment scorer). Compare with the main model on the same test split. Label it plainly as the interpretation chosen. Report training curves and any instability.

**Explainable AI (required):** token-level attribution with **Integrated Gradients** (`captum`), aggregated to words and diff lines, for both intent and hunk. Include a **faithfulness test** (mask the top-attributed tokens versus random tokens and measure the real drop in misalignment probability). Show explanations in the UI.

**Ablations (same test split, per tier):** full model; hunk-only; intent-only; no intent consolidation (raw turns); no focal loss; no multi-task heads; no denoising; with/without k-means features; distilled student; INT8 student. Use several seeds if time allows and report mean and spread; if one seed, say so.

**Metrics:** accuracy, macro-F1, precision, recall, specificity, PR-AUC/ROC-AUC, within-session pairwise AUC, NDCG@K, Precision@K, MRR, ECE, model size, p50/p95 CPU latency, GPU memory.

## 5. Result artifacts (generated by scripts into `results/`, shown in the app)

Every number and figure comes from a script that writes JSON/CSV/PNG. **At least 8 graphs** comparing the model with the baselines, and **at least 5 tables, one of them the master comparison of all models.** Suggested set (add more if useful):

Graphs: (1) training/validation loss curves; (2) PR curves, all models; (3) ROC curves per tier; (4) confusion matrix; (5) NDCG@K versus K, all models; (6) per-tier pairwise AUC bars; (7) calibration reliability diagram before/after temperature scaling; (8) noise-robustness curve (metric versus injected label-noise rate, with and without denoising); (9) confident-learning detection precision/recall versus the known flips; (10) k-means clusters (UMAP/t-SNE) plus elbow/silhouette; (11) ablation bars; (12) latency versus size scatter (full, student, INT8); (13) GAN baseline versus main model; (14) XAI faithfulness curve.
Tables: dataset statistics and splits; hyperparameters; **master model comparison**; ablation; noise/denoising results; latency and size; test summary and coverage.

Also write `results/SUMMARY.md` (plain-language: what was trained, settings, headline numbers, honest weaknesses).

## 6. Backend

FastAPI (Python). Endpoints (suggested): `POST /api/analyze` (session turns plus a unified diff, or a sample id → consolidated intent, per-hunk probability, calibrated confidence, drift type, cluster id, ranking, optional attributions), `GET /api/samples` and `/api/samples/{id}` (curated test-set examples with ground truth, to demo and to prove accuracy), `GET /api/models` and a model parameter so the UI can switch between **main model, distilled/INT8 student and baselines**, `GET /api/results/*` (all JSON for the dashboard), `GET /api/dataset/stats`, `GET /api/health`, `GET /api/tests` (latest test and coverage summary). Parse real unified diffs into hunks robustly (git diff format, multiple files, renames, binary-file notes). Validate inputs, handle errors with clear messages, load models once, run on CPU or GPU automatically, and keep latency low. Provide OpenAPI docs.

## 7. Frontend (the most important part; make it genuinely good)

Use **React + Vite + TypeScript + Tailwind** (or an equivalent modern stack you prefer), a clean component library style, a chart library (Recharts or Plotly), and a **diff viewer** component. Design like a real developer tool: professional, consistent, not a student template. Dark and light themes, responsive layout, smooth but restrained motion, loading skeletons, empty and error states, keyboard shortcuts for stepping through hunks, accessible contrast and labels. Spend real effort on typography, spacing and visual hierarchy.

Pages:
1. **Review workspace (the main screen).** Left: session intent panel (paste chat turns; show which turns were dropped as noise and the consolidated intent). Centre: the diff, split into hunks as cards ordered by misalignment risk, with a risk colour, probability and confidence, a drift-type badge, and code with syntax highlighting. Right: details for the selected hunk: explanation with **highlighted tokens** in intent and code (word and line attribution), the nearest similar flagged hunks (k-means cluster), and Accept / Flag / Needs-discussion actions that update a review summary ("2 of 5 hunks need attention"). Controls: load an example from the test set (with the ground truth shown after analysis so I can demo correctness), paste your own diff, upload a `.diff` or `.patch` file, **model switcher** (main, student/INT8, baselines) with latency shown, threshold slider.
2. **Results dashboard.** All graphs and tables from section 5, interactive and well laid out, with a headline-metrics strip, the master comparison table (sortable), per-tier toggles, a "what this means" sentence under each chart, and an honest-limitations box.
3. **Experiments page:** training run info (hardware, settings, epochs, time), ablation, noise and denoising study, k-means analysis, GAN comparison, XAI faithfulness.
4. **Dataset explorer:** statistics, splits, noise types with counts, search and browse samples.
5. **Testing page:** test results by category, coverage, defect register.
6. **About:** the architecture diagram (draw a clean one; the file `sift_overall_architecture.png` may be in the folder, otherwise generate it), the problem, the SDG relevance, honest novelty and limitations.

The UI must work with a single command (see section 9) and must not show fake data: when it shows a number it comes from the real result files or a real model call.

**Visual QA loop (required):** use Playwright to screenshot every page at desktop and mobile widths, look at the screenshots yourself, and fix layout, overflow, contrast, empty-state and error problems until it looks professional. Save the final screenshots to `docs/screenshots/`.

## 8. Testing (include and report)

`pytest` plus `pytest-cov`, and Playwright end-to-end tests. Cover: smoke/readiness; white-box (shapes, heads, loss, tokenisation, diff parser, intent consolidation); black-box API (equivalence partitions, boundary values, malformed input); data pipeline and **split-leakage** (zero repositories shared across splits); **metamorphic relations** (consistent identifier rename must not flip the verdict; whitespace or comment-only edits must not change the label; reordering independent hunks must not change each hunk's score; determinism); **adversarial/red-team** cases (misleading comments, intent text injected into a diff, very long/empty/unicode input); back-to-back test of the full model against the distilled/INT8 student; UI end-to-end flows. Record only real defects in a defect register (`xfail(strict=True)` for known ones). Align the categories with the framework PDF in the folder (performance, testing methods, data, models, context, documentation). Fix bugs you find; keep the suite green; report coverage.

## 9. Repository, run and reproducibility

```
Sift/
  CLAUDE.md  PROGRESS.md  DECISIONS.md  README.md  Makefile
  data/                  (verified dataset files)
  src/sift/              data, diffparse, consolidate, model, losses, train, calibrate,
                         denoise, distill, kmeans, gan_baseline, explain, rank, serve
  experiments/           baselines, comparison, ablation, noise, kmeans, gan, xai, latency
  results/               JSON, CSV, PNG, SUMMARY.md
  checkpoints/           (gitignored; keep best and final)
  app/backend/  app/frontend/
  tests/  e2e/
  docs/screenshots/
```

- Fix seeds (dataset seed 473), pin versions in `requirements.txt` and the frontend lockfile, log every run's config and environment to JSON, save checkpoints and make runs resumable.
- Run long training in the background (`nohup`, logs to files), poll, and continue other work meanwhile. Commit to git at the end of each phase.
- **One command to start everything:** `make demo` (or `./run.sh`) installs missing pieces, starts the backend and the frontend, and prints the URL. Another, `make reproduce`, re-runs experiments and regenerates `results/`.

## 10. Rules

1. **No fabrication.** No invented results, metrics, graphs, screenshots, papers or URLs. Planned values are labelled "target". Anything measured is measured by a script you ran.
2. **If a result is poor, report it honestly** and make reasonable, documented attempts to improve it (hyperparameters, longer training, better preprocessing) within the time available; never tune on the test set.
3. **Decide on your own.** Where the spec leaves room, choose the sensible option, write it in `DECISIONS.md` (what, why, alternative) and move on. Only stop for a genuine blocker (hardware failure, no network when a download is essential), and even then continue with everything else.
4. **Keep scope honest:** do not claim a new neural architecture; state dataset limits; state that latency numbers belong to this machine.
5. Update `PROGRESS.md` after every phase (done / in progress / blocked).

## 11. Phases (do them in this order)

1. Environment: venv (Python 3.11/3.12), CUDA check, install dependencies, unzip and verify the dataset against section 3, write loaders, reproduce B1–B5.
2. Diff parser and intent consolidation, with unit tests.
3. Model, losses, metrics, training loop; smoke-train on a tiny subset; fix memory settings for 6 GB.
4. Train the full model; small hyperparameter study; train baselines; evaluate everything on all test subsets.
5. Calibration, denoising and the noise study, k-means, distillation, INT8, latency benchmark.
6. XAI (Integrated Gradients and faithfulness test) and the GAN baseline.
7. Ablations.
8. Backend API and inference service.
9. Frontend (all six pages), connect to the real backend and result files.
10. Test suite, end-to-end tests, coverage, defect register; fix everything found.
11. Visual QA with Playwright screenshots; iterate on the UI until it looks polished.
12. Final verification, `results/SUMMARY.md`, README, and a clean `make demo` from scratch.

## 12. Definition of done (all must be true)

- Dataset verified; baselines reproduced; the main model trained with saved checkpoint and logged results.
- All comparison models, ablations, noise/denoising, k-means, GAN baseline and XAI experiments run, with at least 8 graphs and 5 tables (including the master comparison) in `results/` and in the app.
- Calibrated model, distilled and INT8 student exported, real size and latency measured.
- Backend and frontend run with one command; every page works with real data; the review workspace analyses a pasted or uploaded diff end to end; the model switcher works.
- Test suite and end-to-end tests pass; coverage reported; defects recorded honestly.
- Screenshots in `docs/screenshots/`, `README.md`, `DECISIONS.md`, `PROGRESS.md` and `results/SUMMARY.md` complete.

## 13. Final message to me

Tell me, briefly: how to launch the app, what was trained (model, settings, time, hardware), the headline results versus the baselines (including the weak spots), what could not be done and why, and where everything is. Do not narrate your steps.
