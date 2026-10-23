# Sift: viva walkthrough (what it does, what goes in, what comes out, how to check it)

All numbers here are from `results/SUMMARY.md` and `results/*.json` (single seed 473, test split). If asked for a number, say where it comes from.

## 1. The problem in one minute
AI coding agents (Claude Code, Cursor, Copilot) change code across many files. A human must review the diff. A diff is split into **hunks** (each `@@ ... @@` block). Some hunks are not what the developer asked for. Sift reads **what the developer asked** (the chat turns) and **each hunk**, and says: *Aligned* or *Misaligned*, then ranks hunks so the likeliest misaligned ones are reviewed first.

Misaligned hunks have a drift type:
- **Collateral**: an unrequested change next to the requested one (e.g. reformatting while fixing one function).
- **Scope-Expansion**: an unrequested feature or extra behaviour.
- **Purpose-Drift**: a change serving a different purpose.

This is message-code inconsistency detection (a text/code pair classification), **not** defect prediction.

**Why deep learning?** Any diff-level number (lines changed, files touched) is identical for every hunk in one diff, so it cannot rank hunks inside it. Ordering by diff lines equals random order (NDCG@5 0.7236 both). Deciding if one hunk serves a stated intent needs both texts read together, which a cross-encoder does.

## 2. Input and output
**Input**
1. Session intent: the developer's chat turns, one per line. Some are noise (chit-chat like "thanks", a retracted request, a repeated turn).
2. A unified diff (git diff format), pasted, uploaded as `.diff`/`.patch`, or a curated test-set sample.
3. Choices: model (main, student, INT8, baselines), threshold, "explain with Integrated Gradients".

**Pipeline (what happens on Analyse)**
1. **Intent consolidation** (`src/sift/consolidate.py`): drops chit-chat, retracted and repeated turns, leaves one consolidated intent. Rule-based, 94.8% exact match with the dataset's ground truth over all 18,000 rows (test split 85.4%; rules were written from the visible noise templates, so this is in-distribution).
2. **Diff parsing** (`src/sift/diffparse.py`): splits files/hunks, handles renames, binary notes, rejects malformed headers.
3. **Model** (`src/sift/model.py`): input is `[CLS] consolidated_intent [SEP] hunk_text`, truncated to 512 tokens.
4. **Calibration**: temperature scaling (T = 1.275 fitted on validation) so probabilities mean what they say.
5. **Ranking**: hunks sorted by P(misaligned); threshold turns a probability into a flag.
6. **Explanation** (optional): Integrated Gradients gives each token a score.

**Output (per hunk)**: calibrated P(misaligned), confidence, aligned/misaligned at the threshold, drift-type probabilities (3-way), k-means cluster and similar flagged training hunks, rank, and token attributions.

## 3. The model
- **ModernBERT-base cross-encoder**, 149 M parameters, 22 layers, hidden 768, 12 heads, RoPE, GeGLU, 8k context. Cross-encoder means intent and hunk go through the transformer together, so attention can compare them word by word (a siamese/bi-encoder compares two separate vectors, which is weaker for this).
- **Three heads on the pooled [CLS] vector**: (1) alignment (binary), (2) drift type (3-way, only misaligned examples), (3) confidence/calibration.
- **Loss**: `0.6·L_align + 0.3·L_drift + 0.1·L_cal`. The alignment term is **focal loss** (gamma 2) because classes are about 78:22 aligned:misaligned. Focal loss down-weights easy examples.
- **Training**: AdamW, lr 2e-5, cosine schedule with 10% warm-up, up to 6 epochs, early stopping on validation macro-F1, bf16, gradient checkpointing, micro-batch 8 x accumulation 2, max length 512. About 28 minutes on the RTX 4050 (6 GB). Max length is 512, not the planned 1,024, because of the 6 GB GPU.
- **Novelty (do not overclaim)**: the encoder and training methods already exist. The contribution is the hunk-level, multi-turn-intent task framing, the pipeline, and the dataset. No new architecture.

## 4. The dataset (Sift-IA v1.0)
- 18,000 hunk-intent pairs, 6,111 sessions, 15 open-source repos, Python/JS/TS.
- **Splits are by repository** (no repo in two splits): train 12,600, val 2,700, test 2,700 hunks (test repos: httpx, attrs, axios).
- **Labels come from construction**: aligned hunks are real hunks from the seed commit; a misaligned hunk is a donor hunk from a different commit inserted into the session. No human labels yet.
- **Difficulty tiers**: `random` (random donor) and `hard` (donor chosen by identifier overlap, so it looks similar).
- **Noise shipped with ground truth** (25.2% of rows): noisy intent turns, duplicate hunks, non-semantic hunks (whitespace, lockfile, version-bump), label flips (train only, 569 rows), truncation.
- **Limitations to say out loud**: constructed misalignments (a model could learn construction artefacts), tangled seed commits, a donor may coincidentally fit, only three languages, test split has no TypeScript (known defect D-DATA-2), the brief's language counts differ from the released data.

## 5. Metrics: what each one means
- **Macro-F1**: F1 averaged over both classes, fair under imbalance. Main model 0.680 (B5 logistic regression 0.698).
- **Precision / recall / specificity**: of flagged hunks how many are truly misaligned; of misaligned how many are caught; of aligned how many are left alone.
- **NDCG@5**: quality of the ranking of the top 5 hunks in a session. It has a high floor (0.72) because the median session has only 2 hunks, so always read per-tier AUC beside it. Main 0.868, B4 0.830, B5 0.816, random 0.724.
- **Within-session pairwise AUC (per tier)**: chance that a misaligned hunk is ranked above an aligned one in the same session. Hard tier: main 0.726 vs B4/B5 0.551. This is the strongest evidence for the model.
- **ECE** (expected calibration error): gap between stated confidence and actual accuracy. 0.064 before, 0.028 after temperature scaling.
- **ROC/PR-AUC, MRR, P@K**: standard ranking/threshold-free metrics.

## 6. Frontend: page by page (what you see, what it proves, how to check)
Start with `make demo`, open the printed URL. Top right shows "backend online".

### Review (main page)
- **Left**: "Test-set example" dropdown loads a real curated test sample (ground truth hidden until you analyse). Intent box = the chat turns. Unified diff box (or upload `.diff`/`.patch`). Model dropdown shows size and last latency. Checkbox "Explain with Integrated Gradients" (slow on CPU). **Analyse** button.
- **Consolidated intent / Dropped as noise** panel: shows which turns were removed. Check by adding a turn like "thanks!" and watching it appear under dropped (note: the single word "hi" is not currently dropped, a known minor gap).
- **Centre**: Review summary ("1 of 2 hunks need attention") and the **threshold slider** (default 0.34 for the main model, chosen on validation, not test). Moving it re-flags hunks live. Then **hunk cards ordered by risk**: coloured bar, P(misaligned), confidence, drift-type badge, cluster id, and a "model correct" / "truth: ..." chip (ground truth only for test samples). Accept / Flag / Needs-discussion (keys `a`, `f`, `d`; `j`/`k` step through hunks) update the review summary.
- **Right (selected hunk)**: calibrated probability, drift-type probabilities (3 bars), the confidence head's output, latency and temperature T, **token highlights** on the intent and the code (red pushes toward misaligned, blue toward aligned), and **similar flagged hunks** from the k-means cluster.
- **How to check correctness**: load a test sample, press Analyse, compare the verdict with the "truth" chip. Do 5-10 samples, including the same sample on different models.
- **How to explain it**: "Probability is calibrated; a 0.43 means about 43% chance misaligned. The threshold is a precision/recall trade-off the reviewer controls."

### Results
Headline metrics strip, **master model comparison** (sortable; B1-B5, BiLSTM, MiniLM, CodeBERT, main, frozen-embedding and GAN-style baselines, student, INT8), per-tier toggles, graphs (NDCG@5 by model, within-session pairwise AUC by tier, PR/ROC, confusion matrix, reliability diagram, latency vs size...), each with a "what this means" sentence, and an honest-limitations box.

### Experiments
Training runs and settings; **Ablation** (full, hunk-only, intent-only, no consolidation, no focal, no multi-task, no denoising, student, INT8); **Noise and denoising study**; **K-means analysis**; **GAN comparison**; **Explainability: faithfulness**; **Deployment: size and CPU latency**; XAI worked cases (one per class).

### Dataset
Splits, drift types and tiers, noise types with counts, search and browse samples.

### Testing
Results by category, coverage, **defect register** (six known defects as strict expected failures).

### About
Architecture diagram, the problem, SDG 9 (primary) and SDG 8 (secondary) relevance, what is and is not novel, limitations.

## 7. Each extra component: what and why
- **Temperature scaling**: one scalar T divides the logits before sigmoid, fitted on validation. Fixes over-confidence without changing the ranking. ECE 0.064 to 0.028.
- **Denoising**: rules remove non-semantic and duplicate hunks; confident learning (`cleanlab`) flags likely label errors. Scored against the 569 known flips: precision 0.150, recall 0.387, a weak detector (random would be about 0.045). An autoencoder was rejected because reconstruction error is meaningless for discrete text/code pairs.
- **Noise-robustness sweep** (reduced setting, 3,000 train rows, 2 epochs): denoising raised NDCG@5 at all four noise rates (0, 10, 20, 30%), macro-F1 clearly only at 20% and 30%.
- **Distillation + INT8**: a 22 M MiniLM student learns from the big model (tau 4, alpha 0.5), exported to INT8 ONNX: 22.9 MB, macro-F1 0.667, CPU p50 8.5 ms and p95 66.9 ms (targets under 30 MB, p95 under 100 ms met on this machine, measured while the CPU was shared). Main model on CPU: p95 95.1 ms, 596 MB; on GPU p95 11.9 ms.
- **K-means** on pooled embeddings (k=4 by elbow and silhouette): purity 0.800 vs a 0.780 majority rate, so almost no structure. Used for the cluster id and the similar-hunk lookup. Honest result: it adds almost nothing.
- **Integrated Gradients** (`captum`): attribution = path integral of gradients from a baseline (all PAD) to the input, 16 steps, summed over embedding dimensions, aggregated to words and lines. **Faithfulness test**: masking the top-5 attributed tokens drops mean P(misaligned) from 0.579 to 0.386, versus 0.482 for 5 random tokens (n=100). So attribution is informative, but the random drop is large too (masking is out-of-distribution). Attribution is evidence, not proof of the cause. Latency about 0.5 s per explanation.
- **GAN-style baseline**: Sift has no natural GAN, so this is an interpretation: an adversarially trained baseline on frozen (intent, hunk) embeddings, a generator conditioned on the intent makes aligned-hunk embeddings, a discriminator separates real from generated and is used as the scorer. It did not beat the main model on everything and the generator showed collapse (see `results/gan_vs_main.json`). A simple frozen cosine dissimilarity baseline beat the main model on NDCG@5 (0.874) and random-tier AUC (0.814). Say this plainly.

## 8. What the ablations say (be honest)
Hunk-only, intent-only, no consolidation, no focal loss, no multi-task, no denoising: each is a full-protocol run with one seed. Ablated models scored 0.008 to 0.026 macro-F1 higher than the full model, so the components are **not shown to help** here. With one seed, differences this small are within noise. Never claim they proved each component necessary. The intent-only and hunk-only rows are the interesting ones: they test whether the model really reads both texts (see `results/ablation.csv`).

## 9. Testing (Deliverable categories)
193 tests: 187 passed, 6 expected failures, 0 failed; 73.4% line coverage.
- **Smoke, white-box** (shapes, heads, loss, tokenisation, parser), **black-box** (equivalence partitions, boundary values, malformed input), **data pipeline** (zero repositories shared across splits, deterministic loading).
- **Metamorphic** relations: consistent identifier rename, whitespace/comment-only edits, reordering independent hunks, determinism, removing chit-chat must not change the verdict.
- **Adversarial / red-team**: misleading comments, intent text injected into a hunk, very long, empty and unicode inputs.
- **Back-to-back**: full model against the student/INT8.
- **UI end-to-end** with Playwright (10 tests).
- **Defect register (real defects the tests found)**: D1 plain `diff -u` multi-file headers merge into the previous hunk; D2 chit-chat filter matches only 8 fixed phrases; **D3 (High) a comment copying the intent into a misaligned hunk flips 69% of detections to aligned**; D4 a "this change implements: <intent>" comment flips 50%; D-DATA-1 and D-DATA-2 dataset differences. D3 and D4 mean the model can be fooled by text that echoes the intent, a real weakness for an adversarial setting.

## 10. Likely viva questions, short answers
- *Why cross-encoder, not embeddings?* Needs token-level interaction between intent and hunk; a bi-encoder compares two summaries.
- *Why NDCG and also AUC?* NDCG has a 0.72 floor on 2-hunk sessions; per-tier pairwise AUC is the sharper test.
- *Your model's macro-F1 is below logistic regression. Why use it?* Its advantage is ranking and the hard tier (0.726 vs 0.551), which is the real product need (review order). Recall at the chosen threshold is only 0.476.
- *Is the data real?* Hunks are real, but misaligned labels are constructed, so there is no human validation and the model may learn construction artefacts. The hard tier mitigates but does not remove it.
- *Why 512 tokens?* 6 GB GPU. Median pair is about 132 tokens, p95 455.
- *Did you tune on test?* No. Threshold, temperature and early stopping use validation only.
- *Why is the baseline better on some metrics?* Frozen cosine dissimilarity ranks random-tier pairs well because random donors are lexically unrelated; hard-tier AUC is where the model wins.
- *What would you do next?* Human-labelled agent-authored sessions, multiple seeds, adversarial training against D3/D4, more languages, a fuller hyperparameter search (6 trials were run instead of 60).
- *What is novel?* Task framing, pipeline, dataset. Not the architecture.

## 11. What was reduced (say it before they ask)
Hyperparameter search: 6 trials not 60. Baselines, noise sweep and HPO at reduced settings (max length 256, subsets, 1-2 epochs); CodeBERT saw only 4,000 rows. UniXcoder dropped (download stalled). `rules_only` ablation dropped. Single seed throughout. Latency belongs to this machine (i7-13650HX; student numbers measured while the CPU was shared).

## 12. Commands
- Launch: `make demo` (stop: `make stop`)
- UI tests: `make e2e`
- Re-run tables and graphs: `make reproduce`
- Full tests: `PYTHONPATH=src .venv/bin/python -m pytest tests -q`
- Key files: `src/sift/` (model, losses, train, calibrate, denoise, distill, explain, rank, serve), `app/backend/main.py`, `app/frontend/src/pages/`, `results/`, `DECISIONS.md`.

## 13. Comparison with standard base papers
See `docs/RELATED_WORK.md` (task comparison with the closest papers, the existing methods Sift reuses, an honest table against the baselines we ran, and the reference list).
