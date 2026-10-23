# Sift results summary (plain language)

Every number below is copied from a results JSON/CSV written by a script in `experiments/` (see the generated tables in this folder). Single seed (473) everywhere, no confidence intervals.

## What was trained and how
- **Main model:** ModernBERT-base cross-encoder (149.0 M parameters) with three heads (alignment, drift type, confidence), focal-loss multi-task training. Input length L = 512 (the brief's 1,024 does not fit), micro-batch 8 x gradient accumulation 2 (effective 16), bf16, gradient checkpointing, lr 2e-5, up to 6 epochs with early stopping. Trained on 10,128 of 12,600 train rows after denoising (384 non-semantic and 502 duplicate hunks removed by rules, 1,586 flagged by confident learning). Training took 1,692 s (about 28 min), peak GPU memory 3.1 GB, on an RTX 4050 Laptop GPU (6 GB). Best validation macro-F1 0.716. Run: `experiments/runs/full`.
- **Post-training:** temperature scaling (T = 1.275), distilled MiniLM-L6 student (8 epochs, 542 s), INT8 ONNX export, k-means index on pooled embeddings, Integrated Gradients explanations.
- **Comparison models (reduced protocol, same denoised rows):** MiniLM-L6 cross-encoder (2 epochs, L 256), CodeBERT cross-encoder (1 epoch, L 256), Siamese BiLSTM (15 epochs). These got far less training than the main model, so the gap to it is partly a budget gap, not purely an architecture gap.
- **Ablations:** full protocol, 6 epochs, L 512, one seed each (hunk-only, intent-only, no consolidation, no focal loss, alignment head only, no denoising).

## Headline results (test split, repository-disjoint)
- Main model: macro-F1 0.680, NDCG@5 0.868, pairwise AUC 0.800 (random tier) and 0.726 (hard tier), ECE 0.066 (0.064 -> 0.028 after temperature scaling).
- Versus baselines: B4 TF-IDF dissimilarity NDCG@5 0.830, B5 logistic regression 0.816; random order 0.724. On the hard tier the main model clearly leads (0.726 vs 0.551 for B4/B5). Diff-size baselines B2/B3 sit at chance, as the brief argued.
- Deployed target (INT8 ONNX student): 22.9 MB, macro-F1 0.667, NDCG@5 0.869, CPU p50 8.5 ms / p95 66.9 ms; fp32 ONNX 90.4 MB, p50 13.4 / p95 68.7 ms. Targets (<30 MB, p95 <100 ms) are met on this machine only.
- Latency was measured on an Intel i7-13650HX (4 threads, batch 1, tokenisation included) while the CPU was shared with a training job. Main ModernBERT (fp32 PyTorch, 596 MB, L 512, median-length test pairs, 20 warm-up + 100 timed, measured separately on the idle machine with 14 torch threads): CPU p50 94.5 ms / p95 95.1 ms; GPU (RTX 4050 Laptop) p50 11.6 ms / p95 11.9 ms, peak 631 MB. The main model misses the <30 MB size target but just meets p95 <100 ms on CPU; student numbers were taken under load and are not strictly comparable (`results/latency_full.json`).

## Weak spots (stated plainly)
- **Main macro-F1 (0.680) is below B5 (0.698)**, and its recall is only 0.476 at the validation-tuned threshold. The advantage is in ranking, especially the hard tier, not in thresholded classification.
- **On the random tier the main model does not beat simple baselines:** random-tier AUC 0.800 vs B4 0.812; NDCG@5 margin over B4 is only +0.039.
- **The GAN-style/frozen-embedding baselines win some metrics:** frozen cosine dissimilarity NDCG@5 0.874 and random-tier AUC 0.814, above the main model on both; the unsupervised GAN-style baseline gets macro-F1 0.630 and NDCG@5 0.864. Its generator was collapsed at epoch 1 (pairwise cosine 0.98) and diversified to 0.29 by epoch 40.
- **K-means adds almost nothing:** purity 0.800 vs majority rate 0.780; the hybrid changes macro-F1 by +0.010 and NDCG@5 by +0.005.
- **Confident learning is a weak detector** of the 569 known label flips: precision 0.150, recall 0.387 (final CL + rules drop list precision 0.138, recall 0.385; random would be about 0.045).
- **Ablation differences are mostly within noise (one seed):** dropping consolidation (+0.009 macro-F1), focal loss (+0.016), multi-task heads (+0.026) or denoising (+0.008) did not hurt and each ablated model scores slightly higher macro-F1 than the full model. Only removing the hunk or the intent is clearly harmful (-0.218 and -0.133 macro-F1; intent-only is near chance on tier AUC). The justification for these components is therefore not shown by these runs.
- **Red-team defects (tests/test_redteam.py, xfail strict):** D3 (High): a comment copying the intent text into a misaligned hunk flips 69% (18/26) of detected hunks to aligned; D4 (Medium): a comment "this change implements: <intent>" flips 50% (13/26). Threshold was 20%. Mitigation not implemented.
- Other recorded defects: D1 (plain `diff -u` multi-file headers merge into the previous hunk), D2 (chit-chat filter is exact-match on 8 phrases), D-DATA-1/2 (language counts differ from the brief; test split has no TypeScript). Test suite: 192 tests, 186 passed, 6 xfailed, 0 failed; line coverage 69.9% (src/sift + app/backend, e2e not measured).
- XAI: masking the top-5 Integrated-Gradients tokens drops mean P(misaligned) 0.579 -> 0.386 vs 0.482 for random masking (n = 100); the random-masking drop shows PAD masking is itself off-distribution. Explanation latency p50 about 506 ms on a shared GPU. Attribution is evidence, not proof.

## Reduced-protocol studies (not the full brief)
- **Noise robustness (`noise_robustness.json`):** ModernBERT on a 3,000-row train subset, 2 epochs, L 256, TF-IDF-based confident learning for the denoise arm. Test macro-F1 at noise 0.0 / 0.1 / 0.2 / 0.3: no-denoise 0.537 / 0.586 / 0.520 / 0.511, denoise 0.530 / 0.592 / 0.607 / 0.539. Denoising helps at 0.2 and 0.3 but not at 0.0, and the curve is non-monotonic (single seed, small subset), so treat it as indicative only.
- **Hyper-parameter search (`hpo_small.json`):** Optuna TPE, 6 trials (brief planned 60), 1,600 train / 800 val rows, 2 epochs. Best validation macro-F1 0.597 (lr 1.1e-5, L 384). No clearly better config, so the main model was not retrained; there is no `full_tuned` run.

## Not done / dropped
UniXcoder (weight download stalled; dropped), the rules-only ablation (dropped for time), 60-trial Optuna (6 run), multiple seeds, any human-labelled evaluation. All requested graphs (19 PNG) and tables (7 CSV) were generated; `make_tables_status.json` lists only `full_tuned` (never run) as skipped.

## Dataset limits
Labels are construction-determined (misaligned hunks are donor hunks from other commits), with no human validation, so a model may learn construction artefacts; the hard tier mitigates but does not remove this. Test repositories (httpx, attrs, axios) are repository-disjoint but still open-source Python/JavaScript only; seed commits may be tangled.

<!-- make_tables:begin (auto-generated by experiments/make_tables.py; do not edit by hand) -->
## Measured comparison (auto-generated, test split, single seed 473)

| Model | macro-F1 | NDCG@5 | AUC random | AUC hard | ROC-AUC | ECE |
|---|---|---|---|---|---|---|
| B1 Random order | n/a | 0.724 | n/a | n/a | n/a | n/a |
| B2 Diff lines changed | n/a | 0.724 | n/a | n/a | n/a | n/a |
| B3 Hunk size | n/a | 0.721 | 0.479 | 0.493 | n/a | n/a |
| B4 TF-IDF dissimilarity | n/a | 0.830 | 0.812 | 0.551 | n/a | n/a |
| B5 Logistic regression | 0.698 | 0.816 | 0.777 | 0.551 | n/a | n/a |
| Siamese BiLSTM | 0.562 | 0.783 | 0.641 | 0.526 | 0.630 | 0.234 |
| MiniLM-L6 cross-encoder | 0.616 | 0.835 | 0.741 | 0.636 | 0.670 | 0.059 |
| CodeBERT cross-encoder | 0.460 | 0.779 | 0.549 | 0.600 | 0.552 | 0.103 |
| Main: ModernBERT-base | 0.680 | 0.868 | 0.800 | 0.726 | 0.777 | 0.066 |
| Frozen-embedding frozen cosine dissimilarity | 0.614 | 0.874 | 0.814 | 0.719 | n/a | n/a |
| GAN-style baseline: gan unsupervised | 0.630 | 0.864 | 0.784 | 0.708 | n/a | n/a |
| GAN-style baseline: gan hybrid with misaligned negatives | 0.625 | 0.871 | 0.807 | 0.705 | n/a | n/a |
| Distilled MiniLM student (fp32) | 0.667 | 0.869 | 0.794 | 0.678 | 0.750 | 0.107 |
| MiniLM student, no distillation | 0.634 | 0.842 | 0.759 | 0.632 | 0.719 | 0.078 |
| Student ONNX fp32 | 0.669 | 0.874 | n/a | n/a | n/a | 0.058 |
| Student ONNX INT8 (deployed target) | 0.667 | 0.869 | n/a | n/a | n/a | 0.051 |

Empty (n/a) means not measured for that model; models that have not finished are absent: full_tuned.

Weak spots (stated plainly):
- The main model reaches macro-F1 0.680 and NDCG@5 0.868; hard-tier pairwise AUC is only 0.726 (random tier 0.800).
- The non-neural TF-IDF dissimilarity baseline B4 gets NDCG@5 0.830 and random-tier AUC 0.812; the main model's margin on those is +0.039 NDCG@5 and -0.012 random-tier AUC. Its real gain is on the hard tier (0.726 vs B4 0.551).
- Ablations (single seed; differences of ~0.01 are noise): largest macro-F1 gain vs full: 6. alignment head only (+0.026); largest loss: 2. hunk-only (-0.218).
- Labels are construction-determined with no human validation; misaligned hunks are constructed, so construction artefacts may be learnt; one seed, no confidence intervals.

Tables: master_comparison, ablation, per_tier_table, dataset_statistics, hyperparameters, latency_size (+ noise_robustness_table when the sweep exists). Graphs: see PNGs in results/.
<!-- make_tables:end -->

## Calibration (results/calibration.json, calibration_reliability.png)
Temperature scaling of the full model fitted on validation: T = 1.275. Test ECE 0.0638 -> 0.0280; test NLL 0.458 -> 0.442.

## Files
Tables: master_comparison, ablation, per_tier_table, dataset_statistics, hyperparameters, latency_size, noise_robustness_table (CSV). Graphs: PNGs in this folder (loss curves, PR, ROC per tier, confusion matrix, NDCG@K, per-tier AUC bars, calibration, noise robustness, confident learning, k-means elbow and t-SNE, ablation bars, latency vs size, GAN vs main, XAI faithfulness and four XAI cases). B4/B5 curves use test scores from `experiments/baseline_scores.py`.
