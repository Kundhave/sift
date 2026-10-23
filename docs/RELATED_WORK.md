# Sift compared with standard base papers

**Read this first.** The papers below solve neighbouring tasks on different data with different metrics. Their reported numbers are **not comparable** with Sift's numbers, so this section compares *task, input, granularity, method and evaluation design*, never "our score beats their score". No paper's results are quoted here because none were reproduced on Sift-IA.

## 1. Closest research area: comment-code inconsistency and code-change tasks

| | Sift (this project) | Panthaplackel et al., AAAI 2021 | Panthaplackel et al., ACL 2020 | Herzig & Zeller, MSR 2013 |
|---|---|---|---|---|
| Question | Does this **hunk** serve the developer's stated **intent**? | Did a code change make an existing **comment** inconsistent? | Given a code change, how should the **comment** be updated? | How much do **tangled changes** (unrelated edits in one commit) pollute mining results? |
| Pair classified | consolidated multi-turn intent + one diff hunk | comment + code change (just-in-time) | comment + code change | commit contents (untangling) |
| Granularity | hunk-level inside one diff, ranked | method-level comment/code pairs | method-level | file/commit-level |
| Text side | chat turns, some noisy (chit-chat, retracted, repeated) | code comments | code comments | none (change structure) |
| Output | aligned/misaligned, 3 drift types, calibrated probability, ranking, token explanation | inconsistent / consistent | edit sequence for the comment | untangled change groups |
| Method | ModernBERT cross-encoder, multi-task heads, focal loss | deep model correlating comment with code change | edit-based neural model | multi-predictor untangling |
| Evaluation | macro-F1, per-tier pairwise AUC, NDCG@5, ECE | classification metrics on their corpus | edit and update metrics | effect on bug-count bias |
| Relation to Sift | Sift borrows the "text vs code pair" framing | same family of task, closest prior | related task | motivates our limitation: seed commits may be tangled |

What is different in Sift: intent is **multi-turn and noisy** (consolidated first), the unit is the **hunk inside a diff**, and the goal is a **ranking for human review**, not a single yes/no on a comment.

## 2. Building blocks Sift reuses (all existing methods)

| Component in Sift | Base paper | What Sift does with it |
|---|---|---|
| Cross-encoder on a text pair | Nogueira & Cho, *Passage Re-ranking with BERT* (2019) | Same pattern: concatenate the two texts, classify from the [CLS] vector. Sift applies it to intent + hunk |
| Encoder | Warner et al., *Smarter, Better, Faster, Longer: ModernBERT* (2024) | ModernBERT-base (RoPE, GLU-family activation, alternating local/global attention, 8k context), fine-tuned |
| Code-aware encoder baseline | Feng et al., *CodeBERT* (EMNLP Findings 2020) | Used as a baseline only; under-trained here (4,000 rows, 1 epoch), so not a fair comparison |
| Focal loss | Lin et al., *Focal Loss for Dense Object Detection* (ICCV 2017) | Alignment term, gamma 2, for the ~78:22 class ratio |
| Calibration | Guo et al., *On Calibration of Modern Neural Networks* (ICML 2017) | Temperature scaling on validation, ECE 0.064 to 0.028 |
| Distillation | Hinton et al., *Distilling the Knowledge in a Neural Network* (2015) | tau 4, alpha 0.5, MiniLM student, then INT8 ONNX |
| Attribution | Sundararajan et al., *Axiomatic Attribution for Deep Networks* (ICML 2017) | Integrated Gradients, 16 steps, aggregated to words and lines |
| Label-noise detection | Northcutt et al., *Confident Learning* (JAIR 2021) | `cleanlab` scored against the 569 known flips (precision 0.150, recall 0.387) |
| Noisy-label training (option) | Han et al., *Co-teaching* (NeurIPS 2018) | Named in the design; **not run** in this project |
| Tangled commits | Herzig & Zeller (MSR 2013) | Stated as a dataset limitation |

## 3. Honest comparison against the baselines we actually ran

Same dataset, same split, same metrics (single seed 473; see `results/master_comparison.csv`).

| Model | Idea | Macro-F1 | NDCG@5 | Hard-tier pairwise AUC |
|---|---|---|---|---|
| B4 TF-IDF dissimilarity | lexical overlap | not reported | 0.830 | 0.551 |
| B5 logistic regression (4 features) | simple features | 0.698 | 0.816 | 0.551 |
| Siamese BiLSTM | two separate encoders | 0.562 | 0.783 | 0.527 |
| MiniLM-L6 cross-encoder (reduced training) | small cross-encoder | 0.616 | 0.835 | 0.636 |
| CodeBERT cross-encoder (reduced training) | code-aware encoder | 0.460 | 0.779 | 0.600 |
| **ModernBERT cross-encoder (main)** | cross-encoder, multi-task | 0.680 | 0.868 | 0.726 |
| Frozen cosine dissimilarity (embeddings) | no training | n/a | 0.874 | 0.719 |

Reading it honestly:
- Reading both texts jointly helps most where lexical overlap fails: on the **hard tier** the main model reaches 0.726 pairwise AUC against 0.551 for B4 and B5.
- The main model's macro-F1 is **below** B5, and a frozen cosine baseline is slightly better on NDCG@5. The case for the model is hard-tier ranking, not every metric.
- MiniLM, BiLSTM and CodeBERT were trained with a smaller budget than the main model, so part of their gap is budget, not architecture.
- The one comparison the literature supports directly is architectural: joint (cross) encoding beats separate encoding for pair relevance (Nogueira & Cho), and our BiLSTM and frozen-cosine rows are consistent with, but do not prove, that.

## 4. What Sift does not claim
- No new neural architecture. The contribution is the task framing, the pipeline and the dataset.
- No state-of-the-art claim against any published system: no published system was evaluated on Sift-IA, and Sift was not evaluated on their corpora.
- Labels are constructed (donor hunks from other commits), with no human validation yet.

## 5. References
Confirmed by web search while writing this section (titles, authors and venues as found):
1. S. Panthaplackel, J. J. Li, M. Gligoric, R. J. Mooney. *Deep Just-In-Time Inconsistency Detection Between Comments and Source Code.* AAAI 2021. https://arxiv.org/abs/2010.01625
2. S. Panthaplackel, P. Nie, M. Gligoric, J. J. Li, R. J. Mooney. *Learning to Update Natural Language Comments Based on Code Changes.* ACL 2020. https://aclanthology.org/2020.acl-main.168/
3. K. Herzig, A. Zeller. *The Impact of Tangled Code Changes.* MSR 2013. https://www.st.cs.uni-saarland.de/publications/details/herzig-msr-2013
4. B. Warner et al. *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference* (ModernBERT). 2024. https://arxiv.org/abs/2412.13663
5. R. Nogueira, K. Cho. *Passage Re-ranking with BERT.* 2019. https://arxiv.org/abs/1901.04085
6. Z. Feng et al. *CodeBERT: A Pre-Trained Model for Programming and Natural Languages.* Findings of EMNLP 2020. https://arxiv.org/abs/2002.08155

Cited from memory, **not re-verified in this session** (check the details before submitting):
7. T.-Y. Lin et al. *Focal Loss for Dense Object Detection.* ICCV 2017.
8. C. Guo et al. *On Calibration of Modern Neural Networks.* ICML 2017.
9. G. Hinton, O. Vinyals, J. Dean. *Distilling the Knowledge in a Neural Network.* 2015.
10. M. Sundararajan, A. Taly, Q. Yan. *Axiomatic Attribution for Deep Networks.* ICML 2017.
11. C. G. Northcutt, L. Jiang, I. L. Chuang. *Confident Learning: Estimating Uncertainty in Dataset Labels.* JAIR 2021.
12. B. Han et al. *Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels.* NeurIPS 2018.
