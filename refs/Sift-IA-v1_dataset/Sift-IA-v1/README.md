# Sift-IA v1.0 — Intent–Implementation Alignment for Code Diff Hunks

**18,000 diff hunks · 6,111 sessions · 8,016 real commits · 15 permissively-licensed repositories · Python / JavaScript / TypeScript**

Each sample pairs a **stated task intent** with **one diff hunk** and asks a single question:

> Does this hunk implement what was asked?

This is different from every existing commit or code-review dataset we are aware of. Those pair a diff
with a *post-hoc* commit message and ask whether the message *describes* the change. Sift-IA pairs a diff
hunk with an *a-priori* task specification and labels whether the hunk *serves* it — at hunk granularity,
with an intra-diff grouping key so that **ranking hunks inside one diff** is evaluable.

---

## 1. Files

| File | Rows | Description |
|---|---|---|
| `data/sift_ia_v1_train.csv` | 12,600 | Training split. **Contains injected label noise** (see §5). |
| `data/sift_ia_v1_val.csv` | 2,700 | Validation split. Clean labels. |
| `data/sift_ia_v1_test.csv` | 2,700 | Test split. Clean labels. |
| `data/sift_ia_v1_all.jsonl` | 18,000 | Lossless JSON Lines version of all three splits (`intent_turns` kept as a list). |
| `data/sessions.csv` | 6,111 | One row per session (= one diff). Use `session_id` to group hunks for ranking. |
| `data/noise_manifest.csv` | 4,529 | Every sample carrying at least one noise type, with ground truth. |
| `data/gold_annotation_sheet.csv` | 900 | **Blind** annotation sheet (labels withheld) for the planned human validation. |
| `data/source_repositories.csv` | 15 | Provenance: repository, licence, split, sample and commit counts, date range. |
| `data/dataset_statistics.json` | — | All counts reported in this README, machine-readable. |
| `data/baseline_results.json` | — | Non-neural baseline results (§7), reproducible via `code/baselines.py`. |
| `licenses/` | 15 | Upstream licence text of every source repository. |
| `code/` | 4 | `mine.py` → `generate.py` → `baselines.py` → `export.py`. Deterministic, `SEED = 473`. |

---

## 2. Schema

| Column | Type | Description |
|---|---|---|
| `sample_id` | str | Primary key, `SIFT-000001` … |
| `split` | str | `train` / `val` / `test` — **repository-disjoint** (§4) |
| `session_id` | str | Groups the hunks of one diff. **Ranking unit.** |
| `position_in_diff`, `n_hunks_in_diff` | int | Position within the session (order is shuffled, not label-ordered) |
| `repo`, `repo_slug`, `language`, `license` | str | Provenance of the hunk |
| `seed_commit_sha` | str | Commit whose message produced the intent |
| `source_commit_sha`, `source_commit_date` | str | Commit the **hunk** came from (differs from `seed_commit_sha` for mutated hunks) |
| `file_path`, `file_status`, `hunk_header`, `hunk` | str | The code change; `hunk` is a unified diff with 3 lines of context |
| `intent_turns` | str | Multi-turn request, ` \|\| `-separated (a list in the JSONL) |
| `consolidated_intent` | str | Denoised single-sentence task specification (target of intent consolidation) |
| **`label`** | str | **Observed** label — `aligned` / `misaligned`. *Train it on this.* |
| **`drift_type`** | str | Observed `none` / `collateral` / `scope_expansion` / `purpose_drift` |
| `label_clean`, `drift_type_clean` | str | **Ground truth.** Identical to `label` except on flipped train rows. *Evaluate on this.* |
| `mutation_difficulty` | str | `hard` / `random` / `n/a` — see §6 |
| `provenance` | str | `commit:<sha>` or `mutation:<type>:donor_commit=<sha>` |
| `noise_types` | str | `\|`-separated: `label_flip`, `non_semantic_hunk`, `intent_turn`, `truncation`, `duplicate_hunk` |
| `label_noise`, `intent_noise_type`, `non_semantic_reason` | — | Per-noise-type ground truth |
| `approx_tokens`, `exceeds_max_len` | int/bool | Estimated subword length of intent + hunk (`len/3.6`), flag for > 1024 |
| `is_duplicate_hunk`, `hunk_sha1` | bool/str | SHA-1 over the `+`/`−` lines only |
| `gold_candidate` | bool | In the 900-sample blind annotation subset |

---

## 3. Labels

**Level 1 — `aligned` vs `misaligned`.** **Level 2 —** for misaligned hunks only:

| Drift type | Construction | Meaning |
|---|---|---|
| `collateral` | Hunk from another commit in the same repo, **different file** | Agent touched an unrelated component |
| `scope_expansion` | Hunk from another commit, **same file**, original hunks kept | Agent did *more* than asked |
| `purpose_drift` | Hunk from another commit, **same file**, and the seed commit's hunk for that file is **removed** | Agent did something *else* to the right component |

Labels are **construction-determined**: an `aligned` hunk is one the seed commit actually contained;
a `misaligned` hunk provably originates from a *different* commit. No annotator judgement is involved,
which removes the subjectivity that limits comment-consistency datasets.

| Split | Hunks | Sessions | Aligned | Misaligned | +ve rate | collateral / scope / purpose | hard / random |
|---|---|---|---|---|---|---|---|
| train | 12,600 | 4,266 | 9,828 | 2,772 | 22.0 % | 1,248 / 915 / 609 | 1,462 / 1,310 |
| val | 2,700 | 921 | 2,106 | 594 | 22.0 % | 267 / 196 / 131 | 277 / 317 |
| test | 2,700 | 924 | 2,106 | 594 | 22.0 % | 267 / 196 / 131 | 261 / 333 |

The 22.0 % positive rate is deliberate and sits between the class balance of the standard JIT
defect-prediction datasets (QT 8 % defective, OpenStack 13 %).

---

## 4. Splits and leakage control

Splitting is **repository-level**, not random. This is not a stylistic choice: CommitBench (Schall et al.,
IEEE, 2024) shows that MCMD draws ~2M commits from only 500 repositories, so training and test commits are
not independent, and that model performance degrades by **up to 73.41 %** when a repository-based split is
used instead of a random one. A random split of this dataset would therefore overstate accuracy.

* train — requests, flask, click, rich, black, express, fastify
* val — typer, sqlmodel, got, koa, chalk
* test — httpx, attrs, axios

Every split contains both Python and JavaScript/TypeScript. Verified after construction:

* repositories spanning more than one split: **0**
* `(consolidated_intent, hunk)` pairs shared across splits: **0**
* full hunk texts (with context) shared across splits: **1**
* changed-line bodies recurring across splits: 29 — all trivial one-liners (e.g. `+import sys`,
  a CI action version bump) that differ in file, context and intent. Flagged via `is_duplicate_hunk`.

---

## 5. Noise — deliberately included, with ground truth

Real training data for this task is noisy. Rather than hide that, v1.0 **injects or detects five noise
types and ships the ground truth for each**, so the dataset doubles as a *denoising benchmark*.

| Noise type | Rows | % | Ground-truth column | Origin |
|---|---|---|---|---|
| `intent_turn` | 2,730 | 15.2 % | `intent_noise_type` | Chit-chat turn, a retracted decoy request (`"Actually, scratch that…"`), or a repeated turn injected into `intent_turns`. `consolidated_intent` stays clean. |
| `duplicate_hunk` | 937 | 5.2 % | `is_duplicate_hunk` | Identical changed-line bodies (cherry-picks, backports, trivial one-liners) |
| `non_semantic_hunk` | 714 | 4.0 % | `non_semantic_reason` | whitespace-only 343, lockfile 202, version-bump-only 143, generated/minified 26 |
| `label_flip` | 569 | 3.2 % | `label_noise`, `label_clean` | **Train split only.** Class-conditional: P(flip \| aligned) = 0.03, P(flip \| misaligned) = 0.10. Observed directions: 285 aligned→misaligned, 284 misaligned→aligned. |
| `truncation` | 198 | 1.1 % | `exceeds_max_len` | Intent + hunk exceeds 1,024 estimated subword tokens |

4,529 of 18,000 rows (25.2 %) carry at least one noise type. **Val and test labels are clean** — noisy
labels appear only in train, following the convention of noisy-label benchmarks so that evaluation remains
trustworthy.

**How to use it.** Train on `label`; evaluate on `label_clean`. To benchmark a denoising method (e.g.
confident learning), score its flagged set against `label_noise` — the 569 flips are exactly known.

**Noise that is *not* quantified.** Two sources are real but unmeasured in v1.0, and the 900-sample
annotation sheet exists to estimate them:
1. **Tangled commits** — a seed commit may itself bundle unrelated changes, so an `aligned` hunk may not
   serve the intent (Herzig & Zeller, *The Impact of Tangled Code Changes*, MSR 2013).
2. **Residual mutation noise** — a donor hunk may coincidentally be relevant to the intent despite the
   dissimilarity guard.

---

## 6. Difficulty tiers (`mutation_difficulty`)

Donor hunks are mined in two tiers:

* **`random`** — the first admissible donor. Typically lexically distant from the intent → the easy tier.
* **`hard`** — among admissible donors, the one maximising **size-normalised identifier overlap** with the
  intent, restricted to donors of comparable size to the session's hunks. These *look* relevant but
  provably come from another commit.

Both constraints matter. Without size-normalisation, hard mining silently selected larger hunks and
**hunk size alone** reached 0.68 pairwise AUC; after normalisation it is 0.49 (chance). Report results
per tier — a model that only solves the `random` tier has learned lexical overlap, not alignment.

---

## 7. Baseline results (real, reproducible: `code/baselines.py`)

Non-neural baselines on the **test** split. NDCG@5 over the 447 *rankable* sessions (≥ 2 hunks, ≥ 1
misaligned); ties broken randomly and averaged over 50 seeds.

| Baseline | NDCG@5 |
|---|---|
| B1 Random ordering | 0.7236 |
| B2 Lines changed in diff (diff-level scalar) | **0.7236** |
| B3 Hunk size (per-hunk) | 0.7206 |
| B4 TF-IDF dissimilarity | 0.8295 |
| B5 Logistic regression (4 features) | 0.8159 |

> **B2 equals B1 exactly.** A diff-level scalar is *constant within a diff*, so it cannot order hunks inside
> one — its discriminative power for this task is not weak, it is zero by construction. This is the
> empirical case for a model that reads intent and code jointly.
>
> Note NDCG@5 has a **high floor (0.72)** here because the median session has 2 hunks. Report
> per-tier pairwise AUC as well; do not compare these numbers to NDCG on datasets with longer lists.

Within-session pairwise AUC (misaligned ranked above aligned), by tier:

| Baseline | `random` tier | `hard` tier |
|---|---|---|
| B3 Hunk size | 0.479 | 0.493 |
| B4 TF-IDF dissimilarity | 0.812 | **0.551** |
| B5 Logistic regression | 0.777 | **0.551** |

Lexical and size features essentially solve the `random` tier and collapse to near chance (0.50) on the
`hard` tier. The `hard` tier is the part of this dataset that requires semantic modelling.

B5 classification on test (threshold tuned on val): macro-F1 0.698, precision 0.506, recall 0.574,
specificity 0.842.

---

## 8. Collection methodology

1. **Mine** — 15 GitHub repositories with redistribution-permitting licences (MIT / BSD-3 / Apache-2.0),
   bare-cloned; all non-merge commits parsed with `git log -p -M --unified=3` into file- and hunk-level
   records. 64,914 candidate hunks from 24,165 commits.
2. **Filter** — drop merges, releases, reverts, version bumps and changelog commits; subjects of < 3 words
   or > 120 chars; commits touching > 8 files or > 12 hunks; hunks over ~4,000 estimated tokens. Retain
   code and configuration files. These filters follow the quality practices CommitBench establishes for
   commit datasets (bot/release exclusion, duplicate awareness, redistributable licences).
3. **Split** the repositories (§4), then partition each split's commits into a **seed pool (75 %)** and a
   disjoint **donor pool (25 %)** so a mutated hunk never comes from a commit used as a seed.
4. **Build sessions** — the seed commit's hunks become `aligned`; the intent is derived from its subject
   plus, when informative, one sentence of its body.
5. **Mutate** — inject donor hunks under the three drift definitions (§3), guarded by a subject-similarity
   constraint (token Jaccard < 0.2) so the donor is not doing the same job as the intent.
6. **Inject noise** with ground truth (§5); shuffle hunk order within each session.
7. **Budget-exact assembly** — sessions are accepted only while they fit the remaining per-class budget,
   so all split and drift-type totals land exactly on target.
8. **Quality control** — leakage audit (§4), baseline probing (§7), shortcut removal (§6).

**Data provenance is fully traceable**: every hunk carries its repository, commit SHA and date, and every
upstream licence is included in `licenses/`.

---

## 9. Real vs synthetic vs human-validated — stated plainly

| Component | Status in v1.0 |
|---|---|
| Code hunks | **Real.** Every hunk is a verbatim change from a real commit in a public repository. |
| `aligned` labels | **Real**, derived from commit membership (subject to tangled-commit noise, §5). |
| `misaligned` labels | **Synthetic by construction** — real code recombined across commits. The *code* is real; the *pairing* is synthetic. |
| Intents | **Derived**, not captured. Built from real commit messages and given a synthetic multi-turn structure. They are not transcripts of real agent sessions. |
| Injected noise | **Synthetic**, fully specified and reversible. |
| Human validation | **Not yet performed.** `gold_annotation_sheet.csv` (900 blind samples) is provided for it. No human-validated labels are claimed in v1.0. |

Planned for v2.0: a real agent-session subset captured from instrumented coding-agent runs, plus the
completed human annotations with inter-annotator agreement.

---

## 10. Limitations

* **Derived intents.** Commit messages are written *after* the change; a real prompt is written *before*.
  The two differ in specificity and phrasing. This is the dataset's main construct-validity limitation.
* **Synthetic misalignment.** Recombination across commits is a proxy for genuine agent drift. It
  reproduces the *structure* of drift (right file, wrong purpose) but not an agent's error distribution.
* **Unmeasured label noise** from tangled commits and residual donor relevance (§5).
* **Language and domain coverage** — Python, JavaScript and TypeScript library/framework code only. No
  Java, Go, C++, and no application or enterprise codebases.
* **Short sessions.** Median 2 hunks, max 14 — smaller than the 40-hunk diffs large agent runs produce.
  NDCG has a high floor as a result (§7).
* **Not a defect dataset.** `misaligned` means *not what was asked*, not *buggy*. A misaligned hunk is
  usually correct code in the wrong place.

---

## 11. Licence and citation

Dataset compilation and generation code: **CC BY 4.0**. Code hunks remain under their upstream licences
(MIT / BSD-3-Clause / Apache-2.0), reproduced in `licenses/`; each sample records its origin repository,
licence and commit SHA. Only repositories whose licences permit redistribution were used.

Reproduce end to end (deterministic, `SEED = 473`):

```bash
python code/mine.py        # bare clones -> mined_hunks.jsonl
python code/generate.py    # sessions, mutations, noise -> built.json
python code/baselines.py   # baseline_results.json
python code/export.py      # this release
```
