# Discrepancies between CLAUDE.md and the released data (found in Phases 1-2)

| Item | CLAUDE.md | Measured from released files |
|---|---|---|
| Hunks by language | Python 11,900 / JS 5,076 / TS 1,024 | Python 10,205 / JavaScript 6,726 / TypeScript 1,069 (experiments/results/data_verification.json) |
| TypeScript in test split | implied present | **none**: TypeScript appears only in val (1,069). Test is 1,686 Python + 1,014 JavaScript. Train is Python + JavaScript only. |
| Rankable test sessions | 447 | **456** (same in data/baseline_results.json and in our re-run) |
| Distinct source commits (8,016) | 8,016 | Confirmed as distinct `source_commit_sha`; distinct seed commits = 6,111 |
| Everything else in sections 4 and 5 | | Matches (split sizes, label/drift/tier counts, noise counts, NDCG@5 and AUC to 4 d.p.) |
