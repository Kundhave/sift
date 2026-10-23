"""Phase 1: verify split sizes/counts against CLAUDE.md section 4 and the dataset README. Writes results JSON."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sift.data import load_all, ROOT

dfs = load_all()
out = {"splits": {}, "checks": {}}
for s, d in dfs.items():
    mis = d[d.label_clean == "misaligned"]
    out["splits"][s] = {
        "repos": sorted(d.repo.unique()), "sessions": int(d.session_id.nunique()), "hunks": len(d),
        "aligned": int((d.label_clean == "aligned").sum()), "misaligned": len(mis),
        "drift": mis.drift_type_clean.value_counts().to_dict(),
        "tiers": mis.mutation_difficulty.value_counts().to_dict(),
        "languages": d.language.value_counts().to_dict(),
        "median_hunks_per_session": float(d.groupby("session_id").size().median()),
        "max_hunks_per_session": int(d.groupby("session_id").size().max()),
    }
allr = [set(d.repo) for d in dfs.values()]
out["checks"]["repos_shared_across_splits"] = len(allr[0] & allr[1]) + len(allr[0] & allr[2]) + len(allr[1] & allr[2])
import pandas as pd
a = pd.concat(dfs.values())
out["checks"]["total_hunks"] = len(a)
out["checks"]["total_sessions"] = int(a.session_id.nunique())
out["checks"]["distinct_source_commits"] = int(a.seed_commit_sha.nunique())
out["checks"]["language_counts"] = a.language.value_counts().to_dict()
tr = dfs["train"]
out["checks"]["train_label_flips"] = int(tr.label_noise.sum())
out["checks"]["train_label_ne_clean"] = int((tr.label != tr.label_clean).sum())
out["checks"]["val_test_label_eq_clean"] = bool(all((dfs[s].label == dfs[s].label_clean).all() for s in ("val", "test")))
out["checks"]["noise_rows_any"] = int((a.noise_types != "").sum())
out["checks"]["noise_rows_any_notnone"] = int((~a.noise_types.isin(["", "none"])).sum())
out["checks"]["noise_types_values_sample"] = a.noise_types.value_counts().head(8).to_dict()
for col in ("intent_noise_type", "non_semantic_reason"):
    out["checks"][col] = a[col].value_counts().to_dict()
out["checks"]["is_duplicate_hunk"] = int(a.is_duplicate_hunk.sum())
out["checks"]["exceeds_max_len"] = int(a.exceeds_max_len.sum())
out["checks"]["approx_tokens_median_p95_max"] = [float(a.approx_tokens.median()), float(a.approx_tokens.quantile(.95)), float(a.approx_tokens.max())]
json.dump(out, open(ROOT / "experiments/results/data_verification.json", "w"), indent=2)
print(json.dumps(out, indent=1))
