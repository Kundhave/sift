"""Evaluate rule-based intent consolidation against intent_noise_type ground truth."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sift.data import load_all
from sift.consolidate import consolidate_detailed, split_turns

KINDS = ["none", "chit_chat", "retraction", "repeated_turn"]
res = {"per_split": {}, "per_noise_type": {}, "overall": {}, "errors_sample": []}
rows = []
for sp, df in load_all().items():
    for r in df.itertuples():
        out, found = consolidate_detailed(split_turns(r.intent_turns))
        rows.append((sp, r.intent_noise_type, out == r.consolidated_intent, found, r.session_id, r.intent_turns, out, r.consolidated_intent))
def agg(sub):
    n = len(sub); return {"n": n, "exact_match": round(sum(x[2] for x in sub) / n, 4) if n else None}
res["overall"] = agg(rows)
for sp in ("train", "val", "test"): res["per_split"][sp] = agg([x for x in rows if x[0] == sp])
for k in KINDS:
    sub = [x for x in rows if x[1] == k]
    d = agg(sub)
    if k == "none":
        d["false_positive_rate"] = round(sum(bool(x[3]) for x in sub) / len(sub), 4)  # flagged noise on clean rows
    else:
        d["detection_recall"] = round(sum(k in x[3] for x in sub) / len(sub), 4)
    res["per_noise_type"][k] = d
res["errors_sample"] = [{"session": x[4], "turns": x[5][:200], "got": x[6][:100], "want": x[7][:100]} for x in rows if not x[2]][:5]
res["n_errors"] = sum(not x[2] for x in rows)
res["note"] = "Rules were written from visible noise templates; results are in-distribution for Sift-IA v1."
Path(__file__).parent.joinpath("results/consolidation.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("overall", "per_noise_type", "n_errors")}, indent=1))
