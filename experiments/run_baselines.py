"""Phase 2: re-run B1-B5 and compare with the released numbers (CLAUDE.md s5 / data/baseline_results.json)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sift.data import load_all, ROOT
from sift.baselines import run_baselines

d = load_all()
res = run_baselines(d["train"], d["val"], d["test"])
rel = json.load(open(ROOT / "data/baseline_results.json"))
res["released"] = rel
res["comparison"] = {
    "ndcg@5_diff": {k: round(res["ndcg@5"][k] - v, 4) for k, v in zip(res["ndcg@5"], rel["ndcg@5"].values())},
    "rankable_sessions_here": res["rankable_sessions"], "rankable_sessions_released_json": rel["rankable_test_sessions"],
    "rankable_sessions_in_CLAUDE_md": 447,
}
json.dump(res, open(ROOT / "experiments/results/baselines.json", "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "released"}, indent=1))
