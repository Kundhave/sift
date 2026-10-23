"""Save per-hunk test scores of B4 (TF-IDF dissimilarity) and B5 (logistic regression) for the PR/ROC/NDCG@K curves.
Same code path as sift.baselines.run_baselines (replicated, since it does not return scores). Writes
experiments/results/baseline_scores.npz (order = load_split('test')) and prints NDCG@5 for a cross-check with baselines.json."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from sklearn.linear_model import LogisticRegression
from sift.data import load_split
from sift.baselines import LexicalFeatures
from sift import rank as R

tr, te = load_split("train"), load_split("test")
lf = LexicalFeatures(tr); Xtr, _, _ = lf(tr); Xte, sim, _ = lf(te)
lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Xtr, (tr.label == "misaligned").values.astype(int))
b5 = lr.predict_proba(Xte)[:, 1]; b4 = 1 - sim
y = (te.label_clean == "misaligned").values.astype(int); sid = list(te.session_id)
np.savez(ROOT / "experiments/results/baseline_scores.npz", B4=b4, B5=b5, test_ids=np.array(te.session_id.astype(str)))
ref = json.load(open(ROOT / "experiments/results/baselines.json"))["ndcg@5"]
for n, s in (("B4 TF-IDF dissimilarity", b4), ("B5 Logistic regression", b5)):
    print(n, "recomputed", round(R.rank_metrics(s, y, sid, k=5, n_seeds=50)["ndcg@5"], 4), "baselines.json", ref[n])
