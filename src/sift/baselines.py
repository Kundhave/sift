"""Non-neural baselines B1-B5 re-implemented on the released CSVs (follows data/builder_code/baselines.py)."""
from __future__ import annotations
import math, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .data import changed_text, n_changed
from .rank import rank_metrics, pairwise_auc_by_tier, cls_metrics, best_threshold


def _tok(s): return re.findall(r"[A-Za-z_][A-Za-z0-9_]+", s)


def ident_split(s):
    out = []
    for t in _tok(s):
        out += [p.lower() for p in re.sub(r"([a-z])([A-Z])", r"\1 \2", t).replace("_", " ").split() if len(p) > 2]
    return out


class LexicalFeatures:
    def __init__(self, train_df):
        self.vec = TfidfVectorizer(tokenizer=ident_split, lowercase=False, token_pattern=None, min_df=2, sublinear_tf=True)
        self.vec.fit(list(train_df.consolidated_intent) + [changed_text(h) + " " + p for h, p in zip(train_df.hunk, train_df.file_path)])

    def cos(self, a, b):
        A, B = self.vec.transform(a), self.vec.transform(b)
        num = np.asarray(A.multiply(B).sum(1)).ravel()
        den = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()) * np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel())
        return num / np.where(den == 0, 1, den)

    def __call__(self, df):
        sim = self.cos(list(df.consolidated_intent), [changed_text(h) + " " + p for h, p in zip(df.hunk, df.file_path)])
        size = np.array([math.log1p(n_changed(h)) for h in df.hunk])
        pov = np.array([len(set(ident_split(p)) & set(ident_split(i))) > 0 for p, i in zip(df.file_path, df.consolidated_intent)], float)
        ds = np.array([math.log1p(n) for n in df.n_hunks_in_diff])
        return np.c_[sim, size, pov, ds], sim, size


def run_baselines(tr, va, te):
    lf = LexicalFeatures(tr)
    Xtr, _, _ = lf(tr); Xva, _, _ = lf(va); Xte, sim_te, size_te = lf(te)
    ytr = (tr.label == "misaligned").values.astype(int)           # observed (noisy) labels
    yva = (va.label_clean == "misaligned").values.astype(int)
    yte = (te.label_clean == "misaligned").values.astype(int)
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Xtr, ytr)
    p_va, p_te = lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1]
    thr = best_threshold(yva, p_va, np.linspace(0.05, 0.95, 91))
    sid, tiers = list(te.session_id), list(te.mutation_difficulty)
    diff_lines = te.assign(_n=[n_changed(h) for h in te.hunk]).groupby("session_id")._n.transform("sum").values
    scores = {"B1 Random order": np.zeros(len(te)), "B2 Diff lines changed": diff_lines.astype(float),
              "B3 Hunk size": size_te, "B4 TF-IDF dissimilarity": 1 - sim_te, "B5 Logistic regression": p_te}
    out = {"ndcg@5": {}, "pairwise_auc": {}, "rankable_sessions": None}
    for k, s in scores.items():
        rm = rank_metrics(s, yte, sid, k=5, n_seeds=50)
        out["ndcg@5"][k] = rm["ndcg@5"]; out["rankable_sessions"] = rm["n_rankable_sessions"]
        if k not in ("B1 Random order", "B2 Diff lines changed"):
            out["pairwise_auc"][k] = {t: v["auc"] for t, v in pairwise_auc_by_tier(s, yte, sid, tiers).items()}
    out["pairs_per_tier"] = {t: v["pairs"] for t, v in pairwise_auc_by_tier(size_te, yte, sid, tiers).items()}
    out["B5_classification_test"] = {**cls_metrics(yte, p_te >= thr), "threshold_from_val": thr}
    return out
