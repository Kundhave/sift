#!/usr/bin/env python3
"""Stage 3 - non-neural baselines on the released test split (real numbers, no DL yet).

Ranking metric: NDCG@5 over 'rankable' test sessions (>=2 hunks and >=1 misaligned).
Ties are broken randomly and averaged over 50 seeds, so a constant score = random ordering.
Classification metrics use the clean test labels.
"""
import json, math, random, re
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

B = json.load(open("/home/claude/sift_ds/built.json")); rows = B["rows"]
tr = [r for r in rows if r["split"] == "train"]
va = [r for r in rows if r["split"] == "val"]
te = [r for r in rows if r["split"] == "test"]

def changed_text(h): return "\n".join(l[1:] for l in h.split("\n")[1:] if l[:1] in "+-")
def n_changed(h): return sum(1 for l in h.split("\n")[1:] if l[:1] in "+-")
def tok(s): return re.findall(r"[A-Za-z_][A-Za-z0-9_]+", s)
def ident_split(s):
    out = []
    for t in tok(s):
        out += [p.lower() for p in re.sub(r"([a-z])([A-Z])", r"\1 \2", t).replace("_", " ").split() if len(p) > 2]
    return out

vec = TfidfVectorizer(tokenizer=ident_split, lowercase=False, token_pattern=None, min_df=2, sublinear_tf=True)
vec.fit([r["consolidated_intent"] for r in tr] + [changed_text(r["hunk"]) + " " + r["file_path"] for r in tr])

def cos(a, b):
    A = vec.transform(a); Bm = vec.transform(b)
    num = np.asarray(A.multiply(Bm).sum(1)).ravel()
    den = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()) * np.sqrt(np.asarray(Bm.multiply(Bm).sum(1)).ravel())
    return num / np.where(den == 0, 1, den)

def feats(R):
    sim = cos([r["consolidated_intent"] for r in R], [changed_text(r["hunk"]) + " " + r["file_path"] for r in R])
    size = np.array([math.log1p(n_changed(r["hunk"])) for r in R])
    path_ov = np.array([len(set(ident_split(r["file_path"])) & set(ident_split(r["consolidated_intent"]))) > 0 for r in R], float)
    dsize = np.array([math.log1p(r["n_hunks_in_diff"]) for r in R])
    return np.c_[sim, size, path_ov, dsize], sim, size

Xtr, _, _ = feats(tr); ytr = np.array([r["label"] == "misaligned" for r in tr])          # observed (noisy) labels
Xte, sim_te, size_te = feats(te); yte = np.array([r["label_clean"] == "misaligned" for r in te])
Xva, _, _ = feats(va); yva = np.array([r["label_clean"] == "misaligned" for r in va])

lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Xtr, ytr)
p_te = lr.predict_proba(Xte)[:, 1]
# threshold chosen on validation for macro-F1
p_va = lr.predict_proba(Xva)[:, 1]
ths = np.linspace(0.05, 0.95, 91)
best_t = max(ths, key=lambda t: f1_score(yva, p_va >= t, average="macro"))

def ndcg_at_k(scores, rel, k=5):
    order = np.argsort(-np.asarray(scores), kind="stable")
    g = sum(rel[i] / math.log2(pos + 2) for pos, i in enumerate(order[:k]))
    ideal = sorted(rel, reverse=True)
    ig = sum(v / math.log2(pos + 2) for pos, v in enumerate(ideal[:k]))
    return g / ig if ig else None

sess = defaultdict(list)
for i, r in enumerate(te): sess[r["session_id"]].append(i)
rankable = [ix for ix in sess.values() if len(ix) >= 2 and any(yte[j] for j in ix)]

def eval_rank(score_fn, n_seeds=50):
    vals = []
    for s in range(n_seeds):
        rr = random.Random(s)
        for ix in rankable:
            sc = [score_fn(j) + rr.random() * 1e-9 for j in ix]
            v = ndcg_at_k(sc, [int(yte[j]) for j in ix]);  vals.append(v)
    return float(np.mean(vals))

commit_lines = defaultdict(int)
for j, r in enumerate(te): commit_lines[r["session_id"]] += n_changed(r["hunk"])

res = {
 "B1 Random ordering":                     eval_rank(lambda j: 0.0),
 "B2 Lines changed in diff (diff-level)":  eval_rank(lambda j: commit_lines[te[j]["session_id"]]),
 "B3 Hunk size (per-hunk lines changed)":  eval_rank(lambda j: size_te[j]),
 "B4 TF-IDF dissimilarity (1 - cos)":      eval_rank(lambda j: 1 - sim_te[j]),
 "B5 Logistic regression (4 features)":    eval_rank(lambda j: p_te[j]),
}
pred = p_te >= best_t
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
cls = {"threshold_from_val": round(float(best_t), 2),
       "macro_f1": round(f1_score(yte, pred, average="macro"), 4),
       "precision_pos": round(precision_score(yte, pred), 4),
       "recall_pos": round(recall_score(yte, pred), 4),
       "specificity": round(tn / (tn + fp), 4)}
out = {"ndcg@5": {k: round(v, 4) for k, v in res.items()}, "rankable_test_sessions": len(rankable),
       "B5_classification_test": cls}
json.dump(out, open("/home/claude/sift_ds/baseline_results.json", "w"), indent=2)
print(json.dumps(out, indent=2))

# ---- tier analysis: within-session pairwise AUC (misaligned scored above aligned)
def pair_auc(score, tier):
    wins = tot = 0
    for ix in sess.values():
        al = [j for j in ix if not yte[j]]
        mi = [j for j in ix if yte[j] and te[j]["mutation_difficulty"] == tier]
        for m in mi:
            for a in al:
                tot += 1; wins += 1 if score[m] > score[a] else (0.5 if score[m] == score[a] else 0)
    return round(wins / tot, 4) if tot else None, tot
tier = {}
for name, sc in [("B3 hunk size", size_te), ("B4 TF-IDF dissimilarity", 1 - sim_te), ("B5 logistic regression", p_te)]:
    tier[name] = {t: pair_auc(sc, t) for t in ("random", "hard")}
out["within_session_pairwise_auc_by_tier"] = {k: {t: v[0] for t, v in d.items()} for k, d in tier.items()}
out["pairs_per_tier"] = {t: pair_auc(size_te, t)[1] for t in ("random", "hard")}
from collections import Counter
out["mutation_difficulty_counts_test"] = dict(Counter(r["mutation_difficulty"] for r in te if r["label_clean"] == "misaligned"))
json.dump(out, open("/home/claude/sift_ds/baseline_results.json", "w"), indent=2)
print(json.dumps({k: out[k] for k in ["within_session_pairwise_auc_by_tier", "pairs_per_tier", "mutation_difficulty_counts_test"]}, indent=2))
