"""Ranking and classification metrics (NDCG@K, P@K, MRR, within-session pairwise AUC, macro-F1, ECE)."""
from __future__ import annotations
import math
import random
from collections import defaultdict
import numpy as np


def ndcg_at_k(scores, rel, k=5):
    order = np.argsort(-np.asarray(scores, float), kind="stable")
    g = sum(rel[i] / math.log2(p + 2) for p, i in enumerate(order[:k]))
    ig = sum(v / math.log2(p + 2) for p, v in enumerate(sorted(rel, reverse=True)[:k]))
    return g / ig if ig else None


def precision_at_k(scores, rel, k=1):
    order = np.argsort(-np.asarray(scores, float), kind="stable")[:k]
    return float(np.mean([rel[i] for i in order]))


def mrr(scores, rel):
    order = np.argsort(-np.asarray(scores, float), kind="stable")
    for p, i in enumerate(order):
        if rel[i]:
            return 1.0 / (p + 1)
    return 0.0


def session_index(session_ids):
    d = defaultdict(list)
    for i, s in enumerate(session_ids):
        d[s].append(i)
    return d


def rankable(sess_idx, y):
    return [ix for ix in sess_idx.values() if len(ix) >= 2 and any(y[j] for j in ix)]


def rank_metrics(scores, y, session_ids, k=5, n_seeds=20):
    """Mean NDCG@K / P@1 / MRR over rankable sessions; ties broken randomly (averaged over seeds)."""
    y = np.asarray(y, int)
    scores = np.asarray(scores, float)
    groups = rankable(session_index(session_ids), y)
    nd, p1, rr = [], [], []
    for s in range(n_seeds):
        rng = random.Random(s)
        for ix in groups:
            sc = [scores[j] + rng.random() * 1e-9 for j in ix]
            rel = [int(y[j]) for j in ix]
            nd.append(ndcg_at_k(sc, rel, k)); p1.append(precision_at_k(sc, rel, 1)); rr.append(mrr(sc, rel))
    return {f"ndcg@{k}": float(np.mean(nd)), "p@1": float(np.mean(p1)), "mrr": float(np.mean(rr)),
            "n_rankable_sessions": len(groups)}


def pairwise_auc_by_tier(scores, y, session_ids, tiers):
    """Within-session pairwise AUC: P(score(misaligned) > score(aligned)) per mutation tier."""
    y = np.asarray(y, int)
    out = {}
    sess = session_index(session_ids)
    for tier in ("random", "hard"):
        wins = tot = 0.0
        for ix in sess.values():
            al = [j for j in ix if not y[j]]
            mi = [j for j in ix if y[j] and tiers[j] == tier]
            for m in mi:
                for a in al:
                    tot += 1
                    wins += 1 if scores[m] > scores[a] else (0.5 if scores[m] == scores[a] else 0)
        out[tier] = {"auc": (wins / tot if tot else None), "pairs": int(tot)}
    return out


def cls_metrics(y, pred):
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
    y = np.asarray(y, int); pred = np.asarray(pred, int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "f1_aligned": float(f1_score(y, pred, pos_label=0)), "f1_misaligned": float(f1_score(y, pred, pos_label=1)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def best_threshold(y, p, grid=None):
    from sklearn.metrics import f1_score
    grid = np.linspace(0.02, 0.98, 97) if grid is None else grid
    return float(max(grid, key=lambda t: f1_score(y, np.asarray(p) >= t, average="macro")))


def ece(y, p, n_bins=15):
    y = np.asarray(y, float); p = np.asarray(p, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def full_eval(scores, y, session_ids, tiers, threshold, k=5):
    """One-stop evaluation block used by every model/ablation config."""
    scores = np.asarray(scores, float)
    out = cls_metrics(y, scores >= threshold)
    out["threshold"] = float(threshold)
    out.update(rank_metrics(scores, y, session_ids, k=k))
    out["pairwise_auc"] = pairwise_auc_by_tier(scores, y, session_ids, tiers)
    out["ece"] = ece(y, scores)
    return out
