"""Denoising: rule filters, confident learning (cleanlab) scored against the known label flips, co-teaching."""
from __future__ import annotations
import numpy as np


def rule_filter_mask(df):
    """True for rows removed by the rule-based stage (non-semantic hunks, repeated hunks after the first)."""
    ns = (df.non_semantic_reason != "none").values
    dup = df.duplicated("hunk_sha1", keep="first").values
    return ns | dup


def score_flag_set(flagged_mask, true_flip_mask):
    """Precision / recall / F1 of a flagged set against the shipped ground-truth flips (`label_noise`)."""
    f, t = np.asarray(flagged_mask, bool), np.asarray(true_flip_mask, bool)
    tp = int((f & t).sum()); fp = int((f & ~t).sum()); fn = int((~f & t).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"flagged": int(f.sum()), "true_flips": int(t.sum()), "tp": tp, "fp": fp, "fn": fn, "precision": pr,
            "recall": rc, "f1": (2 * pr * rc / (pr + rc)) if pr + rc else 0.0}


def confident_learning_issues(labels, oof_prob_misaligned, **kw):
    """Label issues via cleanlab.filter.find_label_issues given OUT-OF-FOLD probabilities."""
    from cleanlab.filter import find_label_issues
    p1 = np.clip(np.asarray(oof_prob_misaligned, float), 1e-6, 1 - 1e-6)
    probs = np.c_[1 - p1, p1]
    return find_label_issues(np.asarray(labels, int), probs, **kw)


def co_teaching_forget_rate(epoch, final_rate=0.1, ramp=5):
    return final_rate * min(1.0, epoch / ramp)


def co_teaching_select(loss_a, loss_b, forget_rate):
    """Han et al. 2018: each network keeps the (1 - forget_rate) smallest-loss samples *selected by its peer*."""
    import torch
    n = loss_a.numel(); k = max(1, int(round(n * (1 - forget_rate))))
    keep_a = torch.argsort(loss_a)[:k]   # small-loss according to A -> used to update B
    keep_b = torch.argsort(loss_b)[:k]
    return keep_a, keep_b
