"""Metamorphic testing: output relations that must hold under meaning-preserving input changes.
Run on a deterministic 120-pair test-split sample (60 aligned + 60 misaligned) with the trained model.
Tolerances were fixed before the first run."""
import numpy as np
import pytest
from conftest import record
from sift.perturb import rename_identifier, add_context_comment, add_trailing_whitespace_context
from sift.data import split_turns

pytestmark = pytest.mark.needs_artifact


def _probs(scorer, intents, hunks):
    a, _, _ = scorer._logits(intents, hunks)
    return 1 / (1 + np.exp(-a / scorer.T))


def _texts(df):
    return list(df.consolidated_intent), [f"{p}\n{h}" for p, h in zip(df.file_path, df.hunk)]


def _rel(scorer, name, p0, p1, flip_tol, dp_tol=None):
    thr = scorer.threshold
    flips = float(np.mean((p0 >= thr) != (p1 >= thr)))
    record(f"mr_{name}", {"n": int(len(p0)), "flip_rate": flips, "mean_abs_dp": float(np.mean(np.abs(p0 - p1))),
                          "max_abs_dp": float(np.max(np.abs(p0 - p1))), "tolerance_flip_rate": flip_tol})
    return flips


def test_mr1_determinism(scorer, test_samples):
    i, h = _texts(test_samples)
    p0, p1 = _probs(scorer, i, h), _probs(scorer, i, h)
    record("mr_determinism", {"n": len(p0), "max_abs_dp": float(np.max(np.abs(p0 - p1)))})
    assert np.array_equal(p0, p1)


def test_mr2_batch_composition_independence(scorer, test_samples):
    """Scoring each pair alone vs in a batch must agree (padding/bucketing must not leak)."""
    i, h = _texts(test_samples.head(40))
    pb = _probs(scorer, i, h)
    ps = np.concatenate([_probs(scorer, [a], [b]) for a, b in zip(i, h)])
    record("mr_batch_independence", {"n": 40, "max_abs_dp": float(np.max(np.abs(pb - ps))), "mean_abs_dp": float(np.mean(np.abs(pb - ps)))})
    assert np.max(np.abs(pb - ps)) < 0.05


def test_mr3_reordering_hunks_in_session_keeps_each_score(scorer, data):
    t = data["test"]; sess = [s for s, g in t.groupby("session_id") if 3 <= len(g) <= 6][:25]
    worst = 0.0
    for s in sess:
        g = t[t.session_id == s]
        i, h = _texts(g)
        p = _probs(scorer, i, h)
        order = np.random.RandomState(473).permutation(len(g))
        q = _probs(scorer, [i[k] for k in order], [h[k] for k in order])
        worst = max(worst, float(np.max(np.abs(p[order] - q))))
    record("mr_reorder", {"sessions": len(sess), "max_abs_dp": worst})
    assert worst < 0.05


def test_mr4_context_comment_does_not_flip(scorer, test_samples):
    i, h = _texts(test_samples)
    h2 = [add_context_comment(x, p) for x, p in zip(h, test_samples.file_path)]
    f = _rel(scorer, "context_comment", _probs(scorer, i, h), _probs(scorer, i, h2), 0.10)
    assert f <= 0.10


def test_mr5_trailing_whitespace_in_context_does_not_flip(scorer, test_samples):
    i, h = _texts(test_samples)
    h2 = [add_trailing_whitespace_context(x) for x in h]
    f = _rel(scorer, "context_whitespace", _probs(scorer, i, h), _probs(scorer, i, h2), 0.10)
    assert f <= 0.10


def test_mr6_consistent_identifier_rename_does_not_flip(scorer, test_samples):
    i, h, i2, h2 = [], [], [], []
    for (_, r), a, b in zip(test_samples.iterrows(), *_texts(test_samples)):
        res = rename_identifier(a, b, seed=473)
        if res:
            i.append(a); h.append(b); i2.append(res[0]); h2.append(res[1])
    f = _rel(scorer, "identifier_rename", _probs(scorer, i, h), _probs(scorer, i2, h2), 0.10)
    assert len(i) > 80 and f <= 0.10


def test_mr7_removing_chitchat_turns_does_not_flip(scorer, data):
    t = data["test"]; g = t[t.intent_noise_type == "chit_chat"].head(60)
    with_noise = [scorer_intent(scorer, x) for x in g.intent_turns]
    clean = list(g.consolidated_intent)
    h = [f"{p}\n{x}" for p, x in zip(g.file_path, g.hunk)]
    f = _rel(scorer, "chitchat_removal", _probs(scorer, with_noise, h), _probs(scorer, clean, h), 0.05)
    assert f <= 0.05


def scorer_intent(scorer, turns):
    from sift.data import consolidate_intent
    return consolidate_intent(split_turns(turns))


def test_mr8_sensitivity_swapping_intent_between_sessions_raises_misalignment(scorer, test_samples):
    """Directional relation: pairing an aligned hunk with another session's intent should raise p on average."""
    al = test_samples[test_samples.label_clean == "aligned"]
    i, h = _texts(al)
    shifted = i[1:] + i[:1]
    p0, p1 = _probs(scorer, i, h), _probs(scorer, shifted, h)
    record("mr_intent_swap_sensitivity", {"n": len(p0), "mean_p_original": float(p0.mean()), "mean_p_swapped": float(p1.mean())})
    assert p1.mean() > p0.mean()
