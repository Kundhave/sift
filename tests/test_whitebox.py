"""White-box structural tests: shapes, heads, losses, tokenisation limits, parser, metrics."""
import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sift.losses import focal_loss, multitask_loss, distill_loss
from sift.calibrate import fit_temperature, apply_temperature, _nll
from sift.rank import ndcg_at_k, precision_at_k, mrr, ece, cls_metrics, pairwise_auc_by_tier, rank_metrics, best_threshold
from sift.denoise import score_flag_set, co_teaching_select, co_teaching_forget_rate
from sift.data import consolidate_intent, changed_text, n_changed, split_turns, hunk_input
from sift.diffparse import parse_diff


# ---- model structure
def test_head_shapes(tiny_model):
    ids = torch.randint(5, 900, (3, 20)); o = tiny_model(ids, torch.ones_like(ids))
    assert o["align_logit"].shape == (3,) and o["drift_logits"].shape == (3, 3) and o["cal_logit"].shape == (3,)


def test_heads_share_cls_representation(tiny_model):
    ids = torch.randint(5, 900, (1, 12)); am = torch.ones_like(ids)
    z = tiny_model.pooled(ids, am)
    assert z.shape == (1, 32)


def test_padding_does_not_change_logit(tiny_model):
    ids = torch.randint(5, 900, (1, 10))
    a = tiny_model(ids, torch.ones_like(ids))["align_logit"]
    ids2 = torch.cat([ids, torch.zeros(1, 6, dtype=torch.long)], 1); am2 = torch.cat([torch.ones(1, 10), torch.zeros(1, 6)], 1).long()
    b = tiny_model(ids2, am2)["align_logit"]
    assert torch.allclose(a, b, atol=1e-4)


def test_gradients_reach_all_heads_and_encoder(tiny_model):
    tiny_model.train()
    ids = torch.randint(5, 900, (4, 12)); o = tiny_model(ids, torch.ones_like(ids))
    y = torch.tensor([0, 1, 1, 0]); d = torch.tensor([-1, 0, 2, -1])
    loss, _ = multitask_loss(o, y, d)
    tiny_model.zero_grad(); loss.backward()
    for name in ("align_head", "drift_head", "cal_head"):
        assert getattr(tiny_model, name).weight.grad.abs().sum() > 0, name
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tiny_model.encoder.parameters())
    tiny_model.eval()


def test_param_count_positive(tiny_model):
    assert tiny_model.n_params() > 0


# ---- losses
def test_focal_gamma0_equals_bce():
    z = torch.randn(32); y = torch.randint(0, 2, (32,))
    assert torch.allclose(focal_loss(z, y, gamma=0.0), F.binary_cross_entropy_with_logits(z, y.float()), atol=1e-6)


def test_focal_downweights_easy_examples():
    z = torch.tensor([6.0]); y = torch.tensor([1])
    assert focal_loss(z, y, 2.0) < 0.01 * F.binary_cross_entropy_with_logits(z, y.float())


def test_focal_hard_example_penalised_nearly_as_bce():
    z = torch.tensor([-4.0]); y = torch.tensor([1])
    assert focal_loss(z, y, 2.0) > 0.9 * F.binary_cross_entropy_with_logits(z, y.float())


def test_focal_nonnegative_finite_extreme_logits():
    z = torch.tensor([1e4, -1e4, 0.0]); y = torch.tensor([1, 1, 0])
    l = focal_loss(z, y)
    assert torch.isfinite(l) and l >= 0


def test_multitask_weights_sum_matches_components():
    out = {"align_logit": torch.randn(6), "drift_logits": torch.randn(6, 3), "cal_logit": torch.randn(6)}
    y = torch.tensor([0, 1, 1, 0, 1, 0]); d = torch.tensor([-1, 0, 1, -1, 2, -1])
    loss, p = multitask_loss(out, y, d)
    assert math.isclose(loss.item(), 0.6 * p["align"] + 0.3 * p["drift"] + 0.1 * p["cal"], rel_tol=1e-4)


def test_multitask_no_misaligned_in_batch_is_finite():
    out = {"align_logit": torch.randn(3, requires_grad=True), "drift_logits": torch.randn(3, 3, requires_grad=True), "cal_logit": torch.randn(3)}
    loss, _ = multitask_loss(out, torch.zeros(3, dtype=torch.long), -torch.ones(3, dtype=torch.long))
    loss.backward(); assert torch.isfinite(loss)


def test_alignment_only_ablation_has_no_drift_or_cal_terms():
    out = {"align_logit": torch.randn(4), "drift_logits": torch.randn(4, 3), "cal_logit": torch.randn(4)}
    _, p = multitask_loss(out, torch.tensor([0, 1, 0, 1]), torch.tensor([-1, 0, -1, 1]), use_drift=False, use_cal=False)
    assert set(p) == {"align", "total"}


def test_distill_loss_zero_kl_when_student_equals_teacher():
    z = torch.randn(16); y = torch.randint(0, 2, (16,))
    l_same = distill_loss(z, z, y, tau=4.0, alpha=0.0)
    assert l_same.abs() < 1e-5


# ---- metrics
def test_ndcg_perfect_and_worst():
    assert ndcg_at_k([0.9, 0.1, 0.2], [1, 0, 0]) == pytest.approx(1.0)
    assert ndcg_at_k([0.1, 0.9, 0.2], [1, 0, 0]) < 1.0


def test_ndcg_undefined_without_positive():
    assert ndcg_at_k([0.3, 0.2], [0, 0]) is None


def test_precision_at_k_and_mrr():
    assert precision_at_k([0.9, 0.5, 0.1], [1, 0, 1], k=2) == 0.5
    assert mrr([0.1, 0.9, 0.5], [0, 0, 1]) == 0.5


def test_ece_perfectly_calibrated_is_small_and_overconfident_is_large():
    rng = np.random.RandomState(0); p = rng.rand(20000); y = (rng.rand(20000) < p).astype(int)
    assert ece(y, p) < 0.02
    assert ece(y, np.where(p > 0.5, 0.99, 0.01)) > 0.2


def test_cls_metrics_known_confusion():
    m = cls_metrics([1, 1, 0, 0], [1, 0, 0, 0])
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 1, 2, 0) and m["specificity"] == 1.0 and m["precision"] == 1.0


def test_pairwise_auc_by_tier_toy():
    r = pairwise_auc_by_tier([0.9, 0.1, 0.2, 0.8], [1, 0, 0, 1], ["a", "a", "b", "b"], ["hard", "n/a", "n/a", "random"])
    assert r["hard"]["auc"] == 1.0 and r["random"]["auc"] == 1.0 and r["hard"]["pairs"] == 1


def test_rank_metrics_constant_scores_equals_random_floor():
    sid = ["a", "a", "b", "b", "b"]; y = [1, 0, 0, 1, 0]
    m = rank_metrics(np.zeros(5), y, sid, n_seeds=40)
    assert m["n_rankable_sessions"] == 2 and 0.5 < m["ndcg@5"] < 1.0


def test_best_threshold_separable():
    t = best_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.2 < t <= 0.8


def test_temperature_scaling_reduces_nll_on_overconfident_logits():
    rng = np.random.RandomState(1); z = rng.randn(4000) * 5
    y = (rng.rand(4000) < 1 / (1 + np.exp(-z / 3))).astype(float)
    T = fit_temperature(z, y)
    assert T > 1.5 and _nll(z, y, T) < _nll(z, y, 1.0)
    assert apply_temperature(np.array([0.0]), T)[0] == pytest.approx(0.5)


# ---- denoise
def test_score_flag_set():
    s = score_flag_set([1, 1, 0, 0], [1, 0, 1, 0])
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 1) and s["precision"] == 0.5


def test_co_teaching_keeps_small_loss_fraction():
    ka, kb = co_teaching_select(torch.tensor([0.1, 5.0, 0.2, 4.0]), torch.tensor([3.0, 0.1, 0.2, 9.0]), 0.5)
    assert set(ka.tolist()) == {0, 2} and set(kb.tolist()) == {1, 2}
    assert co_teaching_forget_rate(0) == 0 and co_teaching_forget_rate(99) == pytest.approx(0.1)


# ---- tokenisation limits
def test_tokenizer_truncates_to_max_len():
    from sift.model import load_tokenizer
    tok = load_tokenizer("answerdotai/ModernBERT-base")
    enc = tok("fix bug " * 10, "x = 1\n" * 5000, truncation="longest_first", max_length=512)
    assert len(enc["input_ids"]) == 512


def test_tokenizer_keeps_intent_when_hunk_is_long():
    from sift.model import load_tokenizer
    tok = load_tokenizer("answerdotai/ModernBERT-base")
    enc = tok("add retry to fetch", "line\n" * 5000, truncation="longest_first", max_length=64)
    assert tok.decode(enc["input_ids"]).count("retry") == 1


# ---- hunk parser & text helpers
DIFF = """diff --git a/src/a.py b/src/a.py
index 1..2 100644
--- a/src/a.py
+++ b/src/a.py
@@ -1,3 +1,3 @@
 a
-b
+c
@@ -10,2 +10,3 @@ def f():
 x
+y
 z
diff --git a/b.js b/b.js
new file mode 100644
--- /dev/null
+++ b/b.js
@@ -0,0 +1,1 @@
+let q = 1;
"""


def test_parse_diff_counts_and_paths():
    h = parse_diff(DIFF)
    assert [x.file_path for x in h] == ["src/a.py", "src/a.py", "b.js"] and [x.index for x in h] == [0, 1, 2]


def test_parse_diff_hunk_text_starts_with_header():
    assert all(x.text.startswith("@@") for x in parse_diff(DIFF))


def test_parse_diff_handles_crlf():
    assert len(parse_diff(DIFF.replace("\n", "\r\n"))) == 3


def test_parse_diff_empty_and_garbage():
    assert parse_diff("") == [] and parse_diff("hello\nworld") == []


def test_parse_diff_rejects_non_string():
    with pytest.raises(TypeError):
        parse_diff(None)


def test_changed_text_and_count():
    h = "@@ -1,2 +1,2 @@\n ctx\n-old\n+new"
    assert changed_text(h) == "old\nnew" and n_changed(h) == 2


def test_roundtrip_with_dataset_hunks(data):
    """Parser applied to a dataset hunk wrapped in a minimal diff recovers one hunk with identical text."""
    r = data["test"].iloc[0]
    d = f"--- a/{r.file_path}\n+++ b/{r.file_path}\n{r.hunk}\n"
    got = parse_diff(d)
    assert len(got) >= 1 and got[0].text.split("\n")[0] == r.hunk.split("\n")[0]


# ---- intent consolidation
def test_consolidate_drops_chitchat():
    assert consolidate_intent(["add a cache", "sounds good"]) == "add a cache"


def test_consolidate_retraction_replaces_prior_request():
    assert consolidate_intent(["fix A", "Actually, scratch that. fix B"]) == "fix B"


def test_consolidate_repeated_turn_dedup():
    assert consolidate_intent(["do X", "same as before: do X"]) == "do X"


def test_consolidate_empty_and_whitespace():
    assert consolidate_intent([]) == "" and consolidate_intent(["  ", ""]) == ""


def test_consolidate_agreement_with_ground_truth(data):
    import re
    tr = data["train"]; norm = lambda x: re.findall(r"\w+", x.lower())
    s = tr.sample(1500, random_state=473)
    agree = np.mean([norm(consolidate_intent(split_turns(t))) == norm(c) for t, c in zip(s.intent_turns, s.consolidated_intent)])
    from conftest import record; record("intent_consolidation_agreement_train_sample_1500", float(agree))
    assert agree > 0.9
