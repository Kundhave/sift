"""Data pipeline: counts, split leakage, determinism, noise ground truth (CDAO 'Data': verify and validate)."""
import hashlib
import json
import pandas as pd
import pytest
from sift.data import load_split, load_jsonl, SPLITS, ROOT


def test_split_sizes(data):
    assert [len(data[s]) for s in SPLITS] == [12600, 2700, 2700]


def test_zero_repositories_shared_across_splits(data):
    r = {s: set(data[s].repo) for s in SPLITS}
    assert not (r["train"] & r["val"]) and not (r["train"] & r["test"]) and not (r["val"] & r["test"])


def test_expected_repositories(data):
    assert set(data["test"].repo) == {"httpx", "attrs", "axios"}


def test_zero_sessions_shared_across_splits(data):
    s = {k: set(v.session_id) for k, v in data.items()}
    assert not (s["train"] & s["val"] | s["train"] & s["test"] | s["val"] & s["test"])


def test_no_intent_hunk_pair_shared_across_splits(data):
    pairs = {k: set(zip(v.consolidated_intent, v.hunk)) for k, v in data.items()}
    assert not (pairs["train"] & pairs["test"]) and not (pairs["val"] & pairs["test"]) and not (pairs["train"] & pairs["val"])


def test_class_balance_per_split(data):
    for s, (a, m) in {"train": (9828, 2772), "val": (2106, 594), "test": (2106, 594)}.items():
        assert (data[s].label_clean == "aligned").sum() == a and (data[s].label_clean == "misaligned").sum() == m


def test_val_test_labels_clean(data):
    assert (data["val"].label == data["val"].label_clean).all() and (data["test"].label == data["test"].label_clean).all()


def test_train_label_flips_exactly_569(data):
    t = data["train"]
    assert int(t.label_noise.sum()) == 569 and int((t.label != t.label_clean).sum()) == 569


def test_drift_type_only_on_misaligned(data):
    for d in data.values():
        assert (d[d.label_clean == "aligned"].drift_type_clean == "none").all()
        assert set(d[d.label_clean == "misaligned"].drift_type_clean) <= {"collateral", "scope_expansion", "purpose_drift"}


def test_tiers_only_on_misaligned(data):
    for d in data.values():
        assert set(d[d.label_clean == "aligned"].mutation_difficulty) == {"n/a"}


def test_each_session_has_unique_sample_ids_and_positions(data):
    for d in data.values():
        assert d.sample_id.is_unique
        assert (d.groupby("session_id").position_in_diff.nunique() == d.groupby("session_id").size()).all()


def test_loading_is_deterministic():
    a, b = load_split("test"), load_split("test")
    assert a.equals(b)
    assert hashlib.sha1(pd.util.hash_pandas_object(a).values.tobytes()).hexdigest() == hashlib.sha1(pd.util.hash_pandas_object(b).values.tobytes()).hexdigest()


def test_jsonl_matches_csv(data):
    j = load_jsonl()
    assert len(j) == 18000
    ids = {r["sample_id"]: r for r in j}
    r = data["test"].iloc[5]
    assert ids[r.sample_id]["label"] == r.label and ids[r.sample_id]["hunk"] == r.hunk


def test_unknown_split_rejected():
    with pytest.raises(ValueError):
        load_split("holdout")


def test_noise_ground_truth_counts(data):
    a = pd.concat(data.values())
    assert a.is_duplicate_hunk.sum() == 937 and a.exceeds_max_len.sum() == 198
    assert a.non_semantic_reason.value_counts().to_dict() == {"none": 17286, "whitespace_only": 343, "lockfile": 202, "version_bump_only": 143, "generated_or_minified": 26}
    assert a.intent_noise_type.value_counts().drop("none").sum() == 2730


def test_hunks_non_empty_and_start_with_header(data):
    for d in data.values():
        assert (d.hunk.str.len() > 0).all() and d.hunk.str.startswith("@@").all()


def test_dataset_statistics_json_consistent(data):
    st = json.load(open(ROOT / "data/dataset_statistics.json"))
    for s in SPLITS:
        assert st[s]["hunks"] == len(data[s]) and st[s]["sessions"] == data[s].session_id.nunique()


# ---- documented discrepancy between the documented counts and the released data
@pytest.mark.xfail(strict=True, reason="D-DATA-1: documented counts are 11,900/5,076/1,024 hunks per language; released data differs")
def test_documented_language_counts_match_release(data):
    a = pd.concat(data.values())
    assert a.language.value_counts().to_dict() == {"Python": 11900, "JavaScript": 5076, "TypeScript": 1024}


@pytest.mark.xfail(strict=True, reason="D-DATA-2: test split has no TypeScript although val does; TS generalisation is untested")
def test_test_split_covers_typescript(data):
    assert "TypeScript" in set(data["test"].language)
