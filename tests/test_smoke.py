"""Smoke / readiness: the system is alive and its parts import and run (DoD CDAO: readiness)."""
import pytest
from conftest import ART


def test_package_imports():
    import sift, sift.data, sift.model, sift.losses, sift.rank, sift.calibrate, sift.denoise, sift.diffparse, sift.infer
    assert sift.SEED == 473


def test_dataset_loads(data):
    assert {k: len(v) for k, v in data.items()} == {"train": 12600, "val": 2700, "test": 2700}


def test_tiny_model_forward(tiny_model):
    import torch
    ids = torch.randint(5, 900, (2, 16)); o = tiny_model(ids, torch.ones_like(ids))
    assert set(o) == {"align_logit", "drift_logits", "cal_logit"}


def test_tokenizer_available():
    from sift.model import load_tokenizer
    assert load_tokenizer("answerdotai/ModernBERT-base")("hello")["input_ids"]


@pytest.mark.needs_artifact
def test_artifact_scores_a_diff(scorer):
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-x = 1\n+x = 2\n y = 3\n"
    r = scorer.score(["change x to 2"], diff)
    assert r["n_hunks"] == 1 and 0 <= r["hunks"][0]["p_misaligned"] <= 1


def test_api_health():
    from fastapi.testclient import TestClient
    import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from app.main import app
    assert TestClient(app).get("/health").json() == {"status": "ok"}
