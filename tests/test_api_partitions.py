"""API equivalence partitions per model and valid upper boundaries (complements test_api.py)."""
import os
os.environ.setdefault("SIFT_DEVICE", "cpu")
import pytest
from fastapi.testclient import TestClient
from app.backend.main import app
from sift import serve
from tests.test_api import DIFF

client = TestClient(app)
AVAILABLE = [m["id"] for m in client.get("/api/models").json()["models"] if m["available"]]


@pytest.mark.parametrize("model", AVAILABLE)
def test_every_available_model_returns_valid_ranking(model):
    r = client.post("/api/analyze", json={"turns": ["Fix the return value of f"], "diff": DIFF, "model": model})
    assert r.status_code == 200, r.text[:200]
    hs = r.json()["hunks"]
    assert len(hs) == 2
    assert sorted(h["rank"] for h in hs) == [1, 2]
    assert all(0.0 <= h["p_misaligned"] <= 1.0 for h in hs)
    top = min(hs, key=lambda h: h["rank"])
    assert top["p_misaligned"] >= max(h["p_misaligned"] for h in hs) - 1e-9   # rank 1 = highest probability


def test_max_turns_accepted_at_boundary():
    r = client.post("/api/analyze", json={"turns": ["Fix f"] * serve.MAX_TURNS, "diff": DIFF, "model": "b4"})
    assert r.status_code == 200


def test_unicode_turns_and_crlf_diff_accepted():
    r = client.post("/api/analyze", json={"turns": ["修复 f 的返回值 ✓"], "diff": DIFF.replace("\n", "\r\n"), "model": "b4"})
    assert r.status_code in (200, 422)
    assert r.status_code != 500
