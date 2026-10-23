"""API tests (FastAPI TestClient, CPU). Real-model tests are marked needs_artifact and skipped if artifacts/full is absent."""
import os
os.environ.setdefault("SIFT_DEVICE", "cpu")
import pytest
from fastapi.testclient import TestClient
from app.backend.main import app
from sift import serve

client = TestClient(app)
HAS_FULL = serve._artifact_status("full")[0]
needs_full = pytest.mark.skipif(not HAS_FULL, reason="artifacts/full missing")
DIFF = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,3 @@\n def f():\n-    return 1\n+    return 2\n+# note\n"
        "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n")


def test_health_and_models():
    r = client.get("/api/health"); assert r.status_code == 200 and r.json()["status"] == "ok"
    m = client.get("/api/models").json()
    ids = {x["id"]: x for x in m["models"]}
    assert {"full", "student", "student_int8", "b4", "b5"} <= set(ids)
    for x in ids.values():
        assert isinstance(x["available"], bool) and (x["available"] or x["reason"])
    assert m["default"] == "full"


def test_dataset_stats_and_results():
    s = client.get("/api/dataset/stats"); assert s.status_code == 200 and "train" in s.json()
    idx = client.get("/api/results").json(); assert isinstance(idx, list) and idx
    assert client.get("/api/results/baselines.json").status_code == 200
    assert client.get("/api/results/nope.json").status_code == 404


@pytest.mark.parametrize("p", ["../CLAUDE.md", "..%2fCLAUDE.md", "../../data/sift_ia_v1_test.csv", "/etc/passwd", "x.py", "eval_full.pt"])
def test_results_path_traversal_and_type_blocked(p):
    assert client.get(f"/api/results/{p}").status_code in (404, 422)


def test_tests_endpoint_shape():
    assert "available" in client.get("/api/tests").json()


def test_samples_list_and_detail():
    lst = client.get("/api/samples").json()
    assert len(lst) >= 10 and {"id", "n_hunks", "turns"} <= set(lst[0])
    assert any(s["n_misaligned"] == 0 for s in lst) and any(s["n_misaligned"] > 0 for s in lst)
    d = client.get(f"/api/samples/{lst[0]['id']}").json()
    assert d["diff"].startswith("diff --git") and d["hunks"][0]["label"] in ("aligned", "misaligned")
    assert client.get("/api/samples/NOPE").status_code == 404


def test_sample_diff_round_trips_to_dataset_hunks():
    from sift.diffparse import parse_diff
    for s in serve._load_samples().values():
        hs = parse_diff(s["diff"])
        assert [h.text.strip() for h in hs] == [h["text"].strip() for h in s["hunks"]]


@pytest.mark.parametrize("body,status", [
    ({}, 422),                                                  # nothing given
    ({"turns": ["fix f"]}, 422),                                # no diff
    ({"diff": DIFF}, 422),                                      # no turns
    ({"turns": [], "diff": DIFF}, 422),                         # empty list
    ({"turns": ["   "], "diff": DIFF}, 422),                    # whitespace only
    ({"turns": ["ok cool", "thanks"], "diff": DIFF}, 422),      # all chit-chat -> empty intent
    ({"turns": ["fix"], "diff": ""}, 422),                      # empty diff
    ({"turns": ["fix"], "diff": "hello world"}, 422),           # not a diff
    ({"turns": ["fix"], "diff": "@@ -x +y @@\n-a\n+b\n"}, 422),  # malformed header
    ({"turns": "fix", "diff": DIFF}, 422),                      # wrong type
    ({"turns": ["fix"], "diff": DIFF, "sample_id": "S1"}, 422), # both modes
    ({"sample_id": "NOPE"}, 404),
    ({"turns": ["fix"], "diff": DIFF, "model": "nope"}, 404),
    ({"turns": ["x" * (serve.MAX_TURN_CHARS + 1)], "diff": DIFF}, 422),       # boundary: turn too long
    ({"turns": ["fix"] * (serve.MAX_TURNS + 1), "diff": DIFF}, 422),          # boundary: too many turns
    ({"turns": ["fix"], "diff": DIFF + "x" * serve.MAX_DIFF_CHARS}, 413),     # boundary: diff too large
    ({"turns": ["fix"], "diff": DIFF, "model": "student_int8", "explain": True}, (422, 503)),
])
def test_analyze_rejects_bad_input(body, status):
    r = client.post("/api/analyze", json=body)
    assert r.status_code in (status if isinstance(status, tuple) else (status,)), r.text
    assert "error" in r.json() or "detail" in r.json()


def test_binary_only_diff_has_clear_error():
    r = client.post("/api/analyze", json={"turns": ["fix"], "diff": "diff --git a/x.png b/x.png\nBinary files a/x.png and b/x.png differ\n"})
    assert r.status_code == 422 and "binary" in r.json()["error"]


def test_dropped_turns_logic():
    t = ["add retries to fetch", "ok cool", "scratch that. rename helper"]
    intent = serve.consolidate_detailed(t)[0]
    d = {x["turn"]: x["reason"] for x in serve.dropped_turns(t, intent)}
    assert d.get("ok cool") == "chit_chat" and "add retries to fetch" in d


def test_baseline_model_analyze():
    r = client.post("/api/analyze", json={"turns": ["change return value of f"], "diff": DIFF, "model": "b5"})
    assert r.status_code == 200, r.text
    j = r.json(); assert j["n_hunks"] == 2 and sorted(j["ranking"]) == [0, 1]
    assert all(0 <= h["p_misaligned"] <= 1 and h["cluster_id"] is None and h["attributions"] is None for h in j["hunks"])


@needs_full
def test_analyze_full_custom_diff():
    r = client.post("/api/analyze", json={"turns": ["ok cool", "change the return value of f to 2"], "diff": DIFF})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model"] == "full" and j["n_hunks"] == 2 and j["intent"].startswith("change the return value")
    assert [d["reason"] for d in j["dropped_turns"]] == ["chit_chat"]
    ps = [h["p_misaligned"] for h in j["hunks"]]
    assert ps == sorted(ps, reverse=True) and [h["rank"] for h in j["hunks"]] == [1, 2]
    for h in j["hunks"]:
        assert 0.5 <= h["confidence"] <= 1 and h["verdict"] in ("aligned", "misaligned")
        assert (h["drift_type"] is None) == (h["verdict"] == "aligned")


@needs_full
def test_analyze_full_deterministic():
    b = {"turns": ["change the return value of f to 2"], "diff": DIFF}
    a1, a2 = client.post("/api/analyze", json=b).json(), client.post("/api/analyze", json=b).json()
    assert [h["p_misaligned"] for h in a1["hunks"]] == [h["p_misaligned"] for h in a2["hunks"]]


@needs_full
def test_analyze_sample_returns_ground_truth():
    sid = next(s["id"] for s in client.get("/api/samples").json() if s["n_misaligned"] > 0)
    r = client.post("/api/analyze", json={"sample_id": sid})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["sample_id"] == sid and all(h["ground_truth"] is not None for h in j["hunks"])
    assert all("correct" in h for h in j["hunks"])


@needs_full
def test_full_model_returns_cluster_and_similar():
    r = client.post("/api/analyze", json={"turns": ["change return value of f"], "diff": DIFF, "model": "full"})
    assert r.status_code == 200, r.text
    for h in r.json()["hunks"]:
        if (serve.ARTIFACTS / "kmeans" / "kmeans.npz").exists():
            assert isinstance(h["cluster_id"], int) and 0 <= h["cluster_id"] < 64
            assert 0 <= h["cluster_prior"] <= 1 and len(h["similar"]) <= 3


def test_spa_fallback_and_api_404(tmp_path, monkeypatch):
    """With a static dir, client routes get index.html; unknown /api paths stay JSON 404; /api/health still works."""
    import importlib
    (tmp_path / "index.html").write_text("<html>spa</html>")
    (tmp_path / "x.js").write_text("1")
    monkeypatch.setenv("SIFT_STATIC_DIR", str(tmp_path))
    import app.backend.main as m
    m2 = importlib.reload(m)
    try:
        c = TestClient(m2.app)
        assert "spa" in c.get("/dev/results").text and "spa" in c.get("/app").text
        assert c.get("/x.js").text == "1"
        assert "spa" in c.get("/../../etc/passwd").text
        assert c.get("/api/nope").status_code == 404 and "error" in c.get("/api/nope").json()
        assert c.get("/api/health").json()["status"] == "ok"
    finally:
        monkeypatch.delenv("SIFT_STATIC_DIR"); importlib.reload(m)


def test_sift_models_env_restricts_registry(monkeypatch):
    monkeypatch.setenv("SIFT_MODELS", "student_int8,b4")
    ids = [m["id"] for m in client.get("/api/models").json()["models"]]
    assert ids == ["student_int8", "b4"]
    assert client.post("/api/analyze", json={"turns": ["x"], "diff": DIFF, "model": "full"}).status_code == 404
