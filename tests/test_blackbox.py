"""Black-box functional tests: equivalence partitions, boundary values, error guessing, API contract."""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sift.diffparse import parse_diff
from sift.data import consolidate_intent, split_turns


def mk(n):
    return "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n" + "".join(f"@@ -{i*10+1},1 +{i*10+1},1 @@\n-a{i}\n+b{i}\n" for i in range(n))


# ---- parse_diff: equivalence partitions & boundaries
@pytest.mark.parametrize("n", [0, 1, 2, 14, 15, 200])
def test_hunk_count_boundaries(n):
    assert len(parse_diff(mk(n))) == n


@pytest.mark.parametrize("diff", ["", "\n", "   ", "not a diff"])
def test_non_diff_inputs_yield_no_hunks(diff):
    assert parse_diff(diff) == []


@pytest.mark.parametrize("diff", ["@@", "@@ garbage @@"])
def test_malformed_hunk_header_is_rejected_explicitly(diff):
    from sift.diffparse import DiffParseError
    with pytest.raises(DiffParseError):
        parse_diff(diff)


def test_binary_and_mode_change_only_diff_has_no_hunks():
    d = "diff --git a/x.png b/x.png\nBinary files a/x.png and b/x.png differ\ndiff --git a/s.sh b/s.sh\nold mode 100644\nnew mode 100755\n"
    assert parse_diff(d) == []


def test_deleted_file_path_taken_from_old_name():
    d = "diff --git a/old.py b/old.py\ndeleted file mode 100644\n--- a/old.py\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-x\n"
    h = parse_diff(d)
    assert len(h) == 1 and h[0].file_path == "old.py"


def test_unicode_paths_and_content():
    d = "diff --git a/データ.py b/データ.py\n--- a/データ.py\n+++ b/データ.py\n@@ -1,1 +1,1 @@\n-x = 'é'\n+x = '日本語'\n"
    h = parse_diff(d)
    assert h[0].file_path == "データ.py" and "日本語" in h[0].text


def test_removed_line_starting_with_dashes_inside_hunk_is_content():
    d = "diff --git a/a.sql b/a.sql\n--- a/a.sql\n+++ b/a.sql\n@@ -1,2 +1,1 @@\n--- a comment\n keep\n"
    h = parse_diff(d)
    assert len(h) == 1 and "--- a comment" in h[0].text


@pytest.mark.xfail(strict=True, reason="D1: plain `diff -u` multi-file output (no `diff --git` lines) merges the second file's header into the previous hunk")
def test_plain_unified_multifile_diff_splits_files():
    d = "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-a\n+b\n--- a/y.py\n+++ b/y.py\n@@ -1,1 +1,1 @@\n-c\n+d\n"
    h = parse_diff(d)
    assert [x.file_path for x in h] == ["x.py", "y.py"] and "--- a/y.py" not in h[0].text


# ---- consolidate_intent partitions
@pytest.mark.parametrize("turns,exp", [
    ([], ""), (["add x"], "add x"), (["add x", "ok cool"], "add x"), (["add x", "add x"], "add x"),
    (["a", "Actually, scratch that. b"], "b"), (["a", "Actually, scratch that."], ""),
    (["ok cool", "sounds good"], ""),
])
def test_consolidate_partitions(turns, exp):
    assert consolidate_intent(turns) == exp


def test_consolidate_unicode_and_long_turn():
    assert consolidate_intent(["修正 バグ"]) == "修正 バグ"
    assert len(consolidate_intent(["x" * 100_000])) == 100_000


@pytest.mark.xfail(strict=True, reason="D2: chit-chat filter is exact-match on 8 template phrases; paraphrased pleasantries are kept in the intent")
def test_paraphrased_chitchat_dropped():
    assert consolidate_intent(["add a cache", "great job, thanks a lot!"]) == "add a cache"


def test_split_turns_separator():
    assert split_turns("a || b ||  ") == ["a", "b"]


# ---- API contract
@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app)


def test_api_rejects_empty_intent(client):
    r = client.post("/api/score", json={"intent_turns": ["   "], "diff": mk(1)})
    assert r.status_code == 422


def test_api_rejects_missing_fields(client):
    assert client.post("/api/score", json={"diff": "x"}).status_code == 422


def test_api_rejects_oversized_diff(client):
    assert client.post("/api/score", json={"intent_turns": ["x"], "diff": "a" * 2_000_001}).status_code == 413


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Sift" in r.text


@pytest.mark.needs_artifact
def test_api_score_contract(client, scorer):
    import app.main as m
    m._state["s"] = scorer
    r = client.post("/api/score", json={"intent_turns": ["rename a to b"], "diff": mk(3)})
    assert r.status_code == 200
    j = r.json()
    assert j["n_hunks"] == 3 and [h["rank"] for h in j["hunks"]] == [1, 2, 3]
    ps = [h["p_misaligned"] for h in j["hunks"]]
    assert ps == sorted(ps, reverse=True) and all(0 <= p <= 1 for p in ps)
    for h in j["hunks"]:
        assert abs(sum(h["drift_probs"].values()) - 1) < 1e-4 and h["verdict"] in ("aligned", "misaligned")
        assert (h["drift_type"] is None) == (h["verdict"] == "aligned")


@pytest.mark.needs_artifact
def test_api_no_hunks_returns_empty_list(client, scorer):
    import app.main as m
    m._state["s"] = scorer
    r = client.post("/api/score", json={"intent_turns": ["x"], "diff": "nothing here"})
    assert r.status_code == 200 and r.json()["hunks"] == []
