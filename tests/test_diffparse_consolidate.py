import json
import pytest
from sift.diffparse import parse_diff, parse_diff_ex, DiffParseError
from sift.consolidate import consolidate_intent, consolidate_detailed, split_turns
from sift.data import load_split

GIT = """diff --git a/a.py b/a.py
index 111..222 100644
--- a/a.py
+++ b/a.py
@@ -1,2 +1,2 @@
-x = 1
+x = 2
 y = 3
@@ -10,1 +10,2 @@ def f():
 z
+w
diff --git a/old.py b/new.py
similarity index 90%
rename from old.py
rename to new.py
index 1..2 100644
--- a/old.py
+++ b/new.py
@@ -1 +1 @@
--- not a header
+++ also content
diff --git a/img.png b/img.png
Binary files a/img.png and b/img.png differ
diff --git a/n.txt b/n.txt
new file mode 100644
--- /dev/null
+++ b/n.txt
@@ -0,0 +1 @@
+hi
"""


def test_multi_file_hunks_and_paths():
    h = parse_diff(GIT)
    assert [x.file_path for x in h] == ["a.py", "a.py", "new.py", "n.txt"]
    assert [x.index for x in h] == [0, 1, 2, 3]
    assert all(x.text.startswith("@@") for x in h)


def test_rename_status_and_notes():
    h, notes = parse_diff_ex(GIT)
    assert h[2].status == "renamed" and h[2].old_path == "old.py"
    assert any(n.kind == "rename" and "old.py -> new.py" in n.detail for n in notes)


def test_content_lines_that_look_like_headers_stay_in_hunk():
    h = parse_diff(GIT)
    assert "--- not a header" in h[2].text and "+++ also content" in h[2].text


def test_binary_note_without_hunk():
    h, notes = parse_diff_ex(GIT)
    assert any(n.kind == "binary" and n.file_path == "img.png" for n in notes)
    assert all(x.file_path != "img.png" for x in h)


def test_added_file_status():
    assert parse_diff(GIT)[3].status == "added"


def test_binary_only_diff_is_ok():
    h, n = parse_diff_ex("diff --git a/x.bin b/x.bin\nBinary files a/x.bin and b/x.bin differ\n")
    assert h == [] and n[0].kind == "binary"


def test_crlf_and_empty():
    assert len(parse_diff(GIT.replace("\n", "\r\n"))) == 4
    assert parse_diff("") == [] and parse_diff("  \n") == []


def test_malformed_inputs():
    with pytest.raises(DiffParseError):
        parse_diff("--- a/x\n+++ b/x\n@@ broken @@\n+x\n")
    with pytest.raises(DiffParseError):
        parse_diff("hello world", strict=True)
    with pytest.raises(DiffParseError):
        parse_diff("a\x00b")
    with pytest.raises(TypeError):
        parse_diff(None)
    assert parse_diff("hello world") == []


def test_deterministic_and_unicode():
    d = "--- a/é.py\n+++ b/é.py\n@@ -1 +1 @@\n-α\n+β\n"
    assert parse_diff(d) == parse_diff(d) and parse_diff(d)[0].file_path == "é.py"


def test_consolidate_rules():
    assert consolidate_intent("fix bug || thanks") == "fix bug"
    assert consolidate_intent(["old task", "Actually, scratch that. new task"]) == "new task"
    assert consolidate_intent("do X || same as before: do X") == "do X"
    assert consolidate_intent("do X || do X") == "do X"
    assert consolidate_intent("") == ""
    assert consolidate_detailed(["a", "ok cool"])[1] == {"chit_chat"}


def test_consolidate_keeps_clean_turns_and_is_idempotent():
    t = "add retry to fetch"
    assert consolidate_intent(t) == t
    out = consolidate_intent("a || nice, go ahead")
    assert consolidate_intent(out) == out
    assert split_turns("a || b ||  ") == ["a", "b"]
    with pytest.raises(TypeError):
        split_turns(5)


def test_consolidation_results_json_matches_data():
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "experiments/results/consolidation.json"
    assert p.exists()
    r = json.loads(p.read_text())
    assert r["overall"]["n"] == 18000
    assert r["overall"]["exact_match"] > 0.94


def test_consolidation_on_test_split_matches_ground_truth_mostly():
    df = load_split("test")
    ok = sum(consolidate_intent(r) == c for r, c in zip(df.intent_turns, df.consolidated_intent))
    assert ok / len(df) > 0.80
