"""Unified-diff parsing into hunks (file path + hunk text, matching the dataset's `hunk` format)."""
from __future__ import annotations
import re
from dataclasses import dataclass

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")


class DiffParseError(ValueError):
    """Raised for input that is not a parseable unified diff."""


@dataclass
class Hunk:
    file_path: str
    header: str
    text: str            # includes the "@@" header line, as in the dataset
    index: int = 0
    old_path: str = ""
    status: str = "modified"   # modified | added | deleted | renamed


@dataclass
class DiffNote:
    file_path: str
    kind: str            # "binary" | "rename" | "mode" | "empty"
    detail: str = ""


def _strip(p: str, prefix: str) -> str:
    p = p.split("\t")[0].strip()
    if len(p) >= 2 and p[0] == '"' and p[-1] == '"':
        p = p[1:-1]
    return p[len(prefix):] if p.startswith(prefix) else p


def parse_diff_ex(diff: str) -> tuple[list[Hunk], list[DiffNote]]:
    """Parse a diff into (hunks, notes). Notes record binary files, pure renames and mode changes,
    which carry no textual hunk. Raises DiffParseError on malformed input."""
    if not isinstance(diff, str):
        raise TypeError("diff must be a string")
    if "\x00" in diff:
        raise DiffParseError("NUL byte in diff text")
    lines = diff.replace("\r\n", "\n").split("\n")
    hunks: list[Hunk] = []
    notes: list[DiffNote] = []
    path = old = ""
    status = "modified"
    cur: list[str] | None = None
    hdr = ""
    saw_marker = False

    def flush():
        nonlocal cur
        if cur is not None:
            hunks.append(Hunk(path, hdr, "\n".join(cur).rstrip("\n"), len(hunks), old, status))
            cur = None

    for line in lines:
        if line.startswith("diff --git "):
            flush()
            saw_marker = True
            m = re.match(r'diff --git "?a/(.*?)"? "?b/(.*?)"?$', line)
            old, path = (m.group(1), m.group(2)) if m else ("", path)
            status = "modified"
        elif line.startswith("@@") and not _HUNK.match(line):
            raise DiffParseError(f"malformed hunk header: {line[:60]!r}")
        elif _HUNK.match(line):
            flush()
            saw_marker = True
            hdr, cur = line, [line]
        elif cur is not None:
            # inside a hunk: "--- "/"+++ " lines are content (removed/added lines) unless a new diff starts
            cur.append(line)
        elif line.startswith("--- "):
            saw_marker = True
            p = _strip(line[4:], "a/")
            if p != "/dev/null":
                old = old or p
                path = path or p
            else:
                status = "added"
        elif line.startswith("+++ "):
            p = _strip(line[4:], "b/")
            if p == "/dev/null":
                status = "deleted"
            else:
                path = p
        elif line.startswith("rename from "):
            old, status = line[12:].strip(), "renamed"
        elif line.startswith("rename to "):
            path, status = line[10:].strip(), "renamed"
            notes.append(DiffNote(path, "rename", f"{old} -> {path}"))
        elif line.startswith("new file mode"):
            status = "added"
        elif line.startswith("deleted file mode"):
            status = "deleted"
        elif line.startswith(("old mode ", "new mode ")):
            notes.append(DiffNote(path, "mode", line.strip()))
        elif line.startswith("Binary files ") or line.startswith("GIT binary patch"):
            notes.append(DiffNote(path, "binary", line.strip()))
    flush()
    if diff.strip() and not saw_marker:
        raise DiffParseError("input does not look like a unified diff (no diff/---/@@ markers)")
    if diff.strip() and not hunks and not notes:
        raise DiffParseError("diff contains no hunks")
    return hunks, notes


def parse_diff(diff: str, strict: bool = False) -> list[Hunk]:
    """Split a unified diff into hunks. Empty input -> []. Non-diff text -> [] unless strict=True,
    in which case DiffParseError is raised (a malformed '@@' header always raises)."""
    if isinstance(diff, str) and not diff.strip():
        return []
    try:
        return parse_diff_ex(diff)[0]
    except DiffParseError as e:
        if strict or "malformed hunk header" in str(e) or "NUL" in str(e):
            raise
        return []
