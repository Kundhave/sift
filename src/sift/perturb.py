"""Input transformations used by metamorphic and adversarial tests."""
from __future__ import annotations
import random
import re

_KW = set("""def class return import from if else elif for while try except finally with as lambda pass break continue and or not in is None True False
const let var function new this async await export default typeof null true false undefined self cls print len range str int list dict set tuple""".split())
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")


def comment_prefix(path: str) -> str:
    return "#" if path.endswith((".py", ".toml", ".yml", ".yaml", ".cfg", ".sh")) else "//"


def rename_identifier(intent: str, hunk: str, new="renamed_var_7", seed=0):
    """Consistently rename one identifier occurring in the changed lines (in hunk, and in intent if present).
    Returns (intent2, hunk2, old) or None if no candidate."""
    body = "\n".join(l for l in hunk.split("\n")[1:] if l[:1] in "+-")
    cands = sorted({m for m in _ID.findall(body) if m not in _KW})
    if not cands:
        return None
    old = random.Random(seed).choice(cands)
    pat = re.compile(rf"\b{re.escape(old)}\b")
    return pat.sub(new, intent), pat.sub(new, hunk), old


def add_context_comment(hunk: str, path: str, text="note: unchanged helper kept for readability"):
    """Append a *context* (unchanged) comment line: the change itself is untouched."""
    return hunk.rstrip("\n") + f"\n {comment_prefix(path)} {text}"


def add_trailing_whitespace_context(hunk: str):
    return "\n".join(l + "  " if l.startswith(" ") else l for l in hunk.split("\n"))


def inject_comment_added(hunk: str, path: str, text: str):
    """Append an ADDED comment line (an attacker-controlled line inside the diff)."""
    return hunk.rstrip("\n") + f"\n+{comment_prefix(path)} {text}"


def long_hunk(hunk: str, n_lines=50_000):
    return hunk + "\n" + "\n".join(f"+value_{i} = {i}" for i in range(n_lines))
