"""Loaders, intent handling and hunk text preparation for Sift-IA v1."""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SPLITS = ("train", "val", "test")
TURN_SEP = " || "


def load_split(split: str, data_dir: Path | None = None) -> "pd.DataFrame":
    """Load one split CSV. Booleans/NaN are normalised; order is the file order (deterministic)."""
    import pandas as pd   # lazy: the lite (Render) deployment serves without pandas
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    d = Path(data_dir) if data_dir else DATA_DIR
    df = pd.read_csv(d / f"sift_ia_v1_{split}.csv", keep_default_na=False)
    for c in ("exceeds_max_len", "is_duplicate_hunk", "gold_candidate", "label_noise"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().isin(("true", "1"))
    return df


def load_all(data_dir: Path | None = None) -> dict:
    return {s: load_split(s, data_dir) for s in SPLITS}


def split_turns(intent_turns: str) -> list[str]:
    return [t.strip() for t in intent_turns.split(TURN_SEP.strip()) if t.strip()]


def raw_intent(row) -> str:
    """Unconsolidated multi-turn text (used by the 'no intent consolidation' ablation)."""
    return " ".join(split_turns(row["intent_turns"]))


from .consolidate import consolidate_intent  # noqa: E402,F401  (re-export; logic lives in consolidate.py)


def changed_text(hunk: str) -> str:
    return "\n".join(l[1:] for l in hunk.split("\n")[1:] if l[:1] in "+-")


def n_changed(hunk: str) -> int:
    return sum(1 for l in hunk.split("\n")[1:] if l[:1] in "+-")


def hunk_input(row) -> str:
    """Hunk text with the file path prepended (path is part of the diff header)."""
    return f"{row['file_path']}\n{row['hunk']}"


def load_jsonl(path: Path | None = None) -> list[dict]:
    p = Path(path) if path else DATA_DIR / "sift_ia_v1_all.jsonl"
    with open(p) as f:
        return [json.loads(l) for l in f]
