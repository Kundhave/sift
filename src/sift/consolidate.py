"""Rule-based intent consolidation: drop chit-chat, retracted requests and repeated turns.

No labels are used. The patterns come from the noise templates visible in Sift-IA v1, so they are
tuned to that distribution (stated as a limitation in the Model Card).
"""
from __future__ import annotations
import re

TURN_SEP = "||"
_CHATTER = {"nice, go ahead", "can you run it again?", "sorry, i was away for a minute", "great, keep going",
            "thanks, that looks good so far", "ok cool", "sounds good", "hmm, let me think about it for a second",
            "thanks", "thank you", "ok", "okay", "cool", "great", "lgtm", "got it"}
_RETRACT = re.compile(r"^(actually,?\s*)?(scratch that|ignore that|never ?mind|forget that|disregard that)[.!:,]?\s*", re.I)
_REPEAT = re.compile(r"^(same as before|as before|again|same thing)\s*:\s*", re.I)


def split_turns(intent_turns: str | list[str]) -> list[str]:
    """Split the dataset's ' || '-separated string (or accept a list) into stripped non-empty turns."""
    if isinstance(intent_turns, (list, tuple)):
        return [str(t).strip() for t in intent_turns if str(t).strip()]
    if not isinstance(intent_turns, str):
        raise TypeError("intent_turns must be str or list[str]")
    return [t.strip() for t in intent_turns.split(TURN_SEP) if t.strip()]


def _join(parts: list[str]) -> str:
    """Join turns the way the dataset does: a space, preceded by '.' when the turn has no end punctuation."""
    out = ""
    for p in parts:
        out = p if not out else out + ("" if out[-1] in ".!?:;" else ".") + " " + p
    return out


def consolidate_detailed(turns: list[str]) -> tuple[str, set[str]]:
    """Return (consolidated intent, set of noise kinds removed: chit_chat / retraction / repeated_turn)."""
    kept: list[str] = []
    found: set[str] = set()
    for t in turns:
        t = t.strip()
        if t.lower().strip(" .!") in _CHATTER:
            found.add("chit_chat")
            continue
        m = _RETRACT.match(t)
        if m:
            found.add("retraction")
            kept = []
            t = t[m.end():].strip()
            if not t:
                continue
        m = _REPEAT.match(t)
        if m:
            found.add("repeated_turn")
            t = t[m.end():].strip()
        if t and t in kept:
            found.add("repeated_turn")
            continue
        if t:
            kept.append(t)
    return _join(kept), found


def consolidate_intent(turns: list[str] | str) -> str:
    return consolidate_detailed(split_turns(turns))[0]
