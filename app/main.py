"""Minimal local triage UI: paste intent turns + a unified diff -> hunks ranked by misalignment probability."""
from __future__ import annotations
import json, os, sqlite3, sys, time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sift.infer import SiftScorer

ROOT = Path(__file__).resolve().parents[1]
MAX_DIFF_CHARS = 2_000_000
app = FastAPI(title="Sift")
_state: dict = {}
DB_PATH = Path(os.environ.get("SIFT_DB", ROOT / "app" / "sift_feedback.db"))


def db() -> sqlite3.Connection:
    """Local SQLite for analysis history and reviewer feedback (never touches the user's repo)."""
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.executescript("""
    CREATE TABLE IF NOT EXISTS analyses(id INTEGER PRIMARY KEY, ts REAL, intent TEXT, n_hunks INT, result TEXT);
    CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY, ts REAL, analysis_id INT, hunk_index INT, action TEXT);""")
    return c


class Feedback(BaseModel):
    analysis_id: int
    hunk_index: int
    action: str  # "accept" (model is right) | "dismiss" (model is wrong)


class ScoreRequest(BaseModel):
    intent_turns: list[str]
    diff: str
    explain: bool = False


def scorer() -> SiftScorer:
    if "s" not in _state:
        _state["s"] = SiftScorer(os.environ.get("SIFT_ARTIFACT", str(ROOT / "artifacts" / "full")))
    return _state["s"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/score")
def score(req: ScoreRequest):
    if len(req.diff) > MAX_DIFF_CHARS:
        raise HTTPException(413, "diff too large")
    if not any(t.strip() for t in req.intent_turns):
        raise HTTPException(422, "at least one non-empty intent turn is required")
    s = scorer()
    out = s.score(req.intent_turns, req.diff)
    if req.explain and out["hunks"] and s.backend == "torch":
        from sift.explain import Explainer
        ex = _state.setdefault("ex", Explainer(s.model, s.tok, s.device, s.max_len))
        for h in out["hunks"]:
            r = ex.attribute(out["intent"], f"{h['file']}\n{h['text']}", n_steps=16)
            h["intent_words"] = [(w, sc) for w, sc, *_ in Explainer.to_words(r, out["intent"], f"{h['file']}\n{h['text']}", 0)]
            h["hunk_lines"] = Explainer.to_lines(r, f"{h['file']}\n{h['text']}")
    with db() as c:
        slim = {**out, "hunks": [{k: v for k, v in h.items() if k not in ("hunk_lines", "intent_words")} for h in out["hunks"]]}
        out["analysis_id"] = c.execute("INSERT INTO analyses(ts,intent,n_hunks,result) VALUES(?,?,?,?)",
                                       (time.time(), out["intent"], out["n_hunks"], json.dumps(slim))).lastrowid
    return out


@app.post("/api/feedback")
def feedback(fb: Feedback):
    if fb.action not in ("accept", "dismiss"):
        raise HTTPException(422, "action must be 'accept' or 'dismiss'")
    with db() as c:
        if not c.execute("SELECT 1 FROM analyses WHERE id=?", (fb.analysis_id,)).fetchone():
            raise HTTPException(404, "unknown analysis_id")
        c.execute("DELETE FROM feedback WHERE analysis_id=? AND hunk_index=?", (fb.analysis_id, fb.hunk_index))
        c.execute("INSERT INTO feedback(ts,analysis_id,hunk_index,action) VALUES(?,?,?,?)",
                  (time.time(), fb.analysis_id, fb.hunk_index, fb.action))
    return {"ok": True}


@app.get("/api/history")
def history(limit: int = 20):
    with db() as c:
        rows = c.execute("SELECT id,ts,intent,n_hunks FROM analyses ORDER BY id DESC LIMIT ?", (max(1, min(limit, 200)),)).fetchall()
        fb = c.execute("SELECT analysis_id, action, COUNT(*) n FROM feedback GROUP BY analysis_id, action").fetchall()
    counts: dict = {}
    for r in fb:
        counts.setdefault(r["analysis_id"], {})[r["action"]] = r["n"]
    return [{**dict(r), "feedback": counts.get(r["id"], {})} for r in rows]


@app.get("/api/analysis/{aid}")
def analysis(aid: int):
    with db() as c:
        r = c.execute("SELECT result FROM analyses WHERE id=?", (aid,)).fetchone()
    if not r:
        raise HTTPException(404, "unknown analysis_id")
    return {**json.loads(r["result"]), "analysis_id": aid}


@app.get("/api/model")
def model_info():
    try:
        s = scorer()
    except Exception as e:  # model-unavailable state for the UI
        raise HTTPException(503, f"model unavailable: {e}")
    return {"backend": s.backend, "encoder": s.meta["encoder"], "max_len": s.max_len,
            "temperature": s.T, "threshold": s.threshold, "artifact": s.dir.name}


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()
