"""Sift FastAPI service. Run:  PYTHONPATH=src .venv/bin/uvicorn app.backend.main:app --port 8000
Docs: http://localhost:8000/docs   (all inference is local; models load lazily, once)."""
from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sift import serve  # noqa: E402

app = FastAPI(title="Sift API", version="1.0",
              description="Local verification layer: rank diff hunks by misalignment with the stated intent.")


class AnalyzeRequest(BaseModel):
    turns: Optional[list[str]] = Field(None, description="Intent turns of the session (chit-chat/retractions allowed)")
    diff: Optional[str] = Field(None, description="Unified diff (git format, multi-file)")
    sample_id: Optional[str] = Field(None, description="Curated test-set sample id; replaces turns+diff")
    model: str = Field("full", description="Model id from GET /api/models")
    explain: bool = Field(False, description="Integrated-Gradients attributions for the top hunks (PyTorch models only, slow on CPU)")


@app.exception_handler(serve.ApiError)
async def api_error(_: Request, e: serve.ApiError):
    return JSONResponse({"error": e.message, "status": e.status}, status_code=e.status)


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if req.sample_id is not None:
        if req.turns is not None or req.diff is not None:
            raise serve.ApiError(422, "give either sample_id or turns+diff, not both")
        return serve.analyze_sample(req.sample_id, req.model, req.explain)
    if req.turns is None or req.diff is None:
        raise serve.ApiError(422, "provide 'turns' and 'diff', or a 'sample_id'")
    return serve.analyze(req.turns, req.diff, req.model, req.explain)


@app.get("/api/samples")
def samples():
    return serve.list_samples()


@app.get("/api/samples/{sample_id}")
def sample(sample_id: str):
    return serve.get_sample(sample_id)


@app.get("/api/models")
def models():
    ms = serve.REGISTRY.list()
    ok = [m["id"] for m in ms if m["available"]]
    default = "full" if "full" in ok or not ok else ok[0]   # lite deployment: first servable model
    return {"default": default, "models": ms}


@app.get("/api/results")
def results_index():
    return serve.list_results()


@app.get("/api/results/{path:path}")
def result_file(path: str):
    f = serve.find_result(path)
    return FileResponse(f, media_type=serve.RESULT_EXT[f.suffix.lower()])


@app.get("/api/dataset/stats")
def dataset_stats():
    return serve.dataset_stats()


@app.get("/api/tests")
def tests():
    return serve.tests_summary()


@app.get("/api/health")
def health():
    return serve.health()


# ---- optional: serve the built SPA (single-container deployment) -----------------------------
def _static_dir() -> Optional[Path]:
    d = os.environ.get("SIFT_STATIC_DIR")
    p = Path(d) if d else Path(__file__).resolve().parents[1] / "frontend" / "dist"
    return p if (p / "index.html").is_file() else None


_STATIC = _static_dir()
if _STATIC is not None:
    _root = _STATIC.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path == "api" or path.startswith("api/"):
            return JSONResponse({"error": f"unknown API route /{path}", "status": 404}, status_code=404)
        f = (_root / path).resolve()
        if path and _root in f.parents and f.is_file():
            return FileResponse(f)
        if Path(path).suffix:   # missing asset-like path: a real 404, not the SPA shell
            return JSONResponse({"error": f"not found: /{path}", "status": 404}, status_code=404)
        return FileResponse(_root / "index.html")   # client-side route (React Router)
