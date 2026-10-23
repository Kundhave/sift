"""Serving logic for the Sift API: model registry (lazy, load-once), analysis, curated samples, result files.

No web framework imports here, so it is unit-testable and reusable from the CLI.
Missing artifacts are listed as unavailable with a reason; nothing is ever faked.
"""
from __future__ import annotations
import json, math, os, threading, time
from pathlib import Path
import numpy as np
from .consolidate import consolidate_detailed, split_turns
from .data import ROOT, load_split
from .diffparse import DiffParseError, parse_diff, parse_diff_ex

ARTIFACTS = ROOT / "artifacts"
RESULT_DIRS = (ROOT / "results", ROOT / "experiments" / "results")
RESULT_EXT = {".json": "application/json", ".csv": "text/csv", ".png": "image/png", ".md": "text/markdown",
              ".txt": "text/plain", ".svg": "image/svg+xml"}

MAX_TURNS, MAX_TURN_CHARS, MAX_DIFF_CHARS, MAX_HUNKS = 50, 5_000, 1_000_000, int(os.environ.get("SIFT_MAX_HUNKS", 200))
EXPLAIN_TOP = 3


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


# --------------------------------------------------------------------------- baselines
class BaselineScorer:
    """B4 (TF-IDF dissimilarity) / B5 (logistic regression, 4 features), fitted lazily on the train split."""
    backend = "sklearn"

    def __init__(self, kind: str):
        self.kind, self.T = kind, 1.0
        pre = ARTIFACTS / "baselines_lite.joblib"   # pre-fitted B4/B5 (deploy/lite/build_lite_data.py); skips CSV + pandas
        if not (ROOT / "data" / "sift_ia_v1_train.csv").exists() and pre.exists():
            import joblib
            d = joblib.load(pre)
            self.lf, self.lr, self.threshold = d["lf"], d["lr"], float(d["threshold_" + kind])
            self._finish()
            return
        from sklearn.linear_model import LogisticRegression
        from .baselines import LexicalFeatures
        from .rank import best_threshold
        tr, va = load_split("train"), load_split("val")
        self.lf = LexicalFeatures(tr)
        if kind == "b5":
            ytr = (tr.label == "misaligned").values.astype(int)
            self.lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(self.lf(tr)[0], ytr)
        yva = (va.label_clean == "misaligned").values.astype(int)
        self.threshold = float(best_threshold(yva, self._p(va), np.linspace(0.05, 0.95, 91)))
        self._finish()

    def _finish(self):
        self.meta = {"encoder": None, "max_len": None, "temperature": 1.0, "threshold": self.threshold, "backend": "sklearn"}
        self.dir = Path(self.kind)

    def _p(self, df):
        X, sim, _ = self.lf(df)
        return self.lr.predict_proba(X)[:, 1] if self.kind == "b5" else 1 - sim

    def score(self, turns, diff: str) -> dict:
        from types import SimpleNamespace
        intent = consolidate_detailed(split_turns(turns))[0]
        hunks = parse_diff(diff)
        if not hunks:
            return {"intent": intent, "hunks": [], "n_hunks": 0}
        t0 = time.perf_counter()
        df = SimpleNamespace(consolidated_intent=[intent] * len(hunks), hunk=[h.text for h in hunks],
                             file_path=[h.file_path for h in hunks], n_hunks_in_diff=[len(hunks)] * len(hunks))
        p = self._p(df)
        rows = [{"index": h.index, "file": h.file_path, "header": h.header, "p_misaligned": float(p[i]),
                 "verdict": "misaligned" if p[i] >= self.threshold else "aligned", "drift_type": None,
                 "drift_probs": None, "confidence_head": None, "text": h.text} for i, h in enumerate(hunks)]
        rows.sort(key=lambda r: -r["p_misaligned"])
        for k, r in enumerate(rows, 1):
            r["rank"] = k
        return {"intent": intent, "n_hunks": len(rows), "latency_s": time.perf_counter() - t0, "hunks": rows}


# --------------------------------------------------------------------------- registry
# id -> (display name, kind, description, artifact dir name or None)
_KNOWN = {
    "full": ("Sift ModernBERT-base (main)", "neural", "Cross-encoder, 149M parameters, three heads", "full"),
    "student": ("MiniLM student (distilled)", "neural", "22M-parameter distilled student, PyTorch fp32", "student"),
    "student_onnx_fp32": ("MiniLM student (ONNX fp32)", "neural", "Distilled student exported to ONNX", "student_onnx_fp32"),
    "student_int8": ("MiniLM student (INT8 ONNX)", "neural", "Deployed target: INT8-quantised ONNX student", "student_onnx_int8"),
    "b4": ("B4 TF-IDF dissimilarity", "baseline", "1 - cosine(intent, changed text); ranking score, not a probability", None),
    "b5": ("B5 Logistic regression", "baseline", "4 lexical features, class-balanced logistic regression", None),
}


def _artifact_status(dirname: str) -> tuple[bool, str]:
    d = ARTIFACTS / dirname
    if not d.is_dir():
        return False, f"artifact directory artifacts/{dirname} not found"
    if not (d / "meta.json").exists():
        return False, f"artifacts/{dirname}/meta.json missing (artifact not finished)"
    meta = json.load(open(d / "meta.json"))
    w = meta.get("onnx_file", "model.onnx") if meta.get("backend") == "onnx" else "model.pt"
    if not (d / w).exists():
        return False, f"artifacts/{dirname}/{w} missing"
    return True, ""


class Registry:
    def __init__(self):
        self._scorers: dict = {}
        self._lock = threading.Lock()
        self._run_locks: dict[str, threading.Lock] = {}
        self.explainers: dict = {}

    def entries(self) -> dict:
        ent = {k: v for k, v in _KNOWN.items()}
        allow = {m.strip() for m in os.environ.get("SIFT_MODELS", "").split(",") if m.strip()}
        if allow:   # deployment restriction, e.g. SIFT_MODELS=student_int8,b4,b5 on low-RAM hosts
            show = {m.strip() for m in os.environ.get("SIFT_LISTED_UNAVAILABLE", "").split(",") if m.strip()}
            return {k: v for k, v in ent.items() if k in allow or k in show}
        if ARTIFACTS.is_dir():   # pick up any other finished artifact automatically
            known_dirs = {v[3] for v in _KNOWN.values()}
            for d in sorted(ARTIFACTS.iterdir()):
                if d.is_dir() and d.name not in known_dirs and d.name != "kmeans" and (d / "meta.json").exists():
                    ent[d.name] = (d.name, "neural", f"artifacts/{d.name}", d.name)
        return ent

    def info(self, mid: str) -> dict:
        name, kind, desc, dirname = self.entries()[mid]
        allow = {m.strip() for m in os.environ.get("SIFT_MODELS", "").split(",") if m.strip()}
        if allow and mid not in allow:   # listed only so the UI can say why it is missing
            return {"id": mid, "name": name, "kind": kind, "description": desc, "available": False, "loaded": False,
                    "reason": "not served in this lite deployment (512 MB RAM); run `make demo` locally to use it",
                    "supports_explain": False}
        if dirname:
            ok, why = _artifact_status(dirname)
        else:
            ok = (ROOT / "data" / "sift_ia_v1_train.csv").exists() or (ARTIFACTS / "baselines_lite.joblib").exists()
            why = "training CSV not found"
        out = {"id": mid, "name": name, "kind": kind, "description": desc, "available": ok,
               "reason": None if ok else why, "loaded": mid in self._scorers}
        if ok and dirname:
            d = ARTIFACTS / dirname
            meta = json.load(open(d / "meta.json"))
            f = d / (meta.get("onnx_file", "model.onnx") if meta.get("backend") == "onnx" else "model.pt")
            out.update(encoder=meta.get("encoder"), backend=meta.get("backend"), max_len=meta.get("max_len"),
                       temperature=meta.get("temperature"), threshold=meta.get("threshold"),
                       source_run=meta.get("source_run"), size_mb=round(f.stat().st_size / 2**20, 1),
                       supports_explain=meta.get("backend", "torch") == "torch")
        elif ok:
            out.update(backend="sklearn", supports_explain=False, size_mb=None)
        if mid == "full":
            out["source_run"] = out.get("source_run") or "full"
            out["source_run_dir"] = "experiments/runs/full"
        return out

    def list(self) -> list[dict]:
        return [self.info(m) for m in self.entries()]

    def get(self, mid: str):
        ents = self.entries()
        if mid not in ents:
            raise ApiError(404, f"unknown model {mid!r}; known: {sorted(ents)}")
        info = self.info(mid)
        if not info["available"]:
            raise ApiError(503, f"model {mid!r} is unavailable: {info['reason']}")
        with self._lock:
            lk = self._run_locks.setdefault(mid, threading.Lock())
            if mid not in self._scorers:
                try:
                    dirname = ents[mid][3]
                    if dirname:
                        from .infer import SiftScorer
                        self._scorers[mid] = SiftScorer(ARTIFACTS / dirname)
                    else:
                        self._scorers[mid] = BaselineScorer(mid)
                except Exception as e:
                    raise ApiError(503, f"model {mid!r} failed to load: {type(e).__name__}: {e}")
        return self._scorers[mid], lk


REGISTRY = Registry()


# --------------------------------------------------------------------------- input validation
def validate_turns(turns) -> list[str]:
    if not isinstance(turns, list) or not all(isinstance(t, str) for t in turns):
        raise ApiError(422, "'turns' must be a list of strings")
    turns = [t.strip() for t in turns if t.strip()]
    if not turns:
        raise ApiError(422, "at least one non-empty intent turn is required")
    if len(turns) > MAX_TURNS:
        raise ApiError(422, f"too many turns ({len(turns)} > {MAX_TURNS})")
    if any(len(t) > MAX_TURN_CHARS for t in turns):
        raise ApiError(422, f"an intent turn exceeds {MAX_TURN_CHARS} characters")
    return turns


def validate_diff(diff) -> tuple[list, list]:
    if not isinstance(diff, str) or not diff.strip():
        raise ApiError(422, "'diff' must be a non-empty unified diff")
    if len(diff) > MAX_DIFF_CHARS:
        raise ApiError(413, f"diff too large ({len(diff)} > {MAX_DIFF_CHARS} characters)")
    try:
        hunks, notes = parse_diff_ex(diff)
    except DiffParseError as e:
        raise ApiError(422, f"could not parse diff: {e}")
    if not hunks:
        det = "; ".join(f"{n.file_path}: {n.kind}" for n in notes) or "no hunks found"
        raise ApiError(422, f"diff has no textual hunks to score ({det})")
    if len(hunks) > MAX_HUNKS:
        raise ApiError(413, f"too many hunks ({len(hunks)} > {MAX_HUNKS})")
    return hunks, notes


def dropped_turns(turns: list[str], intent: str) -> list[dict]:
    """Which input turns did consolidation not keep, and why (per-turn view of consolidate_detailed)."""
    out = []
    for t in turns:
        kinds = consolidate_detailed([t])[1]
        kept = consolidate_detailed([t])[0]
        if kept and kept.rstrip(".") in intent:
            continue
        if "chit_chat" in kinds:
            reason = "chit_chat"
        elif "retraction" in kinds:
            reason = "retraction"
        elif "repeated_turn" in kinds or kept:
            reason = "repeated_turn" if kept else "retracted_by_later_turn"
        else:
            reason = "retracted_by_later_turn"
        out.append({"turn": t, "reason": reason})
    return out


# --------------------------------------------------------------------------- analysis
def _explain(scorer, mid: str, intent: str, rows: list[dict]) -> None:
    from .explain import Explainer
    if mid not in REGISTRY.explainers:
        REGISTRY.explainers[mid] = Explainer(scorer.model, scorer.tok, scorer.device, scorer.max_len)
    ex = REGISTRY.explainers[mid]
    for r in rows[:EXPLAIN_TOP]:
        h = f"{r['file']}\n{r['text']}"
        res = ex.attribute(intent, h, n_steps=16)
        r["attributions"] = {"method": "integrated_gradients", "n_steps": 16,
                             "intent_words": [[w, sc] for w, sc, *_ in Explainer.to_words(res, intent, h, 0)],
                             "hunk_lines": [[l, sc] for l, sc in Explainer.to_lines(res, h)]}


_kmeans = {"idx": None, "tried": False, "train": None}


def _kmeans_index():
    """k-means index over the full model's pooled [CLS] embeddings (artifacts/kmeans); None if absent."""
    if not _kmeans["tried"]:
        _kmeans["tried"] = True
        try:
            from .kmeans import KMeansIndex
            if (ARTIFACTS / "kmeans" / "kmeans.npz").exists():
                _kmeans["idx"] = KMeansIndex.load(ARTIFACTS / "kmeans")
        except Exception:
            _kmeans["idx"] = None
    return _kmeans["idx"]


def _train_lookup(ids: set[str]) -> dict:   # torch/main-model path only
    if _kmeans["train"] is None:
        df = load_split("train")
        _kmeans["train"] = df.set_index("sample_id")[["repo", "file_path", "hunk", "label", "drift_type"]]
    t = _kmeans["train"]
    return {i: t.loc[i].to_dict() for i in ids if i in t.index}


def _add_clusters(scorer, intent: str, rows: list[dict]) -> None:
    """cluster_id, cluster misaligned prior and nearest flagged (misaligned) train hunks. Never fails the request."""
    idx = _kmeans_index()
    if idx is None or getattr(scorer, "backend", "") != "torch" or not rows:
        return
    try:
        emb = scorer.embed([intent] * len(rows), [f"{r['file']}\n{r['text']}" for r in rows])
        if emb.shape[1] != idx.c.shape[1]:
            return
        cl = idx.assign(emb)[0]
        for r, e, c in zip(rows, emb, cl):
            r["cluster_id"] = int(c)
            r["cluster_prior"] = float(idx.prior[int(c)])
            sim = idx.similar(e, n=3, only_misaligned=True)
            look = _train_lookup({x["sample_id"] for x in sim})
            for x in sim:
                m = look.get(x["sample_id"], {})
                x.update({"repo": m.get("repo"), "file": m.get("file_path"), "drift_type": m.get("drift_type"),
                          "text": str(m.get("hunk", ""))[:1200]})
            r["similar"] = sim
    except Exception:
        pass


def analyze(turns, diff: str, model: str = "full", explain: bool = False) -> dict:
    turns = validate_turns(turns)
    hunks, notes = validate_diff(diff)
    scorer, lock = REGISTRY.get(model)
    if explain and getattr(scorer, "backend", "") != "torch":
        raise ApiError(422, f"explain=true is only supported for PyTorch models, not {model!r}")
    t0 = time.perf_counter()
    with lock:
        out = scorer.score(turns, diff)
        intent = out["intent"]
        if not intent.strip():
            raise ApiError(422, "after consolidation no intent remains (all turns were chit-chat or retracted)")
        for r in out["hunks"]:
            p = r["p_misaligned"]
            r["confidence"] = max(p, 1 - p)
            r["cluster_id"] = None
            r["cluster_prior"] = None
            r["similar"] = []
            r["attributions"] = None
        if model == "full":
            _add_clusters(scorer, intent, out["hunks"])
        if explain:
            _explain(scorer, model, intent, out["hunks"])
    return {"model": model, "intent": intent, "turns": turns, "dropped_turns": dropped_turns(turns, intent),
            "threshold": scorer.threshold, "temperature": scorer.T, "n_hunks": out["n_hunks"],
            "notes": [{"file": n.file_path, "kind": n.kind, "detail": n.detail} for n in notes],
            "model_latency_ms": round(out["latency_s"] * 1000, 1), "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            "ranking": [r["index"] for r in out["hunks"]], "hunks": out["hunks"]}


# --------------------------------------------------------------------------- samples
_samples: dict | None = None
_samples_lock = threading.Lock()


def _diff_from_rows(rows) -> str:
    out = []
    for r in rows:
        f, st = r["file_path"], r["file_status"]
        a = "/dev/null" if st == "added" else f"a/{f}"
        b = "/dev/null" if st == "deleted" else f"b/{f}"
        out.append(f"diff --git a/{f} b/{f}\n--- {a}\n+++ {b}\n{r['hunk'].rstrip(chr(10))}\n")
    return "".join(out)


def _load_samples() -> dict:
    """Deterministic curated set from the TEST split (seed 473): aligned-only, each drift x tier, noisy intents."""
    global _samples
    with _samples_lock:
        if _samples is not None:
            return _samples
        pre = ROOT / "data" / "lite_samples.json"   # frozen copy of the curated set (deploy/lite/build_lite_data.py)
        if not (ROOT / "data" / "sift_ia_v1_test.csv").exists() and pre.exists():
            _samples = json.load(open(pre))
            return _samples
        df = load_split("test")
        rng = np.random.default_rng(473)
        by = {sid: g.sort_values("position_in_diff") for sid, g in df.groupby("session_id", sort=True)}
        ok = [s for s, g in by.items() if 2 <= len(g) <= 8 and len(_diff_from_rows(g.to_dict("records"))) < 20_000]
        picks: list[tuple[str, str]] = []

        def take(cands, n, tag):
            cands = [c for c in cands if c not in {p[0] for p in picks}]
            for i in rng.permutation(len(cands))[:n]:
                picks.append((cands[i], tag))
        for drift in ("collateral", "scope_expansion", "purpose_drift"):
            for tier in ("random", "hard"):
                take([s for s in ok if ((by[s].drift_type_clean == drift) & (by[s].mutation_difficulty == tier)).any()],
                     3, f"{drift}/{tier}")
        take([s for s in ok if (by[s].label_clean == "aligned").all()], 5, "all aligned")
        take([s for s in ok if (by[s].intent_noise_type != "none").any() and (by[s].label_clean == "misaligned").any()],
             4, "noisy intent")
        out = {}
        for sid, tag in picks:
            g = by[sid]
            recs = g.to_dict("records")
            out[sid] = {"id": sid, "tag": tag, "repo": recs[0]["repo"], "language": recs[0]["language"],
                        "turns": split_turns(recs[0]["intent_turns"].replace(" || ", "||")),
                        "diff": _diff_from_rows(recs),
                        "hunks": [{"index": i, "sample_id": r["sample_id"], "file": r["file_path"],
                                   "label": r["label_clean"], "drift_type": r["drift_type_clean"],
                                   "difficulty": r["mutation_difficulty"], "noise_types": r["noise_types"],
                                   "text": r["hunk"]} for i, r in enumerate(recs)]}
        _samples = out
        return out


def list_samples() -> list[dict]:
    return [{"id": s["id"], "tag": s["tag"], "repo": s["repo"], "language": s["language"], "n_hunks": len(s["hunks"]),
             "n_misaligned": sum(h["label"] == "misaligned" for h in s["hunks"]), "turns": s["turns"],
             "drift_types": sorted({h["drift_type"] for h in s["hunks"] if h["label"] == "misaligned"})}
            for s in _load_samples().values()]


def get_sample(sid: str) -> dict:
    s = _load_samples().get(sid)
    if not s:
        raise ApiError(404, f"unknown sample id {sid!r}")
    return s


def analyze_sample(sid: str, model: str = "full", explain: bool = False) -> dict:
    s = get_sample(sid)
    res = analyze(s["turns"], s["diff"], model, explain)
    gt = {h["index"]: h for h in s["hunks"]}
    for r in res["hunks"]:
        g = gt.get(r["index"])
        r["ground_truth"] = None if g is None else {k: g[k] for k in ("label", "drift_type", "difficulty", "noise_types")}
        if g is not None:
            r["correct"] = (r["verdict"] == g["label"])
    res["sample_id"] = sid
    return res


# --------------------------------------------------------------------------- results / stats / tests
def find_result(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts or p.suffix.lower() not in RESULT_EXT:
        raise ApiError(404, "result not found")
    for base in RESULT_DIRS:
        f = (base / p).resolve()
        if base.resolve() in f.parents and f.is_file():
            return f
    raise ApiError(404, f"result {rel!r} not found")


def list_results() -> list[dict]:
    out = []
    for base in RESULT_DIRS:
        if base.is_dir():
            for f in sorted(base.rglob("*")):
                if f.is_file() and f.suffix.lower() in RESULT_EXT:
                    rel = f.relative_to(base).as_posix()
                    out.append({"path": rel, "source": base.relative_to(ROOT).as_posix(), "bytes": f.stat().st_size,
                                "type": RESULT_EXT[f.suffix.lower()], "url": f"/api/results/{rel}"})
    return out


def dataset_stats() -> dict:
    p = ROOT / "data" / "dataset_statistics.json"
    if not p.exists():
        raise ApiError(404, "data/dataset_statistics.json not found")
    return json.load(open(p))


def tests_summary() -> dict:
    for name in ("test_summary.json", "tests.json"):
        for base in RESULT_DIRS:
            f = base / name
            if f.exists():
                out = {"available": True, "source": f.relative_to(ROOT).as_posix(), "summary": json.load(open(f))}
                cov = base / "coverage.json"
                if cov.exists():
                    out["coverage"] = json.load(open(cov))
                return out
    return {"available": False, "reason": "no test summary written yet (results/test_summary.json)"}


def health() -> dict:
    ms = REGISTRY.list()
    return {"status": "ok", "models_available": [m["id"] for m in ms if m["available"]],
            "models_loaded": [m["id"] for m in ms if m["loaded"]]}
