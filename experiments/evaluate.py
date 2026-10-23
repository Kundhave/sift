"""Score any trained run (or any preds file) on the test split and subsets with the CLAUDE.md s4 metrics.

Usage:
  python experiments/evaluate.py --run full --infer        # fresh inference from runs/full/best.pt (batch<=8)
  python experiments/evaluate.py --run minilm_ce           # use runs/<name>/preds.npz (val_logit,test_logit,val_ids,test_ids)
Writes results/eval_<name>.json. Threshold is tuned on validation only (macro-F1) and applied to test.
Subsets: full, python, js_ts, tier_random, tier_hard, repo_<name>. A tier subset keeps the sessions that contain
a misaligned hunk of that tier, drops misaligned hunks of the other tier, and keeps all aligned hunks of those sessions.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sift.data import load_split                       # noqa: E402
from sift import rank as R                             # noqa: E402


def subset_masks(te) -> dict:
    y = (te.label_clean == "misaligned").values
    m = {"full": np.ones(len(te), bool),
         "python": (te.language == "Python").values,
         "js_ts": te.language.isin(["JavaScript", "TypeScript"]).values}
    for repo in sorted(te.repo.unique()):
        m[f"repo_{repo}"] = (te.repo == repo).values
    for tier in ("random", "hard"):
        mis_t = y & (te.mutation_difficulty == tier).values
        sess = set(te.session_id[mis_t])
        in_s = te.session_id.isin(sess).values
        m[f"tier_{tier}"] = in_s & (~y | mis_t)
    return m


def eval_subset(p, y, sess, tiers, thr, ks=(1, 3, 5, 10)):
    from sklearn.metrics import roc_auc_score, average_precision_score
    out = R.cls_metrics(y, p >= thr)
    out["threshold"] = float(thr); out["n"] = int(len(y)); out["n_misaligned"] = int(y.sum())
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, p)); out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = out["pr_auc"] = None
    out["ece"] = R.ece(y, p)
    out["pairwise_auc"] = R.pairwise_auc_by_tier(p, y, sess, tiers)
    groups = R.rankable(R.session_index(sess), y)
    out["n_rankable_sessions"] = len(groups)
    for k in ks:
        rm = R.rank_metrics(p, y, sess, k=k)
        out[f"ndcg@{k}"] = rm[f"ndcg@{k}"] if groups else None
        if k == 1:
            out["mrr"] = rm["mrr"] if groups else None
        if k in (1, 3):
            rng = np.random.RandomState(0)
            vals = []
            for ix in groups:
                sc = p[ix] + rng.rand(len(ix)) * 1e-9
                vals.append(R.precision_at_k(sc, [int(y[j]) for j in ix], k))
            out[f"p@{k}"] = float(np.mean(vals)) if vals else None
    return out


def infer_run(name: str, bs: int):
    import torch
    from sift.model import SiftModel, load_tokenizer
    from sift.train import PairDataset, predict
    cfg = json.load(open(ROOT / "experiments/runs" / name / "config.json"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(cfg["encoder"])
    model = SiftModel(cfg["encoder"]).to(dev)
    model.load_state_dict(torch.load(ROOT / "experiments/runs" / name / "best.pt", map_location=dev))
    va, te = load_split("val"), load_split("test")
    mk = lambda d: PairDataset(d, tok, cfg["max_len"], cfg.get("input_mode", "full"), cfg.get("intent_source", "consolidated"), "label_clean")
    t0 = time.time()
    pv = predict(model, mk(va), dev, bs=bs)["align_logit"]; pt = predict(model, mk(te), dev, bs=bs)["align_logit"]
    info = {"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None, "batch": bs,
            "infer_time_s": time.time() - t0,
            "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None}
    return pv, pt, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run name under experiments/runs")
    ap.add_argument("--preds", help="alternative npz with val_logit/test_logit (+ val_ids/test_ids)")
    ap.add_argument("--name", help="output name (default: run)")
    ap.add_argument("--infer", action="store_true", help="fresh model inference instead of preds.npz")
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    name = a.name or a.run or Path(a.preds).parent.name
    va, te = load_split("val"), load_split("test")
    info = {}
    if a.infer:
        lv, lt, info = infer_run(a.run, a.bs)
        check = ROOT / "experiments/runs" / a.run / "preds.npz"
        if check.exists():
            z = np.load(check, allow_pickle=True)
            info["max_abs_logit_diff_vs_saved_preds"] = float(np.abs(z["test_logit"] - lt).max())
    else:
        z = np.load(a.preds or ROOT / "experiments/runs" / a.run / "preds.npz", allow_pickle=True)
        lv, lt = z["val_logit"], z["test_logit"]
        assert list(z["test_ids"]) == list(te.sample_id), "preds not aligned with test CSV order"
        assert list(z["val_ids"]) == list(va.sample_id), "preds not aligned with val CSV order"
        info["source"] = str(a.preds or f"runs/{a.run}/preds.npz")
    pv, pt = 1 / (1 + np.exp(-lv)), 1 / (1 + np.exp(-lt))
    yv = (va.label_clean == "misaligned").values.astype(int); yt = (te.label_clean == "misaligned").values.astype(int)
    thr = R.best_threshold(yv, pv)
    res = {"name": name, "threshold_source": "validation macro-F1 grid (0.02..0.98)", "info": info, "val": eval_subset(pv, yv, list(va.session_id), list(va.mutation_difficulty), thr), "test": {}}
    for k, m in subset_masks(te).items():
        idx = np.where(m)[0]
        res["test"][k] = eval_subset(pt[idx], yt[idx], list(te.session_id.values[idx]), list(te.mutation_difficulty.values[idx]), thr)
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    json.dump(res, open(out / f"eval_{name}.json", "w"), indent=2, default=float)
    f = res["test"]["full"]
    print(name, {k: (round(f[k], 4) if isinstance(f[k], float) else f[k]) for k in ("macro_f1", "roc_auc", "pr_auc", "ndcg@5", "mrr", "ece")})
    print({k: round(v["macro_f1"], 3) for k, v in res["test"].items()})


if __name__ == "__main__":
    main()
