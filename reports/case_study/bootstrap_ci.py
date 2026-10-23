"""Session-level bootstrap confidence intervals and the drift-type head result, from stored test predictions.
Reads experiments/runs/*/preds.npz, experiments/results/baseline_scores.npz and results/eval_*.json (thresholds chosen on validation).
Writes reports/case_study/bootstrap_ci.json. Run: PYTHONPATH=src .venv/bin/python reports/case_study/bootstrap_ci.py"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sift.data import load_split          # noqa: E402
from sift.rank import ndcg_at_k           # noqa: E402

te = load_split("test")
y = (te.label_clean == "misaligned").values.astype(int)
sid = te.session_id.values
usess, inv = np.unique(sid, return_inverse=True)
S = len(usess)


def per_session_counts(pred):
    c = np.zeros((S, 4))
    for k, (yy, pp) in enumerate(((1, 1), (0, 1), (1, 0), (0, 0))):    # tp, fp, fn, tn
        np.add.at(c[:, k], inv, ((y == yy) & (pred == pp)).astype(float))
    return c


def macro_f1(c):
    tp, fp, fn, tn = c.sum(0)
    f1p = 2 * tp / max(2 * tp + fp + fn, 1); f1n = 2 * tn / max(2 * tn + fn + fp, 1)
    return (f1p + f1n) / 2


def per_session_ndcg(score):
    out, w = np.full(S, np.nan), np.zeros(S)
    for s in range(S):
        ix = np.where(inv == s)[0]
        if len(ix) >= 2 and y[ix].any():
            out[s] = ndcg_at_k(list(score[ix]), [int(v) for v in y[ix]], 5); w[s] = 1
    return out, w


models = {}
for name, run, ev in (("main", "full", "full"), ("no_denoising", "abl_no_denoise", "abl_no_denoise"), ("student_fp32", "student", "student"), ("minilm_ce", "minilm_ce", "minilm_ce")):
    z = np.load(ROOT / "experiments/runs" / run / "preds.npz", allow_pickle=True)
    thr = json.load(open(ROOT / f"results/eval_{ev}.json"))["test"]["full"]["threshold"]
    lg = z["test_logit"]; p = 1 / (1 + np.exp(-lg))
    models[name] = {"score": lg, "pred": (p >= thr).astype(int), "thr": thr, "stored_macro_f1": json.load(open(ROOT / f"results/eval_{ev}.json"))["test"]["full"]["macro_f1"]}
b = np.load(ROOT / "experiments/results/baseline_scores.npz", allow_pickle=True)
thr5 = json.load(open(ROOT / "experiments/results/baselines.json"))["B5_classification_test"]["threshold_from_val"]
models["B5"] = {"score": b["B5"], "pred": (b["B5"] >= thr5).astype(int), "thr": thr5, "stored_macro_f1": json.load(open(ROOT / "experiments/results/baselines.json"))["B5_classification_test"]["macro_f1"]}
models["B4"] = {"score": b["B4"], "pred": None}
for k, m in models.items():
    m["counts"] = per_session_counts(m["pred"]) if m["pred"] is not None else None
    m["ndcg"], m["w"] = per_session_ndcg(m["score"])
    if m["counts"] is not None:
        m["mf1"] = macro_f1(m["counts"])
        print(k, "macro-F1 recomputed", round(m["mf1"], 4), "stored", round(m["stored_macro_f1"], 4), "NDCG@5 (fixed tie-break)", round(np.nansum(m["ndcg"]) / m["w"].sum(), 4))

rng = np.random.default_rng(473)
B = 2000
idx = rng.integers(0, S, size=(B, S))


def boot_f1(m):
    return np.array([macro_f1(m["counts"][i]) for i in idx])


def boot_ndcg(m):
    nd = np.nan_to_num(m["ndcg"]); w = m["w"]
    return np.array([nd[i].sum() / w[i].sum() for i in idx])


res = {"method": "cluster (session-level) bootstrap, 2000 resamples of the 924 test sessions, seed 473; thresholds fixed from validation; NDCG@5 uses a fixed tie-break",
       "n_sessions": int(S), "recomputed_vs_stored_macro_f1": {k: [float(m["mf1"]), float(m["stored_macro_f1"])] for k, m in models.items() if m["pred"] is not None}}
f1b = {k: boot_f1(m) for k, m in models.items() if m["pred"] is not None}
ndb = {k: boot_ndcg(m) for k, m in models.items()}
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
res["macro_f1"] = {k: {"point": float(models[k]["mf1"]), "ci95": ci(v)} for k, v in f1b.items()}
res["ndcg@5"] = {k: {"point": float(np.nansum(models[k]["ndcg"]) / models[k]["w"].sum()), "ci95": ci(v)} for k, v in ndb.items()}
res["paired_difference_main_minus"] = {}
for k in ("B5", "no_denoising", "student_fp32", "minilm_ce"):
    d = f1b["main"] - f1b[k]; dn = ndb["main"] - ndb[k]
    res["paired_difference_main_minus"][k] = {"macro_f1": {"point": float(models["main"]["mf1"] - models[k]["mf1"]), "ci95": ci(d), "p_main_greater": float((d > 0).mean())},
                                              "ndcg@5": {"point": float(res["ndcg@5"]["main"]["point"] - res["ndcg@5"][k]["point"]), "ci95": ci(dn), "p_main_greater": float((dn > 0).mean())}}
dn = ndb["main"] - ndb["B4"]
res["paired_difference_main_minus"]["B4"] = {"ndcg@5": {"point": float(res["ndcg@5"]["main"]["point"] - res["ndcg@5"]["B4"]["point"]), "ci95": ci(dn), "p_main_greater": float((dn > 0).mean())}}

# drift-type head on truly misaligned test hunks
z = np.load(ROOT / "experiments/runs/full/preds.npz", allow_pickle=True)
classes = ["collateral", "scope_expansion", "purpose_drift"]
mis = y == 1
true_d = te.drift_type.values[mis]
pred_d = np.array(classes)[z["test_drift"][mis].argmax(1)]
cm = [[int(((true_d == a) & (pred_d == b_)).sum()) for b_ in classes] for a in classes]
f1s = []
for i, c in enumerate(classes):
    tp = cm[i][i]; fp = sum(cm[j][i] for j in range(3)) - tp; fn = sum(cm[i]) - tp
    f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
res["drift_type"] = {"note": "misaligned test hunks only; argmax of the drift head; ground truth = drift_type column", "n": int(mis.sum()), "classes": classes, "confusion_rows_true_cols_pred": cm,
                     "accuracy": float(np.trace(np.array(cm)) / mis.sum()), "macro_f1": float(np.mean(f1s)), "f1_per_class": dict(zip(classes, map(float, f1s))),
                     "majority_class_rate": float(max(sum(r) for r in cm) / mis.sum())}
json.dump(res, open(ROOT / "reports/case_study/bootstrap_ci.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("macro_f1", "ndcg@5", "paired_difference_main_minus", "drift_type")}, indent=0)[:3500])
