"""Phase 7: build every comparison table and graph from real result JSONs. Idempotent; run again whenever more runs finish.

  CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python experiments/make_tables.py

Inputs (all optional; missing ones are skipped and listed under 'not yet available' in results/make_tables_status.json):
  experiments/runs/<run>/{results.json,preds.npz,config.json,best.pt}, results/eval_<run>.json, experiments/results/baselines.json,
  experiments/results/gan.json, results/{gan_vs_main,kmeans,calibration,latency_size_onnx,noise_robustness,hpo}.json, data/dataset_statistics.json.
Missing evaluations of finished runs are created with experiments/evaluate.py (preds.npz mode, CPU, seconds). No numbers are invented:
a model without a results file simply has no row. Everything is a single seed (473) unless a source file says otherwise.
Outputs in results/: master_comparison.{csv,json}, ablation.{csv,json}, dataset_statistics.csv, hyperparameters.csv, latency_size.csv,
noise_robustness_table.csv (if the sweep exists), per_tier_table.csv, and PNG graphs. SUMMARY.md gets an auto-generated block.
"""
from __future__ import annotations
import csv, json, os, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

RES, RUNS, XRES = ROOT / "results", ROOT / "experiments/runs", ROOT / "experiments/results"
RES.mkdir(exist_ok=True)
status: dict = {"made": [], "skipped": {}}
PAL = ["#1b6ca8", "#d1495b", "#2a9d8f", "#e9a03b", "#7b5ea7", "#555555", "#8c564b", "#17becf", "#bcbd22", "#e377c2", "#4c9f38", "#9c6b30"]


def jload(p):
    p = Path(p)
    try:
        return json.load(open(p)) if p.exists() else None
    except Exception:
        return None


def skip(name, why): status["skipped"][name] = why


def made(name): status["made"].append(name)


def mb(p):
    p = Path(p)
    return round(p.stat().st_size / 1e6, 1) if p.exists() else None


def write_csv(name, rows, cols=None):
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    with open(RES / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k))) for k in cols})
    made(name)


def save_fig(fig, name):
    fig.tight_layout(); fig.savefig(RES / name, dpi=140); plt.close(fig); made(name)


# ---------------------------------------------------------------- 0. make sure finished runs have eval_<run>.json
DISPLAY = {  # run dir -> (label, group)
    "full": ("Main: ModernBERT-base", "main"), "minilm_ce": ("MiniLM-L6 cross-encoder", "dl_baseline"),
    "codebert": ("CodeBERT cross-encoder", "dl_baseline"), "bilstm": ("Siamese BiLSTM", "dl_baseline"),
    "student": ("Distilled MiniLM student (fp32)", "student"), "student_nodistill": ("MiniLM student, no distillation", "student"),
    "abl_hunk_only": ("Ablation: hunk-only", "ablation"), "abl_intent_only": ("Ablation: intent-only", "ablation"),
    "abl_no_consolidation": ("Ablation: no intent consolidation", "ablation"), "abl_no_focal": ("Ablation: no focal loss (CE)", "ablation"),
    "abl_no_multitask": ("Ablation: alignment head only", "ablation"), "abl_no_denoise": ("Ablation: no denoising", "ablation"),
    "full_tuned": ("Main, Optuna-tuned", "main"),
}
SKIP_RUNS = {"smoke"}
runs = sorted(p.parent.name for p in RUNS.glob("*/results.json") if p.parent.name not in SKIP_RUNS and not p.parent.name.startswith("cl_oof"))
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "src"))
for r in runs:
    if not (RES / f"eval_{r}.json").exists() and (RUNS / r / "preds.npz").exists():
        cp = subprocess.run([sys.executable, str(ROOT / "experiments/evaluate.py"), "--run", r], env=env, capture_output=True, text=True, timeout=900)
        if cp.returncode != 0:
            skip(f"eval_{r}.json", "evaluate.py failed: " + cp.stderr.strip().splitlines()[-1][:160])
EV = {r: jload(RES / f"eval_{r}.json") for r in runs}
EV = {r: e for r, e in EV.items() if e}
for r in DISPLAY:
    if r not in EV:
        skip(f"model:{r}", "no finished results.json / eval json yet (not run or still training); no row, no invented numbers")

# ---------------------------------------------------------------- helpers
def sub(e, s="full"): return (e or {}).get("test", {}).get(s, {})
def auc(e, tier): return (sub(e, f"tier_{tier}").get("pairwise_auc", {}).get(tier, {}) or {}).get("auc")
FIELDS = ["accuracy", "macro_f1", "precision", "recall", "specificity", "roc_auc", "pr_auc", "ndcg@5", "mrr", "p@1", "ece"]


def row_from_eval(label, group, e, run=None):
    f = sub(e)
    d = {"model": label, "group": group, "run": run, "seed": 473}
    d.update({k: f.get(k) for k in FIELDS})
    d["auc_random"], d["auc_hard"] = auc(e, "random"), auc(e, "hard")
    d["ndcg5_random_tier"], d["ndcg5_hard_tier"] = sub(e, "tier_random").get("ndcg@5"), sub(e, "tier_hard").get("ndcg@5")
    d["macro_f1_random_tier"], d["macro_f1_hard_tier"] = sub(e, "tier_random").get("macro_f1"), sub(e, "tier_hard").get("macro_f1")
    d["threshold"] = f.get("threshold")
    d["ckpt_size_mb"] = mb(RUNS / run / "best.pt") if run else None
    d["source"] = f"results/eval_{run}.json"
    return d


# ---------------------------------------------------------------- 1. master comparison
base = jload(XRES / "baselines.json") or {}
rows = []
if base:
    nd, pa = base["ndcg@5"], base["pairwise_auc"]
    b5 = base.get("B5_classification_test", {})
    for k, v in nd.items():
        d = {"model": k, "group": "non_neural", "seed": 473, "ndcg@5": v, "source": "experiments/results/baselines.json"}
        if k in pa: d["auc_random"], d["auc_hard"] = pa[k]["random"], pa[k]["hard"]
        if k.startswith("B5"):
            d.update({x: b5.get(x) for x in ("accuracy", "macro_f1", "precision", "recall", "specificity")}); d["threshold"] = b5.get("threshold_from_val")
        rows.append(d)
else:
    skip("baselines", "experiments/results/baselines.json missing")
for r in ["bilstm", "minilm_ce", "codebert", "full", "full_tuned"]:
    if r in EV: rows.append(row_from_eval(*DISPLAY[r], EV[r], r))
gan = jload(RES / "gan_vs_main.json") or {}
for k, v in (gan.get("compared") or {}).items():
    if k.startswith("full"): continue
    rows.append({"model": ("GAN-style baseline: " + k) if k.startswith("gan") else "Frozen-embedding " + k, "group": "gan" if k.startswith("gan") else "frozen_embedding",
                 "seed": 473, "macro_f1": v.get("macro_f1"), "ndcg@5": v.get("ndcg@5"), "auc_random": v.get("auc_random"), "auc_hard": v.get("auc_hard"),
                 "source": "results/gan_vs_main.json"})
if not gan: skip("gan row", "results/gan_vs_main.json missing")
if "student" in EV: rows.append(row_from_eval(*DISPLAY["student"], EV["student"], "student"))
if "student_nodistill" in EV: rows.append(row_from_eval(*DISPLAY["student_nodistill"], EV["student_nodistill"], "student_nodistill"))
onnx = jload(RES / "latency_size_onnx.json") or {}
for key, label in (("student_onnx_fp32", "Student ONNX fp32"), ("student_onnx_int8", "Student ONNX INT8 (deployed target)")):
    o = onnx.get(key)
    if o:
        t = o["test"]
        rows.append({"model": label, "group": "student", "seed": 473, **{k: t.get(k) for k in ("accuracy", "macro_f1", "precision", "recall", "specificity", "ndcg@5", "p@1")},
                     "ece": (o.get("calibration") or {}).get("ece_test_after"), "threshold": t.get("threshold"), "size_mb": o["size_mb"],
                     "cpu_p50_ms": o["latency_ms"]["p50_ms"], "cpu_p95_ms": o["latency_ms"]["p95_ms"], "source": "results/latency_size_onnx.json (tier AUC / ROC not stored)"})
if not onnx: skip("INT8 row", "results/latency_size_onnx.json missing")
# sizes/latency for others
for d in rows:
    if d.get("run") and d.get("size_mb") is None: d["size_mb"] = d.get("ckpt_size_mb")
write_csv("master_comparison.csv", rows)
json.dump({"seed": "single seed (473) for every row; no confidence intervals", "test_split": "Sift-IA v1.0 test (httpx, attrs, axios), threshold tuned on validation",
           "note": "B1/B2 are scalar orderings; B3 has no calibrated classifier; GAN/frozen rows come from experiments/results/gan.json (embedding-based, own thresholds); "
                   "INT8 row: tier AUC not stored. Empty cell = not measured.",
           "rows": rows}, open(RES / "master_comparison.json", "w"), indent=2, default=float); made("master_comparison.json")

# per-tier table (macro-F1 / NDCG / AUC) from eval jsons
pt = []
for r in ["full", "full_tuned", "minilm_ce", "codebert", "bilstm", "student", "student_nodistill"] + [x for x in DISPLAY if x.startswith("abl_")]:
    if r in EV:
        for s in ("full", "tier_random", "tier_hard", "python", "js_ts"):
            f = sub(EV[r], s)
            if f: pt.append({"model": DISPLAY[r][0], "subset": s, "n": f.get("n"), "macro_f1": f.get("macro_f1"), "ndcg@5": f.get("ndcg@5"),
                             "roc_auc": f.get("roc_auc"), "pr_auc": f.get("pr_auc"), "pairwise_auc_random": (f["pairwise_auc"].get("random") or {}).get("auc"),
                             "pairwise_auc_hard": (f["pairwise_auc"].get("hard") or {}).get("auc")})
if pt: write_csv("per_tier_table.csv", pt)

# ---------------------------------------------------------------- 2. ablation
abl = []
full_row = row_from_eval(*DISPLAY["full"], EV["full"], "full") if "full" in EV else None
order = [("full", "1. Full model")] + [(r, f"{i}. {DISPLAY[r][0].replace('Ablation: ', '')}") for i, r in
         enumerate(["abl_hunk_only", "abl_intent_only", "abl_no_consolidation", "abl_no_focal", "abl_no_multitask", "abl_no_denoise"], 2)]
for r, lab in order:
    if r in EV: abl.append({**row_from_eval(lab, "ablation", EV[r], r), "size_mb": mb(RUNS / r / "best.pt")})
    else: abl.append({"model": lab, "group": "ablation", "run": r, "source": "NOT RUN / not finished (no results)"})
km = jload(RES / "kmeans.json")
if km:
    h = km["hybrid"]
    for key, lab in (("logit_only_platt", "9a. k-means features OFF (Platt on logit)"), ("logit_plus_cluster_features", "9b. k-means features ON (hybrid)")):
        t = h[key]["test"]
        abl.append({"model": lab, "group": "ablation_kmeans", "run": "kmeans", "seed": 473, "macro_f1": t.get("macro_f1"), "accuracy": t.get("accuracy"), "precision": t.get("precision"),
                    "recall": t.get("recall"), "specificity": t.get("specificity"), "ndcg@5": t.get("ndcg@5"), "ece": t.get("ece"),
                    "auc_random": (t["pairwise_auc"].get("random") or {}).get("auc") if isinstance(t.get("pairwise_auc"), dict) else None,
                    "auc_hard": (t["pairwise_auc"].get("hard") or {}).get("auc") if isinstance(t.get("pairwise_auc"), dict) else None, "source": "results/kmeans.json hybrid"})
else:
    skip("ablation k-means rows", "results/kmeans.json missing")
if "student" in EV: abl.append({**row_from_eval("10. Distilled MiniLM student (fp32)", "ablation", EV["student"], "student"), "size_mb": mb(RUNS / "student/best.pt")})
else: abl.append({"model": "10. Distilled MiniLM student", "group": "ablation", "source": "NOT RUN"})
i8 = next((r for r in rows if r["model"].startswith("Student ONNX INT8")), None)
if i8: abl.append({**i8, "model": "11. INT8 ONNX student (deployed target)", "group": "ablation"})
else: abl.append({"model": "11. INT8 ONNX student", "group": "ablation", "source": "NOT RUN"})
if full_row:
    for d in abl:
        if d.get("macro_f1") is not None and d["group"] == "ablation" and d is not abl[0]:
            d["delta_macro_f1_vs_full"] = d["macro_f1"] - full_row["macro_f1"]
            if d.get("ndcg@5") is not None: d["delta_ndcg5_vs_full"] = d["ndcg@5"] - full_row["ndcg@5"]
            if d.get("auc_hard") is not None and full_row.get("auc_hard") is not None: d["delta_auc_hard_vs_full"] = d["auc_hard"] - full_row["auc_hard"]
write_csv("ablation.csv", abl)
json.dump({"seeds": "single seed (473) for every row: differences of about 0.01 macro-F1 are within the run-to-run noise one should expect and are not significant",
           "rows": abl}, open(RES / "ablation.json", "w"), indent=2, default=float); made("ablation.json")

# ---------------------------------------------------------------- 3. dataset statistics, hyperparameters, latency/size
ds = jload(ROOT / "data/dataset_statistics.json")
if ds:
    flat = []
    def walk(d, p=""):
        if isinstance(d, dict):
            for k, v in d.items(): walk(v, f"{p}{k}.")
        else: flat.append({"statistic": p[:-1], "value": d if not isinstance(d, list) else json.dumps(d)})
    walk(ds); write_csv("dataset_statistics.csv", flat)
else: skip("dataset_statistics.csv", "data/dataset_statistics.json missing")

hp = []
for r in runs:
    c = jload(RUNS / r / "config.json") or {}
    if c: hp.append({"run": r, **{k: c.get(k) for k in ("encoder", "max_len", "micro_bs", "accum", "epochs", "patience", "lr", "seed", "input_mode", "intent_source", "ce", "no_multitask", "rule_filter", "tau", "alpha")},
                     "best_val_macro_f1": (jload(RUNS / r / "results.json") or {}).get("best_val_macro_f1"), "train_time_s": (jload(RUNS / r / "results.json") or {}).get("train_time_s"),
                     "peak_gpu_mem_gb": (jload(RUNS / r / "results.json") or {}).get("peak_gpu_mem_gb")})
if hp: write_csv("hyperparameters.csv", hp)
hpo = jload(RES / "hpo.json")
if hpo: json.dump(hpo, open(RES / "hyperparameters_optuna.json", "w"), indent=2); made("hyperparameters_optuna.json (copy of hpo.json)")

ls = []
cpu = (onnx.get("cpu") or {}).get("cpu")
for r in runs:
    if (RUNS / r / "best.pt").exists(): ls.append({"model": DISPLAY.get(r, (r,))[0], "format": "PyTorch checkpoint", "size_mb": mb(RUNS / r / "best.pt"), "cpu_p50_ms": None, "cpu_p95_ms": None, "note": "latency not measured"})
for key, fmt in (("student_onnx_fp32", "ONNX fp32"), ("student_onnx_int8", "ONNX INT8")):
    o = onnx.get(key)
    if o: ls.append({"model": "MiniLM student", "format": fmt, "size_mb": o["size_mb"], "cpu_p50_ms": o["latency_ms"]["p50_ms"], "cpu_p95_ms": o["latency_ms"]["p95_ms"],
                     "note": f"{cpu}, {onnx.get('threads')} threads, batch 1, tokenisation included; CPU shared with a training job"})
for fn in sorted(RES.glob("latency*full*.json")):
    j = jload(fn) or {}
    ls = [d for d in ls if not (d["model"].startswith("Main:") and d["format"] == "PyTorch checkpoint")]   # replaced by the measured row
    g = j.get("cuda") or {}
    ls.append({"model": "Main ModernBERT", "format": "PyTorch fp32 (CPU)", "size_mb": j.get("size_mb"), "cpu_p50_ms": j.get("p50_ms"), "cpu_p95_ms": j.get("p95_ms"),
               "note": f"{(j.get('cpu_info') or {}).get('cpu')}, {j.get('torch_threads')} torch threads, batch 1, L 512, tokenisation incl., idle machine; GPU p50/p95 {g.get('p50_ms', float('nan')):.1f}/{g.get('p95_ms', float('nan')):.1f} ms, peak {g.get('peak_gpu_mem_mb', float('nan')):.0f} MB (from {fn.name})"})
if ls: write_csv("latency_size.csv", ls)

nr = jload(RES / "noise_robustness.json")
if nr and nr.get("results"):
    write_csv("noise_robustness_table.csv", [{"noise_rate": x.get("noise_rate"), "arm": x.get("arm"), "n_train_used": x.get("n_train_used"), "realised_flip_rate": x.get("realised_flip_rate"),
                                              "macro_f1": x["test"].get("macro_f1"), "ndcg@5": x["test"].get("ndcg@5"), "roc_auc": x["test"].get("roc_auc")} for x in nr["results"]])
else:
    skip("noise_robustness_table.csv", "results/noise_robustness.json not produced yet (sweep not run)")

# ---------------------------------------------------------------- 4. graphs
def logits(r):
    p = RUNS / r / "preds.npz"
    if not p.exists(): return None
    z = np.load(p, allow_pickle=True); return z["test_logit"]

from sift.data import load_split                                    # noqa: E402
from evaluate import subset_masks                                    # noqa: E402
te = load_split("test"); yt = (te.label_clean == "misaligned").values.astype(int); masks = subset_masks(te)
curve_runs = [r for r in ["full", "full_tuned", "minilm_ce", "codebert", "bilstm", "student", "student_nodistill"] if r in EV and logits(r) is not None]
col = {r: PAL[i % len(PAL)] for i, r in enumerate(curve_runs)}

# (1) loss curves
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8)); n = 0
for r in runs:
    h = (jload(RUNS / r / "results.json") or {}).get("history") or []
    if r in SKIP_RUNS or not h or "train_loss" not in h[0]: continue
    c = col.get(r, PAL[(n + 5) % len(PAL)]); n += 1
    ep = [x["epoch"] for x in h]
    ax[0].plot(ep, [x["train_loss"] for x in h], "-o", ms=3, label=DISPLAY.get(r, (r,))[0], color=c)
    if "val_macro_f1" in h[0]: ax[1].plot(ep, [x["val_macro_f1"] for x in h], "-o", ms=3, color=c)
ax[0].set(title="Training loss per epoch (each run's own loss; not comparable across loss types)", xlabel="epoch", ylabel="train loss"); ax[1].set(title="Validation macro-F1 per epoch", xlabel="epoch", ylabel="macro-F1")
ax[0].title.set_fontsize(8); ax[0].legend(fontsize=6)
if n: save_fig(fig, "training_validation_loss_curves.png")
else: plt.close(fig); skip("training_validation_loss_curves.png", "no run histories")

# (2) PR, (3) ROC per tier
from sklearn.metrics import precision_recall_curve, roc_curve   # noqa: E402
_bs = XRES / "baseline_scores.npz"
BSC = {k: np.load(_bs)[k] for k in ("B4", "B5")} if _bs.exists() else {}     # written by experiments/baseline_scores.py
BLAB = {"B4": "B4 TF-IDF dissimilarity", "B5": "B5 Logistic regression"}
if curve_runs:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for r in curve_runs:
        pr, rc, _ = precision_recall_curve(yt, logits(r)); ax.plot(rc, pr, label=DISPLAY[r][0], color=col[r], lw=1.4)
    for k, v in BSC.items():
        pr, rc, _ = precision_recall_curve(yt, v); ax.plot(rc, pr, "--", lw=1.1, color="black" if k == "B4" else "dimgray", label=BLAB[k])
    ax.axhline(yt.mean(), ls=":", color="grey", label=f"prevalence {yt.mean():.2f}"); ax.set(xlabel="recall", ylabel="precision", title="Precision-recall, full test set"); ax.legend(fontsize=6)
    save_fig(fig, "pr_curves_all_models.png")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for a, s in zip(axs, ("full", "tier_random", "tier_hard")):
        m = masks[s]
        for r in curve_runs:
            fpr, tpr, _ = roc_curve(yt[m], logits(r)[m]); a.plot(fpr, tpr, color=col[r], lw=1.3, label=DISPLAY[r][0])
        for k, v in BSC.items():
            fpr, tpr, _ = roc_curve(yt[m], v[m]); a.plot(fpr, tpr, "--", lw=1.0, color="black" if k == "B4" else "dimgray", label=BLAB[k])
        a.plot([0, 1], [0, 1], ":", color="grey"); a.set(title={"full": "all test", "tier_random": "random tier", "tier_hard": "hard tier"}[s], xlabel="FPR", ylabel="TPR")
    axs[0].legend(fontsize=6); save_fig(fig, "roc_curves_per_tier.png")
else: skip("pr_curves_all_models.png / roc_curves_per_tier.png", "no preds.npz")

# (4) confusion matrix of main model
if "full" in EV:
    f = sub(EV["full"]); cm = np.array([[f["tn"], f["fp"]], [f["fn"], f["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2): ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.0%})", ha="center", va="center", color="black" if cm[i, j] / cm[i].sum() < .6 else "white")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["aligned", "misaligned"], yticklabels=["aligned", "misaligned"], xlabel="predicted", ylabel="true",
           title=f"Main model, test, threshold {f['threshold']:.2f}")
    save_fig(fig, "confusion_matrix_main.png")

# (5) NDCG@K vs K
fig, ax = plt.subplots(figsize=(6, 4)); ks = [1, 3, 5, 10]; plotted = 0
for r in [x for x in ["full", "full_tuned", "minilm_ce", "codebert", "bilstm", "student", "student_nodistill"] if x in EV]:
    f = sub(EV[r]); v = [f.get(f"ndcg@{k}") for k in ks]
    if all(x is not None for x in v): ax.plot(ks, v, "-o", ms=3, label=DISPLAY[r][0], color=col.get(r, PAL[plotted % 12])); plotted += 1
from sift import rank as _R                                                     # noqa: E402
if BSC:
    for k, v in BSC.items():
        ax.plot(ks, [_R.rank_metrics(v, yt, list(te.session_id), k=kk)[f"ndcg@{kk}"] for kk in ks], "--s", ms=3, color="black" if k == "B4" else "dimgray", label=BLAB[k])
else:
    for k, v in (base.get("ndcg@5") or {}).items(): ax.plot([5], [v], "s", ms=6, label=k + " (NDCG@5 only)")
ax.set(xlabel="K", ylabel="NDCG@K", title="NDCG@K versus K, test (447-456 rankable sessions)", xticks=ks); ax.legend(fontsize=6, ncol=2)
if plotted: save_fig(fig, "ndcg_vs_k_all_models.png")
else: plt.close(fig); skip("ndcg_vs_k_all_models.png", "no eval jsons")

# (6) per-tier pairwise AUC bars
labs, ar, ah = [], [], []
for d in rows:
    if d.get("auc_random") is not None and d.get("auc_hard") is not None: labs.append(d["model"]); ar.append(d["auc_random"]); ah.append(d["auc_hard"])
if labs:
    fig, ax = plt.subplots(figsize=(10, 4.2)); x = np.arange(len(labs))
    ax.bar(x - .2, ar, .4, label="random tier", color=PAL[0]); ax.bar(x + .2, ah, .4, label="hard tier", color=PAL[1]); ax.axhline(.5, ls=":", color="grey")
    ax.set_xticks(x); ax.set_xticklabels([l.replace("GAN-style baseline: ", "GAN: ").replace("Main: ", "") for l in labs], rotation=40, ha="right", fontsize=7)
    ax.set(ylabel="within-session pairwise AUC", ylim=(.4, 1), title="Pairwise AUC by difficulty tier (0.5 = chance)"); ax.legend(fontsize=7); save_fig(fig, "per_tier_pairwise_auc_bars.png")

# (11) ablation bars
ar_ = [d for d in abl if d.get("macro_f1") is not None]
if len(ar_) >= 2:
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), sharey=False)
    names = [d["model"].split(". ", 1)[-1][:34] for d in ar_]
    for a, (k, t) in zip(ax, (("macro_f1", "macro-F1"), ("ndcg@5", "NDCG@5"), ("auc_hard", "hard-tier pairwise AUC"))):
        v = [d.get(k) if d.get(k) is not None else np.nan for d in ar_]
        a.barh(range(len(ar_)), v, color=[PAL[0] if "Full" in d["model"] else PAL[2] if "student" in d["model"].lower() else PAL[3] for d in ar_]); a.set_yticks(range(len(ar_)))
        a.set_yticklabels(names if a is ax[0] else [], fontsize=7); a.invert_yaxis(); a.set_title(t, fontsize=9)
        lo = np.nanmin(v); a.set_xlim(max(0, lo - .08), np.nanmax(v) + .03)
        for i, x in enumerate(v):
            if not np.isnan(x): a.text(x, i, f" {x:.3f}", va="center", fontsize=6)
    fig.suptitle("Ablations (test, single seed; truncated x-axes; bars with no value = not measured)", fontsize=9); save_fig(fig, "ablation_bars.png")
else: skip("ablation_bars.png", "fewer than 2 ablation rows measured")

# (12) latency vs size
pts = [d for d in ls if d.get("cpu_p95_ms") is not None and d.get("size_mb") is not None]
if pts:
    fig, ax = plt.subplots(figsize=(5.4, 4))
    for i, d in enumerate(pts): ax.scatter(d["size_mb"], d["cpu_p95_ms"], s=60, color=PAL[i]); ax.annotate(f"{d['model']} {d['format']}", (d["size_mb"], d["cpu_p95_ms"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(100, ls="--", color="grey", lw=.8); ax.axvline(30, ls="--", color="grey", lw=.8); ax.set(xlabel="model size (MB)", ylabel="CPU p95 latency (ms)", title=f"Latency vs size (targets dashed)\n{cpu}")
    ax.title.set_fontsize(8); save_fig(fig, "latency_vs_size_scatter.png")
    if not any(d["model"].startswith("Main") for d in pts): status["skipped"]["latency point: main ModernBERT"] = "no CPU latency measured for the full model; plotted points are the measured student ONNX variants only"
else: skip("latency_vs_size_scatter.png", "no latency measurements")

# (8) noise-robustness curve (the sweep script writes its own PNG; this makes the same plot name only if absent)
if nr and nr.get("results") and not any(RES.glob("noise_robust*.png")):
    fig, ax = plt.subplots(figsize=(5.4, 4))
    for arm in sorted({x["arm"] for x in nr["results"]}):
        pts_ = sorted((x["noise_rate"], x["test"]["macro_f1"]) for x in nr["results"] if x["arm"] == arm); ax.plot(*zip(*pts_), "-o", label=arm)
    ax.set(xlabel="injected label-noise rate", ylabel="test macro-F1", title="Noise robustness (reduced setting)"); ax.legend(); save_fig(fig, "noise_robustness_curve.png")

# ---------------------------------------------------------------- 5. SUMMARY.md auto block
def fmt(x): return "n/a" if x is None else f"{x:.3f}"
L = ["<!-- make_tables:begin (auto-generated by experiments/make_tables.py; do not edit by hand) -->", "## Measured comparison (auto-generated, test split, single seed 473)", ""]
L.append("| Model | macro-F1 | NDCG@5 | AUC random | AUC hard | ROC-AUC | ECE |"); L.append("|---|---|---|---|---|---|---|")
for d in rows: L.append(f"| {d['model']} | {fmt(d.get('macro_f1'))} | {fmt(d.get('ndcg@5'))} | {fmt(d.get('auc_random'))} | {fmt(d.get('auc_hard'))} | {fmt(d.get('roc_auc'))} | {fmt(d.get('ece'))} |")
L += ["", "Empty (n/a) means not measured for that model; models that have not finished are absent: " + (", ".join(sorted(k.split(':', 1)[1] for k in status['skipped'] if k.startswith('model:'))) or "none") + ".", ""]
if full_row:
    b4 = next((d for d in rows if d["model"].startswith("B4")), None)
    L.append("Weak spots (stated plainly):")
    L.append(f"- The main model reaches macro-F1 {fmt(full_row['macro_f1'])} and NDCG@5 {fmt(full_row['ndcg@5'])}; hard-tier pairwise AUC is only {fmt(full_row['auc_hard'])} (random tier {fmt(full_row['auc_random'])}).")
    if b4 and b4.get("ndcg@5") is not None:
        L.append(f"- The non-neural TF-IDF dissimilarity baseline B4 gets NDCG@5 {fmt(b4['ndcg@5'])} and random-tier AUC {fmt(b4.get('auc_random'))}; the main model's margin on those is {full_row['ndcg@5'] - b4['ndcg@5']:+.3f} NDCG@5 and {full_row['auc_random'] - b4['auc_random']:+.3f} random-tier AUC. Its real gain is on the hard tier ({fmt(full_row['auc_hard'])} vs B4 {fmt(b4.get('auc_hard'))}).")
    ab = [d for d in abl if d.get("delta_macro_f1_vs_full") is not None]
    if ab:
        best = max(ab, key=lambda d: d["delta_macro_f1_vs_full"]); worst = min(ab, key=lambda d: d["delta_macro_f1_vs_full"])
        L.append(f"- Ablations (single seed; differences of ~0.01 are noise): largest macro-F1 gain vs full: {best['model']} ({best['delta_macro_f1_vs_full']:+.3f}); largest loss: {worst['model']} ({worst['delta_macro_f1_vs_full']:+.3f}).")
    L.append("- Labels are construction-determined with no human validation; misaligned hunks are constructed, so construction artefacts may be learnt; one seed, no confidence intervals.")
L += ["", "Tables: master_comparison, ablation, per_tier_table, dataset_statistics, hyperparameters, latency_size (+ noise_robustness_table when the sweep exists). Graphs: see PNGs in results/.", "<!-- make_tables:end -->"]
block = "\n".join(L)
sp = RES / "SUMMARY.md"; s = sp.read_text() if sp.exists() else "# Results summary\n"
if "<!-- make_tables:begin" in s:
    a = s.index("<!-- make_tables:begin"); b = s.index("<!-- make_tables:end -->") + len("<!-- make_tables:end -->"); s = s[:a] + block + s[b:]
else:
    parts = s.split("\n", 1); s = parts[0] + "\n\n" + block + "\n" + (parts[1] if len(parts) > 1 else "")
sp.write_text(s)

n_png = len(list(RES.glob("*.png"))); n_tab = len(list(RES.glob("*.csv")))
status["counts"] = {"png_in_results": n_png, "csv_tables_in_results": n_tab}
json.dump(status, open(RES / "make_tables_status.json", "w"), indent=2)
print(f"made {len(status['made'])} files; results/ now has {n_png} PNG and {n_tab} CSV; skipped: {len(status['skipped'])}")
for k, v in status["skipped"].items(): print("  skipped", k, "->", v)
