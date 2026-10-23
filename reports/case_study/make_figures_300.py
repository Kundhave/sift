"""Regenerates every figure of the case study at 300 dpi from results/*.json|csv and experiments/*.
Output: reports/figures_300/*.png (the app's own PNGs in results/ are not touched).
Run: PYTHONPATH=src .venv/bin/python reports/case_study/make_figures_300.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
RES, XR, RUNS = ROOT / "results", ROOT / "experiments" / "results", ROOT / "experiments" / "runs"
OUT = ROOT / "reports" / "figures_300"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.3,
                     "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.dpi": 300, "axes.linewidth": 0.6})
COL = {"main": "#1b3a6b", "minilm": "#d98200", "codebert": "#8c564b", "bilstm": "#7f7f7f", "student": "#2a9d8f", "nodistill": "#9bd0c8",
       "b4": "#222222", "b5": "#666666", "frozen": "#c44e52", "gan": "#8172b3", "gan2": "#b8a9e0"}
W1, W2 = 3.4, 7.0
JSON_SRC: dict[str, list[str]] = {}


def J(p):
    return json.load(open(p))


def save(fig, name, sources):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    JSON_SRC[name] = sources


def ev(name):
    return J(RES / f"eval_{name}.json")


def sub(e, s="full"):
    return e["test"][s]


def logits(run):
    return np.load(RUNS / run / "preds.npz", allow_pickle=True)["test_logit"]


from sift.data import load_split                       # noqa: E402
from evaluate import subset_masks                      # noqa: E402

te = load_split("test")
yt = (te.label_clean == "misaligned").values.astype(int)
masks = subset_masks(te)
CURVE = [("full", "Main ModernBERT-base", COL["main"]), ("minilm_ce", "MiniLM-L6 cross-encoder", COL["minilm"]),
         ("codebert", "CodeBERT cross-encoder", COL["codebert"]), ("bilstm", "Siamese BiLSTM", COL["bilstm"]),
         ("student", "Distilled student", COL["student"]), ("student_nodistill", "Student, no distillation", COL["nodistill"])]
BS = np.load(XR / "baseline_scores.npz", allow_pickle=True)

# ---------------------------------------------------------------- training curves
fig, ax = plt.subplots(1, 2, figsize=(W1, 2.2))
names = [("full", "Main", COL["main"]), ("abl_no_focal", "No focal", "#d98200"), ("abl_no_multitask", "Align. head only", "#2a9d8f"),
         ("abl_no_denoise", "No denoising", "#8172b3"), ("abl_no_consolidation", "Raw turns", "#c44e52")]
for r, lab, c in names:
    h = J(RUNS / r / "results.json")["history"]
    ep = [x["epoch"] for x in h]
    ax[0].plot(ep, [x["train_loss"] for x in h], "-o", ms=2, lw=0.9, color=c, label=lab)
    ax[1].plot(ep, [x["val_macro_f1"] for x in h], "-o", ms=2, lw=0.9, color=c)
ax[0].set(xlabel="epoch", ylabel="training loss", title="Training loss"); ax[1].set(xlabel="epoch", ylabel="validation macro-F1", title="Validation macro-F1")
ax[1].legend(*ax[0].get_legend_handles_labels(), frameon=False, loc="lower right", fontsize=5.8)
fig.tight_layout(w_pad=1.0)
save(fig, "fig_training_curves.png", ["experiments/runs/*/results.json (history)"])

# ---------------------------------------------------------------- NDCG@K
fig, ax = plt.subplots(figsize=(W1, 2.3))
ks = [1, 3, 5, 10]
for r, lab, c in CURVE:
    f = sub(ev(r)); ax.plot(ks, [f[f"ndcg@{k}"] for k in ks], "-o", ms=2.5, lw=1.0, color=c, label=lab)
b = J(XR / "baselines.json")
ax.axhline(0.7236, ls=":", color="grey", lw=0.8)
ax.text(10, 0.7236 + 0.004, "random order, NDCG@5 0.724", ha="right", fontsize=6, color="grey")
ax.set(xlabel="K", ylabel="NDCG@K", xticks=ks); ax.legend(frameon=False, loc="lower right", ncol=1)
save(fig, "fig_ndcg_vs_k.png", ["results/eval_*.json (test.full.ndcg@K)", "experiments/results/baselines.json"])

# ---------------------------------------------------------------- per-tier AUC bars
mc = {r["model"]: r for r in csv.DictReader(open(RES / "master_comparison.csv"))}
rows = [("B3 Hunk size", "B3\nhunk size"), ("B4 TF-IDF dissimilarity", "B4\nTF-IDF"), ("B5 Logistic regression", "B5\nlogistic"),
        ("Siamese BiLSTM", "BiLSTM"), ("MiniLM-L6 cross-encoder", "MiniLM-L6\ncross-enc."), ("CodeBERT cross-encoder", "CodeBERT\ncross-enc."),
        ("Main: ModernBERT-base", "Main"), ("Frozen-embedding frozen cosine dissimilarity", "Frozen\ncosine"),
        ("GAN-style baseline: gan hybrid with misaligned negatives", "GAN-style\nhybrid"), ("Distilled MiniLM student (fp32)", "Student")]
fig, ax = plt.subplots(figsize=(W2, 2.3))
x = np.arange(len(rows)); w = 0.38
ar = [float(mc[m]["auc_random"]) for m, _ in rows]; ah = [float(mc[m]["auc_hard"]) for m, _ in rows]
b1 = ax.bar(x - w / 2, ar, w, label="random tier", color="#9bb7d4"); b2 = ax.bar(x + w / 2, ah, w, label="hard tier", color=COL["main"])
for rect, v in list(zip(b1, ar)) + list(zip(b2, ah)):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=5.8)
ax.axhline(0.5, ls=":", color="grey", lw=0.8); ax.text(len(rows) - 0.5, 0.505, "chance", fontsize=6, color="grey", ha="right")
ax.set_xticks(x); ax.set_xticklabels([l for _, l in rows], rotation=0, fontsize=6.4); ax.set_ylim(0.4, 0.9)
ax.set_ylabel("within-session pairwise AUC"); ax.legend(frameon=False, ncol=2, loc="upper left")
save(fig, "fig_tier_auc.png", ["results/master_comparison.csv"])

# ---------------------------------------------------------------- PR curves and ROC per tier
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score  # noqa: E402
fig, ax = plt.subplots(figsize=(W1, 2.6))
for r, lab, c in CURVE:
    p, rc, _ = precision_recall_curve(yt, logits(r)); ax.plot(rc, p, color=c, lw=1.0, label=lab)
for k, c, lab in (("B4", COL["b4"], "B4 TF-IDF"), ("B5", COL["b5"], "B5 logistic")):
    p, rc, _ = precision_recall_curve(yt, BS[k]); ax.plot(rc, p, "--", color=c, lw=0.9, label=lab)
ax.axhline(yt.mean(), ls=":", color="grey", lw=0.8); ax.text(0.99, yt.mean() + 0.01, f"prevalence {yt.mean():.2f}", ha="right", fontsize=6, color="grey")
ax.set(xlabel="recall", ylabel="precision", ylim=(0, 1.02)); ax.legend(frameon=False, loc="upper right", fontsize=5.8)
save(fig, "fig_pr.png", ["experiments/runs/*/preds.npz", "experiments/results/baseline_scores.npz"])

fig, axs = plt.subplots(1, 3, figsize=(W2, 2.4), sharey=True)
for a, s, t in zip(axs, ("full", "tier_random", "tier_hard"), ("All test hunks", "Random tier", "Hard tier")):
    m = masks[s]
    for r, lab, c in CURVE[:4]:
        f, tpr, _ = roc_curve(yt[m], logits(r)[m]); a.plot(f, tpr, color=c, lw=1.0, label=lab)
    for k, c, lab in (("B4", COL["b4"], "B4 TF-IDF"), ("B5", COL["b5"], "B5 logistic")):
        f, tpr, _ = roc_curve(yt[m], BS[k][m]); a.plot(f, tpr, "--", color=c, lw=0.9, label=lab)
    a.plot([0, 1], [0, 1], ":", color="grey", lw=0.7); a.set(title=t, xlabel="false-positive rate")
axs[0].set_ylabel("true-positive rate"); axs[0].legend(frameon=False, loc="lower right", fontsize=5.8)
fig.tight_layout(w_pad=0.8)
save(fig, "fig_roc_tiers.png", ["experiments/runs/*/preds.npz", "experiments/results/baseline_scores.npz", "experiments/evaluate.py (subset_masks)"])

# ---------------------------------------------------------------- confusion matrix
f = sub(ev("full")); cm = np.array([[f["tn"], f["fp"]], [f["fn"], f["tp"]]])
fig, ax = plt.subplots(figsize=(W1 * 0.72, 2.2))
ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        v = cm[i, j] / cm[i].sum()
        ax.text(j, i, f"{cm[i, j]}\n({v:.0%})", ha="center", va="center", fontsize=7.5, color="white" if v > .6 else "black")
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["aligned", "misaligned"], yticklabels=["aligned", "misaligned"], xlabel="predicted", ylabel="true")
ax.spines[:].set_visible(False)
save(fig, "fig_confusion.png", ["results/eval_full.json (tn, fp, fn, tp)"])

# ---------------------------------------------------------------- reliability before/after
cal = J(RES / "calibration.json")
fig, axs = plt.subplots(1, 2, figsize=(W1, 2.2), sharey=True)
for a, key, t in zip(axs, ("reliability_before", "reliability_after"), (f"Before (ECE {cal['ece_test_before']:.3f})", f"After, T = {cal['temperature']:.3f} (ECE {cal['ece_test_after']:.3f})")):
    bins = [b for b in cal[key] if b["n"] > 0]
    a.plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
    a.plot([b["mean_conf"] for b in bins], [b["frac_pos"] for b in bins], "-o", ms=2.5, lw=1.0, color=COL["main"])
    a.set(title=t, xlabel="mean predicted P(misaligned)", xlim=(0, 1), ylim=(0, 1)); a.title.set_fontsize(6.8)
axs[0].set_ylabel("observed fraction misaligned")
fig.tight_layout(w_pad=0.6)
save(fig, "fig_reliability.png", ["results/calibration.json"])

# ---------------------------------------------------------------- noise robustness
nz = J(RES / "noise_robustness.json")["results"]
fig, axs = plt.subplots(1, 2, figsize=(W1, 2.2))
for a, met, lab in zip(axs, ("macro_f1", "ndcg@5"), ("macro-F1 (test)", "NDCG@5 (test)")):
    for arm, c, ls in (("no_denoise", "#c44e52", "-o"), ("denoise", COL["main"], "-s")):
        pts = sorted((r["noise_rate"], r["test"][met]) for r in nz if r["arm"] == arm)
        a.plot([p[0] * 100 for p in pts], [p[1] for p in pts], ls, ms=2.5, lw=1.0, color=c, label=arm.replace("_", " "))
    a.set(xlabel="injected label noise (%)", ylabel=lab, xticks=[0, 10, 20, 30])
axs[0].legend(frameon=False)
fig.tight_layout(w_pad=0.8)
save(fig, "fig_noise.png", ["results/noise_robustness.json"])

# ---------------------------------------------------------------- confident learning detection
cl = J(RES / "confident_learning.json")
meths = [("confident_learning", "confident\nlearning"), ("prune_by_noise_rate", "prune by\nnoise rate"), ("final drop list (CL + rules)", "final drop list\n(CL + rules)")]
fig, ax = plt.subplots(figsize=(W1, 2.2)); x = np.arange(len(meths)); w = 0.36
pr = [cl["detection"][m]["precision"] for m, _ in meths]; rc = [cl["detection"][m]["recall"] for m, _ in meths]
b1 = ax.bar(x - w / 2, pr, w, color=COL["main"], label="precision"); b2 = ax.bar(x + w / 2, rc, w, color="#d98200", label="recall")
for r_, v in list(zip(b1, pr)) + list(zip(b2, rc)):
    ax.text(r_.get_x() + r_.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.3)
ax.axhline(cl["random_precision"], ls=":", color="grey", lw=0.8); ax.text(2.45, cl["random_precision"] + 0.008, f"random precision {cl['random_precision']:.3f}", ha="right", fontsize=6, color="grey")
ax.set_xticks(x); ax.set_xticklabels([l for _, l in meths], fontsize=6.4); ax.set_ylim(0, 0.52); ax.set_ylabel("against the 569 known flips"); ax.legend(frameon=False, loc="upper right")
save(fig, "fig_cl_detection.png", ["results/confident_learning.json"])

# ---------------------------------------------------------------- k-means
km = J(RES / "kmeans.json")
from sift.kmeans import KMeansIndex                    # noqa: E402
from sklearn.manifold import TSNE                      # noqa: E402
idx = KMeansIndex.load(ROOT / "artifacts" / "kmeans")
Xte = np.load(XR / "pooled_embeddings.npz", allow_pickle=True)["test_emb"]
cid = idx.assign(Xte)[0]
ts = TSNE(2, perplexity=km["tsne"]["perplexity"], init="pca", random_state=473).fit_transform(Xte)
fig = plt.figure(figsize=(W2, 2.5)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.38)
a0, a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
k_ = km["k_scan"]["k"]
a0.plot(k_, km["k_scan"]["inertia"], "-o", ms=2.5, lw=1.0, color=COL["main"]); a0.set(xlabel="k", ylabel="inertia", title="Elbow")
a1.plot(k_, km["k_scan"]["silhouette"], "-o", ms=2.5, lw=1.0, color="#d98200"); a1.axvline(km["k_chosen"], ls=":", color="grey")
a1.set(xlabel="k", ylabel="silhouette", title=f"Silhouette (k = {km['k_chosen']} chosen)")
pal = ["#1b3a6b", "#d98200", "#2a9d8f", "#c44e52", "#8172b3"]
for c in range(km["k_chosen"]):
    m = cid == c; mis = yt[m].mean()
    a2.scatter(ts[m, 0], ts[m, 1], s=2, color=pal[c], alpha=0.6, label=f"cluster {c} (n={m.sum()}, {mis:.0%} misal.)", linewidths=0)
a2.set(xticks=[], yticks=[], title="t-SNE of test embeddings"); a2.spines[:].set_visible(False)
a2.legend(frameon=False, fontsize=5.4, markerscale=3, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
save(fig, "fig_kmeans.png", ["results/kmeans.json", "experiments/results/pooled_embeddings.npz", "artifacts/kmeans"])

# ---------------------------------------------------------------- latency vs size
lat = list(csv.DictReader(open(RES / "latency_size.csv")))
pts = [("Main ModernBERT (fp32, CPU)", "Main ModernBERT", COL["main"]), ("MiniLM student|ONNX fp32", "Student ONNX fp32", COL["minilm"]),
       ("MiniLM student|ONNX INT8", "Student ONNX INT8", COL["student"])]
fig, ax = plt.subplots(figsize=(W1, 2.3))
sel = [("Main ModernBERT", "PyTorch fp32 (CPU)", "Main, PyTorch fp32", COL["main"]), ("MiniLM student", "ONNX fp32", "Student, ONNX fp32", COL["minilm"]),
       ("MiniLM student", "ONNX INT8", "Student, ONNX INT8", COL["student"])]
for mname, fmt, lab, c in sel:
    r = next(x for x in lat if x["model"] == mname and x["format"] == fmt)
    s_, p50, p95 = float(r["size_mb"]), float(r["cpu_p50_ms"]), float(r["cpu_p95_ms"])
    ax.errorbar([s_], [p50], yerr=[[0], [p95 - p50]], fmt="o", color=c, ms=5, capsize=2.5, lw=0.9,
                label=f"{lab}: {s_:.1f} MB, p50 {p50:.1f} / p95 {p95:.1f} ms")
ax.axvline(30, ls=":", color="grey", lw=0.8); ax.axhline(100, ls=":", color="grey", lw=0.8)
ax.text(31, 112, "target < 30 MB", fontsize=5.8, color="grey"); ax.text(900, 102, "target p95 < 100 ms", fontsize=5.8, color="grey", ha="right")
ax.set(xscale="log", xlabel="model size (MB, log scale)", ylabel="CPU latency (ms), bar up to p95", ylim=(0, 135), xlim=(12, 900))
ax.set_xticks([20, 50, 100, 200, 500]); ax.set_xticklabels(["20", "50", "100", "200", "500"]); ax.minorticks_off()
ax.legend(frameon=False, loc="upper center", fontsize=5.8, bbox_to_anchor=(0.5, -0.22))
save(fig, "fig_latency_size.png", ["results/latency_size.csv", "results/latency_full.json", "results/latency_size_onnx.json"])

# ---------------------------------------------------------------- GAN: curves + versus main
gj = J(XR / "gan.json"); gv = J(RES / "gan_vs_main.json")
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
for key, lab, c in (("gan_unsupervised", "unsupervised", COL["gan"]), ("gan_hybrid_with_misaligned_negatives", "hybrid (with misaligned negatives)", COL["minilm"])):
    h = gj["variants"][key]["history"]; ep = [x["epoch"] for x in h]
    axs[0].plot(ep, [x["d_loss"] for x in h], color=c, lw=1.0, label=f"D loss, {lab}"); axs[0].plot(ep, [x["g_loss"] for x in h], "--", color=c, lw=0.9)
    axs[1].plot(ep, [x["val_auc"] for x in h], color=c, lw=1.0, label=lab)
    axs[2].plot(ep, [x["gen_diversity_cos"] for x in h], color=c, lw=1.0, label=lab)
axs[0].set(xlabel="epoch", ylabel="loss (solid D, dashed G)", title="Adversarial losses")
axs[1].set(xlabel="epoch", ylabel="validation ROC-AUC", title="Discriminator as scorer")
axs[2].set(xlabel="epoch", ylabel="mean pairwise cosine of G outputs", title="Generator collapse (1.0 = collapsed)"); axs[2].legend(frameon=False, fontsize=5.8, loc="upper right")
fig.tight_layout(w_pad=0.9)
save(fig, "fig_gan_curves.png", ["experiments/results/gan.json (history)"])

fig, ax = plt.subplots(figsize=(W1, 2.3))
order = [("full ModernBERT", "Main\nModernBERT", COL["main"]), ("frozen cosine dissimilarity", "Frozen\ncosine", COL["frozen"]),
         ("gan unsupervised", "GAN-style\nunsup.", COL["gan"]), ("gan hybrid with misaligned negatives", "GAN-style\nhybrid", COL["gan2"])]
mets = [("ndcg@5", "NDCG@5"), ("auc_random", "AUC random"), ("auc_hard", "AUC hard"), ("macro_f1", "macro-F1")]
w = 0.2; x = np.arange(len(mets))
for i, (k, lab, c) in enumerate(order):
    v = [gv["compared"][k][m] for m, _ in mets]
    ax.bar(x + (i - 1.5) * w, v, w, color=c, label=lab.replace("\n", " "))
ax.set_xticks(x); ax.set_xticklabels([l for _, l in mets]); ax.set_ylim(0.5, 0.92); ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=5.8)
ax.set_ylabel("test value")
save(fig, "fig_gan_vs_main.png", ["results/gan_vs_main.json"])

# ---------------------------------------------------------------- XAI faithfulness
xa = J(RES / "xai.json"); fa = xa["faithfulness"]
ks = sorted(int(k) for k in fa["ig"])
fig, ax = plt.subplots(figsize=(W1, 2.3))
ax.plot(ks, [fa["ig"][str(k)]["mean_drop_top"] for k in ks], "-o", ms=3, lw=1.1, color=COL["main"], label="mask top-k Integrated Gradients tokens")
ax.plot(ks, [fa["ixg"][str(k)]["mean_drop_top"] for k in ks], "-s", ms=3, lw=1.1, color="#2a9d8f", label="mask top-k input x gradient tokens")
ax.plot(ks, [fa["ig"][str(k)]["mean_drop_random"] for k in ks], "--^", ms=3, lw=1.0, color="#c44e52", label="mask k random tokens")
ax.set(xlabel="number of masked tokens k", ylabel="mean drop in P(misaligned)", xticks=ks); ax.legend(frameon=False, loc="lower right")
save(fig, "fig_xai_faithfulness.png", ["results/xai.json (faithfulness)"])

# ---------------------------------------------------------------- XAI worked example
case = J(RES / "xai_cases_ui.json")["collateral"]; cm_ = xa["cases"]["collateral"]
fig, ax = plt.subplots(figsize=(W2, 2.9)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
from matplotlib import colormaps                       # noqa: E402
cmap = colormaps["RdBu_r"]


def shade(s, vmax):
    return cmap(0.5 + 0.5 * float(np.clip(s / vmax, -1, 1)))


words = case["ui"]["intent_words"]; vmax = max(abs(w["score"]) for w in words)
ax.text(0.0, 0.97, f"Sample {cm_['sample_id']} ({cm_['repo']}), truth: {cm_['true']} / {cm_['drift_true']}; model P(misaligned) = {cm_['p']:.3f}", fontsize=7.3, weight="bold", va="top")
ax.text(0.0, 0.88, "Intent words (red pushes toward misaligned, blue toward aligned):", fontsize=6.6, va="top")
xx, yy = 0.0, 0.79
fig.canvas.draw(); rend = fig.canvas.get_renderer()
for w_ in words:
    t = ax.text(xx, yy, w_["word"], fontsize=7, va="center", color="white" if abs(w_["score"]) / vmax > 0.8 else "black", bbox=dict(boxstyle="round,pad=0.18", fc=shade(w_["score"], vmax), ec="none"))
    bb = t.get_window_extent(rend).transformed(ax.transData.inverted()); xx = bb.x1 + 0.012
    if xx > 0.95: xx, yy = 0.0, yy - 0.07; t.set_position((xx, yy)); bb = t.get_window_extent(rend).transformed(ax.transData.inverted()); xx = bb.x1 + 0.012
ax.text(0.0, 0.66, "Hunk lines with the largest attribution, in diff order (score = attribution summed over the line):", fontsize=6.6, va="top")
lines_all = case["ui"]["hunk_lines"]; lv = max(abs(l["score"]) for l in lines_all)
top = sorted(sorted(range(len(lines_all)), key=lambda i: -abs(lines_all[i]["score"]))[:9]); lines = [lines_all[i] for i in top]
for i, l in enumerate(lines[:12]):
    y = 0.58 - i * 0.047
    ax.add_patch(Rectangle((0.0, y - 0.021), 0.97, 0.043, fc=shade(l["score"], lv), ec="none", alpha=0.85))
    ax.text(0.005, y, (l["line"][:118]), fontsize=5.6, family="DejaVu Sans Mono", va="center")
    ax.text(0.995, y, f"{l['score']:+.3f}", fontsize=5.6, va="center", ha="right")
save(fig, "fig_xai_case.png", ["results/xai_cases_ui.json", "results/xai.json"])

# ---------------------------------------------------------------- ablation bars
ab = list(csv.DictReader(open(RES / "ablation.csv")))
sel = [r for r in ab]
short = {"1. Full model": "Full", "2. hunk-only": "Hunk only", "3. intent-only": "Intent only", "4. no intent consolidation": "Raw turns",
         "5. no focal loss (CE)": "No focal", "6. alignment head only": "Align. head only", "7. no denoising": "No denoising",
         "9a. k-means features OFF (Platt on logit)": "k-means off", "9b. k-means features ON (hybrid)": "k-means on",
         "10. Distilled MiniLM student (fp32)": "Student fp32", "11. INT8 ONNX student (deployed target)": "Student INT8"}
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5), sharey=True)
labels = [short[r["model"]] for r in sel]
for a, (m, t) in zip(axs, (("macro_f1", "macro-F1"), ("ndcg@5", "NDCG@5"), ("auc_hard", "hard-tier pairwise AUC"))):
    v = [float(r[m]) if r[m] else np.nan for r in sel]
    cols = [COL["main"] if l == "Full" else "#9bb7d4" for l in labels]
    a.barh(range(len(v)), v, color=cols); a.set_xlim(0.4 if m != "auc_hard" else 0.45, max(v_ for v_ in v if v_ == v_) + 0.07)
    for i, y_ in enumerate(v):
        a.text(y_ + 0.004 if y_ == y_ else 0.41, i, f"{y_:.3f}" if y_ == y_ else "n/a", va="center", fontsize=5.8)
    a.set_title(t); a.invert_yaxis()
axs[0].set_yticks(range(len(labels))); axs[0].set_yticklabels(labels)
fig.tight_layout(w_pad=0.8)
save(fig, "fig_ablation.png", ["results/ablation.csv"])

# ---------------------------------------------------------------- testing figures
ts_ = J(RES / "test_summary.json"); cov = J(RES / "coverage.json"); tm = J(XR / "testing_measurements.json")
CAT = [("smoke", "Smoke"), ("white-box", "White-box"), ("black-box", "Black-box"), ("data-pipeline", "Data pipeline"), ("metamorphic", "Metamorphic"),
       ("adversarial", "Adversarial"), ("back-to-back", "Back-to-back"), ("api", "API"), ("e2e", "UI end-to-end")]
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1, 1.2]})
cs = [c for c in CAT if c[0] in ts_["categories"]][::-1]
ps = [ts_["categories"][c]["passed"] for c, _ in cs]; xf = [ts_["categories"][c]["xfailed"] for c, _ in cs]
axs[0].barh([l for _, l in cs], ps, color=COL["main"], label="passed"); axs[0].barh([l for _, l in cs], xf, left=ps, color="#d98200", label="xfailed (known defect)")
for i, (p_, x_) in enumerate(zip(ps, xf)): axs[0].text(p_ + x_ + 0.8, i, str(p_ + x_), va="center", fontsize=6.3)
axs[0].set_title(f"Tests per category ({ts_['totals']['total']} total, {ts_['totals']['failed']} failed)"); axs[0].legend(frameon=False, loc="center right", fontsize=6)
files = sorted(((k, v["summary"]) for k, v in cov["files"].items() if v["summary"]["num_statements"] > 0), key=lambda kv: kv[1]["percent_covered"])
nm = [k.replace("src/sift/", "").replace("app/backend/", "api/") for k, _ in files]; pc = [s["percent_covered"] for _, s in files]
axs[1].barh(nm, pc, color=["#2a9d8f" if p >= 70 else "#d98200" if p > 0 else "#c44e52" for p in pc]); axs[1].set_xlim(0, 105)
axs[1].axvline(ts_["coverage"]["percent_covered"], ls="--", color="#333", lw=0.8); axs[1].text(ts_["coverage"]["percent_covered"] + 1, 0.1, f"total {ts_['coverage']['percent_covered']}%", fontsize=6.3)
axs[1].set_title("Line + branch coverage per module (%)"); axs[1].tick_params(axis="y", labelsize=5.8)
fig.tight_layout(w_pad=1.0)
save(fig, "fig_testing_summary.png", ["results/test_summary.json", "results/coverage.json"])

fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1.15, 1]})
mr = [("mr_identifier_rename", "identifier rename", 0.10), ("mr_context_comment", "comment added", 0.10), ("mr_context_whitespace", "whitespace change", 0.10),
      ("mr_chitchat_removal", "chit-chat removed", 0.05)]
v = [tm[k]["flip_rate"] * 100 for k, _, _ in mr]; tol = [t * 100 for _, _, t in mr]
axs[0].barh([l for _, l, _ in mr], v, color="#2a9d8f")
for i, (a_, t_) in enumerate(zip(v, tol)):
    axs[0].plot([t_, t_], [i - 0.4, i + 0.4], color="#c44e52", lw=1.2); axs[0].text(a_ + 0.2, i, f"{a_:.1f}% (n = {tm[mr[i][0]]['n']})", va="center", fontsize=6.3)
axs[0].set_xlim(0, 12.5); axs[0].invert_yaxis(); axs[0].set_xlabel("verdict flip rate (%), red line = tolerance"); axs[0].set_title("Metamorphic relations")
at = [("attack_misleading_comment_intent_copy", "intent copied as\ncomment (D3)"), ("attack_context_comment_claims_alignment", "'implements: intent'\ncomment (D4)"),
      ("attack_prompt_injection_string", "prompt-injection\nstring"), ("attack_false_alarm_irrelevant_comment", "irrelevant comment\n(false alarm)")]
v = [tm[k]["success_rate"] * 100 for k, _ in at]
axs[1].barh([l for _, l in at], v, color=["#c44e52", "#c44e52", "#d98200", "#d98200"])
for i, a_ in enumerate(v): axs[1].text(a_ + 1, i, f"{a_:.0f}% ({round(a_ / 100 * tm[at[i][0]]['attempts'])}/{tm[at[i][0]]['attempts']})", va="center", fontsize=6.3)
axs[1].axvline(20, ls=":", color="grey"); axs[1].text(20.8, 3.35, "20% threshold", fontsize=5.8, color="grey")
axs[1].set_xlim(0, 100); axs[1].invert_yaxis(); axs[1].set_xlabel("attack success rate (%)"); axs[1].set_title("Red-team attacks"); axs[1].tick_params(axis="y", labelsize=6)
fig.tight_layout(w_pad=1.0)
save(fig, "fig_redteam_mr.png", ["experiments/results/testing_measurements.json"])

# ---------------------------------------------------------------- architecture
fig, ax = plt.subplots(figsize=(W2, 4.9)); ax.set_xlim(-1, 101); ax.set_ylim(0, 70); ax.axis("off")


def box(x, y, w, h, t, fc="#e8eef7", ec="#1b3a6b", fs=6.0, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", color=tc, linespacing=1.18)


def arr(x1, y1, x2, y2, c="#1b3a6b"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, color=c, lw=0.9, shrinkA=0, shrinkB=0))


def line(xs, ys, c="#888", ls="-", lw=0.8):
    ax.plot(xs, ys, color=c, lw=lw, ls=ls, solid_capstyle="butt")


ax.text(0, 68.6, "A. Inference pipeline (per coding session)", fontsize=7.2, weight="bold", va="center")
bw, gp, y0, h0 = 12.2, 2.4, 52, 12
xs = [i * (bw + gp) for i in range(7)]
spec = [("Session capture\nchat turns +\nunified diff", "#f4f4f4", "#555", False), ("Intent\nconsolidation\n(drop chit-chat,\nretractions, repeats)", "#e8eef7", "#1b3a6b", False),
        ("Hunk\nsegmentation\n(diff parser)", "#e8eef7", "#1b3a6b", False),
        ("ModernBERT-base\ncross-encoder\n3 heads:\nalignment, drift\ntype, confidence", "#1b3a6b", "#1b3a6b", True),
        ("Temperature\ncalibration\n(T fitted on\nvalidation)", "#e8eef7", "#1b3a6b", False), ("Ranking and\nthreshold\n(riskiest hunk\nfirst)", "#e8eef7", "#1b3a6b", False),
        ("Web UI + API\nFastAPI, React\nhighlights, review\nactions", "#e3f3ef", "#2a9d8f", False)]
for x, (t, fc, ec, b_) in zip(xs, spec):
    box(x, y0, bw, h0, t, fc=fc, ec=ec, bold=b_, tc="white" if b_ else "black", fs=5.7)
for i in range(6): arr(xs[i] + bw + 0.2, y0 + h0 / 2, xs[i + 1] - 0.2, y0 + h0 / 2)
ax.text(0, 47.6, "B. Components attached to the trained model", fontsize=7.2, weight="bold", va="center")
cx = xs[3] + bw / 2
arr(cx, y0 - 0.2, cx, 44.2, c="#1b3a6b"); line([12, 88], [44.2, 44.2], c="#1b3a6b", lw=1.0)
bwb, gb = 23.5, 2.0
bx = [i * (bwb + gb) for i in range(4)]
spB = [("Noise analysis and denoising\nrule filter (non-semantic, duplicate);\nconfident learning on out-of-fold\nprobabilities, scored on 569 flips;\ncleans the training rows", "#fdf0dc", "#d98200"),
       ("K-means (k = 4)\non pooled [CLS] embeddings;\ncluster id, similar flagged hunks,\nhybrid cluster features", "#ece7f6", "#8172b3"),
       ("Explainability\nIntegrated Gradients on tokens,\naggregated to words and lines;\nfaithfulness (masking) test", "#fbe6e7", "#c44e52"),
       ("Compression\ndistilled MiniLM-L6 student\n(tau = 4, alpha = 0.5), then\nINT8 ONNX export", "#e3f3ef", "#2a9d8f")]
for x, (t, fc, ec) in zip(bx, spB):
    box(x, 31, bwb, 11.5, t, fc=fc, ec=ec, fs=5.6); line([x + bwb / 2] * 2, [44.2, 42.7], c=ec, lw=1.0)
ax.text(0, 27.4, "C. Comparison and verification, on the same splits and metrics", fontsize=7.2, weight="bold", va="center")
bwc, gc = 32, 2
cxs = [i * (bwc + gc) for i in range(3)]
spC = [("Baselines\nB1 random, B2 diff lines, B3 hunk size,\nB4 TF-IDF, B5 logistic regression;\nSiamese BiLSTM, MiniLM-L6 and\nCodeBERT cross-encoders", "#f4f4f4", "#555"),
       ("GAN-style baseline\non frozen MiniLM embeddings:\ngenerator makes aligned-hunk embeddings;\ndiscriminator is the alignment scorer", "#ece7f6", "#8172b3"),
       ("Evaluation and testing\nper tier, language, repository; ablations;\nmetamorphic, red-team, back-to-back,\nleakage tests; defect register", "#f4f4f4", "#555")]
for x, (t, fc, ec) in zip(cxs, spC): box(x, 13.5, bwc, 11.5, t, fc=fc, ec=ec, fs=5.6)
box(0, 1, 100, 8, "Data: Sift-IA v1.0, 18,000 hunk-intent pairs from 15 open-source repositories, repository-disjoint train / validation / test splits,\nconstructed misaligned hunks, five noise types recorded with ground truth", fc="#f9f9f9", ec="#999", fs=5.9)
arr(50, 9.3, 50, 12.6, c="#888")
save(fig, "fig_architecture.png", ["structure drawn from src/sift and results/*.json (no numbers except those in results)"])

# ---------------------------------------------------------------- UI screenshots (real, cropped)
from PIL import Image                                  # noqa: E402
SS = ROOT / "docs" / "screenshots"
Image.open(SS / "landing_desktop_light.png").crop((0, 0, 1440, 900)).save(OUT / "ui_landing.png")
Image.open(ROOT / "reports" / "screenshots" / "review_explained_desktop_light.png").save(OUT / "ui_review.png")
Image.open(SS / "dev_results_desktop_light.png").crop((224, 0, 1440, 960)).save(OUT / "ui_results.png")
JSON_SRC["ui_landing.png"] = ["docs/screenshots/landing_desktop_light.png (cropped)"]
JSON_SRC["ui_review.png"] = ["reports/screenshots/review_explained_desktop_light.png (copy of a real UI capture, analysed sample with explanation)"]
JSON_SRC["ui_results.png"] = ["docs/screenshots/dev_results_desktop_light.png (cropped)"]
json.dump(JSON_SRC, open(OUT / "figure_sources.json", "w"), indent=1)
print("figures:", len(JSON_SRC))
