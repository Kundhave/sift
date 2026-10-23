"""Plots from existing measured JSON: confident-learning detection vs the 569 known flips; GAN baseline vs main model."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sift.data import ROOT
R = ROOT / "experiments/results"
d = json.load(open(R / "denoise.json"))
rows = {k: v for k, v in d["confident_learning"].items()}
rows["final drop list (CL + rules)"] = d["drop_list_vs_flips"]
n_tr, flips = d["n_train"], 569
rand_prec = flips / n_tr
fig, ax = plt.subplots(figsize=(7.5, 3.8)); x = np.arange(len(rows)); w = .27
for i, (m, c) in enumerate((("precision", "#d1495b"), ("recall", "#00798c"), ("f1", "#edae49"))):
    ax.bar(x + (i - 1) * w, [v[m] for v in rows.values()], w, label=m, color=c)
ax.axhline(rand_prec, ls="--", c="gray", lw=1, label=f"random precision ({rand_prec:.3f})")
ax.set_xticks(x); ax.set_xticklabels([k.replace("_", "\n") for k in rows], fontsize=7); ax.legend(fontsize=7); ax.grid(axis="y", alpha=.3)
ax.set_title(f"Confident learning vs {flips} known label flips (OOF 3-fold; flagged counts: " + ", ".join(str(v["flagged"]) for v in rows.values()) + ")", fontsize=8)
fig.tight_layout(); fig.savefig(ROOT / "results/confident_learning_detection.png", dpi=160); plt.close(fig)
json.dump({"source": "experiments/results/denoise.json", "n_train": n_tr, "true_flips": flips, "random_precision": rand_prec, "detection": rows,
           "oof_auc_vs_clean": d["oof_auc_vs_clean"], "rules": d["rules"], "note": d["note"]}, open(ROOT / "results/confident_learning.json", "w"), indent=2)
# ---- GAN vs main model
g = json.load(open(R / "gan.json")); full = json.load(open(ROOT / "results/eval_full.json"))["test"]
full = full.get("full", full) if isinstance(full, dict) and "macro_f1" not in full else full
models = {"full ModernBERT": full}
for k, v in g["variants"].items(): models[k.replace("_", " ")] = v["test"]
names = list(models); mets = [("macro_f1", "macro-F1"), ("ndcg@5", "NDCG@5"), ("auc_random", "AUC random"), ("auc_hard", "AUC hard")]
get = lambda t, m: t["pairwise_auc"]["random"]["auc"] if m == "auc_random" else t["pairwise_auc"]["hard"]["auc"] if m == "auc_hard" else t[m]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]}); w = .8 / len(names)
for i, n in enumerate(names):
    ax[0].bar(np.arange(4) + i * w, [get(models[n], m) for m, _ in mets], w, label=n)
ax[0].set_xticks(np.arange(4) + .4 - w / 2); ax[0].set_xticklabels([l for _, l in mets]); ax[0].legend(fontsize=6); ax[0].set_ylim(0.4, 1); ax[0].grid(axis="y", alpha=.3)
ax[0].set_title("GAN-style baseline vs main model (test)", fontsize=9)
for k in ("gan_unsupervised", "gan_hybrid_with_misaligned_negatives"):
    h = g["variants"][k]["history"]; e = [r["epoch"] for r in h]
    ax[1].plot(e, [r["d_loss"] for r in h], label=f"D loss ({k[4:11]})"); ax[1].plot(e, [r["gen_diversity_cos"] for r in h], "--", label=f"G diversity cos ({k[4:11]})")
ax[1].set_title("Training curves (D loss, generator pairwise cosine)", fontsize=8); ax[1].legend(fontsize=6); ax[1].set_xlabel("epoch"); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(ROOT / "results/gan_vs_main.png", dpi=160)
json.dump({"source": ["experiments/results/gan.json", "results/eval_full.json"], "compared": {n: {m: get(models[n], m) for m, _ in mets} for n in names},
           "gan_unsupervised_min_diversity": g["variants"]["gan_unsupervised"]["min_gen_diversity"], "gan_unsupervised_final_diversity": g["variants"]["gan_unsupervised"]["final_gen_diversity"]}, open(ROOT / "results/gan_vs_main.json", "w"), indent=2)
print({n: {m: round(get(models[n], m), 3) for m, _ in mets} for n in names})
