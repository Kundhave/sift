"""Deliverable 1: Integrated Gradients / Input x Gradient on the trained cross-encoder; faithfulness test; worked cases; latency."""
import json, sys, time, textwrap
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sift.data import ROOT, load_split
from sift.infer import SiftScorer
from sift.explain import Explainer

N_FAITH = int(sys.argv[1]) if len(sys.argv) > 1 else 100
STEPS = 24
sc = SiftScorer(ROOT / "artifacts/full")
ex = Explainer(sc.model, sc.tok, sc.device, sc.max_len)
te = load_split("test")
txt = lambda r: f"{r.file_path}\n{r.hunk}"
# scores for the whole test split (for picking samples)
a, d, c = sc._logits(list(te.consolidated_intent), [txt(r) for r in te.itertuples()])
te["p"] = 1 / (1 + np.exp(-a / sc.T)); te["pred"] = te.p >= sc.threshold
res = {"n_steps": STEPS, "baseline": "[PAD] embedding at all non-special positions", "target": "sigmoid(align_logit) = P(misaligned)"}

# ---- faithfulness (deletion): hunks predicted misaligned
pool = te[te.pred].sample(min(N_FAITH, int(te.pred.sum())), random_state=473)
rng = np.random.RandomState(473)
rows = {m: {k: [] for k in (3, 5, 10, 20)} for m in ("ig", "ixg")}; rnd = {k: [] for k in (3, 5, 10, 20)}; p0s = []
t0 = time.time()
for r in pool.itertuples():
    for m in ("ig", "ixg"):
        res_ = ex.attribute(r.consolidated_intent, txt(r), method=m, n_steps=STEPS)
        for k in rows[m]:
            f = ex.faithfulness(res_, k, rng)
            rows[m][k].append((f["p0"], f["p_top"], f["p_rand"]))
            if m == "ig": rnd[k].append(f["p_rand"])
    p0s.append(res_["prob"])
res["faithfulness"] = {"n": len(pool), "threshold": sc.threshold, "frac_true_misaligned": float((pool.label_clean == "misaligned").mean()), "seconds": time.time() - t0}
for m in rows:
    res["faithfulness"][m] = {}
    for k, v in rows[m].items():
        v = np.array(v)
        res["faithfulness"][m][str(k)] = {"mean_p_orig": float(v[:, 0].mean()), "mean_p_after_top_k_masked": float(v[:, 1].mean()),
                                         "mean_p_after_random_k_masked": float(v[:, 2].mean()),
                                         "mean_drop_top": float((v[:, 0] - v[:, 1]).mean()), "mean_drop_random": float((v[:, 0] - v[:, 2]).mean()),
                                         "verdict_flip_rate_top": float(((v[:, 0] >= sc.threshold) & (v[:, 1] < sc.threshold)).mean()),
                                         "verdict_flip_rate_random": float(((v[:, 0] >= sc.threshold) & (v[:, 2] < sc.threshold)).mean()),
                                         "frac_top_beats_random": float(((v[:, 0] - v[:, 1]) > (v[:, 0] - v[:, 2])).mean())}
# ---- attribution mass split between intent and hunk
split = []
for r in pool.head(40).itertuples():
    q = ex.attribute(r.consolidated_intent, txt(r), n_steps=STEPS)
    seg = np.array([-1 if s is None else s for s in q["seq"]]); at = np.abs(q["attr"])
    split.append(at[seg == 0].sum() / max(1e-9, at[seg >= 0].sum()))
res["abs_attribution_share_on_intent_tokens_mean"] = float(np.mean(split))

# ---- worked cases: one per class from the TEST split (most confident correct, short hunk)
cases = {}
def pick(mask, ascending_p):
    s = te[mask & (te.hunk.str.count("\n") <= 22)]
    return s.sort_values("p", ascending=ascending_p).iloc[0]
cand = {"aligned": pick((te.label_clean == "aligned") & ~te.pred, True)}
for dt in ("collateral", "scope_expansion", "purpose_drift"):
    cand[dt] = pick((te.drift_type_clean == dt) & te.pred, False)

def render(name, r):
    q = ex.attribute(r.consolidated_intent, txt(r), n_steps=STEPS)
    words = Explainer.to_words(q, r.consolidated_intent, txt(r), 0)
    lines = Explainer.to_lines(q, txt(r))
    m = max([abs(w[1]) for w in words] + [abs(l[1]) for l in lines] + [1e-9])
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.17 * (len(lines) + 4)))
    ax.axis("off"); y = 1.0; dy = 1.0 / (len(lines) + 4)
    ax.text(0, y, f"[{name}] {r.repo}  p(misaligned)={r.p:.3f}  true={r.label_clean}/{r.drift_type_clean}", fontsize=8, weight="bold", va="top", transform=ax.transAxes)
    y -= dy
    ax.text(0, y, "INTENT: " + textwrap.shorten(r.consolidated_intent, 110), fontsize=7, va="top", transform=ax.transAxes)
    y -= dy
    # intent words as coloured chips (top 8 by |attr|)
    top = sorted(words, key=lambda w: -abs(w[1]))[:8]
    ax.text(0, y, "top intent tokens: " + ", ".join(f"{w[0]}({w[1]:+.2f})" for w in top), fontsize=6.5, va="top", transform=ax.transAxes); y -= dy
    for ln, v in lines:
        col = (0.75, 0.22, 0.17, min(1, abs(v) / m) * 0.65) if v > 0 else (0.18, 0.55, 0.34, min(1, abs(v) / m) * 0.45)
        ax.add_patch(plt.Rectangle((0, y - dy * 0.9), 1, dy * 0.95, color=col, transform=ax.transAxes, lw=0))
        ax.text(0.005, y, ln[:105], fontsize=6, family="monospace", va="top", transform=ax.transAxes); y -= dy
    fig.savefig(ROOT / f"reports/figures/xai_case_{name}.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    return {"sample_id": r.sample_id, "repo": r.repo, "p": float(r.p), "true": r.label_clean, "drift_true": r.drift_type_clean,
            "intent": r.consolidated_intent, "top_intent_tokens": [(w[0], w[1]) for w in top[:6]],
            "top_hunk_lines": [(l[0][:90], l[1]) for l in sorted(lines, key=lambda l: -l[1])[:3]]}
for k, r in cand.items():
    cases[k] = render(k, r)
res["cases"] = cases

# ---- explanation latency (single hunk), quiet-machine measurement is done again by experiments/bench_latency.py
lat_s, lat_e = [], []
for r in te.sample(25, random_state=1).itertuples():
    t = time.perf_counter(); sc._logits([r.consolidated_intent], [txt(r)]); torch.cuda.synchronize(); lat_s.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter(); ex.attribute(r.consolidated_intent, txt(r), n_steps=STEPS); torch.cuda.synchronize(); lat_e.append((time.perf_counter() - t) * 1000)
res["latency_ms"] = {"score_p50": float(np.percentile(lat_s[3:], 50)), "explain_ig_p50": float(np.percentile(lat_e[3:], 50)),
                     "explain_ig_p95": float(np.percentile(lat_e[3:], 95)), "device": str(sc.device), "gpu": torch.cuda.get_device_name(0) if sc.device == "cuda" else None,
                     "note": "may be inflated if other jobs shared the GPU"}
json.dump(res, open(ROOT / "experiments/results/xai.json", "w"), indent=2, default=float)
print(json.dumps(res["faithfulness"]["ig"], indent=1)); print(res["latency_ms"])
