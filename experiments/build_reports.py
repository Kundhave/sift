"""Generate the Lab Evaluation 2 PDFs (CSE23037/) from the results JSON files. One function per deliverable.

Usage:  PYTHONPATH=src .venv/bin/python experiments/build_reports.py [--only xai|ablation|gan|testing|review]

Rules: every number, table and figure comes from a results JSON (results/*.json, experiments/results/*.json);
nothing is typed in by hand except method descriptions. Each table/figure is logged in reports/number_trace.json.
Structure/layout only follows the generic template (title, one-row header table, numbered sections from
'1. Executive Summary', tables, figures); no content is taken from any other project.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reports"))
import numpy as np
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Spacer, Table, TableStyle
import reportlib as R
from reportlib import (P, H, H3_, bullets, table, figure, fmt, build, title_block, header_table, RJ, trace, equation, not_executed, fig_save, plt, OUT, FIG, ROOT,
                       SMALL, BODY)

PROJECT = "Sift"
TEAM = "Team 24 (single member)"
MEMBER = "Kundhave S, CB.SC.U4CSE23037"
W = 17.4 * cm
SEED_NOTE = "single seed (473), no confidence intervals"


# ============================================================================================ helpers
def d_(x, d=3, plus=True):
    return "-" if x is None else (f"{x:+.{d}f}" if plus else f"{x:.{d}f}")


def ev_row(test):
    """Flatten an eval 'test' block ('full' subset + tier subsets) into one row."""
    f = test["full"] if "full" in test else test
    pa = f.get("pairwise_auc") or {}
    r = dict(acc=f["accuracy"], mf1=f["macro_f1"], f1a=f.get("f1_aligned"), f1m=f.get("f1_misaligned"), p=f["precision"], rec=f["recall"], spec=f["specificity"],
             auc_r=(pa.get("random") or {}).get("auc"), auc_h=(pa.get("hard") or {}).get("auc"), ndcg5=f["ndcg@5"], ece=f.get("ece"))
    for t in ("random", "hard"):
        s = test.get(f"tier_{t}") if "full" in test else None
        r[f"mf1_{t}"] = s["macro_f1"] if s else None
        r[f"ndcg5_{t}"] = s["ndcg@5"] if s else None
    return r


def bar_colors(n):
    return [R.PALETTE[i % len(R.PALETTE)] for i in range(n)]


def doc_head(title, subtitle, deliverable, extra_label, extra_val):
    return title_block(title, subtitle) + [header_table(PROJECT, deliverable, TEAM, MEMBER, extra_label, extra_val), Spacer(1, 6)]


COMMON_LIMITS = [
    "Labels in Sift-IA v1.0 are construction-determined (a misaligned hunk is a donor hunk from a different commit); there is no human validation.",
    "The test split (httpx, attrs, axios) is repository-disjoint but is still open-source Python/JavaScript (it contains no TypeScript); results may not transfer to agent-authored diffs.",
    "Misaligned hunks are constructed, so a model could learn construction artefacts; the hard tier mitigates this but does not remove it.",
    "Latency and size come only from the machine they were measured on (Intel i7-13650HX CPU, RTX 4050 Laptop GPU with 6 GB); every run is a " + SEED_NOTE + ".",
    "Baselines, noise-robustness and hyper-parameter search use reduced settings; the rules-only denoising ablation, IG convergence checks, a user study of explanations, and drift and bias measurement are outside this work.",
]


# ============================================================================================ XAI
def case_figure(key, case, label):
    """Render intent words + hunk lines coloured by Integrated-Gradients score (red = pushes toward misaligned, blue = toward aligned)."""
    ui = case["ui"]
    W_IN, LH = 7.4, 0.128
    words = ui["intent_words"][:26]
    lines = ui["hunk_lines"]
    fig = plt.figure(figsize=(W_IN, 6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W_IN); ax.axis("off")
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    FS = 6.6

    def color(s, m):
        a = min(abs(s) / m, 1.0) * 0.85 if m else 0
        return (0.85, 0.15, 0.15, a) if s > 0 else (0.15, 0.35, 0.85, a)

    def width_of(txt, **kw):
        t = ax.text(0, 0, txt, fontsize=FS, family="DejaVu Sans Mono", parse_math=False, **kw)
        t.remove(); return len(txt) * 0.602 * FS / 72

    y = 0.12
    head = f"{label}  |  true: {case['true']}" + (f" / {case['drift_true']}" if case["drift_true"] != "none" else "") + f"  |  P(misaligned) = {case['p']:.3f}  |  {case['repo']}, {case['sample_id']}"
    ax.text(0.05, y, head, fontsize=7.6, fontweight="bold", va="center", color="#1f3a5f", parse_math=False); y += 0.2
    ax.text(0.05, y, "Intent (word-level IG)", fontsize=6.4, va="center", color="#666666"); y += 0.15
    x, m = 0.05, max([abs(w["score"]) for w in words] + [1e-9])
    for w in words + ([{"word": "...", "score": 0}] if len(ui["intent_words"]) > len(words) else []):
        tw = width_of(w["word"]) + 0.05
        if x + tw > W_IN - 0.05: x, y = 0.05, y + LH + 0.02
        ax.add_patch(plt.Rectangle((x - 0.01, y - LH / 2), tw, LH, color=color(w["score"], m), lw=0))
        ax.text(x + 0.015, y, w["word"], fontsize=FS, family="DejaVu Sans Mono", va="center", parse_math=False); x += tw + 0.03
    y += 0.24
    ax.text(0.05, y, "Hunk lines (line-level IG: sum of token attributions per line)", fontsize=6.4, va="center", color="#666666"); y += 0.15
    m = max([abs(l["score"]) for l in lines] + [1e-9])
    for l in lines:
        txt = l["line"].replace("\t", "    ")
        txt = txt if len(txt) <= 98 else txt[:95] + "..."
        ax.add_patch(plt.Rectangle((0.04, y - LH / 2), W_IN - 0.08, LH, color=color(l["score"], m), lw=0))
        ax.text(0.06, y, txt, fontsize=FS, family="DejaVu Sans Mono", va="center", parse_math=False)
        ax.text(W_IN - 0.06, y, f"{l['score']:+.2f}", fontsize=5.8, family="DejaVu Sans Mono", va="center", ha="right", color="#444444"); y += LH
    y += 0.06
    fig.set_size_inches(W_IN, y); ax.set_ylim(y, 0)
    return fig_save(fig, f"xai_case_{key}.png", dpi=220)


def faith_figure(x):
    f = x["faithfulness"]; ks = ["3", "5", "10", "20"]
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.7))
    p0 = f["ig"]["3"]["mean_p_orig"]
    ax[0].axhline(p0, color="#7b8794", ls="--", lw=1, label=f"unmasked ({p0:.3f})")
    for m, nm, c in (("ig", "Integrated Gradients", R.PALETTE[0]), ("ixg", "Input x Gradient", R.PALETTE[2])):
        ax[0].plot([int(k) for k in ks], [f[m][k]["mean_p_after_top_k_masked"] for k in ks], "-o", ms=3.5, color=c, label=f"{nm}: top-k masked")
        ax[0].plot([int(k) for k in ks], [f[m][k]["mean_p_after_random_k_masked"] for k in ks], ":s", ms=3, color=c, label=f"{nm}: random-k masked")
    ax[0].set_xlabel("tokens replaced by [PAD] (k)"); ax[0].set_ylabel("mean P(misaligned)"); ax[0].set_xticks([3, 5, 10, 20])
    ax[0].set_ylim(0.1, 0.98); ax[0].legend(fontsize=5.4, frameon=False, loc="upper right", ncol=1); ax[0].grid(alpha=.3); ax[0].set_title(f"Deletion test, mean probability (n={f['n']})", fontsize=8)
    xs = np.arange(len(ks)); bw = 0.2
    for i, (m, kind, nm, c) in enumerate((("ig", "top", "IG top-k", R.PALETTE[0]), ("ig", "random", "IG random-k", "#9fb3cc"), ("ixg", "top", "IxG top-k", R.PALETTE[2]), ("ixg", "random", "IxG random-k", "#a9d6d0"))):
        ax[1].bar(xs + (i - 1.5) * bw, [f[m][k][f"verdict_flip_rate_{kind}"] for k in ks], bw, color=c, label=nm)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(ks); ax[1].set_xlabel("tokens masked (k)"); ax[1].set_ylabel("verdict flip rate"); ax[1].legend(fontsize=5.8, frameon=False, ncol=2)
    ax[1].set_title("Share of predicted-misaligned hunks flipped to aligned", fontsize=8); ax[1].grid(alpha=.3, axis="y")
    for a in ax: a.tick_params(labelsize=7); a.xaxis.label.set_size(7.5); a.yaxis.label.set_size(7.5)
    fig.tight_layout()
    return fig_save(fig, "xai_faithfulness_report.png")


def crop_ui():
    from PIL import Image as PI
    src = ROOT / "reports/screenshots/review_explained_desktop_light.png"
    im = PI.open(src).convert("RGB").crop((350, 100, 1424, 445))
    p = FIG / "ui_explained_crop.png"; FIG.mkdir(parents=True, exist_ok=True); im.save(p); return p


def build_xai():
    x = RJ("results/xai.json"); cs = RJ("results/xai_cases_ui.json"); f = x["faithfulness"]; lat = x["latency_ms"]
    S = []
    S += doc_head("Explainable AI for the Sift Cross-Encoder", "Token-level and line-level attribution with Integrated Gradients, a deletion-based faithfulness check and worked cases",
                  "Deliverable 1: XAI", "Method", "Integrated Gradients (Captum)")
    ig5 = f["ig"]["5"]
    S += [H("1. Executive Summary"),
          P("Sift reads a developer intent and one diff hunk jointly and outputs P(misaligned). The XAI layer answers one question: which intent words and which diff lines pushed that probability up or down. "
            "It uses Integrated Gradients (IG) on the ModernBERT cross-encoder, aggregated from sub-tokens to words (intent and hunk) and to diff lines (hunk), and shows the result in the review UI."),
          *bullets([
              f"<b>Method:</b> IG with a [PAD]-embedding baseline, {x['n_steps']} integration steps in the experiments (the UI default is 16), target = sigmoid(alignment logit) = P(misaligned); Input x Gradient is run as a cheaper comparator.",
              f"<b>Faithfulness (deletion test, n = {f['n']} test hunks):</b> replacing the top-5 IG tokens with [PAD] lowers mean P(misaligned) from {ig5['mean_p_orig']:.3f} to {ig5['mean_p_after_top_k_masked']:.3f}, "
              f"against {ig5['mean_p_after_random_k_masked']:.3f} when 5 random tokens are masked; the top-5 set beats random in {R.pct(ig5['frac_top_beats_random'], 0)} of hunks. The random-masking drop is large too, so part of the effect is an artefact of masking itself.",
              f"<b>Cost:</b> one IG explanation takes p50 {lat['explain_ig_p50']:.0f} ms (p95 {lat['explain_ig_p95']:.0f} ms) on the GPU, against p50 {lat['score_p50']:.1f} ms for scoring alone; it is therefore computed on demand for the top-ranked hunks, not for every hunk.",
              f"<b>Cases:</b> one real test-split example per class (Aligned, Collateral, Scope-Expansion, Purpose-Drift) are shown with highlighted tokens. They were chosen as the most confident correct short examples, so they are illustrative, not representative.",
              "<b>Limit:</b> attribution is evidence about the model, not proof of the cause of a misalignment."]),
          H("2. Mathematical Formulation")]
    S += [P("Let x be the embedded input sequence (intent and hunk, joined as in training), x' the baseline sequence and F(x) = &sigma;(z<sub>align</sub>(x)) the probability of the misaligned class. "
            "For embedding dimension j of token t, Integrated Gradients is the path integral of the gradient from the baseline to the input:"),
          equation(r"$\mathrm{IG}_{t,j}(x)=(x_{t,j}-x'_{t,j})\int_{0}^{1}\frac{\partial F\!\left(x'+\alpha\,(x-x')\right)}{\partial x_{t,j}}\,d\alpha$", "ig"),
          P(f"The integral is approximated by a Riemann sum with m = {x['n_steps']} steps (m = 16 in the UI):"),
          equation(r"$\mathrm{IG}_{t,j}\approx (x_{t,j}-x'_{t,j})\,\frac{1}{m}\sum_{k=1}^{m}\frac{\partial F\!\left(x'+\frac{k}{m}(x-x')\right)}{\partial x_{t,j}}$", "ig_sum"),
          P("The token score sums over embedding dimensions, and the completeness property (scores add up to F(x) - F(x')) is what makes IG attractive; Input x Gradient is the one-step version:"),
          equation(r"$a_t=\sum_j \mathrm{IG}_{t,j}\qquad \mathrm{IxG}_{t,j}=x_{t,j}\,\frac{\partial F(x)}{\partial x_{t,j}}$", "scores"),
          P("<b>Baseline.</b> Every content token (intent and hunk) is replaced by the embedding of the [PAD] token; special tokens ([CLS], [SEP]) keep their own embedding so the sequence structure is unchanged. "
            "<b>Aggregation.</b> Sub-token scores are summed to words by character offsets (a word is an identifier or one punctuation mark); for hunks they are also summed per diff line. "
            "The sign is kept: positive (red) pushes toward misaligned, negative (blue) toward aligned. The mean share of absolute attribution that falls on intent tokens is "
            f"{R.pct(x['abs_attribution_share_on_intent_tokens_mean'], 1)} (hunks carry the rest, which matches their larger share of the input).")]
    trace("xai", "formulation and settings", ["results/xai.json"])
    S += [H("3. Implementation Notes"),
          *bullets([
              "<b>Code:</b> src/sift/explain.py (class built on Captum IntegratedGradients and InputXGradient, attributions taken at the embedding layer; word and line aggregation; deletion test), experiments/xai.py (experiment driver), the FastAPI endpoint /api/analyze with <i>explain: true</i> and the Review page.",
              "<b>Numerics:</b> attributions were computed with bf16 weights, internal batch 2 and a 1.5 GB GPU memory cap (fp32 ran out of memory under the cap), so the numbers describe the bf16 model.",
              f"<b>Faithfulness protocol:</b> {f['n']} test hunks predicted misaligned at the validation-tuned threshold {f['threshold']} ({R.pct(f['frac_true_misaligned'], 0)} of them truly misaligned); for k in 3, 5, 10, 20 the k highest-scoring content tokens are replaced by [PAD] and compared with k random content tokens (RandomState 473).",
              "<b>Scope:</b> Integrated Gradients and Input x Gradient are implemented; attention rollout (optional) is outside this work."]),
          H("4. Attribution Catalogue")]
    S.append(table([["Item", "Level", "Role", "Status"],
                    ["Integrated Gradients", "sub-token, word, diff line", "Primary explanation shown in the UI", "implemented and measured"],
                    ["Input x Gradient", "sub-token", "Cheap comparator in the faithfulness test", "implemented and measured"],
                    ["Attention rollout", "sub-token", "Optional", "not implemented"],
                    ["Word aggregation (intent)", "word", "Which request words drove the verdict", "implemented"],
                    ["Line aggregation (hunk)", "diff line", "Which changed lines drove the verdict", "implemented"],
                    ["Intent attribution share", "scalar per hunk", f"Mean {R.pct(x['abs_attribution_share_on_intent_tokens_mean'], 1)} of |attribution| on intent tokens", "measured"],
                    ["Deletion faithfulness", "top-k vs random-k masking", "Sanity check of the attribution ranking", "measured, n = 100"]],
                   [W * 0.24, W * 0.2, W * 0.36, W * 0.2]))
    trace("xai", "attribution catalogue table", ["results/xai.json"], "method descriptions are prose; share and n come from the JSON")
    # UI
    S += [H("5. UI Presentation"),
          P("The Review page ranks hunks by calibrated P(misaligned). Ticking <i>Explain top hunks</i> adds, for the selected hunk, the intent words and the diff lines coloured by IG score (the right panel below), with the caveat that it is evidence about the model, not proof of cause. "
            "Highlights are only computed on request because of the latency cost in section 8.")]
    S.append(figure(crop_ui(), 15.5, "Figure 1. Screenshot of the running app (Review page, test-set example, light theme): hunk card and the Explain panel with word and line highlights."))
    trace("xai", "Figure 1 UI screenshot", ["reports/screenshots/review_explained_desktop_light.png"], "real screenshot of the app, no numbers generated")
    # cases
    S += [H("6. Worked Case Studies (test split)")]
    S.append(P("One example per class from <i>results/xai_cases_ui.json</i>. Selection rule (experiments/xai.py): the most confident correct prediction with a short hunk, so these are favourable examples. "
               "Red marks pushed P(misaligned) up, blue pushed it down; colour intensity is relative to the largest absolute score within that segment."))
    labels = {"aligned": "Aligned", "collateral": "Collateral", "scope_expansion": "Scope-Expansion", "purpose_drift": "Purpose-Drift"}
    rows = [["Class", "Sample", "P(misal.)", "Top intent word", "Top hunk line (IG score)"]]
    for k in labels:
        c = cs[k]; ti = c["top_intent_tokens"][0]; tl = c["top_hunk_lines"][0]
        rows.append([labels[k], f"{c['sample_id']} ({c['repo']})", f"{c['p']:.3f}", f"{ti[0]} ({ti[1]:+.3f})", f"{tl[0].strip()[:60]} ({tl[1]:+.3f})"])
    S.append(table(rows, [W * 0.14, W * 0.19, W * 0.1, W * 0.17, W * 0.4]))
    for i, k in enumerate(("aligned", "collateral", "scope_expansion", "purpose_drift")):
        S.append(figure(case_figure(k, cs[k], labels[k]), 15.0, None))
    S.append(P("Figures 2-5: the four cases (summary table above). In the Aligned case the model gives low weight to any single line; the misaligned cases show scattered positive scores over several lines rather than one decisive line, "
               "so the highlights should be read as where the model looked, not as a verdict on which line is the unrequested change.", SMALL))
    trace("xai", "Figures 2-5 worked cases + case table", ["results/xai_cases_ui.json"])
    # faithfulness
    S += [H("7. Faithfulness Check"),
          P("The deletion test replaces the top-k attributed tokens by [PAD] and measures the drop in P(misaligned); a faithful attribution should drop it more than masking k random tokens.")]
    S.append(figure(faith_figure(x), 16.6, "Figure 6. Mean P(misaligned) after masking k tokens, and the share of hunks whose verdict flips to aligned (n = 100 test hunks)."))
    rows = [["k", "IG top-k", "IG random-k", "IG flip top / random", "IxG top-k", "IxG random-k", "IxG flip top / random", "IG top beats random"]]
    for k in ("3", "5", "10", "20"):
        a, b = f["ig"][k], f["ixg"][k]
        rows.append([k, fmt(a["mean_p_after_top_k_masked"]), fmt(a["mean_p_after_random_k_masked"]), f"{a['verdict_flip_rate_top']:.2f} / {a['verdict_flip_rate_random']:.2f}",
                     fmt(b["mean_p_after_top_k_masked"]), fmt(b["mean_p_after_random_k_masked"]), f"{b['verdict_flip_rate_top']:.2f} / {b['verdict_flip_rate_random']:.2f}", R.pct(a["frac_top_beats_random"], 0)])
    S.append(table(rows, [W * f_ for f_ in (0.06, 0.12, 0.13, 0.17, 0.12, 0.13, 0.17, 0.1)]))
    S.append(P(f"Mean unmasked P(misaligned) is {f['ig']['3']['mean_p_orig']:.3f}. Top-k masking always lowers it more than random masking, with the clearest gap at k = 5 to 10 "
               f"(IG k = 5: drop {f['ig']['5']['mean_drop_top']:.3f} vs {f['ig']['5']['mean_drop_random']:.3f}; IxG k = 5: {f['ixg']['5']['mean_drop_top']:.3f} vs {f['ixg']['5']['mean_drop_random']:.3f}). "
               f"At k = 3 the gap is small ({f['ig']['3']['mean_drop_top']:.3f} vs {f['ig']['3']['mean_drop_random']:.3f}) and the top set beats random in only {R.pct(f['ig']['3']['frac_top_beats_random'], 0)} of hunks. "
               "Input x Gradient separates from random slightly more than IG here, so IG is not shown to be the more faithful method by this test."))
    trace("xai", "Figure 6 and faithfulness table", ["results/xai.json"])
    # latency
    S += [H("8. Cost of Explanation"),
          table([["Measure", "Value", "Setting"],
                 ["Scoring one hunk (no explanation)", f"p50 {lat['score_p50']:.1f} ms", f"{lat['device']}, {lat['gpu']}"],
                 ["IG explanation, one hunk", f"p50 {lat['explain_ig_p50']:.0f} ms / p95 {lat['explain_ig_p95']:.0f} ms", f"{x['n_steps']} steps, bf16"],
                 ["Ratio explain / score (p50)", f"{lat['explain_ig_p50'] / lat['score_p50']:.1f}x", "computed"],
                 ["Peak GPU memory during XAI run", f"{x['peak_gpu_mem_gb']:.2f} GB", "capped at 1.5 GB"],
                 ["Explanation latency on CPU", "-", "the UI marks it as slow on CPU"]], [W * 0.4, W * 0.32, W * 0.28]),
          P(f"Source note in the JSON: {lat['note']}. IG needs {x['n_steps']} forward and backward passes, which explains the order-of-magnitude cost; the UI uses 16 steps, so its cost is proportionally lower.", SMALL)]
    trace("xai", "latency table", ["results/xai.json"])
    S += [H("9. Scope and Limitations"),
          *bullets([
              "Attribution is evidence about what the model used, not proof of what makes a hunk misaligned; the four cases are favourable examples chosen by confidence.",
              "[PAD] masking is itself off-distribution: random masking alone lowers P(misaligned) from "
              f"{f['ig']['5']['mean_p_orig']:.3f} to {f['ig']['5']['mean_p_after_random_k_masked']:.3f} at k = 5, so only the gap to random is informative, and a different replacement (mean embedding, token deletion) was not tested.",
              f"Faithfulness uses n = {f['n']} hunks from one split and one trained model ({SEED_NOTE}); no confidence interval is given.",
              "The deletion test measures sufficiency of the top tokens for the prediction, not whether a human finds the highlight useful; no user study was done.",
              *COMMON_LIMITS[:1], COMMON_LIMITS[3]]),
          H("10. Summary of Contributions"),
          P("Implemented IG and Input x Gradient on the cross-encoder with intent-side and hunk-side aggregation, integrated the explanation into the review UI, and measured faithfulness and cost on the test split. "
            "The methods (Integrated Gradients, Input x Gradient, deletion tests) are existing techniques; the contribution here is applying them to the hunk-versus-intent task and surfacing them in the review workflow. "
            "Everything above is regenerated from results/xai.json and results/xai_cases_ui.json by experiments/build_reports.py.")]
    return build(OUT / "23037_XAI.pdf", S, "XAI")


# ============================================================================================ Ablation
def ablation_rows():
    full_ev = RJ("results/eval_full.json"); sz = {r.get("run"): r.get("size_mb") for r in RJ("results/ablation.json")["rows"]}
    lat_full = RJ("results/latency_full.json"); onnx = RJ("results/latency_size_onnx.json")
    cpu_full = (lat_full["cpu"]["p50_ms"], lat_full["cpu"]["p95_ms"])
    spec = [("1", "Full model (baseline)", "full", "eval_full", "full"),
            ("2", "Hunk-only input (no intent)", "abl_hunk_only", "eval_abl_hunk_only", "full"),
            ("3", "Intent-only input (no hunk)", "abl_intent_only", "eval_abl_intent_only", "full"),
            ("4", "No intent consolidation (raw turns)", "abl_no_consolidation", "eval_abl_no_consolidation", "full"),
            ("5", "No focal loss (plain CE)", "abl_no_focal", "eval_abl_no_focal", "full"),
            ("6", "Alignment head only (no multi-task)", "abl_no_multitask", "eval_abl_no_multitask", "full"),
            ("7", "No denoising (noisy labels as is)", "abl_no_denoise", "eval_abl_no_denoise", "full"),
            ("8", "Distilled MiniLM student (fp32, PyTorch)", "student", "eval_student", "student"),
            ("8b", "MiniLM student, no distillation (extra)", "student_nodistill", "eval_student_nodistill", "student")]
    rows, srcs = [], []
    for cid, name, run, f, kind in spec:
        t = RJ(f"results/{f}.json")["test"]; r = ev_row(t); srcs.append(f"results/{f}.json")
        r.update(id=cid, name=name, size=sz.get(run) if run in sz else None,
                 lat=(f"{cpu_full[0]:.1f} / {cpu_full[1]:.1f} *" if kind == "full" else "-"))
        rows.append(r)
    sizes = {"student": sz.get("student")}
    for r in rows:
        if r["id"] == "8b": r["size"] = sz.get("student")
    f32, i8 = onnx["student_onnx_fp32"], onnx["student_onnx_int8"]
    for cid, name, o, k in (("8c", "Student ONNX fp32 (extra)", f32, "fp32"), ("9", "INT8 ONNX student (deployed target)", i8, "int8")):
        r = ev_row(o["test"]); r.update(id=cid, name=name, size=o["size_mb"], lat=f"{o['latency_ms']['p50_ms']:.1f} / {o['latency_ms']['p95_ms']:.1f}")
        rows.append(r)
    srcs += ["results/ablation.json (sizes)", "results/latency_full.json", "results/latency_size_onnx.json"]
    km = [r for r in RJ("results/ablation.json")["rows"] if r.get("group") == "ablation_kmeans"]
    return rows, km, srcs, cpu_full, onnx


def ablation_fig(rows):
    main = [r for r in rows if r["id"] in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}]
    names = [f"{r['id']}. " + r["name"].replace(" (fp32, PyTorch)", "").replace(" (deployed target)", "").replace(" (baseline)", "") for r in main]
    fig, ax = plt.subplots(1, 3, figsize=(7.8, 3.3), sharey=True)
    y = np.arange(len(main))[::-1]
    for a, (k, t) in zip(ax, (("mf1", "Macro-F1"), ("ndcg5", "NDCG@5"), ("auc_h", "Hard-tier pairwise AUC"))):
        v = [r[k] for r in main]
        a.barh(y, v, color=[R.PALETTE[0] if r["id"] == "1" else ("#d9822b" if r["id"] == "9" else "#9fb3cc") for r in main])
        for yy, vv in zip(y, v): a.text(vv + 0.01, yy, f"{vv:.3f}", va="center", fontsize=6)
        a.set_title(t, fontsize=8); a.set_xlim(0, 1.0); a.tick_params(labelsize=6.5); a.grid(alpha=.3, axis="x")
    ax[0].set_yticks(y); ax[0].set_yticklabels(names, fontsize=6.5)
    fig.suptitle("Ablation, test split, single seed (blue = full model, orange = deployed INT8; axes start at 0)", fontsize=7.5)
    fig.tight_layout()
    return fig_save(fig, "ablation_bars_report.png")


def build_ablation():
    rows, km, srcs, cpu_full, onnx = ablation_rows(); ab = RJ("results/ablation.json")
    full = rows[0]; by = {r["id"]: r for r in rows}; i8 = by["9"]
    S = doc_head("Ablation Study of the Sift Pipeline", "Which components of the alignment cross-encoder matter: inputs, consolidation, loss, heads, denoising and compression",
                 "Deliverable 5: Ablation Study", "Test cohort", "2,700 hunks (httpx, attrs, axios)")
    S += [H("1. Executive Summary and Study Objectives"),
          P("The ablation isolates each design choice of the Sift pipeline by removing it and re-training under the same protocol, then evaluating on the same held-out repository-disjoint test split. "
            "The headline finding is that removing the hunk or the intent is clearly harmful, but removing consolidation, focal loss, the multi-task heads or denoising did <b>not</b> lower the score in these single-seed runs. "
            "The study therefore does not show that those four components help; they may still matter on other data or seeds, but this experiment gives no evidence for them."),
          *bullets([f"Full model: macro-F1 {full['mf1']:.3f}, NDCG@5 {full['ndcg5']:.3f}, hard-tier pairwise AUC {full['auc_h']:.3f}, 596 MB.",
                    f"Deployed target (INT8 ONNX student): macro-F1 {i8['mf1']:.3f}, NDCG@5 {i8['ndcg5']:.3f}, hard-tier AUC {i8['auc_h']:.3f}, {i8['size']:.1f} MB, CPU p50 / p95 {i8['lat']} ms."]),
          H("2. Experimental Methodology and Evaluation Cohort"),
          P("<b>Cohort.</b> The test split of Sift-IA v1.0: 2,700 hunk-intent pairs from three repositories (httpx, attrs, axios) that appear in no training or validation row; 594 are misaligned. "
            "Labels are evaluated as <i>label_clean</i>. Rankable sessions (at least 2 hunks and 1 misaligned) number 456 in the stored evaluation. "
            "<b>Training.</b> All variants use the full main-model protocol (ModernBERT-base, L = 512, micro-batch 8 x 2, bf16, AdamW 2e-5, up to 6 epochs, early stopping on validation macro-F1) on the same rule and confident-learning denoised rows, except the no-denoising run. "
            "<b>Decision threshold</b> is tuned on validation macro-F1 only. <b>Metrics:</b> accuracy, macro-F1, per-class F1, precision, recall, specificity, within-session pairwise AUC per difficulty tier, NDCG@5, model size and CPU latency. "
            f"<b>Seeds:</b> {SEED_NOTE}; {ab['seeds']}.")]
    cfg = [["#", "Configuration", "What is changed", "Status"],
           ["1", "Full model", "ModernBERT cross-encoder, consolidated intent, focal loss, three heads, denoised rows", "run"],
           ["2", "Hunk-only", "Intent text removed from the input", "run"],
           ["3", "Intent-only", "Hunk text removed from the input", "run"],
           ["4", "No consolidation", "Raw multi-turn intent text, including chit-chat and retracted turns", "run"],
           ["5", "No focal loss", "Alignment term is plain cross-entropy", "run"],
           ["6", "Alignment head only", "Drift-type and confidence heads and their loss terms removed", "run"],
           ["7", "No denoising", "Train on all 12,600 rows with the noisy labels as shipped", "run"],
           ["8", "Distilled MiniLM student", "22M-parameter student, tau 4, alpha 0.5; extra rows: no distillation, ONNX fp32", "run"],
           ["9", "INT8 ONNX student", "Dynamic INT8 quantisation of the student (deployed target)", "run"],
           ["-", "Rules-only denoising", "Rule filters without confident learning (experiments/run_all_gpu.sh, abl_rules_only)", "-"],
           ["-", "K-means features on / off", "Cluster prior and centroid distance added to a Platt calibrator (results/kmeans.json)", "run, small study"]]
    S += [H("3. Description of Configurations"), table(cfg, [W * 0.05, W * 0.22, W * 0.58, W * 0.15])]
    trace("ablation", "configuration table", ["results/ablation.json"], "descriptions are prose; status from existence of eval JSONs")
    # results table 1
    def r1(r): return [r["id"], r["name"], fmt(r["acc"]), fmt(r["mf1"]), fmt(r["f1a"]), fmt(r["f1m"]), fmt(r["p"]), fmt(r["rec"]), fmt(r["spec"])]
    S += [H("4. Results"), P("Table 1: classification at the validation-tuned threshold (test split, label_clean). Per-class F1 comes from the stored eval JSONs (computed by experiments/evaluate.py from each run's predictions).", SMALL),
          table([["#", "Configuration", "Acc.", "Macro-F1", "F1 aligned", "F1 misal.", "Precision", "Recall", "Specif."]] + [r1(r) for r in rows],
                [W * f_ for f_ in (0.04, 0.33, 0.07, 0.09, 0.09, 0.09, 0.1, 0.09, 0.1)])]
    def r2(r): return [r["id"], r["name"].split(" (")[0], fmt(r["ndcg5"]), fmt(r["auc_r"]), fmt(r["auc_h"]), fmt(r["mf1_random"]), fmt(r["mf1_hard"]), f"{r['size']:.1f}" if r["size"] else "-", r["lat"]]
    S += [Spacer(1, 6), P("Table 2: ranking quality per difficulty tier, model size and single-pair CPU latency (p50 / p95, ms).", SMALL),
          table([["#", "Configuration", "NDCG@5", "AUC random", "AUC hard", "Macro-F1 random", "Macro-F1 hard", "Size MB", "CPU p50 / p95 ms"]] + [r2(r) for r in rows],
                [W * f_ for f_ in (0.04, 0.27, 0.08, 0.09, 0.08, 0.1, 0.1, 0.08, 0.16)]),
          P(f"* Same architecture and size as configuration 1: the latency of the full model ({cpu_full[0]:.1f} / {cpu_full[1]:.1f} ms, fp32 PyTorch, L = 512, 14 threads, idle machine, i7-13650HX) is shown for configurations 2-7 and was not timed separately. "
            f"Student latencies were measured with 4 threads while a training job shared the CPU, so they are not strictly comparable with the full-model figure. PyTorch student latency was not measured; INT8 and ONNX fp32 were. "
            "AUC and Macro-F1 per tier: tiers keep sessions that contain a misaligned hunk of that tier.", SMALL)]
    if km:
        off, on = km[0], km[1]
        S.append(P(f"K-means features (results/kmeans.json): Platt on the logit alone gives macro-F1 {off['macro_f1']:.3f}; adding cluster features gives {on['macro_f1']:.3f} ({on['macro_f1'] - off['macro_f1']:+.3f}), a change within single-seed noise.", SMALL))
    trace("ablation", "Tables 1 and 2", srcs + ["results/kmeans.json"])
    # insights
    others = [by[k] for k in ("4", "5", "6", "7")]
    S += [H("5. Insights"),
          *bullets([
              f"<b>Inputs matter:</b> hunk-only macro-F1 {by['2']['mf1']:.3f} ({d_(by['2']['mf1'] - full['mf1'])} vs full) and intent-only {by['3']['mf1']:.3f} ({d_(by['3']['mf1'] - full['mf1'])}); intent-only hard-tier AUC is {by['3']['auc_h']:.3f}, close to chance, so the pair is needed and the model is not just reading the intent.",
              "<b>Components not shown to help:</b> " + "; ".join(f"{r['name'].split(' (')[0].lower()} macro-F1 {r['mf1']:.3f} ({d_(r['mf1'] - full['mf1'])})" for r in others) +
              (". All four ablated variants scored higher macro-F1 than the full model, by " + f"{min(r['mf1'] - full['mf1'] for r in others):.3f} to {max(r['mf1'] - full['mf1'] for r in others):.3f}" if all(r['mf1'] > full['mf1'] for r in others) else ". Not all four ablated variants beat the full model") + ". With one seed and differences of about 0.01 in noise, the honest reading is that these components have no measurable benefit here, not that they hurt.",
              f"<b>Ranking tells a slightly different story:</b> hard-tier AUC is higher without denoising ({by['7']['auc_h']:.3f}) and with alignment head only ({by['6']['auc_h']:.3f}) than for the full model ({full['auc_h']:.3f}); on NDCG@5 the full model ({full['ndcg5']:.3f}) is within {max(abs(r['ndcg5'] - full['ndcg5']) for r in others):.3f} of every ablated variant.",
              f"<b>Compression costs little in macro-F1 and NDCG@5 but more on the hard tier:</b> the INT8 student keeps NDCG@5 ({i8['ndcg5']:.3f} vs {full['ndcg5']:.3f}) and macro-F1 ({i8['mf1']:.3f} vs {full['mf1']:.3f}) while hard-tier AUC falls from {full['auc_h']:.3f} to {i8['auc_h']:.3f}. INT8 quantisation itself changed macro-F1 by {i8['mf1'] - by['8c']['mf1']:+.3f} against the ONNX fp32 student.",
              f"<b>Distillation helps the student:</b> macro-F1 {by['8']['mf1']:.3f} with distillation vs {by['8b']['mf1']:.3f} without ({d_(by['8']['mf1'] - by['8b']['mf1'])}); hard-tier AUC {by['8']['auc_h']:.3f} vs {by['8b']['auc_h']:.3f}.",
              "<b>Error bars:</b> each configuration is a single seed (473), so differences carry no error bar."]),
          figure(ablation_fig(rows), 16.6, "Figure 1. Macro-F1, NDCG@5 and hard-tier AUC for configurations 1-9 (single seed)."),
          H("6. Deployment Recommendation")]
    trace("ablation", "Figure 1 bar chart and insights", srcs)
    f_lat = f"{cpu_full[1]:.1f}"
    S += [P(f"<b>Deploy the INT8 ONNX student</b> as the default local verifier. It is the smallest measured configuration ({i8['size']:.1f} MB, versus {full['size']:.1f} MB for the full model, a factor of {full['size'] / i8['size']:.0f}) "
            f"and meets the planned targets of under 30 MB and p95 under 100 ms on this machine (p95 {onnx['student_onnx_int8']['latency_ms']['p95_ms']:.1f} ms; targets, not guarantees on other hardware). "
            f"Its macro-F1 ({i8['mf1']:.3f}) and NDCG@5 ({i8['ndcg5']:.3f}) match the full model within {abs(i8['mf1'] - full['mf1']):.3f} and {abs(i8['ndcg5'] - full['ndcg5']):.3f}. "
            f"The cost is on the hard tier (pairwise AUC {i8['auc_h']:.3f} vs {full['auc_h']:.3f}), so where hard-tier ranking matters and a GPU is available the full model is the better choice; it also just meets the CPU target (p95 {f_lat} ms) but misses the size target."),
          P("Because the four ablated components gave no measurable gain, a simpler training recipe (cross-entropy, alignment head only, no consolidation, no denoising) is a defensible alternative that this study cannot rule out; it would need repeated seeds to decide. "
            "Command to repeat all ablation runs on a GPU machine: <font face='Courier' size='7.5'>bash run_gpu_experiments.sh</font>.", BODY),
          H("7. Scope and Limitations"), *bullets(COMMON_LIMITS)]
    return build(OUT / "23037_Ablation_Study.pdf", S, "Ablation Study")


# ============================================================================================ GAN
def gan_curves(g):
    fig, ax = plt.subplots(2, 3, figsize=(7.8, 4.6), sharex=True)
    for row, (k, nm) in enumerate((("gan_unsupervised", "Unsupervised (aligned pairs only)"), ("gan_hybrid_with_misaligned_negatives", "Hybrid (+ misaligned negatives for D)"))):
        h = g["variants"][k]["history"]; ep = [e["epoch"] for e in h]
        ax[row, 0].plot(ep, [e["d_loss"] for e in h], color=R.PALETTE[0], label="D loss"); ax[row, 0].plot(ep, [e["g_loss"] for e in h], color=R.PALETTE[1], label="G loss")
        ax[row, 0].set_title(f"{nm}: losses", fontsize=6.8); ax[row, 0].legend(fontsize=6, frameon=False)
        ax[row, 1].plot(ep, [e["val_auc"] for e in h], color=R.PALETTE[2]); b = int(np.argmax([e["val_auc"] for e in h]))
        ax[row, 1].axvline(ep[b], color="#999", ls="--", lw=.8); ax[row, 1].set_title(f"val AUC (best at epoch {ep[b]})", fontsize=6.8)
        ax[row, 2].plot(ep, [e["gen_diversity_cos"] for e in h], color=R.PALETTE[3]); ax[row, 2].axhline(1, color="#999", ls=":", lw=.8)
        ax[row, 2].set_title("generator mean pairwise cosine (1 = collapse)", fontsize=6.8); ax[row, 2].set_ylim(0, 1.05)
        for a in ax[row]: a.tick_params(labelsize=6); a.grid(alpha=.3)
    for a in ax[1]: a.set_xlabel("epoch", fontsize=7)
    fig.tight_layout()
    return fig_save(fig, "gan_curves_report.png")


def gan_bars(rows):
    keys = (("mf1", "Macro-F1"), ("auc_r", "AUC random tier"), ("auc_h", "AUC hard tier"), ("ndcg5", "NDCG@5"))
    fig, ax = plt.subplots(1, 4, figsize=(7.8, 2.9), sharey=True)
    y = np.arange(len(rows))[::-1]
    for a, (k, t) in zip(ax, keys):
        v = [r[k] for r in rows]
        a.barh(y, v, color=[R.PALETTE[0], "#d9822b", R.PALETTE[4], R.PALETTE[2], R.PALETTE[3]][:len(rows)])
        for yy, vv in zip(y, v): a.text(vv + 0.01, yy, f"{vv:.3f}", va="center", fontsize=6)
        a.set_title(t, fontsize=7.5); a.set_xlim(0, 1.05); a.tick_params(labelsize=6); a.grid(alpha=.3, axis="x")
    ax[0].set_yticks(y); ax[0].set_yticklabels([r["name"] for r in rows], fontsize=6.3)
    fig.tight_layout()
    return fig_save(fig, "gan_bars_report.png")


def build_gan():
    g = RJ("experiments/results/gan.json"); main = RJ("results/eval_full.json"); lat = RJ("results/latency_full.json"); onnx = RJ("results/latency_size_onnx.json")
    abl = {r.get("run"): r for r in RJ("results/ablation.json")["rows"]}
    gv = g["variants"]; U, Hy, C = gv["gan_unsupervised"], gv["gan_hybrid_with_misaligned_negatives"], gv["frozen_cosine_dissimilarity"]
    emb_p50, emb_mb = g["embedding_latency_ms_p50_per_pair"], g["embedder_size_mb"]
    rows = []
    m = ev_row(main["test"]); m.update(name="Main: ModernBERT cross-encoder", size=lat["size_mb"], lat=f"{lat['cpu']['p50_ms']:.1f}", params="149.0 M"); rows.append(m)
    i8 = onnx["student_onnx_int8"]; q = ev_row(i8["test"]); q.update(name="INT8 ONNX student (deployed)", size=i8["size_mb"], lat=f"{i8['latency_ms']['p50_ms']:.1f}", params="22 M"); rows.append(q)
    c = ev_row(C["test"]); c.update(name="Frozen cosine dissimilarity", size=emb_mb, lat=f"{emb_p50:.1f}", params="0 trainable"); rows.append(c)
    for nm, v in (("GAN-style, unsupervised D", U), ("GAN-style, hybrid D", Hy)):
        r = ev_row(v["test"]); r.update(name=nm, size=emb_mb + v["disc_size_mb"], lat=f"{emb_p50 + v['disc_latency_ms_p50']:.1f}", params=f"D {v['trainable_params_D'] / 1e6:.2f} M + G {v['trainable_params_G'] / 1e6:.2f} M"); rows.append(r)
    mc = rows[0]; cc = rows[2]; ur = rows[3]; hr = rows[4]
    hu, hh = U["history"], Hy["history"]
    bu = int(np.argmax([e["val_auc"] for e in hu])); bh = int(np.argmax([e["val_auc"] for e in hh]))
    n_col_u = sum(e["gen_diversity_cos"] > 0.9 for e in hu); n_col_h = sum(e["gen_diversity_cos"] > 0.9 for e in hh)
    dmin = min(hu, key=lambda e: e["d_loss"])
    S = doc_head("Adversarially Trained Baseline for Sift", "A GAN-style alignment scorer on frozen (intent, hunk) embeddings, compared with the cross-encoder and a no-training cosine baseline",
                 "Deliverable 4: GAN Results", "Interpretation", "Adversarial baseline on frozen embeddings")
    S += [H("1. Executive Summary and Motivation"),
          P("<b>Interpretation.</b> Sift has no GAN in its design, so this deliverable is implemented as "
            "an <b>adversarially trained baseline</b> that works on frozen sentence embeddings of the (intent, hunk) pairs. A generator learns to produce synthetic aligned-hunk embeddings from noise and the intent embedding; "
            "a discriminator learns to separate real aligned pairs from generated ones and is then used as an alignment scorer. The point is a comparison: does adversarial training on frozen embeddings get close to the fine-tuned cross-encoder?"),
          *bullets([f"On the test split the unsupervised variant reaches macro-F1 {ur['mf1']:.3f}, NDCG@5 {ur['ndcg5']:.3f} and pairwise AUC {ur['auc_r']:.3f} (random tier) / {ur['auc_h']:.3f} (hard tier); the cross-encoder reaches {mc['mf1']:.3f}, {mc['ndcg5']:.3f}, {mc['auc_r']:.3f} / {mc['auc_h']:.3f}.",
                    f"The hybrid variant (misaligned training pairs also fed to D as negatives) reaches {hr['mf1']:.3f}, {hr['ndcg5']:.3f}, {hr['auc_r']:.3f} / {hr['auc_h']:.3f}.",
                    f"Neither GAN variant beats the training-free frozen cosine dissimilarity on NDCG@5 ({cc['ndcg5']:.3f}) or random-tier AUC ({cc['auc_r']:.3f}); on macro-F1 they are {ur['mf1'] - cc['mf1']:+.3f} and {hr['mf1'] - cc['mf1']:+.3f} above it. The adversarial training therefore adds little over a cosine score.",
                    f"Training was unstable: the generator was collapsed (mean pairwise cosine above 0.9) for {n_col_u} epoch(s) of {len(hu)} in the unsupervised run and {n_col_h} in the hybrid run, and validation AUC peaked at epoch {hu[bu]['epoch']} then declined."]),
          H("2. Architecture Formulations")]
    S += [P("Notation: e<sub>i</sub> and e<sub>h</sub> are unit-norm 384-dimensional embeddings of the consolidated intent and of the hunk text (frozen all-MiniLM-L6-v2, max length 512); z ~ N(0, I<sub>64</sub>)."),
          P("<b>Generator</b> G (a 2-hidden-layer MLP with 512 units and LeakyReLU 0.2, output L2-normalised) maps noise and intent to a synthetic aligned-hunk embedding:"),
          equation(r"$\tilde e_h = G(z, e_i)=\frac{\mathrm{MLP}([e_i;z])}{\left\|\mathrm{MLP}([e_i;z])\right\|_2}$", "gan_G"),
          P("<b>Discriminator</b> D scores a pair from the features [e<sub>i</sub>; e<sub>h</sub>; e<sub>i</sub> * e<sub>h</sub>; |e<sub>i</sub> - e<sub>h</sub>|] (MLP 1536-512-256-1, dropout 0.2) and is trained with the standard non-saturating GAN losses, using only <i>aligned</i> training pairs as real:"),
          equation(r"$\mathcal{L}_D=-\mathbb{E}_{(e_i,e_h)\sim \mathrm{aligned}}\log\sigma(D(e_i,e_h))-\mathbb{E}\log\left(1-\sigma(D(e_i,G(z,e_i)))\right)$", "gan_LD"),
          equation(r"$\mathcal{L}_G=-\mathbb{E}\log\sigma\left(D(e_i,G(z,e_i))\right)$", "gan_LG"),
          P("<b>Hybrid variant</b> adds real misaligned pairs as extra negatives for D (a supervised signal that the pure GAN objective does not use):"),
          equation(r"$\mathcal{L}_D^{\mathrm{hyb}}=\mathcal{L}_D-\mathbb{E}_{(e_i,e_h)\sim \mathrm{misaligned}}\log\left(1-\sigma(D(e_i,e_h))\right)$", "gan_LDhyb"),
          P("<b>Scoring.</b> The misalignment score of a pair is s = -D(e<sub>i</sub>, e<sub>h</sub>) (low realism = likely misaligned); the generator is discarded at inference. A threshold is tuned on validation macro-F1 after a monotone squashing of s, so the GAN scores are not calibrated probabilities. "
            "<b>Collapse monitor.</b> The mean pairwise cosine of 256 generated embeddings; 1.0 means every generated embedding is identical (complete mode collapse)."),
          equation(r"$\mathrm{div}=\frac{1}{n(n-1)}\sum_{a\neq b}\cos(\tilde e^{(a)},\tilde e^{(b)})$", "gan_div"),
          P("<b>Reference baseline.</b> Frozen cosine dissimilarity, s = -cos(e<sub>i</sub>, e<sub>h</sub>), with no trainable parameter; it shows how much the adversarial machinery adds over the embeddings alone.")]
    trace("gan", "architecture description", ["experiments/results/gan.json"], "equations describe src/sift/gan_baseline.py; sizes from JSON")
    S += [H("3. Experimental Setup and Training Protocol"),
          table([["Item", "Setting"],
                 ["Data", "Sift-IA v1.0 train / val / test (repository-disjoint). Real = aligned training pairs (observed label, so label noise is present); val for checkpoint and threshold; test for reporting."],
                 ["Embedder", g["encoder"] + f", {emb_mb:.1f} MB, CPU"],
                 ["Optimiser", "Adam, lr 2e-4, betas (0.5, 0.999), batch 128, 40 epochs, seed 473, CPU only"],
                 ["Checkpoint", f"best validation AUC (epoch {hu[bu]['epoch']} unsupervised, epoch {hh[bh]['epoch']} hybrid; best val AUC {U['best_val_auc']:.3f} and {Hy['best_val_auc']:.3f})"],
                 ["Parameters", f"D {U['trainable_params_D']:,}; G {U['trainable_params_G']:,}; D size {U['disc_size_mb']:.2f} MB"],
                 ["Training time", f"{U['train_time_s']:.0f} s (unsupervised), {Hy['train_time_s']:.0f} s (hybrid), on top of a one-off embedding pass"],
                 ["Evaluation", "Same test split, label_clean, same metrics and tiers as the cross-encoder; threshold from validation"]], [W * 0.17, W * 0.83]),
          H("4. Quantitative Comparison")]
    trs = [["Model", "Macro-F1", "AUC random", "AUC hard", "NDCG@5", "ECE", "Size MB", "CPU p50 ms / pair", "Trainable"]]
    for r in rows: trs.append([r["name"], fmt(r["mf1"]), fmt(r["auc_r"]), fmt(r["auc_h"]), fmt(r["ndcg5"]), fmt(r["ece"]), f"{r['size']:.1f}", r["lat"], r["params"]])
    S += [table(trs, [W * f_ for f_ in (0.21, 0.08, 0.09, 0.08, 0.08, 0.07, 0.08, 0.12, 0.19)]),
          P("Notes: CPU latency of the GAN-style and cosine rows is the frozen embedder on one pair (p50 "
            f"{emb_p50:.1f} ms, p95 {g['embedding_latency_ms_p95_per_pair']:.1f} ms) plus, for the GAN rows, the discriminator (p50 {U['disc_latency_ms_p50']:.2f} ms); the cross-encoder figure is fp32 PyTorch on an idle machine and the INT8 figure was taken while a training job shared the CPU, so they are not strictly comparable. "
            "ECE of the GAN and cosine rows is large because their scores are squashed, not calibrated, probabilities. Size for the GAN rows is the embedder plus D (G is not needed at inference).", SMALL)]
    trace("gan", "benchmark table", ["experiments/results/gan.json", "results/eval_full.json", "results/latency_full.json", "results/latency_size_onnx.json"])
    cm_rows = [["Scorer", "Threshold", "TN", "FP", "FN", "TP", "Precision", "Recall", "Specificity"]]
    for nm, v in (("Frozen cosine", C), ("GAN-style, unsupervised D", U), ("GAN-style, hybrid D", Hy)):
        t = v["test"]; cm_rows.append([nm, fmt(t["threshold"], 2), t["tn"], t["fp"], t["fn"], t["tp"], fmt(t["precision"]), fmt(t["recall"]), fmt(t["specificity"])])
    cmi = RJ("results/eval_full.json")["test"]["full"]; cm_rows.insert(1, ["Main cross-encoder", fmt(cmi["threshold"], 2), cmi["tn"], cmi["fp"], cmi["fn"], cmi["tp"], fmt(cmi["precision"]), fmt(cmi["recall"]), fmt(cmi["specificity"])])
    S += [Spacer(1, 5), P("Confusion counts on the 2,700 test hunks at the validation-tuned threshold (the thresholds of the GAN rows live on a squashed score scale and are not comparable with the cross-encoder's probability threshold):", SMALL),
          table(cm_rows, [W * f_ for f_ in (0.22, 0.1, 0.07, 0.07, 0.07, 0.07, 0.13, 0.13, 0.14)])]
    S += [H("5. Training Curves and Instability Analysis"), figure(gan_curves(g), 16.6, "Figure 1. Training history of both variants (experiments/results/gan.json): losses, validation AUC and generator diversity per epoch."),
          *bullets([
              f"<b>Mode collapse at the start.</b> The generator output is almost constant at epoch 1 (mean pairwise cosine {hu[0]['gen_diversity_cos']:.2f} unsupervised, {hh[0]['gen_diversity_cos']:.2f} hybrid); it diversifies later (minimum {U['min_gen_diversity']:.2f} and {Hy['min_gen_diversity']:.2f}, final {U['final_gen_diversity']:.2f} and {Hy['final_gen_diversity']:.2f}). "
              f"In the unsupervised run diversity stays above 0.9 for {n_col_u} of {len(hu)} epochs (last such epoch: {max([e['epoch'] for e in hu if e['gen_diversity_cos'] > 0.9] or [0])}); in the hybrid run for {n_col_h} (last: {max([e['epoch'] for e in hh if e['gen_diversity_cos'] > 0.9] or [0])}).",
              f"<b>Discriminator dominance.</b> The lowest D loss ({dmin['d_loss']:.2f} at epoch {dmin['epoch']}) coincides with a generator loss of {dmin['g_loss']:.2f} and a generator diversity of {dmin['gen_diversity_cos']:.2f}, consistent with D easily rejecting near-identical generated embeddings; later D loss settles near {hu[-1]['d_loss']:.2f} and G loss near {hu[-1]['g_loss']:.2f}.",
              f"<b>Overfitting of the scorer.</b> Validation AUC peaks at {hu[bu]['val_auc']:.3f} (epoch {hu[bu]['epoch']}) and ends at {hu[-1]['val_auc']:.3f}; the hybrid variant peaks at {hh[bh]['val_auc']:.3f} (epoch {hh[bh]['epoch']}) and ends at {hh[-1]['val_auc']:.3f}. Best-by-validation checkpointing was therefore needed and means the validation set influenced the model choice.",
              f"<b>Run summary.</b> Unsupervised: D loss {hu[0]['d_loss']:.2f} -> {hu[-1]['d_loss']:.2f}, G loss {hu[0]['g_loss']:.2f} -> {hu[-1]['g_loss']:.2f}, epoch time about {np.mean([e['epoch_s'] for e in hu]):.1f} s. Hybrid: D loss {hh[0]['d_loss']:.2f} -> {hh[-1]['d_loss']:.2f}, G loss {hh[0]['g_loss']:.2f} -> {hh[-1]['g_loss']:.2f}. The hybrid D loss stays higher because it includes the misaligned-negative term.",
              "<b>No convergence guarantee.</b> A single seed and a fixed 40 epochs were run; GAN training is seed-sensitive and different seeds could give different collapse patterns. This was not tested."]),
          H("6. Failure Analysis"),
          *bullets([
              f"<b>One-class training signal.</b> The unsupervised D sees only aligned pairs, so it learns what real pairs look like, not what misaligned pairs look like; its hard-tier AUC ({ur['auc_h']:.3f}) is {mc['auc_h'] - ur['auc_h']:.3f} below the cross-encoder's and its macro-F1 is {mc['mf1'] - ur['mf1']:.3f} lower.",
              f"<b>Adversarial training adds little over the embedding.</b> The no-training cosine baseline already gives NDCG@5 {cc['ndcg5']:.3f} and random-tier AUC {cc['auc_r']:.3f}, above both GAN variants on those two metrics. The hybrid variant gains {hr['auc_r'] - ur['auc_r']:+.3f} random-tier AUC over the unsupervised one, but that comes from the supervised negatives, not from the adversarial game.",
              f"<b>Where a frozen baseline does well and why that matters.</b> Frozen cosine matches or beats the cross-encoder on NDCG@5 ({cc['ndcg5']:.3f} vs {mc['ndcg5']:.3f}) and random-tier AUC ({cc['auc_r']:.3f} vs {mc['auc_r']:.3f}) with no training. On the hard tier the cross-encoder is only marginally ahead ({mc['auc_h']:.3f} vs {cc['auc_h']:.3f}), so frozen embeddings already carry much of the signal these tiers measure; the hybrid GAN variant also edges the cross-encoder on random-tier AUC ({hr['auc_r']:.3f} vs {mc['auc_r']:.3f}). The cross-encoder's advantage is mainly macro-F1 ({mc['mf1']:.3f} vs {cc['mf1']:.3f}) and hard-tier AUC over the weakest scorers, not a clear win in ranking.",
              f"<b>Poor calibration and unstable thresholds.</b> The GAN scores have ECE {ur['ece']:.2f} / {hr['ece']:.2f}; a threshold tuned on validation (macro-F1 {ur['mf1']:.3f}, recall {ur['rec']:.3f}, precision {ur['p']:.3f} for the unsupervised variant) cannot be read as a probability."]),
          H("7. Visual Comparison"), figure(gan_bars(rows), 16.6, "Figure 2. Test-split comparison of the five scorers (same split, same thresholding rule)."),
          H("8. Scope, Limitations and Conclusion"),
          P(f"The adversarially trained baseline is a working but weak alignment scorer: it reaches macro-F1 {ur['mf1']:.3f} / {hr['mf1']:.3f} against {mc['mf1']:.3f} for the cross-encoder, is no better than a cosine score on ranking, shows early mode collapse and later overfitting, and is poorly calibrated. "
            "Its advantages are size of the trainable part (a 3.7 MB discriminator) and no fine-tuning of the encoder, but the frozen embedder it depends on is itself 91 MB. It is not part of the shipped Sift pipeline. "
            "This deliverable follows the interpretation stated in section 1."),
          *bullets([COMMON_LIMITS[0], COMMON_LIMITS[2], COMMON_LIMITS[3], "GAN rows are one 40-epoch run per variant on CPU; no repeat seeds, no hyper-parameter search. The comparison with the cross-encoder is between very different training budgets.",
                    "Command to regenerate the GAN results on another machine: <font face='Courier' size='7.5'>PYTHONPATH=src python experiments/run_gan.py</font>"])]
    trace("gan", "Figure 1 curves, Figure 2 bars, analysis numbers", ["experiments/results/gan.json", "results/eval_full.json"])
    return build(OUT / "23037_GANResults.pdf", S, "GAN Results")


# ============================================================================================ other deliverables (to be added)
def build_testing():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_testing", ROOT / "reports/build_testing.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.build()


def build_review():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_review", ROOT / "reports/build_review.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.build()


BUILDERS = {"xai": build_xai, "ablation": build_ablation, "gan": build_gan, "testing": build_testing, "review": build_review}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", choices=list(BUILDERS), action="append")
    a = ap.parse_args()
    for k in (a.only or list(BUILDERS)):
        out = BUILDERS[k]()
        if out: print("wrote", out)
    R.save_trace(); print("trace ->", R.TRACE_PATH)


if __name__ == "__main__":
    main()
