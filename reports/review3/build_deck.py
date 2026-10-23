#!/usr/bin/env python3
"""Build Review3/Sift_Review3_Presentation.pptx in the Team 24 template.

Run:  PYTHONPATH=src .venv/bin/python reports/review3/build_deck.py
Every number comes from results/*.json|csv, experiments/results/*.json,
results/SUMMARY.md, the Review 2 deck, ppt/standard_paper.pdf or the dataset
description document; each use is logged to reports/review3/trace.json.
"""
import copy
import csv
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEMPLATE = os.path.join(ROOT, "ppt", "Team_24 (1).pptx")
OUT_DIR = os.path.join(ROOT, "Review3")
OUT = os.path.join(OUT_DIR, "Sift_Review3_Presentation.pptx")
ASSETS = os.path.join(ROOT, "reports", "review3", "assets")
TRACE = os.path.join(ROOT, "reports", "review3", "trace.json")
os.makedirs(ASSETS, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

RED = RGBColor(0xFF, 0x00, 0x00)
CRIM = RGBColor(0xC0, 0x00, 0x00)
MAROON = RGBColor(0xB6, 0x11, 0x4D)
DARK = RGBColor(0x26, 0x26, 0x26)
GREY = RGBColor(0x59, 0x59, 0x59)
CARD = RGBColor(0xF4, 0xF5, 0xFA)
HILITE = RGBColor(0xFB, 0xE3, 0xE3)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BORDER = "CFD8EC"
C_CRIM, C_MAR, C_GREY, C_LGREY, C_DGREY = "#C00000", "#B6114D", "#8C8C8C", "#C9C9C9", "#404040"


def J(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def CSV(path):
    with open(os.path.join(ROOT, path)) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- trace
TRACE_LOG = []
CUR = {"slide": 0}


def V(label, val, src):
    TRACE_LOG.append({"slide": CUR["slide"], "label": label, "value": val, "source": src})
    return val


f3 = lambda v: f"{float(v):.3f}"
f1 = lambda v: f"{float(v):.1f}"
ni = lambda v: f"{int(v):,}"
pct = lambda v: f"{100 * float(v):.0f}%"

# ---------------------------------------------------------------- data
MC = {r["model"]: r for r in CSV("results/master_comparison.csv")}
MCS = "results/master_comparison.csv"
AB = {r["model"]: r for r in CSV("results/ablation.csv")}
ABS = "results/ablation.csv"
DS = {r["statistic"]: r["value"] for r in CSV("results/dataset_statistics.csv")}
DSS = "results/dataset_statistics.csv"
CL = J("results/confident_learning.json")
KM = J("results/kmeans.json")
GAN = J("experiments/results/gan.json")
GVM = J("results/gan_vs_main.json")
XAI = J("results/xai.json")
NZ = J("results/noise_robustness.json")
TS = J("results/test_summary.json")
TM = J("experiments/results/testing_measurements.json")
CAL = J("results/calibration.json")
LATF = J("results/latency_full.json")
LATO = J("results/latency_size_onnx.json")
EF = J("results/eval_full.json")
RUNCFG = J("experiments/runs/full/config.json")
BASE = J("experiments/results/baselines.json")
DEN = J("experiments/results/denoise.json")


# ---------------------------------------------------------------- charts
def style_ax(ax, fs=10):
    ax.tick_params(labelsize=fs)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="both", color="#E6E6E6", lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    p = os.path.join(ASSETS, name)
    fig.savefig(p, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


SHORT = {
    "B3 Hunk size": "B3 hunk size",
    "B4 TF-IDF dissimilarity": "B4 TF-IDF",
    "B5 Logistic regression": "B5 logistic reg.",
    "Siamese BiLSTM": "Siamese BiLSTM",
    "MiniLM-L6 cross-encoder": "MiniLM-L6 CE",
    "CodeBERT cross-encoder": "CodeBERT CE",
    "Main: ModernBERT-base": "Main ModernBERT",
    "Frozen-embedding frozen cosine dissimilarity": "Frozen cosine",
    "GAN-style baseline: gan unsupervised": "GAN-style",
    "Distilled MiniLM student (fp32)": "MiniLM student",
}


def chart_tier_auc():
    keys = list(SHORT)
    rnd = [float(MC[k]["auc_random"]) for k in keys]
    hard = [float(MC[k]["auc_hard"]) for k in keys]
    for k in keys:
        V(f"AUC random/hard {k}", [MC[k]["auc_random"], MC[k]["auc_hard"]], MCS)
    fig, ax = plt.subplots(figsize=(4.7, 3.7))
    y = list(range(len(keys)))[::-1]
    h = 0.38
    ax.barh([i + h / 2 for i in y], rnd, h, color=C_LGREY, label="random tier")
    ax.barh([i - h / 2 for i in y], hard, h, color=C_CRIM, label="hard tier")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[k] for k in keys], fontsize=10)
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlim(0.4, 0.9)
    ax.set_xlabel("within-session pairwise AUC (0.5 = chance)", fontsize=10)
    ax.legend(fontsize=9.5, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False)
    style_ax(ax)
    ax.grid(axis="y", visible=False)
    return save(fig, "tier_auc.png")


def ablation_rows():
    order = ["1. Full model", "2. hunk-only", "3. intent-only", "4. no intent consolidation",
             "5. no focal loss (CE)", "6. alignment head only", "7. no denoising",
             "9a. k-means features OFF (Platt on logit)", "9b. k-means features ON (hybrid)",
             "10. Distilled MiniLM student (fp32)", "11. INT8 ONNX student (deployed target)"]
    label = {"1. Full model": "Full model", "2. hunk-only": "Hunk only", "3. intent-only": "Intent only",
             "4. no intent consolidation": "No consolidation", "5. no focal loss (CE)": "No focal loss (CE)",
             "6. alignment head only": "Alignment head only", "7. no denoising": "No denoising",
             "9a. k-means features OFF (Platt on logit)": "k-means off",
             "9b. k-means features ON (hybrid)": "k-means on",
             "10. Distilled MiniLM student (fp32)": "Student fp32",
             "11. INT8 ONNX student (deployed target)": "Student INT8"}
    return [(label[o], AB[o]) for o in order]


def chart_ablation():
    rows = ablation_rows()
    fig, ax = plt.subplots(figsize=(3.3, 3.6))
    y = list(range(len(rows)))[::-1]
    vals = [float(r["macro_f1"]) for _, r in rows]
    cols = [C_CRIM if n == "Full model" else (C_DGREY if n in ("Hunk only", "Intent only") else C_LGREY) for n, _ in rows]
    ax.barh(y, vals, color=cols)
    for yy, v in zip(y, vals):
        ax.text(v + 0.004, yy, f"{v:.3f}", va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in rows], fontsize=11)
    ax.set_xlim(0.4, 0.8)
    ax.set_xlabel("test macro-F1 (seed 473)", fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    style_ax(ax)
    ax.grid(axis="y", visible=False)
    return save(fig, "ablation_f1.png")


def chart_noise():
    res = NZ["results"]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for arm, col, lab in (("no_denoise", C_GREY, "no denoising"), ("denoise", C_CRIM, "rules + confident learning")):
        pts = sorted((r["noise_rate"], r["test"]["macro_f1"]) for r in res if r["arm"] == arm)
        for r, v in pts:
            V(f"noise {arm} macro-F1 @ {r}", v, "results/noise_robustness.json")
        ax.plot([100 * p[0] for p in pts], [p[1] for p in pts], "o-", color=col, lw=2.2, ms=6, label=lab)
        for p in [q for q in pts if q[0] >= 0.2]:
            other = {r["noise_rate"]: r["test"]["macro_f1"] for r in res if r["arm"] != arm}[p[0]]
            up = p[1] >= other
            ax.annotate(f"{p[1]:.2f}", (100 * p[0], p[1]), textcoords="offset points",
                        xytext=(14 if p[0] == 0 else (-14 if p[0] == 0.3 else 0), 8 if up else -15), ha="center", fontsize=9, color=col)
    ax.set_xlabel("injected label-noise rate (%)", fontsize=10)
    ax.set_ylabel("test macro-F1", fontsize=10)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(0.48, 0.63)
    ax.set_xlim(-4, 34)
    ax.legend(fontsize=9.5, frameon=False, loc="lower left")
    style_ax(ax)
    return save(fig, "noise_robust.png")


def chart_faith():
    ig = XAI["faithfulness"]["ig"]
    ixg = XAI["faithfulness"]["ixg"]
    ks = sorted(int(k) for k in ig)
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    ax.plot(ks, [ig[str(k)]["mean_p_after_top_k_masked"] for k in ks], "o-", color=C_CRIM, lw=2.2, label="mask top-k by IG")
    ax.plot(ks, [ixg[str(k)]["mean_p_after_top_k_masked"] for k in ks], "s-", color="#E08A00", lw=2, label="mask top-k by IxG")
    ax.plot(ks, [ig[str(k)]["mean_p_after_random_k_masked"] for k in ks], "^--", color=C_GREY, lw=2, label="mask k random")
    p0 = ig[str(ks[0])]["mean_p_orig"]
    ax.axhline(p0, color="k", ls=":", lw=1)
    ax.text(ks[-1], p0 + 0.008, f"unmasked {p0:.3f}", ha="right", fontsize=9)
    ax.set_xlabel("tokens masked (k)", fontsize=10)
    ax.set_ylabel("mean P(misaligned)", fontsize=10)
    ax.set_xticks(ks)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    style_ax(ax)
    return save(fig, "faith.png")


def chart_cl():
    d = CL["detection"]
    names = [("prune_by_noise_rate", "prune by\nnoise rate"), ("confident_learning", "confident\nlearning"),
             ("final drop list (CL + rules)", "final drop list\n(CL + rules)")]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    import numpy as np
    x = np.arange(len(names))
    w = 0.26
    for i, (m, c) in enumerate((("precision", C_CRIM), ("recall", "#1F7A8C"), ("f1", "#E0A030"))):
        vals = [d[k][m] for k, _ in names]
        ax.bar(x + (i - 1) * w, vals, w, color=c, label=m)
        for xx, v in zip(x + (i - 1) * w, vals):
            ax.text(xx, v + 0.008, f"{v:.2f}", ha="center", fontsize=8.5)
    rp = CL["random_precision"]
    ax.axhline(rp, color="k", ls="--", lw=1, label=f"random precision {rp:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in names], fontsize=9.5)
    ax.set_ylim(0, 0.5)
    ax.set_ylim(0, 0.58)
    ax.legend(fontsize=9, frameon=False, loc="upper right", ncol=2)
    style_ax(ax)
    ax.grid(axis="x", visible=False)
    return save(fig, "cl_detect.png")


def chart_gan():
    import numpy as np
    comp = GVM["compared"]
    names = [("full ModernBERT", "Main", C_CRIM), ("frozen cosine dissimilarity", "Frozen cosine", C_GREY),
             ("gan unsupervised", "GAN-style", "#1F7A8C"), ("gan hybrid with misaligned negatives", "GAN hybrid", "#E0A030")]
    mets = [("macro_f1", "macro-F1"), ("ndcg@5", "NDCG@5"), ("auc_random", "AUC rand."), ("auc_hard", "AUC hard")]
    fig, ax = plt.subplots(figsize=(4.7, 3.3))
    x = np.arange(len(mets))
    w = 0.2
    for i, (k, lab, c) in enumerate(names):
        vals = [comp[k][m] for m, _ in mets]
        V(f"GAN compare {k}", {m: comp[k][m] for m, _ in mets}, "results/gan_vs_main.json")
        ax.bar(x + (i - 1.5) * w, vals, w, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([m for _, m in mets], fontsize=10)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=9, frameon=False, ncol=4, loc="upper center", columnspacing=0.8, handlelength=1.0)
    style_ax(ax)
    ax.grid(axis="x", visible=False)
    return save(fig, "gan_bars.png")


def chart_gan_curves():
    h = GAN["variants"]["gan_unsupervised"]["history"]
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    ax.plot(ep, [r["d_loss"] for r in h], color=C_CRIM, lw=2, label="discriminator loss")
    ax.set_xlabel("epoch", fontsize=10)
    ax.set_ylabel("discriminator loss", fontsize=10, color=C_CRIM)
    ax2 = ax.twinx()
    ax2.plot(ep, [r["gen_diversity_cos"] for r in h], color="#1F7A8C", lw=2, ls="--", label="generator pairwise cosine")
    ax2.set_ylabel("generator pairwise cosine", fontsize=10, color="#1F7A8C")
    ax2.tick_params(labelsize=10)
    ax2.spines["top"].set_visible(False)
    V("GAN generator cosine epoch 1 / last", [h[0]["gen_diversity_cos"], h[-1]["gen_diversity_cos"]], "experiments/results/gan.json")
    style_ax(ax)
    ax.annotate(f"collapsed: {h[0]['gen_diversity_cos']:.2f}", (1, h[0]["d_loss"]), xytext=(10, h[0]["d_loss"] - 0.05), fontsize=9)
    return save(fig, "gan_curves.png")


def chart_cal():
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.5), sharey=True)
    for ax, key, title in ((axs[0], "reliability_before", f"before (T=1): ECE {CAL['ece_test_before']:.3f}"),
                           (axs[1], "reliability_after", f"after (T={CAL['temperature']:.2f}): ECE {CAL['ece_test_after']:.3f}")):
        rb = CAL[key]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.plot([b["mean_conf"] for b in rb if b["n"]], [b["frac_pos"] for b in rb if b["n"]], "o-", color=C_CRIM, ms=4, lw=1.8)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("mean predicted P", fontsize=9.5)
        style_ax(ax, 9)
    axs[0].set_ylabel("observed fraction", fontsize=9.5)
    V("calibration", {k: CAL[k] for k in ("temperature", "ece_test_before", "ece_test_after")}, "results/calibration.json")
    fig.tight_layout()
    return save(fig, "cal.png")


def chart_latency():
    pts = [("Main ModernBERT\nPyTorch fp32", LATF["size_mb"], LATF["cpu"]["p95_ms"], C_CRIM),
           ("Student ONNX fp32", LATO["student_onnx_fp32"]["size_mb"], LATO["student_onnx_fp32"]["latency_ms"]["p95_ms"], C_GREY),
           ("Student ONNX INT8", LATO["student_onnx_int8"]["size_mb"], LATO["student_onnx_int8"]["latency_ms"]["p95_ms"], "#1F7A8C")]
    fig, ax = plt.subplots(figsize=(4.1, 3.4))
    offs = [(-8, -30, "right"), (10, 10, "left"), (10, -22, "left")]
    for (n, s, p, c), (dx, dy, ha) in zip(pts, offs):
        ax.scatter([s], [p], s=70, color=c, zorder=3)
        ax.annotate(n, (s, p), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=9.5)
    ax.axvline(30, color="k", ls=":", lw=1)
    ax.axhline(100, color="k", ls=":", lw=1)
    ax.text(36, 100.6, "size target 30 MB", fontsize=8.5, va="bottom")
    ax.text(670, 98.6, "p95 target 100 ms", fontsize=8.5, ha="right", va="top")
    ax.set_xlim(-20, 680)
    ax.set_ylim(60, 106)
    ax.set_xlabel("model size (MB)", fontsize=10)
    ax.set_ylabel("CPU p95 latency (ms)", fontsize=10)
    style_ax(ax)
    return save(fig, "latency.png")


def chart_arch():
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(9.3, 3.62))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 39)
    ax.axis("off")

    def box(x, y, w, h, title, sub="", fc="#F4F5FA", ec="#8C8C8C", tc="#262626", lw=1.2, tfs=10.5, sfs=9.4, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.8", fc=fc, ec=ec, lw=lw, ls=ls))
        if sub:
            ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=tfs, fontweight="bold", color=tc)
            ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=sfs, color=tc, linespacing=1.15)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tfs, fontweight="bold", color=tc)

    def arrow(x1, y1, x2, y2, c="#B6114D", ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6, ls=ls))

    ax.text(0.5, 37.6, "RUNTIME PATH (local, CPU or GPU)", fontsize=8.5, fontweight="bold", color="#B6114D")
    w, y, h = 14.6, 27.0, 9.2
    xs = [0.5, 17.4, 34.3, 51.2, 68.1, 85.0]
    box(xs[0], y, w, h, "Agent session", "chat turns +\ngit unified diff")
    box(xs[1], y, w, h, "Consolidator", "drops chit-chat,\nretractions, repeats")
    box(xs[2], y, w, h, "Diff parser", "files and hunks")
    box(xs[3], y, w, h, "Sift model", "ModernBERT cross-\nencoder, 3 heads", fc="#B6114D", ec="#B6114D", tc="white")
    box(xs[4], y, w, h, "Calibrate + rank", "T = 1.275,\nk-means cluster")
    box(xs[5], y, 14.5, h, "Review UI", "FastAPI + React,\nIG evidence")
    for i in range(5):
        arrow(xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.text(0.5, 24.3, "MODEL ZOO (switchable in the UI) AND ANALYSIS", fontsize=8.5, fontweight="bold", color="#B6114D")
    # heads under model
    ax.add_patch(FancyBboxPatch((50.0, 12.2), 49.5, 10.4, boxstyle="round,pad=0,rounding_size=0.8", fc="#FBE3EA", ec="#B6114D", lw=1.2))
    ax.text(74.7, 21.1, "Three heads on pooled [CLS]", ha="center", fontsize=9, fontweight="bold", color="#B6114D")
    for i, (t, s) in enumerate((("Alignment", "focal loss"), ("Drift type", "3-way"), ("Confidence", "calibration"))):
        box(51.0 + i * 16.2, 13.1, 15.2, 6.4, t, s, fc="white", ec="#B6114D", tfs=8.8, sfs=8)
    arrow(58.5, y, 58.5, 22.7)
    box(0.5, 12.2, 23.5, 10.4, "Models compared", "main ModernBERT-base, MiniLM\ncross-encoder, CodeBERT,\nBiLSTM, B1-B5, frozen-embedding\nGAN-style baseline", tfs=9.5, sfs=8.4)
    box(25.2, 12.2, 23.5, 10.4, "Student + INT8 ONNX", "distilled MiniLM-L6\n(tau 4, alpha 0.5)\n22.9 MB, CPU p95 66.9 ms", tfs=9, sfs=8)

    ax.text(0.5, 9.6, "OFFLINE PIPELINE", fontsize=8.5, fontweight="bold", color="#404040")
    labels = [("Sift-IA v1.0", "18,000 pairs,\nrepo-disjoint"), ("Denoise", "rules + confident\nlearning"), ("Train", "multi-task,\nfocal loss"),
              ("Calibrate", "temperature\nscaling"), ("Distil + INT8", "MiniLM student,\nONNX"), ("Analyse", "k-means, GAN-style,\nIntegrated Gradients")]
    for i, (t, s) in enumerate(labels):
        box(xs[i], 0.4, w, 8.4, t, s, fc="#EEEEEE", ec="#8C8C8C", ls="--", tfs=9.5, sfs=8.6)
        if i < 5:
            arrow(xs[i] + w, 4.6, xs[i + 1], 4.6, c="#595959")
    return save(fig, "architecture.png")


# ---------------------------------------------------------------- screenshots
def crop_shot(src, name, w=1440, h=810, top=0, left=0):
    im = Image.open(os.path.join(ROOT, src)).convert("RGB")
    im = im.crop((left, top, min(left + w, im.width), min(top + h, im.height)))
    p = os.path.join(ASSETS, name)
    im.save(p)
    return p


# ---------------------------------------------------------------- pptx helpers
def set_run(r, size=12, bold=False, italic=False, color=DARK, font="Calibri"):
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.name = font
    r.font.color.rgb = color


def add_runs(p, text, size, color=DARK, bold=False, italic=False, font="Calibri", link=None):
    """'**bold**' markup inside text."""
    parts = text.split("**")
    for i, seg in enumerate(parts):
        if not seg:
            continue
        r = p.add_run()
        r.text = seg
        set_run(r, size, bold or (i % 2 == 1), italic, color, font)
        if link:
            r.hyperlink.address = link


def bullet(p, indent=0.16):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(indent))))
    pPr.set("indent", str(-int(Inches(indent))))
    for tag in ("a:buNone", "a:buChar"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    bf = etree.SubElement(pPr, qn("a:buFont"))
    bf.set("typeface", "Arial")
    bc = etree.SubElement(pPr, qn("a:buChar"))
    bc.set("char", "•")


def tb(slide, x, y, w, h, paras, size=12, color=DARK, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, space=4, fill=None):
    """paras: list of str | dict(text, bullet, size, bold, color, italic, align, link, font, space)."""
    shp = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is not None:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    first = True
    for it in paras:
        d = {"text": it} if isinstance(it, str) else it
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = d.get("align", align)
        p.space_after = Pt(d.get("space", space))
        add_runs(p, d["text"], d.get("size", size), d.get("color", color), d.get("bold", False),
                 d.get("italic", False), d.get("font", "Calibri"), d.get("link"))
        if d.get("bullet"):
            bullet(p)
    return shp


def card(slide, x, y, w, h, head, paras, size=11.5, headsize=13, fill=CARD):
    r = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    r.fill.solid()
    r.fill.fore_color.rgb = fill
    r.line.fill.background()
    r.shadow.inherit = False
    items = [{"text": head, "bold": True, "color": RED, "size": headsize, "space": 3}] if head else []
    items += paras
    tb(slide, x + 0.05, y + 0.04, w - 0.1, h - 0.08, items, size=size)
    return r


def tile(slide, x, y, w, h, big, small, bigsize=26):
    r = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    r.fill.solid()
    r.fill.fore_color.rgb = CARD
    r.line.fill.background()
    r.shadow.inherit = False
    tb(slide, x, y + 0.03, w, h * 0.5, [{"text": big, "bold": True, "color": RED, "size": bigsize, "align": PP_ALIGN.CENTER,
                                           "font": "Cambria", "space": 0}], anchor=MSO_ANCHOR.MIDDLE)
    tb(slide, x + 0.05, y + h * 0.5, w - 0.1, h * 0.48, [{"text": small, "size": 11, "align": PP_ALIGN.CENTER, "color": GREY}],
       anchor=MSO_ANCHOR.TOP)


def pic(slide, path, x, y, w, h, align="c"):
    im = Image.open(path)
    ar = im.width / im.height
    if w / h > ar:
        ph, pw = h, h * ar
    else:
        pw, ph = w, w / ar
    px = x + (w - pw) / 2 if align == "c" else x
    py = y + (h - ph) / 2 if align == "c" else y
    return slide.shapes.add_picture(path, Inches(px), Inches(py), Inches(pw), Inches(ph))


def _border(tcPr):
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = etree.Element(qn(tag), w="9525", cap="flat", cmpd="sng")
        sf = etree.SubElement(ln, qn("a:solidFill"))
        etree.SubElement(sf, qn("a:srgbClr")).set("val", BORDER)
        etree.SubElement(ln, qn("a:prstDash")).set("val", "solid")
        yield ln


def table(slide, x, y, w, colw, rows, font=11, row_h=0.27, head_h=None, hilite=(), bold_cells=(), aligns=None, hfont=None):
    """rows[0] is the header. colw are ratios. cell may be str or (str, dict)."""
    nr, nc = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    tblPr.set("firstRow", "1")
    tblPr.set("bandRow", "0")
    tot = float(sum(colw))
    for i, c in enumerate(colw):
        tbl.columns[i].width = Inches(w * c / tot)
    for ri in range(nr):
        tbl.rows[ri].height = Inches(head_h if (ri == 0 and head_h) else row_h)
        for ci in range(nc):
            cell = tbl.cell(ri, ci)
            v = rows[ri][ci]
            opts = {}
            if isinstance(v, tuple):
                v, opts = v
            cell.margin_left = cell.margin_right = Inches(0.05)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            al = (aligns[ci] if aligns else "l")
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[opts.get("align", al)]
            if ri == 0:
                add_runs(p, str(v), hfont or font, WHITE, True)
                bg = CRIM
            else:
                add_runs(p, str(v), font, DARK, opts.get("bold", (ri, ci) in bold_cells or ri in ()))
                bg = HILITE if ri in hilite else WHITE
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            tcPr = cell._tc.get_or_add_tcPr()
            for k, ln in enumerate(_border(tcPr)):
                tcPr.insert(k, ln)
    return gf


# ---------------------------------------------------------------- deck scaffolding
prs = Presentation(TEMPLATE)
SW, SH = prs.slide_width, prs.slide_height
tpl_slide = prs.slides[0]
WORDMARK = None
for shp in tpl_slide.shapes:
    if shp.shape_type == 13 and shp.width == 1440180:
        WORDMARK = shp.image.blob
assert WORDMARK
TOTAL_SLIDES = [32]
BLANK = prs.slide_layouts[10]


def new_slide(title, notes, size=22):
    s = prs.slides.add_slide(BLANK)
    CUR["slide"] = len(prs.slides)
    t = s.shapes.add_textbox(Emu(411491), Emu(150000), Emu(8400000), Emu(480060))
    tf = t.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(34275)
    tf.margin_top = tf.margin_bottom = Emu(20575)
    add_runs(tf.paragraphs[0], title, size, RED, True, False, "Cambria")
    line = s.shapes.add_shape(1, 0, Emu(4574286), Emu(9144015), Emu(34290))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(0, 0, 0)
    line.line.fill.background()
    band = s.shapes.add_shape(1, 0, Emu(4608576), Emu(9144015), Emu(534924))
    band.fill.solid()
    band.fill.fore_color.rgb = RGBColor(0xB6, 0x11, 0x4D)
    band.line.fill.background()
    for sh in (line, band):
        sh.shadow.inherit = False
    s.shapes.add_picture(io.BytesIO(WORDMARK), Emu(7441128), Emu(4677156), Emu(1440180), Emu(458881))
    tb(s, 0.3, 5.14, 5.0, 0.35, [{"text": "Sift  |  Review 3  |  23CSE473 NNDL", "size": 11, "color": WHITE}], anchor=MSO_ANCHOR.MIDDLE)
    tb(s, 6.9, 5.14, 1.1, 0.35, [{"text": f"{CUR['slide']}", "size": 11, "color": WHITE, "align": PP_ALIGN.RIGHT, "bold": True}],
       anchor=MSO_ANCHOR.MIDDLE)
    s.notes_slide.notes_text_frame.text = notes
    return s


def takeaway(s, text, y=4.50, h=0.48):
    tb(s, 0.35, y, 9.3, h, [{"text": text, "size": 12, "italic": True, "color": MAROON}], anchor=MSO_ANCHOR.MIDDLE)


# ================================================================ build slides
def build():
    A = {k: v() for k, v in (("tier", chart_tier_auc), ("abl", chart_ablation), ("noise", chart_noise), ("faith", chart_faith),
                             ("cl", chart_cl), ("gan", chart_gan), ("gancurve", chart_gan_curves), ("cal", chart_cal),
                             ("lat", chart_latency), ("arch", chart_arch))}
    R = os.path.join(ROOT, "results")
    SS = os.path.join(ROOT, "docs", "screenshots")

    # ---- slide 1: template title slide, edited in place
    CUR["slide"] = 1
    for shp in tpl_slide.shapes:
        if shp.name == "Google Shape;130;p25":
            ps = shp.text_frame.paragraphs
            # paragraph 0: 'Sift' <br> 'Intent–Hunk Alignment Verification'  -> 'Sift: Intent–Hunk Alignment Verification'
            r0 = ps[0].runs
            r0[0].text = "Sift: Intent–Hunk Alignment Verification"
            r1 = ps[0]._p.findall(qn("a:r"))
            ps[0]._p.remove(r1[1])
            br = ps[0]._p.find(qn("a:br"))
            ps[0]._p.remove(br)
            shp.left, shp.width = Emu(480000), Emu(7400000)
            newp = copy.deepcopy(ps[1]._p)
            ps[1]._p.addnext(newp)
            rr = newp.find(qn("a:r"))
            rr.find(qn("a:t")).text = "Review 3"
            rr.find(qn("a:rPr")).set("sz", "1800")
            sf = rr.find(qn("a:rPr")).find(qn("a:solidFill"))
            sf.find(qn("a:srgbClr")).set("val", "B6114D")
        if shp.name == "Google Shape;141;p25":
            shp.text_frame.paragraphs[0].runs[0].text = "This presentation is prepared for Course"
            shp.text_frame.paragraphs[1].runs[0].text = "23CSE473 – Neural Networks & Deep Learning  |  Review 3"
    tpl_slide.notes_slide.notes_text_frame.text = (
        "Introduce the project: Sift is a local verification layer that checks whether each diff hunk written by an AI coding agent "
        "serves the developer's stated intent. This is Review 3, so the deck covers the base paper, dataset status, the trained model, "
        "all experiments, the working application and the tests. Team 24 is a single-member team; the faculty guide is Dr. T. Senthil Kumar.")
    # drop template slides 2-5
    lst = prs.slides._sldIdLst
    for sid in list(lst)[1:]:
        prs.part.drop_rel(sid.rId)
        lst.remove(sid)

    # ---- 2 abstract
    s = new_slide("Abstract: what Sift does and what was measured",
                  "Sift consolidates a multi-turn coding session into one intent, scores every diff hunk as aligned or misaligned, "
                  "types the drift, calibrates the confidence and ranks hunks for review. The four tiles are test-split numbers from "
                  "results/master_comparison.csv and results/latency_size_onnx.json. Macro-F1 is honestly below logistic regression; the gain is in ranking.")
    tb(s, 0.35, 0.8, 4.55, 3.1, [
        {"text": "Sift is a local verification layer between an AI coding agent and the developer.", "bold": True, "size": 13, "space": 6},
        {"text": "It consolidates the session's chat turns into one intent, labels each diff hunk **Aligned** or **Misaligned** "
                 "(Collateral, Scope-Expansion, Purpose-Drift), calibrates the confidence, explains the verdict and ranks hunks for review.", "size": 12, "space": 6},
        {"text": "A ModernBERT-base cross-encoder is compared with five non-neural baselines, three neural baselines and a GAN-style "
                 "baseline on repository-disjoint splits of Sift-IA v1.0.", "size": 12},
    ])
    b4h = V("B4 hard AUC", MC["B4 TF-IDF dissimilarity"]["auc_hard"], MCS)
    mh = V("Main hard AUC", MC["Main: ModernBERT-base"]["auc_hard"], MCS)
    mn = V("Main NDCG@5", MC["Main: ModernBERT-base"]["ndcg@5"], MCS)
    b1n = V("B1 NDCG@5", MC["B1 Random order"]["ndcg@5"], MCS)
    i8 = LATO["student_onnx_int8"]
    sz = V("INT8 size MB", i8["size_mb"], "results/latency_size_onnx.json")
    p95 = V("INT8 p95 ms", i8["latency_ms"]["p95_ms"], "results/latency_size_onnx.json")
    nh = V("hunks", sum(int(DS[f"{sp}.hunks"]) for sp in ("train", "val", "test")), DSS)
    tile(s, 5.1, 0.85, 2.2, 1.3, f"{f3(mh)}", f"hard-tier pairwise AUC (B4/B5: {f3(b4h)})")
    tile(s, 7.45, 0.85, 2.2, 1.3, f"{f3(mn)}", f"NDCG@5 (random order: {f3(b1n)})")
    tile(s, 5.1, 2.3, 2.2, 1.3, f"{f1(sz)} MB", f"INT8 ONNX student, CPU p95 {f1(p95)} ms")
    tile(s, 7.45, 2.3, 2.2, 1.3, ni(nh), "hunk-intent pairs, 15 repositories")
    card(s, 0.35, 3.95, 9.3, 0.55, None, [{"text": "Honest weak spot: main macro-F1 " + f3(MC["Main: ModernBERT-base"]["macro_f1"]) +
                                           " is below B5 logistic regression (" + f3(MC["B5 Logistic regression"]["macro_f1"]) + "); the advantage is ranking on the hard tier.",
                                           "size": 12, "color": MAROON, "bold": True}])
    takeaway(s, "All numbers: test split, single seed 473, labels constructed (no human validation).", y=4.52, h=0.4)

    # ---- 3 introduction / problem
    s = new_slide("Introduction: the review bottleneck",
                  "AI coding agents now produce multi-file diffs faster than anyone can read them, so the question becomes whether every hunk serves what the developer asked. "
                  "Diff-level scalars like lines changed are the same for all hunks in a diff, so ordering by them equals random: NDCG@5 0.7236 from baselines.json. "
                  "The drift-type counts are the sums of the three splits in dataset_statistics.csv.")
    b2 = V("B2 NDCG@5", BASE["ndcg@5"]["B2 Diff lines changed"], "experiments/results/baselines.json")
    cards = [
        ("The problem", ["Agents emit large multi-file diffs faster than a developer can review them.",
                         "Code that compiles and passes tests can still contain collateral edits, unrequested features or off-purpose changes."]),
        ("Why scalar signals fail", [f"Lines changed or files touched are identical for every hunk in one diff.",
                                     f"Ordering hunks by diff lines equals random order (NDCG@5 {f3(b2)})."]),
        ("Objective", ["Consolidate the session intent, label every hunk Aligned or Misaligned, type the drift.",
                       "Rank hunks, with calibrated confidence and an explanation."]),
    ]
    for i, (h_, ps) in enumerate(cards):
        card(s, 0.35 + i * 3.12, 0.85, 3.0, 2.05, h_, [{"text": t, "bullet": True, "size": 12.5} for t in ps])
    col = V("drift counts", [sum(int(DS[f"{sp}.drift.{d}"]) for sp in ("train", "val", "test")) for d in ("collateral", "scope_expansion", "purpose_drift")], DSS)
    tb(s, 0.35, 3.0, 9.3, 0.3, [{"text": "Three drift types of a misaligned hunk (counts in Sift-IA v1.0)", "bold": True, "color": RED, "size": 13}])
    defs = [("Collateral", ni(col[0]), "unrequested change next to the requested one, in a different file"),
            ("Scope-Expansion", ni(col[1]), "unrequested feature or behaviour in a file the session already edits"),
            ("Purpose-Drift", ni(col[2]), "serves a different purpose: the requested change is absent")]
    for i, (n, c, d) in enumerate(defs):
        card(s, 0.35 + i * 3.12, 3.33, 3.0, 1.1, f"{n}: {c}", [{"text": d, "size": 12}], headsize=12.5)
    takeaway(s, "Why deep learning: judging one hunk against an intent needs both texts read jointly, which a cross-encoder does.", y=4.5, h=0.45)

    # ---- 4 related work table
    s = new_slide("Related work: five journal papers and the base paper",
                  "These are the five SCImago-indexed journal papers from the Review 2 literature survey. All treat natural-language to code consistency, "
                  "and none conditions on a multi-turn developer intent at hunk level. The last row is the base paper chosen for Review 3, covered on the next two slides.")
    rows = [["#", "Paper, venue, year", "Method and finding", "Relation to Sift (gap)"],
            ["1", "Xu et al., Code Comment Inconsistency Detection Based on Confidence Learning. IEEE TSE 50(3), 2024",
             "MCCL: remove dataset noise by confidence learning, then detect inconsistent pairs", "Motivates our denoising stage. Gap: method-comment pairs, not hunks"],
            ["2", "Huang et al., Are your comments outdated? J. Softw. Evol. Proc. 37(1), 2025",
             "CoCC: supervised classifier for outdated comments", "Validates supervised pair classification. Gap: no calibration or ranking"],
            ["3", "Yang et al., RComSync. Empirical Software Engineering 31(4), 2026",
             "In-context learning plus reranking for comment sync", "Supports ranking. Gap: cloud LLM, not local"],
            ["4", "Fan et al., Exploring LLMs for Code-Change-Related Tasks. ACM TOSEM 34(6), 2025",
             "LLMs with in-context learning vs PEFT; bigger is not always better", "Justifies a compact fine-tuned encoder"],
            ["5", "Wang et al., Best practices for code pre-trained models on SE classification. Expert Syst. Appl., 2025",
             "Embedding vs fine-tuning of code PTMs", "Informs fine-tuning. Gap: single-input, not pair classification"],
            ["B", "Northcutt, Jiang & Chuang, Confident Learning. JAIR, 2021 (base paper)",
             "Prune, count, rank to find label errors; model-agnostic", "Used to find the 569 known label flips"]]
    table(s, 0.35, 0.82, 9.3, [0.3, 3.4, 3.0, 2.9], rows, font=11, row_h=0.5, head_h=0.3, hilite=(6,), aligns=["c", "l", "l", "l"])
    takeaway(s, "Gap: no paper pairs a developer's multi-turn intent with individual diff hunks.", y=4.52, h=0.4)

    # ---- 5 base paper
    s = new_slide("Base paper: Confident Learning (JAIR 2021)",
                  "The standard paper is Northcutt, Jiang and Chuang, Confident Learning, JAIR 2021. Its key idea is to estimate the joint distribution of noisy and true labels "
                  "using per-class thresholds, then prune and rank examples; it is model-agnostic and open-sourced as cleanlab. The CIFAR-10 accuracies are copied from Table 2 of the paper (ppt/standard_paper.pdf).")
    card(s, 0.35, 0.82, 4.5, 1.15, "What it proposes", [
        {"text": "A data-centric framework that finds label errors by estimating the joint distribution of noisy and true labels under class-conditional noise. Model-agnostic; open-source as cleanlab.", "size": 11}])
    tb(s, 0.35, 2.02, 4.5, 0.3, [{"text": "Key idea: prune, count, rank", "bold": True, "color": RED, "size": 13}])
    steps = [("Count", "per-class threshold t_j = mean self-confidence; confident joint C[ỹ,y*] estimates the noise transition matrix"),
             ("Prune", "remove examples likely mislabelled (prune by class or by noise rate)"),
             ("Rank", "order by self-confidence p(ỹ=i; x); train on the cleaned set")]
    for i, (a, b) in enumerate(steps):
        card(s, 0.35, 2.33 + i * 0.7, 4.5, 0.66, None, [{"text": f"**{a}:** {b}", "size": 11}])
    tb(s, 5.05, 0.8, 4.6, 0.3, [{"text": "Reported results (paper, CIFAR-10 test accuracy %)", "bold": True, "color": RED, "size": 12.5}])
    t2 = {"cl20": 91.1, "bl20": 78.4, "ct20": 81.2, "in20": 87.8, "cl40": 86.7, "bl40": 60.2, "ct40": 62.9, "in40": 84.4}
    for k, v in t2.items():
        V("paper Table 2 " + k, v, "ppt/standard_paper.pdf Table 2 (sparsity 0)")
    rows = [["Method", "20% noise", "40% noise"], ["Confident learning (C_ỹ,y*)", "91.1", "86.7"], ["INCV", "87.8", "84.4"],
            ["Co-Teaching", "81.2", "62.9"], ["Baseline", "78.4", "60.2"]]
    table(s, 5.05, 1.12, 4.6, [2.4, 1, 1], rows, font=11, row_h=0.29, hilite=(1,), aligns=["l", "c", "c"])
    tb(s, 5.05, 2.65, 4.6, 1.8, [
        {"text": "Sparsity 0; mean of ten trials in the paper.", "size": 11, "italic": True, "color": GREY},
        {"text": "Also finds real label errors in ImageNet (e.g. 645 missile images labelled as the parent class projectile), MNIST and Amazon Reviews.", "bullet": True, "size": 11.5},
        {"text": "Assumes class-conditional noise; proves conditions under which it exactly finds label errors.", "bullet": True, "size": 11.5},
    ])
    takeaway(s, "Northcutt, Curtis G.; Jiang, Lu; Chuang, Isaac L. Journal of Artificial Intelligence Research, 2021.", y=4.55, h=0.4)

    # ---- 6 use in Sift
    d = CL["detection"]["confident_learning"]
    fd = CL["detection"]["final drop list (CL + rules)"]
    s = new_slide("How Sift uses it: finding the 569 label flips",
                  "Sift-IA's training labels carry 569 deliberately flipped rows with ground truth, so confident learning can be scored directly. cleanlab was run on out-of-fold probabilities "
                  "from the cross-encoder. It flagged 1,470 rows; 220 are true flips, giving precision 0.150 and recall 0.387, about three times the random precision 0.045. It is a weak detector, and we say so.")
    V("CL flagged/tp", [d["flagged"], d["tp"]], "results/confident_learning.json")
    V("CL precision/recall", [d["precision"], d["recall"]], "results/confident_learning.json")
    V("final drop list", [fd["flagged"], fd["precision"], fd["recall"]], "results/confident_learning.json")
    V("random precision", CL["random_precision"], "results/confident_learning.json")
    V("rules removed", CL["rules"], "results/confident_learning.json")
    pic(s, A["cl"], 5.0, 0.82, 4.7, 3.55)
    tb(s, 0.35, 0.8, 4.6, 3.7, [
        {"text": "Procedure", "bold": True, "color": RED, "size": 13},
        {"text": f"Out-of-fold probabilities for all {ni(CL['n_train'])} training rows from the cross-encoder.", "bullet": True},
        {"text": "cleanlab confident learning flags suspected label errors, scored against the known flips.", "bullet": True},
        {"text": "Rules remove non-semantic and duplicate hunks first.", "bullet": True},
        {"text": "Result", "bold": True, "color": RED, "size": 13},
        {"text": f"Flagged {ni(d['flagged'])}, true flips found {d['tp']}: **precision {f3(d['precision'])}, recall {f3(d['recall'])}**.", "bullet": True},
        {"text": f"Random flagging would give precision {f3(CL['random_precision'])}.", "bullet": True},
        {"text": f"Final drop list (CL + rules): precision {f3(fd['precision'])}, recall {f3(fd['recall'])}.", "bullet": True},
    ], size=11.5)
    takeaway(s, "Better than random, but a weak detector: most flagged rows are not true flips.", y=4.5, h=0.45)

    # ---- 7 related model papers
    s = new_slide("Related model papers: what Sift reuses",
                  "Every component of Sift is an existing method; the contribution is the task framing, the pipeline and the dataset. The first three rows are the closest research area (code-comment inconsistency and tangled changes). "
                  "Co-teaching is named in the design and a helper exists, but it was not run. The note: published numbers are not comparable with ours because tasks and data differ.")
    rows = [["Paper", "Role in Sift"],
            ["Panthaplackel et al., Deep Just-In-Time Inconsistency Detection, AAAI 2021", "Closest prior task: comment vs code change"],
            ["Herzig & Zeller, The Impact of Tangled Code Changes, MSR 2013", "Motivates our limitation: seed commits may be tangled"],
            ["Nogueira & Cho, Passage Re-ranking with BERT, 2019", "Cross-encoder pattern: concatenate texts, classify from [CLS]"],
            ["Warner et al., ModernBERT, 2024", "Encoder backbone (8k context), fine-tuned"],
            ["Feng et al., CodeBERT, EMNLP Findings 2020", "Code-aware baseline (reduced training)"],
            ["Lin et al., Focal Loss for Dense Object Detection, ICCV 2017", "Alignment loss for the 78:22 class ratio"],
            ["Guo et al., On Calibration of Modern Neural Networks, ICML 2017", "Temperature scaling on validation"],
            ["Hinton et al., Distilling the Knowledge in a Neural Network, 2015", "Distillation to the MiniLM student"],
            ["Sundararajan et al., Axiomatic Attribution (Integrated Gradients), ICML 2017", "Token attributions for explanations"],
            ["Han et al., Co-teaching, NeurIPS 2018", "Noisy-label training; named, not run"]]
    table(s, 0.35, 0.78, 9.3, [5.2, 4.1], rows, font=11, row_h=0.31, head_h=0.28)
    takeaway(s, "No new architecture is claimed; published results on other tasks are not comparable with ours.", y=4.52, h=0.4)

    # ---- 8 architecture
    s = new_slide("Proposed method: system architecture",
                  "Top row is the runtime path: session, consolidation, diff parsing, the cross-encoder with three heads, calibration and ranking, and the review UI. "
                  "The middle row shows the compared models and the distilled INT8 student. The bottom row is the offline pipeline that produced everything in results/. "
                  "Calibration temperature 1.275 and the INT8 numbers come from calibration.json and latency_size_onnx.json.")
    V("temperature", CAL["temperature"], "results/calibration.json")
    pic(s, A["arch"], 0.35, 0.78, 9.3, 3.62)
    takeaway(s, "Everything runs locally; the UI can switch between the main model, the INT8 student and the baselines.", y=4.5, h=0.45)

    # ---- 9 model, loss, heads
    cfg = RUNCFG
    hp = {r["run"]: r for r in CSV("results/hyperparameters.csv")}["full"]
    best = V("best val macro-F1", float(hp["best_val_macro_f1"]), "results/hyperparameters.csv (run full)")
    V("full run train seconds / gpu mem", [hp["train_time_s"], hp["peak_gpu_mem_gb"]], "results/hyperparameters.csv")
    s = new_slide("Model, loss and heads",
                  "The input is intent, separator, hunk, up to 512 tokens. The pooled vector feeds three heads: alignment, drift type masked to misaligned rows, and a confidence head. "
                  "Loss weights are 0.6, 0.3 and 0.1 with focal loss, gamma 2, on the alignment term. Training took about 28 minutes on the 6 GB RTX 4050; "
                  "the settings table is from experiments/runs/full/config.json and results/hyperparameters.csv.")
    card(s, 0.35, 0.82, 4.55, 1.55, "Cross-encoder (ModernBERT-base)", [
        {"text": "Input: [CLS] consolidated intent [SEP] hunk", "bullet": True, "size": 11.5},
        {"text": f"{cfg['params'] / 1e6:.1f} M parameters, 22 layers, 768 hidden, 12 heads", "bullet": True, "size": 11.5},
        {"text": "Pooled [CLS] feeds three heads", "bullet": True, "size": 11.5}])
    V("params", cfg["params"], "experiments/runs/full/config.json")
    card(s, 0.35, 2.45, 4.55, 2.0, "Heads and loss", [
        {"text": "Alignment: 2-class, focal loss (gamma = 2)", "bullet": True, "size": 11.5},
        {"text": "Drift type: 3-class, masked to misaligned rows", "bullet": True, "size": 11.5},
        {"text": "Confidence: calibration head; temperature scaling afterwards", "bullet": True, "size": 11.5},
        {"text": "L = 0.6 L_align + 0.3 L_drift + 0.1 L_cal", "bold": True, "size": 12.5, "color": MAROON},
        {"text": "FL = -alpha (1 - p_t)^gamma log p_t", "size": 11.5, "italic": True}])
    rows = [["Setting", "Value"], ["Max length / batch", f"{cfg['max_len']} / {cfg['micro_bs']} x {cfg['accum']} (effective {cfg['effective_batch']})"],
            ["Optimiser, lr", f"AdamW, {cfg['lr']:g}, cosine, 10% warm-up"],
            ["Precision", "bf16, gradient checkpointing"], ["Epochs", f"up to {cfg['epochs']}, early stop (patience {cfg['patience']})"],
            ["Training rows", f"{ni(cfg['n_train_used'])} of {ni(cfg['n_train_raw'])} after denoising"],
            ["Best val macro-F1", f3(best)], ["Train time", f"{float(hp['train_time_s']) / 60:.0f} min, peak {float(hp['peak_gpu_mem_gb']):.1f} GB"],
            ["Hardware", "RTX 4050 Laptop, 6 GB"], ["Seed", f"{cfg['seed']} (single seed)"]]
    for k in ("max_len", "micro_bs", "accum", "lr", "epochs", "patience", "n_train_used", "n_train_raw", "seed"):
        V("config " + k, cfg[k], "experiments/runs/full/config.json")
    table(s, 5.05, 0.82, 4.6, [1.7, 2.9], rows, font=11, row_h=0.355)
    takeaway(s, "Known method, new task: ModernBERT, focal loss and temperature scaling are existing techniques.", y=4.52, h=0.4)

    # ---- 10 dataset
    s = new_slide("Dataset: Sift-IA v1.0",
                  "Sift-IA v1.0 has 18,000 hunk-intent pairs from 15 open-source repositories, split by repository so no repository appears in two splits. "
                  "Aligned hunks are real hunks from the seed commit; a misaligned hunk is a donor hunk from a different commit inserted into the session, so labels are determined by construction, not by a human. "
                  "Language counts are the measured ones from the released files (they differ slightly from the planning figures).")
    spl = []
    for sp, nm in (("train", "Train"), ("val", "Validation"), ("test", "Test")):
        reps = json.loads(DS[f"{sp}.repos"])
        al = int(DS[f"{sp}.aligned_clean"])
        mis = int(DS[f"{sp}.misaligned_clean"])
        V(f"split {sp}", [len(reps), DS[f"{sp}.sessions"], DS[f"{sp}.hunks"], al, mis], DSS)
        spl.append([nm, str(len(reps)), ni(DS[f"{sp}.sessions"]), ni(DS[f"{sp}.hunks"]), ni(al), ni(mis)])
    tot = [sum(int(r[k].replace(",", "")) for r in spl) for k in (1, 2, 3, 4, 5)]
    spl.append(["Total"] + [ni(t) for t in tot])
    table(s, 0.35, 0.82, 5.4, [1.3, 0.7, 0.9, 0.9, 0.9, 1.0],
          [["Split", "Repos", "Sessions", "Hunks", "Aligned", "Misaligned"]] + spl, font=11, row_h=0.29, aligns=["l", "c", "c", "c", "c", "c"], hilite=(4,))
    tb(s, 0.35, 2.3, 5.4, 2.2, [
        {"text": "Repositories", "bold": True, "color": RED, "size": 12.5, "space": 2},
        {"text": "Train: " + ", ".join(json.loads(DS["train.repos"])), "size": 11, "space": 1},
        {"text": "Validation: " + ", ".join(json.loads(DS["val.repos"])), "size": 11, "space": 1},
        {"text": "Test: " + ", ".join(json.loads(DS["test.repos"])), "size": 11, "space": 5},
        {"text": "Measured language mix (hunks)", "bold": True, "color": RED, "size": 12.5, "space": 2},
        {"text": "Python " + ni(sum(int(DS[f"{sp}.languages.Python"]) for sp in ("train", "val", "test"))) +
                 ", JavaScript " + ni(sum(int(DS[f"{sp}.languages.JavaScript"]) for sp in ("train", "val", "test"))) +
                 ", TypeScript " + ni(DS["val.languages.TypeScript"]) + " (val only; test has none)", "size": 11},
    ])
    card(s, 5.95, 0.82, 3.7, 3.6, "How labels are built", [
        {"text": "**Aligned:** a real hunk from the seed commit that produced the intent.", "bullet": True, "size": 12},
        {"text": "**Misaligned:** a donor hunk from a different commit inserted into the session.", "bullet": True, "size": 12},
        {"text": "Collateral: donor touches a different file. Scope-Expansion: same file, session hunks kept. Purpose-Drift: same file, own hunk removed.", "bullet": True, "size": 12},
        {"text": "Difficulty tiers (misaligned): random " + ni(sum(int(DS[f"{sp}.difficulty.random"]) for sp in ("train", "val", "test"))) +
                 ", hard " + ni(sum(int(DS[f"{sp}.difficulty.hard"]) for sp in ("train", "val", "test"))), "bullet": True, "size": 12},
        {"text": "Train on label, evaluate on label_clean; val and test labels are clean.", "bullet": True, "size": 12},
    ])
    takeaway(s, "Limit: constructed labels, no human validation yet; only Python and JavaScript in test.", y=4.5, h=0.45)

    # ---- 11 IEEE Dataport
    link = "https://ieee-dataport.org/documents/sift-ia-intent-implementation-alignment-dataset-code-diff-hunks-produced-ai-coding-agents"
    nrow = V("rows with noise", DS["rows_with_any_noise"], DSS)
    s = new_slide("IEEE DataPort: dataset record and status",
                  "The dataset record link is listed in the submission tracker. A dataset description document for Sift-IA v1.0 has been prepared following the IEEE DataPort template; "
                  "that document itself says the DOI will be inserted once the upload is published, so we do not claim the dataset is published or has a DOI. "
                  "The 900-sample gold annotation sheet has labels withheld; human validation is future work.")
    card(s, 0.35, 0.82, 9.3, 0.85, "IEEE DataPort record", [
        {"text": link, "size": 10.5, "color": CRIM, "link": link}], headsize=12.5)
    card(s, 0.35, 1.78, 4.55, 2.7, "Status (supported by our records)", [
        {"text": "Record link is listed in the project submission tracker.", "bullet": True, "size": 12.5},
        {"text": "Dataset description document prepared (IEEE DataPort template).", "bullet": True, "size": 12.5},
        {"text": "DOI to be inserted once the upload is published; **not yet claimed**.", "bullet": True, "size": 12.5},
        {"text": "Licences: MIT, BSD-3-Clause, Apache-2.0 sources; licence texts included.", "bullet": True, "size": 12.5}])
    card(s, 5.05, 1.78, 4.6, 2.7, "Release contents (per the description)", [
        {"text": "Three CSV splits (12,600 / 2,700 / 2,700 rows) and a JSONL copy.", "bullet": True, "size": 12.5},
        {"text": f"Noise manifest: {ni(nrow)} rows with ground-truth noise labels.", "bullet": True, "size": 12.5},
        {"text": "sessions.csv with 6,111 sessions and source repositories file.", "bullet": True, "size": 12.5},
        {"text": "Blind 900-sample gold annotation sheet (labels withheld).", "bullet": True, "size": 12.5},
        {"text": "Four scripts that regenerate the dataset from a fixed seed.", "bullet": True, "size": 12.5}])
    takeaway(s, "Doubles as a denoising benchmark: five noise types shipped with ground truth.", y=4.52, h=0.4)

    # ---- 12 noise & denoising
    s = new_slide("Noise analysis and denoising",
                  "Five noise types are shipped with ground truth, covering 4,529 rows or about a quarter of the data. Rules handle non-semantic and duplicate hunks, consolidation handles intent-turn noise, and confident learning targets label flips. "
                  "The robustness study injects extra label noise on a 3,000-row subset for two epochs: a reduced, single-seed setting, so treat the curve as indicative. Denoising helps at 20 and 30 percent but not at 0.")
    nt = [("Intent-turn", "intent_turn", "consolidation drops chit-chat, retractions, repeats"),
          ("Duplicate hunk", "duplicate_hunk", "rule filter"), ("Non-semantic", "non_semantic_hunk", "rule filter (whitespace, lockfile, version bump)"),
          ("Label flip (train only)", "label_flip", "confident learning"), ("Truncation", "truncation", "flagged as exceeds_max_len")]
    rows = [["Noise type", "Rows", "Handling"]] + [[a, ni(V("noise " + k, DS["noise_rows." + k], DSS)), c] for a, k, c in nt]
    table(s, 0.35, 0.82, 4.75, [1.6, 0.6, 2.6], rows, font=11, row_h=0.46, head_h=0.3, aligns=["l", "c", "l"])
    tb(s, 0.35, 3.45, 4.75, 1.05, [
        {"text": f"{ni(nrow)} of {ni(nh)} rows ({100 * int(nrow) / int(nh):.1f}%) carry at least one noise type.", "size": 11.5, "bullet": True},
        {"text": f"Rules removed {ni(CL['rules']['removed'])} rows; confident learning flagged {ni(CL['detection']['confident_learning']['flagged'])}.", "size": 11.5, "bullet": True}])
    pic(s, A["noise"], 5.25, 0.82, 4.4, 3.6)
    nd = {(r['arm'], r['noise_rate']): r['test']['macro_f1'] for r in NZ['results']}
    takeaway(s, f"At 20% noise: denoise {f3(nd[('denoise', 0.2)])} vs {f3(nd[('no_denoise', 0.2)])}. Reduced setting, one seed, not monotonic.", y=4.5, h=0.45)

    # ---- 13 k-means
    kq = KM["clustering_quality"]["test"]
    ks = KM["k_scan"]
    sil = ks["silhouette"][ks["k"].index(KM["k_chosen"])]
    hy = KM["hybrid"]
    d_ = hy["delta_cluster_minus_baseline"]
    V("kmeans", {"k": KM["k_chosen"], "silhouette": sil, "purity_test": kq["purity_vs_alignment"], "majority": kq["majority_class_rate_alignment"],
                 "nmi_align": kq["nmi_vs_alignment"], "nmi_drift": kq["nmi_vs_drift4"], "delta": d_}, "results/kmeans.json")
    s = new_slide("K-means on the model's pooled embeddings",
                  "K-means was fit on 768-dimensional pooled embeddings of 4,000 training pairs. k = 4 was chosen by silhouette; there is no sharp elbow. "
                  "On the test split, cluster purity against alignment is 0.800 versus a 0.780 majority-class rate, so the clusters carry little label information, and adding cluster features to the ranking stage changed macro-F1 by about one point. "
                  "We report honestly that k-means adds almost nothing; the UI still shows each hunk's cluster and similar flagged hunks.")
    pic(s, os.path.join(R, "kmeans_tsne.png"), 0.35, 0.8, 9.3, 2.85)
    tile(s, 0.35, 3.7, 2.2, 0.78, f"k = {KM['k_chosen']}", f"silhouette {f3(sil)}", bigsize=20)
    tile(s, 2.7, 3.7, 2.4, 0.78, f"{f3(kq['purity_vs_alignment'])}", f"purity vs {f3(kq['majority_class_rate_alignment'])} majority", bigsize=20)
    tile(s, 5.25, 3.7, 2.1, 0.78, f"{f3(kq['nmi_vs_alignment'])}", "NMI vs alignment", bigsize=20)
    tile(s, 7.5, 3.7, 2.15, 0.78, f"+{d_['macro_f1']:.3f}", "macro-F1 with cluster features", bigsize=20)
    takeaway(s, "Honest result: k-means adds almost nothing to ranking (NDCG@5 +0.005, single seed).", y=4.5, h=0.45)

    # ---- 14 GAN
    c = GVM["compared"]
    s = new_slide("GAN comparison: adversarial baseline on frozen embeddings",
                  "Sift has no natural GAN, so we chose this interpretation and label it plainly: a generator conditioned on the intent embedding produces aligned-hunk embeddings, and the discriminator, "
                  "trained to separate real aligned pairs from generated ones, is used as the alignment scorer on frozen MiniLM embeddings. It reaches macro-F1 0.630 and NDCG@5 0.864 against 0.680 and 0.868 for the main model. "
                  "The generator collapsed at epoch 1 (pairwise cosine 0.98) and diversified to about 0.29 by epoch 40, which is the instability we report.")
    pic(s, A["gan"], 0.35, 0.8, 4.7, 2.9)
    pic(s, A["gancurve"], 5.2, 0.8, 4.45, 2.9)
    tb(s, 0.35, 3.72, 9.3, 0.8, [
        {"text": f"Interpretation chosen: adversarially trained scorer on frozen MiniLM embeddings. GAN-style macro-F1 {f3(c['gan unsupervised']['macro_f1'])}, NDCG@5 {f3(c['gan unsupervised']['ndcg@5'])} "
                 f"vs main {f3(c['full ModernBERT']['macro_f1'])} / {f3(c['full ModernBERT']['ndcg@5'])}.", "bullet": True, "size": 11.5},
        {"text": f"Training instability: generator collapsed at epoch 1, minimum diversity {f3(GVM['gan_unsupervised_min_diversity'])}, final {f3(GVM['gan_unsupervised_final_diversity'])}.", "bullet": True, "size": 11.5}])
    takeaway(s, f"A training-free frozen cosine score reaches NDCG@5 {f3(c['frozen cosine dissimilarity']['ndcg@5'])}, above the main model.", y=4.52, h=0.4)

    # ---- 15 XAI
    ig = XAI["faithfulness"]["ig"]
    V("faithfulness k=5", {k: ig["5"][k] for k in ("mean_p_orig", "mean_p_after_top_k_masked", "mean_p_after_random_k_masked", "verdict_flip_rate_top", "verdict_flip_rate_random")}, "results/xai.json")
    V("faithfulness k=10", {k: ig["10"][k] for k in ("mean_p_after_top_k_masked", "mean_p_after_random_k_masked")}, "results/xai.json")
    cs = XAI["cases"]["collateral"]
    V("worked example p", [cs["sample_id"], cs["repo"], cs["p"], cs["true"], cs["drift_true"]], "results/xai.json cases.collateral")
    s = new_slide("Explainability: Integrated Gradients and faithfulness",
                  f"Integrated Gradients, {XAI['n_steps']} steps with a PAD baseline, gives token attributions that are aggregated to words and diff lines. The faithfulness test masks the top-attributed tokens versus random tokens on 100 test pairs. "
                  "Masking the top 5 tokens drops mean P(misaligned) from 0.579 to 0.386 against 0.482 for random masking, so attributions are more faithful than chance, though random masking also moves the score because PAD masking is off-distribution. "
                  "The worked example is an attrs test sample with P(misaligned) 0.957, truly misaligned, collateral.")
    pic(s, A["faith"], 0.3, 0.8, 4.3, 3.3)
    pic(s, os.path.join(R, "xai_case_collateral.png"), 4.7, 0.8, 5.0, 3.0)
    tb(s, 0.35, 4.08, 4.3, 0.45, [{"text": f"n = {XAI['faithfulness']['n']} test pairs; k = 5: {f3(ig['5']['mean_p_after_top_k_masked'])} (IG) vs {f3(ig['5']['mean_p_after_random_k_masked'])} (random)", "size": 11}])
    tb(s, 4.75, 3.8, 4.9, 0.7, [{"text": f"Worked example: {cs['repo']} hunk, P(misaligned) {f3(cs['p'])}, truth {cs['true']} / {cs['drift_true']}; red lines and tokens push toward misaligned.", "size": 11}])
    takeaway(s, "Attribution is evidence about the model, not proof of the cause.", y=4.55, h=0.4)

    # ---- 16 master table
    names = [("B1 Random order", "B1 random order"), ("B2 Diff lines changed", "B2 diff lines"), ("B3 Hunk size", "B3 hunk size"),
             ("B4 TF-IDF dissimilarity", "B4 TF-IDF dissimilarity"), ("B5 Logistic regression", "B5 logistic regression"),
             ("Siamese BiLSTM", "Siamese BiLSTM"), ("MiniLM-L6 cross-encoder", "MiniLM-L6 cross-encoder"),
             ("CodeBERT cross-encoder", "CodeBERT cross-encoder"), ("Main: ModernBERT-base", "Main: ModernBERT-base"),
             ("Frozen-embedding frozen cosine dissimilarity", "Frozen cosine (no training)"),
             ("GAN-style baseline: gan unsupervised", "GAN-style (unsupervised)"),
             ("Distilled MiniLM student (fp32)", "Distilled student (fp32)"), ("Student ONNX INT8 (deployed target)", "Student INT8 ONNX")]
    cols = ["macro_f1", "ndcg@5", "auc_random", "auc_hard", "ece"]
    best = {}
    for c_ in cols:
        vals = [(float(MC[k][c_]), k) for k, _ in names if MC[k][c_] not in ("", None)]
        best[c_] = (min if c_ == "ece" else max)(vals)[1]
    rows = [["Model", "macro-F1", "NDCG@5", "AUC random", "AUC hard", "ECE"]]
    bc = set()
    for ri, (k, nm) in enumerate(names, start=1):
        row = [nm]
        for ci, c_ in enumerate(cols, start=1):
            v = MC[k][c_]
            row.append(f3(v) if v not in ("", None) else "n/a")
            if v not in ("", None) and best[c_] == k:
                bc.add((ri, ci))
        V(f"master row {k}", {c_: MC[k][c_] for c_ in cols}, MCS)
        rows.append(row)
    s = new_slide("Results: master model comparison (test split)",
                  "All rows use the same repository-disjoint test split and metrics, single seed 473. Bold marks the best value in each column. "
                  "The main model leads on hard-tier AUC and NDCG@5 among trained models, but B5 has the higher macro-F1 and the frozen cosine score has the highest NDCG@5. "
                  "The reduced-budget baselines (MiniLM, CodeBERT, BiLSTM) trained far fewer epochs, so part of their gap is budget. Source: results/master_comparison.csv.")
    table(s, 0.35, 0.8, 9.3, [3.2, 1, 1, 1.2, 1.1, 0.9], rows, font=11, row_h=0.245, head_h=0.28, hilite=(9,),
          bold_cells=bc, aligns=["l", "c", "c", "c", "c", "c"])
    takeaway(s, "Best per column in bold. n/a = not measured for that model. Single seed 473; baselines had a smaller training budget.", y=4.38, h=0.55)

    # ---- 17 per-tier AUC + NDCG@K
    s = new_slide("Results: per-tier AUC and NDCG@K",
                  "NDCG@5 has a high floor of 0.72 because sessions are short, so we always report per-tier pairwise AUC as well. On the hard tier the main model reaches 0.726 against 0.551 for B4 and B5, which is where the cross-encoder earns its place. "
                  "On the random tier simple lexical baselines are as good (B4 0.812). The NDCG@K figure covers K of 1, 3, 5 and 10 over the 456 rankable test sessions.")
    pic(s, A["tier"], 0.3, 0.78, 4.7, 3.65)
    pic(s, os.path.join(R, "ndcg_vs_k_all_models.png"), 5.1, 0.85, 4.6, 3.07)
    tb(s, 5.15, 3.95, 4.5, 0.55, [{"text": f"{BASE['rankable_sessions']} rankable test sessions; hard-tier pairs {BASE['pairs_per_tier']['hard']}, random-tier pairs {BASE['pairs_per_tier']['random']}.", "size": 11}])
    V("rankable sessions", BASE["rankable_sessions"], "experiments/results/baselines.json")
    takeaway(s, f"Hard tier: main {f3(mh)} vs {f3(b4h)} for B4. Random tier: B4 {f3(MC['B4 TF-IDF dissimilarity']['auc_random'])} vs main {f3(MC['Main: ModernBERT-base']['auc_random'])}.", y=4.52, h=0.4)

    # ---- 18 PR + confusion
    t = EF["test"]["full"]
    V("main test confusion + curves", {k: t[k] for k in ("tn", "fp", "fn", "tp", "pr_auc", "roc_auc", "precision", "recall", "specificity")}, "results/eval_full.json")
    s = new_slide("Results: precision-recall curves and confusion matrix",
                  "The precision-recall curve is on the full test set with prevalence 0.22, for all models. At the validation-chosen threshold of 0.30 the main model has precision 0.516, recall 0.476 and specificity 0.874; "
                  "the confusion counts come from results/eval_full.json. The thresholded recall is low, which is why macro-F1 trails B5 even though ranking is better.")
    pic(s, os.path.join(R, "pr_curves_all_models.png"), 0.3, 0.8, 5.3, 3.65)
    rows = [["Main model", "Pred aligned", "Pred misaligned"], ["True aligned", ni(t["tn"]), ni(t["fp"])], ["True misaligned", ni(t["fn"]), ni(t["tp"])]]
    table(s, 5.75, 0.95, 3.9, [1.5, 1.2, 1.4], rows, font=11, row_h=0.34, aligns=["l", "c", "c"])
    tb(s, 5.7, 2.2, 4.0, 2.2, [
        {"text": f"PR-AUC {f3(t['pr_auc'])} (prevalence 0.22)", "bullet": True, "size": 12},
        {"text": f"ROC-AUC {f3(t['roc_auc'])}", "bullet": True, "size": 12},
        {"text": f"Precision {f3(t['precision'])}, recall {f3(t['recall'])}", "bullet": True, "size": 12},
        {"text": f"Specificity {f3(t['specificity'])} at threshold {t['threshold']:.2f} (chosen on validation)", "bullet": True, "size": 12}])
    takeaway(s, "Low recall at the chosen threshold: the model ranks better than it classifies.", y=4.52, h=0.4)

    # ---- 18b ROC per tier
    s = new_slide("Results: ROC curves per difficulty tier",
                  "ROC curves for all models on the full test set, the random tier and the hard tier, drawn from the stored test scores. Read them together with the within-session pairwise AUC on the earlier slide, "
                  "which is the tier metric we rely on. Source: results/roc_curves_per_tier.png.")
    pic(s, os.path.join(R, "roc_curves_per_tier.png"), 0.3, 0.8, 9.4, 3.0)
    tb(s, 0.35, 3.85, 9.3, 0.65, [{"text": f"Pairwise AUC on the hard tier for comparison: main {f3(MC['Main: ModernBERT-base']['auc_hard'])}, B4 {f3(MC['B4 TF-IDF dissimilarity']['auc_hard'])}.", "size": 12}])
    takeaway(s, "Panels: all test, random tier, hard tier; every trained model and the lexical baselines.", y=4.52, h=0.4)

    # ---- 19 calibration + latency
    s = new_slide("Calibration, size and CPU latency",
                  "Temperature scaling fitted on validation, T = 1.275, reduces test ECE from 0.064 to 0.028. The latency table is measured on this machine, an Intel i7-13650HX, batch 1, tokenisation included. "
                  "The student numbers were taken while the CPU was shared with a training job, so they are not strictly comparable with the main model, which was measured on an idle machine. "
                  "INT8 reaches 22.9 MB and p95 66.9 ms, meeting the targets of under 30 MB and under 100 ms on this machine only.")
    pic(s, A["cal"], 0.3, 0.78, 5.1, 2.3)
    f32, i8 = LATO["student_onnx_fp32"], LATO["student_onnx_int8"]
    V("latency rows", {"main": [LATF["size_mb"], LATF["cpu"]["p50_ms"], LATF["cpu"]["p95_ms"]],
                       "fp32": [f32["size_mb"], f32["latency_ms"]["p50_ms"], f32["latency_ms"]["p95_ms"]], "int8": [i8["size_mb"], i8["latency_ms"]["p50_ms"], i8["latency_ms"]["p95_ms"]]},
      "results/latency_full.json; results/latency_size_onnx.json")
    rows = [["Model", "Size MB", "p50 ms", "p95 ms"],
            ["Main (PyTorch fp32)", f1(LATF["size_mb"]), f1(LATF["cpu"]["p50_ms"]), f1(LATF["cpu"]["p95_ms"])],
            ["Student ONNX fp32", f1(f32["size_mb"]), f1(f32["latency_ms"]["p50_ms"]), f1(f32["latency_ms"]["p95_ms"])],
            ["Student ONNX INT8", f1(i8["size_mb"]), f1(i8["latency_ms"]["p50_ms"]), f1(i8["latency_ms"]["p95_ms"])]]
    table(s, 0.35, 3.15, 5.0, [2.1, 0.9, 0.9, 0.9], rows, font=11, row_h=0.28, aligns=["l", "c", "c", "c"], hilite=(3,))
    pic(s, A["lat"], 5.5, 0.8, 4.15, 3.5)
    takeaway(s, "Intel i7-13650HX, batch 1. Student timings taken under CPU load; numbers belong to this machine.", y=4.4, h=0.55)

    # ---- 20 ablation
    rows = [["Variant", "macro-F1", "Delta", "NDCG@5", "AUC hard"]]
    full = float(AB["1. Full model"]["macro_f1"])
    for nm, r in ablation_rows():
        dv = float(r["macro_f1"]) - full
        rows.append([nm, f3(r["macro_f1"]), "" if nm == "Full model" else f"{dv:+.3f}", f3(r["ndcg@5"]),
                     f3(r["auc_hard"]) if r["auc_hard"] else "n/a"])
        V("ablation " + nm, {k: r[k] for k in ("macro_f1", "ndcg@5", "auc_hard")}, ABS)
    s = new_slide("Ablation study (same test split, one seed)",
                  "Only removing the hunk or the intent clearly hurts: macro-F1 drops by 0.218 and 0.133, and intent-only is at chance on tier AUC. Dropping consolidation, focal loss, the multi-task heads or denoising did not hurt and each scores slightly higher macro-F1. "
                  "With one seed, differences of about 0.01 are within noise, so these runs do not demonstrate that those components help. That is an honest negative finding.")
    table(s, 0.35, 0.8, 5.6, [2.2, 0.9, 0.8, 0.9, 0.9], rows, font=11, row_h=0.268, head_h=0.28, hilite=(1,), aligns=["l", "c", "c", "c", "c"])
    pic(s, A["abl"], 6.05, 0.8, 3.65, 3.5)
    takeaway(s, "Hunk and intent are both essential; the other components are not shown to help (one seed, differences within noise).", y=4.35, h=0.6)

    # ---- 21 testing: framework + counts
    cats = TS["categories"]
    order = [("smoke", "Smoke / readiness"), ("white-box", "White-box"), ("black-box", "Black-box"), ("api", "API partitions"),
             ("data-pipeline", "Data pipeline + leakage"), ("metamorphic", "Metamorphic"), ("adversarial", "Adversarial / red-team"),
             ("back-to-back", "Back-to-back (full vs student)"), ("e2e", "UI end-to-end")]
    rows = [["Category", "Framework area", "Passed", "xfail"]]
    for k, nm in order:
        c_ = cats[k]
        V("tests " + k, [c_["passed"], c_["xfailed"], c_["failed"], c_["total"]], "results/test_summary.json")
        rows.append([nm, c_["framework_section"], str(c_["passed"]), str(c_["xfailed"])])
    T = TS["totals"]
    cov = TS["coverage"]
    V("test totals", T, "results/test_summary.json")
    V("coverage", cov, "results/test_summary.json")
    rows.append([("Total", {"bold": True}), "", (str(T["passed"]), {"bold": True}), (str(T["xfailed"]), {"bold": True})])
    s = new_slide("Testing and test design",
                  "The suite follows the Test and Evaluation of AI Models framework categories: performance, testing methods, data, models and context. There are 220 tests: 214 passed, 6 expected failures that are recorded defects (strict xfail) and 0 failed. "
                  "Line coverage is 73.4 percent over src/sift and app/backend; end-to-end browser tests are not included in that measurement. Source: results/test_summary.json.")
    table(s, 0.35, 0.8, 5.9, [2.4, 1.9, 0.7, 0.6], rows, font=11, row_h=0.3, head_h=0.3, aligns=["l", "l", "c", "c"], hilite=(10,))
    tile(s, 6.45, 0.85, 1.55, 1.05, str(T["total"]), "tests", bigsize=24)
    tile(s, 8.1, 0.85, 1.55, 1.05, str(T["failed"]), "failed", bigsize=24)
    tile(s, 6.45, 2.0, 1.55, 1.05, f"{cov['percent_covered']:.1f}%", "line coverage", bigsize=22)
    tile(s, 8.1, 2.0, 1.55, 1.05, str(T["xfailed"]), "known defects (xfail)", bigsize=24)
    tb(s, 6.4, 3.15, 3.3, 1.3, [{"text": f"Coverage: {cov['covered_lines']} of {cov['num_statements']} statements, src/sift + app/backend; e2e not measured.", "size": 11, "color": GREY}])
    takeaway(s, "Zero repositories are shared across splits (leakage test passes); every failure found is recorded as a defect.", y=4.52, h=0.4)

    # ---- 22 metamorphic, red-team, defects
    mr = [["Metamorphic relation", "n", "Result"],
          ["Determinism (re-run)", str(TM["mr_determinism"]["n"]), f"max |dp| {TM['mr_determinism']['max_abs_dp']:.0f}"],
          ["Reorder hunks", str(TM["mr_reorder"]["sessions"]), f"max |dp| {TM['mr_reorder']['max_abs_dp']:.1e}"],
          ["Whitespace-only edit", str(TM["mr_context_whitespace"]["n"]), f"flip {100 * TM['mr_context_whitespace']['flip_rate']:.1f}%"],
          ["Comment-only edit", str(TM["mr_context_comment"]["n"]), f"flip {100 * TM['mr_context_comment']['flip_rate']:.1f}%"],
          ["Identifier rename", str(TM["mr_identifier_rename"]["n"]), f"flip {100 * TM['mr_identifier_rename']['flip_rate']:.1f}%"]]
    V("metamorphic", {k: v for k, v in TM.items() if k.startswith("mr_")}, "experiments/results/testing_measurements.json")
    rt = [["Red-team attack", "Success"],
          ["Comment copies the intent", pct(TM["attack_misleading_comment_intent_copy"]["success_rate"])],
          ["Comment claims 'implements intent'", pct(TM["attack_context_comment_claims_alignment"]["success_rate"])],
          ["Prompt-injection string", pct(TM["attack_prompt_injection_string"]["success_rate"])],
          ["Irrelevant comment (false alarm)", pct(TM["attack_false_alarm_irrelevant_comment"]["success_rate"])],
          ["50,000-line hunk", "survived"]]
    V("red-team", {k: v for k, v in TM.items() if k.startswith("attack") or k.startswith("robust")}, "experiments/results/testing_measurements.json")
    sev = {}
    dr = [["ID", "Severity", "Defect", "Status"]]
    short = {"D1": "plain diff -u multi-file headers merge into previous hunk", "D2": "chit-chat filter is exact-match on 8 phrases",
             "D3": "intent-copy comment flips detected hunks to aligned", "D4": "'implements intent' comment flips hunks to aligned",
             "D-DATA-1": "documented language counts differ from release", "D-DATA-2": "test split has no TypeScript"}
    for d_ in TS["defects"]:
        dr.append([d_["id"], d_["severity"], short.get(d_["id"], d_["description"][:55]), "open (xfail)"])
    V("defect register", [(d_["id"], d_["severity"]) for d_ in TS["defects"]], "results/test_summary.json")
    s = new_slide("Metamorphic tests, red-team rates and defect register",
                  "Metamorphic relations hold within tolerance: the model is deterministic and reordering hunks changes scores by about 1e-6, while whitespace, comment and rename edits flip the verdict in 3 to 5 percent of cases. "
                  "The red-team shows the real weakness: a comment that copies the intent flips 68 percent of detected misaligned hunks to aligned, and a comment claiming the change implements the intent flips 52 percent (17 and 13 of 25). "
                  "All six defects are open and recorded as strict xfail; no mitigation is implemented yet.")
    table(s, 0.35, 0.8, 4.6, [2.3, 0.5, 1.5], mr, font=11, row_h=0.28, aligns=["l", "c", "c"])
    table(s, 5.05, 0.8, 4.6, [3.3, 1.3], rt, font=11, row_h=0.28, aligns=["l", "c"], hilite=(1, 2))
    table(s, 0.35, 2.62, 9.3, [0.9, 0.9, 5.7, 1.3], dr, font=11, row_h=0.27, aligns=["l", "l", "l", "l"], hilite=(5, 6))
    takeaway(s, "Open High-severity defect D3: the model partly keys on lexical overlap with the intent.", y=4.55, h=0.4)

    # ---- 23 implementation status
    rows = [["Component", "Status", "Note"],
            ["Data loading and verification", "Done", "counts checked; language mix differs from plan"],
            ["Diff parser, intent consolidation", "Done", "rule-based; 97.4% agreement sample"],
            ["Model, losses, training loop", "Done", "ModernBERT-base, multi-task, focal loss"],
            ["Baselines B1-B5, BiLSTM, MiniLM, CodeBERT", "Done", "reduced budget for neural baselines"],
            ["Calibration, distillation, INT8 ONNX", "Done", "T = 1.275; 22.9 MB student"],
            ["Denoising, noise study", "Reduced", "co-teaching helper not run"],
            ["K-means, GAN-style baseline, XAI", "Done", "k-means adds little"],
            ["Ablations", "Reduced", "6 full runs, one seed; rules-only dropped"],
            ["Hyper-parameter search", "Reduced", "6 Optuna trials, planned 60"],
            ["Backend API, frontend, tests", "Done", "220 tests, 0 failed"]]
    V("consolidation agreement", TM["intent_consolidation_agreement_train_sample_1500"], "experiments/results/testing_measurements.json")
    hs = J("results/hpo_small.json")
    s = new_slide("Implementation status: modules",
                  "Every pipeline stage in the plan exists and ran. Items marked Reduced ran with a smaller budget than planned: six Optuna trials instead of 60, single-seed ablations, and the noise study on a 3,000-row subset. "
                  "UniXcoder was dropped because its weights did not download, and multiple seeds and human-labelled evaluation were not done. These are listed in results/SUMMARY.md.")
    table(s, 0.35, 0.8, 9.3, [3.6, 0.9, 4.8], rows, font=11, row_h=0.31, head_h=0.3, aligns=["l", "c", "l"])
    takeaway(s, "Not done: UniXcoder (download stalled), multiple seeds, human-labelled evaluation.", y=4.52, h=0.4)

    # ---- 24 UI product
    lp = crop_shot("docs/screenshots/landing_desktop_light.png", "ui_landing.png", 1440, 810, 0)
    s = new_slide("Application UI: review workspace",
                  "The review workspace takes a test-set example or a pasted or uploaded diff, shows the consolidated intent and the turns dropped as noise, and orders hunk cards by misalignment risk with probability and drift-type badge. "
                  "The right panel highlights intent words and hunk lines by Integrated Gradients, and Accept, Flag and Needs-discussion actions update the review summary. After analysis the ground truth of a test example is shown, as here. "
                  "This screenshot is from the project's own UI test run.")
    pic(s, os.path.join(ROOT, "reports", "screenshots", "review_explained_desktop_light.png"), 0.3, 0.8, 6.0, 3.75)
    pic(s, lp, 6.45, 0.8, 3.25, 1.83)
    tb(s, 6.4, 2.7, 3.3, 1.8, [
        {"text": "Right: landing page. Left: workspace after analysis", "size": 11, "bold": True, "color": RED},
        {"text": "Ranked hunk cards, threshold slider", "bullet": True, "size": 11}, {"text": "Highlighted tokens and lines (IG)", "bullet": True, "size": 11},
        {"text": "Model switcher: main, INT8 student, baselines", "bullet": True, "size": 11}, {"text": "Keyboard: Accept a, Flag f, Discuss d", "bullet": True, "size": 11}])
    takeaway(s, "Real screenshots from the running application; analysis ran on a real test-set sample.", y=4.55, h=0.4)

    # ---- 25 developer console
    shots = [("Console overview", "docs/screenshots/dev_overview_desktop_light.png"), ("Results dashboard", "docs/screenshots/dev_results_desktop_light.png"),
             ("Experiments", "docs/screenshots/dev_experiments_desktop_light.png"), ("Dataset explorer", "docs/screenshots/dev_dataset_desktop_light.png")]
    s = new_slide("Application UI: developer console",
                  "The developer console is a second interface in the same build with an overview, results dashboard, experiments page, dataset explorer, testing page and related-work page. "
                  "Every number on these pages is read from the real result files served by the backend, nothing is hard-coded. Screenshots are from docs/screenshots at desktop width.")
    for i, (nm, src) in enumerate(shots):
        cp = crop_shot(src, f"ui_dev{i}.png", 1100, 420, 0, left=200)
        x, y = 0.35 + (i % 2) * 4.7, 0.78 + (i // 2) * 1.88
        tb(s, x, y, 4.5, 0.24, [{"text": nm, "bold": True, "size": 11.5, "color": RED}])
        pic(s, cp, x, y + 0.25, 4.5, 1.62)
    takeaway(s, "Console pages: overview, results, experiments, dataset, testing, related work (testing page not shown).", y=4.55, h=0.4)

    # ---- 26 backend + deployment
    dep = os.path.join(ROOT, "deploy", "DEPLOY_URL.txt")
    live = open(dep).read().strip() if os.path.exists(dep) else None
    pkg = os.path.isdir(os.path.join(ROOT, "deploy"))
    V("deploy url present", bool(live), "deploy/DEPLOY_URL.txt (checked at build time)")
    rows = [["Endpoint", "Purpose"],
            ["POST /api/analyze", "turns + diff or sample id: intent, per-hunk P, drift, cluster, rank, attributions"],
            ["GET /api/samples, /{id}", "curated test-set examples with ground truth"],
            ["GET /api/models", "main, student/INT8, baselines for the switcher"],
            ["GET /api/results/*", "JSON behind the dashboards"],
            ["GET /api/dataset/stats, /api/tests", "dataset statistics, test and coverage summary"],
            ["GET /api/health", "readiness; OpenAPI docs at /docs"]]
    s = new_slide("Backend API and deployment status",
                  "The backend is FastAPI; the model is loaded once and runs on CPU or GPU automatically. Inputs are validated and errors return clear messages. "
                  + ("The live deployment link is shown on the right." if live else "A hosted link is not available yet, so the app is launched locally with make demo; the repository link is the one in the project template."))
    table(s, 0.35, 0.8, 5.6, [2.1, 3.5], rows, font=11, row_h=0.5, head_h=0.3)
    dl = [{"text": "Deployment", "bold": True, "color": RED, "size": 13}]
    if live:
        dl.append({"text": f"Live: {live}", "size": 11.5, "bold": True, "link": live if live.startswith("http") else None})
    else:
        dl.append({"text": "Hosted link: not live yet", "size": 11.5, "bold": True})
        dl.append({"text": "Local launch: make demo  (backend :8000, UI :5173)", "size": 11.5})
    dl.append({"text": "Repository: https://github.com/Kundhave/sift", "size": 11.5, "link": "https://github.com/Kundhave/sift"})
    dl.append({"text": "Deployment package: " + ("ready in deploy/ (Dockerfile, Hugging Face Spaces script); not deployed yet, Docker build not run" if pkg else "in preparation"), "size": 11.5})
    dl.append({"text": "CPU-only inference by default; no code leaves the machine.", "size": 11.5})
    card(s, 6.1, 0.8, 3.55, 3.5, None, dl)
    takeaway(s, "One command starts backend and frontend: make demo.", y=4.5, h=0.45)

    # ---- 27 SDG
    s = new_slide("SDG relevance",
                  "SDG 9, Industry, Innovation and Infrastructure, is the primary goal: Sift supports trustworthy AI-assisted software engineering by checking agent output against developer intent. "
                  "SDG 8, Decent Work and Economic Growth, is secondary: ranking hunks by risk is intended to reduce reviewer burden. We have not measured reviewer time saved; the ranking metrics are the only evidence so far.")
    card(s, 0.35, 0.85, 4.55, 1.9, "SDG 9 (primary): Industry, Innovation and Infrastructure", [
        {"text": "Trustworthy AI-assisted software engineering", "bullet": True},
        {"text": "Verification layer between agent and developer", "bullet": True},
        {"text": "Open, documented dataset and reproducible pipeline", "bullet": True}], headsize=12.5)
    card(s, 5.05, 0.85, 4.6, 1.9, "SDG 8 (secondary): Decent Work and Economic Growth", [
        {"text": "Less reviewer burden: likeliest problems first", "bullet": True},
        {"text": "Local, CPU-capable: no cloud dependency or code upload", "bullet": True}], headsize=12.5)
    card(s, 0.35, 2.9, 9.3, 1.5, "Evidence and limits", [
        {"text": f"Ranking evidence: NDCG@5 {f3(mn)} vs {f3(b1n)} for random order (test split).", "bullet": True},
        {"text": "Reviewer time saved is not measured; no user study has been run.", "bullet": True},
        {"text": "Labels are constructed, so real-world benefit needs human validation.", "bullet": True}], headsize=12.5)
    takeaway(s, "SDG 9 primary, SDG 8 secondary.", y=4.52, h=0.4)

    # ---- 28 discussion
    mf = float(MC["Main: ModernBERT-base"]["macro_f1"])
    s = new_slide("Discussion: key findings and weak spots",
                  "Left: what the measurements support. Right: the weak spots we state plainly. The main model's macro-F1 is below logistic regression, the training-free frozen cosine score beats it on NDCG@5, k-means adds almost nothing, "
                  "and four ablated variants score slightly higher than the full model. Everything is single seed with constructed labels, so claims are about ranking on the hard tier and compact deployment, not state of the art.")
    card(s, 0.35, 0.8, 4.55, 3.65, "What the data supports", [
        {"text": f"Hard-tier pairwise AUC {f3(mh)} vs {f3(b4h)} for B4/B5.", "bullet": True, "size": 13, "space": 8},
        {"text": f"NDCG@5 {f3(mn)} vs {f3(b1n)} random order.", "bullet": True, "size": 13, "space": 8},
        {"text": f"Temperature scaling: ECE {f3(CAL['ece_test_before'])} to {f3(CAL['ece_test_after'])}.", "bullet": True, "size": 13, "space": 8},
        {"text": f"INT8 student keeps macro-F1 {f3(MC['Student ONNX INT8 (deployed target)']['macro_f1'])} at {f1(sz)} MB.", "bullet": True, "size": 13, "space": 8},
        {"text": "Intent-aware joint reading matters: hunk-only and intent-only collapse.", "bullet": True, "size": 13, "space": 8},
        {"text": "IG masking drops P(misaligned) more than random masking.", "bullet": True, "size": 13, "space": 8}], headsize=13)
    card(s, 5.05, 0.8, 4.6, 3.65, "Honest weak spots", [
        {"text": f"Macro-F1 {f3(mf)} is below B5 ({f3(MC['B5 Logistic regression']['macro_f1'])}).", "bullet": True, "size": 13, "space": 8},
        {"text": f"Frozen cosine wins NDCG@5 ({f3(MC['Frozen-embedding frozen cosine dissimilarity']['ndcg@5'])}).", "bullet": True, "size": 13, "space": 8},
        {"text": f"K-means adds little: purity {f3(kq['purity_vs_alignment'])} vs {f3(kq['majority_class_rate_alignment'])} majority.", "bullet": True, "size": 13, "space": 8},
        {"text": "Ablated variants are not worse (no focal, no multi-task, no consolidation, no denoising).", "bullet": True, "size": 13, "space": 8},
        {"text": f"Confident learning precision {f3(d['precision'])} on known flips.", "bullet": True, "size": 13, "space": 8},
        {"text": "Single seed; constructed labels; red-team defect D3 open.", "bullet": True, "size": 13, "space": 8}], headsize=13)
    takeaway(s, "Claim scope: hard-tier ranking and compact local deployment, not a new architecture.", y=4.52, h=0.4)

    # ---- 29 conclusion & future work
    s = new_slide("Conclusion and future work",
                  "In conclusion: Sift is a working, tested, locally runnable system with a trained cross-encoder, baselines, calibration, a compact INT8 student, explanations and a UI. "
                  "Its measured strength is hard-tier ranking; its limits are the constructed labels, a single seed and comment-based evasion. Future work lists the concrete next steps, starting with human validation of the 900-sample gold sheet.")
    card(s, 0.35, 0.82, 4.55, 3.6, "Conclusion", [
        {"text": "Hunk-level, multi-turn-intent task framing with its own dataset (Sift-IA v1.0).", "bullet": True, "size": 13, "space": 8},
        {"text": "ModernBERT cross-encoder improves hard-tier ranking over lexical and size baselines.", "bullet": True, "size": 13, "space": 8},
        {"text": "Calibrated, explained and compressed to a 22.9 MB INT8 model for local CPU use.", "bullet": True, "size": 13, "space": 8},
        {"text": "Working UI and API with 220 tests (0 failed) and a recorded defect register.", "bullet": True, "size": 13, "space": 8}], headsize=13)
    card(s, 5.05, 0.82, 4.6, 3.6, "Future work", [
        {"text": "Human validation using the 900-sample gold sheet.", "bullet": True, "size": 13, "space": 8},
        {"text": "Multiple seeds with confidence intervals; full 60-trial search.", "bullet": True, "size": 13, "space": 8},
        {"text": "Mitigate comment-based evasion (strip comment-only lines before scoring).", "bullet": True, "size": 13, "space": 8},
        {"text": "Co-teaching, longer baseline training, TypeScript test repositories.", "bullet": True, "size": 13, "space": 8},
        {"text": "Learned intent consolidation; hosted deployment and user study.", "bullet": True, "size": 13, "space": 8}], headsize=13)
    takeaway(s, "Contribution: task framing, pipeline and dataset; encoder and training methods are existing.", y=4.52, h=0.4)

    # ---- 30 member contributions
    s = new_slide("Member contributions",
                  "Team 24 has a single member, so all work is by Kundhave S. The contribution list covers the full pipeline from dataset to the application and tests. Faculty guide: Dr. T. Senthil Kumar, Professor, CSE Department.")
    rows = [["Member", "Contribution (technical)"],
            ["Kundhave S (CB.SC.U4CSE23037)",
             "One-member team; all work. (1) Task formulation, drift taxonomy and Sift-IA v1.0 dataset construction with noise ground truth. "
             "(2) Diff parser, intent consolidation and the ModernBERT multi-task cross-encoder with focal loss. "
             "(3) Baselines, ablations, noise and denoising study, k-means, GAN-style baseline and Integrated Gradients. "
             "(4) Calibration, distillation and INT8 ONNX export. (5) FastAPI backend, React frontend, test suite and defect register. "
             "(6) IEEE DataPort documentation, reports and presentation."]]
    table(s, 0.35, 0.85, 9.3, [2.3, 7.0], rows, font=13, row_h=1.9, head_h=0.35)
    tb(s, 0.35, 3.4, 9.3, 0.5, [{"text": "Faculty guide: Dr. T. Senthil Kumar, Professor, CSE Department. Course: 23CSE473 Neural Networks and Deep Learning.", "size": 12, "color": GREY}])

    # ---- 31 references
    refs = [
        "[1] C. G. Northcutt, L. Jiang, I. L. Chuang. Confident Learning: Estimating Uncertainty in Dataset Labels. JAIR, 2021.",
        "[2] Z. Xu et al. Code Comment Inconsistency Detection Based on Confidence Learning. IEEE TSE 50(3), 2024.",
        "[3] Huang et al. Are your comments outdated? Detecting code-comment consistency. J. Softw. Evol. Process 37(1), 2025.",
        "[4] Yang et al. RComSync: code-comment synchronization with in-context learning and reranking. Empir. Softw. Eng. 31(4), 2026.",
        "[5] Fan et al. Exploring the Capabilities of LLMs for Code-Change-Related Tasks. ACM TOSEM 34(6), 2025.",
        "[6] Wang et al. Best practices for code pre-trained models on SE classification tasks. Expert Syst. Appl., 2025.",
        "[7] S. Panthaplackel et al. Deep Just-In-Time Inconsistency Detection Between Comments and Source Code. AAAI 2021.",
        "[8] S. Panthaplackel et al. Learning to Update Natural Language Comments Based on Code Changes. ACL 2020.",
        "[9] K. Herzig, A. Zeller. The Impact of Tangled Code Changes. MSR 2013.",
        "[10] B. Warner et al. Smarter, Better, Faster, Longer: ModernBERT. arXiv:2412.13663, 2024.",
        "[11] R. Nogueira, K. Cho. Passage Re-ranking with BERT. arXiv:1901.04085, 2019.",
        "[12] Z. Feng et al. CodeBERT. Findings of EMNLP 2020.",
        "[13] T.-Y. Lin et al. Focal Loss for Dense Object Detection. ICCV 2017.",
        "[14] C. Guo et al. On Calibration of Modern Neural Networks. ICML 2017.",
        "[15] G. Hinton, O. Vinyals, J. Dean. Distilling the Knowledge in a Neural Network. 2015.",
        "[16] M. Sundararajan, A. Taly, Q. Yan. Axiomatic Attribution for Deep Networks. ICML 2017.",
        "[17] B. Han et al. Co-teaching. NeurIPS 2018.",
        "[18] Test and Evaluation of Artificial Intelligence Models Framework (course reference).",
        "[19] Sift-IA v1.0 record on IEEE DataPort (link on the dataset slide).",
    ]
    s = new_slide("References",
                  "References 1 to 6 are the base paper and the five journal papers from the literature survey. References 7 to 17 are the building-block papers listed in docs/RELATED_WORK.md; "
                  "items 13 to 17 were cited from memory there and should be checked once more before the final report. The framework is the course testing reference, and the last item is the dataset record.")
    tb(s, 0.3, 0.78, 4.8, 3.8, [{"text": r, "size": 11, "space": 3} for r in refs[:9]])
    tb(s, 5.05, 0.78, 4.7, 3.8, [{"text": r, "size": 11, "space": 3} for r in refs[9:]])

    prs.core_properties.title = "Sift: Intent-Hunk Alignment Verification for AI-Generated Code Changes (Review 3)"
    prs.core_properties.author = "Kundhave S"
    prs.save(OUT)
    with open(TRACE, "w") as f:
        json.dump(TRACE_LOG, f, indent=1, default=str)
    print("saved", OUT, len(prs.slides), "slides;", len(TRACE_LOG), "trace entries")


if __name__ == "__main__":
    build()
