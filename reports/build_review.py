"""Deliverable 3: Review 2 documentation PDF. Every number is read from results JSON/CSV at build time."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reports"))
import reportlib as R
from reportlib import P, H, H3_, bullets, table, fmt, pct, cm
from reportlab.platypus import PageBreak, Spacer, Paragraph, KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER

SRC = {}


def rj(rel):
    SRC.setdefault("_all", set()).add(rel)
    return json.load(open(ROOT / rel))


def rcsv(rel):
    SRC["_all"].add(rel) if "_all" in SRC else SRC.setdefault("_all", {rel})
    return list(csv.DictReader(open(ROOT / rel)))


def f(x, d=3):
    return "-" if x in (None, "") else f"{float(x):.{d}f}"


def cover(project):
    C = lambda n, **k: ParagraphStyle(n, parent=R.BODY, alignment=TA_CENTER, **k)
    navy = R.NAVY
    s = [Spacer(1, 2.2 * cm),
         Paragraph("Amrita Vishwa Vidyapeetham", C("u", fontName="Helvetica-Bold", fontSize=20, leading=25, textColor=navy)),
         Paragraph("Department of Computer Science and Engineering", C("d", fontSize=12, leading=16)),
         Paragraph("Amrita School of Computing", C("d2", fontSize=12, leading=16, spaceAfter=30)),
         Paragraph("REVIEW 2 - DOCUMENTATION", C("r", fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=colors_grey(), spaceAfter=14)),
         Paragraph(project, C("p", fontName="Helvetica-Bold", fontSize=21, leading=26, textColor=navy, spaceAfter=10)),
         Paragraph("Individual technical report: performance, testing, explainability, ablation, data and model cards", C("s", fontName="Helvetica-Oblique", fontSize=10.5, leading=14, spaceAfter=34)),
         table([["Course", "23CSE473 Neural Networks and Deep Learning"],
                ["Professor", "Dr. T. Senthil Kumar, Professor, CSE Department"],
                ["Team", "Team / Group 24 (single-member team)"]], [3.2 * cm, 11 * cm], header=False, zebra=False),
         Spacer(1, 1.2 * cm),
         Paragraph("SUBMITTED BY", C("sb", fontName="Helvetica-Bold", fontSize=10, textColor=navy, spaceAfter=4)),
         Paragraph("Kundhave S", C("n", fontName="Helvetica-Bold", fontSize=14, leading=18)),
         Paragraph("CB.SC.U4CSE23037", C("rn", fontSize=11, spaceAfter=22)),
         Paragraph("Scope: single-member team; all design, implementation, experiments, testing and documentation were done by Kundhave S.", C("sc", fontName="Helvetica-Oblique", fontSize=9.5, leading=13)),
         PageBreak()]
    return s


def colors_grey():
    from reportlab.lib import colors
    return colors.HexColor("#555555")


def perf_figure(mc):
    import matplotlib.pyplot as plt
    rows = [r for r in mc if r["auc_hard"]]
    names = [r["model"].replace("Main: ", "").replace(" (deployed target)", "").replace("GAN-style baseline: ", "GAN ").replace("Frozen-embedding frozen cosine dissimilarity", "Frozen cosine").replace(" cross-encoder", " CE").replace("GAN gan unsupervised", "GAN unsup.").replace("GAN gan hybrid with misaligned negatives", "GAN hybrid").replace("Distilled MiniLM student (fp32)", "Student fp32").replace("MiniLM student, no distillation", "Student no distill") for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    x = range(len(rows)); w = 0.4
    ax.bar([i - w / 2 for i in x], [float(r["auc_random"]) for r in rows], w, label="random tier", color=R.PALETTE[0])
    ax.bar([i + w / 2 for i in x], [float(r["auc_hard"]) for r in rows], w, label="hard tier", color=R.PALETTE[1])
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=28, ha="right", fontsize=6.5)
    ax.set_ylabel("within-session pairwise AUC", fontsize=7); ax.axhline(0.5, color="#999", lw=.6, ls="--")
    ax.tick_params(axis="y", labelsize=7); ax.set_ylim(0, 1.0); ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    return R.fig_save(fig, "review_tier_auc.png")


def build():
    R.H2.keepWithNext = True; R.H3.keepWithNext = True
    mc = rcsv("results/master_comparison.csv"); ab = rcsv("results/ablation.csv")
    ts = rj("results/test_summary.json"); xai = rj("results/xai.json"); cal = rj("results/calibration.json")
    lf = rj("results/latency_full.json"); lo = rj("results/latency_size_onnx.json"); hpo = rj("results/hpo_small.json")
    ds = rj("data/dataset_statistics.json"); dv = rj("experiments/results/data_verification.json")
    cfg = rj("experiments/runs/full/config.json"); run = rj("experiments/runs/full/results.json")
    cl = rj("results/confident_learning.json"); nr = rj("results/noise_robustness.json"); gv = rj("results/gan_vs_main.json")
    repos = rcsv("data/source_repositories.csv")
    M = {r["model"]: r for r in mc}
    main = M["Main: ModernBERT-base"]; i8 = M["Student ONNX INT8 (deployed target)"]
    T = ts["totals"]; fa = xai["faithfulness"]
    hw = lf["cpu_info"]["cpu"]; gpu = lf["cuda"]["gpu"]
    project = "Sift: a local verification layer for AI coding agents"
    S = cover(project)
    S += R.title_block("Review 2 - Individual Documentation", "Following the headings of the CDAO Test and Evaluation of AI Models framework (April 2024)")
    S.append(R.header_table("Sift", "Review 2 Documentation", "Team 24 (single member)", "Kundhave S, CB.SC.U4CSE23037", "Test cohort", f"{ds['test']['hunks']:,} hunks (Python, JavaScript)"))
    S.append(Spacer(1, 4))

    # 1
    S.append(H("1. Project overview and individual scope"))
    S.append(P("Sift reads the intent a developer expressed over a coding-agent session and classifies every diff hunk as Aligned or Misaligned with it, then ranks hunks so a reviewer sees the likeliest misaligned ones first. Misaligned hunks carry one of three drift types: Collateral, Scope-Expansion, Purpose-Drift. Everything runs locally. The model is a ModernBERT-base cross-encoder over (consolidated intent, hunk) with alignment, drift-type and confidence heads, plus a distilled MiniLM student exported to INT8 ONNX for CPU use."))
    S.append(P("<b>Novelty is not claimed for any component.</b> The encoder and the training methods (focal loss, multi-task learning, temperature scaling, distillation, confident learning) are existing. The contribution is the task framing (hunk-level, multi-turn consolidated intent), the pipeline combining them, and the Sift-IA v1.0 dataset. <b>Individual scope:</b> single-member team, so all work (dataset use, model, experiments, tests, UI, reports) is by Kundhave S."))


    # 2
    S.append(H("2. Performance"))
    S.append(P("Test split, 2,700 hunks, labels <i>label_clean</i>, decision threshold tuned on validation. Pairwise AUC is within-session, per difficulty tier. NDCG@5 is over rankable sessions (at least 2 hunks, at least 1 misaligned)."))
    order = ["B2 Diff lines changed", "B4 TF-IDF dissimilarity", "B5 Logistic regression", "Siamese BiLSTM", "MiniLM-L6 cross-encoder", "CodeBERT cross-encoder",
             "Frozen-embedding frozen cosine dissimilarity", "GAN-style baseline: gan hybrid with misaligned negatives", "Main: ModernBERT-base", "Distilled MiniLM student (fp32)", "Student ONNX INT8 (deployed target)"]
    rows = [["Model", "macro-F1", "Prec.", "Recall", "NDCG@5", "AUC random", "AUC hard", "ECE"]]
    for k in order:
        r = M[k]; nm = k.replace("GAN-style baseline: gan hybrid with misaligned negatives", "GAN-style (hybrid)").replace("Frozen-embedding frozen cosine dissimilarity", "Frozen MiniLM cosine")
        rows.append([("<b>%s</b>" % nm) if k.startswith("Main") else nm] + [f(r[c]) for c in ("macro_f1", "precision", "recall", "ndcg@5", "auc_random", "auc_hard", "ece")])
    S.append(table(rows, [5.2 * cm, 1.6 * cm, 1.5 * cm, 1.5 * cm, 1.6 * cm, 2.0 * cm, 1.8 * cm, 1.4 * cm]))
    S.append(P("Table 1: headline results (results/master_comparison.csv). A dash means the baseline does not produce that quantity.", R.CAP))
    S.append(P(f"The main model has the highest hard-tier AUC ({f(main['auc_hard'])}), but by a small margin over frozen cosine dissimilarity ({f(M['Frozen-embedding frozen cosine dissimilarity']['auc_hard'])}); the frozen-embedding baseline has a slightly higher NDCG@5 ({f(M['Frozen-embedding frozen cosine dissimilarity']['ndcg@5'])} vs {f(main['ndcg@5'])}). On macro-F1 the main model ({f(main['macro_f1'])}) is below the logistic-regression baseline B5 ({f(M['B5 Logistic regression']['macro_f1'])}). NDCG@5 has a high floor ({f(M['B1 Random order']['ndcg@5'], 4)} for random order) because the median session has 2 hunks."))
    S.append(R.figure(perf_figure(mc), 15.5, "Figure 1: within-session pairwise AUC per tier (results/master_comparison.csv; models without stored tier AUC omitted)."))
    S.append(H3_("Latency and size (measured on this machine only)"))
    o8 = lo["student_onnx_int8"]; o32 = lo["student_onnx_fp32"]
    S.append(table([["Model / runtime", "Size MB", "p50 ms", "p95 ms", "Conditions"],
                    ["ModernBERT-base, torch fp32, CPU", f(lf["size_mb"], 1), f(lf["cpu"]["p50_ms"], 1), f(lf["cpu"]["p95_ms"], 1), f"{hw}, {lf['cpu']['torch_threads']} threads, L=512, batch 1, idle"],
                    ["ModernBERT-base, torch fp32, GPU", f(lf["size_mb"], 1), f(lf["cuda"]["p50_ms"], 1), f(lf["cuda"]["p95_ms"], 1), gpu],
                    ["MiniLM student, ONNX fp32, CPU", f(o32["size_mb"], 1), f(o32["latency_ms"]["p50_ms"], 1), f(o32["latency_ms"]["p95_ms"], 1), f"{lo['threads']} threads; CPU shared with a GPU training job"],
                    ["MiniLM student, ONNX INT8, CPU", f(o8["size_mb"], 1), f(o8["latency_ms"]["p50_ms"], 1), f(o8["latency_ms"]["p95_ms"], 1), "same; p95 may be pessimistic"]],
                   [5.2 * cm, 1.5 * cm, 1.4 * cm, 1.4 * cm, 7.6 * cm]))
    S.append(P(f"Table 2: size and latency (results/latency_full.json, results/latency_size_onnx.json). The INT8 student is {f(o8['size_mb'],1)} MB, meeting the planned under-30 MB target; the planned under-100 ms p95 on CPU is met by both the full model ({f(lf['cpu']['p95_ms'],1)} ms) and the INT8 student, but the INT8 p95 was measured while the CPU was shared, so it is not a clean figure. INT8 vs fp32 logit correlation: {f(o8['fp32_vs_int8_logit_corr'])}.", R.CAP))
    S.append(H3_("Other dimensions"))
    d3 = next(d for d in ts["defects"] if d["id"] == "D3"); d4 = next(d for d in ts["defects"] if d["id"] == "D4")
    S += bullets([
        f"<b>Interpretability:</b> Integrated Gradients ({xai['n_steps']} steps, PAD baseline). Masking the top-10 attributed tokens lowers P(misaligned) by {f(fa['ig']['10']['mean_drop_top'])} on average versus {f(fa['ig']['10']['mean_drop_random'])} for 10 random tokens ({fa['n']} test hunks predicted misaligned); attribution beats random in {pct(fa['ig']['10']['frac_top_beats_random'],0)} of hunks. Input x Gradient gives a larger gap ({f(fa['ixg']['10']['mean_drop_top'])} vs {f(fa['ixg']['10']['mean_drop_random'])}).",
        f"<b>Robustness:</b> red-team defects D3 and D4 are open. {d3['description'].split(' flips ')[0].capitalize()} flips {d3['description'].split('flips ')[1].split(' of detected')[0]} of detected misaligned hunks to aligned; the 'this change implements' context comment flips {d4['description'].split('flips ')[1].split(' of detected')[0]}. The model partly keys on lexical overlap with the intent.",
        f"<b>Uncertainty and calibration:</b> temperature scaling (T = {f(cal['temperature'])}, fitted on validation) reduces test ECE from {f(cal['ece_test_before'])} to {f(cal['ece_test_after'])}. The table above reports ECE of the raw evaluation runs.",
        "<b>Drift:</b> see Scope and limitations. Proposed procedure: compute the model's score distribution and the share of flagged hunks on new repositories or a later time window, compare with the test distribution with a two-sample test, and re-evaluate on freshly labelled samples. Commit dates reach 2026, but the splits are by repository, not by time.",
        f"<b>Representativeness:</b> splits are repository-disjoint. Test repositories: attrs, axios, httpx ({dv['splits']['test']['languages']['Python']:,} Python and {dv['splits']['test']['languages']['JavaScript']:,} JavaScript hunks). <b>The test split has no TypeScript</b> (all TypeScript is in validation), so TypeScript generalisation is untested.",
        "<b>Bias:</b> The only imbalances identified are language and repository (Python, JavaScript and TypeScript only, 15 open-source projects). No analysis of developer demographics or other fairness dimensions was done."])

    # 3
    S.append(H("3. Testing methods"))
    S.append(P(f"The pytest suite has {T['total']} tests: {T['passed']} passed, {T['xfailed']} expected failures (strict xfail, each a recorded defect), {T['failed']} failed, {T['skipped']} skipped. Line coverage of src/sift and app/backend is {ts['coverage']['percent_covered']}% ({ts['coverage']['scope']}). Categories cover smoke, white-box, black-box, data pipeline (including zero repositories shared across splits), metamorphic relations, adversarial/red-team, API, back-to-back (full model vs student) and browser end-to-end tests. Details are in the separate Testing document."))
    rows = [["Category", "Files", "Passed", "xfailed", "Failed", "Framework section"]]
    for k, v in ts["categories"].items():
        rows.append([k, ", ".join(v["files"]), v["passed"], v["xfailed"], v["failed"], v["framework_section"]])
    S.append(table(rows, [2.8 * cm, 5.6 * cm, 1.5 * cm, 1.6 * cm, 1.4 * cm, 4 * cm]))
    S.append(P("Table 3: test results by category (results/test_summary.json).", R.CAP))

    # 4
    S.append(H("4. Explainability (XAI)"))
    S.append(P(f"Token-level attribution on the cross-encoder uses captum Integrated Gradients (target: P(misaligned), baseline: {xai['baseline']}, {xai['n_steps']} steps) and Input x Gradient, aggregated to words and diff lines. On average {pct(xai['abs_attribution_share_on_intent_tokens_mean'])} of absolute attribution falls on intent tokens. Faithfulness uses replace-by-[PAD] deletion on {fa['n']} test hunks predicted misaligned:"))
    rows = [["Top-k masked", "IG drop (top)", "IG drop (random)", "IG flip rate (top / random)", "IxG drop (top)", "IxG drop (random)"]]
    for k in ("3", "5", "10", "20"):
        a, b = fa["ig"][k], fa["ixg"][k]
        rows.append([k, f(a["mean_drop_top"]), f(a["mean_drop_random"]), f"{pct(a['verdict_flip_rate_top'],0)} / {pct(a['verdict_flip_rate_random'],0)}", f(b["mean_drop_top"]), f(b["mean_drop_random"])])
    S.append(table(rows, [2.4 * cm, 2.6 * cm, 3 * cm, 4 * cm, 2.6 * cm, 2.8 * cm]))
    S.append(P(f"Table 4: faithfulness (results/xai.json). Explanation costs a median {f(xai['latency_ms']['explain_ig_p50'],0)} ms on the {xai['latency_ms']['gpu']} versus {f(xai['latency_ms']['score_p50'],0)} ms to score ({xai['latency_ms']['note']}). Attribution is evidence about what the model used, not proof of the cause.", R.CAP))

    # 5
    S.append(H("5. Ablation"))
    S.append(P("All variants use the same protocol and test split, one seed (473). Differences of about 0.01 macro-F1 are within expected run-to-run noise and are not significant."))
    rows = [["Configuration", "macro-F1", "NDCG@5", "AUC rand.", "AUC hard", "ECE", "Size MB"]]
    for r in ab:
        if r["group"] != "ablation": continue
        rows.append([r["model"], f(r["macro_f1"]), f(r["ndcg@5"]), f(r["auc_random"]), f(r["auc_hard"]), f(r["ece"]), f(r["size_mb"], 1)])
    S.append(table(rows, [6 * cm, 1.8 * cm, 1.8 * cm, 1.9 * cm, 1.8 * cm, 1.5 * cm, 1.7 * cm]))
    S.append(P("Table 5: ablation (results/ablation.csv; INT8 per-tier AUC was not stored).", R.CAP))
    S.append(P("Findings: removing the hunk or the intent clearly hurts (hunk-only and intent-only are near chance on the hard tier), which supports reading both texts jointly. In contrast, the ablated variants without consolidation, focal loss, multi-task heads or denoising did <b>not</b> underperform the full model on macro-F1 in this single-seed run, so these components are not shown to help here."))

    # 6
    S.append(H("6. Baseline comparison"))
    g = lambda k, c: f(M[k][c])
    S.append(P(f"Table 1 and Figure 1 give the comparison: five non-neural baselines (B1-B5), a Siamese BiLSTM, a MiniLM-L6 cross-encoder and a reduced-protocol CodeBERT cross-encoder, frozen-embedding cosine dissimilarity, and two GAN-style adversarially trained baselines on frozen MiniLM embeddings (an interpretation of the GAN deliverable, since Sift has no GAN). The best non-neural hard-tier AUC is {g('B4 TF-IDF dissimilarity','auc_hard')} (B4); the main model reaches {f(main['auc_hard'])}. A frozen-embedding cosine baseline needs no training and is close to the main model on ranking ({g('Frozen-embedding frozen cosine dissimilarity','auc_hard')} hard AUC), while its macro-F1 is lower ({g('Frozen-embedding frozen cosine dissimilarity','macro_f1')}). The GAN-style baselines did not beat the cross-encoder on hard-tier AUC ({g('GAN-style baseline: gan hybrid with misaligned negatives','auc_hard')} for the hybrid). UniXcoder was dropped, and the CodeBERT run used a reduced protocol."))

    # 7
    S.append(H("7. Data and dataset"))
    nz = ds["noise_rows"]
    S.append(H3_("Data Card: Sift-IA v1.0"))
    lic = {}
    for r in repos: lic.setdefault(r["license"], []).append(r["repo"])
    rows = [["Field", "Value"],
            ["Name / version", "Sift-IA v1.0 (released package; the builder scripts are not in this workspace)"],
            ["Source", f"{len(repos)} open-source GitHub repositories, licences: " + "; ".join(f"{k} ({', '.join(v)})" for k, v in lic.items()) + ". Licence texts are shipped with the package."],
            ["Size", f"{sum(ds[s]['hunks'] for s in ('train','val','test')):,} hunk-intent pairs, {sum(ds[s]['sessions'] for s in ('train','val','test')):,} sessions, {ds['distinct_source_commits']:,} distinct source commits; median {ds['approx_tokens']['median']:.0f} tokens per pair, p95 {ds['approx_tokens']['p95']}, max {ds['approx_tokens']['max']:,}"],
            ["Splits (by repository)"] + [""],
            ["Label construction", "Aligned hunks are real hunks of the seed commit; a misaligned hunk is a donor hunk from a different commit inserted into the session (tiers: random, hard by identifier-overlap mining). Labels follow from construction. Drift types: Collateral, Scope-Expansion, Purpose-Drift."],
            ["Noise shipped with ground truth", f"{ds['rows_with_any_noise']:,} rows carry noise: intent turns {nz['intent_turn']:,}; duplicate hunks {nz['duplicate_hunk']:,}; non-semantic hunks {nz['non_semantic_hunk']:,}; label flips {nz['label_flip']} (train only); over-length {nz['truncation']}"],
            ["Collection dates", f"commits from {ds['commit_date_range'][0]} to {ds['commit_date_range'][1]}"],
            ["Known limitations", "Misaligned hunks are constructed, so a model may learn construction artefacts (the hard tier mitigates, not removes this). No human labels (the 900-sample gold sheet is blank). Seed commits may be tangled. A donor hunk may coincidentally suit the intent. Only Python, JavaScript, TypeScript. Dataset audit: the released per-language hunk counts differ from the earlier documented counts (measured Python " + f"{sum(dv['splits'][s]['languages'].get('Python',0) for s in dv['splits']):,}, JavaScript {sum(dv['splits'][s]['languages'].get('JavaScript',0) for s in dv['splits']):,}, TypeScript {sum(dv['splits'][s]['languages'].get('TypeScript',0) for s in dv['splits']):,}); the test split has no TypeScript; 456 rankable test sessions, not 447."],
            ["Intended use", "Research and benchmarking of hunk-level intent-alignment classification and ranking."],
            ["Not for use", "Judging real developers, defect prediction, security auditing, or any claim about agent-authored code (v1.0 has no agent-authored sessions)."]]
    sp = "; ".join(f"{s}: {ds[s]['hunks']:,} hunks, {ds[s]['sessions']:,} sessions, {ds[s]['misaligned_clean']:,} misaligned ({', '.join(ds[s]['repos'])})" for s in ("train", "val", "test"))
    rows[4][1] = sp
    S.append(table(rows, [3.4 * cm, 14 * cm]))
    S.append(P("Table 6: Data Card (data/dataset_statistics.json, data/source_repositories.csv, experiments/results/data_verification.json).", R.CAP))

    # 8
    S.append(H("8. AI models"))
    S.append(H3_("Model Card: Sift ModernBERT cross-encoder and INT8 student"))
    ot = lambda x: x
    rows = [["Field", "Value"],
            ["Model details", f"answerdotai/ModernBERT-base cross-encoder, {cfg['params']:,} parameters, three heads (alignment, 3-way drift type, confidence) on the pooled representation; input is consolidated intent plus hunk text, max length {cfg['max_len']}. Deployed variant: 22M-parameter MiniLM student distilled (tau {cfg['tau']}, alpha {cfg['alpha']}) and exported to INT8 ONNX."],
            ["Intended use", "Local triage: rank the hunks of one diff by misalignment probability so a human reviews the likeliest ones first, with calibrated confidence and token attributions."],
            ["Out-of-scope use", "Automatic accept/reject of changes; security or correctness verification; languages other than Python/JavaScript/TypeScript; decisions about people."],
            ["Training data", f"Sift-IA v1.0 train split ({cfg['n_train_raw']:,} rows); {cfg['n_train_used']:,} used after rule filters (removed {cfg['rule_removed_non_semantic']} non-semantic, {cfg['rule_removed_duplicates']} duplicate) and confident-learning removal ({cfg['cl_removed']:,}). Validation used for threshold, early stopping and temperature."],
            ["Training settings", f"AdamW, lr {cfg['lr']}, micro-batch {cfg['micro_bs']} x accumulation {cfg['accum']} (effective {cfg['effective_batch']}), up to {cfg['epochs']} epochs with early stopping (patience {cfg['patience']}) on validation macro-F1, focal loss, loss weights 0.6/0.3/0.1, bf16 autocast, gradient checkpointing, seed {cfg['seed']}. Best validation macro-F1 {f(run['best_val_macro_f1'])}; training time {run['train_time_s']/60:.0f} min on {cfg['gpu']} (peak {f(run['peak_gpu_mem_gb'],1)} GB). Context length was reduced from the planned 1,024 to {cfg['max_len']}. Hyperparameter search: {len(hpo['trials'])} TPE trials (planned {hpo['planned_trials_brief']}) on a {hpo['n_train']}-row subset, 2 epochs."],
            ["Metrics (test)", f"Main: macro-F1 {f(main['macro_f1'])}, precision {f(main['precision'])}, recall {f(main['recall'])}, specificity {f(main['specificity'])}, NDCG@5 {f(main['ndcg@5'])}, AUC random/hard {f(main['auc_random'])}/{f(main['auc_hard'])}. INT8 student: macro-F1 {f(i8['macro_f1'])}, NDCG@5 {f(i8['ndcg@5'])}, {f(i8['size_mb'],1)} MB."],
            ["Calibration", f"Temperature scaling, T = {f(cal['temperature'])}; test ECE {f(cal['ece_test_before'])} to {f(cal['ece_test_after'])}."],
            ["Limitations", "Single seed. Constructed labels with no human validation. Open misleading-comment weakness (D3, D4). Recall on misaligned hunks is below 0.5. No TypeScript in the test split. Confident learning found only a fraction of the known label flips (precision "
             f"{pct(cl['detection']['confident_learning']['precision'],0)}, recall {pct(cl['detection']['confident_learning']['recall'],0)} against the {cl['true_flips']} known flips), and the reduced noise-robustness study ({len(nr['results'])} runs) shows no consistent benefit from denoising."],
            ["Ethical considerations", "Built only on permissively licensed public code; no personal data was added. Risk: over-trust. A flagged or unflagged hunk is a ranking signal for a human reviewer, not a verdict, and false negatives are common."],
            ["How to run", "<font face='Courier' size='7'>PYTHONPATH=src .venv/bin/python -m sift.infer --help</font> for the CLI; the web workspace and API are in app/. Checkpoints are in artifacts/."]]
    S.append(table(rows, [3.4 * cm, 14 * cm]))
    S.append(P("Table 7: Model Card (experiments/runs/full/config.json and results.json, results/calibration.json, results/hpo_small.json, results/confident_learning.json, results/noise_robustness.json).", R.CAP))

    # 9
    S.append(H("9. Context"))
    S.append(P(f"<b>Use case:</b> a developer reviews an AI agent's diff and wants the hunks that do not serve the stated request first. <b>Environment:</b> everything runs locally; the deployed target is the INT8 student on CPU. Measurements were made on {hw} ({lf['cpu_info']['logical_cores']} logical cores) and a {gpu} used for training; no other hardware was tested, so the latency figures should not be generalised. No network service or database is required."))

    # 10
    S.append(H("10. Defect register summary"))
    rows = [["ID", "Severity", "Description", "Status"]]
    for d in ts["defects"]:
        rows.append([d["id"], d["severity"], d["description"].replace("<", "&lt;").replace(">", "&gt;"), d["status"]])
    S.append(table(rows, [1.7 * cm, 1.7 * cm, 11.4 * cm, 2.6 * cm]))
    S.append(P(f"Table 8: all {len(ts['defects'])} defects are open and recorded as strict xfail tests (results/test_summary.json). Mitigations are given in the Testing document.", R.CAP))

    # 11
    S.append(H("11. Scope and limitations"))
    S.append(P("Scope notes that apply to the reported numbers: one seed (473) per run; labels are construction-determined with no human validation; the test set is repository-disjoint but is open-source Python/JavaScript only; latency and size are valid only for the machine measured; noise-robustness and baselines used reduced settings; the hyperparameter search ran "
               f"{len(hpo['trials'])} trials, not the planned {hpo['planned_trials_brief']}; the UniXcoder comparison is outside this work; drift and bias were not measured, and the rules-only denoising ablation is outside this work; the GAN-style baseline is our interpretation of the GAN comparison requirement."))
    S.append(H("12. Conclusion"))
    S += bullets([
        f"The ModernBERT cross-encoder has the best hard-tier pairwise AUC ({f(main['auc_hard'])}) and clearly beats the non-neural baselines there (best {g('B4 TF-IDF dissimilarity','auc_hard')}). It does <b>not</b> win on macro-F1, where B5 and several ablated variants score higher, and a frozen-embedding cosine baseline matches it closely on ranking.",
        f"The deployed INT8 student is {f(i8['size_mb'],1)} MB and keeps macro-F1 {f(i8['macro_f1'])} and NDCG@5 {f(i8['ndcg@5'])}, close to the full model, at a CPU p50 of {f(o8['latency_ms']['p50_ms'],1)} ms (shared-CPU measurement).",
        "Individual components (consolidation, focal loss, multi-task heads, denoising) are not shown to help in this single-seed study; only the use of both intent and hunk is clearly supported.",
        "Not done: multiple seeds, human-validated labels, the full 60-trial hyperparameter search, UniXcoder, full-protocol noise robustness, drift and bias measurement, agent-authored sessions, TypeScript test evidence, and fixes for open defects D1-D4.",
        "Novelty: none of the methods is new; the contribution is the task framing, the pipeline, and the dataset."])

    out = R.build(R.OUT / "23037_review2_documentation.pdf", S, "Review 2 Documentation")
    tr = ROOT / "reports" / "number_trace_review.json"
    old = json.load(open(tr)) if tr.exists() else {}
    t = lambda *a: sorted(a)
    old["review"] = {
        "Table 1 and Figure 1": ["results/master_comparison.csv"],
        "Table 2 latency": ["results/latency_full.json", "results/latency_size_onnx.json"],
        "Other dimensions": ["results/xai.json", "results/calibration.json", "results/test_summary.json", "experiments/results/data_verification.json"],
        "Table 3 and Table 8": ["results/test_summary.json"],
        "Table 4": ["results/xai.json"], "Table 5": ["results/ablation.csv"],
        "Table 6 Data Card": ["data/dataset_statistics.json", "data/source_repositories.csv", "experiments/results/data_verification.json", "experiments/results/DISCREPANCIES.md (prose)"],
        "Table 7 Model Card": ["experiments/runs/full/config.json", "experiments/runs/full/results.json", "results/calibration.json", "results/hpo_small.json", "results/confident_learning.json", "results/noise_robustness.json"]}
    json.dump(old, open(tr, "w"), indent=2)
    return out


if __name__ == "__main__":
    print(build())
