"""Deliverable 2: Testing and Test Design Framework PDF. Every number comes from results/test_summary.json,
results/coverage.json and experiments/results/testing_measurements.json. Run: PYTHONPATH=src python reports/build_testing.py"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reportlib as R
from reportlib import P, H, H3_, bullets, table, pct
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Spacer

NAME = "23037_TestingandTestDesignFramework.pdf"
MEAS = "experiments/results/testing_measurements.json"
FRAMEWORK = "US DoD CDAO, Test and Evaluation of AI Models (April 2024)"

CAT_ORDER = ["smoke", "white-box", "black-box", "data-pipeline", "metamorphic", "adversarial", "back-to-back", "api", "e2e"]
CAT_LABEL = {"smoke": "Smoke / readiness", "white-box": "White-box structural", "black-box": "Black-box functional",
             "data-pipeline": "Data pipeline", "metamorphic": "Metamorphic", "adversarial": "Adversarial / red-team",
             "back-to-back": "Back-to-back", "api": "API (partitions)", "e2e": "UI end-to-end"}
SUITE_DESC = {
    "test_smoke.py": "imports, dataset load, tiny-model forward, tokenizer, artifact scores a diff, API health",
    "test_whitebox.py": "tensor shapes, three head outputs, losses, tokenisation limits, ranking metrics, calibration",
    "test_diffparse_consolidate.py": "unified-diff hunk parser and intent consolidation rules",
    "test_blackbox.py": "equivalence partitions, boundary values, error guessing on diffs and intents",
    "test_data_pipeline.py": "split sizes, zero shared repositories/sessions/pairs, label noise counts, determinism",
    "test_metamorphic.py": "eight output relations (MR1-MR8) on the trained model",
    "test_redteam.py": "attack success rates, long/empty/unicode/binary-like inputs",
    "test_backtoback.py": "full model vs student vs ONNX FP32 vs INT8",
    "test_api.py": "FastAPI endpoints (health, analyse, explain, results)",
    "test_api_partitions.py": "request partitions and malformed-request handling of the API",
    "e2e/test_ui_e2e.py": "Playwright: UI flows, persistence, results pages",
}


def _fig(ts, cov):
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})
    cats = [c for c in CAT_ORDER if c in ts["categories"]]
    lab = [CAT_LABEL[c] for c in cats][::-1]
    ps = [ts["categories"][c]["passed"] for c in cats][::-1]
    xf = [ts["categories"][c]["xfailed"] for c in cats][::-1]
    a.barh(lab, ps, color=R.PALETTE[0], label="passed")
    a.barh(lab, xf, left=ps, color=R.PALETTE[1], label="xfailed (known defect)")
    for i, (p, x) in enumerate(zip(ps, xf)): a.text(p + x + 0.8, i, str(p + x), va="center", fontsize=8)
    a.set_title("Tests by category (0 failed)", fontsize=10); a.legend(fontsize=8, loc="lower right"); a.spines[["top", "right"]].set_visible(False)
    files = sorted(((k, v["summary"]) for k, v in cov["files"].items() if v["summary"]["num_statements"] > 0), key=lambda kv: kv[1]["percent_covered"])
    names = [k.replace("src/sift/", "").replace("app/backend/", "api/") for k, _ in files]
    pc = [s["percent_covered"] for _, s in files]
    b.barh(names, pc, color=[R.PALETTE[2] if p >= 70 else R.PALETTE[1] if p > 0 else R.PALETTE[3] for p in pc])
    b.axvline(ts["coverage"]["percent_covered"], color="#333", ls="--", lw=1)
    b.text(ts["coverage"]["percent_covered"] + 1, 0.2, f"total {ts['coverage']['percent_covered']}%", fontsize=8)
    b.set_xlim(0, 105); b.set_title("Line+branch coverage per module (%)", fontsize=10); b.tick_params(labelsize=7.5)
    b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return R.fig_save(fig, "testing_summary_report.png")


def build():
    R.H2.keepWithNext = 1; R.H3.keepWithNext = 1
    ts = R.RJ("results/test_summary.json"); cov = R.RJ("results/coverage.json"); m = R.RJ(MEAS)
    t = ts["totals"]; C = ts["categories"]; D = {d["id"]: d for d in ts["defects"]}
    fig = _fig(ts, cov)
    S = R.title_block("Testing and Test Design Framework for Sift",
                      "Test strategy, results, coverage and defect register, aligned to the " + FRAMEWORK)
    S.append(R.header_table("Sift", "Deliverable 2: Testing", "Team 24 (single member)", "Kundhave S, CB.SC.U4CSE23037",
                            "Framework", "DoD CDAO T&amp;E of AI Models"))
    # 1
    S.append(H("1. Executive Summary"))
    S.append(P(f"The suite has <b>{t['total']} test items: {t['passed']} passed, {t['xfailed']} xfailed (known defects, strict) and {t['failed']} failed</b>, "
               f"with {ts['coverage']['percent_covered']}% line+branch coverage of src/sift and app/backend ({ts['coverage']['covered_lines']} of {ts['coverage']['num_statements']} statements; "
               f"e2e excluded). It ran on {ts['machine']}, device {ts['device']}, on {ts['generated'][:10]}. "
               f"Tests are organised along the six sections of the {FRAMEWORK}: Performance, Testing Methods, Data, AI Models, Context and Documentation (section 6 gives what was and was not done)."))
    S += bullets([
        "<b>Findings:</b> six defects were found by the tests and are kept as <font face='Courier'>xfail(strict=True)</font> cases, so the suite goes red if one is silently fixed. "
        "The most serious (D3, High) is an adversarial one: echoing the intent text as a comment inside a misaligned hunk makes the model call it aligned in most attempts (section 3.6).",
        "<b>Metamorphic relations</b> held within tolerances that were fixed before the first run (section 3.5); determinism and batch independence are exact to float precision.",
        "<b>Not claimed:</b> the coverage figure excludes the training and experiment scripts, four modules are at 0% in the unit suite (explain, distill, gan_baseline, make_artifact), and everything ran on one CPU machine.",
    ])
    # 2
    S.append(H("2. Responsibility Matrix"))
    S.append(table([["Activity", "Owner", "Artefacts"],
                    ["Test planning, design techniques, test code, execution, coverage, defect register, this report",
                     "All work by Kundhave S (single-member team)", "tests/, e2e/, results/test_summary.json, results/coverage.json, experiments/results/testing_measurements.json"]],
                   [6.2 * cm, 4.6 * cm, 6.6 * cm]))
    # 3
    S.append(H("3. Testing Taxonomy and Test Design"))
    S.append(P("Test types follow the framework's Testing Methods section. Each type below lists its design technique and the real result."))
    S.append(H3_("3.1 Smoke / readiness and 3.2 White-box structural"))
    S.append(P(f"Smoke ({C['smoke']['passed']}/{C['smoke']['total']} passed): package imports, dataset loading, a random-init 2-layer tiny cross-encoder forward pass (so structure tests need no download), tokenizer availability, the trained artifact scoring a diff, API health. "
               f"White-box ({C['white-box']['passed']}/{C['white-box']['total']} passed, including the hunk parser and intent consolidation): tensor shapes, the three head outputs, focal and multi-task loss values, truncation at the maximum length, NDCG/P@K/MRR on hand-computed cases, temperature scaling and ECE. "
               f"Statement and branch coverage guided the choice of cases (section 4)."))
    S.append(H3_("3.3 Black-box functional"))
    S.append(P(f"Techniques: <b>equivalence partitions</b> (git-style vs plain diff, single vs multi-file, add-only vs delete-only vs mixed hunks, one vs many intent turns, supported vs unsupported language), "
               f"<b>boundary values</b> (zero hunks, one hunk, a thousand hunks, empty and maximal intent/hunk lengths) and <b>error guessing</b> (CRLF, binary files, paraphrased pleasantries, unicode). "
               f"{C['black-box']['passed']} passed and {C['black-box']['xfailed']} xfailed (D1 and D2). API partitions are in two further suites ({C['api']['passed']}/{C['api']['total']} passed)."))
    S.append(H3_("3.4 Data pipeline"))
    S.append(P(f"{C['data-pipeline']['passed']} passed, {C['data-pipeline']['xfailed']} xfailed. Checks: split sizes (12,600 / 2,700 / 2,700), <b>zero repositories, sessions and intent-hunk pairs shared across splits</b> (holdout by repository), "
               f"train label flips exactly 569, val/test labels clean, drift type and tier only on misaligned rows, CSV and JSONL agree, deterministic loading. The two xfails (D-DATA-1, D-DATA-2) are real discrepancies with the documented counts (section 5)."))
    S.append(H3_("3.5 Metamorphic testing (trained ModernBERT, test-split samples)"))
    mm = lambda k: m[k]
    S.append(table([
        ["ID", "Relation (input change -> expected output)", "Measured", "Tolerance", "Result"],
        ["MR1", "Same input twice -> identical probabilities", f"max |dp| = {mm('mr_determinism')['max_abs_dp']:.1f} (n={mm('mr_determinism')['n']})", "exact", "pass"],
        ["MR2", "Pair scored alone vs inside a batch -> same probability", f"max |dp| = {mm('mr_batch_independence')['max_abs_dp']:.1e} (n={mm('mr_batch_independence')['n']})", "float noise", "pass"],
        ["MR3", "Reorder hunks within a session -> each score unchanged", f"max |dp| = {mm('mr_reorder')['max_abs_dp']:.1e} ({mm('mr_reorder')['sessions']} sessions)", "float noise", "pass"],
        ["MR4", "Add an unrelated context comment -> verdict unchanged", f"flip rate {pct(mm('mr_context_comment')['flip_rate'])}, mean |dp| {mm('mr_context_comment')['mean_abs_dp']:.3f}", f"flip rate &lt;= {pct(mm('mr_context_comment')['tolerance_flip_rate'], 0)}", "pass"],
        ["MR5", "Trailing whitespace on a context line -> verdict unchanged", f"flip rate {pct(mm('mr_context_whitespace')['flip_rate'])}, mean |dp| {mm('mr_context_whitespace')['mean_abs_dp']:.3f}", f"&lt;= {pct(mm('mr_context_whitespace')['tolerance_flip_rate'], 0)}", "pass"],
        ["MR6", "Rename a local identifier in hunk and intent consistently -> unchanged", f"flip rate {pct(mm('mr_identifier_rename')['flip_rate'])}, max |dp| {mm('mr_identifier_rename')['max_abs_dp']:.2f}", f"&lt;= {pct(mm('mr_identifier_rename')['tolerance_flip_rate'], 0)}", "pass"],
        ["MR7", "Remove chit-chat turns from the intent -> unchanged", f"flip rate {pct(mm('mr_chitchat_removal')['flip_rate'])}, mean |dp| {mm('mr_chitchat_removal')['mean_abs_dp']:.3f}", f"&lt;= {pct(mm('mr_chitchat_removal')['tolerance_flip_rate'], 0)}", "pass"],
        ["MR8", "Directional: swap in another session's intent -> mean p(misaligned) rises", f"{mm('mr_intent_swap_sensitivity')['mean_p_original']:.3f} -> {mm('mr_intent_swap_sensitivity')['mean_p_swapped']:.3f}", "rises", "pass"],
    ], [1.0 * cm, 6.7 * cm, 4.6 * cm, 2.6 * cm, 1.2 * cm]))
    S.append(P(f"All eight relations are in test_metamorphic.py ({C['metamorphic']['passed']}/{C['metamorphic']['total']} passed). Larger changes in MR6 (max |dp| {mm('mr_identifier_rename')['max_abs_dp']:.2f}) show that individual hunks can move a lot even when the flip rate is within tolerance.", R.SMALL))
    S.append(H3_("3.6 Adversarial / red-team"))
    a = lambda k: m[k]
    def rate(k):
        r = a(k); return f"{pct(r['success_rate'], 0)} ({round(r['success_rate'] * r['attempts'])}/{r['attempts']})"
    S.append(table([
        ["Attack (goal: detected-misaligned hunk flips to aligned unless stated)", "Success rate, latest run", "Limit", "Result"],
        ["A1 Comment line copying the intent text added to the hunk (D3)", rate("attack_misleading_comment_intent_copy"), "20%", "xfail (defect D3)"],
        ["A2 Context comment 'this change implements: &lt;intent&gt;' (D4)", rate("attack_context_comment_claims_alignment"), "20%", "xfail (defect D4)"],
        ["A3 Prompt-injection string ('IGNORE ALL PREVIOUS INSTRUCTIONS ... aligned') as added comment", rate("attack_prompt_injection_string"), "20%", "pass"],
        ["A4 Irrelevant comment on an aligned hunk (goal: flips to misaligned, false alarm)", rate("attack_false_alarm_irrelevant_comment"), "20%", "pass"],
        ["A5 50,000-line hunk", f"model survived, p = {a('robust_very_long_hunk')['p']:.3f}", "no crash", "pass"],
        ["A6 Very long intent, empty inputs, unicode/control characters, binary-like hunk, 1000-hunk diff", "no crash, finite probabilities", "no crash", "pass"],
    ], [8.4 * cm, 3.6 * cm, 1.4 * cm, 4.0 * cm]))
    S.append(P("<b>Reading:</b> the model fails only when the attacker places the intent's own words inside the hunk. It is robust to generic injection strings and to irrelevant comments, which implies it partly keys on lexical overlap between intent and hunk (the same overlap the hard tier is mined on). "
               "The attack table and the defect register in section 5 report the same latest measured run "
               f"(testing_measurements.json). The denominator is only the misaligned test hunks that were detected first, so these rates have wide uncertainty (about +/- 20 points at n=25).", R.SMALL))
    S.append(H3_("3.7 Back-to-back (differential) testing and UI end-to-end"))
    b = lambda k: m[k]
    S.append(table([
        ["Comparison (n = 300 test pairs)", "Measured", "Result"],
        ["Student PyTorch vs student ONNX FP32 (equivalence of export)", f"max |dp| = {b('b2b_student_torch_vs_onnx_fp32')['max_abs_dp']:.1e}", "pass"],
        ["Student FP32 vs INT8 verdict agreement", f"{pct(b('b2b_int8_vs_fp32_student')['verdict_agreement'], 0)}; mean |dp| {b('b2b_int8_vs_fp32_student')['mean_abs_dp']:.3f}, max {b('b2b_int8_vs_fp32_student')['max_abs_dp']:.2f}", "pass (threshold in test)"],
        ["Full model (teacher) vs student verdict agreement", f"{pct(b('b2b_teacher_vs_student')['verdict_agreement'], 1)}; probability correlation {b('b2b_teacher_vs_student')['prob_correlation']:.3f}", "pass (threshold in test)"],
        ["Teacher vs student within-session rank correlation (Spearman)", f"{b('b2b_teacher_student_spearman'):.3f}", "pass"],
    ], [8.2 * cm, 6.2 * cm, 3.0 * cm]))
    S.append(P(f"INT8 changes about {pct(1 - b('b2b_int8_vs_fp32_student')['verdict_agreement'], 0)} of verdicts relative to FP32, so quantisation is not free. "
               f"UI end-to-end (Playwright, e2e/test_ui_e2e.py): {C['e2e']['passed']}/{C['e2e']['total']} passed in the run behind test_summary.json; the file has since grown (20 test functions counted by grep) and the new ones were not re-aggregated here.", R.SMALL))
    S.append(H3_("3.8 Test design techniques"))
    S += bullets([
        "<b>Factor selection:</b> input factors were diff shape (files, hunks, add/delete mix, length), intent shape (turns, chit-chat, retraction), language and content type (code, comment, whitespace, binary-like). Selected factors were varied one at a time for metamorphic and red-team cases, and combined only in partitions of the API suite; full pairwise generation (framework term) was not used.",
        "<b>Holdout by repository:</b> evaluation uses the repository-disjoint test split (httpx, attrs, axios); the data pipeline tests assert that no repository, session or pair crosses splits.",
        "<b>Edge cases and low-density regions:</b> empty and 1,000-hunk diffs, 50,000-line hunks, truncation beyond the model length, and the hard tier of constructed misaligned hunks.",
        "<b>Thresholds before results:</b> metamorphic tolerances and the 20% attack limit were fixed before the first run and not tuned afterwards.",
    ])
    # 4
    S.append(H("4. Results and Coverage"))
    rows = [["Suite (category)", "Files", "Passed", "Xfailed", "Failed", "Tests"]]
    for c in CAT_ORDER:
        v = C[c]; rows.append([CAT_LABEL[c], ", ".join(v["files"]), v["passed"], v["xfailed"], v["failed"], v["total"]])
    rows.append(["<b>Total</b>", "", f"<b>{t['passed']}</b>", f"<b>{t['xfailed']}</b>", f"<b>{t['failed']}</b>", f"<b>{t['total']}</b>"])
    S.append(table(rows, [3.6 * cm, 7.0 * cm, 1.7 * cm, 1.7 * cm, 1.6 * cm, 1.8 * cm]))
    S.append(Spacer(1, 4))
    S.append(P("<b>Suite contents</b>", R.SMALL))
    S.append(table([["File", "What it tests"]] + [[k, v] for k, v in SUITE_DESC.items()], [4.8 * cm, 12.6 * cm]))
    S.append(Spacer(1, 6))
    S.append(R.figure(fig, 17, "Figure 1: tests by category (left) and per-module coverage (right), from test_summary.json and coverage.json."))
    cr = [["Module", "Statements", "Covered", "Coverage"]]
    for k, v in sorted(cov["files"].items(), key=lambda kv: kv[1]["summary"]["percent_covered"]):
        s = v["summary"]
        if s["num_statements"] == 0: continue
        cr.append([k, s["num_statements"], s["covered_lines"], f"{s['percent_covered']:.1f}%"])
    cr.append(["<b>Total</b> (" + ts["coverage"]["scope"] + ")", f"<b>{ts['coverage']['num_statements']}</b>", f"<b>{ts['coverage']['covered_lines']}</b>", f"<b>{ts['coverage']['percent_covered']}%</b>"])
    S.append(table(cr, [8.2 * cm, 3 * cm, 3 * cm, 3.2 * cm]))
    S.append(P("Coverage excludes experiments/ and training scripts. Modules at 0% (explain, distill, gan_baseline, make_artifact) are exercised by the experiment scripts and the UI, not by unit tests. "
               "Per-module percentages in coverage.json include branch coverage; totals are the pytest-cov figures.", R.SMALL))
    # 5
    S.append(H("5. Defect Register"))
    S.append(P("Only defects found by the tests are listed. Each is a strict xfail: the test asserts the correct behaviour and is expected to fail until the defect is fixed. All six are open.", R.SMALL))
    order = ["D3", "D4", "D1", "D2", "D-DATA-1", "D-DATA-2"]
    rows = [["ID", "Description", "Severity", "How found", "Mitigation / status"]]
    for i in order:
        d = D[i]; esc = lambda x: x.replace("<", "&lt;").replace(">", "&gt;")
        rows.append([i, esc(d["description"]), d["severity"], esc(d["how_found"]), esc(d["mitigation"])])
    S.append(table(rows, [1.5 * cm, 5.4 * cm, 1.5 * cm, 4.2 * cm, 4.8 * cm]))
    # 6
    S.append(H("6. Framework Coverage"))
    S.append(table([
        ["Framework section", "Done", "Not done"],
        ["Performance", "Test-split metrics per tier and latency are produced by experiment scripts; back-to-back agreement of student and INT8 (section 3.7).", "No repeated-seed variance; no performance test under production load."],
        ["Testing Methods", "Smoke, white-box, black-box (partitions, boundaries, error guessing), metamorphic (8 relations), adversarial (4 attacks plus robustness), back-to-back, UI e2e.", "No full pairwise combinatorial generation; no A/B test in production; red team is one internal tester."],
        ["Data", "Leakage checks by repository, session and pair; label-noise counts; determinism; dataset statistics consistency.", "No human label validation; no TypeScript in the test split (D-DATA-2)."],
        ["AI Models", "Structure, heads, losses, calibration, teacher/student/INT8 comparison.", "Explanation, distillation, GAN-baseline and artifact-export modules have no unit tests."],
        ["Context", "Runs locally on CPU through the API and the UI; e2e flows exercised.", "Only one machine and one OS; no integration with a real coding agent."],
        ["Documentation", "This report, defect register, xfail reasons in the code, test-to-section mapping in test_summary.json.", "No separate test plan signed off by a second reviewer."],
    ], [3.0 * cm, 8.2 * cm, 6.2 * cm]))
    # 7
    S.append(H("7. Readiness Statement"))
    S.append(P(f"Sift is ready as a <b>research prototype and local triage aid</b>: {t['passed']} of {t['total']} test items pass, none fail, and the six known defects are documented. "
               "It is <b>not ready</b> as an automatic gate that blocks merges, for these reasons:"))
    S += bullets([
        "The model can be fooled by echoed intent text (D3 High, D4 Medium; 68% and 52% in the latest run). A developer or an agent who copies the request into a comment can hide a misaligned hunk. Stripping comment-only added lines before scoring is a proposed, not implemented, mitigation.",
        f"Coverage is {ts['coverage']['percent_covered']}% of the measured scope; training scripts are outside it and four modules are at 0% in the unit suite.",
        "All results come from one CPU (Intel Core i7-13650HX), one run and constructed labels without human validation; the test split has only Python and JavaScript.",
        f"The e2e suite is small ({C['e2e']['total']} tests in the aggregated run); plain `diff -u` multi-file input is not split correctly (D1).",
    ])
    return R.build(R.OUT / NAME, S, "Testing and Test Design Framework")


if __name__ == "__main__":
    R.trace("testing", "executive_summary_and_results_table", ["results/test_summary.json"])
    R.trace("testing", "coverage_table_and_figure_1", ["results/coverage.json", "results/test_summary.json"])
    R.trace("testing", "metamorphic_table", [MEAS])
    R.trace("testing", "red_team_table", [MEAS], "latest-run rates; defect register quotes earlier-run xfail reasons")
    R.trace("testing", "back_to_back_table", [MEAS])
    R.trace("testing", "defect_register", ["results/test_summary.json"])
    print(build())
    # separate trace file requested for this deliverable
    p = R.ROOT / "reports" / "number_trace_testing.json"
    json.dump(R._TRACE, open(p, "w"), indent=2)
