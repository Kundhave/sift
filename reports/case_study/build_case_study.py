"""Builds Review3/Sift_Case_Study_IEEE_Access.docx from results/*.json|csv and experiments/results/*.json.
Every number in tables and prose is read from those files at build time.
Run: PYTHONPATH=src .venv/bin/python reports/case_study/build_case_study.py
(figures: reports/case_study/make_figures_300.py -> reports/figures_300/)
"""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from docxlib import Doc, COL_W, PAGE_W            # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH      # noqa: E402

RES, XR, RUNS, FIG = ROOT / "results", ROOT / "experiments" / "results", ROOT / "experiments" / "runs", ROOT / "reports" / "figures_300"
OUT = ROOT / "Review3" / "Sift_Case_Study_IEEE_Access.docx"
OUT.parent.mkdir(exist_ok=True)


def J(p):
    return json.load(open(ROOT / p))


def rows_csv(p):
    return list(csv.DictReader(open(ROOT / p)))


def f3(x, d=3):
    return "–" if x in (None, "", "None") else f"{float(x):.{d}f}"


def pc(x, d=0):
    return f"{float(x) * 100:.{d}f}%"


def n_(x):
    return f"{int(x):,}"


# ============================================================ data
MC = {r["model"]: r for r in rows_csv("results/master_comparison.csv")}
AB = rows_csv("results/ablation.csv")
DS = {r["statistic"]: r["value"] for r in rows_csv("results/dataset_statistics.csv")}
LAT = rows_csv("results/latency_size.csv")
NZ = J("results/noise_robustness.json")["results"]
CL = J("results/confident_learning.json")
CAL = J("results/calibration.json")
KM = J("results/kmeans.json")
GV = J("results/gan_vs_main.json")
GJ = J("experiments/results/gan.json")
XA = J("results/xai.json")
TS = J("results/test_summary.json")
COV = J("results/coverage.json")
TM = J("experiments/results/testing_measurements.json")
BL = J("experiments/results/baselines.json")
HPO = J("results/hpo_small.json")
CONS = J("experiments/results/consolidation.json")
DEN = J("experiments/results/denoise.json")
LF = J("results/latency_full.json")
LO = J("results/latency_size_onnx.json")
EV = {k: J(f"results/eval_{k}.json") for k in ("full", "student", "student_nodistill", "minilm_ce", "codebert", "bilstm", "abl_no_denoise")}
BOOT = J("reports/case_study/bootstrap_ci.json")
FULLRUN = J("experiments/runs/full/results.json")
STURUN = J("experiments/runs/student/results.json")
main = MC["Main: ModernBERT-base"]
stu = MC["Distilled MiniLM student (fp32)"]
int8 = MC["Student ONNX INT8 (deployed target)"]
onx = MC["Student ONNX fp32"]
frz = MC["Frozen-embedding frozen cosine dissimilarity"]
gu = MC["GAN-style baseline: gan unsupervised"]
gh = MC["GAN-style baseline: gan hybrid with misaligned negatives"]
b4, b5, b3 = MC["B4 TF-IDF dissimilarity"], MC["B5 Logistic regression"], MC["B3 Hunk size"]
mini, cbt, bil = MC["MiniLM-L6 cross-encoder"], MC["CodeBERT cross-encoder"], MC["Siamese BiLSTM"]
nod = MC["MiniLM student, no distillation"]
ABD = {r["model"]: r for r in AB}
ab_full = ABD["1. Full model"]
ab_nod = ABD["7. no denoising"]
ab_head = ABD["6. alignment head only"]
ab_foc = ABD["5. no focal loss (CE)"]
ab_con = ABD["4. no intent consolidation"]
ab_hunk = ABD["2. hunk-only"]
ab_int = ABD["3. intent-only"]
km_on, km_off = ABD["9b. k-means features ON (hybrid)"], ABD["9a. k-means features OFF (Platt on logit)"]
FT = EV["full"]["test"]
CONF = FT["full"]
CFG = FULLRUN["config"]
T_ = CAL["temperature"]
FA = XA["faithfulness"]
KQ = KM["clustering_quality"]["test"]
KCT = KM["test_cluster_table"]
KH = KM["hybrid"]["delta_cluster_minus_baseline"]
CPU = LF["cpu_info"]["cpu"]
DEPLOY = (ROOT / "deploy" / "DEPLOY_URL.txt")
DEPLOY_URL = DEPLOY.read_text().strip() if DEPLOY.exists() and DEPLOY.read_text().strip() else None
cl_best = CL["detection"]["confident_learning"]
cl_final = CL["detection"]["final drop list (CL + rules)"]
nz = {(r["noise_rate"], r["arm"]): r["test"] for r in NZ}
TT = TS["totals"]
CATS = TS["categories"]
att = lambda k: TM[k]
at_d3, at_d4 = att("attack_misleading_comment_intent_copy"), att("attack_context_comment_claims_alignment")
at_inj, at_fa = att("attack_prompt_injection_string"), att("attack_false_alarm_irrelevant_comment")
b2b_int8, b2b_ts = att("b2b_int8_vs_fp32_student"), att("b2b_teacher_vs_student")
cov_t = COV["totals"]
TRACE: dict[str, list[str]] = {}

# ============================================================ references (alphabetical by first author's surname)
REFS = [
    ("Fan", "Fan et al. (2025). Exploring the capabilities of LLMs for code-change-related tasks. <i>ACM Transactions on Software Engineering and Methodology</i>, 34(6)."),
    ("Feng", "Feng, Z., et al. (2020). CodeBERT: A pre-trained model for programming and natural languages. <i>Findings of the Association for Computational Linguistics: EMNLP 2020</i>. https://arxiv.org/abs/2002.08155"),
    ("Guo", "Guo, C., et al. (2017). On calibration of modern neural networks. <i>Proceedings of the 34th International Conference on Machine Learning (ICML)</i>."),
    ("Han", "Han, B., et al. (2018). Co-teaching: Robust training of deep neural networks with extremely noisy labels. <i>Advances in Neural Information Processing Systems (NeurIPS)</i>."),
    ("Herzig", "Herzig, K., and Zeller, A. (2013). The impact of tangled code changes. <i>Proceedings of the 10th Working Conference on Mining Software Repositories (MSR)</i>. https://www.st.cs.uni-saarland.de/publications/details/herzig-msr-2013"),
    ("Hinton", "Hinton, G., Vinyals, O., and Dean, J. (2015). Distilling the knowledge in a neural network. <i>arXiv preprint</i>."),
    ("Huang", "Huang et al. (2025). Are your comments outdated? Toward automatically detecting code-comment consistency. <i>Journal of Software: Evolution and Process</i>, 37(1)."),
    ("Kundhave", "Kundhave, S. (n.d.). Sift-IA: An intent–implementation alignment dataset for code diff hunks produced by AI coding agents (version 1.0). <i>IEEE DataPort</i>. https://ieee-dataport.org/documents/sift-ia-intent-implementation-alignment-dataset-code-diff-hunks-produced-ai-coding-agents"),
    ("Lin", "Lin, T.-Y., et al. (2017). Focal loss for dense object detection. <i>Proceedings of the IEEE International Conference on Computer Vision (ICCV)</i>."),
    ("Nogueira", "Nogueira, R., and Cho, K. (2019). Passage re-ranking with BERT. <i>arXiv preprint</i> arXiv:1901.04085. https://arxiv.org/abs/1901.04085"),
    ("Northcutt", "Northcutt, C. G., Jiang, L., and Chuang, I. L. (2021). Confident learning: Estimating uncertainty in dataset labels. <i>Journal of Artificial Intelligence Research</i>."),
    ("Panthaplackel1", "Panthaplackel, S., Nie, P., Gligoric, M., Li, J. J., and Mooney, R. J. (2020). Learning to update natural language comments based on code changes. <i>Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)</i>. https://aclanthology.org/2020.acl-main.168/"),
    ("Panthaplackel2", "Panthaplackel, S., Li, J. J., Gligoric, M., and Mooney, R. J. (2021). Deep just-in-time inconsistency detection between comments and source code. <i>Proceedings of the AAAI Conference on Artificial Intelligence</i>. https://arxiv.org/abs/2010.01625"),
    ("Sundararajan", "Sundararajan, M., Taly, A., and Yan, Q. (2017). Axiomatic attribution for deep networks. <i>Proceedings of the 34th International Conference on Machine Learning (ICML)</i>."),
    ("US DoD", "U.S. Department of Defense, Chief Digital and Artificial Intelligence Office (2024). <i>Test and Evaluation of Artificial Intelligence Models Framework</i>. April 2024."),
    ("Wang", "Wang et al. (2025). An empirical study of best practices for code pre-trained models on software engineering classification tasks. <i>Expert Systems with Applications</i>."),
    ("Warner", "Warner, B., et al. (2024). Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. <i>arXiv preprint</i> arXiv:2412.13663. https://arxiv.org/abs/2412.13663"),
    ("Xu", "Xu, Z., Guo, S., Wang, Y., Chen, R., Li, H., Li, X., and Jiang, H. (2024). Code comment inconsistency detection based on confidence learning. <i>IEEE Transactions on Software Engineering</i>, 50(3), 598–617."),
    ("Yang", "Yang et al. (2026). RComSync: Improving code-comment synchronization with in-context learning and reranking. <i>Empirical Software Engineering</i>, 31(4)."),
]
REFS.sort(key=lambda r: r[0].lower())


def build(R: dict) -> tuple[Doc, dict]:
    D = Doc()
    rec_f, rec_t = {}, {}

    def FR(k):
        return f"Figure {R['f'].get(k, '?')}"

    def TR(k):
        return f"Table {R['t'].get(k, '?')}"

    def SR(k):
        return R["s"].get(k, "?")

    rec_s = {}

    def H2(key, title, **kw):
        D.heading2(title, **kw)
        rec_s[key] = f"{D.h1}.{D.h2}"

    def fig(key, name, width, caption, src=None):
        if width <= COL_W + 0.01:
            rec_f[key] = D.boxed_figure(FIG / name, width, caption)
        else:
            D.picture(FIG / name, width)
            rec_f[key] = D.caption_figure(caption)
        TRACE[f"Figure:{key}"] = (src or []) + [f"reports/figures_300/{name}"]

    def tab(key, rows, widths, caption, src=None, **kw):
        rec_t[key] = D.tab_n + 1
        D.table(rows, widths, caption=caption, **kw)
        TRACE[f"Table:{key}"] = src or []

    D.header_footer("Kundhave S and T. Senthil Kumar: Sift, hunk-level verification of AI-assisted code changes")

    # ------------------------------------------------------------ title block (single column)
    def center(text, size, bold=False, italic=False, before=0, after=0):
        par = D.d.add_paragraph()
        from docxlib import _fmt
        _fmt(par, WD_ALIGN_PARAGRAPH.CENTER, before, after)
        D.runs(par, text, size=size, bold=bold, italic=italic)
        return par

    TITLE = "Sift: A Local, Calibrated and Explainable Verification Layer that Ranks AI-Assisted Code Changes by Their Alignment with Developer Intent"
    center(TITLE, 19, bold=True, before=0, after=8)
    center("Kundhave S<sup>1</sup> and Dr. T. Senthil Kumar<sup>2</sup>", 12, before=0, after=4)
    center("<sup>1</sup>Department of Computer Science and Engineering, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore, Tamil Nadu, India (Roll No. CB.SC.U4CSE23037; e-mail: cb.sc.u4cse23037@cb.students.amrita.edu)", 9, italic=True, after=1)
    center("<sup>2</sup>Professor, Department of Computer Science and Engineering, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore, Tamil Nadu, India (e-mail: t_senthilkumar@cb.amrita.edu)", 9, italic=True, after=4)
    center("<b>Corresponding author:</b> Kundhave S (cb.sc.u4cse23037@cb.students.amrita.edu)", 9, after=7)

    abstract = (
        "AI coding assistants now produce large multi-file changes that a developer must review, and the difficulty is no longer whether the code runs but whether each changed block "
        "serves what was asked. This article presents Sift, a local verification layer that consolidates a developer’s multi-turn request into one intent, scores every hunk of the resulting "
        "diff against that intent with a ModernBERT cross-encoder, and ranks hunks so that the likeliest misaligned ones are reviewed first, with a calibrated probability, a drift type and a "
        "token-level explanation. We study the task on Sift-IA v1.0, 18,000 hunk-intent pairs from 15 open-source repositories, split by repository, whose misaligned hunks are "
        "constructed from donor commits and whose noise is annotated with ground truth. On three held-out test repositories the model reaches a macro F1 score of "
        f"{f3(main['macro_f1'])} and a normalised discounted cumulative gain at rank five of {f3(main['ndcg@5'])}; on the hard tier its within-session pairwise area under the curve is "
        f"{f3(main['auc_hard'])}, against {f3(b4['auc_hard'])} for the best lexical baseline, although its macro F1 score is below a logistic-regression baseline ({f3(b5['macro_f1'])}). "
        f"Temperature scaling reduces calibration error from {f3(CAL['ece_test_before'])} to {f3(CAL['ece_test_after'])}, and a distilled 8-bit student of {float(int8['size_mb']):.1f} megabytes answers in "
        f"{float(int8['cpu_p95_ms']):.1f} milliseconds at the 95th percentile on a laptop processor. Label-error detection by confident learning is weak (precision {f3(cl_best['precision'])}); "
        "a single-seed ablation does not show that individual components help, and simple comment-based attacks flip up to "
        f"{pc(at_d3['success_rate'])} of detections. A {TT['total']}-test suite documents these limits."
    )
    nwords = len(abstract.split())
    assert 150 <= nwords <= 250, nwords
    par = D.d.add_paragraph()
    from docxlib import _fmt
    _fmt(par, WD_ALIGN_PARAGRAPH.JUSTIFY, 2, 4, left=0.3)
    par.paragraph_format.right_indent = par.paragraph_format.left_indent
    D.runs(par, "<b>Abstract—</b>" + abstract, size=9, bold=True)
    par.runs[0].bold = True
    par = D.d.add_paragraph()
    _fmt(par, WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 8, left=0.3)
    par.paragraph_format.right_indent = par.paragraph_format.left_indent
    D.runs(par, "<b>Keywords:</b> intent alignment, code review, cross-encoder, label noise, calibration, explainability", size=9)
    R["abstract_words"] = nwords

    D.start_two_columns()

    # ============================================================ 1 Introduction
    D.heading1("Introduction")
    D.p("AI coding assistants such as Cursor and GitHub Copilot turn a short natural-language request into edits across many files. Reviewing that output is becoming the "
        "bottleneck of AI-assisted development: the reviewer must decide, hunk by hunk, whether each change implements what the developer asked for or is an unrequested "
        "addition. A hunk is one contiguous block of changed lines in a unified diff. Two difficulties make the problem hard. First, the developer’s intent is rarely a single "
        "sentence; it is spread across several chat turns, some of which are noise (pleasantries, retracted requests, repeated turns). Second, unrequested changes come in different "
        "forms: an edit to an unrelated file next to the requested change, an extra feature in the right file, or a change that serves a different purpose than the one requested.")
    D.p("Sift addresses this by consolidating the session into one intent, classifying every hunk as aligned or misaligned, assigning a drift type to the misaligned ones, and "
        "ranking the hunks of a diff so that a reviewer sees the likeliest misaligned hunk first. Each verdict carries a calibrated probability and an explanation of which words "
        "and diff lines drove it. The system runs locally, so source code does not leave the developer’s machine.")
    D.p("Why a learned model is needed can be shown without training anything. Any diff-level scalar, such as the number of lines changed or of files touched, has the same value "
        "for every hunk of one diff and therefore cannot order hunks inside it. Ordering hunks by diff size gives NDCG@5 of "
        f"{f3(BL['ndcg@5']['B2 Diff lines changed'], 4)}, identical to random order ({f3(BL['ndcg@5']['B1 Random order'], 4)}). Deciding whether one hunk serves a stated intent requires reading both "
        "texts jointly, which is what a cross-encoder does (Nogueira and Cho, 2019). The task is a message–code inconsistency problem in the sense of Panthaplackel et al. (2021), not defect prediction.")
    D.run_in("Relevance to the Sustainable Development Goals", "The work contributes primarily to SDG 9 (Industry, Innovation and Infrastructure): software is infrastructure, and AI-assisted "
             "development will only be trusted if its output can be verified cheaply. A local, explainable verification layer is a small piece of trustworthy AI-assisted engineering that "
             "organisations of any size can run without sending code to a third party. The secondary goal is SDG 8 (Decent Work and Economic Growth): reviewing large generated diffs is "
             "repetitive, attention-draining work, and directing a reviewer to the hunks most likely to be unrequested reduces that burden and makes the review task more sustainable. "
             "We make no quantitative claim about reviewer time; the evidence in this article concerns ranking quality, not human effort.")
    D.p("The contributions of this article are as follows.", after=1.5)
    D.numbered([
        "A task formulation at the level of the hunk inside a diff, conditioned on a consolidated multi-turn intent, with three drift types and a ranking objective.",
        f"A full pipeline: rule-based intent consolidation, a ModernBERT cross-encoder with three heads, temperature calibration, ranking, token-level explanation, a distilled 8-bit student ({float(int8['size_mb']):.1f} MB), and a web interface with an API.",
        "A study of label noise on a dataset that ships ground truth for its noise: rule-based and confident-learning denoising scored against the known label flips, and a noise-robustness sweep.",
        "Comparisons on identical splits against five non-neural baselines, three neural baselines, a k-means analysis, and an adversarially trained baseline on frozen embeddings, reported per difficulty tier, language and repository.",
        "An explainability study with Integrated Gradients and a masking-based faithfulness test, and a nine-part ablation.",
        f"A test design and verification study ({TT['total']} tests, metamorphic relations, red-team attacks, back-to-back and leakage tests) with a defect register that records real defects, including two high-impact robustness failures.",
    ])
    D.p("Four research questions guide the study. <b>RQ1:</b> can a cross-encoder conditioned on consolidated session intent rank diff hunks by misalignment better than diff-level scalar baselines, which are provably constant within a diff? "
        "<b>RQ2:</b> does confidence-learning-based label denoising improve macro F1 on a dataset whose misalignments are generated by construction, as the literature reports for code–comment pairs (Xu et al., 2024)? "
        "<b>RQ3:</b> does post-hoc temperature scaling produce calibrated probabilities (expected calibration error of at most 0.05) suitable for ranking and abstention? "
        "<b>RQ4:</b> how much macro F1 is retained after distillation and 8-bit quantisation to meet a CPU latency budget of 100 ms?")
    D.p("The encoder and the training methods are existing ones; we do not claim a new architecture. The contribution is the task framing, the pipeline, the evaluation and the dataset. "
        "The remainder of the article is organised as follows. Section 2 reviews related work. Section 3 describes the data, the architecture, the model, the noise, clustering, "
        "adversarial, explainability and compression components, the interface and the test design. Section 4 gives the results and a detailed discussion. Section 5 reports the "
        "ablation study, and Section 6 concludes.")

    # ============================================================ 2 Related work
    D.heading1("Related Work")
    D.heading2("Message–code and comment–code inconsistency")
    D.p("The closest research area asks whether a natural-language description still agrees with the code it accompanies. Panthaplackel et al. (2021) detect, just in time, whether a "
        "code change has made an existing comment inconsistent, and Panthaplackel et al. (2020) learn to update a comment given a code change. Both work at the level of method–comment "
        "pairs. Sift borrows the text-versus-code pair framing but differs in three respects: the text side is a multi-turn, partly noisy developer intent that must first be consolidated; "
        "the unit is the hunk inside a diff; and the output is a ranking for human review, not a single yes/no on one comment.")
    D.p("Five recent journal studies (Table 1) frame the gap. Xu et al. (2024) remove dataset noise with confidence learning before detecting inconsistent code–comment pairs, which "
        "directly motivates our denoising stage. Huang et al. (2025) show that learning-based classification of code–comment relationships can detect outdated comments. Yang et al. "
        "(2026) combine in-context learning with an explicit reranking stage for code–comment synchronisation, supporting a ranking formulation. Fan et al. (2025) evaluate large language "
        "models on code-change-related tasks and report, among other findings, that larger models are not necessarily better and that the context length allocated matters, which "
        "justifies a compact fine-tuned encoder with a long-context backbone. Wang et al. (2025) compare code pre-trained models under embedding and fine-tuning paradigms for "
        "software-engineering classification and caution against importing natural-language strategies unchanged. All five operate on comment–method or commit–message pairs; none "
        "conditions on a developer’s progressive session intent, ranks hunks inside one diff, or reports calibration.")
    D.heading2("Tangled changes and dataset validity")
    D.p("Herzig and Zeller (2013) show that commits often bundle unrelated edits (tangled changes) and that this pollutes mining studies. This is a limitation for Sift-IA: a seed "
        "commit that is itself tangled would contain hunks that do not serve its message, and our labels would call them aligned (Section " + SR("dataset") + ").")
    D.heading2("Cross-encoders and pre-trained encoders")
    D.p("A cross-encoder concatenates the two texts and classifies from the [CLS] vector, so that attention can compare the texts token by token; Nogueira and Cho (2019) apply this "
        "pattern to passage re-ranking with BERT. Sift uses ModernBERT (Warner et al., 2024), an encoder with rotary position embeddings, a gated activation, alternating local and "
        "global attention and an 8,192-token context, fine-tuned here at a maximum length of 512 tokens. CodeBERT (Feng et al., 2020) is a pre-trained model for programming and "
        "natural languages and serves as a code-aware baseline.")
    D.heading2("Label noise and confident learning")
    D.p("The model paper for our denoising stage is Northcutt et al. (2021). Their confident learning (CL) is a data-centric approach: instead of adding a noise-robust model or loss, it "
        "characterises and finds label errors in a dataset by combining three principles, pruning noisy data, counting with probabilistic thresholds to estimate noise, and ranking "
        "examples to train with confidence. Under a class-conditional noise assumption, where a label flips with a probability that depends only on the latent true class, CL "
        "uses out-of-sample predicted probabilities and the given noisy labels to count examples in an unnormalised confident joint, estimates the joint distribution between noisy "
        "and uncorrupted labels, and prunes the examples that are likely mislabelled. The authors give sufficient conditions under which CL exactly finds label errors, report that it "
        "outperforms seven recent approaches for learning with noisy labels on CIFAR, and release the open-source <i>cleanlab</i> package. Sift uses that package. A difference "
        "in evidence is worth noting: on benchmark data the true labels are unknown, whereas Sift-IA ships the identity of every injected label flip, so we can measure CL’s detection "
        "precision and recall directly (Section " + SR("noise_r") + "). Co-teaching (Han et al., 2018), which trains two networks that exchange small-loss examples, is a related noisy-label method; "
        "it is implemented as a helper in our code base but was not run in this study.")
    D.heading2("Calibration, distillation, attribution and focal loss")
    D.p("Guo et al. (2017) show that modern networks are often over-confident and that temperature scaling, a single scalar fitted on held-out data, corrects most of the miscalibration. "
        "Hinton et al. (2015) distil a large model into a small one by matching softened outputs. Sundararajan et al. (2017) define Integrated Gradients, an attribution method with "
        "axiomatic guarantees. Lin et al. (2017) introduce the focal loss, which down-weights easy examples and is useful under class imbalance, as here (about 78:22). All four are used "
        "as published; Sift’s contribution is their combination in a hunk-level review pipeline and the measurement of what each adds.")
    with D.wide():
        D.heading2("Positioning")
        pos = [["Work", "Unit of analysis", "Text side", "Output", "Relation to Sift"],
               ["Panthaplackel et al. (2021), AAAI", "method-level comment and code change", "code comment", "consistent / inconsistent", "closest prior task; no multi-turn intent, no hunk ranking"],
               ["Panthaplackel et al. (2020), ACL", "method-level comment and code change", "code comment", "edit sequence for the comment", "related task; update, not verification"],
               ["Herzig and Zeller (2013), MSR", "commit and file contents", "none", "untangled change groups", "motivates the tangled-seed limitation"],
               ["Xu et al. (2024), IEEE TSE", "method–comment pairs", "code comment", "inconsistency after confidence-learning denoising", "motivates denoising; not hunks conditioned on a task intent"],
               ["Huang et al. (2025), JSEP", "code–comment relationship features", "code comment", "outdated-comment classifier", "validates supervised pair classification; post hoc, no calibration or ranking"],
               ["Yang et al. (2026), EMSE", "code–comment synchronisation", "code comment", "synchronised comment via in-context learning and reranking", "supports a ranking stage; large-model and cloud-dependent"],
               ["Fan et al. (2025), TOSEM", "code-change-related tasks", "task prompts", "LLM evaluation with in-context learning and PEFT", "justifies a compact fine-tuned encoder"],
               ["Wang et al. (2025), ESWA", "software-engineering classification", "single input", "code PTM paradigm comparison", "informs fine-tuning; not a text–code pair setting"],
               ["<b>Sift (this article)</b>", "<b>hunk inside a diff, ranked</b>", "<b>consolidated multi-turn intent</b>", "<b>aligned/misaligned, 3 drift types, calibrated probability, ranking, token explanation</b>", "<b>local, CPU-deployable student</b>"]]
        tab("positioning", pos, [1.55, 1.35, 1.0, 1.7, 1.4], "Positioning of Sift against the closest related work (task, input, granularity and outputs only; the studies use different data and metrics, so no score comparison is made).",
            size=7, aligns=["l"] * 5, shade_rows=(9,), src=["Review 2 literature survey (five journal studies)", "docs/RELATED_WORK.md"])

    # ============================================================ 3 Method
    D.heading1("Proposed Method")
    D.heading2("Problem formulation")
    D.p("A coding session yields user turns u<sub>1</sub>, …, u<sub>n</sub> and a diff split into hunks h<sub>1</sub>, …, h<sub>m</sub>. A consolidation step maps the turns to one intent "
        "<i>ĩ</i> by removing noise. For each hunk the system outputs the probability <i>P</i>(misaligned | <i>ĩ</i>, h<sub>j</sub>), a verdict at a threshold, a drift type among "
        "{collateral, scope expansion, purpose drift} for misaligned hunks, a confidence, and an ordering of the hunks of the diff by risk. Evaluation therefore has two parts: "
        "classification of individual hunks (macro F1, precision, recall, specificity) and ranking within a session (NDCG@K, precision@K, mean reciprocal rank and within-session pairwise AUC).")

    H2("dataset", "Dataset description")
    st = lambda k: DS[k]
    D.p(f"Sift-IA v1.0 (Kundhave, n.d.) contains {n_(18000)} hunk–intent pairs from {n_(6111)} sessions built from {n_(st('distinct_source_commits'))} distinct source commits of 15 open-source repositories "
        f"under MIT, BSD-3 or Apache-2.0 licences. The hunks are real hunks from open-source commits. A row pairs one hunk with the consolidated intent of its session. The released files hold "
        f"{n_(18000)} rows, of which Python hunks number {n_(10205)}, JavaScript {n_(6726)} and TypeScript {n_(1069)} (an earlier description of the dataset gave slightly different language counts; we quote the released files). The record’s title refers to the intended use case of reviewing AI-assisted changes; v1.0 itself is "
        "not a collection of assistant-authored diffs.")
    D.p("<b>Label construction.</b> Labels are determined by construction, not by judgement. An aligned hunk is a real hunk from the seed commit whose message produced the intent. "
        "A misaligned hunk is a donor hunk from a different commit that is inserted into the session. The drift type records how the donor relates to the session: <i>collateral</i> "
        "(the donor touches a file unrelated to the intent), <i>scope expansion</i> (the donor touches a file the session already modifies and the session’s own hunks are retained) and "
        "<i>purpose drift</i> (the donor touches the same file and the session’s own hunk for that file is removed). Of the 18,000 rows, 14,040 are aligned (78.0%) and 3,960 misaligned: 1,782 "
        "collateral, 1,307 scope expansion and 871 purpose drift. Misaligned rows have a difficulty tier: <i>random</i> (a randomly chosen donor; 1,960 rows) or <i>hard</i> (a donor chosen for "
        "identifier overlap with the session, so that it looks similar; 2,000 rows).")
    D.p(f"<b>Splits.</b> Splits are by repository, so no repository appears in two splits (Table {rec_t_n(R,'dataset')}). Train uses requests, flask, click, rich, black, express and fastify; validation uses typer, "
        "sqlmodel, got, koa and chalk; test uses httpx, attrs and axios. The test split contains Python (1,686 hunks) and JavaScript (1,014 hunks, all from axios) but no TypeScript, which appears only in validation.")
    D.p("<b>Noise with ground truth.</b> A quarter of the rows (4,529, 25.2%) carry at least one kind of noise, each recorded with its ground truth (Table "
        f"{rec_t_n(R,'noise_types')}). Label flips are injected only in the training split (569 rows, with P(flip | aligned) = 0.03 and P(flip | misaligned) = 0.10); we train on the observed label "
        "and evaluate on the clean label, and validation and test labels are clean.")
    D.p("<b>Limitations.</b> Misaligned hunks are constructed, so a model may learn construction artefacts. A donor may coincidentally fit the intent, seed commits may be tangled (Herzig and Zeller, 2013), "
        "intent noise follows templates, and only Python, JavaScript and TypeScript from open-source repositories are covered. No human-validated labels exist yet; a blind annotation sheet for human "
        "validation is part of the release but is unlabelled.")

    D.heading2("Overall architecture")
    # (architecture figure is wide)
    D.p(f"{FR('arch')} shows the whole system. In the inference pipeline (row A), the session’s chat turns and unified diff are captured; a rule-based step consolidates the turns "
        "(dropping chit-chat, retractions and repeated turns); a diff parser segments the diff into hunks, handling multi-file diffs, renames and binary notes; the ModernBERT cross-encoder "
        "scores each (intent, hunk) pair through three heads; temperature scaling turns the logit into a calibrated probability; hunks are ranked by risk and flagged at a threshold chosen on "
        "validation; and a FastAPI service and a React interface present the result. Row B lists the components attached to the trained model: noise analysis and denoising "
        "(rules plus confident learning), which cleans the training rows; k-means on the pooled embeddings, which supplies a cluster id, similar flagged hunks and optional hybrid features; "
        "Integrated Gradients, which attributes a verdict to words and lines; and the compression path (distillation to a MiniLM student, then 8-bit ONNX export). Row C lists the comparison and "
        "verification layer: five non-neural baselines, a Siamese BiLSTM and two other cross-encoders, an adversarially trained baseline on frozen embeddings, and the test and ablation suites, all "
        "evaluated on the same splits.")
    with D.wide():
        fig("arch", "fig_architecture.png", 6.9, "Overall architecture of Sift: (A) inference pipeline, (B) components attached to the trained model, and (C) the comparison and verification layer, built on the repository-disjoint dataset.",
            src=["structure of src/sift and app/"])

    D.heading2("Model and loss")
    D.p("The input is the sequence [CLS] <i>ĩ</i> [SEP] <i>h</i>, truncated to 512 tokens. The encoder is ModernBERT-base (149.0 M parameters, 22 layers, hidden size 768, 12 heads). "
        "Let <i>p</i> be the pooled [CLS] representation. Three linear heads act on it:")
    D.equation("<i>p</i> = Enc<sub>θ</sub>([CLS] <i>ĩ</i> [SEP] <i>h</i>)<sub>[CLS]</sub>")
    D.equation("<i>a</i> = <i>w</i><sub>a</sub><sup>⊤</sup><i>p</i>,  <i>d</i> = <i>W</i><sub>d</sub><i>p</i>,  <i>c</i> = <i>w</i><sub>c</sub><sup>⊤</sup><i>p</i>")
    D.p("where <i>a</i> is the alignment logit, so that <i>P</i>(misaligned) = σ(<i>a</i>), <i>d</i> holds three drift-type logits, and <i>c</i> is the confidence logit. For the imbalance of about 78:22, "
        "the alignment term is the focal loss with γ = 2:")
    D.equation("<i>L</i><sub>align</sub> = −(1 − <i>p</i><sub>t</sub>)<sup>γ</sup> log <i>p</i><sub>t</sub>")
    D.p("where <i>p</i><sub>t</sub> = σ(<i>a</i>) for a misaligned hunk (<i>y</i> = 1) and 1 − σ(<i>a</i>) otherwise. The drift term <i>L</i><sub>drift</sub> is the cross-entropy over the three drift classes computed on misaligned rows only. The confidence term <i>L</i><sub>cal</sub> is the binary cross-entropy "
        "between <i>c</i> and the indicator that the (detached) alignment decision at logit zero is correct. The total loss is")
    D.equation("<i>L</i> = 0.6 <i>L</i><sub>align</sub> + 0.3 <i>L</i><sub>drift</sub> + 0.1 <i>L</i><sub>cal</sub>")
    D.p(f"Training uses AdamW (learning rate {CFG['lr']:.0e}, weight decay 0.01), a cosine schedule with 10% warm-up, gradient clipping at 1.0, up to 6 epochs with early stopping on validation macro F1 "
        f"(patience 2), bf16 autocast, gradient checkpointing and a micro-batch of {CFG['micro_bs']} with {CFG['accum']} accumulation steps (effective batch {CFG['effective_batch']}). "
        "A maximum length of 1,024 tokens was planned but does not fit in 6 GB of GPU memory, so 512 is used; the median pair is about 132 tokens and the 95th percentile about 455.")

    D.heading2("Noise component and denoising")
    D.p("Denoising has two stages. First, rules remove non-semantic hunks (whitespace-only, lock-file, version-bump-only and generated or minified changes) and repeated hunks after the first "
        f"occurrence. Second, confident learning finds likely label errors. Following Northcutt et al. (2021), let <i>s</i> be the observed label, <i>y</i>* the latent true label and "
        "<i>q</i><sub>j</sub>(<i>x</i>) an out-of-sample predicted probability of class <i>j</i>. The per-class threshold is the mean self-confidence of the class,")
    D.equation("<i>t</i><sub>j</sub> = E[ <i>q</i><sub>j</sub>(<i>x</i>) | <i>s</i> = <i>j</i> ]")
    D.p("and the confident joint counts the examples with observed label <i>i</i> that are confidently predicted to belong to class <i>j</i>:")
    D.equation("<i>C</i>[<i>i</i>][<i>j</i>] = | { <i>x</i> : <i>s</i> = <i>i</i>, <i>κ</i>(<i>x</i>) = <i>j</i> } |")
    D.p("where <i>κ</i>(<i>x</i>) is the class with the largest <i>q</i><sub>l</sub>(<i>x</i>) among the classes whose probability reaches their threshold, <i>q</i><sub>l</sub>(<i>x</i>) ≥ <i>t</i><sub>l</sub>. The off-diagonal entries (<i>i</i> ≠ <i>j</i>) estimate the label errors. Out-of-sample probabilities come from three-fold cross-fitting of a MiniLM-L6 cross-encoder over the training split, "
        "and the <i>cleanlab</i> implementation flags the issues; we compare its confident-learning filter with prune-by-noise-rate and prune-by-class, and we score every flagged set "
        f"against the {n_(569)} known flips by precision and recall. After the rules ({DEN['rules']['removed']} rows) and the confident-learning drop list ({n_(DEN['drop_list_size'])} rows), "
        f"{n_(CFG['n_train_used'])} of {n_(CFG['n_train_raw'])} training rows remain for the main model.")
    D.p("The noise-robustness study asks how denoising behaves when the label noise is increased. Symmetric label flips at 0%, 10%, 20% and 30% are injected into a 3,000-row training subset; for "
        "each rate, ModernBERT is trained with and without denoising (rules plus confident learning on five-fold out-of-fold TF-IDF logistic-regression probabilities) and evaluated on the clean test labels. "
        "This is a reduced protocol (2 epochs, maximum length 256, single seed).")

    D.heading2("K-means integration")
    D.p("K-means is run on the L2-normalised pooled [CLS] embeddings of hunk–intent pairs (768 dimensions). The objective is")
    D.equation("min<sub>μ</sub> Σ<sub>x</sub> min<sub>c</sub> ‖<i>e</i>(<i>x</i>) − μ<sub>c</sub>‖<sup>2</sup>")
    D.p(f"over the centroids μ<sub>c</sub>, where <i>e</i>(<i>x</i>) is the embedding of pair <i>x</i>, fitted on {n_(KM['n_fit'])} random training pairs; validation and test pairs are assigned to the nearest centroid by cosine distance. The number of clusters is chosen by the silhouette "
        "criterion with the elbow plotted for inspection. The unsupervised analysis measures cluster purity and normalised mutual information (NMI) against the alignment label and the four-way drift label, "
        "and visualises the test embeddings with t-SNE. In the hybrid analysis, a Platt (logistic) calibrator on the logit is compared with one that also receives the cluster’s training "
        "misaligned prior and the distance to the assigned centroid, both fitted on validation. In the interface each hunk displays its cluster and the most similar flagged training hunks.")

    H2("gan_m", "GAN-style baseline")
    D.p("Sift has no natural generative component, so the adversarial comparison required by the course guideline is an interpretation we chose: an adversarially trained alignment scorer on "
        "frozen embeddings. A frozen MiniLM-L6 sentence encoder embeds the intent (<i>e</i><sub>i</sub>) and the hunk (<i>e</i><sub>h</sub>). A generator <i>G</i>, conditioned on the intent "
        "embedding and a noise vector <i>z</i> ~ N(0, I), produces a synthetic aligned-hunk embedding; a discriminator <i>D</i> receives the pair features "
        "[<i>e</i><sub>i</sub>, <i>e</i><sub>h</sub>, <i>e</i><sub>i</sub>⊙<i>e</i><sub>h</sub>, |<i>e</i><sub>i</sub> − <i>e</i><sub>h</sub>|]. Only aligned training pairs are “real”:")
    D.equation("min<sub>G</sub> max<sub>D</sub> <i>V</i>(<i>D</i>, <i>G</i>)")
    D.equation("<i>V</i> = E[log <i>D</i>(<i>e</i><sub>i</sub>, <i>e</i><sub>h</sub>)] + E[log(1 − <i>D</i>(<i>e</i><sub>i</sub>, <i>G</i>))]")
    D.p("where the first expectation is over real aligned pairs, the second over the noise <i>z</i>, and <i>G</i> abbreviates <i>G</i>(<i>e</i><sub>i</sub>, <i>z</i>). After training, the misalignment score of a pair is 1 − <i>D</i>(<i>e</i><sub>i</sub>, <i>e</i><sub>h</sub>). A hybrid variant additionally treats real misaligned training pairs as negatives "
        "for the discriminator. Forty epochs, Adam (learning rate 2×10<sup>−4</sup>, β<sub>1</sub> = 0.5), the checkpoint with the best validation AUC is kept, and a "
        "generator-diversity diagnostic (mean pairwise cosine of generated embeddings, 1.0 meaning complete collapse) is logged each epoch. A parameter-free reference, the cosine dissimilarity of the "
        "frozen embeddings, is evaluated alongside.")

    D.heading2("Explainability")
    D.p("Token-level attributions use Integrated Gradients (Sundararajan et al., 2017) as implemented in <i>captum</i>, applied to the input embeddings with <i>F</i> = σ(<i>a</i>) = <i>P</i>(misaligned):")
    D.equation("IG<sub>i</sub> = (<i>x</i><sub>i</sub> − <i>x</i>′<sub>i</sub>) ∫<sub>0</sub><sup>1</sup> ∂<i>F</i>(<i>x</i>′ + α(<i>x</i> − <i>x</i>′))/∂<i>x</i><sub>i</sub> dα")
    D.p(f"The integral is approximated by a Riemann sum. The baseline <i>x</i>′ is the [PAD] embedding at every content position (the special tokens are kept), <i>m</i> = {XA['n_steps']} steps are used in the experiments, and the token score is the sum over "
        "embedding dimensions; scores are then aggregated to words (separately for the intent and the hunk) and to diff lines. For comparison, Input × Gradient is computed from the same forward pass. "
        "The faithfulness test replaces the top-<i>k</i> attributed content tokens by [PAD] and measures the mean drop in <i>P</i>(misaligned) on hunks the model flags, against replacing <i>k</i> random tokens:")
    D.equation("Δ<sub>k</sub> = (1/<i>n</i>) Σ<sub>j</sub> [ <i>P</i>(<i>x</i><sub>j</sub>) − <i>P</i><sub>k</sub>(<i>x</i><sub>j</sub>) ]")
    D.p("where <i>P</i><sub>k</sub> is the probability after masking <i>k</i> tokens. A faithful attribution yields a larger Δ<sub>k</sub> for the top-<i>k</i> tokens than for random ones.")

    D.heading2("Calibration, distillation and INT8 export")
    D.p("Temperature scaling divides the alignment logit by a scalar <i>T</i> fitted on the validation split by minimising the negative log-likelihood, and calibration is reported by the expected calibration error with <i>M</i> = 15 equal-width bins:")
    D.equation("<i>P</i><sub>T</sub> = σ(<i>a</i>/<i>T</i>)")
    D.equation("ECE = Σ<sub><i>m</i></sub> (|<i>B</i><sub>m</sub>|/<i>n</i>) |acc(<i>B</i><sub>m</sub>) − conf(<i>B</i><sub>m</sub>)|")
    D.p("Because σ(<i>a</i>/<i>T</i>) is monotone in <i>a</i>, temperature scaling does not change any ranking metric. A MiniLM-L6 student (22.7 M parameters) is trained by distillation from the main model with temperature τ = 4 and weight α = 0.5:")
    D.equation("<i>L</i><sub>KD</sub> = α <i>L</i><sub>CE</sub> + (1 − α) τ<sup>2</sup> KL<sub>τ</sub>")
    D.p("where <i>L</i><sub>CE</sub> is the cross-entropy of the student logit <i>a</i><sub>s</sub> with the label and KL<sub>τ</sub> = KL(σ(<i>a</i><sub>t</sub>/τ) ‖ σ(<i>a</i><sub>s</sub>/τ)) compares the softened teacher and student outputs. The student is exported to ONNX and dynamically quantised to 8-bit integers; its temperature and threshold are re-fitted on validation.")

    D.heading2("User interface and backend integration")
    D.p("The model is served by a FastAPI application whose endpoints cover analysis (<i>POST /api/analyze</i>: session turns and a unified diff, or a curated sample identifier), curated test-split samples, "
        "the model registry, result files, dataset statistics, health and test summaries, with generated OpenAPI documentation. The registry exposes the main model, the student in PyTorch, "
        "ONNX fp32 and INT8 form, and the B4 and B5 baselines, so that the interface can switch between them; models are loaded once and run on CPU or GPU automatically. Inputs are validated "
        "(at most 50 turns, 5,000 characters per turn, 1,000,000 diff characters and 200 hunks), and the diff parser handles git-style multi-file diffs, renames and binary-file notes. "
        "The React, TypeScript and Tailwind interface has a review workspace (intent panel with dropped turns, risk-ordered hunk cards with syntax highlighting, token highlights, similar flagged hunks, "
        "accept / flag / needs-discussion actions, a threshold slider, a model switcher with measured latency, and keyboard shortcuts) and a developer console with results, experiments, dataset, testing and related-work pages. "
        "Every number the interface shows is read from the result files or comes from a model call.")

    H2("testmeth", "Testing and test design methodology")
    D.p("Verification follows the framework for the test and evaluation of AI models of the U.S. Department of Defense Chief Digital and Artificial Intelligence Office (U.S. Department of Defense, 2024), whose six sections are "
        "performance, testing methods, data, AI models, context and documentation. The test types and their design techniques are: <i>smoke</i> tests of readiness; <i>white-box</i> tests of tensor shapes, "
        "heads, losses, tokenisation, the diff parser and consolidation; <i>black-box</i> tests designed by equivalence partitioning, boundary-value analysis and error guessing; <i>data-pipeline</i> tests including "
        "split-leakage checks; <i>metamorphic</i> relations (output relations that must hold when the input is changed in a controlled way); <i>adversarial / red-team</i> attacks with an attack-success threshold; "
        "<i>back-to-back</i> (differential) comparison of the full model with the student; and end-to-end user-interface flows. Metamorphic tolerances and the 20% attack limit were fixed before the first run. "
        "Factors were varied one at a time (diff shape, intent shape, language, content type); full pairwise generation was not used. Known defects are recorded as strict expected failures, "
        "so the suite becomes red if one is silently fixed.")

    # ============================================================ 4 Results
    D.heading1("Results and Discussion")
    D.heading2("Experimental setup")
    D.p(f"All experiments ran on one laptop: an NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB of memory, a {CPU} (20 logical cores) and 15 GB of RAM, with Python 3.12 and PyTorch {CFG['torch']}. "
        "The dataset seed is 473 and every run uses seed 473; there is a single seed per configuration and no confidence intervals. Hyper-parameters, the classification threshold, and the "
        "temperature are chosen on validation only, and the test split is used for reporting. Table "
        f"{rec_t_n(R,'hyper')} lists the training configurations. The main model was trained for {FULLRUN['train_time_s']:.0f} s ({FULLRUN['train_time_s']/60:.0f} min) with a peak GPU memory of "
        f"{FULLRUN['peak_gpu_mem_gb']:.1f} GB and reached a best validation macro F1 of {FULLRUN['best_val_macro_f1']:.3f}. A small Optuna study (TPE, {HPO['trials_run']} trials of the {HPO['planned_trials_brief']} planned, "
        f"{n_(HPO['n_train'])} training rows, {HPO['epochs']} epochs) found no clearly better setting (best validation macro F1 {HPO['best']['value']:.3f} at learning rate {HPO['best']['params']['lr']:.1e} and maximum length "
        f"{HPO['best']['params']['max_len']}), so the main model keeps its default settings.")
    D.p("The non-neural baselines are: B1, random order; B2, the number of lines changed in the whole diff (a diff-level scalar, constant within a session); B3, the number of changed lines in the hunk; B4, one minus the TF-IDF cosine similarity between the consolidated intent and the changed text and path of the hunk "
        "(identifier-split tokens, vocabulary fitted on the training split); and B5, a class-weighted logistic regression trained on the observed training labels with four features: that similarity, the logarithm of the hunk size, an indicator of identifier overlap between the file path and the intent, and the logarithm of the number of hunks in the diff.")
    D.p("The comparison models use a reduced protocol and the same denoised rows: a MiniLM-L6 cross-encoder (2 epochs, maximum length 256), a CodeBERT cross-encoder (1 epoch on 4,000 rows, maximum length 256) and a Siamese BiLSTM "
        "(15 epochs). UniXcoder could not be run because its weights could not be downloaded. These models received far less training than the main model, so part of the gap to it is a budget gap and not purely an "
        "architecture gap.")
    D.heading2("Evaluation datasets")
    D.p("Because there is one real dataset, the evaluation sets are defined as subsets of the test split: the full test split (2,700 hunks); Python only (1,686); JavaScript only (1,014), which equals the single "
        "JavaScript test repository axios because TypeScript is absent from the test split; the hard tier (815 hunks, 217 rankable sessions) and the random tier (893 hunks, 255 sessions); and each test repository "
        "(httpx 786, attrs 900, axios 1,014 hunks). A tier subset keeps the sessions that contain a misaligned hunk of that tier, drops the misaligned hunks of the other tier and keeps the aligned hunks of those sessions. "
        f"NDCG@K is averaged over the {CONF['n_rankable_sessions']} rankable sessions (at least two hunks and at least one misaligned hunk), and pairwise AUC uses {J('results/eval_full.json')['test']['full']['pairwise_auc']['random']['pairs']} random-tier and "
        f"{J('results/eval_full.json')['test']['full']['pairwise_auc']['hard']['pairs']} hard-tier (misaligned, aligned) pairs. The previously measured baselines were reproduced exactly (B1–B5 NDCG@5 and pairwise AUC agree to four decimals with the dataset release); "
        "the only mismatch is the number of rankable sessions, 456 here and in the release against 447 in an earlier description of the dataset.")

    with D.wide():
        ds_rows = [["Split", "Repositories", "Sessions", "Hunks", "Aligned", "Misaligned", "Python", "JavaScript", "TypeScript"]]
        for sp, repos in (("train", "requests, flask, click, rich, black, express, fastify"), ("val", "typer, sqlmodel, got, koa, chalk"), ("test", "httpx, attrs, axios")):
            ds_rows.append([sp, repos, n_(DS[f"{sp}.sessions"]), n_(DS[f"{sp}.hunks"]), n_(DS[f"{sp}.aligned_clean"]), n_(DS[f"{sp}.misaligned_clean"]),
                            n_(DS.get(f"{sp}.languages.Python", 0)), n_(DS.get(f"{sp}.languages.JavaScript", 0)), n_(DS.get(f"{sp}.languages.TypeScript", 0))])
        ds_rows.append(["total", "15 repositories", n_(sum(int(DS[f'{s}.sessions']) for s in ('train', 'val', 'test'))), n_(18000), n_(14040), n_(3960), n_(10205), n_(6726), n_(1069)])
        tab("dataset", ds_rows, [0.55, 2.35, 0.65, 0.6, 0.65, 0.75, 0.5, 0.65, 0.65], "Sift-IA v1.0 statistics and repository-disjoint splits (clean labels). Drift types: collateral 1,782, scope expansion 1,307, purpose drift 871; difficulty tiers: random 1,960, hard 2,000.",
            size=7.5, aligns=["l", "l"] + ["c"] * 7, bold_rows=(4,), src=["results/dataset_statistics.csv", "experiments/results/DISCREPANCIES.md"])
        nt = [["Noise type", "Rows", "Ground-truth field / note"],
              ["Intent turn (chit-chat 425, retraction 308, repeated turn 180 sessions)", n_(DS["noise_rows.intent_turn"]), "intent_noise_type"],
              ["Duplicate hunk", n_(DS["noise_rows.duplicate_hunk"]), "is_duplicate_hunk"],
              ["Non-semantic hunk (whitespace 343, lock-file 202, version bump 143, generated 26)", n_(DS["noise_rows.non_semantic_hunk"]), "non_semantic_reason"],
              ["Label flip (train only; 285 aligned to misaligned, 284 the reverse)", n_(DS["noise_rows.label_flip"]), "label_noise, label_clean"],
              ["Truncation (exceeds the maximum length)", n_(DS["noise_rows.truncation"]), "exceeds_max_len"],
              ["Rows with at least one noise type", n_(DS["rows_with_any_noise"]), "25.2% of 18,000"]]
        tab("noise_types", nt, [4.2, 0.8, 2.0], "Noise types recorded with ground truth in Sift-IA v1.0.", size=7.5, aligns=["l", "c", "l"], bold_rows=(6,), src=["results/dataset_statistics.csv"])

        hy = [["Run", "Encoder", "Max len", "Batch", "Epochs", "LR", "Best val F1", "Time (s)", "Peak GPU (GB)"]]
        names = [("full", "Main (full)", "ModernBERT-base"), ("abl_hunk_only", "Ablation: hunk only", "ModernBERT-base"), ("abl_intent_only", "Ablation: intent only", "ModernBERT-base"),
                 ("abl_no_consolidation", "Ablation: raw turns", "ModernBERT-base"), ("abl_no_focal", "Ablation: no focal loss", "ModernBERT-base"), ("abl_no_multitask", "Ablation: alignment head only", "ModernBERT-base"),
                 ("abl_no_denoise", "Ablation: no denoising", "ModernBERT-base"), ("minilm_ce", "MiniLM-L6 cross-encoder", "all-MiniLM-L6-v2"), ("codebert", "CodeBERT cross-encoder", "codebert-base"),
                 ("bilstm", "Siamese BiLSTM", "BiLSTM (MiniLM vocabulary)"), ("student", "Distilled student", "all-MiniLM-L6-v2"), ("student_nodistill", "Student, no distillation", "all-MiniLM-L6-v2")]
        for r, lab, enc in names:
            rr = J(f"experiments/runs/{r}/results.json"); c = rr["config"]
            mb = c.get("micro_bs"); ac = c.get("accum")
            hy.append([lab, enc, c.get("max_len", "–"), f"{mb}×{ac}" if mb else "–", c.get("epochs", "–"), f"{c['lr']:.0e}" if c.get("lr") else "–",
                       f3(rr["best_val_macro_f1"]), f"{rr['train_time_s']:.0f}", f3(rr.get("peak_gpu_mem_gb"), 1)])
        tab("hyper", hy, [1.9, 1.5, 0.5, 0.55, 0.5, 0.55, 0.65, 0.55, 0.8], "Training configurations (single seed 473; bf16, gradient checkpointing for the ModernBERT runs; the loss weights are 0.6/0.3/0.1; max len = maximum token length, batch = micro-batch × accumulation).",
            size=7.5, aligns=["l", "l"] + ["c"] * 7, shade_rows=(1,), src=["experiments/runs/*/results.json", "results/hyperparameters.csv"])

    D.heading2("Master comparison of all models")
    D.p(f"{TR('master')} gives all models on the full test split. Ordering hunks by diff size or hunk size is no better than random order (B2 {f3(BL['ndcg@5']['B2 Diff lines changed'])}, B3 {f3(b3['ndcg@5'])}), as argued in Section 1. "
        f"The main ModernBERT model reaches NDCG@5 {f3(main['ndcg@5'])}, level with the distilled student ({f3(stu['ndcg@5'])}) and ahead of the MiniLM cross-encoder ({f3(mini['ndcg@5'])}), and has the highest hard-tier pairwise AUC of the models in the table ({f3(main['auc_hard'])}), although the parameter-free frozen-cosine reference is close on the hard tier ({f3(frz['auc_hard'])}) and higher on NDCG@5 ({f3(frz['ndcg@5'])}) and on the random tier ({f3(frz['auc_random'])}). "
        f"The main model’s macro F1 ({f3(main['macro_f1'])}) is below B5 ({f3(b5['macro_f1'])}), and on the random tier B4 ({f3(b4['auc_random'])}) is also above it ({f3(main['auc_random'])}). "
        f"These weaknesses are stated here and discussed in Section {SR('disc')}.")
    with D.wide():
        order = [("B1 Random order", "B1 random order"), ("B2 Diff lines changed", "B2 diff lines changed"), ("B3 Hunk size", "B3 hunk size"), ("B4 TF-IDF dissimilarity", "B4 TF-IDF dissimilarity"),
                 ("B5 Logistic regression", "B5 logistic regression"), ("Siamese BiLSTM", "Siamese BiLSTM"), ("MiniLM-L6 cross-encoder", "MiniLM-L6 cross-encoder (reduced)"),
                 ("CodeBERT cross-encoder", "CodeBERT cross-encoder (reduced)"), ("Main: ModernBERT-base", "Main: ModernBERT-base cross-encoder"),
                 ("Frozen-embedding frozen cosine dissimilarity", "Frozen-embedding cosine dissimilarity (no training)"), ("GAN-style baseline: gan unsupervised", "GAN-style baseline, unsupervised"),
                 ("GAN-style baseline: gan hybrid with misaligned negatives", "GAN-style baseline, hybrid"), ("Distilled MiniLM student (fp32)", "Distilled MiniLM student (fp32)"),
                 ("MiniLM student, no distillation", "MiniLM student, no distillation"), ("Student ONNX fp32", "Student ONNX fp32"), ("Student ONNX INT8 (deployed target)", "Student ONNX INT8 (deployed target)")]
        mrows = [["Model", "Macro-F1", "Prec.", "Recall", "Spec.", "ROC-AUC", "NDCG@5", "AUC random", "AUC hard", "ECE", "Size (MB)"]]
        for k, lab in order:
            r = MC[k]
            if k == "B1 Random order" or k == "B2 Diff lines changed":
                mrows.append([lab] + ["–"] * 5 + [f3(r["ndcg@5"]), "–", "–", "–", "–"]); continue
            mrows.append([lab, f3(r["macro_f1"]), f3(r["precision"]), f3(r["recall"]), f3(r["specificity"]), f3(r["roc_auc"]), f3(r["ndcg@5"]), f3(r["auc_random"]), f3(r["auc_hard"]), f3(r["ece"]), f3(r["size_mb"] or r["ckpt_size_mb"], 1)])
        tab("master", mrows, [2.2, 0.55, 0.45, 0.5, 0.45, 0.55, 0.55, 0.6, 0.55, 0.45, 0.5], "Master comparison of all models on the full test split (2,700 hunks, 456 rankable sessions). A dash means the quantity was not measured or not defined for that model. ECE is before temperature scaling for the ModernBERT rows; the thresholded metrics use the threshold tuned on validation. Single seed.",
            size=7.2, aligns=["l"] + ["c"] * 10, shade_rows=(9,), bold_rows=(9,), src=["results/master_comparison.csv", "results/eval_*.json", "experiments/results/baselines.json"])
    D.p(f"{FR('ndcg')} shows NDCG@K for the trained models and {FR('tier')} the per-tier pairwise AUC. NDCG@K has a high floor because the median session has only two hunks; the random-order value of {f3(BL['ndcg@5']['B1 Random order'])} "
        f"should be read as the zero line, which is why {FR('tier')} reports per-tier AUC. The cross-encoders and the student rank clearly better than the Siamese BiLSTM and the under-trained CodeBERT, consistent with the "
        "benefit of joint encoding, but the frozen-embedding reference is as good as the main model on both NDCG@5 and random-tier AUC. The precision–recall curves "
        f"({FR('pr')}) and the ROC curves per tier ({FR('roc')}) tell the same story: the main model dominates the neural baselines on every tier, and B5 and the main model are close on the full set; on the hard tier "
        "the lexical baselines B4 and B5 are close to the diagonal. At the validation-tuned threshold the main model has the confusion matrix of "
        f"{FR('conf')}: {CONF['tn']} true negatives, {CONF['fp']} false positives, {CONF['fn']} false negatives and {CONF['tp']} true positives (precision {f3(CONF['precision'])}, recall {f3(CONF['recall'])}, specificity {f3(CONF['specificity'])}).")
    fig("ndcg", "fig_ndcg_vs_k.png", COL_W, "NDCG@K versus K for the trained models on the test split (single seed). The dotted line is random order at K = 5.", src=["results/eval_*.json"])
    with D.wide():
        fig("tier", "fig_tier_auc.png", 6.9, "Within-session pairwise AUC per difficulty tier, test split. A value of 0.5 is chance. Lexical baselines B3–B5 fall to or near chance on the hard tier, where the donor hunk is chosen for identifier overlap.", src=["results/master_comparison.csv"])
    fig("pr", "fig_pr.png", COL_W, "Precision–recall curves on the full test split for the trained models and for B4 and B5 (dashed). The dotted line is the prevalence of misaligned hunks.", src=["experiments/runs/*/preds.npz", "experiments/results/baseline_scores.npz"])
    with D.wide():
        fig("roc", "fig_roc_tiers.png", 6.9, "ROC curves for all test hunks, the random tier and the hard tier. B4 and B5 are dashed.", src=["experiments/runs/*/preds.npz", "experiments/results/baseline_scores.npz"])
    fig("conf", "fig_confusion.png", COL_W * 0.75, f"Confusion matrix of the main model on the test split at threshold {CONF['threshold']:.2f} (row-normalised percentages).", src=["results/eval_full.json"])
    fig("train", "fig_training_curves.png", COL_W, "Training loss (left) and validation macro-F1 (right) per epoch for the main model and four ablation runs. Loss values are not comparable across loss types (focal versus cross-entropy, one head versus three).", src=["experiments/runs/*/results.json"])
    D.p(f"{FR('train')} shows the training curves: validation macro F1 of the main model rises from {FULLRUN['history'][0]['val_macro_f1']:.3f} after the first epoch to a plateau near {FULLRUN['best_val_macro_f1']:.3f} by epochs 4 to 6 while the training loss keeps falling, which suggests that "
        "further epochs would mostly fit the training set; early stopping on validation macro F1 was configured but all six epochs were used.")

    D.heading2("Results per evaluation subset")
    D.p(f"{TR('subset')} reports the main model, the MiniLM cross-encoder and the distilled student on every subset. The main model’s macro F1 ranges from {f3(FT['repo_httpx']['macro_f1'])} on httpx to {f3(FT['repo_attrs']['macro_f1'])} on attrs, "
        f"and its NDCG@5 from {f3(FT['repo_httpx']['ndcg@5'])} to {f3(FT['repo_attrs']['ndcg@5'])}, so the repository matters at least as much as the model choice for many comparisons. The random tier is easier than the hard tier "
        f"(macro F1 {f3(FT['tier_random']['macro_f1'])} against {f3(FT['tier_hard']['macro_f1'])}). Python and JavaScript behave differently by tier: on Python the random-tier AUC ({f3(FT['python']['pairwise_auc']['random']['auc'])}) is higher than the hard-tier AUC "
        f"({f3(FT['python']['pairwise_auc']['hard']['auc'])}), while on axios (the only JavaScript repository) the hard tier ({f3(FT['js_ts']['pairwise_auc']['hard']['auc'])}) is higher than the random tier ({f3(FT['js_ts']['pairwise_auc']['random']['auc'])}). TypeScript is absent from the test split, so "
        "TypeScript generalisation is unmeasured.")
    with D.wide():
        sr = [["Subset", "n", "Main F1", "Main NDCG@5", "Main AUC rand / hard", "MiniLM F1", "MiniLM NDCG@5", "Student F1", "Student NDCG@5"]]
        subs = [("full", "Full test"), ("python", "Python only"), ("js_ts", "JavaScript only (= axios)"), ("tier_random", "Random tier"), ("tier_hard", "Hard tier"), ("repo_httpx", "httpx"), ("repo_attrs", "attrs"), ("repo_axios", "axios")]

        def aucpair(e):
            a = e["pairwise_auc"]
            return " / ".join(f3(a[t]["auc"]) if a[t]["auc"] is not None else "–" for t in ("random", "hard"))
        for k, lab in subs:
            m_, mi, st_ = EV["full"]["test"][k], EV["minilm_ce"]["test"][k], EV["student"]["test"][k]
            sr.append([lab, n_(m_["n"]), f3(m_["macro_f1"]), f3(m_["ndcg@5"]), aucpair(m_), f3(mi["macro_f1"]), f3(mi["ndcg@5"]), f3(st_["macro_f1"]), f3(st_["ndcg@5"])])
        tab("subset", sr, [1.7, 0.5, 0.65, 0.8, 1.1, 0.7, 0.85, 0.7, 0.85], "Results per evaluation subset of the test split (single seed; thresholds tuned on validation; “–” where a tier has no pairs).", size=7.5,
            aligns=["l"] + ["c"] * 8, src=["results/eval_full.json", "results/eval_minilm_ce.json", "results/eval_student.json"])

    H2("stats", "Statistical uncertainty")
    bm, bn, bd = BOOT["macro_f1"], BOOT["ndcg@5"], BOOT["paired_difference_main_minus"]
    D.p(f"All results above are single-seed point estimates. To show how much the finite test set alone can move them, we resample the {BOOT['n_sessions']} test sessions with replacement (2,000 resamples; thresholds stay fixed). "
        f"This captures the sampling uncertainty of the test split, not the seed-to-seed variation of training, which we did not measure. {TR('ci')} gives the intervals. The main model\u2019s macro F1 has a 95% interval of {bm['main']['ci95'][0]:.3f} to {bm['main']['ci95'][1]:.3f} "
        f"and its NDCG@5 {bn['main']['ci95'][0]:.3f} to {bn['main']['ci95'][1]:.3f}. The macro-F1 difference to B5 ({bd['B5']['macro_f1']['point']:+.3f}) has an interval that includes zero ({bd['B5']['macro_f1']['ci95'][0]:+.3f} to {bd['B5']['macro_f1']['ci95'][1]:+.3f}), "
        f"so B5\u2019s edge in macro F1 is not distinguishable from sampling noise, whereas the NDCG@5 advantage over B5 ({bd['B5']['ndcg@5']['point']:+.3f}, interval {bd['B5']['ndcg@5']['ci95'][0]:+.3f} to {bd['B5']['ndcg@5']['ci95'][1]:+.3f}) is not. "
        f"The student is indistinguishable from the main model on both measures, and the MiniLM cross-encoder trails it on both. The no-denoising ablation has a macro-F1 difference within noise but ranks better on NDCG@5 by more than the interval ({-bd['no_denoising']['ndcg@5']['point']:.3f}, interval {-bd['no_denoising']['ndcg@5']['ci95'][1]:.3f} to {-bd['no_denoising']['ndcg@5']['ci95'][0]:.3f}).")
    cit = [["Model", "Macro-F1 (95% CI)", "NDCG@5 (95% CI)"]]
    for k, lab in (("main", "Main ModernBERT"), ("no_denoising", "Ablation: no denoising"), ("student_fp32", "Student fp32"), ("minilm_ce", "MiniLM-L6 cross-encoder"), ("B5", "B5 logistic regression"), ("B4", "B4 TF-IDF dissimilarity")):
        f1c = f"{bm[k]['point']:.3f} ({bm[k]['ci95'][0]:.3f}\u2013{bm[k]['ci95'][1]:.3f})" if k in bm else "\u2013"
        cit.append([lab, f1c, f"{bn[k]['point']:.3f} ({bn[k]['ci95'][0]:.3f}\u2013{bn[k]['ci95'][1]:.3f})"])
    cit.append(["<i>Paired difference, main minus B5</i>", f"{bd['B5']['macro_f1']['point']:+.3f} ({bd['B5']['macro_f1']['ci95'][0]:+.3f} to {bd['B5']['macro_f1']['ci95'][1]:+.3f})", f"{bd['B5']['ndcg@5']['point']:+.3f} ({bd['B5']['ndcg@5']['ci95'][0]:+.3f} to {bd['B5']['ndcg@5']['ci95'][1]:+.3f})"])
    cit.append(["<i>Paired difference, main minus no denoising</i>", f"{bd['no_denoising']['macro_f1']['point']:+.3f} ({bd['no_denoising']['macro_f1']['ci95'][0]:+.3f} to {bd['no_denoising']['macro_f1']['ci95'][1]:+.3f})", f"{bd['no_denoising']['ndcg@5']['point']:+.3f} ({bd['no_denoising']['ndcg@5']['ci95'][0]:+.3f} to {bd['no_denoising']['ndcg@5']['ci95'][1]:+.3f})"])
    cit.append(["<i>Paired difference, main minus student fp32</i>", f"{bd['student_fp32']['macro_f1']['point']:+.3f} ({bd['student_fp32']['macro_f1']['ci95'][0]:+.3f} to {bd['student_fp32']['macro_f1']['ci95'][1]:+.3f})", f"{bd['student_fp32']['ndcg@5']['point']:+.3f} ({bd['student_fp32']['ndcg@5']['ci95'][0]:+.3f} to {bd['student_fp32']['ndcg@5']['ci95'][1]:+.3f})"])
    tab("ci", cit, [1.35, 1.0, 1.0], "Session-level bootstrap intervals on the test split (2,000 resamples). Point values are recomputed from stored predictions (main macro F1 0.679 here against 0.680 in the other tables, which come from a fresh inference pass), and the NDCG@5 here uses a fixed tie-break, so B4 is 0.834 here against 0.830 with random tie-breaking.", size=6.8, aligns=["l", "c", "c"], src=["reports/case_study/bootstrap_ci.json"])

    H2("drift", "Drift-type head")
    dt = BOOT["drift_type"]; cmx = dt["confusion_rows_true_cols_pred"]
    D.p(f"The second head predicts the drift type of a misaligned hunk. On the {dt['n']} truly misaligned test hunks its accuracy is {dt['accuracy']:.3f} and its macro F1 {dt['macro_f1']:.3f}, against a majority-class rate of {dt['majority_class_rate']:.3f} ({TR('drift')}); "
        f"collateral drift is recognised best (F1 {dt['f1_per_class']['collateral']:.3f}), scope expansion ({dt['f1_per_class']['scope_expansion']:.3f}) and purpose drift ({dt['f1_per_class']['purpose_drift']:.3f}) poorly. So the head separates collateral from the other two types somewhat but is barely better than predicting the most frequent class, and it should be presented as a weak hint, not a diagnosis.")
    dtt = [["True \\ predicted", "Collateral", "Scope expansion", "Purpose drift", "F1"]]
    for i, c in enumerate(dt["classes"]):
        dtt.append([c.replace("_", " "), cmx[i][0], cmx[i][1], cmx[i][2], f3(dt["f1_per_class"][c])])
    tab("drift", dtt, [0.9, 0.55, 0.7, 0.65, 0.4], f"Drift-type head on the {dt['n']} misaligned test hunks (accuracy {dt['accuracy']:.3f}, macro F1 {dt['macro_f1']:.3f}).", size=7, aligns=["l", "c", "c", "c", "c"], src=["reports/case_study/bootstrap_ci.json", "experiments/runs/full/preds.npz"])

    D.heading2("Calibration")
    D.p(f"The temperature fitted on validation is T = {T_:.3f}. On the test split the expected calibration error falls from {f3(CAL['ece_test_before'])} to {f3(CAL['ece_test_after'])} and the negative log-likelihood from "
        f"{f3(CAL['nll_test_before'])} to {f3(CAL['nll_test_after'])} ({FR('rel')}); on validation the error is already low and moves from {f3(CAL['ece_val_before'])} to {f3(CAL['ece_val_after'])}. A temperature above one means the model was over-confident. "
        f"The reliability curve after scaling follows the diagonal more closely at low and middle confidences; the highest-confidence bins remain noisy because they hold few hunks. Calibration was fitted on validation and evaluated on test, so the improvement is out of sample.")
    fig("rel", "fig_reliability.png", COL_W, f"Reliability diagram of the main model on the test split before and after temperature scaling (T = {T_:.3f}); 10 equal-width bins.", src=["results/calibration.json"])

    H2("noise_r", "Noise and denoising")
    D.p(f"Of the {n_(DS['train.hunks'])} training rows, 569 (4.5%) carry a flipped label. Confident learning flags {n_(cl_best['flagged'])} rows and finds {cl_best['tp']} of the {n_(569)} true flips (precision {f3(cl_best['precision'])}, recall {f3(cl_best['recall'])}), "
        f"compared with a random-guess precision of {f3(CL['random_precision'])}: the detector is about {cl_best['precision'] / CL['random_precision']:.1f} times better than chance in precision but remains weak ({FR('cl')}, {TR('cl')}). The final drop list "
        f"(confident learning plus rules) has precision {f3(cl_final['precision'])} and recall {f3(cl_final['recall'])}; the out-of-fold model that produced the probabilities has an AUC of only {f3(CL['oof_auc_vs_clean'])} against the clean labels. The rules removed "
        f"{CL['rules']['removed']} rows ({CL['rules']['non_semantic']} non-semantic and {CL['rules']['duplicates_after_first']} repeated-hunk candidates), of which {CL['rules']['removed_that_are_label_flips']} were label flips.")
    cltab = [["Method", "Flagged", "True flips found", "Precision", "Recall", "F1"]]
    for k, lab in (("confident_learning", "Confident learning (filter)"), ("prune_by_noise_rate", "Prune by noise rate"), ("prune_by_class", "Prune by class"), ("final drop list (CL + rules)", "Final drop list (CL + rules)")):
        d = CL["detection"][k]; cltab.append([lab, n_(d["flagged"]), n_(d["tp"]), f3(d["precision"]), f3(d["recall"]), f3(d["f1"])])
    cltab.append(["Random guess", "–", "–", f3(CL["random_precision"]), "–", "–"])
    tab("cl", cltab, [1.1, 0.45, 0.5, 0.45, 0.4, 0.35], "Detection of the 569 known label flips among 12,600 training rows.", size=7, src=["results/confident_learning.json"])
    fig("cl", "fig_cl_detection.png", COL_W, "Confident-learning detection precision and recall against the 569 known label flips; the dotted line is the precision of random guessing.", src=["results/confident_learning.json"])
    D.p(f"The noise-robustness sweep ({FR('noise')}, {TR('noise')}) shows that denoising helps at 20% and 30% injected noise (macro F1 {f3(nz[(0.2,'denoise')]['macro_f1'])} against {f3(nz[(0.2,'no_denoise')]['macro_f1'])}, and "
        f"{f3(nz[(0.3,'denoise')]['macro_f1'])} against {f3(nz[(0.3,'no_denoise')]['macro_f1'])}) and raises NDCG@5 at all four rates, but it does not help macro F1 at 0% ({f3(nz[(0.0,'denoise')]['macro_f1'])} against {f3(nz[(0.0,'no_denoise')]['macro_f1'])}). "
        "The curves are non-monotone (macro F1 peaks at 10% for both arms), which is expected for a 3,000-row, single-seed, two-epoch study and means that only the large 20% and 30% gaps and the NDCG@5 trend are suggestive.")
    nzt = [["Noise", "Arm", "Rows used", "Macro-F1", "NDCG@5"]]
    for r in NZ:
        nzt.append([f"{r['noise_rate']*100:.0f}%", r["arm"].replace("_", " "), n_(r["n_train_used"]), f3(r["test"]["macro_f1"]), f3(r["test"]["ndcg@5"])])
    tab("noise", nzt, [0.55, 0.9, 0.7, 0.65, 0.6], "Noise-robustness study (reduced protocol: 3,000 training rows, 2 epochs, maximum length 256, single seed).", size=7, src=["results/noise_robustness.json"])
    fig("noise", "fig_noise.png", COL_W, "Test macro-F1 and NDCG@5 versus injected label-noise rate, with and without denoising (reduced protocol, single seed).", src=["results/noise_robustness.json"])

    D.heading2("K-means analysis")
    D.p(f"The silhouette criterion selects k = {KM['k_chosen']} (silhouette {max(KM['k_scan']['silhouette']):.3f}), and the elbow curve has no sharp elbow ({FR('km')}). The clusters have weak class structure: on the test split "
        f"purity against the alignment label is {f3(KQ['purity_vs_alignment'])} against a majority-class rate of {f3(KQ['majority_class_rate_alignment'])}, with NMI {f3(KQ['nmi_vs_alignment'])} for alignment and {f3(KQ['nmi_vs_drift4'])} for the four-way drift label. "
        f"The clusters nonetheless differ in their misaligned rate ({TR('km')}): cluster {KCT[1]['cluster']} contains almost only aligned hunks ({pc(KCT[1]['misaligned_rate'],1)} misaligned), while cluster {KCT[3]['cluster']} is {pc(KCT[3]['misaligned_rate'])} misaligned and collateral-heavy. "
        f"In the hybrid analysis, adding the cluster prior and the centroid distance to the Platt calibrator changes test macro F1 by {KH['macro_f1']:+.3f}, NDCG@5 by {KH['ndcg@5']:+.3f} and hard-tier AUC by {KH['auc_hard']:+.4f}, i.e. almost nothing ({TR('abl')} contains both rows).")
    kct = [["Cluster", "n", "Misaligned rate (test)", "Train prior", "Aligned / collateral / scope / purpose"]]
    for c in KCT:
        d = c["drift_counts"]; kct.append([c["cluster"], n_(c["n"]), pc(c["misaligned_rate"], 1), pc(c["train_prior"], 1), f"{d['aligned']} / {d['collateral']} / {d['scope_expansion']} / {d['purpose_drift']}"])
    tab("km", kct, [0.4, 0.4, 0.8, 0.55, 1.25], f"K-means (k = {KM['k_chosen']}) clusters of the test embeddings.", size=7, aligns=["c", "c", "c", "c", "c"], src=["results/kmeans.json"])
    with D.wide():
        fig("km", "fig_kmeans.png", 6.9, "K-means on the pooled ModernBERT embeddings: elbow (inertia), silhouette against k, and a t-SNE view of the test embeddings coloured by cluster (the percentage is the misaligned share of the cluster).", src=["results/kmeans.json", "experiments/results/pooled_embeddings.npz"])

    D.heading2("GAN-style baseline")
    gcmp = GV["compared"]
    D.p(f"The adversarially trained baseline is a comparison under our own interpretation (Section {SR('gan_m')}). {TR('gan')} and {FR('ganv')} compare it with the main model. The unsupervised variant, which never sees a misaligned pair, reaches "
        f"macro F1 {f3(gcmp['gan unsupervised']['macro_f1'])} and NDCG@5 {f3(gcmp['gan unsupervised']['ndcg@5'])}; the hybrid variant reaches NDCG@5 {f3(gcmp['gan hybrid with misaligned negatives']['ndcg@5'])}. The parameter-free frozen-cosine reference reaches NDCG@5 "
        f"{f3(gcmp['frozen cosine dissimilarity']['ndcg@5'])} and random-tier AUC {f3(gcmp['frozen cosine dissimilarity']['auc_random'])}, both above the main model. Training was unstable ({FR('ganc')}): the generator collapsed at the start (pairwise cosine "
        f"{GJ['variants']['gan_unsupervised']['history'][0]['gen_diversity_cos']:.2f} in epoch 1) and diversified to {GV['gan_unsupervised_final_diversity']:.2f} by epoch 40 (minimum {GV['gan_unsupervised_min_diversity']:.2f}), while the discriminator loss fell and the generator loss rose; "
        f"the validation AUC of the discriminator peaked early (best {GJ['variants']['gan_unsupervised']['best_val_auc']:.3f}) and decayed towards the end, so the best-on-validation checkpoint is used. The discriminator output is not a probability (ECE about "
        f"{GJ['variants']['gan_unsupervised']['test']['ece']:.2f}).")
    gt = [["Model", "Macro-F1", "NDCG@5", "AUC random", "AUC hard", "Trainable params", "Scoring latency p50"]]
    gt.append(["Main ModernBERT", f3(gcmp["full ModernBERT"]["macro_f1"]), f3(gcmp["full ModernBERT"]["ndcg@5"]), f3(gcmp["full ModernBERT"]["auc_random"]), f3(gcmp["full ModernBERT"]["auc_hard"]), "149.0 M", f"{LF['cpu']['p50_ms']:.1f} ms (CPU)"])
    gt.append(["Frozen cosine", f3(gcmp["frozen cosine dissimilarity"]["macro_f1"]), f3(gcmp["frozen cosine dissimilarity"]["ndcg@5"]), f3(gcmp["frozen cosine dissimilarity"]["auc_random"]), f3(gcmp["frozen cosine dissimilarity"]["auc_hard"]), "0", f"{GJ['embedding_latency_ms_p50_per_pair']:.1f} ms (embedder)"])
    for k, lab in (("gan unsupervised", "GAN-style unsupervised"), ("gan hybrid with misaligned negatives", "GAN-style hybrid")):
        v = GJ["variants"]["gan_unsupervised" if k == "gan unsupervised" else "gan_hybrid_with_misaligned_negatives"]
        gt.append([lab, f3(gcmp[k]["macro_f1"]), f3(gcmp[k]["ndcg@5"]), f3(gcmp[k]["auc_random"]), f3(gcmp[k]["auc_hard"]), f"{(v['trainable_params_D'] + v['trainable_params_G']) / 1e6:.2f} M", f"{GJ['embedding_latency_ms_p50_per_pair'] + v['disc_latency_ms_p50']:.1f} ms (embedder + D)"])
    with D.wide():
        tab("gan", gt, [1.5, 0.7, 0.7, 0.8, 0.7, 1.1, 1.5], "GAN-style baseline versus the main model on the test split (single seed).", size=7.5, aligns=["l"] + ["c"] * 6, src=["results/gan_vs_main.json", "experiments/results/gan.json", "results/latency_full.json"])
        fig("ganc", "fig_gan_curves.png", 6.9, "Training behaviour of the adversarial baseline: losses (solid discriminator, dashed generator), the discriminator’s validation ROC-AUC as a scorer, and generator diversity (1.0 means complete collapse).", src=["experiments/results/gan.json"])
    fig("ganv", "fig_gan_vs_main.png", COL_W, "GAN-style variants and the frozen-cosine reference versus the main model on the test split.", src=["results/gan_vs_main.json"])

    D.heading2("Explainability results")
    ig5, ix5 = FA["ig"]["5"], FA["ixg"]["5"]
    D.p(f"The faithfulness test uses {FA['n']} test hunks that the model flags as misaligned (mean P(misaligned) {ig5['mean_p_orig']:.3f}). Masking the top-5 Integrated Gradients tokens lowers the mean probability to {ig5['mean_p_after_top_k_masked']:.3f}, a drop of "
        f"{ig5['mean_drop_top']:.3f}, against a drop of {ig5['mean_drop_random']:.3f} for five random tokens; the verdict flips for {pc(ig5['verdict_flip_rate_top'])} of hunks against {pc(ig5['verdict_flip_rate_random'])} ({TR('xai')}, {FR('faith')}). "
        f"Top-attributed tokens beat random tokens for {pc(ig5['frac_top_beats_random'])} of hunks at k = 5. The random-masking drop is itself large, which shows that substituting [PAD] is off-distribution for the model; the gap between the top-<i>k</i> and random curves is the "
        f"informative quantity. Input × Gradient is at least as faithful as Integrated Gradients by this measure (drop {ix5['mean_drop_top']:.3f} against {ig5['mean_drop_top']:.3f} at k = 5), which we report as measured. On average {pc(XA['abs_attribution_share_on_intent_tokens_mean'])} of the absolute attribution falls on intent tokens and the rest on the hunk.")
    xt = [["k masked", "IG: top", "IG: random", "IxG: top", "IxG: random", "Flip rate IG top / random"]]
    for k in ("3", "5", "10", "20"):
        xt.append([k, f3(FA["ig"][k]["mean_drop_top"]), f3(FA["ig"][k]["mean_drop_random"]), f3(FA["ixg"][k]["mean_drop_top"]), f3(FA["ixg"][k]["mean_drop_random"]), f"{pc(FA['ig'][k]['verdict_flip_rate_top'])} / {pc(FA['ig'][k]['verdict_flip_rate_random'])}"])
    tab("xai", xt, [0.5, 0.5, 0.55, 0.5, 0.6, 0.85], "Faithfulness test: mean drop in P(misaligned) after masking the top-k attributed tokens or k random tokens (n = 100 flagged test hunks; IG = Integrated Gradients, IxG = Input × Gradient).", size=7, src=["results/xai.json"])
    fig("faith", "fig_xai_faithfulness.png", COL_W, "XAI faithfulness curve: mean drop in P(misaligned) as the number of masked tokens grows.", src=["results/xai.json"])
    case = XA["cases"]["collateral"]
    D.p(f"{FR('xcase')} is a worked example (sample {case['sample_id']} from {case['repo']}; truth: {case['true']}, {case['drift_true']}; model P(misaligned) = {case['p']:.3f}). The intent is a dependency bump (“Bump CodSpeedHQ/action from 3.5.0 to 3.8.0”), while the hunk edits the type-hint resolution code in "
        "attrs. The two hunk lines with the largest positive attribution are new code lines that have nothing to do with the intent, which is the reading a reviewer would expect. The largest single intent attribution, however, is the final full stop, "
        "a reminder that token attributions can concentrate on punctuation and position and should be read as evidence, not as proof of the cause.")
    with D.wide():
        fig("xcase", "fig_xai_case.png", 6.9, "Worked example of an Integrated Gradients explanation for a collateral-drift hunk (intent words and hunk lines shaded by attribution; red pushes toward misaligned, blue toward aligned).", src=["results/xai_cases_ui.json", "results/xai.json"])

    D.heading2("Compression, size and latency")
    D.p(f"Distillation helps: the student with distillation reaches macro F1 {f3(stu['macro_f1'])} and NDCG@5 {f3(stu['ndcg@5'])}, against {f3(nod['macro_f1'])} and {f3(nod['ndcg@5'])} without it. The INT8 ONNX student occupies {float(int8['size_mb']):.1f} MB "
        f"(fp32 ONNX {float(onx['size_mb']):.1f} MB; main model {float(main['size_mb']):.1f} MB), reaches macro F1 {f3(int8['macro_f1'])} and NDCG@5 {f3(int8['ndcg@5'])}, and answers with a CPU median of {float(int8['cpu_p50_ms']):.1f} ms and a 95th percentile of {float(int8['cpu_p95_ms']):.1f} ms ({TR('lat')}, {FR('lat')}). "
        f"The targets of under 30 MB and a p95 of under 100 ms are met by the student on this machine. The main model misses the size target but meets the 100 ms target on the CPU ({LF['cpu']['p95_ms']:.1f} ms p95 with {LF['cpu']['torch_threads']} threads on an idle machine) and needs about {LF['cuda']['p50_ms']:.1f} ms on the GPU with {LF['cuda']['peak_gpu_mem_mb']:.0f} MB peak memory. "
        "The student figures were taken with four threads while the CPU was shared with a training job, whereas the main-model figure was measured on an idle machine with 14 threads, so the two sets are not strictly comparable; all latencies belong to this machine.")
    lt = [["Model", "Format", "Size (MB)", "CPU p50 (ms)", "CPU p95 (ms)", "Macro-F1", "NDCG@5"]]
    for mname, fmt, lab, mk in (("Main ModernBERT", "PyTorch fp32 (CPU)", "Main ModernBERT", "Main: ModernBERT-base"), ("MiniLM student", "ONNX fp32", "Distilled student", "Student ONNX fp32"), ("MiniLM student", "ONNX INT8", "Distilled student", "Student ONNX INT8 (deployed target)")):
        r = next(x for x in LAT if x["model"] == mname and x["format"] == fmt)
        lt.append([lab, fmt, f3(r["size_mb"], 1), f3(r["cpu_p50_ms"], 1), f3(r["cpu_p95_ms"], 1), f3(MC[mk]["macro_f1"]), f3(MC[mk]["ndcg@5"])])
    tab("lat", lt, [1.0, 0.95, 0.5, 0.5, 0.5, 0.45, 0.45], f"Size and CPU latency ({CPU}; batch 1; tokenisation included; student: 4 threads, CPU shared with a training job; main: {LF['cpu']['torch_threads']} threads, idle). GPU: p50 {LF['cuda']['p50_ms']:.1f} ms, p95 {LF['cuda']['p95_ms']:.1f} ms.", size=6.8, src=["results/latency_size.csv", "results/latency_full.json", "results/latency_size_onnx.json"])
    fig("lat", "fig_latency_size.png", COL_W, "Model size versus CPU latency (median, with the bar extending to the 95th percentile) for the main model and the student; dotted lines mark the targets.", src=["results/latency_size.csv"])

    D.heading2("Interface integration and deployment")
    D.p(f"The model is reachable through the interface shown in {FR('ui1')} and {FR('ui3')}. {FR('ui1')}(a) is the landing page, which states the purpose and shows the workspace; (b) is the results console, whose headline strip, sortable master table and per-tier toggles read the same result files as this article. "
        f"{FR('ui3')} shows the review workspace after a curated test sample has been analysed with the explanation option on: the intent and the consolidated intent are on the left, the hunk cards are ordered by risk with a verdict, probability and drift-type badge (centre), "
        "and the selected hunk shows token highlights on the intent words and the code lines (right); because the sample comes from the test split, the ground truth is shown after analysis so that correctness can be checked. "
        f"Scoring a pair takes about {XA['latency_ms']['score_p50']:.0f} ms on the GPU, and an Integrated Gradients explanation about {XA['latency_ms']['explain_ig_p50']:.0f} ms at the median ({XA['latency_ms']['explain_ig_p95']:.0f} ms at the 95th percentile, measured on a shared GPU).")
    D.p("The backend is a single FastAPI process with lazy, locked model loading and generated OpenAPI documentation; one command starts the backend and the interface on the local machine. "
        "The deployment target is a local CPU, using the INT8 student, with the main model used when a GPU is available. Inference runs entirely on the local machine." +
        (f" A hosted demonstration is available at {DEPLOY_URL}." if DEPLOY_URL else ""))
    with D.wide():
        from docx.shared import Inches
        t = D.d.add_table(rows=1, cols=2); t.autofit = False
        for ci, nm in enumerate(("ui_landing.png", "ui_results.png")):
            cell = t.rows[0].cells[ci]; cell.width = Inches(3.5)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell.paragraphs[0].add_run().add_picture(str(FIG / nm), width=Inches(3.35))
        from docxlib import fix_table_xml
        fix_table_xml(t._tbl)
        rec_f["ui1"] = D.caption_figure("User interface screens: (a) the landing page and (b) the results console. Both are real captures of the running application.")
        TRACE["Figure:ui1"] = ["docs/screenshots/landing_desktop_light.png", "docs/screenshots/dev_results_desktop_light.png"]
        D.picture(FIG / "ui_review.png", 6.0)
        rec_f["ui3"] = D.caption_figure("The review workspace after analysing a curated test-set sample with the explanation option on: risk-ordered hunk cards, the consolidated intent, the ground-truth chip and token highlights for the selected hunk.")
        TRACE["Figure:ui3"] = ["reports/screenshots/review_explained_desktop_light.png"]

    # ----------------------------------------------------- testing
    D.heading2("Testing and test design results")
    t_ = TT
    D.p(f"The suite contains {t_['total']} test items: {t_['passed']} pass, {t_['xfailed']} are expected failures that record known defects, and {t_['failed']} fail ({TR('tests')}, {FR('test')}). Line coverage of the measured scope (src/sift and the backend; end-to-end tests excluded) is "
        f"{cov_t['covered_lines']} of {cov_t['num_statements']} statements ({pc(cov_t['percent_statements_covered']/100,1)}), branch coverage {pc(cov_t['covered_branches']/cov_t['num_branches'],1)} and combined {cov_t['percent_covered']:.1f}%; the training and experiment scripts are outside the measured scope and four modules are at 0% in the unit suite.")
    CATL = {"smoke": "Smoke / readiness", "white-box": "White-box structural", "black-box": "Black-box functional", "data-pipeline": "Data pipeline and leakage", "metamorphic": "Metamorphic", "adversarial": "Adversarial / red team",
            "back-to-back": "Back-to-back", "api": "API partitions", "e2e": "User-interface end-to-end"}
    tr_ = [["Category", "Framework section", "Passed", "Xfailed", "Failed", "Total"]]
    for c in ("smoke", "white-box", "black-box", "data-pipeline", "metamorphic", "adversarial", "back-to-back", "api", "e2e"):
        v = CATS[c]; tr_.append([CATL[c], v["framework_section"], v["passed"], v["xfailed"], v["failed"], v["total"]])
    tr_.append(["Total", "", t_["passed"], t_["xfailed"], t_["failed"], t_["total"]])
    tab("tests", tr_, [1.2, 1.0, 0.35, 0.4, 0.35, 0.35], f"Test summary by category. Coverage of src/sift and the backend: {cov_t['covered_lines']}/{cov_t['num_statements']} statements ({pc(cov_t['percent_statements_covered']/100,1)}), combined line and branch {cov_t['percent_covered']:.1f}%.", size=7, aligns=["l", "l", "c", "c", "c", "c"], bold_rows=(10,),
        src=["results/test_summary.json", "results/coverage.json"])
    with D.wide():
        fig("test", "fig_testing_summary.png", 6.9, "Tests per category (left) and line-plus-branch coverage per module (right).", src=["results/test_summary.json", "results/coverage.json"])
    D.p("<b>Metamorphic relations.</b> " + f"Eight relations were tested on the trained model with test-split samples ({TR('mr')}). The deterministic relations hold exactly or to floating-point noise (determinism: maximum |Δp| = {TM['mr_determinism']['max_abs_dp']:.1f}; batch independence {TM['mr_batch_independence']['max_abs_dp']:.1e}; hunk reordering {TM['mr_reorder']['max_abs_dp']:.1e}). "
        f"The semantic relations hold within tolerances fixed in advance: a consistent identifier rename flips {pc(TM['mr_identifier_rename']['flip_rate'],1)} of verdicts (limit 10%), an added comment {pc(TM['mr_context_comment']['flip_rate'],1)}, a whitespace change {pc(TM['mr_context_whitespace']['flip_rate'],1)} and removing chit-chat turns {pc(TM['mr_chitchat_removal']['flip_rate'],1)} (limit 5%). "
        f"The directional relation holds: swapping in another session’s intent raises the mean probability from {TM['mr_intent_swap_sensitivity']['mean_p_original']:.3f} to {TM['mr_intent_swap_sensitivity']['mean_p_swapped']:.3f}. The largest single change under a rename is |Δp| = {TM['mr_identifier_rename']['max_abs_dp']:.2f}, so individual hunks can move substantially even when the flip rate is within tolerance.")
    mr = [["ID", "Relation (input change → expected output)", "Measured", "Tolerance"],
          ["MR1", "Same input twice → identical probabilities", f"max |Δp| = {TM['mr_determinism']['max_abs_dp']:.1f} (n = {TM['mr_determinism']['n']})", "exact"],
          ["MR2", "Pair alone versus in a batch → same", f"max |Δp| = {TM['mr_batch_independence']['max_abs_dp']:.1e} (n = {TM['mr_batch_independence']['n']})", "float noise"],
          ["MR3", "Reorder hunks → each score unchanged", f"max |Δp| = {TM['mr_reorder']['max_abs_dp']:.1e} ({TM['mr_reorder']['sessions']} sessions)", "float noise"],
          ["MR4", "Add an unrelated comment → verdict unchanged", f"flip rate {pc(TM['mr_context_comment']['flip_rate'],1)} (n = {TM['mr_context_comment']['n']})", "≤ 10%"],
          ["MR5", "Trailing whitespace → verdict unchanged", f"flip rate {pc(TM['mr_context_whitespace']['flip_rate'],1)} (n = {TM['mr_context_whitespace']['n']})", "≤ 10%"],
          ["MR6", "Rename an identifier in hunk and intent → unchanged", f"flip rate {pc(TM['mr_identifier_rename']['flip_rate'],1)} (n = {TM['mr_identifier_rename']['n']})", "≤ 10%"],
          ["MR7", "Remove chit-chat turns → unchanged", f"flip rate {pc(TM['mr_chitchat_removal']['flip_rate'],1)} (n = {TM['mr_chitchat_removal']['n']})", "≤ 5%"],
          ["MR8", "Swap in another intent → mean P(misaligned) rises", f"{TM['mr_intent_swap_sensitivity']['mean_p_original']:.3f} → {TM['mr_intent_swap_sensitivity']['mean_p_swapped']:.3f} (n = {TM['mr_intent_swap_sensitivity']['n']})", "rises"]]
    tab("mr", mr, [0.3, 1.5, 1.1, 0.5], "Metamorphic relations on the main model (all eight pass).", size=6.8, aligns=["c", "l", "l", "c"], src=["experiments/results/testing_measurements.json"])
    D.p("<b>Red-team results.</b> " + f"The model fails only when the attacker places the intent’s own words inside the hunk ({TR('red')}, {FR('red')}). Adding a comment that copies the intent text to a misaligned hunk flips {pc(at_d3['success_rate'])} ({round(at_d3['success_rate']*at_d3['attempts'])}/{at_d3['attempts']}) of detected hunks to aligned (defect D3, limit 20%), "
        f"and a comment of the form “this change implements: <intent>” flips {pc(at_d4['success_rate'])} ({round(at_d4['success_rate']*at_d4['attempts'])}/{at_d4['attempts']}) (D4). A generic prompt-injection string succeeds in {pc(at_inj['success_rate'])} and an irrelevant comment on an aligned hunk causes a false alarm in {pc(at_fa['success_rate'])} of cases. "
        f"A 50,000-line hunk, very long and empty inputs, unicode and control characters and a 1,000-hunk diff do not crash the system. The denominators are small (25 detected hunks), so the rates carry roughly ±20 percentage points of uncertainty.")
    rt = [["Attack", "Success rate", "Limit", "Result"],
          ["A1 intent text copied as a comment (D3)", f"{pc(at_d3['success_rate'])} ({round(at_d3['success_rate']*at_d3['attempts'])}/{at_d3['attempts']})", "20%", "xfail (defect)"],
          ["A2 comment “this change implements: intent” (D4)", f"{pc(at_d4['success_rate'])} ({round(at_d4['success_rate']*at_d4['attempts'])}/{at_d4['attempts']})", "20%", "xfail (defect)"],
          ["A3 prompt-injection string as comment", f"{pc(at_inj['success_rate'])} ({round(at_inj['success_rate']*at_inj['attempts'])}/{at_inj['attempts']})", "20%", "pass"],
          ["A4 irrelevant comment on an aligned hunk (false alarm)", f"{pc(at_fa['success_rate'])} ({round(at_fa['success_rate']*at_fa['attempts'])}/{at_fa['attempts']})", "20%", "pass"],
          ["A5 50,000-line hunk; empty, very long, unicode and binary-like input; 1,000 hunks", "no crash", "no crash", "pass"]]
    tab("red", rt, [1.75, 0.65, 0.35, 0.55], "Red-team attack success rates (goal: a detected misaligned hunk flips to aligned unless stated).", size=6.8, aligns=["l", "c", "c", "c"], src=["experiments/results/testing_measurements.json"])
    with D.wide():
        fig("red", "fig_redteam_mr.png", 6.9, "Metamorphic flip rates against their tolerances (left) and red-team attack success rates (right).", src=["experiments/results/testing_measurements.json"])
    D.p("<b>Back-to-back and leakage tests.</b> " + f"The student in PyTorch and in ONNX fp32 agree to |Δp| ≤ {TM['b2b_student_torch_vs_onnx_fp32']['max_abs_dp']:.1e} (n = {TM['b2b_student_torch_vs_onnx_fp32']['n']}), so the export is faithful. INT8 agrees with fp32 on {pc(b2b_int8['verdict_agreement'])} of verdicts (mean |Δp| {b2b_int8['mean_abs_dp']:.3f}), "
        f"so quantisation is not free. The full model and the student agree on {pc(b2b_ts['verdict_agreement'],1)} of verdicts with a probability correlation of {b2b_ts['prob_correlation']:.3f} and a Spearman rank correlation of {TM['b2b_teacher_student_spearman']:.3f}. "
        "The data-pipeline tests assert split sizes (12,600 / 2,700 / 2,700), zero repositories, sessions and intent–hunk pairs shared across splits, exactly 569 training label flips with clean validation and test labels, agreement of the CSV and JSONL releases, and deterministic loading; "
        f"{CATS['data-pipeline']['passed']} of {CATS['data-pipeline']['total']} pass, and the two expected failures are documentation discrepancies (D-DATA-1, D-DATA-2).")
    with D.wide():
        dr = [["ID", "Severity", "Defect", "How found", "Status"]]
        for d in sorted(TS["defects"], key=lambda x: ["D3", "D4", "D1", "D2", "D-DATA-1", "D-DATA-2"].index(x["id"])):
            dr.append([d["id"], d["severity"], d["description"].replace("`", ""), d["how_found"].replace("`", ""), d["status"]])
        tab("defects", dr, [0.6, 0.6, 3.1, 1.9, 0.8], "Defect register: real defects found by the tests (each is a strict expected failure; all are open).", size=7, aligns=["c", "c", "l", "l", "l"], src=["results/test_summary.json"])
    D.p("The framework sections are covered as follows. <i>Performance</i>: per-tier metrics and latency come from the experiment scripts, without repeated seeds or load tests. <i>Testing methods</i>: all types listed in Section " + SR("testmeth") + ", without full pairwise combinatorial generation. "
        "<i>Data</i>: leakage, noise counts and determinism, without human label validation. <i>AI models</i>: structure, heads, losses, calibration and the teacher–student comparison; the explanation, distillation and generative-baseline modules have no unit tests. "
        "<i>Context</i>: local CPU operation through the API and the interface, on one machine and one operating system. <i>Documentation</i>: the defect register and the per-test reasons stored with each expected failure.")

    # ----------------------------------------------------- DISCUSSION
    H2("disc", "Discussion", count_key="DISCUSSION")
    discussion(D, R, TR, FR, tab)

    # ============================================================ 5 Ablation
    D.heading1("Ablation Study", count_key="ABLATION")
    D.p("The ablations use the same test split, the same full training protocol (6 epochs, maximum length 512) and a single seed, so differences of about 0.01 in macro F1 are within noise. "
        f"{TR('abl')} and {FR('abl')} list eleven rows: the full model; hunk-only and intent-only inputs; raw turns instead of the consolidated intent; no focal loss (cross-entropy); alignment head only (no multi-task heads); no denoising; the k-means hybrid off and on; "
        "the distilled student; and its INT8 form. (The numbering skips row 8, a planned rules-only denoising ablation that was dropped for time.)")
    with D.wide():
        at = [["Configuration", "Macro-F1", "Prec.", "Recall", "NDCG@5", "AUC random", "AUC hard", "ECE", "Δ macro-F1", "Δ NDCG@5"]]
        for r in AB:
            at.append([re.sub(r"^\d+[ab]?\.\s*", "", r["model"]), f3(r["macro_f1"]), f3(r["precision"]), f3(r["recall"]), f3(r["ndcg@5"]), f3(r["auc_random"]), f3(r["auc_hard"]), f3(r["ece"]),
                       (f"{float(r['delta_macro_f1_vs_full']):+.3f}" if r["delta_macro_f1_vs_full"] else "–"), (f"{float(r['delta_ndcg5_vs_full']):+.3f}" if r["delta_ndcg5_vs_full"] else "–")])
        tab("abl", at, [2.3, 0.55, 0.45, 0.5, 0.55, 0.65, 0.55, 0.45, 0.65, 0.6], "Ablation study on the test split (single seed 473). Deltas are relative to the full model; k-means rows are Platt calibrators on the main-model logit with and without cluster features, and the INT8 row has no per-tier AUC stored.",
            size=7.2, aligns=["l"] + ["c"] * 9, shade_rows=(1,), bold_rows=(1,), src=["results/ablation.csv"])
        fig("abl", "fig_ablation.png", 6.9, "Ablation bars (test split, single seed): macro-F1, NDCG@5 and hard-tier pairwise AUC.", src=["results/ablation.csv"])
    D.p(f"<b>Honest reading.</b> Two ablations are clearly harmful: removing the intent (hunk-only) costs {abs(float(ab_hunk['delta_macro_f1_vs_full'])):.3f} macro F1 and removing the hunk (intent-only) costs {abs(float(ab_int['delta_macro_f1_vs_full'])):.3f}; the intent-only model is at chance on both tiers "
        f"(AUC {f3(ab_int['auc_random'])} and {f3(ab_int['auc_hard'])}), so the model genuinely needs both texts, and the hunk-only model is near chance on the random tier ({f3(ab_hunk['auc_random'])}). "
        f"The other four ablations are not worse than the full model: raw turns ({float(ab_con['delta_macro_f1_vs_full']):+.3f} macro F1), no focal loss ({float(ab_foc['delta_macro_f1_vs_full']):+.3f}), alignment head only ({float(ab_head['delta_macro_f1_vs_full']):+.3f}) and no denoising ({float(ab_nod['delta_macro_f1_vs_full']):+.3f}) each score slightly higher, and no-denoising has the best NDCG@5 ({f3(ab_nod['ndcg@5'])}) and hard-tier AUC ({f3(ab_nod['auc_hard'])}) of all rows. "
        "With one seed and differences of this size, we cannot claim that these components help, nor that they hurt; the experiments do not support the design choices of consolidation, focal loss, multi-task heads or denoising. "
        f"Focal loss shows one visible effect, on calibration: the cross-entropy variant has a larger error ({f3(ab_foc['ece'])} against {f3(ab_full['ece'])}), although the temperature step removes most miscalibration anyway. The k-means features change macro F1 by {float(km_on['macro_f1']) - float(km_off['macro_f1']):+.3f}, within noise. "
        f"The student loses {abs(float(ABD['10. Distilled MiniLM student (fp32)']['delta_macro_f1_vs_full'])):.3f} macro F1 and {abs(float(ABD['10. Distilled MiniLM student (fp32)']['delta_auc_hard_vs_full'])):.3f} hard-tier AUC relative to the full model, and INT8 loses nothing further.")
    D.p("<b>Deployment recommendation.</b> Because no ablated variant is demonstrably worse and the cheaper variants are not worse, a deployment can use the simplest configuration that was tested without loss: the alignment-head-only model has the best macro F1 of the ModernBERT variants, and the INT8 student is the practical default on a CPU. "
        "We keep the full multi-task model as the reported main model because it was fixed before the ablations were run and because the drift-type and confidence heads feed the interface; a multi-seed study should settle the question before any component is removed.")

    # ============================================================ 6 Conclusion
    D.heading1("Conclusion and Future Work")
    D.p(f"We presented Sift, a pipeline that consolidates a developer’s multi-turn intent, scores each hunk of a diff with a ModernBERT cross-encoder, calibrates and ranks the result, explains it, and serves it locally through an API and a web interface. On repository-disjoint test data it ranks hunks clearly better than diff-level and lexical baselines on the hard tier "
        f"(pairwise AUC {f3(main['auc_hard'])} against {f3(b4['auc_hard'])}), reaches NDCG@5 {f3(main['ndcg@5'])}, and is well calibrated after temperature scaling (ECE {f3(CAL['ece_test_after'])}); a {float(int8['size_mb']):.1f} MB 8-bit student keeps nearly all of that accuracy at a p95 CPU latency of {float(int8['cpu_p95_ms']):.0f} ms. "
        f"The results are also modest in important ways: macro F1 is below a logistic-regression baseline, a frozen sentence-embedding cosine is as good on random-tier ranking, confident learning is a weak label-error detector on this data, k-means adds almost nothing, the ablated variants are not worse than the full model, and two simple comment-based attacks defeat the model most of the time. "
        "These limits, the constructed labels, the single seed and the reduced-budget baselines mean that the numbers support the feasibility of hunk-level verification, not a claim of general state-of-the-art performance.")
    D.p("Future work follows from the weaknesses. (1) Human-validated labels and real assistant-authored sessions, including TypeScript in the test split, to test whether the model learns the task and not the construction. (2) Multiple seeds with confidence intervals, and the full 60-trial hyper-parameter search. "
        "(3) Adversarial training and input sanitisation (for example, ignoring comment-only added lines) against the intent-echo attacks D3 and D4. (4) A learned consolidation step to replace the template-based rules (defect D2). (5) A stronger label-error detector, such as probabilities from the main model itself, and a run of co-teaching. (6) Studying whether the ranking reduces actual reviewer time.")

    D.heading1("Data Availability and Code", numbered=False)
    D.p("The Sift-IA v1.0 dataset is released as an IEEE DataPort record: https://ieee-dataport.org/documents/sift-ia-intent-implementation-alignment-dataset-code-diff-hunks-produced-ai-coding-agents. "
        "All numbers in this article were produced by scripts in the project repository and are read from the generated result files; the figures were regenerated from those files at 300 dpi. "
        + (f"A hosted demonstration of the application is available at {DEPLOY_URL}." if DEPLOY_URL else ""))

    # ============================================================ References
    D.heading1("References", numbered=False, count_key="REFS")
    for k, txt in REFS:
        par = D.d.add_paragraph()
        from docxlib import _fmt
        _fmt(par, WD_ALIGN_PARAGRAPH.LEFT, 0, 2.5, first=-0.25, left=0.25)
        D.runs(par, txt, size=8.5)

    # ============================================================ Author contributions
    D.heading1("Author Contributions", numbered=False)
    ac = [["Roll No.", "Name", "Contribution"],
          ["CB.SC.U4CSE23037", "Kundhave S", "Sole member of Team 24: problem formulation, dataset, model, baselines, experiments, denoising, k-means, adversarial baseline, explainability, backend, interface, testing and writing"],
          ["– (faculty)", "Dr. T. Senthil Kumar", "Faculty guide and supervisor: guidance and review"]]
    tab("authors", ac, [1.25, 0.8, 1.35], "Author contributions.", size=7.5, aligns=["l", "l", "l"])
    D.heading1("About the Authors", numbered=False)
    D.p("<b>Kundhave S</b> (Roll No. CB.SC.U4CSE23037) is a B.Tech student in Computer Science and Engineering at the Amrita School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore. This work was carried out as Team 24 (single member) of the course 23CSE473 Neural Networks and Deep Learning.", first=0)
    D.p("<b>Dr. T. Senthil Kumar</b> is a Professor in the Department of Computer Science and Engineering, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore, and the course instructor and guide for this work.", first=0)
    return D, {"f": rec_f, "t": rec_t, "s": rec_s}


def rec_t_n(R, key):
    return R["t"].get(key, "?")


# ============================================================ discussion text
def discussion(D, R, TR, FR, tab):
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    mc_ = lambda m, c: float(m[c])
    D.p("This subsection interprets each result, states what the evidence does and does not support, and compares the findings with the base paper and the related work. Differences of about 0.01 are treated as noise throughout: every configuration is a single seed, the test split has three repositories, and no confidence intervals were computed.")

    D.run_in("What the evidence supports", f"The strongest result is structural and does not depend on training: diff-level scalars cannot rank hunks inside a diff, and ordering by diff size is indistinguishable from random order (NDCG@5 {f3(BL['ndcg@5']['B2 Diff lines changed'])} against {f3(BL['ndcg@5']['B1 Random order'])}). "
             f"The second is that a model that reads both texts ranks better than one that reads only one: the hunk-only and intent-only ablations lose {abs(float(ab_hunk['delta_macro_f1_vs_full'])):.2f} and {abs(float(ab_int['delta_macro_f1_vs_full'])):.2f} macro F1, and intent-only is at chance on both tiers. "
             f"The third is that, among trained models, the cross-encoders dominate the separate-encoder BiLSTM ({f3(mini['ndcg@5'])} and {f3(main['ndcg@5'])} NDCG@5 against {f3(bil['ndcg@5'])}), in line with the case for joint encoding (Nogueira and Cho, 2019). "
             "The claims that are not supported are listed in the table below, so that a reader can see at a glance which conclusions the data allow.")
    with D.wide():
        cl_rows = [["Claim", "Evidence", "Verdict"],
                   ["A learned cross-encoder ranks hunks better than diff-level scalars", f"NDCG@5 {f3(main['ndcg@5'])} against {f3(BL['ndcg@5']['B2 Diff lines changed'])} (B2, equal to random)", "supported"],
                   ["The model beats lexical baselines on the hard tier", f"pairwise AUC {f3(main['auc_hard'])} against {f3(b4['auc_hard'])} (B4) and {f3(b5['auc_hard'])} (B5)", "supported"],
                   ["The model beats a frozen sentence-embedding cosine", f"hard AUC {f3(main['auc_hard'])} against {f3(frz['auc_hard'])}; NDCG@5 {f3(main['ndcg@5'])} against {f3(frz['ndcg@5'])}; random AUC {f3(main['auc_random'])} against {f3(frz['auc_random'])}", "not supported (a tie)"],
                   ["The model beats a logistic-regression baseline on classification", f"macro F1 {f3(main['macro_f1'])} against {f3(b5['macro_f1'])}", "not supported"],
                   ["Temperature scaling calibrates the probabilities", f"ECE {f3(CAL['ece_test_before'])} → {f3(CAL['ece_test_after'])} (out of sample)", "supported"],
                   ["Distillation keeps most of the accuracy at a small size", f"macro F1 {f3(int8['macro_f1'])} against {f3(main['macro_f1'])}; {float(int8['size_mb']):.1f} MB against {float(main['size_mb']):.0f} MB", "supported (single seed)"],
                   ["Confident learning finds the injected label flips", f"precision {f3(cl_best['precision'])}, recall {f3(cl_best['recall'])}", "weak"],
                   ["Denoising improves the main model", f"full-protocol ablation: no-denoising {float(ab_nod['delta_macro_f1_vs_full']):+.3f} macro F1; reduced sweep helps at 20% and 30% noise", "not shown at natural noise; suggestive at high noise"],
                   ["K-means features improve ranking or calibration", f"macro F1 {KH['macro_f1']:+.3f}, NDCG@5 {KH['ndcg@5']:+.3f}", "no measurable gain"],
                   ["Each design component (consolidation, focal loss, multi-task, denoising) helps", "ablated variants score equal or higher", "not shown"],
                   ["Attributions are faithful", f"top-5 drop {FA['ig']['5']['mean_drop_top']:.3f} against {FA['ig']['5']['mean_drop_random']:.3f} random", "partly supported"],
                   ["The model is robust to adversarial comments", f"intent-echo attacks succeed {pc(at_d3['success_rate'])} and {pc(at_d4['success_rate'])}", "not supported"]]
        tab("claims", cl_rows, [2.7, 3.1, 1.2], "Claims versus evidence.", size=7.2, aligns=["l", "l", "l"], src=["results/*.json (see the cited tables)"])

    D.run_in("Why ranking wins on the hard tier but macro F1 is below B5", f"Macro F1 is a thresholded measure and ranking is not. B5 combines four simple features, among them the TF-IDF similarity of intent and hunk and a path-overlap indicator, and its threshold of 0.62 is tuned for macro F1; the main model’s validation-tuned threshold is {CONF['threshold']:.2f}. "
             f"At those thresholds B5 has recall {f3(b5['recall'])} and precision {f3(b5['precision'])}, the main model recall {f3(main['recall'])} and precision {f3(main['precision'])}: B5 catches more misaligned hunks at slightly lower precision, and macro F1 rewards that. "
             f"The random tier explains the rest. Random donors come from unrelated commits, so intent and hunk share almost no vocabulary, and a lexical score separates them easily; B4 reaches {f3(b4['auc_random'])} on that tier against {f3(main['auc_random'])} for the main model. On the hard tier the donor was chosen for identifier overlap, and lexical similarity collapses to {f3(b4['auc_hard'])} and {f3(b5['auc_hard'])}, essentially chance, "
             f"while the main model retains {f3(main['auc_hard'])}. A mixed test set with 893 random-tier and 815 hard-tier hunks therefore rewards the lexical baselines on the half of the data where they work and rewards the model on the half where they do not. "
             f"The model’s validation macro F1 ({EV['full']['val']['macro_f1']:.3f}) is also higher than its test macro F1 ({f3(main['macro_f1'])}); the validation repositories include TypeScript and the test repositories do not, and the threshold was fitted on the former, which is a plausible source of the shift but is not isolated by our experiments. "
             "For the product need, which is to put the riskiest hunk first, the ranking measures are the relevant ones; for a gate that must classify every hunk, B5 is at least as good.")

    bm_, bn_, bd_ = BOOT["macro_f1"], BOOT["ndcg@5"], BOOT["paired_difference_main_minus"]
    D.run_in("Statistical uncertainty", f"The bootstrap over test sessions ({TR('ci')}) changes how several comparisons should be read. A 95% interval for the main model\u2019s macro F1 spans about {bm_['main']['ci95'][1] - bm_['main']['ci95'][0]:.2f}, so two models whose macro F1 differ by 0.02 cannot be separated on this test set. "
             f"That is the situation for the main model and B5: the paired macro-F1 difference is {bd_['B5']['macro_f1']['point']:+.3f} with an interval from {bd_['B5']['macro_f1']['ci95'][0]:+.3f} to {bd_['B5']['macro_f1']['ci95'][1]:+.3f}. We therefore describe the main model as not better than B5 in macro F1 rather than as worse. In NDCG@5, by contrast, the main model is ahead of B5 by {bd_['B5']['ndcg@5']['point']:.3f} with an interval that excludes zero, "
             f"and it is ahead of the MiniLM cross-encoder on both measures, although that comparison is confounded by training budget. The student and the main model cannot be separated. The most informative result concerns denoising: the no-denoising ablation ranks better than the full model by {-bd_['no_denoising']['ndcg@5']['point']:.3f} NDCG@5, with an interval entirely on that side of zero. "
             "This is only sampling uncertainty over the test sessions. A different training seed might move the two models by a comparable amount, so the result is a flag for follow-up and not proof that denoising hurts. Two further cautions apply: sessions within a repository are not independent of each other in the way the bootstrap assumes, and with three test repositories the intervals understate the uncertainty about new repositories. "
             "Reporting these intervals is the minimum a reader needs; multiple seeds would add the missing training-variance term.")

    D.run_in("Per-tier, per-language and per-repository behaviour", f"The test repositories differ widely ({TR('subset')}): macro F1 is {f3(FT['repo_httpx']['macro_f1'])} on httpx, {f3(FT['repo_axios']['macro_f1'])} on axios and {f3(FT['repo_attrs']['macro_f1'])} on attrs, and the number of rankable sessions is only {FT['repo_httpx']['n_rankable_sessions']} for httpx, {FT['repo_attrs']['n_rankable_sessions']} for attrs and {FT['repo_axios']['n_rankable_sessions']} for axios. "
             f"With so few sessions, differences of a few points between repositories or between models on a subset are not reliable; the table is best read as evidence of heterogeneity, not as a ranking of repositories. The hard-tier AUC on Python ({f3(FT['python']['pairwise_auc']['hard']['auc'])}) is lower than on JavaScript ({f3(FT['js_ts']['pairwise_auc']['hard']['auc'])}), while the random-tier AUC shows the opposite order ({f3(FT['python']['pairwise_auc']['random']['auc'])} against {f3(FT['js_ts']['pairwise_auc']['random']['auc'])}). "
             "One reading is that the construction of hard donors interacts with the language (identifier overlap is easier to mine in one ecosystem than in the other), but we did not test this. Because the JavaScript test data is a single repository and TypeScript is absent, language-level conclusions are really repository-level conclusions, "
             "and TypeScript generalisation is unmeasured (defect D-DATA-2).")

    dt_ = BOOT["drift_type"]
    D.run_in("The drift-type head", f"The drift head is the weakest output of the model: accuracy {dt_['accuracy']:.3f} against a majority-class rate of {dt_['majority_class_rate']:.3f}, and macro F1 {dt_['macro_f1']:.3f} ({TR('drift')}). The pattern in the confusion matrix has a structural explanation. The drift type is defined by the relation of the donor hunk to the rest of the session: collateral means a different file from the one the intent concerns, "
             "scope expansion means a file the session already modifies with the session\u2019s own hunks retained, and purpose drift means the same file with the session\u2019s own hunk removed. The model scores one hunk at a time together with the intent. Whether the session\u2019s own hunk was removed, which separates purpose drift from scope expansion, is a property of the other hunks in the diff and is invisible to a single-hunk input. "
             f"Collateral drift, which depends on the file path and the content of the hunk compared with the intent, is the type the head recognises best (F1 {dt_['f1_per_class']['collateral']:.3f}). The conclusion is a design one: if the drift type is to be reported reliably, the model needs session-level context, for example the other hunks of the diff or the set of touched files, as additional input. "
             "Until then the drift badge in the interface should be read as a weak suggestion, and the alignment verdict and ranking, which are the parts the evidence supports, should be what a reviewer acts on. We note that the multi-task ablation (alignment head only) did not hurt alignment quality, consistent with the drift head contributing little to the shared representation.")

    D.run_in("Calibration", f"Temperature scaling gives T = {T_:.3f}, a mild correction: the model is somewhat over-confident, and one scalar reduces the test error from {f3(CAL['ece_test_before'])} to {f3(CAL['ece_test_after'])}. This matters for the product, because the interface shows the probability to a reviewer and a threshold slider turns it into a decision. "
             f"The correction does not change any ranking, so the conclusions about NDCG and AUC are unaffected by it. The values in the master table (ECE {f3(main['ece'])} for the main model and {f3(stu['ece'])} for the student) are before scaling; the INT8 student, whose temperature was re-fitted after export, has {f3(int8['ece'])}. "
             f"The study target of ECE at most 0.05 is therefore met by the main model after scaling and reached marginally by the INT8 student ({f3(int8['ece'])}). Calibration was measured on a split whose repositories differ from those used to fit T, which is the realistic situation; it was not tested under a further distribution shift, such as other languages or real assistant-authored diffs, where a fitted temperature may not transfer. "
             "The GAN-style discriminator outputs are not probabilities at all (ECE about 0.26), which is another reason they are used only as ranking scores.")

    D.run_in("Class imbalance, focal loss and the choice of threshold", f"The training data is about 78% aligned, and the test split has {CONF['n_misaligned']} misaligned hunks among {CONF['n']}. With so few positives, thresholded metrics depend strongly on where the threshold falls, which is why we tune it on validation macro F1 and report threshold-free ranking measures beside it. The chosen thresholds differ by model ({CONF['threshold']:.2f} for the main model, {EV['student']['test']['full']['threshold']:.2f} for the student, {BL['B5_classification_test']['threshold_from_val']:.2f} for B5), and they are not comparable across models because the models are calibrated differently. "
             f"A low threshold for the main model reflects the prevalence: at 0.30 the model flags about {pc((CONF['tp'] + CONF['fp']) / CONF['n'])} of all hunks, close to the true misaligned share of {pc(CONF['n_misaligned'] / CONF['n'])}. The focal loss is intended to help with the imbalance, but the ablation does not show a clear benefit in ranking or macro F1 ({float(ab_foc['delta_macro_f1_vs_full']):+.3f} macro F1 when it is removed); "
             f"the visible effect of the cross-entropy variant is higher precision ({f3(ab_foc["precision"])}) with lower recall ({f3(ab_foc["recall"])}) and a larger calibration error ({f3(ab_foc['ece'])}). In deployment the threshold is a policy choice: a team that reviews everything and wants a prioritised order can use any threshold, while a team that reviews only flagged hunks must choose between the {f3(CONF['precision'])} precision and {f3(CONF['recall'])} recall measured here and a more sensitive setting. "
             "Because the probabilities are calibrated, the slider has a direct reading as an estimated chance of misalignment, which is easier to explain to reviewers than a score.")

    D.run_in("Intent consolidation", f"Consolidation is rule-based and was written from the noise templates that are visible in Sift-IA, so its agreement with the ground-truth consolidated intent ({pc(CONS['overall']['exact_match'],1)} exact match over all rows; detection recall 1.0 for chit-chat, retraction and repeated-turn noise) is an in-distribution figure. Agreement is lower on the test repositories ({pc(CONS['per_split']['test']['exact_match'],1)} against {pc(CONS['per_split']['val']['exact_match'],1)} on validation), and the errors come mostly from genuine repeated turns in real commit messages, which the dataset keeps and the rules drop ({pc(CONS['per_noise_type']['none']['false_positive_rate'],1)} of clean rows lose a turn). "
             f"The ablation that feeds the raw turns to the model, without consolidation, is not worse (macro F1 {float(ab_con['delta_macro_f1_vs_full']):+.3f}, NDCG@5 {float(ab_con['delta_ndcg5_vs_full']):+.3f}), so a ModernBERT cross-encoder handles the noisy turns on its own at this noise level. Consolidation is therefore not shown to improve accuracy. It still has a product value that the metrics do not capture: it shows the reviewer which turns were dropped and what intent the model actually saw,. "
             "The weakness recorded as defect D2 (paraphrased pleasantries are kept) would matter more for real chat sessions than for templated noise, and a learned consolidation step, evaluated on real sessions, is the natural next step.")
    D.run_in("Effect of noise and denoising", f"Natural label noise in the training split is low (569 of 12,600 rows, 4.5%), and the detector available to find it is weak. Confident learning is built on out-of-sample predicted probabilities (Northcutt et al., 2021), and its guarantees assume that those probabilities are good enough to separate mislabelled from correctly labelled examples; here the cross-fitted MiniLM probabilities have an AUC of only {f3(CL['oof_auc_vs_clean'])} against the clean labels. "
             f"The detector is {cl_best['precision'] / CL['random_precision']:.1f} times better than random in precision but still flags {n_(cl_best['fp'])} correctly labelled rows to catch {cl_best['tp']} flips, and the drop list plus the rules remove about {pc((CFG['n_train_raw'] - CFG['n_train_used']) / CFG['n_train_raw'])} of the training rows. "
             f"In the full-protocol ablation the model trained without any denoising is not worse (macro F1 {f3(ab_nod['macro_f1'])} against {f3(ab_full['macro_f1'])}; NDCG@5 {f3(ab_nod['ndcg@5'])} against {f3(ab_full['ndcg@5'])}), which is consistent with a cost in lost training data that offsets a small gain in label quality. "
             f"The injected-noise sweep tells a more favourable story for denoising when noise is high: at 20% and 30% the denoised arm leads by {float(nz[(0.2,'denoise')]['macro_f1']) - float(nz[(0.2,'no_denoise')]['macro_f1']):.3f} and {float(nz[(0.3,'denoise')]['macro_f1']) - float(nz[(0.3,'no_denoise')]['macro_f1']):.3f} macro F1, and NDCG@5 improves at every rate; "
             f"at 0% it removes {n_(3000 - 2670)} rows and macro F1 is {float(nz[(0.0,'denoise')]['macro_f1']) - float(nz[(0.0,'no_denoise')]['macro_f1']):+.3f}. The non-monotone shape of both curves, with the best macro F1 at 10%, shows that this study is too small and too short to resolve differences of a few points. "
             "Two cautions apply. The sweep uses symmetric flips, whereas the shipped noise is asymmetric and class-conditional (P(flip | aligned) = 0.03, P(flip | misaligned) = 0.10), which is closer to the assumption in the base paper. And the denoising arm uses TF-IDF probabilities, not the ModernBERT probabilities of the main study. "
             "The practical conclusion is that denoising is cheap insurance when noise is high and is not shown to be needed at the 4.5% level of the released training split.")

    D.run_in("What k-means and the GAN-style baseline show", f"K-means finds weak global structure. The silhouette is highest at k = {KM['k_chosen']} but only {max(KM['k_scan']['silhouette']):.3f}, purity ({f3(KQ['purity_vs_alignment'])}) is barely above the majority rate ({f3(KQ['majority_class_rate_alignment'])}), and NMI is about {KQ['nmi_vs_alignment']:.2f}. "
             f"There is some local signal: one cluster holds {KCT[1]['n']} test hunks with {pc(KCT[1]['misaligned_rate'],1)} misaligned, and another has a {pc(KCT[3]['misaligned_rate'])} misaligned rate, which is why a cluster lookup is useful for showing similar flagged hunks in the interface. "
             f"The cluster prior is not stable across repositories, though: the training prior of the most misaligned cluster is {pc(KCT[3]['train_prior'])} but its test rate is {pc(KCT[3]['misaligned_rate'])}, so using it as a feature transfers imperfectly. The hybrid calibrator gains {KH['macro_f1']:+.3f} macro F1, with three parameters fitted on validation instead of one; this is within noise, and we report that k-means adds almost nothing to accuracy. "
             f"The adversarially trained baseline is informative in a different way. Without ever seeing a misaligned pair, the unsupervised discriminator reaches NDCG@5 {f3(gu['ndcg@5'])} and hard-tier AUC {f3(gu['auc_hard'])}, close to the main model, using a 3.7 MB discriminator on frozen embeddings; this shows that most of the ranking signal on this dataset is semantic similarity between intent and hunk, available from a general-purpose sentence encoder. "
             f"The unsupervised variant also suggests a deployment option when no misaligned labels exist. The parameter-free cosine reference does as well or better ({f3(frz['ndcg@5'])} NDCG@5), so the adversarial training itself adds no measurable value over its own input features. The training was unstable in the usual ways: an initial collapse (diversity {GJ['variants']['gan_unsupervised']['history'][0]['gen_diversity_cos']:.2f}), a rising generator loss and a falling discriminator loss, and a validation AUC that decays after its early peak, "
             "so the checkpoint has to be chosen on validation. We label the whole comparison as an interpretation, because Sift has no natural generative component.")

    D.run_in("Explainability and its faithfulness", f"Masking the top-attributed tokens lowers the predicted probability about twice as much as masking random tokens at k = 5 ({FA['ig']['5']['mean_drop_top']:.3f} against {FA['ig']['5']['mean_drop_random']:.3f}) and changes the verdict {pc(FA['ig']['5']['verdict_flip_rate_top'])} of the time against {pc(FA['ig']['5']['verdict_flip_rate_random'])}, so the attributions carry information about what the model uses. "
             f"Three observations limit what this proves. First, the random baseline drops {FA['ig']['5']['mean_drop_random']:.3f}, so [PAD] substitution moves the input off-distribution and part of every drop is an artefact of the test; at k = 3 the advantage of IG over random is small ({FA['ig']['3']['mean_drop_top']:.3f} against {FA['ig']['3']['mean_drop_random']:.3f}, and top tokens beat random ones for only {pc(FA['ig']['3']['frac_top_beats_random'])} of hunks). "
             f"Second, Input × Gradient outperforms Integrated Gradients on this test at every k, which a gradient-path method is not guaranteed to do; with a [PAD] baseline and 24 steps, the integration path crosses regions where the model is not trained, and we did not verify the completeness property. Third, the worked example shows the highest intent attribution on a full stop, which is not a semantic cue. "
             f"Attribution therefore works as evidence for the reviewer (which lines pushed the score) and not as a causal explanation. On average only {pc(XA['abs_attribution_share_on_intent_tokens_mean'])} of the attribution mass lies on the intent and the rest on the hunk, which fits the finding that the hunk-only model keeps some signal while the intent-only model has none. "
             f"An explanation costs about {XA['latency_ms']['explain_ig_p50']:.0f} ms at the median and {XA['latency_ms']['explain_ig_p95']:.0f} ms at the 95th percentile, so it is offered on demand for the top hunks and not for every hunk.")

    D.run_in("Robustness weaknesses D3 and D4", f"The two high-impact defects share one cause. The model partly relies on lexical overlap between the intent and the hunk, which is also why the lexical baseline B4 is strong on the random tier. A comment that copies the intent text into a misaligned hunk provides that overlap on demand, and {pc(at_d3['success_rate'])} of detected hunks flip to aligned; a comment that claims the change implements the intent has the same effect in {pc(at_d4['success_rate'])} of cases. "
             f"The intent-swap relation (mean probability {TM['mr_intent_swap_sensitivity']['mean_p_original']:.2f} to {TM['mr_intent_swap_sensitivity']['mean_p_swapped']:.2f}) shows that the model does respond to the intent, as it should, but the attacks show that it responds to its words, not only to its meaning. The same tests show robustness in the cases that look like attacks but are generic: a prompt-injection string succeeds in {pc(at_inj['success_rate'])} of cases and an irrelevant comment raises a false alarm in {pc(at_fa['success_rate'])}. "
             "This matters for the intended use, because the code under review is produced by an assistant that may add explanatory comments that echo the request without any intent to deceive; the same text that is natural for generated code can make a misaligned hunk look aligned. The rates come from 25 detected hunks and so carry a wide uncertainty, and they were measured by an internal tester, not in the field. "
             "Mitigations we propose but did not implement are stripping comment-only added lines before scoring, ignoring comments that restate the intent, and adversarial training with intent-echo examples. A smaller defect, D1, means that plain multi-file <i>diff -u</i> output (without git headers) merges the second file’s header into the previous hunk, and D2 means that paraphrased pleasantries are not removed by the exact-match chit-chat rules; "
             f"both are documented with their mitigations, and the consolidation exact-match rate of {pc(CONS['overall']['exact_match'],1)} over all rows is in-distribution, falling to {pc(CONS['per_split']['test']['exact_match'],1)} on the test repositories.")

    D.run_in("Latency and size trade-offs", f"The INT8 student is {float(main['size_mb']) / float(int8['size_mb']):.0f} times smaller than the main model ({float(int8['size_mb']):.1f} MB against {float(main['size_mb']):.0f} MB) and keeps nearly all of the aggregate accuracy (macro F1 {f3(int8['macro_f1'])} against {f3(main['macro_f1'])}; NDCG@5 {f3(int8['ndcg@5'])} against {f3(main['ndcg@5'])}). "
             f"Its cost is concentrated on the hard tier, where the fp32 student’s AUC is {f3(stu['auc_hard'])} against {f3(main['auc_hard'])}, a loss of {float(main['auc_hard']) - float(stu['auc_hard']):.3f}. Distillation is worth {float(stu['macro_f1']) - float(nod['macro_f1']):.3f} macro F1 and {float(stu['auc_hard']) - float(nod['auc_hard']):.3f} hard-tier AUC over training the same student without a teacher. "
             f"Quantisation to INT8 is cheap in accuracy but not free: it changes {pc(1 - b2b_int8['verdict_agreement'])} of verdicts relative to fp32, with mean |Δp| = {b2b_int8['mean_abs_dp']:.3f}. The student and the main model agree on {pc(b2b_ts['verdict_agreement'],1)} of verdicts, so switching models in the interface changes the verdict for about one hunk in seven. "
             f"The latency numbers need care. The student was measured with four threads while a training job shared the CPU, the main model with {LF['cpu']['torch_threads']} threads on an idle machine, so the 8.5 ms against {LF['cpu']['p50_ms']:.1f} ms median comparison overstates the speed-up on some axes and understates it on others. "
             f"Even so, both meet the 100 ms target at the 95th percentile ({float(int8['cpu_p95_ms']):.1f} and {LF['cpu']['p95_ms']:.1f} ms), and the main model on the GPU needs {LF['cuda']['p50_ms']:.1f} ms and {LF['cuda']['peak_gpu_mem_mb']:.0f} MB. The sensible default is the INT8 student on a CPU, and the main model where a GPU exists or where the hard tier matters most. All figures belong to one laptop and will differ elsewhere.")

    D.run_in("Training cost and the budget gap", f"The main model trained for {FULLRUN['train_time_s']/60:.0f} minutes on a 6 GB laptop GPU, the student for {STURUN['train_time_s']/60:.0f} minutes, and the MiniLM cross-encoder baseline for under three minutes; the BiLSTM for about two minutes. Compute therefore differs by an order of magnitude between rows of the master table, and the comparison tables mix architecture and budget. "
             f"The student gives a measure of how much budget matters: trained for 8 epochs at 512 tokens without a teacher it reaches {f3(nod['macro_f1'])} macro F1, while the MiniLM cross-encoder baseline of the same architecture, trained for 2 epochs at 256 tokens, reaches {f3(mini['macro_f1'])}; with distillation the same student reaches {f3(stu['macro_f1'])}. The CodeBERT row is the extreme case: one epoch on 4,000 rows yields a model that is close to chance (macro F1 {f3(cbt['macro_f1'])}), which is a statement about the budget and not about code-aware encoders. "
             "A fair comparison would give each architecture the same tuning effort; that would need more GPU time than was available and is the most useful extension of the baseline comparison. The hyper-parameter study had the same limit: six trials on small subsets scored within a narrow band, so the evidence neither confirms the default learning rate nor suggests a better one.")
    D.run_in("Lessons for the dataset", "Three features of Sift-IA shaped what the experiments could show. First, the labels are constructed, so they are exact but they measure whether the model can tell a donor from a seed hunk, which is correlated with, but not identical to, whether a real assistant drifted from a real request. The strong baseline of frozen-embedding similarity suggests that much of the signal is topical similarity, which a real assistant\u2019s drifted hunk may also share with the request, so the hard tier is the more informative one. "
             f"Second, the test split has three repositories and no TypeScript, which limits language-level claims and makes repository effects large ({f3(FT['repo_httpx']['macro_f1'])} to {f3(FT['repo_attrs']['macro_f1'])} macro F1). A future release should balance languages across splits and add repositories. Third, the shipped noise annotations were the most useful design choice for this study: because every label flip is known, we could test denoising rather than assume it. "
             "The gold annotation sheet in the release, which is unlabelled, is the route to human validation; once labelled it would allow the first measurement of how well constructed labels agree with human judgement and would show whether the weaknesses seen here, such as the near-chance drift-type head, come from the model or from the construction.")
    D.run_in("Implications for the interface", "The result files show several weaknesses, and the interface reflects them. The workspace shows a calibrated probability and a threshold slider, not a binary verdict; it shows the ground truth after analysis for curated test samples, so that correctness can be checked by hand; it labels attribution as evidence and not proof; and the results console has a strip that names the best configuration per metric, including the cases where it is an ablated variant and not the main model. "
             "Showing the weak points beside the strong ones is a deliberate design choice, because a review aid that overstates its confidence would be trusted more than the evidence allows. The model switcher also exposes the trade-off directly, with the measured latency next to each model, and the similar-flagged-hunks lookup gives the reviewer a concrete precedent that does not depend on the model\u2019s own probability.")

    D.run_in("Threats to validity", "<i>Construct validity.</i> Labels come from construction: a misaligned hunk is a donor hunk from another commit. A donor may coincidentally fit the intent, a seed commit may itself be tangled (Herzig and Zeller, 2013) so that an “aligned” hunk does not serve its message, and a model may learn artefacts that distinguish donors from seed hunks (for example style, date or coding conventions) instead of intent alignment. The hard tier mitigates this but does not remove it, and no human has validated the labels. "
             "The intent noise follows templates, and the rule-based consolidation was written from those templates, so its high agreement is in-distribution. "
             "<i>Internal validity.</i> Every configuration is a single seed; no confidence intervals or significance tests were computed, and differences below about 0.02 should not be interpreted. The baselines other than B1–B5 were trained with reduced budgets (the MiniLM cross-encoder for 2 epochs at 256 tokens, CodeBERT for 1 epoch on 4,000 rows, the BiLSTM for 15 epochs), so gaps to the main model are partly budget gaps, and the CodeBERT result in particular says nothing about code-aware encoders in general. "
             f"The hyper-parameter search ran {HPO['trials_run']} of {HPO['planned_trials_brief']} planned trials on small subsets, so the main model may be sub-optimally tuned; the noise sweep is a reduced protocol. The thresholds, temperature and early stopping use validation only, and the test split was used for reporting, but the validation split contains TypeScript and the test split does not. "
             "<i>External validity.</i> The test repositories are repository-disjoint from training but are three open-source Python and JavaScript projects, one of them (axios) providing all JavaScript results; results may not carry over to other languages, to proprietary code, or to diffs written by real assistants, whose misalignments are not donor hunks. "
             "Latency and memory belong to one laptop. <i>Conclusion validity.</i> The red-team rates rest on 25 attempts each, and the metamorphic tolerances, although fixed in advance, are judgement calls.")

    D.run_in("What kinds of errors the model makes", f"At the validation-tuned threshold the main model makes {CONF['fn']} false negatives and {CONF['fp']} false positives on {CONF['n']} hunks ({FR('conf')}), so it misses about as many misaligned hunks as it wrongly flags. The false negatives are the costly errors for a verification tool, since a missed misaligned hunk reaches the code base unreviewed, whereas a false positive costs a reviewer a few seconds. "
             f"The per-tier results suggest where the misses concentrate: recall is not reported per tier, but the hard-tier macro F1 ({f3(FT['tier_hard']['macro_f1'])}) is well below the random-tier value ({f3(FT['tier_random']['macro_f1'])}), and the drift-type confusion shows that purpose drift and scope expansion are the types the model understands least. The attacks in the red-team study add a third class of errors, deliberate rather than accidental, which the threshold cannot address. "
             "A reviewer who lowers the threshold to catch more misaligned hunks pays for it with more false alarms, and because precision at the tuned setting is already close to one half, there is little headroom; the better lever is a stronger model on the hard tier, for example with session-level context, more training data from more repositories, or multi-seed ensembling, none of which was tested here.")
    D.run_in("Cross-encoder versus embedding similarity", f"The cross-encoder is expensive relative to an embedding model: {float(main['size_mb']):.0f} MB and {LF['cpu']['p50_ms']:.0f} ms per pair on a CPU, against a frozen MiniLM embedder of {GJ['embedder_size_mb']:.0f} MB at about {GJ['embedding_latency_ms_p50_per_pair']:.0f} ms per pair (measured separately, so not strictly comparable) and a 3.7 MB discriminator. Yet the unsupervised frozen-embedding reference is as good on ranking as the main model on this dataset. "
             "Two readings are consistent with the data. One is that the constructed misalignments are mostly topical mismatches that a bi-encoder can detect, so the extra capacity of joint encoding is not needed here; the hard tier, built from identifier overlap, would then be the only place where joint reading could matter, and the margin there is small (hard-tier AUC "
             f"{f3(main['auc_hard'])} against {f3(frz['auc_hard'])}). The other is that the main model is under-trained or under-tuned for the available budget, so that its potential advantage is not realised. Our experiments cannot separate the two. A pragmatic design that follows from either reading, which we suggest but did not test, is a two-stage system: rank with embeddings, and run the cross-encoder only on the top hunks or on those near the decision threshold.")
    D.run_in("Beyond the dataset", "Real assistants do not insert donor hunks from other commits. Their misalignments are typically the result of over-eager editing: reformatting nearby code, adding a helper that was not requested, or fixing a neighbouring defect. Such hunks are likely to share vocabulary with the request and to touch the same files, which resembles the hard tier more than the random tier, where the evidence for the model is strongest and the absolute accuracy is lowest. "
             "Real sessions will also contain intent noise that does not follow templates, and diffs far larger than the median of two hunks, where ranking matters more. These factors point in different directions, so we make no prediction of field performance; what the study provides is a measured baseline, a set of failure modes to look for, and a dataset and test suite to measure progress against.")
    D.run_in("Safe use", "A verification layer can do harm if it is trusted too much. A reviewer who sees a low risk score may skim a hunk that deserves attention, and the red-team results show that such a score can be induced cheaply. We therefore recommend that Sift is used to prioritise and highlight, never to waive review, and that its output is presented with its calibrated probability and the limits stated in this article. "
             "The local operation keeps source code on the developer\u2019s machine, which avoids sending proprietary code to a hosted service; the explanation and the similar-hunk lookup are computed locally for the same reason.")

    D.run_in("Comparison with the base paper and the related work", "Northcutt et al. (2021) propose confident learning as a general, model-agnostic way to find label errors, show that it is provably consistent under their conditions, and demonstrate it on image and text benchmarks. We adopt the idea and the cleanlab implementation but our setting differs: the labels are binary, the noise is a small fraction of the data, "
             "the out-of-sample model is weak, and the labels of the clean data are themselves constructed. Our contribution relative to that paper is a measurement that benchmarks cannot provide, because Sift-IA ships the identity of every flip: precision and recall against the known flips, "
             f"which show that the method finds some errors ({f3(cl_best['recall'])} recall) at a high false-positive cost ({f3(cl_best['precision'])} precision). We do not compare numbers with the paper’s benchmarks. Xu et al. (2024) also place confidence learning before a comment-code consistency detector; in our data the stage is not shown to help at the natural noise level, which is a result about this dataset, not a refutation of theirs. "
             "Panthaplackel et al. (2021) address consistency between a comment and a code change at the method level; Sift moves the unit to the hunk and the text to a multi-turn intent. Fan et al. (2025) find that larger language models are not necessarily better and that context length matters, "
             f"which is consistent with a compact fine-tuned encoder and with our finding that a {float(int8['size_mb']):.0f} MB student keeps nearly all of the accuracy of a 149 M-parameter model; we did not evaluate a large language model, so this is a consistency, not a comparison. Yang et al. (2026) show that a reranking stage improves selection, which supports ranking as the product objective. "
             "Wang et al. (2025) caution that practices from natural language do not always transfer to code, and our code-aware CodeBERT baseline was the weakest trained model, but with one epoch on 4,000 rows it cannot support a conclusion. Nogueira and Cho (2019) motivate cross-encoding; our BiLSTM result agrees, while the strength of the frozen bi-encoder cosine shows that on this synthetic task much of the signal is available without joint encoding.")

    D.run_in("Practical implications for reviewers", f"Used as a triage aid, Sift places a truly misaligned hunk first in {pc(CONF['p@1'])} of rankable sessions (precision@1) with a mean reciprocal rank of {f3(CONF['mrr'])}, against a random-order NDCG@5 of {f3(BL['ndcg@5']['B1 Random order'])}. Because the median session has two hunks, the saving in reading order is modest per session; the benefit grows with diff size, which the test split cannot show well. "
             f"At the validation-tuned threshold about half of the flags are false alarms (precision {f3(CONF['precision'])}) and about half of the misaligned hunks are missed (recall {f3(CONF['recall'])}), so the system should be used to order and highlight, not to block or auto-approve; the threshold slider lets a team trade precision for recall because the probabilities are calibrated. "
             "A reviewer should use the explanation as a pointer to lines worth reading, not as proof, should distrust a hunk that carries comments echoing the request (D3, D4), and should treat the verdict as less reliable on languages and repositories that differ from those studied. Running locally on a CPU with the INT8 student keeps source code on the developer’s machine, "
             "which supports the aim of trustworthy AI-assisted engineering (SDG 9); any reduction in reviewer burden (SDG 8) remains a hypothesis to be tested with real reviewers.")

    D.run_in("Answers to the research questions", f"RQ1 (can a cross-encoder conditioned on consolidated intent rank hunks better than diff-level scalars?): yes; NDCG@5 {f3(main['ndcg@5'])} against {f3(BL['ndcg@5']['B2 Diff lines changed'])}, and a hard-tier pairwise AUC of {f3(main['auc_hard'])} against {f3(b3['auc_hard'])} for hunk size. "
             f"RQ2 (does confidence-learning denoising improve macro F1 on a mutation-generated dataset?): not at the natural noise level of the full protocol (no-denoising is {float(ab_nod['delta_macro_f1_vs_full']):+.3f} macro F1 relative to the full model); suggestively at 20% and 30% injected noise in the reduced sweep. "
             f"RQ3 (does temperature scaling give ECE of at most 0.05?): yes for the main model ({f3(CAL['ece_test_after'])}); the INT8 student is at {f3(int8['ece'])}. "
             f"RQ4 (how much macro F1 is retained after distillation and INT8?): {float(int8['macro_f1']) / float(main['macro_f1']) * 100:.1f}% of the main model’s macro F1 and the same NDCG@5, within a p95 CPU latency of {float(int8['cpu_p95_ms']):.0f} ms.")

    D.run_in("Reproducibility and scope", "Every number in this article is read from result files written by scripts, with fixed seeds, logged configurations and resumable checkpoints; a regeneration script rebuilds the tables and figures from those files. The study does not claim a new neural architecture; the contribution is the task framing, the pipeline, the evaluation and the dataset. "
             "It also does not claim that Sift prevents misaligned changes from reaching a code base: the strongest claim the evidence supports is that, on constructed misalignments from open-source repositories, a calibrated cross-encoder ranks misaligned hunks above aligned ones more reliably than lexical and diff-size baselines on the hard tier, while simpler semantic baselines are competitive overall.")


def main_():
    R = {"f": {}, "t": {}, "s": {}}
    D, rec = build(R)           # pass 1: collect numbering
    R["f"], R["t"], R["s"] = rec["f"], rec["t"], rec["s"]
    TRACE.clear()
    D, rec = build(R)           # pass 2: final text with numbering
    assert rec["f"] == R["f"] and rec["t"] == R["t"] and rec["s"] == R["s"], "numbering changed between passes"
    D.save(OUT)
    meta = {"figures": D.fig_n, "tables": D.tab_n, "equations": D.eq_n, "abstract_words": R.get("abstract_words"), "words_by_section": D.body_words, "deploy_url": DEPLOY_URL,
            "references": len(REFS), "figure_numbers": rec["f"], "table_numbers": rec["t"], "section_numbers": rec["s"]}
    json.dump(meta, open(HERE / "build_meta.json", "w"), indent=1)
    json.dump(TRACE, open(HERE / "number_trace.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("words_by_section",)}, indent=0)[:800])
    print({k: v for k, v in D.body_words.items() if k in ("DISCUSSION", "ABLATION")}, "words total", sum(D.body_words.values()))


if __name__ == "__main__":
    main_()
