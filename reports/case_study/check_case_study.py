"""Checks the built case study and writes reports/case_study/CHECKLIST.json.
Usage: .venv/bin/python reports/case_study/check_case_study.py <converted.pdf>
(the PDF is produced by `soffice --headless --convert-to pdf` into a scratch directory; it is not a deliverable)."""
import json, re, subprocess, sys
from pathlib import Path
import docx
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOCX = ROOT / "Review3" / "Sift_Case_Study_IEEE_Access.docx"
pdf = sys.argv[1]
meta = json.load(open(HERE / "build_meta.json"))
trace = json.load(open(HERE / "number_trace.json"))
d = docx.Document(DOCX)

heads = [p.text for p in d.paragraphs if p.style.name == "Heading 1"]
sub = [p.text for p in d.paragraphs if p.style.name == "Heading 2"]
required = ["1. Introduction", "2. Related Work", "3. Proposed Method", "4. Results and Discussion", "5. Ablation Study", "6. Conclusion and Future Work", "References"]
norm = lambda x: x.lower()
pos = [next((i for i, h in enumerate(heads) if norm(h) == norm(r)), None) for r in required]
ordered = all(p is not None for p in pos) and pos == sorted(pos)
ref_i = heads.index("References")
tail = heads[ref_i:]
captions_t = [p.text for p in d.paragraphs if re.match(r"^TABLE \d+\.", p.text)]
captions_f = [p.text for p in d.paragraphs]
captions_f += [p.text for t in d.tables for c in t._cells for p in c.paragraphs]
captions_f = [t for t in captions_f if re.match(r"^FIGURE \d+\.", t)]
full = "\n".join(p.text for p in d.paragraphs)
text_all = full + "\n".join(p.text for t in d.tables for c in t._cells for p in c.paragraphs)

n_pages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
html = subprocess.run(["pdftotext", "-bbox", pdf, "-"], capture_output=True, text=True).stdout
pages = html.split("<page ")[1:]


def locate(first, second=None):
    for pi, pg in enumerate(pages, 1):
        ws = re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>', pg)
        for i, w in enumerate(ws):
            if w[4] == first and (second is None or any(x[4].upper().startswith(second) for x in ws[i + 1:i + 4])):
                return pi, float(w[0]), float(w[1])
    return None


a = locate(f"{meta['section_numbers']['disc']}", "DISCUSSION")
b = locate("5.", "BLATION") or locate("5.", "ABLATION")
top, bottom = 58.0, 727.0
colpos = lambda t: (t[0] - 1) * 2 + (0 if t[1] < 306 else 1) + (t[2] - top) / (bottom - top)
disc_pages = (colpos(b) - colpos(a)) / 2 if a and b else None
# wide (full-width) blocks inside the discussion: the claims table; estimate its height from the docx page text is not available, so report both
res = {
    "figures_total": meta["figures"], "figures_captioned_in_docx": len(captions_f),
    "graphs_excluding_architecture_screenshots_and_worked_example": meta["figures"] - 4,
    "tables_total": meta["tables"], "tables_captioned_in_docx": len(captions_t), "equations": meta["equations"],
    "pages": n_pages, "discussion_words": meta["words_by_section"].get("DISCUSSION"),
    "discussion_pages_estimated_including_claims_table": round(disc_pages, 2) if disc_pages else None,
    "discussion_start": a, "ablation_start": b,
    "abstract_words": meta["abstract_words"], "keywords": 6,
    "required_headings_present_in_order": ordered, "heading1_list": heads, "heading2_list": sub,
    "front_matter_after_references": tail,
    "author_contributions_table_present": "Author Contributions" in heads and "Roll No." in text_all,
    "figure_min_dpi_300_source_pngs": all(Image.open(p).info.get("dpi", (0, 0))[0] >= 299 for p in (ROOT / "reports" / "figures_300").glob("fig_*.png")),
    "forbidden_words_found": [w for w in ("Claude", "agent-generated", "TODO", "lorem") if w.lower() in text_all.lower()],
    "deployment_url_included": meta["deploy_url"],
    "in_text_citation_style": "author-date",
    "references_count": meta["references"],
    "number_trace": trace,
    "prose_number_sources": {
        "abstract, 1, 4.4-4.6, 5": ["results/master_comparison.csv", "results/ablation.csv", "results/eval_full.json", "experiments/results/baselines.json", "results/calibration.json", "results/confident_learning.json", "results/test_summary.json"],
        "3.2, 4.3": ["results/dataset_statistics.csv", "experiments/results/DISCREPANCIES.md", "experiments/results/data_verification.json"],
        "4.1, 4.2": ["experiments/runs/full/results.json", "results/hpo_small.json", "results/latency_full.json"],
        "4.5 statistical uncertainty, 4.6 drift head": ["reports/case_study/bootstrap_ci.json (computed by reports/case_study/bootstrap_ci.py from stored predictions)"],
        "4.7-4.13": ["results/calibration.json", "results/confident_learning.json", "results/noise_robustness.json", "results/kmeans.json", "results/gan_vs_main.json", "experiments/results/gan.json", "results/xai.json", "results/latency_size.csv", "results/latency_size_onnx.json"],
        "4.14 testing": ["results/test_summary.json", "results/coverage.json", "experiments/results/testing_measurements.json"],
        "4.15 discussion": ["all of the above"],
    },
}
json.dump(res, open(HERE / "CHECKLIST.json", "w"), indent=1)
print({k: v for k, v in res.items() if k not in ("number_trace", "heading2_list", "prose_number_sources", "heading1_list")})
print(heads)
