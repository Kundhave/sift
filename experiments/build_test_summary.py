"""Build results/test_summary.json from pytest junit XML (unit + e2e) and results/coverage.json.
Usage: build_test_summary.py <junit_tests.xml> <junit_e2e.xml>"""
import json, sys, re, datetime, platform
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATS = {  # file -> (category, framework section)
    "test_smoke": ("smoke", "Performance / readiness"),
    "test_whitebox": ("white-box", "AI Models"),
    "test_diffparse_consolidate": ("white-box", "AI Models"),
    "test_blackbox": ("black-box", "Testing Methods"),
    "test_api": ("api", "Testing Methods"),
    "test_api_partitions": ("api", "Testing Methods"),
    "test_data_pipeline": ("data-pipeline", "Data"),
    "test_metamorphic": ("metamorphic", "Testing Methods"),
    "test_redteam": ("adversarial", "Testing Methods"),
    "test_backtoback": ("back-to-back", "AI Models"),
    "test_ui_e2e": ("e2e", "Context"),
}
# Severity / how-found / mitigation are authored; the existence of each defect is proven by its strict xfail
# test (reason string carries the measured numbers).
DEFECT_META = {
    "D1": ("Medium", "Black-box test on plain `diff -u` multi-file input (no `diff --git` lines)",
           "Open: only git-style diffs split files correctly. Mitigation: UI/docs state git unified diff is required."),
    "D2": ("Low", "Black-box error guessing on paraphrased pleasantries in intent turns",
           "Open: consolidation is rule-based (8 templates). Mitigation: model still reads the remaining intent; learned filter is future work."),
    "D3": ("High", "Red-team: comment line copying the intent text added to a misaligned hunk",
           "Open. Model partly keys on lexical overlap with the intent. Mitigation: strip comment-only added lines before scoring (not implemented) or adversarial training."),
    "D4": ("Medium", "Red-team: context comment claiming 'this change implements: <intent>'",
           "Open. Same root cause as D3. Mitigation: ignore comments claiming intent; flag hunks whose only intent-overlap is in comments."),
    "D-DATA-1": ("Low", "Data pipeline test vs documented per-language hunk counts",
                 "Open: documentation discrepancy; reports must quote the released counts, not the documented ones."),
    "D-DATA-2": ("Medium", "Data pipeline test of language coverage per split",
                 "Open: test split has no TypeScript, so TS generalisation is unmeasured; stated as a limitation."),
}

def parse(xml):
    rows = []
    for tc in ET.parse(xml).getroot().iter("testcase"):
        f = tc.get("classname", "").split(".")
        mod = next((p for p in f if p.startswith("test_")), "")
        st, msg = "passed", ""
        for k in tc:
            if k.tag in ("failure", "error"): st, msg = "failed", (k.get("message") or "")[:200]
            elif k.tag == "skipped":
                typ = k.get("type", "")
                st, msg = ("xfailed" if "xfail" in typ or (k.get("message") or "").startswith("xfail") else "skipped"), (k.get("message") or "")[:300]
        rows.append((mod, tc.get("name"), st, msg))
    return rows

rows = parse(sys.argv[1]) + parse(sys.argv[2])
cats = defaultdict(lambda: dict(passed=0, xfailed=0, failed=0, skipped=0, total=0, files=set(), framework_section=""))
defects, seen = [], set()
for mod, name, st, msg in rows:
    c, sec = CATS.get(mod, ("other", "Documentation"))
    d = cats[c]; d[st] += 1; d["total"] += 1; d["files"].add(mod + ".py"); d["framework_section"] = sec
    if st == "xfailed":
        m = re.search(r"\b(D-DATA-\d|D\d)\b", msg)
        if m and m.group(1) not in seen:
            seen.add(m.group(1)); i = m.group(1)
            sev, how, mit = DEFECT_META.get(i, ("Unrated", "pytest", "see DECISIONS.md"))
            defects.append(dict(id=i, description=re.sub(r"^xfail:\s*", "", msg).split(": ", 1)[-1] if ": " in msg else msg,
                                severity=sev, how_found=how, mitigation=mit, test=f"{mod}.py::{name}", status="open (xfail strict)"))
for d in cats.values(): d["files"] = sorted(d["files"])
tot = {k: sum(c[k] for c in cats.values()) for k in ("total", "passed", "xfailed", "failed", "skipped")}
cov = json.load(open(ROOT / "results" / "coverage.json"))["totals"]
out = dict(generated=datetime.datetime.now().isoformat(timespec="seconds"), machine=platform.processor() or platform.machine(),
           device="cpu", totals=tot, categories={k: v for k, v in sorted(cats.items())},
           coverage=dict(percent_covered=round(cov["percent_covered"], 2), covered_lines=cov["covered_lines"],
                         num_statements=cov["num_statements"], scope="src/sift + app/backend (unit/API tests; e2e not measured)"),
           defects=sorted(defects, key=lambda x: x["id"]))
json.dump(out, open(ROOT / "results" / "test_summary.json", "w"), indent=2)
print(json.dumps(dict(totals=tot, cov=out["coverage"]["percent_covered"], defects=[d["id"] for d in defects])))
