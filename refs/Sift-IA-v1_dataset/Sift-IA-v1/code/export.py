#!/usr/bin/env python3
"""Stage 4 - write the IEEE DataPort release package."""
import json, csv, os, shutil, subprocess, statistics
from collections import Counter, defaultdict

B = json.load(open("/home/claude/sift_ds/built.json")); rows, sessions = B["rows"], B["sessions"]
BL = json.load(open("/home/claude/sift_ds/baseline_results.json"))
R = "/home/claude/sift_ds/release/Sift-IA-v1"
shutil.rmtree(R, ignore_errors=True)
for d in ["data", "licenses", "code"]: os.makedirs(f"{R}/{d}")

COLS = ["sample_id", "split", "session_id", "position_in_diff", "n_hunks_in_diff",
        "repo", "repo_slug", "language", "license", "seed_commit_sha", "source_commit_sha", "source_commit_date",
        "file_path", "file_status", "hunk_header", "hunk",
        "intent_turns", "consolidated_intent",
        "label", "drift_type", "label_clean", "drift_type_clean", "mutation_difficulty", "provenance",
        "noise_types", "label_noise", "intent_noise_type", "non_semantic_reason",
        "approx_tokens", "exceeds_max_len", "is_duplicate_hunk", "hunk_sha1", "gold_candidate"]

def flat(r):
    o = {k: r[k] for k in COLS}
    o["intent_turns"] = " || ".join(r["intent_turns"])
    for k in ("label_noise", "exceeds_max_len", "is_duplicate_hunk", "gold_candidate"): o[k] = int(bool(r[k]))
    return o

for split in ["train", "val", "test"]:
    with open(f"{R}/data/sift_ia_v1_{split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, quoting=csv.QUOTE_NONNUMERIC); w.writeheader()
        for r in rows:
            if r["split"] == split: w.writerow(flat(r))
with open(f"{R}/data/sift_ia_v1_all.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps({k: r[k] for k in COLS}, ensure_ascii=False) + "\n")

with open(f"{R}/data/sessions.csv", "w", newline="") as f:
    keys = ["session_id", "split", "repo", "seed_commit_sha", "n_hunks", "n_misaligned_clean", "mutations", "intent_noise_type"]
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for s in sessions: w.writerow({k: s[k] for k in keys})

with open(f"{R}/data/noise_manifest.csv", "w", newline="") as f:
    keys = ["sample_id", "split", "noise_types", "label", "label_clean", "drift_type", "drift_type_clean",
            "label_noise", "intent_noise_type", "non_semantic_reason", "exceeds_max_len", "is_duplicate_hunk"]
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for r in rows:
        if r["noise_types"] != "none":
            o = {k: r[k] for k in keys}
            for k in ("label_noise", "exceeds_max_len", "is_duplicate_hunk"): o[k] = int(bool(o[k]))
            w.writerow(o)

# blind annotation sheet (labels deliberately withheld)
with open(f"{R}/data/gold_annotation_sheet.csv", "w", newline="", encoding="utf-8") as f:
    keys = ["sample_id", "session_id", "repo", "intent_turns", "consolidated_intent", "file_path", "hunk",
            "annotator_1_label", "annotator_1_drift_type", "annotator_2_label", "annotator_2_drift_type",
            "adjudicated_label", "adjudicated_drift_type", "notes"]
    w = csv.DictWriter(f, fieldnames=keys, quoting=csv.QUOTE_NONNUMERIC); w.writeheader()
    for r in sorted([r for r in rows if r["gold_candidate"]], key=lambda x: x["sample_id"]):
        w.writerow({"sample_id": r["sample_id"], "session_id": r["session_id"], "repo": r["repo"],
                    "intent_turns": " || ".join(r["intent_turns"]), "consolidated_intent": r["consolidated_intent"],
                    "file_path": r["file_path"], "hunk": r["hunk"],
                    **{k: "" for k in keys[7:]}})

# source repositories + licenses
REPOS = {}
exec(open("/home/claude/sift_ds/mine.py").read().split("CODE_EXT")[0])
src = defaultdict(lambda: {"n": 0, "commits": set(), "dates": []})
for r in rows:
    s = src[r["repo"]]; s["n"] += 1; s["commits"].add(r["source_commit_sha"]); s["dates"].append(r["source_commit_date"][:10])
    s["split"] = r["split"]
with open(f"{R}/data/source_repositories.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["repo", "github", "language", "license", "split", "samples", "distinct_commits", "earliest_commit", "latest_commit"])
    for name, (slug, lang, lic) in REPOS.items():
        s = src[name]
        w.writerow([name, f"https://github.com/{slug}", lang, lic, s["split"], s["n"], len(s["commits"]), min(s["dates"]), max(s["dates"])])
        gd = f"/home/claude/repos/{name}.git"
        lf = subprocess.run(["git", "--git-dir", gd, "ls-tree", "--name-only", "HEAD"], capture_output=True, text=True).stdout.split()
        lf = [x for x in lf if x.lower().startswith(("licen", "copying"))][0]
        txt = subprocess.run(["git", "--git-dir", gd, "show", f"HEAD:{lf}"], capture_output=True, text=True).stdout
        open(f"{R}/licenses/{name}_{lf}".replace(".md", ".txt") if not lf.endswith(".txt") else f"{R}/licenses/{name}_{lf}", "w").write(txt)

for f in ["mine.py", "generate.py", "baselines.py", "export.py"]:
    shutil.copy(f"/home/claude/sift_ds/{f}", f"{R}/code/{f}")
json.dump(BL, open(f"{R}/data/baseline_results.json", "w"), indent=2)

# ---------------- statistics
st = {}
for sp in ["train", "val", "test"]:
    R_ = [r for r in rows if r["split"] == sp]; S_ = [s for s in sessions if s["split"] == sp]
    st[sp] = {"hunks": len(R_), "sessions": len(S_),
              "aligned_clean": sum(r["label_clean"] == "aligned" for r in R_),
              "misaligned_clean": sum(r["label_clean"] == "misaligned" for r in R_),
              "drift": dict(Counter(r["drift_type_clean"] for r in R_ if r["label_clean"] == "misaligned")),
              "difficulty": dict(Counter(r["mutation_difficulty"] for r in R_ if r["label_clean"] == "misaligned")),
              "label_flips": sum(r["label_noise"] for r in R_),
              "languages": dict(Counter(r["language"] for r in R_)),
              "repos": sorted({r["repo"] for r in R_})}
noise = Counter()
for r in rows:
    for t in r["noise_types"].split("|"):
        if t != "none": noise[t] += 1
st["noise_rows"] = dict(noise); st["rows_with_any_noise"] = sum(r["noise_types"] != "none" for r in rows)
st["non_semantic_reasons"] = dict(Counter(r["non_semantic_reason"] for r in rows if r["non_semantic_reason"] != "none"))
st["intent_noise_sessions"] = dict(Counter(s["intent_noise_type"] for s in sessions))
st["label_flip_directions"] = {f"{a}->{b}": c for (a, b), c in Counter((r["label_clean"], r["label"]) for r in rows if r["label_noise"]).items()}
tok = [r["approx_tokens"] for r in rows]
st["approx_tokens"] = {"median": statistics.median(tok), "p95": sorted(tok)[int(.95 * len(tok))], "max": max(tok)}
st["hunks_per_session"] = {"median": statistics.median(s["n_hunks"] for s in sessions), "max": max(s["n_hunks"] for s in sessions)}
st["commit_date_range"] = [min(r["source_commit_date"][:10] for r in rows), max(r["source_commit_date"][:10] for r in rows)]
st["distinct_source_commits"] = len({r["source_commit_sha"] for r in rows})
json.dump(st, open(f"{R}/data/dataset_statistics.json", "w"), indent=2)
json.dump(st, open("/home/claude/sift_ds/stats.json", "w"), indent=2)
print(json.dumps(st, indent=1)[:2500])
