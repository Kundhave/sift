#!/usr/bin/env python3
"""Stage 2 - build the Sift Intent-Alignment dataset (v1.0) from mined hunks.

Deterministic: every random choice flows from SEED.
Labels are construction-determined (the mutation applied defines the label).
Noise is injected / detected with ground-truth flags so denoising can be evaluated.
"""
import json, random, re, hashlib, math, csv, os
from collections import defaultdict, Counter

SEED = 473
rng = random.Random(SEED)
OUT = "/home/claude/sift_ds/release"
os.makedirs(OUT, exist_ok=True)

SPLIT_REPOS = {
    "train": ["requests", "flask", "click", "rich", "black", "express", "fastify"],
    "val":   ["typer", "sqlmodel", "got", "koa", "chalk"],
    "test":  ["httpx", "attrs", "axios"],
}
# exact targets (clean labels) - match the Review-2 specification
TARGET = {  # aligned, collateral, scope_expansion, purpose_drift
    "train": (9828, 1248, 915, 609),
    "val":   (2106, 267, 196, 131),
    "test":  (2106, 267, 196, 131),
}
LABEL_FLIP = {"aligned": 0.03, "misaligned": 0.10}   # train split only
INTENT_NOISE = [("chit_chat", 0.07), ("retraction", 0.05), ("repeated_turn", 0.03)]
MAX_TOKENS = 1024
DROP_TOKENS = 4000
LOCKFILES = ("package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "pdm.lock", "uv.lock")
CODE_EXT = (".py", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".pyi")
STOP = set("a an the to of in on for and or with from by is are be as at into this that it its fix fixes fixed "
           "add adds added update updates updated use uses remove removes removed make makes change changes "
           "support when not no new more some".split())
CHIT_CHAT = ["thanks, that looks good so far", "ok cool", "hmm, let me think about it for a second",
             "nice, go ahead", "sounds good", "great, keep going", "can you run it again?",
             "sorry, I was away for a minute"]

# ---------------------------------------------------------------- helpers
def approx_tokens(t): return math.ceil(len(t) / 3.6)          # BPE estimate for mixed code/text

def changed(h):
    lines = h.split("\n")[1:]
    return [l[1:] for l in lines if l[:1] == "+"], [l[1:] for l in lines if l[:1] == "-"]

def non_semantic_reason(r):
    p, h = r["file_path"], r["hunk"]
    base = os.path.basename(p)
    if base in LOCKFILES: return "lockfile"
    if any(seg in p for seg in ("/dist/", "dist/", "/build/", ".min.js", "_generated", ".pb.")): return "generated_or_minified"
    add, rem = changed(h)
    if any(len(l) > 500 for l in add): return "generated_or_minified"
    strip = lambda ls: sorted(re.sub(r"\s+", "", l) for l in ls if l.strip())
    if strip(add) == strip(rem): return "whitespace_only"
    imp = re.compile(r"^\s*(import |from \S+ import |const .* = require\(|import\{|export \* from)")
    ch = [l for l in add + rem if l.strip()]
    if ch and all(imp.match(l) for l in ch) and sorted(map(str.strip, add)) == sorted(map(str.strip, rem)):
        return "import_reorder_only"
    ver = re.compile(r'^\s*("version"\s*:|__version__\s*=|version\s*=)')
    if ch and all(ver.match(l) for l in ch): return "version_bump_only"
    return ""

def clean_subject(s):
    s = re.sub(r"\s*\(#\d+\)\s*$", "", s).strip()
    s = re.sub(r"^\[[^\]]{1,20}\]\s*", "", s)
    return s

def body_refinement(b):
    for line in b.split("\n"):
        line = line.strip(" -*\t")
        if not line or re.match(r"(Signed-off-by|Co-authored-by|Reviewed-by|Closes|Fixes|Refs?|See|http|```)", line, re.I):
            continue
        sent = re.split(r"(?<=[.!?])\s", line)[0]
        if 4 <= len(sent.split()) <= 35:
            return sent
        return ""
    return ""

def words(s): return {w for w in re.findall(r"[a-z][a-z0-9_]+", s.lower()) if w not in STOP and len(w) > 2}
def jacc(a, b): return len(a & b) / max(1, len(a | b))
def ident_tokens(s):
    out = set()
    for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", s):
        for p in re.sub(r"([a-z])([A-Z])", r"\1 \2", t).replace("_", " ").split():
            p = p.lower()
            if len(p) > 2 and p not in STOP: out.add(p)
    return out
HARD_P = 0.65   # share of misaligned hunks mined as hard negatives

BAD_SUBJ = re.compile(r"^(merge|release|revert|wip\b|v?\d+\.\d+|bump version|prepare|changelog|update changelog|"
                      r"version\b|\d+\.\d+\.\d+)", re.I)

# ---------------------------------------------------------------- load
commits = defaultdict(list)
with open("/home/claude/sift_ds/mined_hunks.jsonl") as f:
    for line in f:
        r = json.loads(line)
        commits[(r["repo"], r["commit_sha"])].append(r)

def usable_commit(hs):
    s = clean_subject(hs[0]["subject"])
    if BAD_SUBJ.match(s) or len(s.split()) < 3 or len(s) > 120: return False
    if any(approx_tokens(h["hunk"]) > DROP_TOKENS for h in hs): return False
    return True

by_repo = defaultdict(list)
for (repo, sha), hs in commits.items():
    if usable_commit(hs):
        by_repo[repo].append(sorted(hs, key=lambda x: x["hunk_index"]))
for repo in by_repo:
    by_repo[repo].sort(key=lambda hs: hs[0]["commit_sha"])   # stable before shuffle

def semantic_donor(h):
    return h["file_path"].endswith(CODE_EXT) and not non_semantic_reason(h) and approx_tokens(h["hunk"]) <= MAX_TOKENS

# ---------------------------------------------------------------- build per split
rows, sessions, counters = [], [], defaultdict(Counter)
hunk_seen = {}
sid_n = 0

def make_row(h, split, session_id, label, drift, provenance, donor=None):
    return {"split": split, "session_id": session_id, "repo": h["repo"], "repo_slug": h["repo_slug"],
            "language": h["language"], "license": h["license"], "source_commit_sha": h["commit_sha"],
            "source_commit_date": h["commit_date"], "file_path": h["file_path"], "file_status": h["file_status"],
            "hunk_header": h["hunk_header"], "hunk": h["hunk"],
            "label_clean": label, "drift_type_clean": drift, "provenance": provenance, "mutation_difficulty": "n/a"}

for split in ["train", "val", "test"]:
    A, C, S, P = TARGET[split]
    rem = {"aligned": A, "collateral": C, "scope_expansion": S, "purpose_drift": P}
    pool = [hs for repo in SPLIT_REPOS[split] for hs in by_repo[repo]]
    rng.shuffle(pool)
    n_donor = int(len(pool) * (0.40 if split == "val" else 0.25))
    donors, seeds = pool[:n_donor], pool[n_donor:]
    donor_hunks = [h for hs in donors for h in hs if semantic_donor(h)]
    by_file = defaultdict(list)
    for h in donor_hunks: by_file[(h["repo"], h["file_path"])].append(h)
    by_repo_d = defaultdict(list)
    for h in donor_hunks: by_repo_d[h["repo"]].append(h)
    decoys = defaultdict(list)
    for hs in donors: decoys[hs[0]["repo"]].append(clean_subject(hs[0]["subject"]))
    used_donor = set(); used_seed = set(); dtok = {}; intent_tok = set()

    def pick_donor(kind, session_hunks, intent_words, repo):
        files = {h["file_path"] for h in session_hunks}
        if kind in ("scope_expansion", "purpose_drift"):
            cands = [d for f in files for d in by_file.get((repo, f), [])]
        else:
            dirs = {os.path.dirname(f) for f in files}
            cands = [d for d in by_repo_d[repo] if d["file_path"] not in files]
            diffdir = [d for d in cands if os.path.dirname(d["file_path"]) not in dirs]
            cands = diffdir or cands
        rng.shuffle(cands)
        bodies = {h["hunk"] for h in session_hunks}
        ok = []
        for d in cands[:600]:
            key = (d["commit_sha"], d["hunk_index"])
            if key in used_donor or d["hunk"] in bodies: continue
            if jacc(intent_words, words(clean_subject(d["subject"]))) >= 0.2: continue
            ok.append(d)
        if not ok: return None, None
        hard = rng.random() < HARD_P
        if hard:
            it = intent_tok
            def score(d):
                t = dtok.get(id(d))
                if t is None:
                    t = ident_tokens("\n".join(l[1:] for l in d["hunk"].split("\n")[1:] if l[:1] in "+-") + " " + d["file_path"])
                    dtok[id(d)] = t
                return len(it & t) / math.sqrt(max(1, len(it)) * max(1, len(t)))   # size-normalised
            nch = lambda h: sum(1 for l in h["hunk"].split("\n")[1:] if l[:1] in "+-")
            sizes = sorted(nch(h) for h in session_hunks)
            med = sizes[len(sizes) // 2] if sizes else 6
            pool_ = [d for d in ok if 0.5 * med <= nch(d) <= 2.0 * med + 2] or ok     # size-matched
            best = max(pool_, key=score)
            if score(best) == 0: hard = False; best = ok[0]
        else:
            best = ok[0]
        used_donor.add((best["commit_sha"], best["hunk_index"]))
        return best, ("hard" if hard else "random")

    def build_session(hs, plan):
        """plan: list of drift kinds to inject. Returns (hunk_rows, counts) or None."""
        repo = hs[0]["repo"]
        subj = clean_subject(hs[0]["subject"]); ref = body_refinement(hs[0]["body"])
        iw = words(subj + " " + ref)
        intent_tok.clear(); intent_tok.update(ident_tokens(subj + " " + ref))
        base = list(hs); out = []
        for kind in plan:
            d, diff_ = pick_donor(kind, base, iw, repo)
            if d is None: return None
            if kind == "purpose_drift":
                same = [h for h in base if h["file_path"] == d["file_path"]]
                if not same: return None
                base.remove(rng.choice(same))
            out.append((d, kind, diff_))
        return subj, ref, base, out

    def accept(hs, plan, final=False):
        global sid_n
        built = build_session(hs, plan)
        if not built: return False
        subj, ref, base, muts = built
        cnt = Counter({"aligned": len(base)}); cnt.update(k for _, k, _ in muts)
        if any(cnt[k] > rem[k] for k in cnt): return False
        if not final:
            left_a = rem["aligned"] - cnt["aligned"]
            left_cs = (rem["collateral"] - cnt["collateral"]) + (rem["scope_expansion"] - cnt["scope_expansion"]) \
                      + (rem["purpose_drift"] - cnt["purpose_drift"])
            if left_a < left_cs + 60: return False
        for k in cnt: rem[k] -= cnt[k]
        used_seed.add(hs[0]["commit_sha"])
        sid_n += 1
        sid = f"S{sid_n:05d}"
        turns = [subj] + ([ref] if ref else [])
        consolidated = subj + (f". {ref}" if ref else "")
        # ---- intent-turn noise (all splits: it is an input characteristic)
        u = rng.random(); acc = 0; noise = ""
        for name, p in INTENT_NOISE:
            acc += p
            if u < acc: noise = name; break
        if noise == "chit_chat":
            turns.insert(rng.randint(1, len(turns)), rng.choice(CHIT_CHAT))
        elif noise == "retraction":
            decoy = rng.choice(decoys[hs[0]["repo"]])
            turns = [decoy, "Actually, scratch that. " + subj] + turns[1:]
        elif noise == "repeated_turn":
            turns.append("same as before: " + subj)
        hunk_rows = [make_row(h, split, sid, "aligned", "none", f"commit:{h['commit_sha'][:12]}") for h in base]
        for d, kind, diff_ in muts:
            rr = make_row(d, split, sid, "misaligned", kind, f"mutation:{kind}:donor_commit={d['commit_sha'][:12]}")
            rr["mutation_difficulty"] = diff_
            hunk_rows.append(rr)
        rng.shuffle(hunk_rows)
        for i, r in enumerate(hunk_rows):
            r.update({"intent_turns": turns, "consolidated_intent": consolidated, "intent_noise_type": noise or "none",
                      "position_in_diff": i, "n_hunks_in_diff": len(hunk_rows),
                      "seed_commit_sha": hs[0]["commit_sha"]})
            rows.append(r)
        sessions.append({"session_id": sid, "split": split, "repo": hs[0]["repo"], "seed_commit_sha": hs[0]["commit_sha"],
                         "n_hunks": len(hunk_rows), "n_misaligned_clean": sum(r["label_clean"] == "misaligned" for r in hunk_rows),
                         "mutations": "|".join(k for _, k, _ in muts) or "none", "intent_noise_type": noise or "none"})
        return True

    singles = [hs for hs in seeds if len(hs) == 1]
    multi = [hs for hs in seeds if len(hs) > 1]
    reserve_singles = singles[: len(singles) // 2]
    phase1 = multi + singles[len(singles) // 2:]
    rng.shuffle(phase1)
    for hs in phase1:
        need = rem["collateral"] + rem["scope_expansion"] + rem["purpose_drift"]
        if rem["aligned"] <= 60 + rem["collateral"] + rem["scope_expansion"] + rem["purpose_drift"]: break
        plan = []
        if need and rng.random() < 0.52:
            k = 1 if rng.random() < 0.72 else 2
            for _ in range(k):
                kinds = [x for x in ("collateral", "scope_expansion", "purpose_drift") if rem[x] - plan.count(x) > 0]
                if not kinds: break
                w = [rem[x] - plan.count(x) for x in kinds]
                plan.append(rng.choices(kinds, weights=w)[0])
        accept(hs, plan)
    # ---- end game with single-hunk seeds: exact totals
    def unused(): return sorted([hs for hs in seeds if hs[0]["commit_sha"] not in used_seed], key=len)
    for kind, plan in [("purpose_drift", ["purpose_drift"]), ("collateral", ["collateral"]),
                       ("scope_expansion", ["scope_expansion"])]:
        for hs in unused():
            if rem[kind] <= 0: break
            accept(hs, plan, final=True)
    for hs in unused():
        if rem["aligned"] <= 0: break
        accept(hs, [], final=True)
    counters[split] = Counter({k: TARGET[split][i] - rem[k] for i, k in
                               enumerate(["aligned", "collateral", "scope_expansion", "purpose_drift"])})
    print(split, dict(counters[split]), "remaining:", {k: v for k, v in rem.items() if v})

# ---------------------------------------------------------------- per-hunk noise flags
seen = {}
for i, r in enumerate(rows):
    r["sample_id"] = f"SIFT-{i+1:06d}"
    r["approx_tokens"] = approx_tokens(r["consolidated_intent"] + r["hunk"] + r["file_path"])
    r["exceeds_max_len"] = r["approx_tokens"] > MAX_TOKENS
    r["non_semantic_reason"] = non_semantic_reason(r) or "none"
    norm = "\n".join(l for l in r["hunk"].split("\n")[1:] if l[:1] in "+-")
    hsh = hashlib.sha1(norm.encode()).hexdigest()
    r["hunk_sha1"] = hsh
    r["is_duplicate_hunk"] = hsh in seen
    seen.setdefault(hsh, r["sample_id"])
    # ---- label noise: train only, class-conditional, ground truth retained
    r["label"], r["drift_type"], r["label_noise"] = r["label_clean"], r["drift_type_clean"], False
    if r["split"] == "train":
        if r["label_clean"] == "aligned" and rng.random() < LABEL_FLIP["aligned"]:
            r["label"] = "misaligned"
            r["drift_type"] = rng.choices(["collateral", "scope_expansion", "purpose_drift"], weights=[45, 33, 22])[0]
            r["label_noise"] = True
        elif r["label_clean"] == "misaligned" and rng.random() < LABEL_FLIP["misaligned"]:
            r["label"], r["drift_type"], r["label_noise"] = "aligned", "none", True
    types = []
    if r["label_noise"]: types.append("label_flip")
    if r["non_semantic_reason"] != "none": types.append("non_semantic_hunk")
    if r["intent_noise_type"] != "none": types.append("intent_turn")
    if r["exceeds_max_len"]: types.append("truncation")
    if r["is_duplicate_hunk"]: types.append("duplicate_hunk")
    r["noise_types"] = "|".join(types) or "none"

# gold-annotation candidates: stratified 900 from test
test = [r for r in rows if r["split"] == "test"]
strata = defaultdict(list)
for r in test: strata[r["drift_type_clean"]].append(r)
gold = []
for k, lst in strata.items():
    n = round(900 * len(lst) / len(test)); rng.shuffle(lst); gold += lst[:n]
gold = gold[:900]
gold_ids = {r["sample_id"] for r in gold}
for r in rows: r["gold_candidate"] = r["sample_id"] in gold_ids

json.dump({"rows": rows, "sessions": sessions}, open("/home/claude/sift_ds/built.json", "w"))
print("rows:", len(rows), "sessions:", len(sessions), "gold candidates:", len(gold_ids))
