#!/usr/bin/env python3
"""Stage 1 - mine non-merge commits from local bare clones into hunk records.

Output: mined_hunks.jsonl  (one record per hunk, grouped by commit)
"""
import json, re, subprocess, sys, os, hashlib

REPOS = {  # name -> (github slug, language, license)
    "requests": ("psf/requests", "Python", "Apache-2.0"),
    "flask": ("pallets/flask", "Python", "BSD-3-Clause"),
    "click": ("pallets/click", "Python", "BSD-3-Clause"),
    "httpx": ("encode/httpx", "Python", "BSD-3-Clause"),
    "rich": ("Textualize/rich", "Python", "MIT"),
    "typer": ("fastapi/typer", "Python", "MIT"),
    "attrs": ("python-attrs/attrs", "Python", "MIT"),
    "black": ("psf/black", "Python", "MIT"),
    "sqlmodel": ("tiangolo/sqlmodel", "Python", "MIT"),
    "express": ("expressjs/express", "JavaScript", "MIT"),
    "axios": ("axios/axios", "JavaScript", "MIT"),
    "got": ("sindresorhus/got", "TypeScript", "MIT"),
    "chalk": ("chalk/chalk", "JavaScript", "MIT"),
    "fastify": ("fastify/fastify", "JavaScript", "MIT"),
    "koa": ("koajs/koa", "JavaScript", "MIT"),
}
CODE_EXT = (".py", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".pyi")
CONFIG_EXT = (".json", ".toml", ".yml", ".yaml", ".cfg", ".ini")
LOCKFILES = ("package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "pdm.lock", "uv.lock")
SEP = "\x1e@@COMMIT@@"

def keep_path(p):
    b = os.path.basename(p)
    return p.endswith(CODE_EXT) or p.endswith(CONFIG_EXT) or b in LOCKFILES

def parse_diff(diff_text):
    files, cur = [], None
    for line in diff_text.split("\n"):
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            cur = {"path": m.group(2) if m else None, "hunks": [], "binary": False, "new": False, "deleted": False}
            files.append(cur)
        elif cur is None:
            continue
        elif line.startswith("Binary files"):
            cur["binary"] = True
        elif line.startswith("new file mode"):
            cur["new"] = True
        elif line.startswith("deleted file mode"):
            cur["deleted"] = True
        elif line.startswith("@@"):
            cur["hunks"].append({"header": line, "lines": []})
        elif cur["hunks"] and (line[:1] in (" ", "+", "-") or line == ""):
            if line.startswith("+++") or line.startswith("---"):
                continue
            cur["hunks"][-1]["lines"].append(line)
    return files

def mine(name, max_commits=None):
    slug, lang, lic = REPOS[name]
    gd = f"/home/claude/repos/{name}.git"
    fmt = SEP + "%n%H%n%aI%n%s%n%b%n@@ENDMSG@@"
    cmd = ["git", "--git-dir", gd, "log", "--no-merges", "-M", "--unified=3",
           "--no-color", f"--format={fmt}", "-p", "HEAD"]
    out = subprocess.run(cmd, capture_output=True, text=True, errors="replace").stdout
    records, n_commits = [], 0
    for block in out.split(SEP)[1:]:
        head, _, diff = block.partition("@@ENDMSG@@")
        parts = head.strip("\n").split("\n")
        if len(parts) < 3:
            continue
        sha, date, subject = parts[0], parts[1], parts[2].strip()
        body = "\n".join(parts[3:]).strip()
        files = [f for f in parse_diff(diff) if f["path"] and not f["binary"] and keep_path(f["path"])]
        if not files or len(files) > 8:
            continue
        hunks = []
        for f in files:
            for h in f["hunks"]:
                body_lines = [l for l in h["lines"] if l != ""]
                if not any(l[:1] in "+-" for l in body_lines):
                    continue
                hunks.append({"file_path": f["path"], "hunk_header": h["header"],
                              "hunk": "\n".join([h["header"]] + body_lines),
                              "file_status": "added" if f["new"] else ("deleted" if f["deleted"] else "modified")})
        if not hunks or len(hunks) > 12:
            continue
        n_commits += 1
        for i, h in enumerate(hunks):
            records.append({"repo": name, "repo_slug": slug, "language": lang, "license": lic,
                            "commit_sha": sha, "commit_date": date, "subject": subject, "body": body,
                            "hunk_index": i, "n_hunks_in_commit": len(hunks), **h})
        if max_commits and n_commits >= max_commits:
            break
    return records, n_commits

if __name__ == "__main__":
    total = 0
    with open("/home/claude/sift_ds/mined_hunks.jsonl", "w") as fo:
        for name in REPOS:
            recs, nc = mine(name)
            for r in recs:
                fo.write(json.dumps(r) + "\n")
            total += len(recs)
            print(f"{name:10} commits={nc:5}  hunks={len(recs):6}", flush=True)
    print("TOTAL hunks:", total)
