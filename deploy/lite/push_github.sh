#!/usr/bin/env bash
# Create the public GitHub repo "sift" and push the assembled folder. Needs GITHUB_USER and GITHUB_TOKEN (repo scope) in the
# environment. The token is passed only as an in-memory http header for this one git command: never written to git config or a file.
set -euo pipefail
: "${GITHUB_USER:?set GITHUB_USER}" "${GITHUB_TOKEN:?set GITHUB_TOKEN}"
DIR="${1:-/home/kundhave/Projects/sift-render}"
REPO="${REPO_NAME:-sift}"
code=$(curl -sS -o /tmp/gh_create.$$ -w '%{http_code}' -X POST https://api.github.com/user/repos \
  -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" \
  -d "{\"name\":\"$REPO\",\"private\":false,\"description\":\"Sift: hunk-level intent-alignment check for AI coding agent diffs\"}")
if [ "$code" != 201 ] && [ "$code" != 422 ]; then cat /tmp/gh_create.$$ >&2; rm -f /tmp/gh_create.$$; echo "GitHub API returned $code" >&2; exit 1; fi
rm -f /tmp/gh_create.$$   # 422 = repo already exists: continue and push
cd "$DIR"
git add -A
git -c user.name="${GIT_AUTHOR_NAME:-$GITHUB_USER}" -c user.email="${GIT_AUTHOR_EMAIL:-$GITHUB_USER@users.noreply.github.com}" \
  commit -q -m "Sift lite deployment (Render)" || true
b64=$(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 -w0)
git -c "http.extraheader=Authorization: Basic $b64" push -u "https://github.com/$GITHUB_USER/$REPO.git" HEAD:main
echo "pushed: https://github.com/$GITHUB_USER/$REPO"
