#!/usr/bin/env bash
set -euo pipefail

# Safe helper to create a GitHub repo (if needed) and push the current repo.
# Important: This script DOES NOT store your token. You must export GITHUB_TOKEN
# in your shell before running.
# Usage: 
#   export GITHUB_TOKEN="ghp_xxx" 
#   bash scripts/git_push_repo.sh itz-sayak/LOFT public main

REPO="${1:-itz-sayak/LOFT}"
VISIBILITY="${2:-public}"
BRANCH="${3:-main}"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Error: GITHUB_TOKEN environment variable is not set."
  echo "Export it and re-run. Example:"
  echo "  export GITHUB_TOKEN='ghp_xxx'"
  exit 1
fi

command -v git >/dev/null 2>&1 || { echo "git not found in PATH"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl not found in PATH"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found in PATH (used to parse API responses)."; exit 1; }

# require a clean working tree to avoid accidental partial pushes
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is not clean. Please commit or stash changes before running." >&2
  git status --porcelain
  exit 1
fi

REMOTE_URL="https://github.com/${REPO}.git"

# Check repo existence using GitHub API
echo "Checking repository ${REPO} on GitHub..."
status=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: token ${GITHUB_TOKEN}" "https://api.github.com/repos/${REPO}")

if [[ "$status" == "200" ]]; then
  echo "Repository ${REPO} already exists."
elif [[ "$status" == "404" ]]; then
  echo "Repository not found. Attempting to create under the authenticated user..."
  OWNER=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" "https://api.github.com/user" | python3 -c 'import sys, json; print(json.load(sys.stdin)["login"])')
  TARGET_OWNER=$(echo "$REPO" | cut -d'/' -f1)
  TARGET_NAME=$(echo "$REPO" | cut -d'/' -f2)

  if [[ "$OWNER" != "$TARGET_OWNER" ]]; then
    echo "Authenticated user ($OWNER) does not match target owner ($TARGET_OWNER)." >&2
    echo "You must have permission to create the repo under $TARGET_OWNER (or use your own account)." >&2
    exit 1
  fi

  # Build payload without requiring jq
  if [[ "$VISIBILITY" == "private" ]]; then
    priv=true
  else
    priv=false
  fi
  payload=$(printf '{"name":"%s","private":%s}' "$TARGET_NAME" "$priv")

  created=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" -d "$payload" "https://api.github.com/user/repos")
  if echo "$created" | grep -q '"full_name"'; then
    echo "Repository created successfully."
  else
    echo "Failed to create repository. API response:" >&2
    echo "$created" >&2
    exit 1
  fi
else
  echo "Unexpected response from GitHub API: HTTP $status" >&2
  exit 1
fi

# Ensure remote origin points to the expected HTTPS URL
if git remote get-url origin >/dev/null 2>&1; then
  echo "Setting origin to ${REMOTE_URL}"
  git remote set-url origin "$REMOTE_URL"
else
  echo "Adding origin ${REMOTE_URL}"
  git remote add origin "$REMOTE_URL"
fi

# Ensure local branch exists and is checked out
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  echo "Using existing local branch $BRANCH"
  git checkout "$BRANCH"
else
  current=$(git rev-parse --abbrev-ref HEAD)
  echo "Creating branch $BRANCH from current branch $current"
  git checkout -b "$BRANCH"
fi

# Push using http.extraHeader to avoid embedding token into URL
# This sends an Authorization header with the token for this push only.
echo "Pushing to https://github.com/${REPO}.git (branch: $BRANCH)"

git -c http.extraHeader="Authorization: token ${GITHUB_TOKEN}" push --set-upstream origin "$BRANCH"

if [[ $? -ne 0 ]]; then
  echo "Push failed." >&2
  exit 1
fi

# Reset origin URL to HTTPS without token (it already is)
echo "Ensuring origin URL remains without embedded token..."
git remote set-url origin "$REMOTE_URL"

echo "Push complete. Repository: https://github.com/${REPO} (branch: $BRANCH)"

echo "NOTE: Do NOT paste your token into chat. This script expects you to export it locally as GITHUB_TOKEN."
