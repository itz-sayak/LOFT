Pushing this repository to GitHub (safe helper)
===============================================

This repository includes a helper script to create a GitHub repository (if needed) and push the current code.

Important safety notes
- I will NOT and CANNOT use any token you pasted into chat. Do NOT paste tokens into chat.
- Run the helper script locally and provide the token via an environment variable only.

Quick steps
1. Inspect and commit everything you want uploaded:

```bash
cd /mnt/zone/B/NEW/P2P-Bridge-OT-real-latent
git add -A
git commit -m "Snapshot: upload LOFT repo"
```

2. Export your GitHub personal access token in your shell (do NOT paste it to anyone):

```bash
export GITHUB_TOKEN="ghp_your_token_here"
```

3. Run the helper script:

```bash
# Example: push to itz-sayak/LOFT as a public repo on branch 'main'
bash scripts/git_push_repo.sh itz-sayak/LOFT public main
```

What the script does
- Verifies your working tree is clean.
- Checks whether `itz-sayak/LOFT` exists on GitHub via the API.
  - If it doesn't exist and the authenticated user matches the target owner,
    the script will attempt to create the repo under that account.
  - If the authenticated user is different from the target owner, the script exits
    with a clear error (to avoid accidental permission issues).
- Sets the `origin` remote to `https://github.com/<owner>/<repo>.git`.
- Pushes the specified branch using a temporary HTTP Authorization header
  so the token is not embedded in the remote URL.

Troubleshooting
- If you prefer an interactive approach, use the GitHub CLI (`gh`) instead:

```bash
# login once interactively
gh auth login
# create the repo (if needed) and push
gh repo create itz-sayak/LOFT --public --source=. --remote=origin --push
```

- If you get permission errors when creating the repo, you likely do not own `itz-sayak`.
  Push to a repo under your own account (e.g., `yourusername/LOFT`) instead, or ask the
  owner to grant you the necessary permissions.

Security reminder
- Do not paste or upload your personal access tokens in public places or chats.
- The helper script reads the token from `GITHUB_TOKEN` and does not save it to disk.
