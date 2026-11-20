#!/usr/bin/env bash
set -euo pipefail

# Publish the built Showcase static site (`showcase/`) as a clean orphan snapshot
# onto the `gh-pages` branch without polluting `main` git history.
#
# Usage:
#   ./scripts/publish_gh_pages.sh            # publish current showcase/ to origin gh-pages
#   ./scripts/publish_gh_pages.sh --rebuild  # rebuild showcase data first, then publish
#   ./scripts/publish_gh_pages.sh --dry-run  # stage temp gh-pages commit and inspect without pushing

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOWCASE_DIR="$REPO_ROOT/showcase"
REMOTE_NAME="origin"
BRANCH_NAME="gh-pages"
REBUILD=0
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --rebuild)
      REBUILD=1
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      sed -n '3,11p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ "$REBUILD" -eq 1 ]]; then
  echo "[1/4] Rebuilding showcase data..."
  python3 "$REPO_ROOT/scripts/build_showcase.py"
fi

if [[ ! -f "$SHOWCASE_DIR/index.html" ]] || [[ ! -f "$SHOWCASE_DIR/data/overview.js" ]]; then
  echo "Showcase build artifacts missing in $SHOWCASE_DIR. Run with --rebuild first." >&2
  exit 1
fi

REMOTE_URL="$(git -C "$REPO_ROOT" remote get-url "$REMOTE_NAME")"
GIT_NAME="$(git -C "$REPO_ROOT" config user.name)"
GIT_EMAIL="$(git -C "$REPO_ROOT" config user.email)"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "[2/4] Staging clean showcase snapshot in $TMP_DIR..."
cp -a "$SHOWCASE_DIR/." "$TMP_DIR/"
touch "$TMP_DIR/.nojekyll"

git -C "$TMP_DIR" init -b "$BRANCH_NAME" >/dev/null
git -C "$TMP_DIR" config user.name "$GIT_NAME"
git -C "$TMP_DIR" config user.email "$GIT_EMAIL"
git -C "$TMP_DIR" add -A

COMMIT_MSG="Publish Showcase snapshot ($(date '+%Y-%m-%d %H:%M:%S %z'))"
git -C "$TMP_DIR" commit -m "$COMMIT_MSG" >/dev/null

echo "[3/4] Packaged files:"
git -C "$TMP_DIR" ls-tree -r --name-only HEAD | sed 's/^/  - /'

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[4/4] Dry run complete; nothing pushed to $REMOTE_URL ($BRANCH_NAME)."
  exit 0
fi

echo "[4/4] Pushing orphan snapshot to $REMOTE_URL ($BRANCH_NAME)..."
git -C "$TMP_DIR" push --force "$REMOTE_URL" "$BRANCH_NAME"
echo "Done! Enable GitHub Pages on branch '$BRANCH_NAME' / (root) in repository Settings -> Pages."
