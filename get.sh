#!/usr/bin/env bash
# l9s one-line install:
#   curl -fsSL https://raw.githubusercontent.com/JongoDB/l9s/main/get.sh | bash
# Options (pass after `bash -s --`):
#   --model <org/repo>   serve this model instead of the starter model
#   --no-model           install only
#   --ref <branch|tag>   install a specific version (default: main)
# e.g. curl -fsSL https://raw.githubusercontent.com/JongoDB/l9s/main/get.sh | bash -s -- --no-model
# Re-running it updates l9s and keeps your settings, profiles and history.
set -euo pipefail
REPO="JongoDB/l9s"
REF="main"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref) REF="${2:?--ref needs a branch or tag}"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
command -v curl >/dev/null || { echo "l9s needs curl" >&2; exit 1; }
command -v tar >/dev/null || { echo "l9s needs tar" >&2; exit 1; }
SRC="${L9S_HOME:-$HOME/.l9s}/src"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "==> downloading l9s ($REF)"
curl -fsSL "https://codeload.github.com/$REPO/tar.gz/$REF" | tar xz -C "$TMP"
mkdir -p "$(dirname "$SRC")"
rm -rf "$SRC"
mv "$TMP"/l9s-* "$SRC"
exec bash "$SRC/install.sh" ${ARGS[@]+"${ARGS[@]}"}
