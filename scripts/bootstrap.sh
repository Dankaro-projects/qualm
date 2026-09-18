#!/usr/bin/env bash
# One-time setup for a fresh checkout.
#
# rustup installs entirely under ~/.cargo and ~/.rustup and needs no sudo.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "── Python ──"
if ! command -v uv >/dev/null; then
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
(cd python && uv sync && uv run pytest -q)

echo
echo "── Rust ──"
if command -v cargo >/dev/null; then
  (cd crates/qualm-harness && cargo build)
else
  cat <<'MSG'
cargo not found — the Rust half is not buildable yet. Install with:

  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  . "$HOME/.cargo/env"

No sudo required. Then re-run this script.
MSG
fi

echo
echo "── Reachability ──"
# Check the real inference path rather than assuming it. Per the verification
# standard: a config that has never been exercised is not verified.
OLLAMA="${QUALM_OLLAMA:-http://localhost:11434}"
if curl -fsS -m 5 "$OLLAMA/api/version" >/dev/null 2>&1; then
  echo "Ollama OK at $OLLAMA — $(curl -fsS -m 5 "$OLLAMA/api/version")"
else
  echo "Ollama NOT reachable at $OLLAMA (is the server running? set QUALM_OLLAMA to its address)" >&2
fi
