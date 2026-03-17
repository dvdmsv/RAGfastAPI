#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

is_wsl() {
  grep -qiE "microsoft|wsl" /proc/version 2>/dev/null
}

if is_wsl && [[ -z "${OPENAI_API_BASE:-}" ]]; then
  OPENAI_API_BASE="http://localhost:1234/v1"
  export OPENAI_API_BASE
  printf 'Usando OPENAI_API_BASE=%s\n' "$OPENAI_API_BASE"
fi

if [[ -f "$ROOT_DIR/.venv" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.venv"
  set +a
  printf 'Usando OPENAI_API_BASE=%s\n' "${OPENAI_API_BASE:-http://localhost:1234/v1}"
fi

if [[ $# -eq 0 ]]; then
  set -- up --build
fi

docker compose -f "$ROOT_DIR/docker-compose.wsl.yml" "$@"
