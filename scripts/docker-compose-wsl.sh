#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

is_wsl() {
  grep -qiE "microsoft|wsl" /proc/version 2>/dev/null
}

default_windows_host_ip() {
  awk '/^nameserver[[:space:]]+/ { print $2; exit }' /etc/resolv.conf
}

if is_wsl && [[ -z "${HOST_DOCKER_INTERNAL_TARGET:-}" ]]; then
  HOST_DOCKER_INTERNAL_TARGET="$(default_windows_host_ip)"
  export HOST_DOCKER_INTERNAL_TARGET
  printf 'Usando IP de Windows para host.docker.internal: %s\n' "$HOST_DOCKER_INTERNAL_TARGET"
fi

if [[ -f "$ROOT_DIR/.venv" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.venv"
  set +a
  printf 'Usando OPENAI_API_BASE=%s\n' "${OPENAI_API_BASE:-http://host.docker.internal:1234/v1}"
fi

if [[ $# -eq 0 ]]; then
  set -- up --build
fi

docker compose "$@"
