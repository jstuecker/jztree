#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-all}
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

run_main() {
  "$REPO_ROOT/packaging/docker-wheel-builder/run-build-main.sh"
}

run_cu13() {
  "$REPO_ROOT/packaging/docker-wheel-builder/run-build.sh"
}

run_cu12() {
  "$REPO_ROOT/packaging/docker-wheel-builder-cu12/run-build.sh"
}

case "$MODE" in
  main)
    run_main
    ;;
  cu13)
    run_cu13
    ;;
  cu12)
    run_cu12
    ;;
  all|both)
    run_main
    run_cu13
    run_cu12
    ;;
  *)
    echo "Usage: $0 [main|cu13|cu12|all]" >&2
    exit 2
    ;;
esac
