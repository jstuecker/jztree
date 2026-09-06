#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-all}
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

run_main() {
  bash "$REPO_ROOT/packaging/run-builder.sh" main
}

run_cu13() {
  bash "$REPO_ROOT/packaging/run-builder.sh" cu13
}

run_cu12() {
  bash "$REPO_ROOT/packaging/run-builder.sh" cu12
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
