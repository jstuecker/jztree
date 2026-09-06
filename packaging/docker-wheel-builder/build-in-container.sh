#!/usr/bin/env bash
set -euo pipefail
export BUILD_MODE=${BUILD_MODE:-cu13}
exec /opt/python/cp311-cp311/bin/python "${REPO_ROOT:-/workspace}/packaging/wheel_builder.py"
