#!/usr/bin/env bash
# Primary arm. The reported plan is 12,000 rounds at a learning rate of 3.2e-3; bound the run
# with CAPS_MAX_ROUNDS when a shorter confirmation pass is intended.
set -euo pipefail

# The scripts run from a clone without requiring an install, so src/ leads the module path.
ROOT="${CAPS_ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
MAX_ROUNDS="${CAPS_MAX_ROUNDS:-}"

if [[ -n "$MAX_ROUNDS" ]]; then
  "$PYTHON_BIN" -m private_vascular_motion_generation.cli.run_study \
    --study caps_main --root "$ROOT" --max-rounds "$MAX_ROUNDS"
else
  "$PYTHON_BIN" -m private_vascular_motion_generation.cli.run_study --study caps_main --root "$ROOT"
fi
