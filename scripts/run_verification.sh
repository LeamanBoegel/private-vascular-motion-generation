#!/usr/bin/env bash
# Both verification layers, then the three written reports and the status roll-up.
set -euo pipefail

# The scripts run from a clone without requiring an install, so src/ leads the module path.
ROOT="${CAPS_ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m private_vascular_motion_generation.cli.verify --root "$ROOT" --scratch reports/scratch
"$PYTHON_BIN" -m private_vascular_motion_generation.cli.summarise --root "$ROOT" --show all
