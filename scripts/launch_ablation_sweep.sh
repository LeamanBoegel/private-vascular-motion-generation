#!/usr/bin/env bash
# Component ablation, window ladder, factorial and the three scaling panels.
set -euo pipefail

# The scripts run from a clone without requiring an install, so src/ leads the module path.
ROOT="${CAPS_ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
MAX_ROUNDS="${CAPS_MAX_ROUNDS:-2}"

for study in \
  ablation_components \
  ablation_window_ladder \
  factorial_resolution_conditioning \
  scaling_substrate_volume \
  scaling_reference_corpus \
  scaling_window_horizon
do
  "$PYTHON_BIN" -m private_vascular_motion_generation.cli.run_study \
    --study "$study" --root "$ROOT" --max-rounds "$MAX_ROUNDS"
done
