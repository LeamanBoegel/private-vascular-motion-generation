#!/usr/bin/env bash
# Stage the auxiliary public reference layer used by the shaping estimator.
#
# The manuscript names no public angiographic cohort in full, so this script does not
# download anything on its own. It validates whatever the operator has staged under
# $CAPS_REFERENCE_ROOT and writes the manifest the loader expects.
set -euo pipefail

ROOT="${CAPS_REFERENCE_ROOT:-data/reference}"
EXPECTED_CLIPS="${CAPS_EXPECT_CLIPS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - "$ROOT" "$EXPECTED_CLIPS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
clips = sorted(root.glob("clip_*.pt"), key=lambda item: int(item.stem.split("_")[1]))
if not clips:
    raise SystemExit(
        f"no reference clips staged under {root.name}; place clip_<index>.pt payloads there first"
    )
if len(clips) < expected:
    raise SystemExit(f"expected at least {expected} reference clips, found {len(clips)}")
manifest = {
    "clips": [
        {"index": index, "source_kind": "public_angiography" if position % 2 == 0 else "open_simulator"}
        for position, index in enumerate(int(path.stem.split("_")[1]) for path in clips)
    ]
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"manifest written for {len(manifest['clips'])} reference clips")
PY
