"""Read back the written reports and print a compact status roll-up.

Ref: Sec. 2.8 (retained null results are reported, not smoothed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..verification import CLAIM_TO_CODE_FILE, MANIFEST_FILE, VERIFICATION_FILE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarise the written CAPS release reports.")
    parser.add_argument("--root", default=".", help="release root")
    parser.add_argument("--show", choices=("status", "claims", "unresolved", "all"), default="status")
    return parser


def _load(root: Path, name: str) -> dict[str, object]:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(f"{name} has not been written yet; run the verify entry point first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must hold a mapping")
    return payload


def _entries(payload: dict[str, object], key: str) -> list[dict[str, object]]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise TypeError(f"{key} must hold a list of records")
    return [entry for entry in value if isinstance(entry, dict)]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    report = _load(root, VERIFICATION_FILE)
    print(f"overall_status        : {report['overall_status']}")
    print(f"counts                : {report['counts']}")
    print(f"manuscript_family     : {report['manuscript_family_status']}")
    print(f"execution_family      : {report['execution_family_status']}")
    discrepancies = _entries(report, "discrepancies")
    print(f"discrepancies         : {len(discrepancies)}")
    if args.show in ("status", "all"):
        unresolved = _entries(report, "unresolved")
        print(f"unresolved            : {len(unresolved)}")
    if args.show in ("claims", "all"):
        claims = _load(root, CLAIM_TO_CODE_FILE)
        print(f"claim mappings        : {claims['counts']}")
    if args.show in ("unresolved", "all"):
        for entry in _entries(report, "unresolved"):
            print(f"  {entry['status']} {entry['id']}: {str(entry['description'])[:88]}")
    if args.show in ("all",):
        manifest = _load(root, MANIFEST_FILE)
        print(f"manifest files        : {manifest['file_count']}")
        print(f"manifest aggregate    : {manifest['aggregate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
