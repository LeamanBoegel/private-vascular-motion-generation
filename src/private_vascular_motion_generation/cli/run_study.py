"""Run one or more study arms and write their measurements to the reports directory.

Ref: Sec. 4.9, Sec. 2.1-2.9.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..runner.logging_utils import LoggingConfig, configure_logging
from ..runner.studies import STUDY_REGISTRY, execute_study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a study arm of the CAPS release.")
    parser.add_argument("--study", required=True, choices=sorted(STUDY_REGISTRY), help="study arm to run")
    parser.add_argument("--root", default=".", help="release root containing the configuration tree")
    parser.add_argument("--out", default="reports", help="directory receiving the measurement file")
    parser.add_argument("--override", action="append", default=[], help="dotted key=value override")
    parser.add_argument("--max-rounds", type=int, default=None, help="bound the private pass explicitly")
    parser.add_argument("--log-file", default=None, help="optional log file path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    configure_logging(LoggingConfig(log_file=Path(args.log_file) if args.log_file else None))
    kwargs: dict[str, object] = {}
    if args.max_rounds is not None:
        kwargs["max_rounds"] = args.max_rounds
    outcome = execute_study(args.study, root, overrides=args.override, **kwargs)
    destination = root / args.out
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": str(Path(args.root)),
        "overrides": list(args.override),
        "outcome": outcome.to_dict(),
    }
    target = destination / f"{args.study}.json"
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"{outcome.name}: {outcome.status} -> {target.name}")
    for note in outcome.notes:
        print(f"  note: {note}")
    return 0 if outcome.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
