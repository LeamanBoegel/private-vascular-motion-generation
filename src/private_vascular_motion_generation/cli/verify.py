"""Run the two verification layers and write the three release reports.

Ref: Sec. 2.8, Sec. 4.6, Sec. 4.9.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..runner.logging_utils import LoggingConfig, configure_logging
from ..verification import run_execution_checks, write_all_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the CAPS release and write its reports.")
    parser.add_argument("--root", default=".", help="release root")
    parser.add_argument("--scratch", default="reports/scratch", help="scratch directory for checkpoint tests")
    parser.add_argument("--log-file", default=None, help="optional log file path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    scratch = (root / args.scratch).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    configure_logging(LoggingConfig(log_file=Path(args.log_file) if args.log_file else None))
    execution = run_execution_checks(root, scratch)
    summary = write_all_reports(root, execution, {"root": ".", "scratch": str(Path(args.scratch))})
    print(json.dumps(summary, indent=2, sort_keys=True))
    for result in execution.results:
        if result.status in ("FAIL", "BLOCKED", "NOT_RUN"):
            print(f"{result.status} {result.identifier}: {result.description}")
    return 0 if summary["overall_status"] in ("PASS", "PARTIALLY_VERIFIED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
