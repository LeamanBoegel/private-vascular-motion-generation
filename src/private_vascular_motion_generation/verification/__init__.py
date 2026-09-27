"""Two-layer verification of the release.

Ref: Sec. 2.8 (pre-specified criteria and retained null results), Sec. 4.9.
"""

from __future__ import annotations

from .checks import (
    EXECUTION_FAMILY,
    MANUSCRIPT_FAMILY,
    STATUSES,
    CheckLedger,
    CheckResult,
    close_enough,
    relative_close_enough,
)
from .execution_checks import run_execution_checks
from .manuscript_checks import run_manuscript_checks
from .reporting import (
    CLAIM_MAPPINGS,
    CLAIM_TO_CODE_FILE,
    MANIFEST_FILE,
    VERIFICATION_FILE,
    build_claim_to_code,
    build_integrity_manifest,
    build_verification_report,
    iter_tracked_files,
    sha256_of,
    write_all_reports,
    write_json,
)

__all__ = [
    "CLAIM_MAPPINGS",
    "CLAIM_TO_CODE_FILE",
    "EXECUTION_FAMILY",
    "MANIFEST_FILE",
    "MANUSCRIPT_FAMILY",
    "STATUSES",
    "VERIFICATION_FILE",
    "CheckLedger",
    "CheckResult",
    "build_claim_to_code",
    "build_integrity_manifest",
    "build_verification_report",
    "close_enough",
    "iter_tracked_files",
    "relative_close_enough",
    "run_execution_checks",
    "run_manuscript_checks",
    "sha256_of",
    "write_all_reports",
    "write_json",
]
