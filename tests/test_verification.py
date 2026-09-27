"""Verification-layer tests: the ledger, the manuscript checks and the report builders.

Ref: Sec. 2.8, Sec. 4.6.
"""

from __future__ import annotations

import json
from pathlib import Path

from private_vascular_motion_generation.verification import (
    EXECUTION_FAMILY,
    MANUSCRIPT_FAMILY,
    CheckLedger,
    CheckResult,
    build_claim_to_code,
    build_integrity_manifest,
    build_verification_report,
    close_enough,
    iter_tracked_files,
    relative_close_enough,
    run_manuscript_checks,
    sha256_of,
)


def test_check_result_validates_its_vocabulary() -> None:
    result = CheckResult(
        identifier="X-1",
        family=MANUSCRIPT_FAMILY,
        description="probe",
        source="Sec. 1",
        code_ref="probe:file",
        status="PASS",
    )
    assert result.to_dict()["status"] == "PASS"
    import pytest

    with pytest.raises(ValueError):
        CheckResult(
            identifier="X-2",
            family=MANUSCRIPT_FAMILY,
            description="probe",
            source="Sec. 1",
            code_ref="probe:file",
            status="MAYBE",
        )


def test_ledger_counts_and_statuses() -> None:
    ledger = CheckLedger()
    ledger.add(_result("A", MANUSCRIPT_FAMILY, "PASS"))
    ledger.add(_result("B", EXECUTION_FAMILY, "PASS"))
    assert ledger.counts()["PASS"] == 2
    assert ledger.overall_status() == "PASS"
    ledger.add(_result("C", EXECUTION_FAMILY, "BLOCKED"))
    assert ledger.overall_status() == "PARTIALLY_VERIFIED"
    ledger.add(_result("D", EXECUTION_FAMILY, "FAIL"))
    assert ledger.overall_status() == "FAIL"


def _result(identifier: str, family: str, status: str) -> CheckResult:
    return CheckResult(
        identifier=identifier,
        family=family,
        description=f"probe {identifier}",
        source="Sec. 1",
        code_ref="probe:file",
        status=status,
    )


def test_comparison_helpers() -> None:
    assert close_enough(0.842, 0.840, 0.003)
    assert not close_enough(0.842, 0.840, 0.001)
    assert relative_close_enough(0.0024, 0.0024, 1e-9)


def test_manuscript_checks_report_both_families() -> None:
    ledger = run_manuscript_checks()
    assert ledger.counts()["PASS"] > 80
    assert ledger.family_status(MANUSCRIPT_FAMILY) in ("PASS", "FAIL")
    assert ledger.family_status(EXECUTION_FAMILY) == "NOT_RUN"
    ids = [result.identifier for result in ledger.results]
    assert len(ids) == len(set(ids))


def test_manuscript_discrepancies_are_named() -> None:
    ledger = run_manuscript_checks()
    discrepancies = [result.identifier for result in ledger.results if result.status == "FAIL"]
    assert "MS-SC2-FIT" in discrepancies
    described = [result.note for result in ledger.results if result.identifier == "MS-SC2-FIT"]
    assert described and "order of magnitude" in described[0]


def test_claim_map_covers_every_status() -> None:
    payload = build_claim_to_code()
    statuses = {mapping["status"] for mapping in payload["mappings"]}
    assert {"IMPLEMENTED", "BLOCKED", "NOT_RUN"} <= statuses
    assert payload["counts"]["IMPLEMENTED"] >= 40
    refs = [mapping["code_ref"] for mapping in payload["mappings"]]
    assert len(refs) == len(set(refs))


def test_manifest_covers_the_tree_and_excludes_only_itself(release_root: Path) -> None:
    from private_vascular_motion_generation.verification.reporting import MANIFEST_EXCLUDES

    tracked = iter_tracked_files(release_root, MANIFEST_EXCLUDES)
    names = {str(path) for path in tracked}
    assert "integrity_manifest.json" not in names
    assert "verification_report.json" in names
    assert "claim_to_code.json" in names
    assert "README.md" in names
    assert not any(name.startswith("reports/") for name in names)
    digest = sha256_of(release_root / "README.md")
    assert len(digest) == 64


def test_manifest_is_deterministic(release_root: Path) -> None:
    first = build_integrity_manifest(release_root)
    second = build_integrity_manifest(release_root)
    assert first["aggregate"] == second["aggregate"]
    assert first["file_count"] == second["file_count"]


def test_verification_report_merges_both_layers() -> None:
    execution = CheckLedger()
    execution.add(_result("E-1", EXECUTION_FAMILY, "PASS"))
    report = build_verification_report(execution, run_manuscript_checks(), {"root": "."})
    assert report["execution_family_status"] == "PASS"
    assert report["counts"]["PASS"] > 0
    assert "measurements" in report
    json.dumps(report, default=str)


def test_shipped_reports_are_the_complete_ones(release_root: Path) -> None:
    """The committed reports must carry the full run, not a partial writer's output."""
    claim_map = json.loads((release_root / "claim_to_code.json").read_text(encoding="utf-8"))
    report = json.loads((release_root / "verification_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((release_root / "integrity_manifest.json").read_text(encoding="utf-8"))
    assert len(claim_map["mappings"]) >= 45
    assert len(report["execution_layer"]["results"]) >= 100
    assert len(report["manuscript_layer"]["results"]) >= 45
    assert report["overall_status"] in ("PASS", "PARTIALLY_VERIFIED")
    assert manifest["file_count"] == len(manifest["files"])
    names = {entry["path"] for entry in manifest["files"]}
    assert "verification_report.json" in names
    assert "integrity_manifest.json" not in names
