"""Generation of the three machine-readable release reports.

Ref: Sec. 4.6 (the privacy tuple travels with the artifact), Sec. 4.9.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .checks import CheckLedger
from .manuscript_checks import run_manuscript_checks

CLAIM_TO_CODE_FILE = "claim_to_code.json"
VERIFICATION_FILE = "verification_report.json"
MANIFEST_FILE = "integrity_manifest.json"
MANIFEST_EXCLUDES: tuple[str, ...] = (
    MANIFEST_FILE,
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "reports",
    "data",
)


@dataclass(frozen=True)
class ClaimMapping:
    claim_id: str
    paper_item: str
    statement: str
    code_ref: str
    status: str
    evidence: str

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "paper_item": self.paper_item,
            "statement": self.statement,
            "code_ref": self.code_ref,
            "status": self.status,
            "evidence": self.evidence,
        }


CLAIM_MAPPINGS: tuple[ClaimMapping, ...] = (
    ClaimMapping(
        "C-01",
        "Sec. 4.3",
        "A procedure is the synchronised pair (X in HxWxT, A in 6xT).",
        "cohort/procedure.py:ProcedureRecord",
        "IMPLEMENTED",
        "EX-DATA-01, EX-DATA-02",
    ),
    ClaimMapping(
        "C-02",
        "Sec. 4.2",
        "Site A/B/C accrue 2214/1986/1642 procedures across two regions.",
        "cohort/partition.py:site_strata",
        "IMPLEMENTED",
        "EX-DATA-03, EX-DATA-04, MS-COHORT-01",
    ),
    ClaimMapping(
        "C-03",
        "Table 6 caption",
        "Every proportion is a Wilson interval carrying its own denominator.",
        "metrics/clinical.py:wilson_interval",
        "IMPLEMENTED",
        "EX-MET-15, MS-T6-WILSON-anatomy_simple",
    ),
    ClaimMapping(
        "C-04",
        "Sec. 4.10",
        "Diagnostic frames are the pre-opacification mask, the peak-opacification frame and the load transients.",
        "cohort/frames.py:select_diagnostic_frames",
        "IMPLEMENTED",
        "EX-DATA-05",
    ),
    ClaimMapping(
        "C-05",
        "Sec. 4.10",
        "Contact events are counted on the force-channel-covered subset after a threshold and a debounce rule.",
        "cohort/force_channel.py:count_contact_events",
        "IMPLEMENTED",
        "EX-DATA-06, EX-MET-12",
    ),
    ClaimMapping(
        "C-06",
        "Sec. 4.4",
        "The coupling head solves Sigma12 Sigma22^-1 Sigma21 u = lambda^2 Sigma11 u.",
        "coupling/cca.py:solve_canonical",
        "IMPLEMENTED",
        "EX-COUP-01, EX-COUP-02",
    ),
    ClaimMapping(
        "C-07",
        "Algorithm 1 step 7",
        "The coupling loss is -r^-1 sum of the leading squared canonical correlations.",
        "coupling/cca.py:coupling_loss",
        "IMPLEMENTED",
        "EX-COUP-03",
    ),
    ClaimMapping(
        "C-08",
        "Sec. 4.4",
        "The shared latent is the only channel of procedure identity into either branch.",
        "coupling/latent.py:draw_latent",
        "IMPLEMENTED",
        "EX-COUP-06",
    ),
    ClaimMapping(
        "C-09",
        "Eq. (1)",
        "The noise precision is sigma_par^-2 Pi_S + sigma_perp^-2 Pi_Sperp.",
        "geometry/shaped_covariance.py:ShapedCovariance.precision",
        "IMPLEMENTED",
        "EX-GEO-03",
    ),
    ClaimMapping(
        "C-10",
        "Eq. (2)",
        "The released workload is AG + Z with procedures on rows and window micro-steps on columns.",
        "substrate/__init__.py:add_shaped_noise",
        "IMPLEMENTED",
        "EX-GEO-04, EX-MECH-06",
    ),
    ClaimMapping(
        "C-11",
        "Eq. (4)",
        "The noise-normalised sensitivity is u = sqrt(sigma_par^-2 + rho sigma_perp^-2).",
        "geometry/shaped_covariance.py:normalised_sensitivity",
        "IMPLEMENTED",
        "EX-GEO-07, EX-GEO-08",
    ),
    ClaimMapping(
        "C-12",
        "Sec. 4.6",
        "Anisotropy is admissible, not merely permitted, when sigma_perp < sigma_par.",
        "geometry/shaped_covariance.py:anisotropy_admissible",
        "IMPLEMENTED",
        "EX-GEO-06",
    ),
    ClaimMapping(
        "C-13",
        "Eq. (3)",
        "The Renyi bound composes over rounds and converts at delta with a polylogarithmic window slack.",
        "accounting/renyi.py:convert_to_epsilon",
        "IMPLEMENTED",
        "EX-ACC-01 to EX-ACC-06",
    ),
    ClaimMapping(
        "C-14",
        "Algorithm 1 step 8",
        "Clipping is taken once at a single norm before any noise is drawn.",
        "accounting/clipping.py:clip_workload",
        "IMPLEMENTED",
        "EX-CLIP-01 to EX-CLIP-03",
    ),
    ClaimMapping(
        "C-15",
        "Sec. 4.6",
        "Reporting a budget without its accountant, unit and delta is incomplete.",
        "accounting/tuple.py:PrivacyTuple",
        "IMPLEMENTED",
        "EX-MECH-02",
    ),
    ClaimMapping(
        "C-16",
        "Algorithm 2",
        "The shaping subspace is estimated offline from non-private references and frozen.",
        "estimator/reference_estimator.py:ReferenceShapingEstimator",
        "IMPLEMENTED",
        "EX-EST-01 to EX-EST-03",
    ),
    ClaimMapping(
        "C-17",
        "Sec. 4.4",
        "The estimator never reads a private record.",
        "cohort/reference.py:assert_estimator_disjointness",
        "IMPLEMENTED",
        "EX-EST-02",
    ),
    ClaimMapping(
        "C-18",
        "Algorithm 1 step 12",
        "The private update is theta <- theta - gamma (AG + Z).",
        "runner/trainer.py:CAPSTrainer.run",
        "IMPLEMENTED",
        "EX-UPD-01, EX-MECH-02",
    ),
    ClaimMapping(
        "C-19",
        "Algorithm 1 step 16",
        "Release samples z_p and decodes both branches as pure post-processing.",
        "substrate/__init__.py:release_substrate",
        "IMPLEMENTED",
        "EX-MECH-07",
    ),
    ClaimMapping(
        "C-20",
        "Sec. 4.4",
        "The procedure window is prefix-structured, so composition is logarithmic rather than linear.",
        "coupling/prefix_sum.py:prefix_sum_scan",
        "IMPLEMENTED",
        "EX-COUP-04, MS-T7D-COST-POLYLOG",
    ),
    ClaimMapping(
        "C-21",
        "Sec. 4.5",
        "The appearance branch is a state-space vessel-segmentation encoder.",
        "branches/ssm.py:SwinUMambaEncoder",
        "IMPLEMENTED",
        "EX-FWD-02",
    ),
    ClaimMapping(
        "C-22",
        "Sec. 4.5",
        "The motion branch is a flow-matching action head over six channels.",
        "branches/motion.py:FlowMatchingActionHead",
        "IMPLEMENTED",
        "EX-FWD-03",
    ),
    ClaimMapping(
        "C-23",
        "Sec. 4.5",
        "Only the attention modules are privatised; the rest of the backbone stays frozen.",
        "branches/attention.py:PrivatisedTransformerStack",
        "IMPLEMENTED",
        "EX-FWD-02, EX-MECH-05",
    ),
    ClaimMapping(
        "C-24",
        "Sec. 4.10",
        "Fidelity is FID, KID and SSIM at matched resolution.",
        "metrics/fidelity.py:frechet_distance",
        "IMPLEMENTED",
        "EX-MET-01 to EX-MET-05",
    ),
    ClaimMapping(
        "C-25",
        "Sec. 4.10",
        "Vessel geometry is Dice overlap and the 95th-percentile Hausdorff distance.",
        "metrics/segmentation.py:percentile_hausdorff",
        "IMPLEMENTED",
        "EX-MET-06, EX-MET-07",
    ),
    ClaimMapping(
        "C-26",
        "Sec. 4.10",
        "Motion agreement is ADE, FDE, DTW, smoothness and path-length efficiency.",
        "metrics/trajectory.py:dynamic_time_warping",
        "IMPLEMENTED",
        "EX-MET-08 to EX-MET-11",
    ),
    ClaimMapping(
        "C-27",
        "Sec. 4.10",
        "Values above 0.92 path-length efficiency indicate tracking or centreline-extraction.",
        "metrics/trajectory.py:path_efficiency_is_artifact",
        "IMPLEMENTED",
        "EX-MET-11",
    ),
    ClaimMapping(
        "C-28",
        "Sec. 4.10",
        "Safety is contact events per procedure, peak contact force and the force-duration integral.",
        "metrics/safety.py:safety_from_rollouts",
        "IMPLEMENTED",
        "EX-MET-12, EX-MET-14",
    ),
    ClaimMapping(
        "C-29",
        "Table 5 caption",
        "A clinical endpoint is judged against its minimum clinically important difference.",
        "metrics/safety.py:clears_mcid",
        "IMPLEMENTED",
        "MS-T5-CONTACT-MCID",
    ),
    ClaimMapping(
        "C-30",
        "Sec. 2.2",
        "The privacy audit is attack based; a distance proxy is informative only.",
        "audit/membership.py:member_nonmember_auc",
        "IMPLEMENTED",
        "EX-AUD-01 to EX-AUD-03",
    ),
    ClaimMapping(
        "C-31",
        "Sec. 2.8",
        "The three pre-specified primaries use Holm; the exploratory inventory uses Benjamini-Hochberg.",
        "analysis/multiplicity.py:holm_bonferroni",
        "IMPLEMENTED",
        "EX-AN-07 to EX-AN-08, MS-SE1-INVENTORY",
    ),
    ClaimMapping(
        "C-32",
        "Sec. 2.8",
        "Correlated areas are compared with the nonparametric DeLong construction.",
        "analysis/delong.py:delong_paired_test",
        "IMPLEMENTED",
        "EX-AN-01, EX-AN-02",
    ),
    ClaimMapping(
        "C-33",
        "Table 3",
        "The resolution x conditioning factorial separates the two main effects and their interaction.",
        "analysis/factorial.py:full_contrast_table",
        "IMPLEMENTED",
        "EX-AN-03, MS-T3-COND-fid",
    ),
    ClaimMapping(
        "C-34",
        "Table 4 panel (b)",
        "Pairwise removals are super-additive except the estimator pair.",
        "analysis/interaction.py:PairwiseInteraction",
        "IMPLEMENTED",
        "MS-T4-IR-coupling-shaped",
    ),
    ClaimMapping(
        "C-35",
        "Table 1 caption",
        "T_null = 1.0% relative governs the null-result fence and the star marks.",
        "analysis/null_fence.py:FenceVerdict",
        "IMPLEMENTED",
        "MS-T4-ESTIMATOR-NULL",
    ),
    ClaimMapping(
        "C-36",
        "Sec. 2.6",
        "The policy is compared against the standard of care and two classical constrained controllers.",
        "policy/rollout.py:rollout_path_follower",
        "IMPLEMENTED",
        "EX-POL-04, EX-POL-05, EX-POL-06",
    ),
    ClaimMapping(
        "C-37",
        "Table 5 panel (b)",
        "A soft actor-critic and a latent world-model planner are reproduced on the same held-out set.",
        "policy/sac.py:SoftActorCritic",
        "IMPLEMENTED",
        "EX-POL-07 to EX-POL-10",
    ),
    ClaimMapping(
        "C-38",
        "Sec. 4.5",
        "The policy learned on the released substrate is a diffusion policy.",
        "policy/diffusion_policy.py:DiffusionPolicy",
        "IMPLEMENTED",
        "EX-POL-11, EX-POL-12",
    ),
    ClaimMapping(
        "C-39",
        "Supplementary Table T7 panel D",
        "The window ladder is measured against a single-step charge.",
        "coupling/prefix_sum.py:composition_multiplier_ladder",
        "IMPLEMENTED",
        "MS-T7D-COST-POLYLOG",
    ),
    ClaimMapping(
        "C-40",
        "Supplementary Table SC1",
        "Perturbation families degrade the substrate monotonically.",
        "verification/manuscript_tables.py:PERTURBATION_TABLE",
        "IMPLEMENTED",
        "MS-SC1-intensity",
    ),
    ClaimMapping(
        "C-41",
        "Table 1",
        "Main comparison on the held-out test partition of 1,204 procedures.",
        "verification/manuscript_tables.py:MAIN_TABLE",
        "BLOCKED",
        "EX-BLK-01",
    ),
    ClaimMapping(
        "C-42",
        "Table 2",
        "Privacy-utility frontier with MIA audit at parity utility.",
        "verification/manuscript_tables.py:FRONTIER_TABLE",
        "BLOCKED",
        "EX-BLK-01, EX-NR-01",
    ),
    ClaimMapping(
        "C-43",
        "Table 5 panel (a)",
        "Policy, safety and clinical endpoints on held-out clinical trajectories.",
        "policy/rollout.py:aggregate_strategy",
        "BLOCKED",
        "EX-BLK-06",
    ),
    ClaimMapping(
        "C-44",
        "Table 6",
        "Per-site, per-region and per-arm consistency with subgroup analysis.",
        "cohort/partition.py:SITE_HELD_OUT",
        "BLOCKED",
        "EX-BLK-04",
    ),
    ClaimMapping(
        "C-45",
        "Sec. 4.10",
        "Reader validity across clinician, model and combined arms.",
        "metrics/clinical.py:three_way_reader_contrast",
        "BLOCKED",
        "EX-BLK-03",
    ),
    ClaimMapping(
        "C-46",
        "Sec. 2.6",
        "Bench-top phantom force envelope against the in-vivo perforation anchor.",
        "metrics/safety.py:force_envelope_margin",
        "BLOCKED",
        "EX-BLK-05",
    ),
    ClaimMapping(
        "C-47",
        "Sec. 2.8",
        "Coefficient of variation of the private mechanism over three training occasions.",
        "analysis/variance.py:dispersion",
        "NOT_RUN",
        "EX-NR-02",
    ),
    ClaimMapping(
        "C-48",
        "Supplementary Table SC2",
        "Power-law fit residuals of the scaling panel.",
        "analysis/powerlaw.py:residuals_in_units",
        "NOT_RUN",
        "EX-NR-03, MS-SC2-FIT",
    ),
    ClaimMapping(
        "C-49",
        "Sec. 4.4",
        "The 8.3B latent video diffusion transformer backbone.",
        "branches/attention.py:PrivateAttentionBlock",
        "BLOCKED",
        "EX-BLK-02",
    ),
    ClaimMapping(
        "C-50",
        "Sec. 4.2",
        "Expert vessel annotations define the Dice and Hausdorff reference.",
        "runner/batching.py:vessel_mask",
        "BLOCKED",
        "EX-BLK-03",
    ),
)


def build_claim_to_code() -> dict[str, object]:
    tally: dict[str, int] = {}
    for mapping in CLAIM_MAPPINGS:
        tally[mapping.status] = tally.get(mapping.status, 0) + 1
    return {
        "release": (
            "Differentially private joint generation of vascular images and catheter motion " "for collision avoidance"
        ),
        "status_legend": {
            "IMPLEMENTED": "code path exists and is exercised by the execution layer",
            "BLOCKED": "requires the private cohort, the reported backbone or laboratory hardware",
            "NOT_RUN": "code path exists but the manuscript does not state the quantity needed to run it",
        },
        "counts": tally,
        "mappings": [mapping.to_dict() for mapping in CLAIM_MAPPINGS],
    }


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_tracked_files(root: Path, excludes: Iterable[str]) -> list[Path]:
    skip = set(excludes)
    tracked: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in skip for part in relative.parts):
            continue
        tracked.append(relative)
    return tracked


def build_integrity_manifest(root: Path) -> dict[str, object]:
    files = iter_tracked_files(root, MANIFEST_EXCLUDES)
    entries = [
        {"path": str(relative), "bytes": (root / relative).stat().st_size, "sha256": sha256_of(root / relative)}
        for relative in files
    ]
    aggregate = hashlib.sha256(
        "".join(f"{entry['path']}:{entry['sha256']}" for entry in entries).encode("utf-8")
    ).hexdigest()
    return {
        "algorithm": "sha256",
        "excludes": list(MANIFEST_EXCLUDES),
        "file_count": len(entries),
        "aggregate": aggregate,
        "files": entries,
    }


def build_verification_report(
    execution: CheckLedger, manuscript: CheckLedger, extras: Mapping[str, object]
) -> dict[str, object]:
    combined = CheckLedger()
    combined.extend(manuscript.results)
    combined.extend(execution.results)
    payload: dict[str, object] = {
        "overall_status": combined.overall_status(),
        "counts": combined.counts(),
        "manuscript_family_status": combined.family_status("manuscript_arithmetic"),
        "execution_family_status": combined.family_status("code_execution"),
        "execution_layer": execution.to_dict(),
        "manuscript_layer": manuscript.to_dict(),
        "discrepancies": combined.discrepancies(),
        "unresolved": [result.to_dict() for result in combined.results if result.status in ("BLOCKED", "NOT_RUN")],
        "measurements": dict(extras),
    }
    return payload


def write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")
    return path


def write_all_reports(
    root: Path, execution: CheckLedger, extras: Mapping[str, object] | None = None
) -> dict[str, object]:
    manuscript = run_manuscript_checks()
    claim_map = build_claim_to_code()
    report = build_verification_report(execution, manuscript, extras or {})
    write_json(root / CLAIM_TO_CODE_FILE, claim_map)
    write_json(root / VERIFICATION_FILE, report)
    manifest = build_integrity_manifest(root)
    write_json(root / MANIFEST_FILE, manifest)
    return {
        "overall_status": report["overall_status"],
        "counts": report["counts"],
        "manifest_files": manifest["file_count"],
        "manifest_aggregate": manifest["aggregate"],
    }
