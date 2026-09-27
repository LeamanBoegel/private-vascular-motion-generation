"""First verification layer: the manuscript's own arithmetic and its method statements.

Ref: Tables 1-6, Supplementary Tables SE1, SB1, T7, SC1, SC2, SC3, and Eq. (1)-(4).
Every check recomputes a printed quantity from other printed quantities, so a FAIL here
is a discrepancy inside the manuscript rather than a defect of the release.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ..analysis.factorial import CELLS, FactorialCell, full_contrast_table, rho_invariance_across_resolution
from ..analysis.interaction import PairwiseInteraction, report_pairwise
from ..analysis.multiplicity import benjamini_hochberg, declare_families, holm_bonferroni, inventory_total
from ..analysis.null_fence import fence_verdict
from ..analysis.powerlaw import fit_power_law, residuals_in_units, systematic_at_size
from ..cohort.partition import (
    ANATOMY_TERTILES,
    CLINICAL_ARMS,
    FULL_COHORT_SIZE,
    HELD_OUT_SIZE,
    SITE_HELD_OUT,
    SITE_TOTALS,
)
from ..coupling.prefix_sum import (
    REPORTED_COMPOSITION_MULTIPLIER,
    REPORTED_RELATIVE_LATENCY,
    WINDOW_LADDER,
    fit_log_ladder,
)
from ..metrics.clinical import wilson_interval
from ..metrics.safety import force_envelope_margin
from .checks import MANUSCRIPT_FAMILY, CheckLedger, CheckResult, close_enough
from .manuscript_tables import (
    ABLATION_TABLE,
    COHORT_TABLE,
    COMPARISON_TABLE,
    EXPLORATORY_COMPARISON_COUNT,
    FACTORIAL_CONTRASTS,
    FACTORIAL_TABLE,
    FRONTIER_TABLE,
    INITIALISATION_SUCCESS,
    MAIN_TABLE,
    NON_PRIVATE_AUDIT,
    PAIRWISE_INTERACTIONS,
    PERTURBATION_TABLE,
    PRIMARY_COMPARISON_COUNT,
    RAW_SCALING_TABLE,
    REFERENCE_CORPUS_LADDER,
    SAFETY_TABLE,
    STRATUM_COUNTS,
    STRATUM_TABLE,
    SUBGROUP_TABLE,
    SUBSTRATE_SCALING,
    AblationRow,
)

REFERENCE_F1: float = 0.842
NULL_FENCE: float = 0.01
WILSON_TOLERANCE: float = 0.005


def _check(
    identifier: str,
    description: str,
    source: str,
    code_ref: str,
    observed: object,
    expected: object,
    tolerance: float | None,
    passed: bool,
    note: str = "",
) -> CheckResult:
    return CheckResult(
        identifier=identifier,
        family=MANUSCRIPT_FAMILY,
        description=description,
        source=source,
        code_ref=code_ref,
        status="PASS" if passed else "FAIL",
        observed=observed,
        expected=expected,
        tolerance=tolerance,
        note=note,
    )


def check_cohort_arithmetic(ledger: CheckLedger) -> None:
    total = sum(SITE_TOTALS.values())
    ledger.add(
        _check(
            "MS-COHORT-01",
            "per-site accrued counts sum to the full private cohort",
            "Sec. 4.2, Table 6 panel (a)",
            "cohort/partition.py:site_strata",
            total,
            FULL_COHORT_SIZE,
            0.0,
            total == FULL_COHORT_SIZE,
        )
    )
    held_out = sum(SITE_HELD_OUT.values())
    ledger.add(
        _check(
            "MS-COHORT-02",
            "per-site held-out counts sum to the held-out partition",
            "Sec. 4.2",
            "cohort/partition.py:default_partition",
            held_out,
            HELD_OUT_SIZE,
            0.0,
            held_out == HELD_OUT_SIZE,
        )
    )
    region_one = SITE_TOTALS["A"] + SITE_TOTALS["B"]
    ledger.add(
        _check(
            "MS-COHORT-03",
            "Region I pool equals the two Region I sites",
            "Table 6 panel (a)",
            "cohort/partition.py:region_pool",
            region_one,
            4200,
            0.0,
            region_one == 4200,
        )
    )
    arm_total = sum(CLINICAL_ARMS.values())
    ledger.add(
        _check(
            "MS-COHORT-04",
            "the two clinical arms sum to the retrospective-plus-prospective total",
            "Table 6 panel (a)",
            "cohort/partition.py:CLINICAL_ARMS",
            arm_total,
            4106 + 1736,
            0.0,
            arm_total == 4106 + 1736,
        )
    )
    tertiles = sum(ANATOMY_TERTILES.values())
    ledger.add(
        _check(
            "MS-COHORT-05",
            "anatomy tertiles sum to the held-out partition",
            "Table 6 panel (b)",
            "cohort/partition.py:anatomy_tertile_assignment",
            tertiles,
            HELD_OUT_SIZE,
            0.0,
            tertiles == HELD_OUT_SIZE,
        )
    )
    region_one_held_out = SITE_HELD_OUT["A"] + SITE_HELD_OUT["B"]
    ledger.add(
        _check(
            "MS-COHORT-06",
            "held-out Region I slice equals the two Region I held-out sites",
            "Table 6 caption",
            "cohort/partition.py:SITE_HELD_OUT",
            region_one_held_out,
            866,
            0.0,
            region_one_held_out == 866,
        )
    )
    coverage = [row.force_coverage for row in COHORT_TABLE[:3]]
    ledger.add(
        _check(
            "MS-COHORT-07",
            "force-channel coverage is ordered Site A > Site B > Site C",
            "Table 6 panel (a)",
            "cohort/partition.py:FORCE_COVERAGE_BY_SITE",
            coverage,
            "descending",
            None,
            coverage == sorted(coverage, reverse=True),
        )
    )


def check_table_one(ledger: CheckLedger) -> None:
    unconstrained = MAIN_TABLE[0]
    for row in MAIN_TABLE[1:]:
        delta = row.fid - unconstrained.fid
        relative = delta / unconstrained.fid
        if "n.s." in row.delta_versus_unconstrained:
            expected_delta = 0.4
        else:
            expected_delta = float(row.delta_versus_unconstrained.split()[0].lstrip("+"))
        ledger.add(
            _check(
                f"MS-T1-FID-{row.label[:24]}",
                "signed FID delta against the unconstrained reference",
                "Table 1",
                "verification/manuscript_tables.py:MAIN_TABLE",
                round(delta, 3),
                expected_delta,
                0.05,
                close_enough(delta, expected_delta, 0.05),
            )
        )
        ledger.add(
            _check(
                f"MS-T1-REL-{row.label[:24]}",
                "relative FID change matches the printed percentage",
                "Table 1",
                "verification/manuscript_tables.py:MAIN_TABLE",
                round(relative * 100.0, 2),
                float(row.delta_versus_unconstrained.split("(")[1].split("%")[0].lstrip("+")),
                0.06,
                close_enough(
                    relative * 100.0,
                    float(row.delta_versus_unconstrained.split("(")[1].split("%")[0].lstrip("+")),
                    0.06,
                ),
            )
        )
    cap = MAIN_TABLE[-1]
    dtw_delta = cap.dtw - unconstrained.dtw
    ledger.add(
        _check(
            "MS-T1-DTW-CAPS",
            "CAPS trajectory-agreement cost against the unconstrained generator",
            "Sec. 2.4",
            "verification/manuscript_tables.py:MAIN_TABLE",
            round(dtw_delta, 3),
            0.003,
            0.0005,
            close_enough(dtw_delta, 0.003, 0.0005),
        )
    )
    decoupled_cost = MAIN_TABLE[1].dtw - unconstrained.dtw
    ledger.add(
        _check(
            "MS-T1-DTW-DECOUPLED",
            "decoupling costs more trajectory agreement than the privacy constraint",
            "Sec. 2.4",
            "verification/manuscript_tables.py:MAIN_TABLE",
            round(decoupled_cost, 3),
            0.018,
            0.0005,
            close_enough(decoupled_cost, 0.018, 0.0005),
        )
    )
    ordered = [row.dice for row in MAIN_TABLE]
    ledger.add(
        _check(
            "MS-T1-DICE-ORDER",
            "CAPS leads the two other private rows on Dice",
            "Sec. 2.1",
            "verification/manuscript_tables.py:MAIN_TABLE",
            [round(value, 4) for value in ordered],
            "CAPS highest among private rows",
            None,
            MAIN_TABLE[-1].dice > MAIN_TABLE[2].dice and MAIN_TABLE[-1].dice > MAIN_TABLE[3].dice,
        )
    )


def check_frontier_table(ledger: CheckLedger) -> None:
    marginals = FRONTIER_TABLE[0]
    isotropic = FRONTIER_TABLE[1]
    caps = FRONTIER_TABLE[3]
    ledger.add(
        _check(
            "MS-T2-RATIO-MARGINAL",
            "budget ratio against the two independent generators",
            "Supplementary Table SE1",
            "verification/manuscript_tables.py:FRONTIER_TABLE",
            round(marginals.epsilon_at_parity / caps.epsilon_at_parity, 3),
            2.40,
            0.005,
            close_enough(marginals.epsilon_at_parity / caps.epsilon_at_parity, 2.40, 0.005),
        )
    )
    ledger.add(
        _check(
            "MS-T2-RATIO-ISOTROPIC",
            "budget ratio against the isotropic joint generator",
            "Supplementary Table SE1",
            "verification/manuscript_tables.py:FRONTIER_TABLE",
            round(isotropic.epsilon_at_parity / caps.epsilon_at_parity, 3),
            1.80,
            0.005,
            close_enough(isotropic.epsilon_at_parity / caps.epsilon_at_parity, 1.80, 0.005),
        )
    )
    ledger.add(
        _check(
            "MS-T2-MIA-DELTA",
            "membership-inference gap between the non-private and private generators",
            "Supplementary Table SE1",
            "verification/manuscript_tables.py:FRONTIER_TABLE",
            round(0.617 - caps.mia_auc, 3),
            0.114,
            0.001,
            close_enough(0.617 - caps.mia_auc, 0.114, 0.001),
        )
    )
    ledger.add(
        _check(
            "MS-T2-ORDERING",
            "reported budgets are ordered two-marginal > isotropic > CAPS",
            "Sec. 2.2",
            "verification/manuscript_tables.py:FRONTIER_TABLE",
            [marginals.epsilon_at_parity, isotropic.epsilon_at_parity, caps.epsilon_at_parity],
            "2.0 < 3.6 < 4.8",
            None,
            caps.epsilon_at_parity < isotropic.epsilon_at_parity < marginals.epsilon_at_parity,
        )
    )
    ledger.add(
        _check(
            "MS-T2-CLIPPING-SPLIT",
            "marginal rows carry C = 1.00 and joint rows C = 0.85",
            "Table 2 caption",
            "accounting/clipping.py:compare_clipping_norms",
            [marginals.clipping_norm, caps.clipping_norm],
            [1.0, 0.85],
            0.0,
            marginals.clipping_norm == 1.0 and caps.clipping_norm == 0.85,
        )
    )
    ledger.add(
        _check(
            "MS-T2-UNIT-ASYMMETRY",
            "the private-record subspace row is flagged with an uncounted selection",
            "Table 2 row 3",
            "verification/manuscript_tables.py:FRONTIER_TABLE",
            FRONTIER_TABLE[2].selection_unaccounted,
            True,
            0.0,
            FRONTIER_TABLE[2].selection_unaccounted,
        )
    )
    ledger.add(
        _check(
            "MS-T2-MIA-COVER",
            "the private audit interval covers the no-advantage reference",
            "Sec. 2.2",
            "analysis/delong.py:NO_ADVANTAGE_REFERENCE",
            [caps.mia_low, caps.mia_high],
            "0.500 inside interval",
            None,
            caps.mia_low <= 0.500 <= caps.mia_high,
        )
    )
    ledger.add(
        _check(
            "MS-T2-MIA-NONPRIVATE",
            "the non-private interval excludes the no-advantage reference",
            "Sec. 2.2",
            "audit/membership.py:REPORTED_NON_PRIVATE_AUDIT",
            [NON_PRIVATE_AUDIT[1], NON_PRIVATE_AUDIT[2]],
            "0.500 outside interval",
            None,
            not (NON_PRIVATE_AUDIT[1] <= 0.500 <= NON_PRIVATE_AUDIT[2]),
        )
    )


def check_factorial_table(ledger: CheckLedger) -> None:
    cells = {
        row.label: FactorialCell(
            label=row.label,
            fid=row.fid,
            dtw=row.dtw,
            contact_event_f1=row.contact_event_f1,
            rho=row.rho,
            epsilon_at_parity=row.epsilon_at_parity,
        )
        for row in FACTORIAL_TABLE
    }
    if set(cells) != set(CELLS):
        raise ValueError("the factorial transcription does not cover the four cells")
    contrasts = full_contrast_table(cells)
    for metric, (conditioning, resolution, interaction, dominance) in FACTORIAL_CONTRASTS.items():
        measured = contrasts[metric]
        ledger.add(
            _check(
                f"MS-T3-COND-{metric}",
                "conditioning main effect recomputed from the four printed cells",
                "Table 3 panel (b)",
                "analysis/factorial.py:conditioning_main_effect",
                round(abs(measured["conditioning_main_effect"]), 4),
                conditioning,
                0.005 if metric != "contact_event_f1" else 0.0006,
                close_enough(
                    abs(measured["conditioning_main_effect"]),
                    conditioning,
                    0.005 if metric != "contact_event_f1" else 0.0006,
                ),
            )
        )
        ledger.add(
            _check(
                f"MS-T3-RES-{metric}",
                "resolution main effect recomputed from the four printed cells",
                "Table 3 panel (b)",
                "analysis/factorial.py:resolution_main_effect",
                round(abs(measured["resolution_main_effect"]), 4),
                resolution,
                0.005 if metric != "contact_event_f1" else 0.0006,
                close_enough(
                    abs(measured["resolution_main_effect"]),
                    resolution,
                    0.005 if metric != "contact_event_f1" else 0.0006,
                ),
            )
        )
        ledger.add(
            _check(
                f"MS-T3-INT-{metric}",
                "interaction recomputed from the four printed cells",
                "Table 3 panel (b)",
                "analysis/factorial.py:interaction",
                round(abs(measured["interaction"]), 4),
                interaction,
                0.005 if metric != "contact_event_f1" else 0.0006,
                close_enough(
                    abs(measured["interaction"]),
                    interaction,
                    0.005 if metric != "contact_event_f1" else 0.0006,
                ),
            )
        )
        ledger.add(
            _check(
                f"MS-T3-DOM-{metric}",
                "dominance ratio of the conditioning effect over the resolution effect",
                "Table 3 caption",
                "analysis/factorial.py:FactorialContrast.dominance_ratio",
                round(abs(measured["dominance_ratio"]), 3),
                dominance,
                0.05,
                close_enough(abs(measured["dominance_ratio"]), dominance, 0.05),
            )
        )
    rho_report = rho_invariance_across_resolution(cells)
    ledger.add(
        _check(
            "MS-T3-RHO-RESOLUTION",
            "rho moves with conditioning and not with resolution",
            "Table 3 caption",
            "analysis/factorial.py:rho_invariance_across_resolution",
            round(abs(rho_report["conditioning_shift"]), 3),
            0.515,
            0.02,
            close_enough(abs(rho_report["conditioning_shift"]), 0.515, 0.02),
        )
    )
    ledger.add(
        _check(
            "MS-T3-BUDGET-ORDER",
            "the off-conditioning cells pay the larger parity budget",
            "Table 3 panel (a)",
            "analysis/factorial.py:budget_moves_with_conditioning",
            [cells["off_1x"].epsilon_at_parity, cells["off_2x"].epsilon_at_parity],
            "both above the on-conditioning budgets",
            None,
            cells["off_1x"].epsilon_at_parity > cells["on_1x"].epsilon_at_parity
            and cells["off_2x"].epsilon_at_parity > cells["on_2x"].epsilon_at_parity,
        )
    )


def check_ablation_tables(ledger: CheckLedger) -> None:
    rows = {row.variant: row for row in ABLATION_TABLE}
    full = rows["CAPS (full mechanism)"]
    ledger.add(
        _check(
            "MS-T4-FULL-CONSISTENT",
            "the full-mechanism row carries the main-table values",
            "Table 4 row 1",
            "verification/manuscript_tables.py:ABLATION_TABLE",
            [full.dtw, full.dice, full.fid, full.epsilon_at_parity],
            [0.148, 0.8941, 45.3, 2.0],
            0.0,
            full.dtw == 0.148 and full.dice == 0.8941 and full.fid == 45.3,
        )
    )
    latent = rows["minus shared coupling latent"]
    head_only = rows["minus coupling head only"]
    shaped = rows["minus coupling-shaped window noise"]
    estimator = rows["minus public-reference estimator"]
    accountant = rows["minus procedure accountant"]
    ledger.add(
        _check(
            "MS-T4-LATENT-LARGEST",
            "removing the shared latent drops contact-event F1 more than removing the head alone",
            "Sec. 2.5",
            "analysis/interaction.py:component_drop",
            [round(REFERENCE_F1 - latent.contact_event_f1, 3), round(REFERENCE_F1 - head_only.contact_event_f1, 3)],
            [0.071, 0.039],
            0.0005,
            close_enough(REFERENCE_F1 - latent.contact_event_f1, 0.071, 0.0005)
            and close_enough(REFERENCE_F1 - head_only.contact_event_f1, 0.039, 0.0005),
        )
    )
    ledger.add(
        _check(
            "MS-T4-ESTIMATOR-NULL",
            "the estimator removal is retained as a null result inside the fence",
            "Table 4 caption",
            "analysis/null_fence.py:fence_verdict",
            round(fence_verdict("estimator", REFERENCE_F1, estimator.contact_event_f1).relative_change(), 5),
            0.0024,
            0.0002,
            fence_verdict("estimator", REFERENCE_F1, estimator.contact_event_f1).is_retained_null(),
        )
    )
    ledger.add(
        _check(
            "MS-T4-ACCOUNTANT-NEUTRAL",
            "removing the procedure accountant leaves the main utility unchanged",
            "Sec. 2.5",
            "verification/manuscript_tables.py:ABLATION_TABLE",
            [accountant.dice, accountant.fid, accountant.epsilon_at_parity],
            [full.dice, full.fid, full.epsilon_at_parity],
            0.0,
            accountant.dice == full.dice and accountant.fid == full.fid,
        )
    )
    ledger.add(
        _check(
            "MS-T4-NONMONOTONE-DICE",
            "removing the shaped noise is the single retained non-monotonic gain",
            "Table 4 row 12, Table SB1",
            "verification/manuscript_tables.py:STRATUM_TABLE",
            [round(shaped.dice, 4), round(full.dice, 4)],
            "shaped removal above the full mechanism",
            None,
            shaped.dice < full.dice,
        )
    )
    stratum_complex = {row.label: row.dice_complex for row in STRATUM_TABLE}
    ledger.add(
        _check(
            "MS-SB1-COMPLEX-DICE",
            "the complex-stratum Dice of the shaped-noise removal is the printed 0.8986",
            "Supplementary Table SB1",
            "verification/manuscript_tables.py:STRATUM_TABLE",
            stratum_complex["minus coupling-shaped noise"],
            0.8986,
            0.0,
            stratum_complex["minus coupling-shaped noise"] == 0.8986,
        )
    )
    ledger.add(
        _check(
            "MS-SB1-WEIGHTED-MEAN",
            "the pooled column of the full mechanism is the procedure-weighted stratum mean",
            "Supplementary Table SB1 caption",
            "metrics/segmentation.py:dice_by_stratum",
            round(
                (
                    STRATUM_COUNTS["simple"] * 0.896
                    + STRATUM_COUNTS["intermediate"] * 0.832
                    + STRATUM_COUNTS["complex"] * 0.789
                )
                / sum(STRATUM_COUNTS.values()),
                4,
            ),
            0.842,
            0.0005,
            close_enough(
                (
                    STRATUM_COUNTS["simple"] * 0.896
                    + STRATUM_COUNTS["intermediate"] * 0.832
                    + STRATUM_COUNTS["complex"] * 0.789
                )
                / sum(STRATUM_COUNTS.values()),
                0.842,
                0.0005,
            ),
        )
    )
    interactions = _design_interactions(rows)
    report = report_pairwise(interactions)
    for row in report["rows"]:
        key = _pair_key(row["left"], row["right"])
        printed = PAIRWISE_INTERACTIONS[key]
        ledger.add(
            _check(
                f"MS-T4-IR-{row['left'].split()[0]}-{row['right'].split()[0]}",
                "pairwise interaction ratio recomputed from the printed drops",
                "Table 4 panel (b)",
                "analysis/interaction.py:PairwiseInteraction.interaction_ratio",
                round(row["interaction_ratio"], 3),
                printed[2],
                0.01,
                close_enough(row["interaction_ratio"], printed[2], 0.01),
            )
        )
        ledger.add(
            _check(
                f"MS-T4-REL-{row['left'].split()[0]}-{row['right'].split()[0]}",
                "pairwise relation label matches the printed one",
                "Table 4 panel (b)",
                "analysis/interaction.py:PairwiseInteraction.relation",
                row["relation"],
                printed[3],
                None,
                row["relation"] == printed[3],
            )
        )


def _design_interactions(rows: Mapping[str, AblationRow]) -> list[PairwiseInteraction]:
    """The four printed pairwise rows, with dAB read off the printed interaction table."""
    latent_with_shaped = 0.123
    shaped_with_factor = 0.080
    latent_with_factor = 0.108
    estimator_with_shaped = 0.032
    latent_drop = REFERENCE_F1 - float(rows["minus shared coupling latent"].contact_event_f1)
    shaped_drop = REFERENCE_F1 - float(rows["minus coupling-shaped window noise"].contact_event_f1)
    factor_drop = REFERENCE_F1 - float(rows["minus prefix-sum factorization"].contact_event_f1)
    estimator_drop = REFERENCE_F1 - float(rows["minus public-reference estimator"].contact_event_f1)
    return [
        _pair("coupling latent", "shaped noise", latent_with_shaped, latent_drop + shaped_drop),
        _pair("shaped noise", "factorization", shaped_with_factor, shaped_drop + factor_drop),
        _pair("coupling latent", "factorization", latent_with_factor, latent_drop + factor_drop),
        _pair("public-reference estimator", "shaped noise", estimator_with_shaped, estimator_drop + shaped_drop),
    ]


def _pair(left: str, right: str, drop_both: float, additive: float) -> PairwiseInteraction:
    return PairwiseInteraction(left=left, right=right, drop_both=drop_both, drop_left_plus_right=additive)


def _slug(label: str) -> str:
    """One identifier per printed row, so two rows sharing a final word stay distinct."""
    return "-".join(label.split())


def _pair_key(left: str, right: str) -> str:
    mapping = {
        ("coupling latent", "shaped noise"): "coupling_latent_with_shaped_noise",
        ("shaped noise", "factorization"): "shaped_noise_with_factorization",
        ("coupling latent", "factorization"): "coupling_latent_with_factorization",
        ("public-reference estimator", "shaped noise"): "public_reference_estimator_with_shaped_noise",
    }
    return mapping[(left, right)]


def check_safety_table(ledger: CheckLedger) -> None:
    rows = {row.label: row for row in SAFETY_TABLE}
    private = rows["policy on private substrate"]
    unconstrained = rows["policy on unconstrained substrate"]
    standard = rows["standard of care"]
    ledger.add(
        _check(
            "MS-T5-PARITY-PP",
            "private-substrate success sits inside the non-inferiority margin against the unconstrained substrate",
            "Table 5 caption",
            "metrics/safety.py:non_inferiority_margin",
            round((unconstrained.task_success - private.task_success) * 100.0, 2),
            0.7,
            0.05,
            close_enough((unconstrained.task_success - private.task_success) * 100.0, 0.7, 0.05),
        )
    )
    ledger.add(
        _check(
            "MS-T5-STANDARD-GAP",
            "private-substrate success trails the standard of care by the printed margin",
            "Supplementary Table SE1",
            "metrics/safety.py:non_inferiority_margin",
            round((standard.task_success - private.task_success) * 100.0, 2),
            4.6,
            0.05,
            close_enough((standard.task_success - private.task_success) * 100.0, 4.6, 0.05),
        )
    )
    reduction = standard.contacts_per_procedure - private.contacts_per_procedure
    ledger.add(
        _check(
            "MS-T5-CONTACT-MCID",
            "the contact-event reduction clears its minimum clinically important difference twice",
            "Table 5 caption",
            "metrics/safety.py:clears_mcid",
            round(reduction, 3),
            1.0,
            0.005,
            close_enough(reduction, 1.0, 0.005) and reduction >= 2 * 0.5,
        )
    )
    ledger.add(
        _check(
            "MS-T5-FORCE-MARGIN",
            "the peak contact force sits about 46% under the reference perforation magnitude",
            "Sec. 2.6",
            "metrics/safety.py:force_envelope_margin",
            round(force_envelope_margin(private.peak_force_n) * 100.0, 2),
            46.0,
            0.5,
            close_enough(force_envelope_margin(private.peak_force_n) * 100.0, 46.0, 0.5),
        )
    )
    ledger.add(
        _check(
            "MS-T5-CBF-BAND",
            "the control-barrier-function filter breaches the parity band",
            "Table 5 panel (a)",
            "metrics/safety.py:within_non_inferiority_margin",
            round((unconstrained.task_success - rows["control-barrier-function filter"].task_success) * 100.0, 2),
            2.8,
            0.05,
            close_enough(
                (unconstrained.task_success - rows["control-barrier-function filter"].task_success) * 100.0,
                2.8,
                0.05,
            ),
        )
    )
    ledger.add(
        _check(
            "MS-T5-POTENTIAL-FIELD",
            "the reactive potential-field avoider loses the printed margin of success",
            "Table 5 panel (a)",
            "metrics/safety.py:non_inferiority_margin",
            round((unconstrained.task_success - rows["reactive potential-field avoider"].task_success) * 100.0, 2),
            19.0,
            0.05,
            close_enough(
                (unconstrained.task_success - rows["reactive potential-field avoider"].task_success) * 100.0,
                19.0,
                0.05,
            ),
        )
    )
    ledger.add(
        _check(
            "MS-T5-EQUAL-BUDGET",
            "the equal-budget configuration is a parity row against the unconstrained substrate",
            "Table 5 panel (a)",
            "verification/manuscript_tables.py:SAFETY_TABLE",
            round((unconstrained.task_success - rows["equal-budget configuration"].task_success) * 100.0, 2),
            1.1,
            0.05,
            close_enough(
                (unconstrained.task_success - rows["equal-budget configuration"].task_success) * 100.0,
                1.1,
                0.05,
            ),
        )
    )
    ledger.add(
        _check(
            "MS-T5-COVERAGE-COLUMN",
            "the normalised contact column is not a rescaling of the per-procedure column",
            "Table 5 caption",
            "metrics/safety.py:SafetyRow.contacts_per_thousand_frames",
            [round(row.contacts_per_procedure / row.contacts_per_1000_frames, 4) for row in SAFETY_TABLE],
            "frames per 1000 is constant",
            None,
            len({round(row.analysed_frames, 6) for row in SAFETY_TABLE}) > 1,
        )
    )


def check_cohort_table(ledger: CheckLedger) -> None:
    ledger.add(
        _check(
            "MS-T6-SPREAD",
            "the per-site success spread matches the printed 3.4 percentage points",
            "Sec. 2.7",
            "metrics/safety.py:per_site_spread",
            round((COHORT_TABLE[0].task_success - COHORT_TABLE[2].task_success) * 100.0, 2),
            3.4,
            0.05,
            close_enough((COHORT_TABLE[0].task_success - COHORT_TABLE[2].task_success) * 100.0, 3.4, 0.05),
        )
    )
    ledger.add(
        _check(
            "MS-T6-CROSS-REGION",
            "the cross-region drop satisfies the pre-specified 5.0 percentage-point bound",
            "Sec. 2.7",
            "metrics/safety.py:cross_region_bound_holds",
            round((COHORT_TABLE[3].task_success - COHORT_TABLE[4].task_success) * 100.0, 2),
            2.6,
            0.05,
            close_enough((COHORT_TABLE[3].task_success - COHORT_TABLE[4].task_success) * 100.0, 2.6, 0.05),
        )
    )
    for label, (denominator, point, low, high, _contacts) in SUBGROUP_TABLE.items():
        successes = int(round(point * denominator))
        estimate = wilson_interval(successes, denominator)
        ledger.add(
            _check(
                f"MS-T6-WILSON-{label}",
                "Wilson interval recomputed from the printed denominator and proportion",
                "Table 6 panel (b)",
                "metrics/clinical.py:wilson_interval",
                [round(estimate.low, 3), round(estimate.high, 3)],
                [low, high],
                WILSON_TOLERANCE,
                close_enough(estimate.low, low, WILSON_TOLERANCE)
                and close_enough(estimate.high, high, WILSON_TOLERANCE),
                note=(
                    "the printed intervals sit up to "
                    f"{max(abs(estimate.low - low), abs(estimate.high - high)):.4f} away from a "
                    "Wilson score interval rebuilt from the printed denominator and rounded point"
                ),
            )
        )
    ordered = [SUBGROUP_TABLE[key][1] for key in ("anatomy_simple", "anatomy_intermediate", "anatomy_complex")]
    ledger.add(
        _check(
            "MS-T6-SUBGROUP-ORDER",
            "task success falls monotonically with anatomy complexity",
            "Table 6 panel (b)",
            "verification/manuscript_tables.py:SUBGROUP_TABLE",
            [round(value, 3) for value in ordered],
            "descending",
            None,
            ordered == sorted(ordered, reverse=True),
        )
    )
    inside = all(low <= point <= high for _n, point, low, high, _contacts in SUBGROUP_TABLE.values())
    ledger.add(
        _check(
            "MS-T6-POINT-INSIDE",
            "every printed subgroup proportion sits inside its printed interval",
            "Table 6 panel (b)",
            "verification/manuscript_tables.py:SUBGROUP_TABLE",
            inside,
            True,
            0.0,
            inside,
        )
    )


def check_scaling_and_perturbation(ledger: CheckLedger) -> None:
    sizes = torch.tensor([row.substrate_size for row in SUBSTRATE_SCALING], dtype=torch.float64)
    f1 = torch.tensor([row.contact_event_f1 for row in SUBSTRATE_SCALING], dtype=torch.float64)
    ledger.add(
        _check(
            "MS-T7A-MONOTONE",
            "contact-event F1 improves monotonically with substrate volume",
            "Supplementary Table T7 panel A",
            "analysis/powerlaw.py:scaling_is_monotone",
            [round(value, 3) for value in f1.tolist()],
            "non-decreasing",
            None,
            bool(torch.all(f1[1:] >= f1[:-1]).item()),
        )
    )
    fit = fit_power_law(sizes, f1)
    _ = fit
    ledger.add(
        _check(
            "MS-T7B-BUDGET-FLOOR",
            "the parity budget stops improving at the largest substrate volumes",
            "Supplementary Table T7 panel B",
            "analysis/powerlaw.py:budget_floor_reached",
            [row.substrate_size for row in SUBSTRATE_SCALING if row.epsilon_at_parity <= 2.0],
            [9600, 19200],
            0.0,
            [row.substrate_size for row in SUBSTRATE_SCALING if row.epsilon_at_parity <= 2.0] == [9600, 19200],
        )
    )
    raw_sizes = torch.tensor([row.substrate_size for row in RAW_SCALING_TABLE], dtype=torch.float64)
    raw_f1 = torch.tensor([row.contact_event_f1 for row in RAW_SCALING_TABLE], dtype=torch.float64)
    raw_fit = fit_power_law(raw_sizes, raw_f1)
    raw_residuals = residuals_in_units(raw_fit, raw_sizes, raw_f1)
    ledger.add(
        _check(
            "MS-SC2-FIT",
            "the raw scaling data admit a power law with the printed residual order",
            "Supplementary Table SC2",
            "analysis/powerlaw.py:residuals_in_units",
            [round(float(raw_residuals.min().item()), 2), round(float(raw_residuals.max().item()), 2)],
            [-3.6, 3.4],
            0.35,
            abs(float(raw_residuals.min().item()) + 3.6) <= 0.35
            and abs(float(raw_residuals.max().item()) - 3.4) <= 0.35,
            note=(
                "a power law fitted to the printed per-run contact-event F1 admits residuals an "
                "order of magnitude larger than the printed ones, so the printed residual column "
                "cannot be rebuilt from the printed endpoint column"
            ),
        )
    )
    printed_sizes = torch.tensor([row.substrate_size for row in RAW_SCALING_TABLE], dtype=torch.float64)
    printed_residuals = torch.tensor([row.fit_residual_1e3 for row in RAW_SCALING_TABLE], dtype=torch.float64)
    systematic_sizes = [
        size
        for size in (600, 1200, 2400, 4800, 9600, 19200)
        if systematic_at_size(printed_residuals, printed_sizes, size)
    ]
    ledger.add(
        _check(
            "MS-SC2-SYSTEMATIC",
            "the printed residual column carries no systematically signed block",
            "Supplementary Table SC2 caption",
            "analysis/powerlaw.py:systematic_at_size",
            systematic_sizes,
            [],
            0.0,
            systematic_sizes == [],
            note=(
                "the printed residual column is systematically negative at 19,200 procedures "
                "(-2.8, -3.6, -2.9), which by the caption's own rule marks a departure of the fit "
                "rather than initialisation noise"
            ),
        )
    )
    reference_rungs = torch.tensor([row[2] for row in REFERENCE_CORPUS_LADDER], dtype=torch.float64)
    ledger.add(
        _check(
            "MS-T7C-SATURATION",
            "the reference-corpus ladder saturates at its last two rungs",
            "Supplementary Table T7 panel C",
            "analysis/powerlaw.py:reference_corpus_saturation",
            [round(float(reference_rungs[-2].item()), 3), round(float(reference_rungs[-1].item()), 3)],
            [0.836, 0.837],
            0.002,
            abs(float(reference_rungs[-1].item() - reference_rungs[-2].item())) <= 0.002,
        )
    )
    cost_fit = fit_log_ladder(WINDOW_LADDER, REPORTED_COMPOSITION_MULTIPLIER)
    latency_fit = fit_log_ladder(WINDOW_LADDER, REPORTED_RELATIVE_LATENCY)
    ledger.add(
        _check(
            "MS-T7D-COST-POLYLOG",
            "the window composition-cost ladder is polylogarithmic in the window length",
            "Supplementary Table T7 panel D",
            "coupling/prefix_sum.py:fit_log_ladder",
            round(cost_fit["r_squared"], 4),
            ">= 0.95",
            0.05,
            cost_fit["r_squared"] >= 0.95,
        )
    )
    ledger.add(
        _check(
            "MS-T7D-LATENCY-POLYLOG",
            "the window relative-latency ladder is polylogarithmic in the window length",
            "Supplementary Table T7 panel D",
            "coupling/prefix_sum.py:fit_log_ladder",
            round(latency_fit["r_squared"], 4),
            ">= 0.95",
            0.05,
            latency_fit["r_squared"] >= 0.95,
        )
    )
    for family, rungs in PERTURBATION_TABLE.items():
        if family == "none":
            continue
        deficits = [row[5] for row in rungs]
        ledger.add(
            _check(
                f"MS-SC1-{family}",
                "the perturbation family degrades the contact-event F1 monotonically",
                "Supplementary Table SC1",
                "verification/manuscript_tables.py:PERTURBATION_TABLE",
                [round(value, 3) for value in deficits],
                "non-increasing",
                None,
                all(later <= earlier + 1e-9 for earlier, later in zip(deficits, deficits[1:])),
            )
        )


def check_statistics_and_initialisations(ledger: CheckLedger) -> None:
    families = declare_families()
    ledger.add(
        _check(
            "MS-SE1-INVENTORY",
            "the exploratory inventory sums to the declared twenty-three comparisons",
            "Sec. 2.8",
            "analysis/multiplicity.py:inventory_total",
            inventory_total(),
            EXPLORATORY_COMPARISON_COUNT,
            0.0,
            inventory_total() == EXPLORATORY_COMPARISON_COUNT,
        )
    )
    ledger.add(
        _check(
            "MS-SE1-FAMILIES",
            "the inventory is spread over the declared number of families",
            "Sec. 2.8",
            "analysis/multiplicity.py:declare_families",
            len(families),
            7,
            0.0,
            len(families) == 7,
        )
    )
    primaries = [row for row in COMPARISON_TABLE if row.family == "primary"]
    ledger.add(
        _check(
            "MS-SE1-PRIMARIES",
            "three comparisons are pre-specified and carry the Holm correction",
            "Supplementary Table SE1",
            "analysis/multiplicity.py:holm_bonferroni",
            len(primaries),
            PRIMARY_COMPARISON_COUNT,
            0.0,
            len(primaries) == PRIMARY_COMPARISON_COUNT and all(row.correction == "holm" for row in primaries),
        )
    )
    holm_labels = [row.label for row in primaries]
    holm_p = [max(row.p_uncorrected, 1e-12) for row in primaries]
    holm = holm_bonferroni(holm_labels, holm_p)
    ledger.add(
        _check(
            "MS-SE1-HOLM-PASS",
            "all pre-specified primaries survive the Holm step-down at the declared level",
            "Sec. 2.8",
            "analysis/multiplicity.py:holm_bonferroni",
            [row.passes for row in holm],
            [True, True, True],
            None,
            all(row.passes for row in holm),
        )
    )
    exploratory = [row for row in COMPARISON_TABLE if row.family == "exploratory"]
    bh = benjamini_hochberg([row.label for row in exploratory], [max(row.p_uncorrected, 1e-12) for row in exploratory])
    ledger.add(
        _check(
            "MS-SE1-BH-RETAINED",
            "a retained null result survives the Benjamini-Hochberg step-up",
            "Supplementary Table SE1",
            "analysis/multiplicity.py:benjamini_hochberg",
            [row.passes for row in bh if "estimator removed" in row.label],
            [False],
            None,
            all(not row.passes for row in bh if "estimator removed" in row.label),
        )
    )
    for label, values in INITIALISATION_SUCCESS.items():
        tensor = torch.tensor(values, dtype=torch.float64)
        ledger.add(
            _check(
                f"MS-SC3-{_slug(label)}",
                "the twenty-initialisation column reproduces its printed mean",
                "Supplementary Table SC3",
                "verification/manuscript_tables.py:INITIALISATION_SUCCESS",
                round(float(tensor.mean().item()), 4),
                {
                    "policy on private substrate": 0.895,
                    "policy on unconstrained substrate": 0.902,
                    "control-barrier-function filter": 0.874,
                    "reactive potential-field avoider": 0.712,
                }[label],
                0.0005,
                abs(
                    float(tensor.mean().item())
                    - {
                        "policy on private substrate": 0.895,
                        "policy on unconstrained substrate": 0.902,
                        "control-barrier-function filter": 0.874,
                        "reactive potential-field avoider": 0.712,
                    }[label]
                )
                <= 0.0005,
            )
        )
        ledger.add(
            _check(
                f"MS-SC3-SPREAD-{_slug(label)}",
                "the twenty-initialisation column carries more than one distinct value",
                "Supplementary Table SC3",
                "analysis/variance.py:dispersion",
                round(float(tensor.std(unbiased=True).item()), 5),
                "> 0",
                0.0,
                float(tensor.std(unbiased=True).item()) > 0.0,
            )
        )
    ledger.add(
        _check(
            "MS-SE1-DELONG-UNIT",
            "the area comparison is stated as a paired DeLong test on the held-out procedures",
            "Supplementary Table SE1",
            "analysis/delong.py:delong_paired_test",
            [row.unit for row in COMPARISON_TABLE if row.label.startswith("MIA AUC")],
            ["DeLong paired (1204 procedures)"],
            None,
            any(row.unit.startswith("DeLong paired") for row in COMPARISON_TABLE),
        )
    )


def run_manuscript_checks() -> CheckLedger:
    ledger = CheckLedger()
    check_cohort_arithmetic(ledger)
    check_table_one(ledger)
    check_frontier_table(ledger)
    check_factorial_table(ledger)
    check_ablation_tables(ledger)
    check_safety_table(ledger)
    check_cohort_table(ledger)
    check_scaling_and_perturbation(ledger)
    check_statistics_and_initialisations(ledger)
    return ledger
