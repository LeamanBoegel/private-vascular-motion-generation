"""Values transcribed from the manuscript, used only to check its internal arithmetic.

Ref: Tables 1-6 of the main body and Supplementary Tables SE1, T7, SB1, SC1, SC2, SC3.
Every entry here is a transcription of a printed number, never a measured result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

MAIN_TABLE_SIZE: int = 1204
SUBSTRATE_VOLUME: int = 4800


@dataclass(frozen=True)
class MainRow:
    label: str
    fid: float
    kid: float
    ssim: float
    dice: float
    ade_mm: float
    fde_mm: float
    dtw: float
    path_efficiency: float
    delta_versus_unconstrained: str
    private: bool


MAIN_TABLE: tuple[MainRow, ...] = (
    MainRow("unconstrained joint generator", 44.9, 0.0141, 0.712, 0.8964, 2.9, 5.3, 0.145, 0.79, "reference", False),
    MainRow(
        "unconstrained decoupled pair",
        45.6,
        0.0152,
        0.704,
        0.8891,
        3.2,
        5.9,
        0.163,
        0.77,
        "+0.7 FID (+1.6% rel.)",
        False,
    ),
    MainRow(
        "two independent private generators",
        47.1,
        0.0163,
        0.689,
        0.8712,
        3.4,
        6.2,
        0.171,
        0.74,
        "+2.2 FID (+4.9% rel.)",
        True,
    ),
    MainRow(
        "joint private, isotropic window noise",
        46.4,
        0.0159,
        0.693,
        0.8786,
        3.3,
        6.0,
        0.166,
        0.75,
        "+1.5 FID (+3.3% rel.)",
        True,
    ),
    MainRow("CAPS", 45.3, 0.0146, 0.708, 0.8941, 3.0, 5.4, 0.148, 0.78, "+0.4 FID (+0.9% rel.), n.s.", True),
)


@dataclass(frozen=True)
class FrontierRow:
    label: str
    epsilon_at_parity: float
    clipping_norm: float
    sigma_parallel: float
    sigma_perpendicular: float
    rank: int
    rho: float | None
    mia_auc: float
    mia_low: float
    mia_high: float
    selection_unaccounted: bool = False


FRONTIER_TABLE: tuple[FrontierRow, ...] = (
    FrontierRow("two independent private generators", 4.8, 1.00, 4.6, 4.6, 0, None, 0.541, 0.523, 0.559),
    FrontierRow("joint, isotropic window noise", 3.6, 0.85, 3.9, 3.9, 12, 0.41, 0.528, 0.511, 0.545),
    FrontierRow("joint, subspace from private records", 2.0, 0.85, 5.4, 2.6, 12, 0.20, 0.511, 0.495, 0.527, True),
    FrontierRow("CAPS", 2.0, 0.85, 5.4, 2.6, 12, 0.19, 0.503, 0.487, 0.519),
)

NON_PRIVATE_AUDIT: tuple[float, float, float] = (0.617, 0.598, 0.636)


@dataclass(frozen=True)
class FactorialCellRow:
    label: str
    fid: float
    dtw: float
    contact_event_f1: float
    rho: float
    epsilon_at_parity: float


FACTORIAL_TABLE: tuple[FactorialCellRow, ...] = (
    FactorialCellRow("on_1x", 45.3, 0.148, 0.842, 0.19, 2.0),
    FactorialCellRow("on_2x", 45.9, 0.156, 0.833, 0.19, 2.1),
    FactorialCellRow("off_1x", 52.4, 0.233, 0.762, 0.71, 5.6),
    FactorialCellRow("off_2x", 58.1, 0.271, 0.744, 0.70, 6.2),
)

FACTORIAL_CONTRASTS: Mapping[str, tuple[float, float, float, float]] = {
    "fid": (9.65, 3.15, 5.1, 3.06),
    "dtw": (0.100, 0.023, 0.030, 4.35),
    "contact_event_f1": (0.085, 0.014, 0.009, 6.26),
}


@dataclass(frozen=True)
class AblationRow:
    index: int
    tier: str
    variant: str
    dtw: float
    contact_event_f1: float
    dice: float
    fid: float
    epsilon_at_parity: float


ABLATION_TABLE: tuple[AblationRow, ...] = (
    AblationRow(1, "full", "CAPS (full mechanism)", 0.148, 0.842, 0.8941, 45.3, 2.0),
    AblationRow(2, "0", "kinematic channel permuted across procedures", 0.298, 0.318, 0.8602, 51.9, 9.4),
    AblationRow(3, "0", "frame sequence permuted across procedures", 0.246, 0.301, 0.8427, 59.6, 9.1),
    AblationRow(4, "0", "cohort-mean trajectory head", 0.386, 0.352, 0.8936, 45.4, 5.2),
    AblationRow(5, "0", "nearest-neighbour trajectory retrieval head", 0.207, 0.688, 0.8934, 45.5, 4.6),
    AblationRow(6, "0", "kinematic channels zeroed", 0.284, 0.339, 0.8564, 52.8, 9.7),
    AblationRow(7, "0", "force channel zeroed", 0.171, 0.612, 0.8929, 45.6, 2.9),
    AblationRow(8, "0", "projection-geometry metadata zeroed", 0.178, 0.797, 0.8873, 47.9, 3.3),
    AblationRow(9, "0", "contrast-phase label zeroed", 0.163, 0.826, 0.8917, 46.1, 2.5),
    AblationRow(10, "1", "minus shared coupling latent", 0.221, 0.771, 0.8721, 47.6, 4.4),
    AblationRow(11, "1", "minus coupling head only", 0.186, 0.803, 0.8802, 46.7, 3.9),
    AblationRow(12, "1", "minus coupling-shaped window noise", 0.166, 0.812, 0.8786, 46.4, 3.6),
    AblationRow(13, "1", "minus prefix-sum factorization", 0.163, 0.819, 0.8897, 46.0, 5.1),
    AblationRow(14, "1", "minus public-reference estimator", 0.149, 0.840, 0.8938, 45.4, 2.0),
    AblationRow(15, "1", "minus procedure accountant", 0.148, 0.842, 0.8941, 45.3, 2.0),
    AblationRow(16, "3", "public-reference source, public corpora only", 0.150, 0.839, 0.8936, 45.4, 2.1),
    AblationRow(17, "3", "window ladder at k = 8", 0.168, 0.815, 0.8861, 46.3, 2.6),
    AblationRow(18, "3", "window ladder at k = 64", 0.162, 0.829, 0.8908, 45.8, 5.9),
    AblationRow(19, "ref", "two independent private generators", 0.171, 0.781, 0.8712, 47.1, 4.8),
    AblationRow(20, "ref", "unconstrained joint generator", 0.145, 0.858, 0.8964, 44.9, float("inf")),
)

PAIRWISE_INTERACTIONS: Mapping[str, tuple[float, float, float, str]] = {
    "coupling_latent_with_shaped_noise": (0.123, 0.101, 1.22, "super-additive"),
    "shaped_noise_with_factorization": (0.080, 0.053, 1.51, "super-additive"),
    "coupling_latent_with_factorization": (0.108, 0.094, 1.15, "super-additive"),
    "public_reference_estimator_with_shaped_noise": (0.032, 0.032, 1.01, "prerequisite, not synergy"),
}


@dataclass(frozen=True)
class StratumRow:
    label: str
    f1_simple: float
    f1_intermediate: float
    f1_complex: float
    f1_pooled: float
    dice: float
    dice_complex: float
    dtw: float
    epsilon_at_parity: float


STRATUM_TABLE: tuple[StratumRow, ...] = (
    StratumRow("CAPS", 0.896, 0.832, 0.789, 0.842, 0.8941, 0.8812, 0.148, 2.0),
    StratumRow("minus shared coupling latent", 0.836, 0.776, 0.706, 0.771, 0.8721, 0.8604, 0.221, 4.4),
    StratumRow("minus coupling head only", 0.858, 0.806, 0.749, 0.803, 0.8802, 0.8701, 0.186, 3.9),
    StratumRow("minus coupling-shaped noise", 0.872, 0.815, 0.751, 0.812, 0.8786, 0.8986, 0.166, 3.6),
    StratumRow("minus prefix-sum factorization", 0.869, 0.822, 0.767, 0.819, 0.8897, 0.8772, 0.163, 5.1),
    StratumRow("minus reference estimator", 0.895, 0.841, 0.787, 0.840, 0.8938, 0.8809, 0.149, 2.0),
    StratumRow("minus procedure accountant", 0.896, 0.843, 0.789, 0.842, 0.8941, 0.8812, 0.148, 2.0),
)

STRATUM_COUNTS: Mapping[str, int] = {"simple": 412, "intermediate": 458, "complex": 334}


@dataclass(frozen=True)
class SafetyRowReport:
    label: str
    task_success: float
    contacts_per_procedure: float
    contacts_per_1000_frames: float
    analysed_frames: int
    peak_force_n: float
    force_duration_integral_ns: float
    fluoroscopy_min: float
    contrast_ml: float
    navigation_min: float
    path_efficiency: float
    delta_success_pp: str


SAFETY_TABLE: tuple[SafetyRowReport, ...] = (
    SafetyRowReport(
        "standard of care", 0.941, 2.4, 4.4, 548, 0.58, 0.71, 12.6, 148, 42.8, 0.76, "re-simulated reference"
    ),
    SafetyRowReport(
        "control-barrier-function filter",
        0.874,
        1.5,
        3.0,
        501,
        0.44,
        0.52,
        10.9,
        131,
        37.4,
        0.79,
        "-2.8 pp (parity-band breach)",
    ),
    SafetyRowReport(
        "reactive potential-field avoider", 0.712, 2.1, 3.5, 602, 0.55, 0.66, 14.2, 156, 45.1, 0.71, "-19.0 pp"
    ),
    SafetyRowReport(
        "policy on unconstrained substrate", 0.902, 1.6, 3.3, 487, 0.47, 0.55, 10.4, 126, 35.9, 0.82, "reference"
    ),
    SafetyRowReport(
        "policy on private substrate",
        0.895,
        1.4,
        3.0,
        462,
        0.41,
        0.48,
        9.8,
        121,
        34.6,
        0.81,
        "-0.7 pp (parity, inside CI)",
    ),
    SafetyRowReport(
        "equal-budget configuration", 0.891, 1.5, 3.2, 471, 0.43, 0.50, 10.1, 124, 35.2, 0.80, "-1.1 pp (parity)"
    ),
)

PUBLISHED_COMPARATORS: Mapping[str, tuple[float, int | None, str]] = {
    "strongest 2026 world-model controller": (0.90, 10, "published"),
    "base world-model controller": (0.65, 10, "published"),
    "soft actor-critic": (0.37, None, "reproduced"),
}


@dataclass(frozen=True)
class CohortStratumRow:
    label: str
    procedures: int
    fid: float
    dtw: float
    task_success: float
    success_low: float
    success_high: float
    contacts_per_procedure: float
    peak_force_n: float
    procedure_success: float
    force_coverage: float


COHORT_TABLE: tuple[CohortStratumRow, ...] = (
    CohortStratumRow("Site A (Region I)", 2214, 45.1, 0.146, 0.905, 0.893, 0.917, 1.3, 0.39, 0.948, 0.71),
    CohortStratumRow("Site B (Region I)", 1986, 45.4, 0.149, 0.889, 0.875, 0.903, 1.5, 0.42, 0.936, 0.64),
    CohortStratumRow("Site C (Region II)", 1642, 46.0, 0.157, 0.871, 0.855, 0.887, 1.6, 0.44, 0.921, 0.49),
    CohortStratumRow("Region I (pooled)", 4200, 45.2, 0.147, 0.897, 0.888, 0.906, 1.4, 0.41, 0.942, 0.68),
    CohortStratumRow("Region II (Site C)", 1642, 46.0, 0.157, 0.871, 0.855, 0.887, 1.6, 0.44, 0.921, 0.49),
    CohortStratumRow("retrospective supporting arm", 4106, 45.3, 0.148, 0.892, 0.883, 0.901, 1.4, 0.41, 0.939, 0.61),
    CohortStratumRow("prospective observational arm", 1736, 45.4, 0.150, 0.878, 0.863, 0.893, 1.5, 0.42, 0.928, 0.67),
)

SUBGROUP_TABLE: Mapping[str, tuple[int, float, float, float, float]] = {
    "anatomy_simple": (412, 0.938, 0.912, 0.958, 0.9),
    "anatomy_intermediate": (458, 0.901, 0.874, 0.928, 1.4),
    "anatomy_complex": (334, 0.836, 0.796, 0.876, 2.1),
    "access_transradial": (731, 0.899, 0.877, 0.921, 1.3),
    "access_transfemoral": (473, 0.889, 0.861, 0.917, 1.5),
    "operator_experienced": (648, 0.911, 0.889, 0.933, 1.3),
    "operator_early_career": (556, 0.876, 0.849, 0.903, 1.6),
}


@dataclass(frozen=True)
class ScalingRungRow:
    substrate_size: int
    rho: float
    contact_event_f1: float
    fid: float
    epsilon_at_parity: float


SUBSTRATE_SCALING: tuple[ScalingRungRow, ...] = (
    ScalingRungRow(600, 0.34, 0.774, 48.2, 3.4),
    ScalingRungRow(1200, 0.29, 0.801, 47.1, 2.9),
    ScalingRungRow(2400, 0.24, 0.822, 46.1, 2.4),
    ScalingRungRow(4800, 0.20, 0.836, 45.6, 2.1),
    ScalingRungRow(9600, 0.18, 0.841, 45.4, 2.0),
    ScalingRungRow(19200, 0.17, 0.843, 45.3, 2.0),
)

BUDGET_SWEEP: tuple[tuple[float, float, float, float, float], ...] = (
    (1.0, 0.797, 47.0, 0.26, 0.505),
    (2.0, 0.836, 45.6, 0.20, 0.503),
    (4.0, 0.841, 45.4, 0.21, 0.512),
    (8.0, 0.843, 45.3, 0.22, 0.531),
)

REFERENCE_CORPUS_LADDER: tuple[tuple[int, float, float, float, float], ...] = (
    (2, 0.27, 0.827, 46.0, 2.3),
    (4, 0.23, 0.832, 45.8, 2.2),
    (8, 0.20, 0.836, 45.6, 2.0),
    (16, 0.19, 0.837, 45.6, 2.0),
)

WINDOW_LADDER_TABLE: tuple[tuple[int, float, float, float, float], ...] = (
    (4, 0.0, 1.42, 1.10, 0.828),
    (8, 0.0, 1.68, 1.19, 0.832),
    (16, 0.2, 1.89, 1.28, 0.835),
    (32, 0.6, 2.07, 1.37, 0.836),
    (64, 1.1, 2.21, 1.45, 0.834),
    (128, 2.4, 2.33, 1.53, 0.826),
)


@dataclass(frozen=True)
class RawScalingRow:
    substrate_size: int
    initialisation: int
    contact_event_f1: float
    fid: float
    epsilon_at_parity: float
    fit_residual_1e3: float


RAW_SCALING_TABLE: tuple[RawScalingRow, ...] = (
    RawScalingRow(600, 2, 0.769, 48.4, 3.5, -2.1),
    RawScalingRow(600, 7, 0.777, 47.9, 3.3, 3.4),
    RawScalingRow(600, 11, 0.776, 48.3, 3.4, -0.9),
    RawScalingRow(1200, 2, 0.798, 47.3, 3.0, 1.8),
    RawScalingRow(1200, 7, 0.805, 46.8, 2.8, -1.2),
    RawScalingRow(1200, 11, 0.800, 47.2, 2.9, 0.6),
    RawScalingRow(2400, 2, 0.819, 46.3, 2.5, 2.2),
    RawScalingRow(2400, 7, 0.826, 45.8, 2.3, -2.0),
    RawScalingRow(2400, 11, 0.821, 46.2, 2.4, 1.4),
    RawScalingRow(4800, 2, 0.833, 45.8, 2.2, 0.8),
    RawScalingRow(4800, 7, 0.839, 45.3, 2.0, -1.6),
    RawScalingRow(4800, 11, 0.836, 45.7, 2.1, 1.1),
    RawScalingRow(9600, 2, 0.839, 45.5, 2.1, -1.9),
    RawScalingRow(9600, 7, 0.844, 45.2, 1.9, 2.7),
    RawScalingRow(9600, 11, 0.840, 45.5, 2.0, 0.4),
    RawScalingRow(19200, 2, 0.841, 45.4, 2.0, -2.8),
    RawScalingRow(19200, 7, 0.846, 45.1, 1.9, -3.6),
    RawScalingRow(19200, 11, 0.842, 45.4, 2.1, -2.9),
)


PERTURBATION_TABLE: Mapping[str, tuple[tuple[float, float, float, float, float, float], ...]] = {
    "none": ((0.00, 45.3, 0.148, 0.842, 0.895, 0.000),),
    "intensity": (
        (0.01, 45.5, 0.147, 0.840, 0.893, -0.002),
        (0.02, 45.9, 0.151, 0.836, 0.891, -0.006),
        (0.04, 46.4, 0.159, 0.830, 0.887, -0.012),
        (0.08, 47.2, 0.172, 0.819, 0.878, -0.023),
        (0.16, 48.9, 0.192, 0.798, 0.861, -0.044),
    ),
    "dropout": (
        (0.02, 45.6, 0.153, 0.838, 0.892, -0.004),
        (0.05, 46.0, 0.159, 0.833, 0.888, -0.009),
        (0.10, 46.8, 0.170, 0.824, 0.881, -0.018),
        (0.20, 48.1, 0.189, 0.806, 0.869, -0.036),
        (0.30, 50.2, 0.216, 0.781, 0.846, -0.061),
    ),
    "drift": (
        (20, 45.4, 0.152, 0.839, 0.893, -0.003),
        (50, 45.7, 0.157, 0.835, 0.889, -0.007),
        (100, 46.2, 0.166, 0.828, 0.884, -0.014),
        (200, 47.0, 0.180, 0.815, 0.874, -0.027),
        (400, 48.4, 0.203, 0.793, 0.856, -0.049),
    ),
}

INITIALISATION_SUCCESS: Mapping[str, tuple[float, ...]] = {
    "policy on private substrate": (
        0.912,
        0.901,
        0.887,
        0.900,
        0.894,
        0.879,
        0.905,
        0.898,
        0.883,
        0.891,
        0.909,
        0.876,
        0.896,
        0.902,
        0.888,
        0.914,
        0.899,
        0.881,
        0.893,
        0.892,
    ),
    "policy on unconstrained substrate": (
        0.924,
        0.905,
        0.896,
        0.909,
        0.899,
        0.886,
        0.913,
        0.902,
        0.891,
        0.898,
        0.915,
        0.883,
        0.903,
        0.908,
        0.894,
        0.920,
        0.906,
        0.889,
        0.900,
        0.899,
    ),
    "control-barrier-function filter": (
        0.898,
        0.871,
        0.865,
        0.879,
        0.868,
        0.859,
        0.882,
        0.876,
        0.863,
        0.870,
        0.889,
        0.855,
        0.875,
        0.880,
        0.866,
        0.901,
        0.878,
        0.861,
        0.872,
        0.872,
    ),
    "reactive potential-field avoider": (
        0.745,
        0.703,
        0.698,
        0.716,
        0.706,
        0.706,
        0.722,
        0.713,
        0.700,
        0.708,
        0.725,
        0.700,
        0.711,
        0.718,
        0.702,
        0.732,
        0.719,
        0.706,
        0.707,
        0.703,
    ),
}


@dataclass(frozen=True)
class ComparisonRow:
    label: str
    unit: str
    statistic: float
    p_uncorrected: float
    p_corrected: float
    effect_size: str
    correction: str
    family: str


COMPARISON_TABLE: tuple[ComparisonRow, ...] = (
    ComparisonRow(
        "parity budget CAPS 2.0 vs two marginal 4.8",
        "paired Wilcoxon (20 initialisations)",
        0.0,
        0.0,
        0.0,
        "ratio 2.40x",
        "holm",
        "primary",
    ),
    ComparisonRow(
        "parity budget CAPS 2.0 vs isotropic joint 3.6",
        "paired Wilcoxon (20 initialisations)",
        0.0,
        0.0,
        0.0,
        "ratio 1.80x",
        "holm",
        "primary",
    ),
    ComparisonRow(
        "MIA AUC non-private 0.617 vs private 0.503",
        "DeLong paired (1204 procedures)",
        9.0,
        0.0,
        0.0,
        "delta AUC 0.114",
        "holm",
        "primary",
    ),
    ComparisonRow(
        "contact events private 1.4 vs standard 2.4",
        "paired permutation",
        -7.9,
        0.0,
        0.0,
        "-1.0 event (0.7-1.3)",
        "benjamini_hochberg",
        "exploratory",
    ),
    ComparisonRow(
        "task success standard 0.941 vs private 0.895",
        "two-proportion",
        4.13,
        0.0,
        0.0,
        "-4.6 pp",
        "benjamini_hochberg",
        "exploratory",
    ),
    ComparisonRow(
        "task success complex 0.836 vs simple 0.938",
        "two-proportion",
        4.34,
        0.0,
        0.0,
        "-10.2 pp",
        "benjamini_hochberg",
        "exploratory",
    ),
    ComparisonRow(
        "contact events complex 2.1 vs simple 0.9",
        "permutation",
        6.02,
        0.0,
        0.0,
        "+1.2 events",
        "benjamini_hochberg",
        "exploratory",
    ),
    ComparisonRow(
        "inter-reader kappa with model 0.74 vs alone 0.62",
        "paired bootstrap (9 readers)",
        0.0,
        0.004,
        0.013,
        "delta kappa 0.12",
        "benjamini_hochberg",
        "exploratory",
    ),
    ComparisonRow(
        "reader discrimination 0.863 vs 0.803",
        "paired bootstrap (9 readers)",
        0.0,
        0.006,
        0.017,
        "delta 0.060",
        "benjamini_hochberg",
        "exploratory",
    ),
    ComparisonRow(
        "cross-region task success 0.871 vs 0.897",
        "two-proportion",
        1.24,
        0.215,
        0.55,
        "-2.6 pp",
        "benjamini_hochberg",
        "exploratory",
    ),
    ComparisonRow(
        "procedure success private 0.895 vs unconstrained 0.902",
        "paired permutation",
        -1.06,
        0.29,
        0.67,
        "-0.7 pp (-1.9 to +0.5)",
        "benjamini_hochberg",
        "exploratory",
    ),
    ComparisonRow(
        "factorial interaction contact-event F1",
        "two-way factorial",
        0.009,
        0.37,
        0.71,
        "0.009 (about 0.9 SD)",
        "benjamini_hochberg",
        "exploratory",
    ),
    ComparisonRow(
        "factorial conditioning main effect FID",
        "two-way factorial",
        9.65,
        0.0,
        1.00,
        "9.65 (3.1x resolution)",
        "benjamini_hochberg",
        "exploratory",
    ),
    ComparisonRow(
        "contact-event F1 estimator removed 0.840 vs full 0.842",
        "paired Wilcoxon (20 initialisations)",
        96.0,
        0.68,
        1.00,
        "-0.002 (0.24% rel.)",
        "benjamini_hochberg",
        "exploratory",
    ),
)

EXPLORATORY_COMPARISON_COUNT: int = 23
PRIMARY_COMPARISON_COUNT: int = 3
TOTAL_COHORT: int = 5842
PREFIX_WINDOW_DEFAULT: int = 16
