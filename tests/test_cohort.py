"""Cohort, frame-selection, force-channel and partition tests.

Ref: Sec. 4.2, Sec. 4.3, Sec. 4.10.
"""

from __future__ import annotations

import pytest
import torch

from private_vascular_motion_generation.cohort.force_channel import (
    contacts_per_procedure,
    contacts_per_thousand_frames,
    count_contact_events,
    force_duration_integral,
    peak_force_envelope,
)
from private_vascular_motion_generation.cohort.frames import (
    frame_axis_alignment,
    select_diagnostic_frames,
    transient_ladder,
)
from private_vascular_motion_generation.cohort.partition import (
    ANATOMY_TERTILES,
    FULL_COHORT_SIZE,
    HELD_OUT_SIZE,
    SITE_HELD_OUT,
    SITE_TOTALS,
    anatomy_tertile_assignment,
    default_partition,
    region_pool,
    site_strata,
)
from private_vascular_motion_generation.cohort.procedure import (
    FORCE_CHANNEL_INDEX,
    MOTION_CHANNELS,
    MOTION_DIM,
    ProcedureRecord,
    ProcedureTensorSpec,
    stack_procedures,
)


def test_motion_channel_semantics() -> None:
    assert MOTION_DIM == 6
    assert MOTION_CHANNELS[FORCE_CHANNEL_INDEX] == "axial_force"
    assert MOTION_CHANNELS[:3] == ("tip_x", "tip_y", "tip_z")


def test_tensor_spec_contract(small_fixture) -> None:
    record = small_fixture.records[0]
    spec = ProcedureTensorSpec(height=32, width=32, frames=16)
    assert spec.covers(record.appearance, record.motion)
    assert spec.appearance_shape == (32, 32, 16)


def test_record_rejects_misaligned_frame_axis() -> None:
    with pytest.raises(ValueError):
        ProcedureRecord(
            procedure_id="A::00000",
            site="A",
            region="I",
            appearance=torch.zeros(8, 8, 4),
            motion=torch.zeros(6, 5),
            force_channel_present=False,
        )


def test_stack_requires_shared_extent(small_fixture) -> None:
    records = list(small_fixture.records[:3])
    appearance, motion = stack_procedures(records)
    assert tuple(appearance.shape) == (3, 32, 32, 16)
    assert tuple(motion.shape) == (3, 6, 16)
    with pytest.raises(ValueError):
        stack_procedures([])


def test_partition_is_procedure_level_and_disjoint() -> None:
    partition = default_partition()
    assert len(partition.held_out) == HELD_OUT_SIZE
    assert partition.is_disjoint()
    assert sum(SITE_HELD_OUT.values()) == HELD_OUT_SIZE
    assert partition.region_mix() == {"I": 866, "II": 338}


def test_strata_totals() -> None:
    strata = site_strata()
    assert sum(stratum.accrued for stratum in strata) == FULL_COHORT_SIZE
    assert sum(stratum.held_out for stratum in strata) == HELD_OUT_SIZE
    assert [stratum.site for stratum in strata] == ["A", "B", "C"]
    assert region_pool(["A", "B", "C"]) == {"I": ["A", "B"], "II": ["C"]}
    assert SITE_TOTALS["A"] + SITE_TOTALS["B"] == 4200


def test_tertile_assignment_follows_the_declared_sizes() -> None:
    scores = torch.arange(sum(ANATOMY_TERTILES.values()), dtype=torch.float32)
    labels = anatomy_tertile_assignment(scores)
    assert labels.count("simple") == ANATOMY_TERTILES["simple"]
    assert labels.count("complex") == ANATOMY_TERTILES["complex"]
    assert labels[0] == "simple" and labels[-1] == "complex"


def test_diagnostic_frame_selection() -> None:
    intensity = torch.zeros(24)
    intensity[6:10] = torch.tensor([0.1, 0.9, 0.4, 0.2])
    intensity[14:18] = torch.tensor([0.2, 0.3, 0.8, 0.1])
    tip_load = torch.zeros(24)
    tip_load[12] = 1.0
    plan = select_diagnostic_frames(intensity, [(6, 10), (14, 18)], tip_load)
    assert plan.peak_opacification == (7, 16)
    assert plan.pre_opacification == (5, 13)
    assert plan.union() == tuple(sorted(set(plan.union())))
    assert plan.mask(24).sum().item() == len(plan.union())


def test_diagnostic_frame_selection_rejects_bad_runs() -> None:
    with pytest.raises(ValueError):
        select_diagnostic_frames(torch.zeros(8), [(2, 2)], torch.zeros(8))
    with pytest.raises(ValueError):
        select_diagnostic_frames(torch.zeros(8), [(0, 4)], torch.zeros(4))


def test_transient_ladder_rate() -> None:
    """The transient subset is acquired at about 0.25 Hz against a 15 Hz motion stream.

    Ref: Sec. 4.10 (a 16 s window carries four transient samples).
    """
    assert transient_ladder(60).numel() == 1
    assert transient_ladder(240).numel() == 4
    assert frame_axis_alignment(60, 60, 1e-6)


def test_contact_event_threshold_and_debounce() -> None:
    trace = torch.zeros(40)
    trace[4:9] = 0.2
    trace[20] = 0.3
    trace[30:34] = 0.1
    events = count_contact_events(trace, threshold_n=0.05, debounce_samples=2)
    onsets = [event.onset for event in events]
    assert 4 in onsets and 30 in onsets
    assert 20 not in onsets


def test_contact_rate_columns_are_not_rescalings() -> None:
    counts = torch.tensor([1.0, 2.0, 3.0])
    assert contacts_per_procedure(counts) == pytest.approx(2.0)
    assert contacts_per_thousand_frames(10, 5000) == pytest.approx(2.0)
    assert peak_force_envelope(torch.tensor([0.1, -0.4, 0.2])) == pytest.approx(0.4)
    assert force_duration_integral(torch.tensor([0.0, 0.2, 0.4])) > 0.0
