# CAPS — coupling-anisotropic private synthesis of vascular images and catheter motion

This is the code release for *Differentially private joint generation of vascular images and
catheter motion for collision avoidance*. The manuscript asks a question the image-synthesis and
trajectory-synthesis literatures answer separately: can a single mechanism release a synchronised
angiographic frame stack and six-channel catheter-motion tensor under one procedure-level privacy
account? The release implements the answer the manuscript gives — design the *geometry* of the
privacy noise rather than its magnitude — and the machinery needed to check that answer.

Four components carry the mechanism, and each one has a single home in this tree:

| component | what it does | module |
| --- | --- | --- |
| shared coupling latent | carries procedure identity into both branches, and is the only channel that does | `coupling/latent.py` |
| coupling head | solves `Σ12 Σ22⁻¹ Σ21 u = λ² Σ11 u` over the paired branch gradients and freezes the leading directions | `coupling/cca.py`, `coupling/head.py` |
| public-reference shaping estimator | computes the coupling subspace from non-private corpora only, then freezes it | `estimator/reference_estimator.py` |
| coupling-shaped noise generator | draws `Z` with `Σ⁻¹ = σ∥⁻² Π_S + σ⊥⁻² Π_S⊥` and applies `θ ← θ − γ(AG + Z)` | `geometry/`, `substrate/` |

Around those sit the two branches (a state-space vessel-segmentation encoder and a flow-matching
action head), the Rényi accountant, the membership-inference audit, the factorial and ablation
analysis, the constrained-control comparators, and a two-layer verification driver.

## What is measured here and what is not

The mechanism is checked end to end on a schema-compatible substrate of the same tensor contract,
and the manuscript's own arithmetic is checked separately. The **private multi-centre cohort is not
exportable**, so every cohort-level number in the manuscript — the fidelity, Dice, contact-event and
clinical columns of Tables 1, 2, 4, 5 and 6 — is reported as `BLOCKED` rather than estimated. The
same applies to the 8.3B backbone weights, the expert vessel annotations, the nine-reader study and
the bench-top phantom arm. `verification_report.json` carries the per-item status; nothing in this
release is presented as a reproduced clinical result.

## Installation

```bash
python -m pip install -e ".[dev]"          # editable install with the lint and test toolchain
python -m pip install -r requirements.txt  # pinned runtime plus toolchain, no editable install
conda env create -f environment.yml        # conda path, same versions
docker build -t caps:local .               # container path, CUDA 12.4 runtime base
```

The container entry point is the verification driver, so `docker run --rm caps:local` reproduces the
status table without any further setup.

## Data

The release reads two things from disk and downloads nothing on its own.

**Private layer (blocked).** 5,842 anonymised procedures accrued at Sites A, B and C across two
regions, of which 1,204 form the held-out partition (866 from Region I, 338 from Region II). The
site sizes, held-out slices, force-channel coverage and clinical-arm split are declared once, in
`cohort/partition.py`, and checked against the manuscript transcription by the manuscript layer of
the verification driver. The procedures themselves are not part of this release.

**Auxiliary public layer.** The shaping estimator needs a non-private reference set disjoint from
the private cohort. The manuscript names it as public angiographic collections and open-simulator
expert trajectories without naming the corpora, so no URL is asserted here. Place `clip_<index>.pt`
payloads holding an `appearance` tensor (`H × W × T`) and a `motion` tensor (`6 × T`) under
`data/reference/`, then run:

```bash
bash scripts/prepare_reference_corpus.sh   # writes manifest.json next to the clips
```

`dataset_urls.txt` lists the upstream projects this release points at — the backbone weights and
their upstream code, the segmentation architecture, the attention-privatisation recipe, the
world-model comparator and the tensor framework — each with the content that was read at that URL
and the licence stated there. The sources the manuscript leaves unnamed are recorded in the same
file as unresolved rather than filled in with a plausible-looking substitute.

**Schema-compatible substrate.** Every execution path that needs procedures uses
`private_vascular_motion_generation.cohort.fixture`, a seeded generator that emits the same
`H × W × T` appearance tensor and `6 × T` motion tensor with the same channel order, the same
frame provenance and the same tube geometry. Its shape report is checked by `EX-DATA-01`.

## Training

One command per arm; the reported plan lives in the configuration, not in the command line.

```bash
bash scripts/launch_caps_main.sh                 # primary arm, 12,000 rounds at lr 3.2e-3
CAPS_MAX_ROUNDS=4 bash scripts/launch_caps_main.sh   # confirmatory pass, override stated in the report
bash scripts/launch_ablation_sweep.sh            # ablation, window ladder, factorial, three scalings
```

or directly:

```bash
python -m private_vascular_motion_generation.cli.run_study --study caps_main --root .
python -m private_vascular_motion_generation.cli.run_study --study ablation_components --root . --max-rounds 2
python -m private_vascular_motion_generation.cli.run_study --study overfit_single_batch --root .
```

Study arms: `caps_main`, `ablation_components`, `ablation_window_ladder`, `factorial_resolution_conditioning`,
`audit_membership`, `scaling_substrate_volume`, `scaling_reference_corpus`, `scaling_window_horizon`,
`perturbation_response`, `policy_comparison`, `null_results`, `component_interaction`, `metric_suite`,
`release_path`, `overfit_single_batch`. Any configuration key can be overridden with `--override key=value`,
and `--max-rounds` bounds the private pass explicitly — the report records the override, so a bounded
run is never mistaken for the reported plan.

Training the private pass applies plain gradient descent on the single declared workload: per-procedure
gradients are formed on the concatenated appearance-and-motion residual, clipped once at a single norm
before any noise is drawn, aggregated into the procedure-by-micro-step workload, given the coupling-shaped
noise, and applied with `θ ← θ − γ(AG + Z)`. The accountant accumulates one Rényi charge per round and
converts once at the end.

## Evaluation

```bash
bash scripts/run_verification.sh
PYTHONPATH=src python -m private_vascular_motion_generation.cli.verify --root . --scratch reports/scratch
PYTHONPATH=src python -m private_vascular_motion_generation.cli.summarise --root . --show all
pytest -q
```

The shell scripts put `src/` on the module path themselves, so they run from a fresh clone. The
bare `python -m ...` forms and `pytest -q` work without the prefix once `pip install -e .` has run;
otherwise set `PYTHONPATH=src` as shown.

The verification driver runs two layers that are reported apart on purpose:

* the **execution layer** actually reads the substrate, runs the forward pass, computes both branch
  losses, differentiates, applies the privatised update, saves and reloads a checkpoint, fits a
  single batch until the residual falls, and exercises every metric, audit, contrast and policy
  path. Its independent references — a brute-force dynamic-time-warping recursion, a rank-sum area,
  an inverted score test for the Wilson interval, a quadrature form of the Rényi divergence — are
  written inside the checks so that no check validates an implementation against itself. It also
  resolves every reference in the claim map to a live symbol and confirms that every check cited as
  evidence exists.
* the **manuscript layer** recomputes the printed quantities of the manuscript from the other printed
  quantities of the manuscript: the signed deltas of Table 1, the four factorial cells and their two
  main effects and interaction, the pairwise interaction ratios, the stratum-weighted pooled column,
  the subgroup Wilson intervals, the twenty-initialisation columns, the window ladder and the raw
  scaling residuals.

A failure in the second layer is a disagreement inside the manuscript's own tables, not a broken
release, which is why `verification_report.json` reports `execution_family_status` and
`manuscript_family_status` side by side. Where a printed quantity could not be rebuilt from the
printed data, the check fails with a note saying exactly what did not close. Two such disagreements
are recorded and left standing: the printed residual column of Supplementary Table SC2 cannot be
rebuilt from the printed endpoint column, and that same printed column is systematically negative at
the largest substrate size, which the caption's own rule reads as a departure of the fit.

## Compute budget

The reported plan is 12,000 rounds at a learning rate of 3.2e-3 with a privatised attention width of
`d = 1024`, a procedure window of `k = 16` micro-steps, coupling rank `r = 12`, noise multipliers
`σ∥ = 5.4` and `σ⊥ = 2.6`, clipping norm `C = 0.85` and `δ = 10⁻⁵`. Per round the mechanism pays
`O(d² + dk)` in time and `O(d + Bk)` in space, and the accountant pays `O(K)` once rather than `O(N)`.
The manuscript does not report the GPU type, count, wall-clock or storage, and it does not state the
procedure-level sampling rate; `configs/study/caps_main.yaml` marks the sampling rate, the schedule
and the small architecture choices as engineering defaults under `defaults_not_reported_in_manuscript`.
The verification run in this tree uses a bounded round count and the reduced-width `_smoke`
configuration for the unit-level paths; that configuration is labelled as a test configuration and
its numbers are never quoted as mechanism results.

## Layout

```
configs/{scheme,cohort,runner,study}   study arms, noise schemes, cohort definitions, optimisation plans
src/private_vascular_motion_generation/
    cohort/      procedures, frame selection, partitions, force channel, reference layer, fixture
    coupling/    shared latent, canonical-correlation head, prefix-sum window
    geometry/    projectors, coupling-shaped covariance, concentration
    accounting/  Rényi divergence, order grids, clipping, privacy tuple
    branches/    appearance encoder, privatised attention, flow-matching head, joint decoder
    estimator/   the public-reference shaping pass
    substrate/   workload assembly, noise application, release
    metrics/     fidelity, vessel geometry, motion agreement, safety, clinical endpoints
    audit/       membership inference, distance to closest record
    analysis/    factorial contrasts, interactions, multiplicity, intervals, resampling, scaling fits
    policy/      environment, diffusion policy, barrier filter, potential field, actor-critic, world model
    runner/      configuration, batching, seeding, optimisation, checkpointing, trainer, study arms
    verification/ manuscript arithmetic, execution checks, report writers
    cli/         run_study, verify, summarise
scripts/    staging, training, ablation sweep, verification
tests/      unit, integration, shape, single-batch and smoke tests
```

`LICENSE` carries the release's own terms; `NOTICE` lists every third-party component with its
upstream licence and the URL that licence was read from.

## Reports

Running the verification driver writes `claim_to_code.json` (every manuscript item mapped to the
file and symbol that implements it, with its status), `verification_report.json` (both layers, their
counts, the discrepancies and the unresolved items) and `integrity_manifest.json` (SHA-256 per file
plus an aggregate over the release). Per-arm measurements land under `reports/`.
