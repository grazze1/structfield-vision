# V0.3 experimental protocol

## Pre-registration status

The tracked V0.3 experiment configurations are **result-free a-priori
definitions**. Each declares `contains_results: false`; the baseline matrix is
also marked `frozen: true`. They freeze comparison semantics, controlled prior
corruptions, a geometry-consistency smoke study, and failure-score definitions.

This is a partial pre-registration, not a completed training manifest. Dataset
instances, model widths for a final study, optimizer, learning-rate schedule,
batch size, stopping rule, compute budget, and statistical power remain
unspecified and must be frozen before any formal run. No formal long-duration
training has been performed for this protocol.

## Research question and conceptual path

The protocol tests:

> When should neural networks trust explicit structural priors?

through the declared flow:

```text
Image → Probabilistic Structural Field → Structural Uncertainty → Reliability Gate → Task Backbone → Prediction
```

Definitions for each stage are in [Method](method.md). Every study must keep
field estimation outside the application `forward` boundary so timing,
corruption, device placement, detachment, and provenance remain explicit.

## Frozen comparison matrix

The six V0.3 protocol variants are:

| Protocol | Prior | Adaptive gate | Uncertainty | Prior corruption | Consistency loss |
| --- | --- | --- | --- | --- | --- |
| `no_prior` | None | No | No | No | No |
| `fixed_prior` | Fixed concatenation | No | No | No | No |
| `probabilistic_prior` | Probabilistic affine conditioning | No | Yes | No | No |
| `reliability_gated_prior` | Probabilistic residual | Yes | Yes | No | No |
| `reliability_gated_prior_corruption` | Probabilistic residual | Yes | Yes | Yes | No |
| `reliability_gated_prior_corruption_equivariance` | Probabilistic residual | Yes | Yes | Yes | Yes |

These rows identify implemented components, not trained outcomes. Any comparison
must match the task backbone, input preprocessing, data split, optimizer,
training steps, augmentation, evaluation mask, and selection rule. Parameter
count and runtime must be reported because adapter modes change channel count
and learned projections.

The future matrix also reserves optional `escnn` variants. They are marked
planned and must not be included as completed baselines until representation
types, transformation actions, installation, training, and evaluation have all
been independently verified.

## Structural-field configuration

The default analytic estimator uses:

- 16 uniform axial orientation bins over `[0,π)`;
- frequencies `0.04`, `0.08`, and `0.16` cycles/pixel;
- structure-tensor scales `1.2`, `2.4`, and `4.8` pixels;
- Gaussian weak denoising with sigma `0.8`;
- orientation temperature `0.15` and scale temperature `0.10`;
- two top modes with minimum separation `π/8`; and
- no Hessian contribution to the public field by default.

Tracked 8-, 16-, and 32-bin variants provide an orientation-resolution
sensitivity study. They must not be mixed within one matched comparison unless
the application projection and checkpoint are explicitly compatible.

## Frozen structural-prior corruption sweep

The seed base is `20260830`. Stochastic seeds follow the declared base plus
corruption-index and level-index offsets. The sweep leaves structural
uncertainty unchanged unless a future protocol explicitly changes
`recompute_structural_uncertainty`.

| Corruption | Frozen levels |
| --- | --- |
| Orientation shift | `0`, `π/16`, `π/8` radians |
| Orientation-noise standard deviation | `0`, `5°`, `10°` |
| Frequency scale | `0.75`, `1.00`, `1.25` |
| Frequency-noise standard deviation | `0`, `0.01`, `0.03` cycles/pixel |
| Spatial shift `(y,x)` | `(0,0)`, `(2,0)`, `(0,2)`, `(2,2)` pixels |
| Scale-bin shift | `-1`, `0`, `+1` |
| Coherence-noise standard deviation | `0`, `0.10`, `0.25` |
| Mode exchange | stored modes `0` and `1` |
| Field dropout probability | `0`, `0.25`, `0.50`, `1.00` |

Each run compares against the uncorrupted field, preserves corruption metadata,
and reports per seed before aggregation. Structural prior corruption must not be
confused with observation degradation in
[Benchmark](benchmark.md).

## Frozen geometry-consistency smoke study

The declared transforms are identity, rotations of `-15°` and `+15°`, and
isotropic scales `0.8` and `1.25`. Observed and expected branches are aligned
using the implemented field transform. Only the intersection of their strict
validity masks contributes to orientation-posterior, frequency, and uncertainty
losses.

Both branches remain attached to autograd. The configuration explicitly
disallows a claim of exact architectural group equivariance. See
[Equivariance](equivariance.md).

## Frozen failure-prediction definitions

Failure labels are explicit caller-supplied binary maps; no metric or loss
creates them implicitly. Two score families are reported separately:

1. structural uncertainty; and
2. one minus predicted reliability.

Higher scores mean greater predicted failure. The reference reliability target
uses equal orientation/frequency error weights, temperature `0.25`, threshold
`0.5`, reliable-region weight `1`, and unreliable-region weight `2`. AUROC
requires both classes; AUPRC requires at least one failure. The configuration
does not permit a calibration claim.

Before a formal task run, the failure event itself must be frozen in domain
terms—for example, a declared threshold on a task error—without inspecting test
scores. Threshold choice and score direction must appear in the run manifest.

## Required evaluation sequence

### Phase A — software and semantics

- Validate configuration schema and tracked file identity.
- Run CPU and real-CUDA forward/backward smoke tests where supported.
- Check deterministic replay on the same device.
- Check axial wrap, mask behavior, dtype/device preservation, and finite values.

This phase establishes executable contracts only.

### Phase B — controlled structural estimation

- Evaluate the analytic estimator on deterministic benchmark test identities.
- Report axial error, orientation-distribution divergence, recall at a declared
  tolerance, and frequency error.
- Stratify by family, difficulty, junction status, and degradation.
- Run the 8/16/32-bin sensitivity comparison without choosing a variant from
  test results.

### Phase C — matched task study

- Freeze all missing training choices and seeds in a run manifest.
- Train all applicable comparison rows under equal data and schedule.
- Report no-prior, fixed-prior, probabilistic-prior, and gated controls before
  adding corruption or consistency terms.
- Report task error together with gate/residual diagnostics, parameter count,
  throughput, and field-estimation cost.
- Preserve per-seed outputs and aggregate only after per-seed reporting.

This phase is planned and has not been executed as a formal long-duration
study.

### Phase D — external real-data validation

- Proceed only after the data, license, privacy, split, target, and decoding
  contracts in [Benchmark](benchmark.md) are documented.
- Keep synthetic and real-data results in separate tables.
- Freeze any domain-specific threshold and subgroup plan before test
  evaluation.

This phase is not currently supported by distributed data or dedicated dataset
adapters.

## Research Hypotheses and decision rules

### Research Hypothesis H1 — selective prior benefit

The gated prior improves the declared task outcome over no-prior and fixed-prior
controls under matched conditions, without hiding a larger compute or parameter
budget.

### Research Hypothesis H2 — corruption robustness

As controlled prior error increases, the gated model degrades less than fixed
or ungated prior conditioning while maintaining nontrivial use on clean fields.

### Research Hypothesis H3 — failure information

Structural uncertainty and predicted failure scores provide ranking information
for the frozen failure event, with separate score reports and subgroup checks.

### Research Hypothesis H4 — geometric consistency

The declared consistency objective reduces valid-region transformed-branch
error without a degenerate loss of task accuracy or structural sensitivity.

Each hypothesis must be reported as supported, unsupported, or inconclusive
under the frozen study. The protocol does not authorize changing corruption
levels, masks, score direction, or subgroup definitions after viewing test
results without labeling the analysis exploratory.

## Result and artifact policy

- Configuration files contain no measurements.
- Software-test outputs are labeled as engineering verification.
- Formal runs require immutable config, code revision, environment, device,
  seeds, data manifest, and output checksums.
- Failed and incomplete runs remain distinguishable from completed runs.
- No task-performance statement may be inferred from an untrained model, a
  unit test, a smoke run, or a synthetic visualization.

Related conceptual references and unresolved literature-review work are listed
in [Related work](related_work.md).
