# Reliability gating

## Purpose

Reliability gating provides an inspectable place for a task network to vary how
strongly an explicit structural prior changes its features. It addresses the
central question—**When should neural networks trust explicit structural
priors?**—by turning “trust” into a measurable residual-use map rather than a
verbal attribute.

## Adapter comparison boundary

Let `x ∈ R^(B×C×H×W)` be a task feature tensor, `z` an aligned structural
feature tensor, and `u ∈ [0,1]^(B×1×H×W)` the structural-uncertainty map. The
implemented adapters are:

```text
No prior:                 y = x
Fixed concatenation:     y = concat(x,z)
Affine conditioning:     y = (1+γ(z)) · x + β(z)
Reliability-gated:       δ = structural_projection(z)
                         g = sigmoid(gate_network(concat(x,z,u)))
                         y = x + g · δ
```

For the gated adapter, `g` and `δ` have shape `[B,C,H,W]`. The adapter returns
both maps alongside `y`, making it possible to inspect candidate prior effect
separately from actual prior use. An all-zero gate produces exact residual
identity.

Inputs must already agree in batch, spatial shape, dtype, and device. The
adapter never silently resizes, casts, moves, or estimates a field. This keeps
estimation cost, corruption state, and data provenance visible to the caller.

## What the gate means

The gate is trained through the downstream application objective. It measures
how much of a candidate structural residual a particular stage, channel, and
pixel uses. It is therefore:

- task- and backbone-dependent;
- allowed to differ between channels at one pixel;
- not constrained to be monotone in structural uncertainty;
- not a calibrated reliability or failure probability; and
- not evidence that the prior itself is physically correct.

Structural uncertainty, predicted reliability, and the adapter gate remain
separate signals, as detailed in [Uncertainty](uncertainty.md).

## Auxiliary reliability supervision

The repository also implements `ReliabilityLoss`, an optional reference
objective for a predicted reliability map. It derives a detached soft target
from normalized axial-orientation error and relative-frequency error, then uses
configured reliable/unreliable region weights in a mean-squared objective.

That auxiliary predictor is not wired into the adapter by default. Any study
that shares it with, supervises, or regularizes the gate must declare the new
coupling as an experimental choice. Otherwise, interpreting gate values as the
reference reliability target would be incorrect.

## Corruption as a causal stress control

The frozen corruption sweep perturbs one declared aspect of a valid field while
preserving field schema and recording immutable metadata. Available controls
cover:

- axial orientation shift and pixelwise orientation noise;
- frequency scaling and frequency noise;
- integer spatial shift and discrete scale-bin shift;
- coherence noise;
- top-mode exchange; and
- whole-field dropout to a neutral prior.

Stochastic operations require explicit local seeds. Most operations leave the
existing uncertainty map unchanged, enabling a study to distinguish “the field
was damaged” from “the uncertainty signal noticed the damage.” The experiment
must not imply automatic uncertainty recomputation when it was disabled.

### Research Hypothesis H1 — damage containment

Under matched training and controlled prior corruption, reliability-gated
conditioning will limit task degradation relative to fixed or ungated prior
injection.

### Research Hypothesis H2 — useful selective use

On uncorrupted or locally reliable structure, the learned gate will retain a
nonzero structural contribution where that contribution improves the declared
task objective, rather than collapsing to a universal no-prior solution.

Both statements are unverified. Reporting only average task error is
insufficient: the study must also report gate statistics, candidate residual
magnitude, corruption family/level, valid-region policy, and no-prior behavior.

## Falsification and diagnostics

Evidence against reliability-aware gating includes any of the following under
matched conditions:

- no improvement over the no-prior and fixed-prior controls;
- equal or greater sensitivity to controlled field corruption;
- gate saturation that is unrelated to declared error or corruption;
- apparent robustness caused only by a vanishing structural residual;
- gains that disappear when parameter count or training schedule is matched;
  or
- failure-ranking behavior that does not survive subgroup reporting.

These are experiment diagnostics, not assumptions about expected outcomes.
Gate histograms and maps must always be paired with task and field-error
evidence; visualization alone is not validation.

## Current status

The adapters, returned gate/residual diagnostics, corruption functions, and
reference reliability objective are implemented and software-tested. The V0.3
comparison matrix is frozen with no embedded results. No formal long-duration
training has established robustness, calibration, or a task benefit.

See [Method](method.md) for the application boundary,
[Benchmark](benchmark.md) for controlled evidence, and
[Experimental protocol](experimental_protocol.md) for the pre-registered
comparisons.
