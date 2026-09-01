# Heuristic structural uncertainty

## Contract

`HeuristicStructuralUncertainty` is the deterministic v1 baseline heuristic for turning a validated `StructuralField` into one uncertainty map shaped `[B,1,H,W]`. Its output preserves the field device and public dtype and is bounded to `[0,1]`.

This score is a transparent heuristic, not a calibrated probability, a theoretically optimal estimator, or a learned model. It has no trainable parameters or buffers. V0.2 does not train a learnable uncertainty combiner.

## Components

All categorical calculations renormalize non-negative probability mass. Float16 and bfloat16 fields are evaluated internally in float32.

For normalized orientation entropy,

```text
u_orientation = -sum_k p_k log(p_k) / log(K)
```

with `0 log(0) = 0`. The estimator and uncertainty module share the same entropy implementation.

For orientation-mode margin, let `a` be the highest orientation probability and `b` the highest probability whose bin is at least `metadata.default_min_separation_rad` away from `a` under axial distance. Then

```text
u_margin = 1 - (a - b) / (a + b) = 2b / (a + b)
```

If no separated runner-up exists, `b=0` and `u_margin=0`; orientation entropy still represents diffuse evidence. This mode-level definition deliberately avoids treating leakage into an adjacent bin as a second structural direction.

For frequency variance with at least two candidate frequencies,

```text
u_frequency = clamp(4 * frequency_var / (f_max - f_min)^2, 0, 1)
```

The denominator is the maximum variance of a distribution supported on the candidate interval. A single candidate has zero frequency uncertainty and retains its weight. Missing candidate metadata is an error when this component has positive weight.

The tensor term is

```text
u_coherence = 1 - coherence
```

For `S >= 2` scale candidates,

```text
u_scale = -sum_s q_s log(q_s) / log(S)
```

A single scale has zero scale disagreement and retains its weight. If `scale_prob` is absent, the scale component is omitted entirely.

## Combination and missing scale

The default weights, in component order, are:

```text
orientation entropy       0.25
orientation mode margin   0.20
frequency variance        0.20
one minus coherence       0.25
scale entropy             0.10
```

Weights are configurable, finite, and non-negative. The active weighted sum is always divided by the active weight total. Consequently, a field without `scale_prob` uses denominator `0.90` with the defaults; the missing component is not silently replaced by zero.

## Estimator integration

`AnalyticStructuralFieldEstimator` writes the default heuristic result into `StructuralField.structural_uncertainty` without mutating an existing field. Metadata records `baseline-heuristic-v1` and the five default weights. Intermediate component maps remain internal and do not expand the public `StructuralField` schema.

## Limitations

- The five signals are correlated, and the fixed weights are not fitted or calibrated.
- Scores are pointwise and do not model spatially correlated failure.
- Mode selection is piecewise differentiable; exact ties need not have a unique gradient.
- Frequency normalization depends on the declared candidate support.
- Downstream failure-prediction quality must be measured later; it is not implied by a high or low heuristic score.

## Algorithm provenance

The implementation is project-authored and contains no copied third-party code. Normalized categorical entropy follows the standard information-entropy definition described by C. E. Shannon, “A Mathematical Theory of Communication,” 1948, [DOI 10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x). Frequency normalization uses the classical bounded-support variance inequality associated with T. Popoviciu. The relative separated-mode margin, missing-component renormalization, and fixed convex combination are project-specific baseline design choices.

Mathematical references do not introduce source-code licensing obligations. Project-authored code and documentation are Apache-2.0; third-party artifacts remain under their own terms.
