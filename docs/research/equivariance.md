# Covariance and equivariance

## Terminology

Geometric language is used narrowly in StructField-Vision:

- **Field covariance** asks whether estimating a transformed image agrees with
  transforming the estimated field: `F(Tx) ≈ T_F(F(x))`.
- **Task equivariance** asks whether a model output transforms in the declared
  output representation: `N(Tx) ≈ T_Y(N(x))`.
- **Exact architectural group equivariance** requires a model construction and
  transformation group that guarantee the relevant equality, subject to stated
  discretization assumptions.

V0.2 provides reference transforms that can express the first two approximate
comparisons, but the frozen losses and smoke protocol cover only selected
structural-field signals: orientation posterior, frequency mean, and structural
uncertainty. No dedicated task-output equivariance objective is frozen. V0.2
does **not** claim exact architectural group equivariance. Loss classes retain
“Equivariance” in their API names, but their implemented contract is
geometry-aware structural-field consistency on aligned branches.

## Implemented geometric transport

`GeometricTransform(rotation_rad=φ, scale=s)` describes a same-canvas rotation
and isotropic scale about the pixel center. Positive rotation is
counter-clockwise in displayed image coordinates. Bilinear sampling uses
`align_corners=False`.

The structural transform applies the spatial warp and the semantic value
transport together:

- orientation-posterior mass shifts by `+φ` on the axial interval `[0,π)`;
- top orientation modes are interpolated through doubled-angle vectors, shifted
  by `+φ`, and wrapped modulo π;
- orientation entropy is recomputed from the transported posterior;
- frequency mean divides by `s` and frequency variance divides by `s²`;
- frequency-candidate metadata divides by `s`;
- selected spatial scale and scale-candidate metadata multiply by `s`; and
- coherence and structural uncertainty are spatially warped and clamped to
  their valid ranges.

These rules follow the axial and unit conventions in
[Structural field](structural_field.md). A spatial warp without semantic value
transport is not a valid comparison.

## Strict validity mask

Every tensor or field transform returns an explicit
`equivariance_valid_mask[B,1,H,W]`. A pixel is valid only when its complete
bilinear interpolation footprint lies in valid source support. Pre-existing
holes can be propagated through a source mask.

All reference losses require a mask; there is no unmasked default. Invalid
locations are removed before subtraction and reduction, preventing padding or
sentinel values from affecting the statistic.

## Reference losses

Three parameter-free losses compare already aligned maps:

| Loss | Signal | Reference reduction |
| --- | --- | --- |
| `OrientationEquivarianceLoss` | Orientation posterior | Channel-mean squared error, then masked pixel mean |
| `FrequencyEquivarianceLoss` | Frequency mean | Channel-mean absolute error, then masked pixel mean |
| `UncertaintyEquivarianceLoss` | Structural uncertainty | Masked absolute error |

Both branches remain in the autograd graph. The implementation is a symmetric
consistency regularizer, not an implicit stopped-gradient teacher/student
method.

## Frozen smoke protocol

The result-free V0.3 smoke configuration declares:

- identity;
- rotations of `-15°` and `+15°`;
- scales `0.8` and `1.25`;
- intersection of observed and expected strict validity masks; and
- separate orientation, frequency, and uncertainty losses.

The contract explicitly sets `claim_exact_group_equivariance: false`. The smoke
study checks that aligned comparisons are finite, differentiable, and
nonempty. It is not a long-run robustness study.

### Research Hypothesis — consistency under transforms

Adding the declared masked consistency objectives may reduce transformed-branch
disagreement and may improve robustness to matching geometric changes without
removing the value of explicit structural conditioning.

This must be tested against matched no-consistency controls. A lower consistency
loss alone is not sufficient; a degenerate constant predictor can also be
consistent.

## Optional architecture boundary

The experiment matrix reserves optional comparisons using `escnn`. That package
is not a core dependency, and those architectures are marked planned rather
than implemented V0.2 baselines. Installing an optional package would not by
itself establish correct representation types, transformation actions, or exact
equivariance for the task pipeline.

## Limits

- Same-canvas bilinear transport introduces interpolation approximation.
- Arbitrary rotation of a discrete orientation posterior redistributes mass
  between bins.
- The transform does not cover perspective, anisotropic scale, lens distortion,
  or learned registration.
- CPU and CUDA may differ within floating-point tolerances.
- Software tests do not establish robustness, generalization, or a trained
  performance benefit.

The full comparison sequence is frozen in
[Experimental protocol](experimental_protocol.md); controlled transform and
degradation evidence is described in [Benchmark](benchmark.md).
