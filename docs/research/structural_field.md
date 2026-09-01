# Probabilistic structural field

## Purpose

`StructuralField` is the task-neutral interface between analytic image evidence
and downstream neural conditioning. It represents local structure as validated
probability distributions and moments instead of treating one orientation map
as universally sufficient.

The field is one stage in the research flow defined in
[Method](method.md):

```text
Image → Probabilistic Structural Field → Structural Uncertainty → Reliability Gate → Task Backbone → Prediction
```

## Axial orientation convention

Local orientations are unoriented line axes, not directed arrows:

```text
θ ∈ [0, π)
θ ≡ θ + π
```

The period is therefore **π**, not `2π`. Axial angular error is the shortest
distance modulo π, with range `[0,π/2]`. Doubled-angle vectors
`(cos(2θ), sin(2θ))` are used when angles must be interpolated or projected into
continuous channels.

Tensor coordinates use `[B,C,H,W]`, `x` increases rightward, and `y` increases
downward. Positive angle and rotation are counter-clockwise in the displayed
image. Under this convention, a displayed tangent for angle `θ` is
`(cos(θ), -sin(θ))` in tensor-coordinate components.

## Public schema

| Field | Shape | Meaning |
| --- | --- | --- |
| `orientation_prob` | `[B,K,H,W]` | Normalized posterior over uniform axial bins |
| `frequency_mean` | `[B,1,H,W]` | Posterior mean in cycles/pixel |
| `frequency_var` | `[B,1,H,W]` | Posterior variance in `(cycles/pixel)²` |
| `coherence` | `[B,1,H,W]` | Structure-tensor coherence in `[0,1]` |
| `orientation_entropy` | `[B,1,H,W]` | Normalized orientation entropy |
| `structural_uncertainty` | `[B,1,H,W]` | Deterministic ambiguity baseline in `[0,1]` |
| `scale_prob` | optional `[B,S,H,W]` | Normalized posterior over declared scales |
| `selected_scale` | optional `[B,1,H,W]` | Posterior-expected scale in pixels |

All present tensors share a floating dtype, device, batch size, and spatial
shape. Metadata records orientation-bin centers, frequency and scale
candidates, units, estimator/schema versions, coordinate conventions, and
JSON-safe extras. The schema excludes task targets, image paths, backbone
features, and application-specific labels.

## Posterior construction

The deterministic analytic baseline combines multi-scale structure-tensor
evidence with multi-frequency complex Gabor evidence. The normalized joint
orientation-frequency response yields:

- an orientation posterior over `K` uniform centers `θ_k = kπ/K`;
- a frequency posterior over the configured candidate frequencies;
- frequency mean and variance;
- orientation entropy; and
- axially separated top modes derived on demand with circular non-maximum suppression.

The default uses 16 directions. The tracked 8-, 16-, and 32-direction variants
change posterior resolution while preserving the same axial period and public
contract. Optional Hessian maps are diagnostics and do not alter the stable
schema.

## Why a probability field

A single best angle suppresses evidence about competing directions. The
posterior can instead represent diffuse evidence, neighboring-bin uncertainty,
or multiple separated modes. This matters in controlled scenes containing
crossings, bifurcations, junctions, occlusions, and low-coherence regions.

Probability mass near zero and near π is adjacent under the axial topology.
Mode suppression, angular metrics, corruptions, and geometric transforms all
use that same wraparound rule. Treating those endpoints as linearly distant is
an invalid evaluation.

### Research Hypothesis — retained ambiguity

For downstream tasks in which local structure can be multimodal, retaining the
orientation posterior and its ambiguity signals will be more informative than
passing only an argmax angle. This hypothesis must be tested using matched
backbones and training conditions; the schema alone does not establish a
benefit.

## Diagnostics boundary

`StructuralFieldDiagnostics` may include normalized input, joint responses,
frequency probabilities, tensor eigensystem maps, scale responses, and optional
Hessian responses. These values support estimator debugging but are not stable
task inputs. Keeping them separate prevents experiments from silently changing
the field definition by selecting convenient internal channels.

## Geometric behavior

For a displayed rotation `φ`, orientation values transport by `+φ` modulo π.
For isotropic magnification by scale `s`, frequencies divide by `s` and spatial
scales multiply by `s`. Probability maps are spatially warped and renormalized;
invalid interpolation footprints are excluded through an explicit validity
mask. These operations test covariance of the representation and are detailed
in [Equivariance](equivariance.md).

## Validation and limits

- Construction rejects non-finite tensors, invalid shapes, inconsistent
  devices/dtypes, out-of-range maps, and non-normalized probabilities.
- Uniform posteriors on non-informative pixels are a neutral storage choice, not
  evidence that all orientations are physically present.
- Posterior values are analytic baseline scores, not Bayesian posterior
  guarantees.
- Candidate frequencies and scales discretize the supported range; behavior
  outside that support is not characterized.
- The estimator is deterministic and untrained, but deterministic does not mean
  correct or calibrated.
- Synthetic ground truth can establish controlled estimator behavior only; the
  real-data boundary is defined in [Benchmark](benchmark.md).

The uncertainty derived from this field is specified separately in
[Uncertainty](uncertainty.md), and task-specific use is specified in
[Reliability gating](reliability_gating.md).
