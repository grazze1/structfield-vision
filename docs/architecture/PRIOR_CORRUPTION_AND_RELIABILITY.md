# Prior corruption and reference reliability supervision

## Scope

Stage 6 provides controlled perturbations for asking how downstream models
behave when an explicit structural prior is wrong, noisy, displaced, or absent.
It also provides one transparent reference objective for an auxiliary
reliability prediction. These utilities are experimental infrastructure, not a
claim that the chosen corruptions or loss are a final scientific definition.

## Functional contract

Every corruption has the form:

```python
corrupted_field, corruption_metadata = corruption(field, ...)
```

The input `StructuralField` is never modified. Every tensor in the returned
field has independent storage, including tensors unaffected by that particular
corruption. Dtype, device, batch shape, schema metadata, and valid probability
constraints are preserved. Operations remain differentiable where their
semantics permit it; discrete random masks and mode/bin choices are not treated
as learnable decisions.

Stochastic functions require an explicit keyword-only `seed`. They create a
generator local to the field device and do not consume the global CPU or CUDA
random-number stream. Repeating the same operation on the same device and input
with the same seed is deterministic. CPU and CUDA generators are not promised
to produce bit-identical samples across devices.

`CorruptionMetadata` is a frozen dataclass containing the canonical operation
name, an immutable ordered parameter tuple, the seed for stochastic operations,
and schema version `0.2`. `to_dict()` returns a JSON-compatible record.

## Canonical corruptions

| Function | Semantics |
| --- | --- |
| `orientation_shift` | Transports orientation posterior mass on the axial `[0,π)` circle using periodic linear interpolation; shifts top modes and recomputes normalized entropy. |
| `orientation_noise` | Samples one Gaussian angular displacement per pixel and applies the same periodic posterior transport. |
| `frequency_scale` | Multiplies frequency mean by `a` and variance by `a²`. |
| `frequency_noise` | Adds Gaussian cycles/pixel noise to frequency mean and clamps at zero. |
| `spatial_shift` | Circularly rolls every spatial tensor by integer `(y,x)` offsets. |
| `scale_shift` | Moves discrete scale-posterior mass by integer bins, accumulating overflow at endpoints, then recomputes expected selected scale. |
| `coherence_noise` | Adds Gaussian noise to coherence and clamps to `[0,1]`. |
| `mode_swap` | Exchanges two stored top modes and their corresponding posterior-bin mass. |
| `field_dropout` | Samples a per-example mask and replaces selected complete fields with a neutral valid prior. |

The neutral dropout prior uses uniform orientation and scale distributions,
zero frequency mean/variance, zero coherence, maximal orientation entropy and
structural uncertainty, and the mean candidate scale. The realized mask is
reproducible from the recorded probability and seed.

These functions deliberately do not recompute the heuristic structural
uncertainty after every corruption. The uncertainty map is itself part of the
prior being controlled: operations leave it unchanged unless their documented
semantics explicitly replace the full field. Experiments that need a freshly
derived uncertainty map must invoke `HeuristicStructuralUncertainty` explicitly.

## Reference reliability target

`ReliabilityLossConfig` exposes both error weights, reliable and unreliable
region weights, temperature `tau`, classification `threshold`, and numerical
`eps`. Inputs are:

```text
prediction                    [B,C,H,W] in [0,1]
orientation_error_rad         [B,1,H,W] in [0,π/2]
frequency_error_relative      [B,1,H,W] non-negative and dimensionless
```

The detached continuous target is:

```text
e_theta = orientation_error_rad / (π/2)
e = (w_theta e_theta + w_frequency e_frequency) / (w_theta + w_frequency)
r = exp(-e / tau)
```

The error weights are normalized, so multiplying both by a common positive
constant does not change the target. `r >= threshold` selects the configured
reliable-region weight; other pixels select the unreliable-region weight. The
loss is the region-weighted mean squared error between every prediction channel
and broadcast soft target `r`.

The threshold does not harden or redefine the soft target. Error-derived
targets are detached so the objective trains the reliability predictor rather
than encouraging an estimator to manipulate its error labels. Float16 and
bfloat16 inputs use float32 working arithmetic; float32/float64 precision is
preserved.

## Reliability, uncertainty, and gate are distinct

- Structural uncertainty is a task-neutral heuristic property of an estimated
  field.
- The reference reliability prediction is supervised against synthetic
  orientation/frequency error and is not asserted to be calibrated probability.
- The Stage-5 spatial-channel gate measures task-specific structural residual
  usage. It is not automatically equal to the reference reliability prediction
  or to `1 - structural_uncertainty`.

An application may later decide how these signals interact, but Stage 6 does
not silently couple them.

## Provenance and limitations

The corruption functions, metadata contract, and reliability objective are
project-authored independent implementations using only PyTorch and Python
standard-library primitives. No third-party source, dependency, dataset, or
weight was copied or vendored. Defaults are transparent baselines and require
experimental validation; they are not theoretically optimal or a final research
claim. Whole-repository licensing remains subject to the final provenance gate
and the third-party boundaries recorded in `NOTICE.md`.
