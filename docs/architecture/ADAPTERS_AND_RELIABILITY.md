# Structural adapters and downstream reliability

## Tensor boundary

Stage 5 adapters operate on already aligned feature tensors:

```text
backbone features x        [B,C,H,W]
structural features z      [B,D,H,W]
structural uncertainty u   [B,1,H,W]  (reliability-gated adapter only)
```

The adapter layer does not silently resize, cast, or move these tensors. Batch and spatial dimensions, dtype, and device must already agree. This keeps interpolation and structural-feature selection explicit at the caller boundary and avoids freezing an unrequested universal `StructuralField` encoder before the application migration.

`u` is not duplicated inside `z` for the reliability gate. An upstream encoder may choose task-neutral field features, but it must pass the scalar structural uncertainty separately.

## Unified output

Every adapter returns:

```python
AdapterOutput(
    features,
    gate=None,
    structural_delta=None,
)
```

All present tensors are finite BCHW maps. A present `gate` or `structural_delta` has exactly the same shape, dtype, and device as the returned features. Gate values lie in `[0,1]`.

`output_channels(feature_channels, structural_channels)` reports the output channel count before execution. `NoPriorAdapter` also permits the structural channel argument to be omitted.

## Baselines

### No prior

```text
y = x
```

`NoPriorAdapter` accepts no structural tensor and returns the original feature tensor exactly. It represents the no-prior ablation and does not require the structural estimator to run.

### Fixed concatenation

```text
y = concat(x, z, channel)
```

`ConcatStructuralAdapter` has no learned state. Its output has `C+D` channels and preserves the first `C` channels exactly.

### FiLM conditioning

A learned `1x1` projection produces spatial-channel affine parameters:

```text
[gamma, beta] = projection(z)
delta = gamma * x + beta
y = (1 + gamma) * x + beta = x + delta
```

The raw residual `delta` is returned for inspection. This is an independent, lightweight implementation of the feature-wise affine-conditioning idea; it is not copied from an external implementation.

### Reliability-gated residual

```text
delta = structural_projection(z)
gate = sigmoid(gate_network(concat(x, z, u)))
y = x + gate * delta
```

The V0.2 gate is spatial-channel: its actual shape is `[B,C,H,W]`, not a broadcast scalar map. Both the learned gate and ungated candidate delta are returned. A pure residual fusion path is used so an explicit all-zero gate gives exact residual identity.

One lightweight adapter instance can be attached to each backbone stage. Stage 5 intentionally does not add attention, multiple gate families, implicit pyramid management, or a per-stage controller.

## Structural uncertainty is not the gate

`structural_uncertainty` is a task-neutral, parameter-free estimate of ambiguity in the structural field; higher means the field estimate is less certain.

The downstream `gate` is learned through the application objective and represents how much structural residual a particular stage, channel, and pixel uses. It is not a calibrated reliability/failure probability, need not equal `1-u`, and need not be monotonic in `u`. A gate heatmap visualizes structural-prior usage, not certainty by itself.

## Numerical behavior

- Inputs are validated as finite, non-empty floating BCHW tensors.
- Learned convolutions follow normal PyTorch autocast behavior; the analytic estimator's autocast-disabled policy is not imposed on neural adapters.
- The returned tensors are explicitly brought back to the backbone feature dtype.
- Invalid channel counts, batch/spatial mismatches, dtype/device mismatches, out-of-range uncertainty, and invalid gate fusion fail fast.

## Provenance and limitations

The adapter contracts, concatenation baseline, residual gate, and lightweight gate network are project-authored. FiLM is an independent implementation of the affine-conditioning concept described by E. Perez et al., “FiLM: Visual Reasoning with a General Conditioning Layer,” AAAI 2018, [AAAI article 11671](https://ojs.aaai.org/index.php/AAAI/article/view/11671).

No external source code or new dependency is included. The gate is uncalibrated until downstream experiments establish otherwise. Project-authored code and documentation are covered by Apache-2.0; external artifacts retain their own terms.
