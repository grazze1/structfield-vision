# Application Semantic Migration

## Boundary and non-compatibility

The three V0.2 applications consume an explicit `StructuralField` supplied by
the caller. They never run an estimator inside `forward`, so estimation cost,
field corruption, device placement, and provenance remain visible to the
experiment. Only the `none` adapter mode may omit the field.

This is a semantic rewrite, not checkpoint or import compatibility. Historical
V0.1 source, results, and research records are excluded from the public
repository. A shape-compatible old tensor is not evidence of semantic
compatibility and must not be loaded silently with `strict=False`.

Every application supports the Stage-5 comparison boundary:

```text
none | concat | film | reliability_gated
```

The learned reliability gate receives `structural_uncertainty` through its
dedicated scalar input. Uncertainty is not duplicated inside the fixed
structural feature tensor. Restoration models can return their `AdapterOutput`
on request; the segmentation model exposes a present gate/delta in its output
mapping. A gate map describes learned prior usage, not calibrated confidence.

## Versioned field projections

The feature channel order is part of the application schema and is not inferred
from arbitrary dataclass field order.

### Fringe: `fringe-structfield-k-plus-5-v0.2`

```text
orientation_prob[K]
cos(2 theta_top1), sin(2 theta_top1)
frequency_mean
selected_scale / max(scale_candidates), or zero when absent
coherence
```

### Texture: `texture-structfield-k-plus-8-v0.2`

```text
orientation_prob[K]
cos/sin(2 theta_top1)
cos/sin(2 theta_top2)
frequency_mean
sqrt(frequency_var)
coherence
selected_scale / max(scale_candidates), or zero when absent
```

If a field stores only one top mode, the texture mapping selects two distinct
distribution modes with zero minimum angular separation. This makes the
channel count explicit without reviving V0.1 mode-specific frequency fields.

### Curvilinear: `curvilinear-structfield-8-v0.2`

```text
orientation first moment: sum p(theta) cos(2 theta)
orientation first moment: sum p(theta) sin(2 theta)
frequency_mean / max(frequency_candidates)
sqrt(frequency_var) / max(frequency_candidates)
coherence
orientation_entropy
normalized entropy of scale_prob, or zero when absent/single-scale
selected_scale / max(scale_candidates), or zero when absent
```

This fixed-eight representation replaces the historical task-specific Hessian,
blob, and line-confidence channel bundle with fields from the stable public
schema. It is a new semantic mapping and does not claim numerical parity.

## Fringe restoration

`FringeRestorer` retains the high-level V0.1 scientific path: a two-channel
image encoder (`image` plus an explicitly supplied analytic normalized image),
a separate structural encoder, fixed/adaptive fusion, a three-level residual
U-Net, final-decoder noise FiLM, and an additive one-channel residual head
clamped to `[0,1]`.

The no-prior mode fixes the second image channel to zero and omits the
structural encoder entirely, so it has no unused trainable prior branch. A
missing noise descriptor means a zero vector of the configured dimension; the
old hard-coded ten-dimensional fallback is not retained.

The reference objective is:

```text
L1 + axial Sobel-gradient direction + Sobel-gradient magnitude
```

The last term is deliberately not called a frequency-field or Fourier loss.

## Texture restoration

`TextureRestorer` retains separate image/field encoders, adapter fusion, a
three-level residual U-Net, and a global Fourier-domain residual block at the
bottleneck. The block uses `rfft2`, learned real/imaginary `1x1` projections,
`irfft2`, and a residual connection. It is not a full Fourier U-Net. The
provenance-uncertain historical NAFNet-like path is excluded.

The reference objective exposes L1, spatial-gradient, axial-gradient-direction,
and Fourier-amplitude terms. Float16/bfloat16 FFT and loss work is promoted to
float32; float64 is preserved.

## Curvilinear segmentation

`CurvilinearSegmenter(adapter_mode="none")` preserves the one-channel,
width-24, three-level plain U-Net baseline and its historical parameter count
(`1,085,305`) as an architecture check, not a performance claim. Conditioned
variants inject the fixed-eight field representation at the input adapter
boundary. The active head returns binary segmentation logits only.

The active objective is per-image soft Dice plus BCE-with-logits. The historical
clDice-derived source, centerline/orientation/width heads, private dataset
targets, and Qilu/SNP-Net runtime are not part of the new application. Low
precision losses accumulate in float32, and float64 is retained.

## Numerical and test boundary

- Inputs are finite grayscale BCHW tensors with spatial dimensions at least
  `8x8`; odd sizes are supported by explicit decoder-size alignment.
- Supplied fields must match batch, spatial size, orientation-bin count, dtype,
  and device. There is no implicit resize, cast, or estimator call.
- Float16/bfloat16 losses use float32 work tensors. The Fourier bottleneck also
  promotes low-precision FFT operations to float32.
- Tiny tests instantiate every application and adapter mode and execute
  forward/backward on CPU and real CUDA. These are software checks, not model
  training or performance evaluation.
- Axial-gradient direction uses normalized absolute 2-D cross product (sine
  misalignment). Identical or opposite gradients therefore have exact zero
  direction loss, and flat regions are not assigned a spurious unit penalty.
  This deliberately corrects a V0.1 pathology and is one reason
  checkpoint/numeric parity is not claimed.

## Provenance

All active application code is independently written for V0.2. The
encoder/decoder follows the high-level U-Net idea from O. Ronneberger,
P. Fischer, and T. Brox, [“U-Net: Convolutional Networks for Biomedical Image
Segmentation”](https://arxiv.org/abs/1505.04597). Feature-wise affine
conditioning follows the FiLM concept from E. Perez et al., [“FiLM: Visual
Reasoning with a General Conditioning Layer”](https://arxiv.org/abs/1709.07871).
The Dice objective is an independently written scalar formulation related to
the objective described by F. Milletari, N. Navab, and S.-A. Ahmadi,
[“V-Net”](https://arxiv.org/abs/1606.04797).

No external implementation was copied. No NAFNet-like source, clDice source,
BM3D source, dataset adapter, external image, checkpoint, pretrained weight,
organizer label, or competition runtime was added to `applications/`.
Whole-repository licensing remains pending final provenance review.
