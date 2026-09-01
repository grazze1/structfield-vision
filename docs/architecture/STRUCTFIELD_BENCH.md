# StructField-Bench-v0.1

## Purpose and version boundary

StructField-Bench-v0.1 is a deterministic procedural benchmark for testing
structural-field semantics and failure behavior before external datasets are
introduced. Its primary product is analytically controlled ground truth, not
photorealistic imagery or a claim of real-world model performance.

The public constants are:

```text
benchmark_version = StructField-Bench-v0.1
generator_version = structfield-procedural-v0.1
```

Changing sample semantics, family definitions, seed mapping, or tensor schema
requires a new generator and, when incompatible, benchmark version.

## Sample schema

Each unbatched `BenchmarkSample` contains:

```text
image                         [1,H,W] float in [0,1]
orientation_distribution_gt  [K,H,W] float, sum_K = 1
orientation_count            [1,H,W] int64 in [0,M]
component_frequency_gt       [M,H,W] float, cycles/pixel
valid_mask                    [1,H,W] bool
junction_mask                 [1,H,W] bool
metadata                      frozen BenchmarkMetadata
```

`orientation_count` is the number of analytic components at a pixel, and the
first `orientation_count` frequency channels store those components in
deterministic renderer order. Remaining channels are zero. `valid_mask` equals
`orientation_count > 0`; `junction_mask` is a subset of it.

Invalid pixels retain a schema-valid neutral target: uniform orientation
probability, zero component count and frequency, and false junction status.
They must still be excluded from metrics through `valid_mask`; the uniform
distribution is a storage convention, not orientation evidence.

`BenchmarkMetadata` records benchmark/generator versions, stable sample ID,
split-domain seed, family, difficulty, degradation, orientation bins, component
capacity, and recursively immutable JSON-compatible parameters. `to_dict()`
returns plain serializable containers. `BenchmarkSample.to(...)` returns
independently stored tensors and keeps integer/mask dtypes intact.

## Orientation and frequency ground truth

Analytic component orientations use the project convention: axial radians in
`[0,π)`, `x-right,y-down`, with positive angles counter-clockwise on the
displayed image. A continuous component angle is linearly splatted between its
two neighboring axial bins with periodic wrap. At a multi-component pixel,
each component contributes equal orientation mass before normalization.

Frequency is stored per analytic component in cycles/pixel. It is not inferred
from the rendered image. This permits exact tests of frequency changes,
crossings, and degradation synchronization.

## Canonical families

The frozen v0.1 vocabulary contains fourteen project-authored analytic scenes:

| Family | Controlled structure |
| --- | --- |
| `straight` | Single constant orientation and frequency plane pattern. |
| `parallel` | Parallel band pattern with controlled duty cycle. |
| `curved` | Parabolic level sets with analytic local tangent and frequency. |
| `sinusoidal` | Sinusoidally curved level sets and analytic local tangent. |
| `radial` | Radial spokes, with the polar singularity marked invalid. |
| `concentric` | Concentric rings with tangential orientation. |
| `crossing` | Two orientation/frequency corridors crossing at one junction. |
| `dual_crossing` | Two spatially separated two-component crossings. |
| `bifurcation` | One parent and two branches with a three-component junction. |
| `junction` | Three directed arms meeting at a junction. |
| `varying_frequency` | One orientation field with analytic spatial frequency variation. |
| `varying_thickness` | Constant frequency with a spatially varying band duty cycle. |
| `domain_boundary` | Two orientation/frequency domains separated by an invalid boundary. |
| `occlusion` | A single field with an explicit invalid elliptical occlusion. |

Difficulty changes transparent family parameters such as curvature, angular
separation, band width, invalid extent, or contrast. The generator stores every
resolved analytic parameter in metadata.

## Deterministic split identity

Train, validation, and test use pairwise disjoint half-open seed domains of size
`2^50`. A seed is derived arithmetically from:

```text
split domain + base-seed block + canonical family offset + sample index
```

Python's process-randomized `hash()` is never used. Each family permits one
million indices per base-seed block, with validated bounds. Difficulty variants
reuse the same local random draw to support paired comparisons, but difficulty
is included in `sample_id`, so their identities remain distinct.

Generation uses a function-local CPU `torch.Generator` and does not consume the
global random stream. Repeating the same identity tuple yields identical tensor
and metadata content.

## Degradation contract

`apply_degradation(sample, config, seed=...)` is pure: it does not mutate the
input and every returned tensor has independent storage. The degradation
sample ID includes name, difficulty, explicit seed, and a stable SHA-256 digest
of resolved parameters. Metadata preserves the source sample identity and
records all realized settings.

Image-only degradations preserve GT values exactly:

```text
gaussian, poisson, speckle, salt_and_pepper,
gaussian_blur, motion_blur, contrast, brightness, jpeg
```

JPEG uses an actual in-memory Pillow JPEG encode/decode with recorded quality;
it is not mislabeled quantization. No file is written.

Motion-blur angles follow the same displayed-image counter-clockwise convention
as structural orientations; tensor rows increasing downward are accounted for
when constructing the kernel.

Geometry-aware degradations synchronize GT:

- `rotation` uses the Stage-7 same-canvas transform, spatially warps the field,
  shifts axial orientation by the displayed-image angle, and excludes padding
  or incomplete source footprints.
- `scaling` applies the same spatial rule and divides component frequency by
  scale (`0.10` at scale `2` becomes `0.05`).
- degradation `occlusion` zeros the chosen image rectangle and replaces every
  affected GT field with the neutral invalid convention.

After a geometric transform, component count is the rounded interpolated
discrete count on the strict valid footprint, and junction status is derived
from `orientation_count >= 2`; it is not thresholded independently. Bilinear
orientation-distribution transport can remain softly multimodal immediately at
a component-support boundary even where the discrete rounded count is one.
This is a documented interpolation boundary convention, and callers should use
the count and valid mask for discrete component decisions.

Gaussian, Poisson, speckle, salt-and-pepper, and occlusion randomness uses an
explicit local per-device generator. Determinism is guaranteed for a repeated
operation on the same device; CPU and CUDA random streams are not promised to
match bit-for-bit.

## Provenance and limitations

The schema, split mapping, analytic families, renderer, degradation wrappers,
and tests are project-authored independent implementations. They use PyTorch,
Python standard-library primitives, NumPy for a decoded array boundary, and
Pillow for the JPEG codec. No external dataset, image, generator source,
checkpoint, or pretrained weight is copied or vendored.

The images are controlled synthetic evidence. They do not establish external
validity, calibration, downstream task performance, or benchmark superiority.
Whole-repository licensing and dependency notices remain subject to the final
provenance/dependency gates and the Apache-2.0/third-party boundaries in `NOTICE.md`.
