# V0.1 to V0.2 Migration Notes

## Compatibility policy

V0.2 is a semantic migration, not a path-preserving or checkpoint-compatible
release. The frozen tag is the compatibility boundary. There are no forwarding
packages for `q1_denoising`, `q2_restoration`, `q3_segmentation`, or
`structure`; old imports must be run from `legacy-v0.1`.

Old state dictionaries must not be loaded with `strict=False` into V0.2. Even
when a tensor shape happens to match, orientation, frequency, scale,
confidence/uncertainty, feature-channel, adapter, and package semantics differ.
Any future V0.2 checkpoint format must identify its application schema,
StructuralField schema, field-feature schema, adapter mode, and estimator
configuration. This stage provides no checkpoint converter.

## Application mapping

| Frozen V0.1 area | Active V0.2 area | Disposition |
| --- | --- | --- |
| `q1_denoising` | `applications/fringe_restoration` | Dual encoder, residual U-Net, explicit StructuralField conditioning, and noise FiLM semantically rewritten. |
| `q2_restoration` | `applications/texture_restoration` | Dual encoder and Fourier-bottleneck residual U-Net semantically rewritten. |
| `q3_segmentation` | `applications/curvilinear_segmentation` | Plain U-Net baseline plus explicit StructuralField adapter boundary independently rewritten. |
| `structure` | `structfield` | Replaced by the typed probabilistic V0.2 core and analytic estimator. |
| synthetic Q1/Q2 generators | `benchmarks/structfield_bench` | Replaced by versioned, independently written benchmark families and degradations. |
| basic evaluation concepts | `structfield.metrics` and application losses | Independently rewritten with explicit masks and conventions. |
| training, inference, calibration, and campaign scripts | frozen tag only | Not migrated in Stage 10; no long training or competition runtime is retained. |

## Deliberate exclusions

- The provenance-uncertain historical NAFNet-like blocks are legacy-only.
- The historical clDice implementation was classified as derived from the
  MIT-licensed `jocpae/clDice` reference and is excluded from active V0.2. Its
  old numerical results remain historical facts, not active metric support.
- Qilu/SNP-Net loaders, private paths, labels, clinical tables, checkpoints,
  outputs, and pretrained weights are not migrated.
- BM3D remains an external non-commercial historical option and is neither
  vendored nor required by the core.
- Old dataset defaults, training loops, inference scripts, formal submission
  naming, and organizer-public calibration are not active V0.2 interfaces.

## Scientific interpretation

V0.1 results motivated the new question—when should a downstream network trust
an explicit structural prior?—but they do not evaluate the V0.2 estimator,
uncertainty, corruption, equivariance, adapters, applications, or benchmark.
Only new experiments under a disclosed V0.2 protocol may make V0.2 performance
claims.
