# StructuralField schema

## Purpose

`StructuralField` is a validated, task-neutral batch representation. It contains structural estimates only; restoration targets, segmentation masks, model features, and competition metadata are forbidden.

## Tensor contract

| Field | Shape | Range / unit |
|---|---|---|
| `orientation_prob` | `[B,K,H,W]` | non-negative; sum over `K` is one |
| `frequency_mean` | `[B,1,H,W]` | non-negative cycles/pixel |
| `frequency_var` | `[B,1,H,W]` | non-negative `(cycles/pixel)²` |
| `coherence` | `[B,1,H,W]` | `[0,1]` |
| `orientation_entropy` | `[B,1,H,W]` | normalized `[0,1]` |
| `structural_uncertainty` | `[B,1,H,W]` | `[0,1]` |
| `scale_prob` | optional `[B,S,H,W]` | non-negative; sum over `S` is one |
| `selected_scale` | optional `[B,1,H,W]` | non-negative pixels |

All present tensors share floating dtype, device, batch size, and spatial dimensions. The metadata orientation count and optional scale candidates must match tensor channels.

## Metadata

`FieldMetadata` records orientation bin centers, frequency/scale candidates, units, coordinate/rotation conventions, estimator/schema versions, and JSON-safe extra facts. It contains no tensor and no private path.

## Diagnostics boundary

`StructuralFieldDiagnostics` may carry normalized input, joint orientation–frequency response/probability, marginal responses, tensor eigensystem maps, and optional Hessian maps. Diagnostics are analytic evidence, not stable task inputs. Keeping them separate prevents task-specific channels from expanding the public field schema.

## Validation and immutability

- Construction validates shape, finite values, probability normalization, units, ranges, dtype, and device.
- `.to(...)` returns a new field or diagnostics object and converts every tensor consistently.
- Dataclasses are frozen and APIs do not mutate inputs. Contained tensors retain normal PyTorch mutability; callers must not modify shared tensors in place.
- Invalid inputs raise `TypeError` or `ValueError` at the schema boundary rather than fail later inside an estimator or model.

## Orientation modes

`topk_orientations(k, min_separation)` dynamically selects probability modes using axial circular non-maximum suppression. No top-k tensor is persisted. It returns typed angles, probabilities, and bin indices. The selection is discrete; gathered probabilities remain differentiable with respect to the selected probability values.
