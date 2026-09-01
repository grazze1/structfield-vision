# StructField-Vision overview

StructField-Vision asks:

> **When should neural networks trust explicit structural priors?**

The V0.2 repository supplies the testable infrastructure around that question:
a checkpoint-free analytic estimator, a typed probabilistic field, uncertainty
and reliability signals, prior adapters, controlled corruptions, geometric
consistency losses, structural metrics, deterministic synthetic benchmarks,
and three compact task-backbone integrations.

## Research flow

```text
Image
  ↓
Probabilistic Structural Field
  ↓
Structural Uncertainty
  ↓
Reliability Gate
  ↓
Task Backbone
  ↓
Prediction
```

The field represents axial orientation as a categorical distribution over
`[0, π)`, plus top modes, local frequency, scale, coherence, and heuristic
structural uncertainty. Valid support is carried separately by benchmark or
geometric-transform masks rather than stored in `StructuralField`. A
reliability-gated adapter can attenuate a prior before it enters a task model.
This creates a measurable alternative to always trusting or always omitting an
explicit prior.

## Implemented surfaces

- `structfield.core`: field schema, validation, device/dtype movement, and
  axial conventions.
- `structfield.estimators`: analytic tensor/Gabor/frequency/Hessian components
  and their orchestrating estimator.
- `structfield.uncertainty`, `structfield.adapters`, and `structfield.losses`:
  heuristic uncertainty, fixed/adaptive conditioning, reliability supervision,
  and equivariance objectives.
- `structfield.corruption`, `structfield.transforms`, and
  `structfield.metrics`: controlled failure injection, geometry-aware field
  transforms, calibration/failure/field metrics.
- `structfield.benchmarks`: deterministic synthetic families,
  degradations, splits, and structured sample metadata.
- `applications`: fringe restoration, texture restoration, and curvilinear
  segmentation model/loss integrations.
- `structfield-estimate`: local PNG/NPY analytic estimation on CPU, CUDA, or
  automatic device selection.

## Evidence boundary

V0.2 has unit, integration, CPU, and GPU smoke/regression evidence. It has not
performed a formal long-training campaign and reports no V0.2 task-performance
claim. The experiment matrix under `configs/experiments/` is frozen before such
results are collected.

Earlier task measurements are not part of the public repository or release
artifacts. They do not validate the current schema, estimator, adapters,
applications, or benchmark.

## Navigation

- Research framing: [`research/`](research/)
- Architecture contracts: [`architecture/`](architecture/)
- Local data boundary: [`datasets.md`](datasets.md)
- Reproducibility: [`reproducibility.md`](reproducibility.md)
- Limitations: [`limitations.md`](limitations.md)
- Refactor evidence and stage reports: [`refactor_v02/`](refactor_v02/)
