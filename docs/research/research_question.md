# Research question

## Central question

> When should neural networks trust explicit structural priors?

StructField-Vision treats this as an empirical question, not as a premise. An
explicit structural prior is a caller-supplied `StructuralField` estimated from
an image before a task model uses it. “Trust” has the operational meaning of
allowing that field to influence task features through a learned, inspectable
residual gate. It does not mean human trust, a calibrated probability, or a
guarantee that the field is correct.

The conceptual path is:

```text
Image → Probabilistic Structural Field → Structural Uncertainty → Reliability Gate → Task Backbone → Prediction
```

This path is a research abstraction. In the implementation, the image and the
precomputed field enter the application through separate, explicit arguments;
the estimator is not hidden inside the backbone. The exact execution boundary
is documented in [Method](method.md).

## Operational scope

The question is decomposed into four observable quantities:

1. **Prior content:** axial orientation distributions, spatial frequency,
   scale, and coherence represented by a validated probabilistic field.
2. **Prior ambiguity:** a task-neutral heuristic structural-uncertainty map.
3. **Prior use:** the task-specific spatial-channel gate and its gated
   structural residual.
4. **Outcome:** task error, structural-field error, or an explicitly defined
   binary failure event, always evaluated on a declared valid region.

These quantities are intentionally distinct. Structural uncertainty is not the
gate; the gate is not a calibrated confidence map; and a low task error does not
prove the field itself is accurate. See [Uncertainty](uncertainty.md) and
[Reliability gating](reliability_gating.md).

## Testable statements

### Research Hypothesis H1 — selective use

When a useful structural field varies in quality across an image, a
reliability-gated residual adapter will be less sensitive to locally wrong
priors than fixed concatenation or ungated conditioning, while retaining useful
prior effects in reliable regions.

This statement remains unverified. It requires matched training and evaluation
under the frozen comparison matrix in
[Experimental protocol](experimental_protocol.md).

### Research Hypothesis H2 — ambiguity and failure ranking

Structural uncertainty and a separately predicted failure score will contain
information about caller-defined downstream failures. The two scores must be
reported separately because one is a deterministic field heuristic and the
other is a learned task signal.

This hypothesis concerns ranking against explicit labels. It does not assert
probability calibration or causal explanation.

### Research Hypothesis H3 — geometric consistency

Geometry-aware consistency losses may reduce disagreement between transformed
image branches on valid pixels. The implemented losses test approximate
covariance of selected field signals; they do not establish exact architectural
group equivariance. See [Equivariance](equivariance.md).

### Research Hypothesis H4 — multimodal representation

A posterior over axial orientations may preserve useful ambiguity at crossings,
junctions, and weak-evidence regions that a single orientation angle would
discard. The controlled benchmark can test this representation-level claim
before any real-data generalization claim is considered.

## Evidence ladder and current status

| Evidence level | Current status | What it can support |
| --- | --- | --- |
| Typed field, estimator, adapters, corruptions, transforms, metrics | Implemented and software-tested | API and numerical-contract claims |
| Deterministic procedural benchmark with analytic targets | Implemented | Controlled structural-error studies |
| Frozen V0.3 comparison configurations | Defined with `contains_results: false` | A-priori experiment definitions |
| Formal long-duration task training | Not run for this research program | No task-performance conclusion |
| Licensed external real-data evaluation | Not run and no external data shipped | No external-validity conclusion |

Passing unit or CUDA tests establishes software behavior only. It is not a
substitute for trained comparisons, uncertainty calibration, robustness
evidence, or real-data validation.

## Reading map

- [Method](method.md) defines the end-to-end software and research boundary.
- [Structural field](structural_field.md) fixes the probabilistic representation.
- [Uncertainty](uncertainty.md) defines the deterministic ambiguity baseline.
- [Reliability gating](reliability_gating.md) separates uncertainty, reliability,
  and task-specific prior use.
- [Benchmark](benchmark.md) defines controlled evidence and real-data limits.
- [Experimental protocol](experimental_protocol.md) records the result-free
  V0.3 comparison plan.
- [Related work](related_work.md) records the verified concept map and open
  literature-review tasks.
