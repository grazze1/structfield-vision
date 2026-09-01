# Method

StructField-Vision maps a finite grayscale or RGB tensor `[B,C,H,W]` to a
validated probabilistic structural field. The analytic estimator combines
multi-scale structure-tensor evidence with complex Gabor responses on uniform
axial bins. It marginalizes the joint orientation-frequency response into a
normalized orientation posterior, frequency moments, coherence, entropy, and
an explicitly named baseline heuristic uncertainty map.

`orientation_prob` is the only persisted orientation state. Top modes are
derived with `topk_orientations()` using axial distance modulo π, so adjacent
bin leakage is not automatically treated as a separate direction. Optional
diagnostics remain outside `StructuralField` and are created only when the
caller requests them.

The estimator is differentiable through continuous tensor operations. Mode
selection and stochastic experimental corruptions include discrete choices.
Application models consume a caller-supplied field and never hide estimation,
resizing, casting, or device transfer inside `forward`.

Adapters, corruptions, equivariance losses, the heuristic uncertainty baseline,
and the three lightweight applications are experimental research references.
They support controlled software experiments; they are not pretrained models
or evidence of downstream benefit. See [Concepts](../concepts.md),
[Uncertainty](uncertainty.md), and [Benchmark](benchmark.md).

No formal long-duration training or external-data study is part of v1
qualification. Any claim that reliability-aware conditioning improves a task
remains a research hypothesis requiring a separately governed experiment.
