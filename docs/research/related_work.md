# Related work and concept provenance

## Scope

This document is a verified concept map for the implemented V0.2 components. It
is not a comprehensive literature review and does not position the V0.3
research hypotheses against every relevant method. Entries below are limited
to references and dependencies already recorded in the repository.

No cited publication is a source of copied implementation code. The active
components are independent implementations built from PyTorch and standard
Python interfaces; repository-wide licensing remains governed by the project
provenance audit and license notices.

## Analytic local structure

The estimator combines established analytic ideas rather than learning the
field end to end:

- Bigün, Granlund, and Wiklund, “Multidimensional orientation estimation with
  applications to texture analysis and optical flow” (1991),
  [DOI 10.1109/34.85668](https://doi.org/10.1109/34.85668), provides context for
  structure-tensor orientation analysis.
- Gabor, “Theory of communication. Part 1: The analysis of information” (1946),
  [DOI 10.1049/ji-3-2.1946.0074](https://doi.org/10.1049/ji-3-2.1946.0074),
  provides context for localized frequency-selective filters.
- Frangi et al., “Multiscale vessel enhancement filtering” (1998),
  [DOI 10.1007/BFb0056195](https://doi.org/10.1007/BFb0056195), provides context
  for the optional Hessian line diagnostics.

StructField-Vision's stable output is a probabilistic axial field with explicit
frequency and scale summaries. The exact project schema, normalization,
posterior construction, and diagnostics boundary are described in
[Structural field](structural_field.md); the citations above do not validate
those project-specific choices.

## Entropy and structural ambiguity

Normalized categorical entropy follows the standard information-entropy
definition associated with Shannon, “A Mathematical Theory of Communication”
(1948),
[DOI 10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x).

The separated-mode term, bounded frequency-variance normalization,
missing-component renormalization, selected weights, and their combination are
project baseline choices. They have not been fitted or calibrated. Their role
and limits are explicit in [Uncertainty](uncertainty.md).

## Feature conditioning and task backbones

- Perez et al., “FiLM: Visual Reasoning with a General Conditioning Layer”
  (AAAI 2018),
  [AAAI article 11671](https://ojs.aaai.org/index.php/AAAI/article/view/11671),
  provides the feature-wise affine-conditioning concept used by the lightweight
  affine adapter.
- Ronneberger, Fischer, and Brox, “U-Net: Convolutional Networks for Biomedical
  Image Segmentation” (2015),
  [arXiv:1505.04597](https://arxiv.org/abs/1505.04597), provides the high-level
  encoder/decoder concept used by the reference applications.
- Milletari, Navab, and Ahmadi, “V-Net: Fully Convolutional Neural Networks for
  Volumetric Medical Image Segmentation” (2016),
  [arXiv:1606.04797](https://arxiv.org/abs/1606.04797), provides context for a
  Dice-style training objective.

The residual reliability gate, its spatial-channel output contract, and the
separation of uncertainty from task-specific prior use are project baseline
choices. Their scientific value remains a hypothesis; see
[Reliability gating](reliability_gating.md).

## Geometric consistency

The current implementation uses PyTorch affine sampling, explicit semantic
transport of the structural field, strict validity masks, and symmetric
consistency losses. It tests approximate covariance/equivariance relations; it
does not provide an exact group-equivariant architecture.

`escnn` is listed only in a research requirements file for planned comparison rows; it is not a distribution extra.
No `escnn` architecture, representation audit, trained baseline, or functional
validation is part of the current core evidence. A dedicated group-equivariant
literature review and implementation review are required before those rows can
be reported. See [Equivariance](equivariance.md).

## Controlled benchmarks and real data

The procedural families, analytic target renderer, deterministic split mapping,
and degradation wrappers in `StructField-Bench-v0.1` are project-authored. They
do not import an external generator or dataset. Synthetic control supports
mechanism and error analysis but does not establish external validity.

The repository includes a path-enumeration boundary for caller-owned local
files, not distributed real data. Data-source literature, domain-specific
annotation standards, and external benchmark protocols must be reviewed only
after a lawful dataset and intended task are selected. The boundary is defined
in [Benchmark](benchmark.md).

## Research Hypothesis — combined framework

Representing an explicit prior as a probabilistic axial field, exposing a
task-neutral ambiguity signal, and learning an inspectable residual-use gate may
provide a useful framework for studying when structural conditioning helps or
harms a task model.

This is a research hypothesis. The listed references motivate individual
concepts but do not establish the combined framework's effectiveness,
distinctiveness, or generality.

## Literature-review work required before a formal paper claim

1. Review probabilistic and multimodal orientation-field estimators using a
   documented search strategy and date range.
2. Review uncertainty, selective prediction, confidence calibration, and
   failure-detection work relevant to dense vision.
3. Review adaptive prior fusion and reliability gating beyond feature-wise
   affine conditioning.
4. Review group-equivariant and steerable architectures under the exact axial
   representation and transformation group used by the study.
5. Review robustness evaluation under corrupt or misspecified auxiliary
   inputs, separating observation corruption from prior corruption.
6. Review domain-specific methods only after a licensed real-data task is
   selected, then freeze inclusion/exclusion criteria before positioning the
   results.

Until those tasks are complete, the repository should describe its claims using
the explicit `Research Hypothesis` labels in
[Research question](research_question.md) and the result-free protocol in
[Experimental protocol](experimental_protocol.md).
