# Metrics and Failure Analysis

## Scope

Stage 9 provides small, stateless reference metrics for structural-field
evaluation. They are transparent evaluation primitives, not accumulated
training statistics, dataset-specific claims, or evidence that an uncertainty
score is calibrated. Every metric is an `nn.Module` with no parameters or
buffers and returns one scalar tensor.

All public `forward` methods require an explicit boolean `valid_mask` with
shape `[B,1,H,W]`. At least one pixel must be valid. Invalid inputs may contain
sentinel values, including NaN or infinity: implementations remove masked
locations before subtraction, modulo, logarithms, sorting, or reduction.
NaN/infinity at a valid location is rejected.

Inputs must be floating BCHW tensors on one device and, where two field values
are compared, must use the same dtype. Float16 and bfloat16 inputs accumulate
in float32; float32 and float64 retain their dtype. The metrics run on CPU and
CUDA and explicitly disable surrounding autocast within their calculations.

## Metric definitions

| Metric | Definition and unit | Required evaluation choices |
| --- | --- | --- |
| `AxialAngularError` | Mean shortest angular distance modulo π, in radians; range `[0,π/2]`. | Single predicted and target angle per pixel. |
| `OrientationRecallAtK` | Fraction of active target modes matched by at least one predicted mode, averaged over valid pixels. | Constructor requires an angular threshold in radians. Optional integer `target_count` declares that only the first count target channels are active. |
| `FrequencyMAE` | Mean absolute frequency error across channels, then valid pixels. | Inputs must already share units and channel correspondence. The project convention is cycles/pixel. |
| `OrientationDistributionJS` | Jensen–Shannon divergence between channel-wise orientation PMFs using natural logarithms; range `[0,ln(2)]`. | Both distributions must be non-negative and sum to one at valid pixels. |
| `EquivarianceError` | Channel-mean L1 error between two already aligned field branches. | The caller is responsible for geometric and semantic alignment. |
| `OrientationCalibrationECE` | Equal-width-bin expected calibration error for binary axial-angle correctness. | Constructor requires both the angular correctness threshold in radians and bin count. Confidence must lie in `[0,1]`. |
| `FailureDetectionAUROC` | Tie-aware ranking area for explicit binary failure labels; equal-score positive/negative pairs contribute `0.5`. | Scores must increase with likelihood of failure. The caller supplies the 0/1 failure target. |
| `FailureDetectionAUPRC` | Grouped-unique-threshold average precision, `Σ precision × Δrecall`. | Scores must increase with likelihood of failure. The caller supplies the 0/1 failure target. |

Angles are axial: `0` and `π` denote the same axis. Angular correctness is
inclusive (`error <= threshold`). ECE uses equal-width bins over `[0,1]` that
are left-closed/right-open, except confidence `1` belongs to the final bin.

## Failure targets and undefined cases

The failure metrics never synthesize a label and contain no hidden quality
threshold. A study must define the failure event, construct
`failure_target[B,1,H,W]`, and disclose that choice. Reversing label semantics
without reversing the score direction changes the measurement as expected.

AUROC requires at least one valid failure and one valid non-failure. AUPRC
requires at least one valid failure; an all-positive set is defined as `1`.
When every score is tied, grouped AUPRC equals failure prevalence. These edge
conditions raise an explicit error where the statistic is undefined rather
than returning a misleading NaN.

## Interpretation limits

- ECE depends on both its correctness threshold and bin count. Comparisons are
  meaningful only when those choices and the evaluated population are held
  fixed.
- AUROC and AUPRC measure ranking against a caller-defined failure event; they
  do not establish calibrated probabilities or a causal failure mechanism.
- Recall@K depends on the angular tolerance and target-mode annotation policy.
- Equivariance error assumes the caller has already transformed and aligned
  both branches under the Stage-7 convention.
- Scalar averages can hide family, corruption, junction, or difficulty
  subgroups. Later experiments should report relevant stratified results.

## Provenance

The metric implementations, tests, and definitions are project-authored.
They use PyTorch tensor operations and Python standard-library primitives. No
scikit-learn implementation, external evaluation source, dataset, checkpoint,
or pretrained weight is copied or vendored.
