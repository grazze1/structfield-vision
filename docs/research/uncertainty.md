# Structural uncertainty

`HeuristicStructuralUncertainty` is an experimental deterministic baseline,
not a calibrated probability. It combines normalized orientation entropy,
separated-mode ambiguity, normalized frequency variance, one minus coherence,
and optional scale entropy. Missing optional components are omitted and the
active weights are renormalized.

Three quantities remain distinct:

| Quantity | Meaning |
| --- | --- |
| Coherence | Local structure-tensor anisotropy evidence |
| Structural uncertainty | Task-neutral heuristic ambiguity in the field |
| Reliability gate | Task-specific learned amount of structural residual use |

A gate may depend on uncertainty but is not defined as `1 - uncertainty`.
Likewise, an optional predicted-reliability output is a separate learned score.
Calibration or failure-prediction claims require an explicit target, mask,
threshold, binning policy, and held-out evaluation; v1 makes no such empirical
claim.

Controlled corruptions preserve the existing uncertainty map unless
recomputation is explicitly requested. This allows prior error and stated
uncertainty to be varied independently in research tests.
