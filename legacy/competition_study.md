# Historical Competition Study

This document records V0.1 output lineage only. Organizer-public labels were
used for calibration, adaptation, or selection in these routes. Public-example
metrics are not independent hidden-set results, and no official hidden score or
rank is available. The implementations remain only at `legacy-v0.1`; they are
not part of the StructField core or active V0.2 applications.

## Fringe restoration

The independently selected seed-42 `ConditionedRestorer` was followed by
per-image moment matching to the mean (`0.49997109`) and standard deviation
(`0.35371420`) of an organizer-public label, then clipping. The checkpoint had
independent synthetic validation evidence, but the submitted output
distribution was public-label calibrated.

## Texture restoration

A 75-epoch normalized-structure seed-3407 model was adapted for 250 supervised
steps on one organizer-public pair with synthetic replay. The submitted route
used full adapted weights (`alpha=1.0`) and an affine contrast transform. Its
public-example MAE was `0.06971558`, while recorded synthetic validation/OOD
MAE worsened to `0.02544452`/`0.02495769`. A predeclared OOD-tolerance rule
would instead have selected `alpha=0.3`; therefore the submitted route is not
the independent benchmark.

## Curvilinear segmentation

A geometry-multitask seed-2026 source model was supervised on an
organizer-public example and interpolated at `alpha=0.75`. Historical inference
used CLAHE, three scales, probability averaging, a fixed threshold, component
filtering, and dilation. Public-example Dice was `0.80345090`. Complete
adaptation reduced recorded internal-test mean-image Dice from `0.81354166` to
`0.77788723`; no independent alpha-0.75 test result was archived.

## Reproduction boundary

The V0.1 tag contains the historical scripts and exact source paths. They
require user-supplied organizer examples and private checkpoints. V0.2 neither
ships those assets nor keeps an executable competition runtime on its active
branch. Reconstructing the historical workflow requires an explicit checkout
of `legacy-v0.1` and a separate rights/security review of all inputs and
checkpoints.
