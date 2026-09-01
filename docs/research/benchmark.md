# Benchmark and evidence boundary

`StructField-Bench-v0.1` is the stable v1-supported synthetic benchmark format.
It generates fourteen deterministic analytic families with orientation,
frequency, component-count, validity, junction, seed, degradation, and family
metadata. Invalid pixels contain neutral stored values and must be excluded by
the explicit valid mask.

Train, validation, and test identities use disjoint arithmetic seed domains.
Photometric degradation preserves targets; rotation, scaling, and occlusion
update every affected target and record resolved parameters. Metrics include
axial angular error, orientation recall, frequency MAE, orientation-distribution
Jensen–Shannon divergence, calibration diagnostics, and failure ranking.

The release regression baseline records one deterministic sample from every
family in `benchmarks/v1.0.0_baseline.json`. Its tolerances detect numerical
software regressions; they are not scientific performance claims.

No external data, weights, clinical tables, or dataset-specific support claims
are distributed. The dataset API only discovers and validates caller-owned
local files. Real-data identity, license, privacy, split, and evaluation
decisions remain the caller's responsibility.
