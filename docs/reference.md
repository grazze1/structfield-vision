# Stable and experimental boundaries

The stable top-level API is intentionally small. Stable domain modules are
`core`, `benchmarks`, `metrics`, `configs`, `datasets`, `checkpoint`, `cli`, and
`visualization`.

Adapters, corruption, equivariance and reliability losses, heuristic
uncertainty, and the three reference applications are experimental. Import them
through `structfield.experimental`; they do not receive the v1 compatibility
guarantee.

The application and checkpoint schemas remain
`structfield-applications-v0.3` and
`structfield-application-checkpoint-v1`. The CLI bundle schema is
`structfield-estimate-v1`; the benchmark schema remains
`StructField-Bench-v0.1` and is supported by software v1.
