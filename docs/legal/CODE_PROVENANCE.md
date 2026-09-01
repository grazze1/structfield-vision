# Code provenance

This inventory covers the active `structfield/`, `scripts/`, and `examples/`
release surface. Classification is reviewed at release time.

| Class | Meaning | v1 status |
|---|---|---|
| A | Project-authored implementation | Included |
| B | Independent implementation based on published ideas | Included with references in documentation |
| C | Vendored permissive third-party source | None |
| D | Vendored source requiring separate terms | None |
| E | Unknown or incompatible provenance | Excluded |

The analytic estimator, schema validation, benchmark renderer, metrics, CLI,
checkpoint format, examples, and release scripts are A/B implementations. The
historical clDice code and provenance-uncertain NAFNet-like code are not present
in the active package or distribution. Dependencies are installed separately
and are not vendored.

No source file classified C, D, or E is shipped in the v1 wheel or sdist.
