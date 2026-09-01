# Packaging and configuration contract

`structfield-vision==1.0.0` is an Apache-2.0 Python distribution.

The wheel publishes these namespaces:

```text
structfield
structfield.adapters
structfield.applications
structfield.benchmarks
structfield.cli
structfield.configs
structfield.core
structfield.corruption
structfield.datasets
structfield.estimators
structfield.losses
structfield.metrics
structfield.transforms
structfield.uncertainty
structfield.visualization
```

Generic top-level `applications` and `benchmarks` packages are intentionally
absent. `structfield.applications` uses schema
`structfield-applications-v0.3`; the structural-field and benchmark schemas
remain `0.2` and `v0.1`.

The wheel includes `LICENSE`, `NOTICE.md`, and the eight active YAML resources
listed by `structfield.configs.available_configs()`. It excludes tests,
datasets, internal handoff/audit documents, historical templates, and
reproducibility-only configurations.

escnn and BM3D are research-only dependencies. They are absent from
distribution extras; BM3D remains in an explicitly noncommercial requirements
file with its independent upstream terms.
