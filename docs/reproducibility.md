# Reproducibility contract

Version 1.0.0 separates deterministic software checks from scientific
experiments that require external data.

## Release environments

The release uses Python 3.11 and two PEP 751 lock files:

- `pylock.cpu.toml`: Linux x86_64, PyTorch `2.13.0+cpu`
- `pylock.cu126.toml`: Linux x86_64, PyTorch `2.13.0+cu126`

They are generated with `uv 0.12.3` using the corresponding PyTorch backend.
Each lock has a deterministic CycloneDX 1.6 SBOM and third-party license
inventory under `sbom/`. PyTorch 2.13.0 is the minimum supported runtime and
the first available release whose complete CPU and CUDA 12.6 lock profiles
pass the release vulnerability gate. CUDA 12.6 replaces the 12.8 profile
because PyTorch 2.13 does not publish a CUDA 12.8 release wheel.

## Packaged configurations

Only four active analytic-estimator configurations and four active experiment
protocols are included in the wheel. Consumers must use:

```python
from structfield.configs import available_configs, read_config_text
```

The API returns text for a canonical resource ID. It does not expose a
persistent filesystem path and does not require PyYAML. Historical templates
and reproduction-only configurations remain repository material, not wheel
resources.

## Deterministic checks

The CPU gate covers Ruff, compileall, pytest, wheel and sdist construction,
isolated installation, CLI publication, resource loading, package contents,
SBOM/license reproduction, vulnerability audit, and `twine check`. It does not
read external data or perform long training.

Application checkpoints record estimator configuration, preprocessing,
training provenance, application constructor including validation mode,
producer version, weight size, SHA-256, and state-dict shape/dtype signatures.
