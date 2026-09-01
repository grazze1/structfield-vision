# 06 CPU qualification

Qualification uses a fresh `/tmp` Python 3.11 environment synchronized from
`pylock.cpu.toml` by uv 0.12.3. The final record includes Ruff, formatting,
Mypy, compileall, tests/coverage, examples, benchmark, performance, docs,
distributions, Twine, wheel audit and isolated installation.

Final qualification result: **PASS** on Python 3.11.15 with PyTorch
2.13.0+cpu. The public CPU suite reported 820 passed and 80 CUDA-only skips.
Coverage over `structfield` was 92%; the core field, estimator, benchmark,
checkpoint, and CLI critical paths each met or exceeded 90%. The six examples,
14-family regression baseline, CPU performance recorder, strict documentation
build, sdist/wheel build, Twine check, wheel-content audit, and clean wheel
installation all passed.
