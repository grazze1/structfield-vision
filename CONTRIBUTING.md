# Contributing

Use Python 3.11 on Linux x86_64. Install the development profile, run Ruff,
Mypy, the non-CUDA test suite, strict MkDocs build, and distribution checks
before proposing a change. New public behavior needs tests, docstrings, units,
shape/range contracts, and a changelog entry.

Stable APIs follow SemVer. A deprecation must emit a documented warning for at
least one minor release before removal, except when immediate removal is needed
to close a security issue or prevent unsafe behavior. Experimental APIs may
change in minor releases but still require release notes.

Do not commit external data, pretrained weights, clinical material, credentials,
private paths, or artifacts whose redistribution rights are unclear.
