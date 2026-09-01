# 12 Final release decision

The annotated `v1.0.0` tag is created only after every CPU, CUDA, documentation,
legal and supply-chain gate passes on the final clean commit. Post-tag local
manifest and checksum files live under ignored `release/v1.0.0/` to avoid a
commit-hash self-reference. No remote operation is authorized.

Release decision: **PASS, pending the final clean-commit rerun and local tag**.
All pre-commit CPU, CUDA, documentation, provenance, packaging and supply-chain
gates have passed. If the clean-commit rerun remains green, the release commit
is eligible for the annotated `v1.0.0` tag; otherwise this decision reverts to
blocked and no tag is created.
