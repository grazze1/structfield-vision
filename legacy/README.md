# V0.1 Legacy Evidence

This directory is a compact historical index for the pre-StructField project.
It is not an active runtime, compatibility package, benchmark result for V0.2,
or weight release.

The complete frozen V0.1 source is available from the annotated Git tag:

```bash
git show legacy-v0.1:README.md
git archive legacy-v0.1
```

The tag resolves to commit
`e39f681c771e7cc0f7846293cb280c1121bd1357`. No historical checkpoint,
dataset, organizer input/label, prediction, clinical table, or submission is
included in this repository.

- `competition_study.md` records the separately calibrated competition route
  and its limitations.
- `result_manifest.json` preserves selected V0.1 facts with explicit evidence
  classification.
- `migration_notes.md` explains what was semantically rewritten, excluded, or
  deferred in V0.2.

All numbers here are **Legacy Evidence**. They cannot be presented as
performance of the V0.2 models, estimator, StructuralField schema, or
StructField-Bench.
