# Migrating from 0.3 to 1.0

Version 1 intentionally removes the broad 0.3 top-level re-export surface.
Import metrics, conventions, visualization, datasets, and benchmark objects
from their stable domain modules. Import research components through
`structfield.experimental`.

`StructuralField.orientation_topk` was removed. Replace reads with:

```python
modes = field.topk_orientations(k=2)
angles = modes.angles
probabilities = modes.probabilities
```

Construct fields with `orientation_prob` as the sole orientation state. The
field schema is now `1.0`; config schemas are `1.0`; CLI bundles are
`structfield-estimate-v1`. The benchmark, application, and checkpoint schema
identifiers remain unchanged.

The CLI adds `--config`, `--version`, and an optional `--num-orientations`
override. Its output includes `orientation_probability.npy` and integrity
metadata. Resolver validation is stricter, so bundles manually modified after
publication are rejected.

Safe checkpoint manifests carrying field schema `0.2` can only be loaded by
explicitly passing `migrate_legacy_field_schema=True`. Unsafe legacy pickle
checkpoints remain unsupported.
