# CLI publication and visualization

`structfield-estimate` accepts one local PNG or NPY input and performs strict
input validation before analytic estimation. Its estimate metadata schema
is `structfield-estimate-v1`.

## Published layout

```text
output/
+-- bundles/
|   +-- <32-lowercase-hex-generation>/
|       +-- metadata.json
|       +-- orientation.npy
|       +-- entropy.npy
|       +-- frequency.npy
|       +-- coherence.npy
|       +-- uncertainty.npy
|       +-- optional SVG and PNG figures
+-- current.json
```

`current.json` contains only:

```json
{"schema_version":"structfield-estimate-current-v1","generation":"..."}
```

All generation files are written and `fsync`ed in a sibling staging directory.
The complete directory is renamed into `bundles/`; only then is the pointer
published. The first writer uses a no-clobber hard-link operation, so concurrent
non-overwrite writers cannot both claim the current pointer. `--overwrite`
atomically replaces the pointer with a newly completed generation and never
changes an older bundle.

If publication fails before the pointer switch, the existing pointer remains
valid and a complete orphan generation may remain. Known legacy flat files are
removed only after a successful pointer switch; unknown user files are never
removed. `resolve_current_bundle()` validates the pointer and returns the
selected immutable directory.

This contract is limited to local POSIX filesystems with atomic rename, hard
links, and directory `fsync`. It does not claim equivalent semantics for
network or object storage.
