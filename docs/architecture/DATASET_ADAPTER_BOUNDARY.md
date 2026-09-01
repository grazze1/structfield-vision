# Dataset Adapter Boundary

## Scope

Stage 11 establishes a small, read-only boundary between caller-owned local
files and future StructField-Vision data pipelines. It implements four things:

- a runtime-checkable `DatasetAdapter` protocol;
- an immutable, path-only `DatasetRecord`;
- a deterministic `GenericFolderDatasetAdapter`;
- fail-fast root, record, collection, and adapter validation functions.

This stage intentionally does not implement a PyTorch `Dataset`, image decoder,
transform pipeline, download manager, cache, archive extractor, or any of the
five planned dataset-specific runtimes.

## Public contract

`DatasetRecord` contains a dataset/sample identity, a required image path,
optional target and valid-region-mask paths, an optional split, and deeply
frozen JSON-compatible metadata. Construction performs no disk access. On-disk
checks are explicit through `validate_dataset_record` or
`validate_dataset_records`.

`DatasetAdapter` requires only three members:

```text
dataset_id: str
root: pathlib.Path
records() -> tuple[DatasetRecord, ...]
```

`GenericFolderDatasetAdapter` accepts an explicit root and relative image,
target, and valid-mask directories. It recognizes path suffixes only. The
default suffix vocabulary is PNG, JPEG, BMP, TIFF, GIF, and NPY; supporting a
suffix here does not imply decoding support.

## Pairing and validation semantics

Each folder is indexed by a file's relative path with its final extension
removed. The index is case-insensitive only for extension filtering; relative
stems retain their exact spelling. Configured target and valid-mask folders must
have exactly the same key set as the image folder. Results are sorted by key.

Validation rejects:

- absent or non-directory roots;
- absent or non-file record paths;
- empty image discovery;
- duplicate record identities, image paths, target paths, or relative stems,
  plus cross-record reuse of one file in image/target/mask roles;
- missing and orphan target or valid-mask paths;
- absolute/traversing configured subdirectories;
- files or symlinks resolving outside the declared root;
- URL-like values where local paths are required;
- non-finite or non-JSON metadata.

Shared valid-region masks remain legal at the record-validation layer because
some future layouts may use a common field-of-view template. The generic folder
adapter itself uses one-to-one relative-stem pairing.

Adapter-level validation checks the declared dataset identifier and resolves
the root before invoking discovery. Every returned record must use that same
identifier; malformed record collections fail before being accepted.

## Planned, not supported

The public tuple `RESERVED_DATASET_IDENTIFIERS` freezes these future names:

```text
DRIVE
CHASE_DB1
HRF
Crack500
DeepGlobe
```

No dedicated adapter, URL, checksum, downloader, split definition, class map,
or performance claim is attached to those names in V0.2. Local-layout success
does not validate dataset identity or task semantics.

Qilu and SNP-Net remain historical-only Legacy Evidence. No active Python
symbol, registry entry, default path, loader, or test fixture is provided for
either dataset.

## Side-effect and data boundary

The adapter implementation uses standard-library protocol, dataclass, mapping,
and filesystem path primitives. Discovery performs no network access and never
writes, decodes, copies, moves, extracts, caches, or vendors data. Tests create
only tiny empty files under pytest-managed temporary directories.

No external image, mask, array, archive, clinical table, checkpoint, or
pretrained weight belongs in this repository. Users are responsible for
obtaining data and checking the current source terms. Layout validation is not
license, privacy, ethics, medical-use, or redistribution clearance.

## Package-name boundary

The governing V0.2 specification fixes the tracked physical source boundary as
`datasets/adapters/`. Stage 13 maps that directory into the distribution as
`structfield.datasets` and deliberately omits a top-level `datasets` package.
This prevents StructField-Vision from owning or shadowing the import name used
by unrelated distributions such as Hugging Face Datasets. A source checkout is
therefore expected to use the verified editable installation before importing
the adapter API.

## Provenance

The protocol, record schema, generic discovery logic, validation functions,
tests, and documentation are independently written for StructField-Vision
V0.2. No third-party adapter implementation or dataset asset was copied. The
repository-wide source license remains pending the final provenance review in
`LICENSE`; external data always remains under its own terms as described in `NOTICE.md`.
