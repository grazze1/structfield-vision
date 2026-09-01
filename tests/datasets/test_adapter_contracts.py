"""Contract tests for the local-only dataset adapter boundary."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import structfield.datasets as datasets
from structfield.datasets import (
    RESERVED_DATASET_IDENTIFIERS,
    DatasetAdapter,
    DatasetRecord,
    GenericFolderDatasetAdapter,
    validate_dataset_adapter,
    validate_dataset_record,
    validate_dataset_records,
    validate_dataset_root,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _record(root: Path, sample_id: str = "sample") -> DatasetRecord:
    return DatasetRecord(
        dataset_id="local-fixture",
        sample_id=sample_id,
        image_path=_touch(root / "images" / f"{sample_id}.png"),
        target_path=_touch(root / "targets" / f"{sample_id}.png"),
        valid_mask_path=_touch(root / "masks" / f"{sample_id}.png"),
        split="test",
    )


def test_public_api_and_reserved_identifiers_are_exact() -> None:
    required_exports = {
        "COMMON_IMAGE_EXTENSIONS",
        "RESERVED_DATASET_IDENTIFIERS",
        "DatasetAdapter",
        "DatasetRecord",
        "GenericFolderDatasetAdapter",
        "validate_dataset_adapter",
        "validate_dataset_record",
        "validate_dataset_records",
        "validate_dataset_root",
    }

    assert datasets.__name__ == "structfield.datasets"
    assert required_exports <= set(datasets.__all__)
    assert RESERVED_DATASET_IDENTIFIERS == (
        "DRIVE",
        "CHASE_DB1",
        "HRF",
        "Crack500",
        "DeepGlobe",
    )


def test_dataset_adapter_is_a_runtime_protocol(tmp_path: Path) -> None:
    class MinimalAdapter:
        dataset_id = "minimal"
        root = tmp_path

        def records(self) -> tuple[DatasetRecord, ...]:
            return ()

    assert isinstance(MinimalAdapter(), DatasetAdapter)
    assert isinstance(
        GenericFolderDatasetAdapter(root=tmp_path, dataset_id="generic"),
        DatasetAdapter,
    )
    assert not isinstance(object(), DatasetAdapter)


def test_dataset_record_is_keyword_only_and_immutable(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        DatasetRecord("fixture", "sample", tmp_path / "image.png")  # type: ignore[misc]

    record = DatasetRecord(
        dataset_id="fixture",
        sample_id="sample",
        image_path=tmp_path / "not-created.png",
    )
    with pytest.raises((AttributeError, TypeError)):
        record.sample_id = "changed"  # type: ignore[misc]


def test_record_construction_is_structural_and_does_not_require_files(tmp_path: Path) -> None:
    record = DatasetRecord(
        dataset_id="fixture",
        sample_id="not-yet-materialized",
        image_path=tmp_path / "absent.png",
        target_path=tmp_path / "absent-target.png",
    )

    assert record.image_path == tmp_path / "absent.png"
    with pytest.raises(FileNotFoundError):
        validate_dataset_record(record)


def test_metadata_is_deep_frozen_and_detached_from_input(tmp_path: Path) -> None:
    source: dict[str, Any] = {
        "nested": {"sequence": [1, {"accepted": True}]},
        "score": 0.25,
    }
    record = DatasetRecord(
        dataset_id="fixture",
        sample_id="sample",
        image_path=tmp_path / "image.png",
        metadata=source,
    )

    nested = source["nested"]
    assert isinstance(nested, dict)
    sequence = nested["sequence"]
    assert isinstance(sequence, list)
    accepted = sequence[1]
    assert isinstance(accepted, dict)
    accepted["accepted"] = False
    sequence.append(2)
    assert record.metadata["nested"]["sequence"] == (1, {"accepted": True})

    with pytest.raises((AttributeError, TypeError)):
        record.metadata["new"] = "value"  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        record.metadata["nested"]["sequence"].append(3)  # type: ignore[union-attr]
    with pytest.raises((AttributeError, TypeError)):
        record.metadata["nested"]["sequence"][1]["accepted"] = False  # type: ignore[index]

    detached = record.to_dict()
    detached["metadata"]["nested"]["sequence"][1]["accepted"] = False
    assert record.metadata["nested"]["sequence"][1]["accepted"] is True


@pytest.mark.parametrize(
    "metadata",
    (
        {1: "non-string-key"},
        {"bad": math.nan},
        {"bad": math.inf},
        {"bad": {"sets-are-not-json"}},
        {"bad": object()},
    ),
)
def test_record_rejects_metadata_that_cannot_be_safely_frozen(
    tmp_path: Path,
    metadata: dict[object, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        DatasetRecord(
            dataset_id="fixture",
            sample_id="sample",
            image_path=tmp_path / "image.png",
            metadata=metadata,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dataset_id", ""),
        ("dataset_id", " padded "),
        ("sample_id", ""),
        ("sample_id", 3),
        ("split", ""),
    ),
)
def test_record_rejects_invalid_text_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "dataset_id": "fixture",
        "sample_id": "sample",
        "image_path": tmp_path / "image.png",
        "split": None,
    }
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        DatasetRecord(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("path", ("https://example.invalid/image.png", "file:///tmp/image.png"))
def test_record_rejects_url_paths(path: str) -> None:
    with pytest.raises(ValueError):
        DatasetRecord(dataset_id="fixture", sample_id="sample", image_path=path)  # type: ignore[arg-type]


def test_validate_dataset_root_is_fail_fast(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    plain_file = _touch(tmp_path / "plain-file")

    assert validate_dataset_root(root) == root.resolve()
    with pytest.raises(FileNotFoundError):
        validate_dataset_root(tmp_path / "missing")
    with pytest.raises((NotADirectoryError, ValueError)):
        validate_dataset_root(plain_file)
    with pytest.raises((TypeError, ValueError)):
        validate_dataset_root("https://example.invalid/dataset")


def test_validate_record_checks_files_requirements_and_returns_identity(tmp_path: Path) -> None:
    root = tmp_path / "root"
    record = _record(root)

    assert (
        validate_dataset_record(
            record,
            root=root,
            require_target=True,
            require_valid_mask=True,
        )
        is record
    )

    image_only = DatasetRecord(
        dataset_id="fixture",
        sample_id="image-only",
        image_path=_touch(root / "images" / "image-only.png"),
    )
    with pytest.raises(ValueError):
        validate_dataset_record(image_only, root=root, require_target=True)
    with pytest.raises(ValueError):
        validate_dataset_record(image_only, root=root, require_valid_mask=True)

    missing_optional_file = DatasetRecord(
        dataset_id="fixture",
        sample_id="missing-optional",
        image_path=image_only.image_path,
        target_path=root / "targets" / "missing.png",
    )
    with pytest.raises(FileNotFoundError):
        validate_dataset_record(missing_optional_file, root=root)


@pytest.mark.parametrize("flag", (None, 0, 1, "yes"))
def test_validation_flags_must_be_actual_bools(tmp_path: Path, flag: object) -> None:
    record = _record(tmp_path / "root")
    with pytest.raises(TypeError):
        validate_dataset_record(record, require_target=flag)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        validate_dataset_records([record], allow_empty=flag)  # type: ignore[arg-type]


def test_record_validation_rejects_outside_and_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = _touch(tmp_path / "outside.png")

    outside_record = DatasetRecord(
        dataset_id="fixture",
        sample_id="outside",
        image_path=outside,
    )
    with pytest.raises(ValueError):
        validate_dataset_record(outside_record, root=root)

    link = root / "linked.png"
    try:
        link.symlink_to(outside)
    except OSError as error:  # pragma: no cover - platform capability guard
        pytest.skip(f"symlinks unavailable: {error}")
    linked_record = DatasetRecord(
        dataset_id="fixture",
        sample_id="linked",
        image_path=link,
    )
    with pytest.raises(ValueError):
        validate_dataset_record(linked_record, root=root)


def test_record_validation_rejects_aliases_to_the_same_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    image = _touch(root / "images" / "sample.png")
    alias = root / "targets" / "sample.png"
    alias.parent.mkdir()
    try:
        alias.symlink_to(image)
    except OSError as error:  # pragma: no cover - platform capability guard
        pytest.skip(f"symlinks unavailable: {error}")

    record = DatasetRecord(
        dataset_id="fixture",
        sample_id="sample",
        image_path=image,
        target_path=alias,
    )
    with pytest.raises(ValueError):
        validate_dataset_record(record, root=root)


def test_validate_records_is_finite_deterministic_and_rejects_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "root"
    first = _record(root, "first")
    second = _record(root, "second")

    result = validate_dataset_records(
        (record for record in (first, second)),
        root=root,
        require_target=True,
        require_valid_mask=True,
    )
    assert result == (first, second)
    assert validate_dataset_records((), root=root, allow_empty=True) == ()
    with pytest.raises(ValueError):
        validate_dataset_records((), root=root)

    duplicate_identity = DatasetRecord(
        dataset_id=first.dataset_id,
        sample_id=first.sample_id,
        image_path=_touch(root / "images" / "identity-copy.png"),
    )
    with pytest.raises(ValueError):
        validate_dataset_records((first, duplicate_identity), root=root)

    duplicate_image = DatasetRecord(
        dataset_id="other-dataset",
        sample_id="other-sample",
        image_path=first.image_path,
    )
    with pytest.raises(ValueError):
        validate_dataset_records((first, duplicate_image), root=root)


def test_validate_records_rejects_cross_record_role_aliases(tmp_path: Path) -> None:
    root = tmp_path / "root"
    first = _record(root, "first")
    second_image = _touch(root / "images" / "second.png")

    image_reused_as_target = DatasetRecord(
        dataset_id="local-fixture",
        sample_id="second",
        image_path=second_image,
        target_path=first.image_path,
    )
    with pytest.raises(ValueError):
        validate_dataset_records((first, image_reused_as_target), root=root)

    target_reused_as_mask = DatasetRecord(
        dataset_id="local-fixture",
        sample_id="second",
        image_path=second_image,
        valid_mask_path=first.target_path,
    )
    with pytest.raises(ValueError):
        validate_dataset_records((first, target_reused_as_mask), root=root)


def test_validate_records_allows_an_intentionally_shared_valid_mask(tmp_path: Path) -> None:
    root = tmp_path / "root"
    shared_mask = _touch(root / "masks" / "field-of-view.png")
    records = tuple(
        DatasetRecord(
            dataset_id="local-fixture",
            sample_id=sample_id,
            image_path=_touch(root / "images" / f"{sample_id}.png"),
            valid_mask_path=shared_mask,
        )
        for sample_id in ("first", "second")
    )

    assert validate_dataset_records(records, root=root, require_valid_mask=True) == records


def test_validate_adapter_enforces_protocol_and_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    record = _record(root)

    class LocalAdapter:
        dataset_id = "local-fixture"

        def __init__(self, dataset_root: Path) -> None:
            self.root = dataset_root

        def records(self) -> tuple[DatasetRecord, ...]:
            return (record,)

    assert validate_dataset_adapter(LocalAdapter(root), require_target=True) == (record,)
    with pytest.raises(TypeError):
        validate_dataset_adapter(object())  # type: ignore[arg-type]


def test_validate_adapter_checks_root_and_identity_before_accepting_records(tmp_path: Path) -> None:
    class InvalidIdentifierAdapter:
        dataset_id = " padded "
        root = tmp_path

        def records(self) -> tuple[DatasetRecord, ...]:
            raise AssertionError("records must not run before identifier validation")

    with pytest.raises(TypeError):
        validate_dataset_adapter(InvalidIdentifierAdapter())

    class MissingRootAdapter:
        dataset_id = "fixture"
        root = tmp_path / "missing"

        def records(self) -> tuple[DatasetRecord, ...]:
            raise AssertionError("records must not run before root validation")

    with pytest.raises(FileNotFoundError):
        validate_dataset_adapter(MissingRootAdapter())

    root = tmp_path / "root"
    root.mkdir()
    record = DatasetRecord(
        dataset_id="different",
        sample_id="sample",
        image_path=_touch(root / "images" / "sample.png"),
    )

    class MismatchedAdapter:
        dataset_id = "declared"

        def __init__(self, dataset_root: Path) -> None:
            self.root = dataset_root

        def records(self) -> tuple[DatasetRecord, ...]:
            return (record,)

    with pytest.raises(ValueError):
        validate_dataset_adapter(MismatchedAdapter(root))

    class InvalidRecordAdapter:
        dataset_id = "declared"

        def __init__(self, dataset_root: Path) -> None:
            self.root = dataset_root

        def records(self) -> tuple[object, ...]:
            return (object(),)

    with pytest.raises(TypeError):
        validate_dataset_adapter(InvalidRecordAdapter(root))  # type: ignore[arg-type]
