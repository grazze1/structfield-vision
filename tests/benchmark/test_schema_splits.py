from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from structfield.benchmarks.schema import (
    BENCHMARK_VERSION,
    CANONICAL_DEGRADATIONS,
    CANONICAL_FAMILIES,
    BenchmarkMetadata,
    BenchmarkSample,
)
from structfield.benchmarks.splits import (
    BASE_SEED_STRIDE,
    MAX_BASE_SEED,
    MAX_SAMPLES_PER_FAMILY,
    SPLIT_SEED_DOMAINS,
    derive_sample_seed,
    get_seed_domain,
    make_sample_id,
)


def _metadata(**updates: object) -> BenchmarkMetadata:
    values: dict[str, object] = {
        "sample_id": "tiny-0",
        "seed": 7,
        "split": "train",
        "family": "straight",
        "difficulty": "easy",
        "degradation": "none",
        "parameters": {"contrast": 0.7, "nested": {"levels": [1, 2, 3]}},
        "num_orientations": 4,
        "orientation_bin_centers_rad": (0.0, torch.pi / 4, torch.pi / 2, 3 * torch.pi / 4),
        "max_components": 2,
    }
    values.update(updates)
    return BenchmarkMetadata(**values)  # type: ignore[arg-type]


def _sample(**updates: object) -> BenchmarkSample:
    height, width = 2, 3
    image = torch.linspace(0.0, 1.0, height * width, dtype=torch.float32).view(1, height, width)
    count = torch.tensor([[[1, 2, 0], [1, 0, 1]]], dtype=torch.int64)
    valid = count > 0
    junction = torch.zeros_like(valid)
    junction[0, 0, 1] = True

    orientation = torch.full((4, height, width), 0.25, dtype=torch.float32)
    one_component = count[0] == 1
    orientation[:, one_component] = 0.0
    orientation[0, one_component] = 1.0
    two_components = count[0] == 2
    orientation[:, two_components] = 0.0
    orientation[0, two_components] = 0.5
    orientation[2, two_components] = 0.5

    frequency = torch.zeros((2, height, width), dtype=torch.float32)
    frequency[0, valid[0]] = 0.125
    frequency[1, two_components] = 0.25
    values: dict[str, object] = {
        "image": image,
        "orientation_distribution_gt": orientation,
        "orientation_count": count,
        "component_frequency_gt": frequency,
        "valid_mask": valid,
        "junction_mask": junction,
        "metadata": _metadata(),
    }
    values.update(updates)
    return BenchmarkSample(**values)  # type: ignore[arg-type]


def test_canonical_vocabulary_is_frozen_and_complete() -> None:
    assert BENCHMARK_VERSION == "StructField-Bench-v0.1"
    assert CANONICAL_FAMILIES == (
        "straight",
        "parallel",
        "curved",
        "sinusoidal",
        "radial",
        "concentric",
        "crossing",
        "dual_crossing",
        "bifurcation",
        "junction",
        "varying_frequency",
        "varying_thickness",
        "domain_boundary",
        "occlusion",
    )
    assert CANONICAL_DEGRADATIONS == (
        "none",
        "gaussian",
        "poisson",
        "speckle",
        "salt_and_pepper",
        "gaussian_blur",
        "motion_blur",
        "contrast",
        "brightness",
        "rotation",
        "scaling",
        "jpeg",
        "occlusion",
    )


def test_tiny_sample_has_computable_cross_field_contract() -> None:
    sample = _sample()
    assert sample.validate() is sample
    assert sample.spatial_shape == (2, 3)
    assert sample.dtype == torch.float32
    assert sample.device.type == "cpu"
    assert torch.equal(sample.valid_mask, sample.orientation_count > 0)
    assert torch.equal(sample.junction_mask & ~sample.valid_mask, torch.zeros_like(sample.valid_mask))
    assert torch.allclose(sample.orientation_distribution_gt.sum(dim=0), torch.ones(2, 3))
    assert torch.equal(sample.component_frequency_gt[:, ~sample.valid_mask[0]], torch.zeros(2, 2))


def test_metadata_and_nested_parameters_are_immutable() -> None:
    source_parameters = {"nested": {"levels": [1, 2, 3]}}
    metadata = _metadata(parameters=source_parameters)
    source_parameters["nested"]["levels"].append(4)  # type: ignore[index,union-attr]

    with pytest.raises(FrozenInstanceError):
        metadata.seed = 99  # type: ignore[misc]
    with pytest.raises(TypeError):
        metadata.parameters["new"] = 1  # type: ignore[index]
    nested = metadata.parameters["nested"]
    assert nested["levels"] == (1, 2, 3)  # type: ignore[index]
    with pytest.raises(TypeError):
        nested["new"] = 1  # type: ignore[index]
    assert metadata.to_dict()["parameters"] == {"nested": {"levels": [1, 2, 3]}}


def test_metadata_to_dict_is_json_serializable_and_detached() -> None:
    metadata = _metadata()
    payload = metadata.to_dict()
    encoded = json.dumps(payload, sort_keys=True)
    assert "StructField-Bench-v0.1" in encoded
    payload["parameters"]["nested"]["levels"].append(99)
    assert metadata.to_dict()["parameters"]["nested"]["levels"] == [1, 2, 3]


@pytest.mark.parametrize(
    ("updates", "error", "message"),
    [
        ({"benchmark_version": "v0"}, ValueError, "benchmark_version"),
        ({"generator_version": ""}, TypeError, "generator_version"),
        ({"sample_id": ""}, TypeError, "sample_id"),
        ({"seed": True}, TypeError, "seed"),
        ({"seed": -1}, ValueError, "seed"),
        ({"split": "dev"}, ValueError, "split"),
        ({"family": "dual crossing"}, ValueError, "family"),
        ({"difficulty": ""}, TypeError, "difficulty"),
        ({"degradation": "blur"}, ValueError, "degradation"),
        ({"num_orientations": 1}, ValueError, "num_orientations"),
        ({"orientation_bin_centers_rad": (0.0, 0.1)}, ValueError, "length"),
        (
            {"orientation_bin_centers_rad": (0.0, 0.2, 0.1, 0.3)},
            ValueError,
            "strictly increasing",
        ),
        ({"orientation_bin_centers_rad": (0.0, 0.1, 0.2, torch.pi)}, ValueError, r"\[0, pi\)"),
        ({"max_components": 0}, ValueError, "max_components"),
        ({"parameters": []}, TypeError, "mapping"),
        ({"parameters": {1: "bad"}}, TypeError, "keys"),
        ({"parameters": {"bad": float("nan")}}, ValueError, "non-finite"),
        ({"parameters": {"bad": torch.tensor(1)}}, TypeError, "JSON-compatible"),
    ],
)
def test_metadata_rejects_invalid_contracts(
    updates: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _metadata(**updates)


def test_sample_dataclass_is_structurally_immutable() -> None:
    sample = _sample()
    with pytest.raises(FrozenInstanceError):
        sample.image = torch.zeros_like(sample.image)  # type: ignore[misc]


def test_sample_to_returns_non_aliasing_copy_and_preserves_discrete_dtypes() -> None:
    sample = _sample()
    converted = sample.to(dtype=torch.float64)
    assert converted is not sample
    assert converted.metadata is sample.metadata
    assert converted.dtype == torch.float64
    assert converted.orientation_distribution_gt.dtype == torch.float64
    assert converted.component_frequency_gt.dtype == torch.float64
    assert converted.orientation_count.dtype == torch.int64
    assert converted.valid_mask.dtype == torch.bool
    assert converted.junction_mask.dtype == torch.bool
    for before, after in zip(
        (
            sample.image,
            sample.orientation_distribution_gt,
            sample.orientation_count,
            sample.component_frequency_gt,
            sample.valid_mask,
            sample.junction_mask,
        ),
        (
            converted.image,
            converted.orientation_distribution_gt,
            converted.orientation_count,
            converted.component_frequency_gt,
            converted.valid_mask,
            converted.junction_mask,
        ),
    ):
        assert before.data_ptr() != after.data_ptr()


def test_sample_rejects_unsupported_float8_schema_and_conversion() -> None:
    float8_dtype = getattr(torch, "float8_e4m3fn", None)
    if float8_dtype is None:
        pytest.skip("float8_e4m3fn is unavailable")
    sample = _sample()
    with pytest.raises(TypeError, match="floating-point dtype supported"):
        replace(sample, image=torch.empty_like(sample.image, dtype=float8_dtype))
    with pytest.raises(TypeError, match="supported floating-point"):
        sample.to(dtype=float8_dtype)


def test_sample_to_renormalizes_probability_mass_across_precision_changes() -> None:
    metadata = _metadata(
        num_orientations=18,
        orientation_bin_centers_rad=tuple(index * torch.pi / 18 for index in range(18)),
    )
    sample = BenchmarkSample(
        image=torch.zeros(1, 1, 1),
        orientation_distribution_gt=torch.full((18, 1, 1), 1.0 / 18),
        orientation_count=torch.zeros(1, 1, 1, dtype=torch.int64),
        component_frequency_gt=torch.zeros(2, 1, 1),
        valid_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        junction_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        metadata=metadata,
    )
    promoted = sample.to(dtype=torch.float16).to(dtype=torch.float64)
    assert torch.allclose(
        promoted.orientation_distribution_gt.sum(dim=0),
        torch.ones(1, 1, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_sample_rejects_invalid_shapes_types_ranges_and_semantics() -> None:
    sample = _sample()

    with pytest.raises(ValueError, match="image must have shape"):
        replace(sample, image=sample.image.unsqueeze(0))
    with pytest.raises(TypeError, match="image must use a floating"):
        replace(sample, image=torch.zeros_like(sample.image, dtype=torch.int64))
    with pytest.raises(ValueError, match="image must lie"):
        replace(sample, image=sample.image.clone().fill_(1.1))
    bad_nan = sample.image.clone()
    bad_nan[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        replace(sample, image=bad_nan)

    with pytest.raises(ValueError, match="orientation_distribution_gt must have shape"):
        replace(sample, orientation_distribution_gt=sample.orientation_distribution_gt[:-1])
    with pytest.raises(ValueError, match="share image dtype"):
        replace(sample, orientation_distribution_gt=sample.orientation_distribution_gt.double())
    negative_probability = sample.orientation_distribution_gt.clone()
    negative_probability[0, 0, 0] = -0.1
    negative_probability[1, 0, 0] += 0.1
    with pytest.raises(ValueError, match="negative probability"):
        replace(sample, orientation_distribution_gt=negative_probability)
    bad_sum = sample.orientation_distribution_gt.clone()
    bad_sum[:, 0, 0] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        replace(sample, orientation_distribution_gt=bad_sum)

    with pytest.raises(TypeError, match="orientation_count must use torch.int64"):
        replace(sample, orientation_count=sample.orientation_count.int())
    count_too_large = sample.orientation_count.clone()
    count_too_large[0, 0, 0] = 3
    with pytest.raises(ValueError, match="orientation_count must lie"):
        replace(sample, orientation_count=count_too_large)
    with pytest.raises(TypeError, match="must use torch.bool"):
        replace(sample, valid_mask=sample.valid_mask.float())

    inconsistent_valid = sample.valid_mask.clone()
    inconsistent_valid[0, 0, 0] = False
    with pytest.raises(ValueError, match="valid_mask must equal"):
        replace(sample, valid_mask=inconsistent_valid)
    invalid_junction = sample.junction_mask.clone()
    invalid_junction[0, 0, 2] = True
    with pytest.raises(ValueError, match="subset"):
        replace(sample, junction_mask=invalid_junction)

    nonuniform_invalid = sample.orientation_distribution_gt.clone()
    nonuniform_invalid[:, 0, 2] = 0.0
    nonuniform_invalid[0, 0, 2] = 1.0
    with pytest.raises(ValueError, match="uniform orientation"):
        replace(sample, orientation_distribution_gt=nonuniform_invalid)
    frequency_at_invalid = sample.component_frequency_gt.clone()
    frequency_at_invalid[0, 0, 2] = 0.1
    with pytest.raises(ValueError, match="invalid pixels must have zero"):
        replace(sample, component_frequency_gt=frequency_at_invalid)
    frequency_in_padding = sample.component_frequency_gt.clone()
    frequency_in_padding[1, 0, 0] = 0.1
    with pytest.raises(ValueError, match="beyond orientation_count"):
        replace(sample, component_frequency_gt=frequency_in_padding)


def test_split_domains_are_pairwise_disjoint_and_cover_derived_seeds() -> None:
    train = get_seed_domain("train")
    validation = get_seed_domain("validation")
    test = get_seed_domain("test")
    assert train.stop <= validation.start
    assert validation.stop <= test.start
    for split, domain in SPLIT_SEED_DOMAINS.items():
        assert derive_sample_seed(split, "straight", 0) in domain
        assert (
            derive_sample_seed(
                split,
                CANONICAL_FAMILIES[-1],
                MAX_SAMPLES_PER_FAMILY - 1,
                MAX_BASE_SEED,
            )
            in domain
        )


def test_seed_derivation_is_deterministic_injective_and_uses_stable_offsets() -> None:
    expected = 17 * BASE_SEED_STRIDE + 5
    assert derive_sample_seed("train", "straight", 5, 17) == expected
    assert derive_sample_seed("train", "straight", 5, 17) == expected

    family_seeds = [derive_sample_seed("train", family, 5, 17) for family in CANONICAL_FAMILIES]
    assert len(family_seeds) == len(set(family_seeds)) == 14
    assert all(right - left == MAX_SAMPLES_PER_FAMILY for left, right in zip(family_seeds, family_seeds[1:]))
    split_seeds = {derive_sample_seed(split, "straight", 5, 17) for split in SPLIT_SEED_DOMAINS}
    assert len(split_seeds) == 3


def test_sample_id_is_stable_unique_and_contains_explicit_identity() -> None:
    identifier = make_sample_id("validation", "dual_crossing", 12, 3, difficulty="easy")
    assert identifier == make_sample_id("validation", "dual_crossing", 12, 3, difficulty="easy")
    assert BENCHMARK_VERSION in identifier
    assert ":validation:dual_crossing:" in identifier
    assert identifier != make_sample_id("validation", "dual_crossing", 13, 3)
    assert identifier != make_sample_id("validation", "dual_crossing", 12, 3, difficulty="hard")
    with pytest.raises(TypeError, match="difficulty"):
        make_sample_id("validation", "dual_crossing", 12, 3, difficulty="")


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        (("dev", "straight", 0, 0), ValueError, "split"),
        (("train", "unknown", 0, 0), ValueError, "family"),
        (("train", "straight", True, 0), TypeError, "index"),
        (("train", "straight", -1, 0), ValueError, "index"),
        (("train", "straight", MAX_SAMPLES_PER_FAMILY, 0), ValueError, "index"),
        (("train", "straight", 0, True), TypeError, "base_seed"),
        (("train", "straight", 0, MAX_BASE_SEED + 1), ValueError, "base_seed"),
    ],
)
def test_split_helpers_fail_fast(
    args: tuple[object, object, object, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        derive_sample_seed(*args)  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sample_to_cuda_preserves_schema_and_round_trips() -> None:
    sample = _sample()
    cuda_sample = sample.to(device="cuda", dtype=torch.float16)
    assert cuda_sample.device.type == "cuda"
    assert cuda_sample.dtype == torch.float16
    assert cuda_sample.orientation_count.device.type == "cuda"
    assert cuda_sample.valid_mask.device.type == "cuda"
    round_trip = cuda_sample.to(device="cpu", dtype=torch.float32)
    assert round_trip.device.type == "cpu"
    assert torch.allclose(round_trip.image, sample.image, atol=5e-4, rtol=5e-4)
    assert torch.equal(round_trip.orientation_count, sample.orientation_count)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sample_rejects_cross_device_tensor_mixture() -> None:
    sample = _sample()
    with pytest.raises(ValueError, match="valid_mask must share image device"):
        replace(sample, valid_mask=sample.valid_mask.cuda())
