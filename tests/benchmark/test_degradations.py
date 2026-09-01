"""Ground-truth synchronization tests for benchmark degradations."""

from __future__ import annotations

import math
from dataclasses import fields, replace

import pytest
import torch

from structfield.benchmarks.degradations import DegradationConfig, apply_degradation
from structfield.benchmarks.generator import StructFieldBenchConfig, generate_sample
from structfield.benchmarks.schema import BenchmarkMetadata, BenchmarkSample


def _sample(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BenchmarkSample:
    height, width, bins, components = 17, 19, 18, 3
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    image = (0.5 + 0.4 * torch.cos(2.0 * math.pi * (xx + yy) / 11.0)).unsqueeze(0).to(dtype=dtype)
    orientation = torch.zeros(bins, height, width, device=device, dtype=dtype)
    orientation[3] = 1.0  # 30 degrees for 18 uniform bins
    count = torch.ones(1, height, width, device=device, dtype=torch.int64)
    frequency = torch.zeros(components, height, width, device=device, dtype=dtype)
    frequency[0] = 0.1
    valid = torch.ones(1, height, width, device=device, dtype=torch.bool)
    junction = torch.zeros_like(valid)
    return BenchmarkSample(
        image=image,
        orientation_distribution_gt=orientation,
        orientation_count=count,
        component_frequency_gt=frequency,
        valid_mask=valid,
        junction_mask=junction,
        metadata=BenchmarkMetadata(
            sample_id="degradation-fixture",
            seed=1,
            split="train",
            family="straight",
            difficulty="medium",
            num_orientations=bins,
            orientation_bin_centers_rad=tuple(index * math.pi / bins for index in range(bins)),
            max_components=components,
            parameters={"fixture": True},
        ),
    )


IMAGE_ONLY = (
    "none",
    "gaussian",
    "poisson",
    "speckle",
    "salt_and_pepper",
    "gaussian_blur",
    "motion_blur",
    "contrast",
    "brightness",
    "jpeg",
)
ALL_DEGRADATIONS = (*IMAGE_ONLY, "rotation", "scaling", "occlusion")


def _tensor_items(sample: BenchmarkSample) -> dict[str, torch.Tensor]:
    return {
        item.name: value for item in fields(sample) if isinstance((value := getattr(sample, item.name)), torch.Tensor)
    }


@pytest.mark.parametrize("name", ALL_DEGRADATIONS)
def test_every_degradation_returns_valid_independently_stored_sample(name: str) -> None:
    sample = _sample()
    snapshot = {key: value.clone() for key, value in _tensor_items(sample).items()}
    output = apply_degradation(sample, DegradationConfig(name=name), seed=17)  # type: ignore[arg-type]

    assert output.validate() is output
    assert output.metadata.degradation == name
    record = output.metadata.parameters["degradation"]
    assert record["name"] == name  # type: ignore[index]
    assert record["seed"] == 17  # type: ignore[index]
    for key, value in _tensor_items(sample).items():
        assert torch.equal(value, snapshot[key]), key
        assert getattr(output, key).data_ptr() != value.data_ptr(), key


@pytest.mark.parametrize("name", IMAGE_ONLY)
def test_image_only_degradations_preserve_gt_values(name: str) -> None:
    sample = _sample()
    output = apply_degradation(sample, DegradationConfig(name=name, difficulty="hard"), seed=23)  # type: ignore[arg-type]
    for key in (
        "orientation_distribution_gt",
        "orientation_count",
        "component_frequency_gt",
        "valid_mask",
        "junction_mask",
    ):
        assert torch.equal(getattr(output, key), getattr(sample, key)), key


@pytest.mark.parametrize("name", ["gaussian", "poisson", "speckle", "salt_and_pepper", "occlusion"])
def test_stochastic_degradation_is_seed_deterministic_and_sensitive(name: str) -> None:
    sample = _sample()
    config = DegradationConfig(name=name, difficulty="hard")  # type: ignore[arg-type]
    first = apply_degradation(sample, config, seed=101)
    repeated = apply_degradation(sample, config, seed=101)
    different = apply_degradation(sample, config, seed=202)
    for key, value in _tensor_items(first).items():
        assert torch.equal(value, getattr(repeated, key)), key
    assert first.metadata.sample_id == repeated.metadata.sample_id
    assert first.metadata.sample_id != different.metadata.sample_id
    changed_key = "valid_mask" if name == "occlusion" else "image"
    assert not torch.equal(getattr(first, changed_key), getattr(different, changed_key))


def test_local_randomness_does_not_consume_global_cpu_rng() -> None:
    sample = _sample()
    torch.manual_seed(991)
    state = torch.random.get_rng_state().clone()
    for name in ("gaussian", "poisson", "speckle", "salt_and_pepper", "occlusion"):
        apply_degradation(sample, DegradationConfig(name=name), seed=5)  # type: ignore[arg-type]
    assert torch.equal(torch.random.get_rng_state(), state)


def test_noise_blur_and_jpeg_actually_change_nonconstant_image() -> None:
    sample = _sample()
    for name in (
        "gaussian",
        "poisson",
        "speckle",
        "salt_and_pepper",
        "gaussian_blur",
        "motion_blur",
        "contrast",
        "brightness",
        "jpeg",
    ):
        output = apply_degradation(
            sample,
            DegradationConfig(name=name, difficulty="hard"),  # type: ignore[arg-type]
            seed=37,
        )
        assert not torch.equal(output.image, sample.image), name
        assert bool(((0.0 <= output.image) & (output.image <= 1.0)).all())


def test_positive_motion_blur_angle_is_counter_clockwise_on_displayed_image() -> None:
    sample = _sample()
    impulse = torch.zeros_like(sample.image)
    center = (sample.spatial_shape[0] // 2, sample.spatial_shape[1] // 2)
    impulse[0, *center] = 1.0
    sample = replace(sample, image=impulse)
    output = apply_degradation(
        sample,
        DegradationConfig(
            name="motion_blur",
            parameters={"kernel_size": 5, "angle_rad": math.pi / 4.0},
        ),
        seed=1,
    )
    row, column = center
    assert output.image[0, row - 1, column + 1] > 0.0  # displayed '/' direction
    assert output.image[0, row + 1, column - 1] > 0.0
    assert output.image[0, row - 1, column - 1] == 0.0
    assert output.image[0, row + 1, column + 1] == 0.0


def test_rotation_synchronizes_spatial_gt_and_axial_orientation() -> None:
    sample = _sample()
    output = apply_degradation(
        sample,
        DegradationConfig(name="rotation", parameters={"angle_rad": math.radians(20.0)}),
        seed=1,
    )
    center = (sample.spatial_shape[0] // 2, sample.spatial_shape[1] // 2)
    assert output.valid_mask[0, *center]
    assert int(output.orientation_distribution_gt[:, center[0], center[1]].argmax()) == 5
    assert float(output.component_frequency_gt[0, *center]) == pytest.approx(0.1, abs=1e-7)
    assert not output.valid_mask[0, 0, 0]
    assert output.orientation_count[0, 0, 0] == 0
    assert torch.allclose(
        output.orientation_distribution_gt[:, 0, 0],
        torch.full_like(output.orientation_distribution_gt[:, 0, 0], 1.0 / 18.0),
    )
    assert torch.equal(output.component_frequency_gt[:, 0, 0], torch.zeros(3))


def test_scaling_synchronizes_component_frequency() -> None:
    sample = _sample()
    output = apply_degradation(
        sample,
        DegradationConfig(name="scaling", parameters={"scale": 2.0}),
        seed=1,
    )
    center = (sample.spatial_shape[0] // 2, sample.spatial_shape[1] // 2)
    assert output.valid_mask[0, *center]
    assert float(output.component_frequency_gt[0, *center]) == pytest.approx(0.05, abs=1e-7)
    assert int(output.orientation_distribution_gt[:, center[0], center[1]].argmax()) == 3


@pytest.mark.parametrize("family", ["bifurcation", "junction"])
def test_arbitrary_rotation_keeps_junction_mask_synchronized_with_component_count(
    family: str,
) -> None:
    sample = generate_sample(
        family,
        index=7,
        difficulty="hard",
        config=StructFieldBenchConfig(height=64, width=64),
    )
    output = apply_degradation(
        sample,
        DegradationConfig(name="rotation", parameters={"angle_rad": math.radians(17.3)}),
        seed=9,
    )
    assert torch.equal(output.junction_mask, output.orientation_count >= 2)
    assert bool(output.junction_mask.any())


def test_occlusion_neutralizes_all_gt_inside_recorded_rectangle() -> None:
    sample = _sample()
    output = apply_degradation(
        sample,
        DegradationConfig(name="occlusion", difficulty="medium"),
        seed=44,
    )
    record = output.metadata.parameters["degradation"]
    top = int(record["top"])  # type: ignore[index]
    left = int(record["left"])  # type: ignore[index]
    height = int(record["height"])  # type: ignore[index]
    width = int(record["width"])  # type: ignore[index]
    region = (slice(top, top + height), slice(left, left + width))
    assert not output.valid_mask[0][region].any()
    assert torch.equal(output.orientation_count[0][region], torch.zeros_like(output.orientation_count[0][region]))
    occluded_frequency = output.component_frequency_gt[:, region[0], region[1]]
    assert torch.equal(occluded_frequency, torch.zeros_like(occluded_frequency))
    assert torch.allclose(
        output.orientation_distribution_gt[:, region[0], region[1]],
        torch.full_like(output.orientation_distribution_gt[:, region[0], region[1]], 1.0 / 18.0),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_noise_and_geometry_preserve_dtype(dtype: torch.dtype) -> None:
    sample = _sample(dtype=dtype)
    for config in (
        DegradationConfig(name="gaussian"),
        DegradationConfig(name="rotation", parameters={"angle_rad": 0.2}),
    ):
        output = apply_degradation(sample, config, seed=8)
        assert output.dtype == dtype
        assert output.orientation_distribution_gt.dtype == dtype
        assert output.component_frequency_gt.dtype == dtype
        assert output.validate() is output


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"name": "unknown"}, "one of"),
        ({"name": "gaussian", "difficulty": "extreme"}, "one of"),
        ({"name": "gaussian", "parameters": {"sigma": True}}, "not bool"),
        ({"name": "gaussian", "parameters": {"sigma": float("nan")}}, "finite"),
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, object], error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        DegradationConfig(**kwargs)  # type: ignore[arg-type]


def test_invalid_runtime_parameters_fail_fast() -> None:
    sample = _sample()
    cases = (
        DegradationConfig(name="gaussian", parameters={"sigma": -1.0}),
        DegradationConfig(name="poisson", parameters={"peak": 0.0}),
        DegradationConfig(name="motion_blur", parameters={"kernel_size": 4}),
        DegradationConfig(name="jpeg", parameters={"quality": 101}),
        DegradationConfig(name="scaling", parameters={"scale": 0.0}),
        DegradationConfig(name="occlusion", parameters={"height_fraction": 1.1}),
        DegradationConfig(name="gaussian", parameters={"sigam": 0.1}),
    )
    for config in cases:
        with pytest.raises((TypeError, ValueError)):
            apply_degradation(sample, config, seed=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_geometric_degradations_preserve_schema_device_and_rng() -> None:
    sample = _sample(device="cuda")
    state = torch.cuda.get_rng_state().clone()
    for config in (
        DegradationConfig(name="rotation", parameters={"angle_rad": 0.23}),
        DegradationConfig(name="scaling", parameters={"scale": 1.4}),
    ):
        output = apply_degradation(sample, config, seed=12)
        assert output.device.type == "cuda"
        assert output.validate() is output
    assert torch.equal(torch.cuda.get_rng_state(), state)
