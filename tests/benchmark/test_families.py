"""Analytic and determinism contracts for the fourteen v0.1 families."""

from __future__ import annotations

import math

import pytest
import torch

from structfield.benchmarks.families import FAMILY_NAMES, AnalyticComponent, FamilyScene
from structfield.benchmarks.generator import (
    StructFieldBenchConfig,
    StructFieldBenchGenerator,
    _assemble_ground_truth,
    generate_sample,
)
from structfield.benchmarks.schema import CANONICAL_FAMILIES, BenchmarkSample


def _generator(*, base_seed: int = 0) -> StructFieldBenchGenerator:
    return StructFieldBenchGenerator(
        StructFieldBenchConfig(
            height=64,
            width=72,
            num_orientations=18,
            max_components=3,
            base_frequency_cpp=0.08,
            base_seed=base_seed,
        ),
    )


def _expected_splat(angles: list[float], num_orientations: int) -> torch.Tensor:
    expected = torch.zeros(num_orientations)
    for angle in angles:
        position = (angle % math.pi) * num_orientations / math.pi
        lower_unwrapped = math.floor(position)
        fraction = position - lower_unwrapped
        expected[lower_unwrapped % num_orientations] += (1.0 - fraction) / len(angles)
        expected[(lower_unwrapped + 1) % num_orientations] += fraction / len(angles)
    return expected


def _assert_samples_equal(first: BenchmarkSample, second: BenchmarkSample) -> None:
    for name in (
        "image",
        "orientation_distribution_gt",
        "orientation_count",
        "component_frequency_gt",
        "valid_mask",
        "junction_mask",
    ):
        assert torch.equal(getattr(first, name), getattr(second, name)), name
    assert first.metadata.to_dict() == second.metadata.to_dict()


@pytest.mark.parametrize("family", CANONICAL_FAMILIES)
def test_all_canonical_families_generate_strict_schema(family: str) -> None:
    sample = _generator().generate_sample(family, split="validation", index=11, difficulty="medium")
    assert isinstance(sample, BenchmarkSample)
    assert sample.image.shape == (1, 64, 72)
    assert sample.orientation_distribution_gt.shape == (18, 64, 72)
    assert sample.orientation_count.shape == (1, 64, 72)
    assert sample.component_frequency_gt.shape == (3, 64, 72)
    assert sample.valid_mask.shape == sample.junction_mask.shape == (1, 64, 72)
    assert sample.image.dtype == sample.orientation_distribution_gt.dtype == torch.float32
    assert sample.component_frequency_gt.dtype == torch.float32
    assert sample.orientation_count.dtype == torch.int64
    assert sample.valid_mask.dtype == sample.junction_mask.dtype == torch.bool
    assert 0.0 <= float(sample.image.min()) <= float(sample.image.max()) <= 1.0
    assert torch.allclose(
        sample.orientation_distribution_gt.sum(dim=0),
        torch.ones(64, 72),
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.equal(sample.valid_mask, sample.orientation_count > 0)
    assert sample.metadata.family == family
    assert sample.metadata.difficulty == "medium"
    assert sample.metadata.degradation == "none"
    assert sample.metadata.parameters["difficulty_level"] == pytest.approx(0.55)
    assert sample.metadata.parameters["sample_seed"] == sample.metadata.seed
    assert sample.metadata.parameters["sample_index"] == 11
    assert sample.metadata.parameters["difficulty"] == "medium"


def test_renderer_vocabulary_is_exactly_the_frozen_fourteen_families() -> None:
    assert FAMILY_NAMES == CANONICAL_FAMILIES
    assert len(FAMILY_NAMES) == 14


def test_split_index_identity_is_repeatable_and_different_indices_change_sample() -> None:
    generator = _generator()
    first = generator.generate_sample("sinusoidal", split="test", index=37, difficulty="hard")
    repeated = generator.generate_sample("sinusoidal", split="test", index=37, difficulty="hard")
    different = generator.generate_sample("sinusoidal", split="test", index=38, difficulty="hard")
    _assert_samples_equal(first, repeated)
    assert first.metadata.seed != different.metadata.seed
    assert first.metadata.sample_id != different.metadata.sample_id
    assert not torch.equal(first.image, different.image)


def test_base_seed_and_split_are_explicit_parts_of_identity() -> None:
    train = _generator(base_seed=1).generate_sample("straight", split="train", index=4)
    validation = _generator(base_seed=1).generate_sample("straight", split="validation", index=4)
    other_base_seed = _generator(base_seed=2).generate_sample("straight", split="train", index=4)
    assert len({train.metadata.seed, validation.metadata.seed, other_base_seed.metadata.seed}) == 3
    assert len({train.metadata.sample_id, validation.metadata.sample_id, other_base_seed.metadata.sample_id}) == 3
    assert not torch.equal(train.image, other_base_seed.image)


def test_generation_uses_local_generator_without_polluting_global_rng() -> None:
    torch.manual_seed(98123)
    state_before = torch.random.get_rng_state().clone()
    _generator().generate_sample("dual_crossing", split="train", index=6, difficulty="hard")
    state_after = torch.random.get_rng_state()
    assert torch.equal(state_before, state_after)


def test_straight_single_component_matches_analytic_orientation_and_frequency() -> None:
    sample = _generator().generate_sample("straight", index=9, difficulty="easy")
    parameters = sample.metadata.parameters
    angle = float(parameters["orientation_rad"])
    frequency = float(parameters["frequency_cpp"])
    expected = _expected_splat([angle], sample.metadata.num_orientations)
    assert torch.allclose(sample.orientation_distribution_gt[:, 32, 36], expected, atol=2e-6, rtol=0.0)
    assert torch.all(sample.orientation_count == 1)
    assert torch.allclose(
        sample.component_frequency_gt[0],
        torch.full((64, 72), frequency),
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.count_nonzero(sample.component_frequency_gt[1:]) == 0


def test_linear_bin_splat_wraps_axially_between_last_and_first_bin() -> None:
    num_orientations = 8
    angle = math.pi - 0.25 * math.pi / num_orientations
    component = AnalyticComponent(
        orientation_rad=torch.full((2, 3), angle),
        frequency_cpp=torch.full((2, 3), 0.1),
        support=torch.ones(2, 3, dtype=torch.bool),
        signal=torch.ones(2, 3),
    )
    scene = FamilyScene(
        components=(component,),
        junction_mask=torch.zeros(2, 3, dtype=torch.bool),
        parameters={},
    )
    distribution, count, frequency, valid = _assemble_ground_truth(
        scene,
        num_orientations=num_orientations,
        max_components=3,
    )
    expected = torch.zeros(num_orientations)
    expected[-1] = 0.25
    expected[0] = 0.75
    assert torch.allclose(distribution[:, 0, 0], expected, atol=2e-6, rtol=0.0)
    assert torch.all(count == 1)
    assert torch.all(valid)
    assert torch.all(frequency[0] == 0.1)
    assert torch.count_nonzero(frequency[1:]) == 0


def test_crossing_center_has_two_splatted_components_and_component_frequencies() -> None:
    sample = _generator().generate_sample("crossing", index=5, difficulty="medium")
    row, column = 32, 36
    parameters = sample.metadata.parameters
    angles = [float(parameters["orientation_1_rad"]), float(parameters["orientation_2_rad"])]
    expected = _expected_splat(angles, sample.metadata.num_orientations)
    assert int(sample.orientation_count[0, row, column]) == 2
    assert sample.junction_mask[0, row, column]
    assert torch.allclose(sample.orientation_distribution_gt[:, row, column], expected, atol=2e-6, rtol=0.0)
    expected_frequency = torch.tensor(
        [float(parameters["frequency_1_cpp"]), float(parameters["frequency_2_cpp"]), 0.0],
    )
    assert torch.allclose(sample.component_frequency_gt[:, row, column], expected_frequency, atol=1e-7, rtol=0.0)


def test_bifurcation_junction_contains_all_three_analytic_branches() -> None:
    sample = _generator().generate_sample("bifurcation", index=3, difficulty="hard")
    triple = torch.nonzero(sample.orientation_count[0] == 3, as_tuple=False)
    assert triple.numel() > 0
    row, column = (int(value) for value in triple[0])
    parameters = sample.metadata.parameters
    angles = [
        float(parameters["parent_orientation_rad"]),
        float(parameters["left_branch_orientation_rad"]),
        float(parameters["right_branch_orientation_rad"]),
    ]
    expected = _expected_splat(angles, sample.metadata.num_orientations)
    assert sample.junction_mask[0, row, column]
    assert torch.allclose(sample.orientation_distribution_gt[:, row, column], expected, atol=2e-6, rtol=0.0)
    frequency = float(parameters["frequency_cpp"])
    assert torch.allclose(
        sample.component_frequency_gt[:, row, column],
        torch.full((3,), frequency),
        atol=1e-7,
        rtol=0.0,
    )


@pytest.mark.parametrize("family", ["radial", "concentric", "domain_boundary", "occlusion"])
def test_invalid_regions_are_uniform_with_zero_frequency(family: str) -> None:
    sample = _generator().generate_sample(family, index=8, difficulty="hard")
    invalid = ~sample.valid_mask[0]
    assert bool(invalid.any())
    expected = torch.full_like(sample.orientation_distribution_gt[:, invalid], 1.0 / 18.0)
    assert torch.allclose(sample.orientation_distribution_gt[:, invalid], expected, atol=2e-6, rtol=0.0)
    assert torch.count_nonzero(sample.component_frequency_gt[:, invalid]) == 0
    assert torch.all(sample.orientation_count[0, invalid] == 0)


def test_frequency_padding_channels_are_zero_for_every_family() -> None:
    generator = _generator()
    for index, family in enumerate(CANONICAL_FAMILIES):
        sample = generator.generate_sample(family, index=index)
        channel = torch.arange(3).view(3, 1, 1)
        padding = channel >= sample.orientation_count[0].unsqueeze(0)
        assert torch.count_nonzero(sample.component_frequency_gt.masked_select(padding)) == 0, family


def test_varying_frequency_ground_truth_changes_across_the_field() -> None:
    sample = _generator().generate_sample("varying_frequency", index=2, difficulty="hard")
    frequency = sample.component_frequency_gt[0]
    assert float(frequency.max() - frequency.min()) > 0.05
    assert float(frequency.min()) == pytest.approx(
        float(sample.metadata.parameters["frequency_min_cpp"]),
        abs=1e-7,
    )
    assert float(frequency.max()) == pytest.approx(
        float(sample.metadata.parameters["frequency_max_cpp"]),
        abs=1e-7,
    )


def test_varying_thickness_has_measurably_different_band_duty() -> None:
    sample = _generator().generate_sample("varying_thickness", index=2, difficulty="hard")
    angle = float(sample.metadata.parameters["orientation_rad"])
    rows = torch.arange(64, dtype=torch.float32)
    columns = torch.arange(72, dtype=torch.float32)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    x = column_grid - 35.5
    y = 31.5 - row_grid
    tangent = math.cos(angle) * x + math.sin(angle) * y
    low_region = tangent <= torch.quantile(tangent, 0.2)
    high_region = tangent >= torch.quantile(tangent, 0.8)
    low_bright_fraction = (sample.image[0, low_region] > 0.4).float().mean()
    high_bright_fraction = (sample.image[0, high_region] > 0.4).float().mean()
    assert float(high_bright_fraction - low_bright_fraction) > 0.25
    assert float(sample.metadata.parameters["thick_duty_cycle"]) > float(
        sample.metadata.parameters["thin_duty_cycle"],
    )


def test_domain_boundary_has_distinct_analytic_orientations_on_both_sides() -> None:
    sample = _generator().generate_sample("domain_boundary", index=12, difficulty="medium")
    parameters = sample.metadata.parameters
    left_expected = _expected_splat([float(parameters["left_orientation_rad"])], 18)
    right_expected = _expected_splat([float(parameters["right_orientation_rad"])], 18)
    assert sample.valid_mask[0, 32, 0] and sample.valid_mask[0, 32, -1]
    assert torch.allclose(sample.orientation_distribution_gt[:, 32, 0], left_expected, atol=2e-6, rtol=0.0)
    assert torch.allclose(sample.orientation_distribution_gt[:, 32, -1], right_expected, atol=2e-6, rtol=0.0)
    assert not torch.allclose(left_expected, right_expected)


def test_occlusion_contains_both_valid_structure_and_invalid_hole() -> None:
    sample = _generator().generate_sample("occlusion", index=0, difficulty="medium")
    assert bool(sample.valid_mask.any())
    assert bool((~sample.valid_mask).any())
    assert torch.allclose(
        sample.image[~sample.valid_mask],
        torch.full_like(sample.image[~sample.valid_mask], 0.05),
        atol=1e-7,
        rtol=0.0,
    )


def test_convenience_function_and_difficulty_are_explicit() -> None:
    config = StructFieldBenchConfig(height=48, width=48)
    easy = generate_sample("curved", index=3, difficulty="easy", config=config)
    hard = generate_sample("curved", index=3, difficulty="hard", config=config)
    assert easy.metadata.difficulty == "easy"
    assert hard.metadata.difficulty == "hard"
    assert easy.metadata.seed == hard.metadata.seed
    assert easy.metadata.sample_id != hard.metadata.sample_id
    assert not torch.equal(easy.image, hard.image)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"height": 23}, ValueError),
        ({"width": True}, TypeError),
        ({"num_orientations": 1}, ValueError),
        ({"max_components": 2}, ValueError),
        ({"base_frequency_cpp": 0.0}, ValueError),
        ({"base_frequency_cpp": float("nan")}, ValueError),
        ({"base_seed": -1}, ValueError),
        ({"background_intensity": 1.0}, ValueError),
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        StructFieldBenchConfig(**kwargs)  # type: ignore[arg-type]


def test_generator_rejects_unknown_family_difficulty_split_and_index() -> None:
    generator = _generator()
    with pytest.raises(ValueError, match="family"):
        generator.generate_sample("not-a-family")
    with pytest.raises(ValueError, match="difficulty"):
        generator.generate_sample("straight", difficulty="extreme")
    with pytest.raises(ValueError, match="split"):
        generator.generate_sample("straight", split="holdout")
    with pytest.raises(ValueError, match="index"):
        generator.generate_sample("straight", index=-1)
