"""Axial-angle conventions and circular top-k mode selection."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from structfield.core import axial_angle_distance, doubled_angle, wrap_axial_angle
from tests.structfield.test_field import make_field


def test_axial_periodicity_near_zero_and_pi() -> None:
    one_degree = torch.tensor(math.radians(1.0))
    one_seventy_nine = torch.tensor(math.radians(179.0))
    distance = axial_angle_distance(one_degree, one_seventy_nine)
    assert float(distance) == pytest.approx(math.radians(2.0), abs=1e-6)
    assert float(wrap_axial_angle(torch.tensor(-math.radians(1.0)))) == pytest.approx(
        math.radians(179.0),
        abs=1e-6,
    )
    wrapped = wrap_axial_angle(torch.tensor([-1e-7], dtype=torch.float32))
    assert 0.0 <= float(wrapped.item()) < math.pi


def test_doubled_angle_makes_zero_and_pi_equivalent() -> None:
    values = torch.tensor([[[[0.0]]], [[[math.pi]]]])
    encoded = doubled_angle(values, dim=1)
    assert torch.allclose(encoded[0], encoded[1], atol=1e-6)


@pytest.mark.parametrize("num_orientations", [8, 16, 32])
def test_topk_modes_have_minimum_axial_separation(num_orientations: int) -> None:
    field = make_field(num_orientations)
    modes = field.topk_orientations(k=2, min_separation=math.pi / 4.0)
    assert modes.angles.shape == (2, 2, 9, 11)
    separation = axial_angle_distance(modes.angles[:, 0], modes.angles[:, 1])
    assert bool((separation >= math.pi / 4.0 - 1e-6).all())
    assert torch.equal(modes.indices[:, 0], torch.zeros_like(modes.indices[:, 0]))
    assert torch.equal(
        modes.indices[:, 1],
        torch.full_like(modes.indices[:, 1], num_orientations // 2),
    )


def test_topk_suppresses_neighbor_across_axial_wraparound() -> None:
    field = make_field(16, batch=1, height=3, width=3)
    probability = torch.zeros_like(field.orientation_prob)
    probability[:, 0] = 0.45
    probability[:, 15] = 0.35
    probability[:, 8] = 0.20
    field = replace(field, orientation_prob=probability)
    modes = field.topk_orientations(k=2, min_separation=math.radians(20.0))
    assert torch.equal(modes.indices[:, 0], torch.zeros_like(modes.indices[:, 0]))
    assert torch.equal(modes.indices[:, 1], torch.full_like(modes.indices[:, 1], 8))


def test_topk_returns_probabilities_and_preserves_gradient() -> None:
    field = make_field(8, batch=1, height=2, width=2)
    logits = torch.randn(1, 8, 2, 2, requires_grad=True)
    field = replace(field, orientation_prob=torch.softmax(logits, dim=1))
    modes = field.topk_orientations(k=2, min_separation=math.pi / 8.0)
    modes.probabilities.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_topk_rejects_impossible_or_invalid_requests() -> None:
    field = make_field(8)
    with pytest.raises(ValueError, match="cannot satisfy"):
        field.topk_orientations(k=3, min_separation=math.pi / 2.0)
    with pytest.raises(ValueError, match="k must lie"):
        field.topk_orientations(k=0)
    with pytest.raises(ValueError, match="min_separation"):
        field.topk_orientations(k=2, min_separation=math.pi)


def test_topk_falls_back_to_feasible_uniform_configuration_after_greedy_dead_end() -> None:
    field = make_field(10, batch=1, height=2, width=2)
    probability = torch.full_like(field.orientation_prob, 0.001125)
    probability[:, 0] = 0.600
    probability[:, 5] = 0.391
    field = replace(field, orientation_prob=probability)
    modes = field.topk_orientations(k=3, min_separation=0.9)
    for first in range(3):
        for second in range(first + 1, 3):
            separation = axial_angle_distance(modes.angles[:, first], modes.angles[:, second])
            assert bool((separation >= 0.9 - 1e-6).all())
    assert torch.equal(modes.indices[:, 0], torch.zeros_like(modes.indices[:, 0]))


def test_topk_fallback_is_independent_across_batch_and_space() -> None:
    field = make_field(10, batch=1, height=1, width=1)
    probability_a = torch.full_like(field.orientation_prob, 0.001)
    probability_a[:, 0] = 0.500
    probability_a[:, 4] = 0.300
    probability_a[:, 7] = 0.193
    probability_b = torch.full_like(field.orientation_prob, 0.001125)
    probability_b[:, 0] = 0.600
    probability_b[:, 5] = 0.391
    modes_a = replace(field, orientation_prob=probability_a).topk_orientations(k=3, min_separation=0.9)
    modes_b = replace(field, orientation_prob=probability_b).topk_orientations(k=3, min_separation=0.9)

    batch_field = make_field(10, batch=2, height=1, width=1)
    batch_probability = torch.cat((probability_a, probability_b), dim=0)
    batch_modes = replace(batch_field, orientation_prob=batch_probability).topk_orientations(k=3, min_separation=0.9)
    assert torch.equal(batch_modes.indices[:1], modes_a.indices)
    assert torch.equal(batch_modes.indices[1:], modes_b.indices)

    spatial_field = make_field(10, batch=1, height=1, width=2)
    spatial_probability = torch.cat((probability_a, probability_b), dim=-1)
    spatial_modes = replace(spatial_field, orientation_prob=spatial_probability).topk_orientations(
        k=3,
        min_separation=0.9,
    )
    assert torch.equal(spatial_modes.indices[..., :1], modes_a.indices)
    assert torch.equal(spatial_modes.indices[..., 1:], modes_b.indices)


def test_orientation_modes_to_keeps_integer_indices() -> None:
    modes = make_field(8).topk_orientations().to(dtype=torch.float64)
    assert modes.angles.dtype == torch.float64
    assert modes.probabilities.dtype == torch.float64
    assert modes.indices.dtype == torch.long


def test_exact_pi_is_rejected_by_mode_contract() -> None:
    field = make_field(8)
    modes = field.topk_orientations(k=1)
    with pytest.raises(ValueError, match=r"\[0, pi\)"):
        replace(modes, angles=torch.full_like(modes.angles, math.pi))
