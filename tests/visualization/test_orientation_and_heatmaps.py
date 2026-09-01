"""Numerical and coordinate-system contracts for visualization helpers."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from structfield.visualization import (
    plot_coherence_heatmap,
    plot_frequency_heatmap,
    plot_gate_heatmap,
    plot_orientation_overlay,
    plot_orientation_probability,
    plot_uncertainty_heatmap,
)


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    yield
    plt.close("all")


def _assert_plot_result(result: tuple[Figure, Axes]) -> tuple[Figure, Axes]:
    fig, ax = result
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.figure is fig
    fig.canvas.draw()
    return fig, ax


def test_orientation_probability_encodes_uniform_and_concentrated_fields() -> None:
    probability = np.zeros((8, 5, 7), dtype=np.float32)
    probability[2] = 1.0
    fig, ax = _assert_plot_result(plot_orientation_probability(probability))
    rgba = np.asarray(ax.images[0].get_array())
    assert rgba.shape == (5, 7, 4)
    assert np.ptp(rgba[..., :3], axis=-1).mean() > 0.05

    with pytest.raises(ValueError, match="sum to one"):
        plot_orientation_probability(probability * 0.5)


def _orientation_segments(ax: Axes) -> np.ndarray:
    collections = [artist for artist in ax.collections if isinstance(artist, LineCollection)]
    assert len(collections) == 1
    segments = np.asarray(collections[0].get_segments(), dtype=np.float64)
    assert segments.ndim == 3 and segments.shape[1:] == (2, 2)
    assert segments.shape[0] > 0
    return segments


@pytest.mark.parametrize(
    ("angle", "horizontal"),
    ((0.0, True), (math.pi / 2.0, False)),
)
def test_orientation_overlay_uses_canonical_axial_image_coordinates(
    angle: float,
    horizontal: bool,
) -> None:
    image = np.linspace(0.0, 1.0, 9 * 13, dtype=np.float32).reshape(9, 13)
    orientation = np.full(image.shape, angle, dtype=np.float32)
    coherence = np.ones_like(image)

    _, ax = _assert_plot_result(plot_orientation_overlay(image, orientation, coherence=coherence, stride=3))
    deltas = np.diff(_orientation_segments(ax), axis=1)[:, 0, :]

    if horizontal:
        assert np.allclose(deltas[:, 1], 0.0, atol=1e-6)
        assert np.all(np.abs(deltas[:, 0]) > 0.0)
    else:
        assert np.allclose(deltas[:, 0], 0.0, atol=1e-6)
        assert np.all(np.abs(deltas[:, 1]) > 0.0)


def test_positive_axial_angle_is_counter_clockwise_on_the_displayed_image() -> None:
    image = np.zeros((7, 11), dtype=np.float32)
    orientation = np.full_like(image, math.pi / 4.0)

    _, ax = _assert_plot_result(plot_orientation_overlay(image, orientation, stride=2))
    deltas = np.diff(_orientation_segments(ax), axis=1)[:, 0, :]

    # Image coordinates are x-right/y-down, hence a positive visual
    # counter-clockwise angle has opposite-sign x and y components.
    assert np.all(np.abs(deltas[:, 0]) > 0.0)
    assert np.all(np.abs(deltas[:, 1]) > 0.0)
    assert np.all(deltas[:, 0] * deltas[:, 1] < 0.0)
    assert np.allclose(np.abs(deltas[:, 0]), np.abs(deltas[:, 1]), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", ((3, 17), (17, 3), (3, 5), (1, 1)))
def test_orientation_overlay_default_stride_samples_every_legal_small_shape(
    shape: tuple[int, int],
) -> None:
    image = np.zeros(shape, dtype=np.float32)
    orientation = np.full(shape, math.pi / 4.0, dtype=np.float32)

    _, ax = _assert_plot_result(plot_orientation_overlay(image, orientation))
    segments = _orientation_segments(ax)
    midpoints = segments.mean(axis=1)

    assert segments.shape[0] >= 1
    assert np.all((0.0 <= midpoints[:, 0]) & (midpoints[:, 0] < shape[1]))
    assert np.all((0.0 <= midpoints[:, 1]) & (midpoints[:, 1] < shape[0]))


def test_orientation_overlay_preserves_odd_sized_inputs() -> None:
    image = np.arange(9 * 13, dtype=np.float32).reshape(9, 13) / float(9 * 13)
    orientation = np.linspace(0.0, math.pi, image.size, endpoint=False, dtype=np.float32).reshape(image.shape)
    coherence = np.linspace(0.1, 1.0, image.size, dtype=np.float32).reshape(image.shape)
    snapshots = tuple(value.copy() for value in (image, orientation, coherence))

    _, ax = _assert_plot_result(plot_orientation_overlay(image, orientation, coherence=coherence, stride=4))

    assert np.asarray(ax.images[0].get_array()).shape == image.shape
    for value, snapshot in zip((image, orientation, coherence), snapshots, strict=True):
        assert np.array_equal(value, snapshot)


@pytest.mark.parametrize(
    ("orientation", "coherence"),
    (
        (np.zeros((5, 7), dtype=np.float32), np.ones((5, 8), dtype=np.float32)),
        (np.full((5, 7), np.nan, dtype=np.float32), None),
        (np.zeros((5, 7), dtype=np.float32), np.full((5, 7), 1.01, dtype=np.float32)),
    ),
)
def test_orientation_overlay_rejects_shape_range_and_finite_violations(
    orientation: np.ndarray,
    coherence: np.ndarray | None,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        plot_orientation_overlay(
            np.zeros((5, 7), dtype=np.float32),
            orientation,
            coherence=coherence,
            stride=2,
        )


def test_orientation_overlay_rejects_exact_axial_period() -> None:
    image = np.zeros((5, 7), dtype=np.float64)
    orientation = np.full_like(image, math.pi)

    with pytest.raises(ValueError):
        plot_orientation_overlay(image, orientation, stride=2)


@pytest.mark.parametrize(
    "image",
    (
        np.full((5, 7), -0.01, dtype=np.float32),
        np.full((5, 7), 1.01, dtype=np.float32),
        np.full((5, 7), np.nan, dtype=np.float32),
        np.zeros((2, 5, 7), dtype=np.float32),
    ),
)
def test_orientation_overlay_rejects_invalid_image_values_and_shapes(image: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        plot_orientation_overlay(image, np.zeros((5, 7), dtype=np.float32), stride=2)


@pytest.mark.parametrize("stride", (0, -1, 1.5, True))
def test_orientation_overlay_requires_a_positive_integer_stride(stride: object) -> None:
    values = np.zeros((5, 7), dtype=np.float32)
    with pytest.raises((TypeError, ValueError)):
        plot_orientation_overlay(values, values, stride=stride)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("plotter", "values", "expected_clim"),
    (
        (
            plot_uncertainty_heatmap,
            np.linspace(0.0, 1.0, 9 * 13, dtype=np.float32).reshape(9, 13),
            (0.0, 1.0),
        ),
        (
            plot_coherence_heatmap,
            np.linspace(0.0, 1.0, 9 * 13, dtype=np.float32).reshape(9, 13),
            (0.0, 1.0),
        ),
        (
            plot_frequency_heatmap,
            np.linspace(0.0, 0.25, 9 * 13, dtype=np.float32).reshape(9, 13),
            (0.0, 0.25),
        ),
    ),
)
def test_scalar_heatmaps_preserve_odd_shapes_and_semantic_color_limits(
    plotter: object,
    values: np.ndarray,
    expected_clim: tuple[float, float],
) -> None:
    snapshot = values.copy()
    _, ax = _assert_plot_result(plotter(values))  # type: ignore[operator]
    rendered = np.asarray(ax.images[0].get_array())

    assert rendered.shape == values.shape
    assert np.allclose(ax.images[0].get_clim(), expected_clim)
    assert np.array_equal(values, snapshot)


def test_zero_frequency_heatmap_uses_nonnegative_nondegenerate_color_limits() -> None:
    values = np.zeros((3, 5), dtype=np.float32)

    _, ax = _assert_plot_result(plot_frequency_heatmap(values))
    vmin, vmax = ax.images[0].get_clim()

    assert vmin == 0.0
    assert vmax > 0.0


def test_scalar_heatmaps_accept_detached_cpu_tensors_without_mutation() -> None:
    values = torch.linspace(0.0, 1.0, 7 * 11, dtype=torch.float32).reshape(7, 11)
    snapshot = values.clone()

    _, ax = _assert_plot_result(plot_uncertainty_heatmap(values))

    assert np.asarray(ax.images[0].get_array()).shape == (7, 11)
    assert torch.equal(values, snapshot)


@pytest.mark.parametrize("shape", ((1, 7, 11), (1, 1, 7, 11)))
def test_scalar_heatmaps_accept_singleton_core_dimensions(shape: tuple[int, ...]) -> None:
    values = np.linspace(0.0, 0.25, 7 * 11, dtype=np.float32).reshape(shape)

    _, ax = _assert_plot_result(plot_frequency_heatmap(values))

    assert np.asarray(ax.images[0].get_array()).shape == (7, 11)


@pytest.mark.parametrize(
    ("plotter", "bad_values"),
    (
        (plot_uncertainty_heatmap, np.full((5, 7), -0.01, dtype=np.float32)),
        (plot_uncertainty_heatmap, np.full((5, 7), 1.01, dtype=np.float32)),
        (plot_coherence_heatmap, np.full((5, 7), -0.01, dtype=np.float32)),
        (plot_coherence_heatmap, np.full((5, 7), np.inf, dtype=np.float32)),
        (plot_frequency_heatmap, np.full((5, 7), -0.01, dtype=np.float32)),
        (plot_frequency_heatmap, np.full((5, 7), np.nan, dtype=np.float32)),
        (plot_frequency_heatmap, np.zeros((2, 5, 7), dtype=np.float32)),
    ),
)
def test_scalar_heatmaps_reject_invalid_shape_range_or_finiteness(
    plotter: object,
    bad_values: np.ndarray,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        plotter(bad_values)  # type: ignore[operator]


def test_gate_heatmap_reduces_channels_and_supports_singleton_batch() -> None:
    gate = np.stack(
        (
            np.full((9, 13), 0.1, dtype=np.float32),
            np.full((9, 13), 0.5, dtype=np.float32),
            np.full((9, 13), 0.9, dtype=np.float32),
        )
    )[None]
    snapshot = gate.copy()

    _, mean_ax = _assert_plot_result(plot_gate_heatmap(gate))
    _, max_ax = _assert_plot_result(plot_gate_heatmap(gate, reduction="max"))

    assert np.allclose(np.asarray(mean_ax.images[0].get_array()), 0.5)
    assert np.allclose(np.asarray(max_ax.images[0].get_array()), 0.9)
    assert mean_ax.images[0].get_clim() == (0.0, 1.0)
    assert max_ax.images[0].get_clim() == (0.0, 1.0)
    assert np.array_equal(gate, snapshot)


@pytest.mark.parametrize(
    ("gate", "reduction"),
    (
        (np.zeros((2, 3, 5, 7), dtype=np.float32), "mean"),
        (np.zeros((3, 5, 7), dtype=np.float32), "median"),
        (np.full((3, 5, 7), 1.01, dtype=np.float32), "mean"),
        (np.full((3, 5, 7), np.nan, dtype=np.float32), "mean"),
    ),
)
def test_gate_heatmap_rejects_ambiguous_batch_invalid_reduction_and_bad_values(
    gate: np.ndarray,
    reduction: str,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        plot_gate_heatmap(gate, reduction=reduction)
