"""Public API and plotting-style contracts for Stage 12 visualizations."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import structfield.visualization as visualization
from structfield.visualization import (
    apply_publication_style,
    plot_orientation_wheel,
)

PUBLIC_FUNCTIONS = {
    "apply_publication_style",
    "plot_orientation_wheel",
    "plot_orientation_overlay",
    "plot_orientation_probability",
    "plot_uncertainty_heatmap",
    "plot_frequency_heatmap",
    "plot_coherence_heatmap",
    "plot_gate_heatmap",
    "save_figure",
}


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


def test_visualization_public_api_is_explicit_and_callable() -> None:
    assert PUBLIC_FUNCTIONS <= set(visualization.__all__)
    for name in PUBLIC_FUNCTIONS:
        value = getattr(visualization, name)
        assert isinstance(value, Callable)


def test_publication_style_sets_editable_vector_and_export_defaults() -> None:
    with matplotlib.rc_context():
        matplotlib.rcParams.update(
            {
                "font.size": 17.0,
                "svg.fonttype": "path",
                "pdf.fonttype": 3,
                "axes.spines.right": True,
                "axes.spines.top": True,
            }
        )

        assert apply_publication_style() is None

        assert 7.0 <= float(matplotlib.rcParams["font.size"]) <= 12.0
        assert matplotlib.rcParams["svg.fonttype"] == "none"
        assert matplotlib.rcParams["pdf.fonttype"] == 42
        assert matplotlib.rcParams["axes.spines.right"] is False
        assert matplotlib.rcParams["axes.spines.top"] is False


def test_orientation_wheel_returns_a_renderable_axes_with_visual_content() -> None:
    fig, ax = _assert_plot_result(plot_orientation_wheel())

    assert ax.has_data() or bool(ax.images) or bool(ax.collections) or bool(ax.patches)
    assert fig.axes
