"""Publication-format export and figure lifecycle tests."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from structfield.visualization import (
    apply_publication_style,
    plot_uncertainty_heatmap,
    save_figure,
)


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    yield
    plt.close("all")


def test_save_figure_exports_editable_svg_png_and_pdf(tmp_path: Path) -> None:
    with matplotlib.rc_context():
        apply_publication_style()
        fig, ax = plot_uncertainty_heatmap(np.full((7, 11), 0.25, dtype=np.float32))
        ax.set_title("Editable structural uncertainty")
        output_base = tmp_path / "nested" / "uncertainty"

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            paths = tuple(
                save_figure(
                    fig,
                    output_base,
                    formats=("svg", "png", "pdf"),
                    dpi=144,
                    close=False,
                )
            )

    assert paths == (
        output_base.with_suffix(".svg"),
        output_base.with_suffix(".png"),
        output_base.with_suffix(".pdf"),
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    svg_text = paths[0].read_text(encoding="utf-8")
    assert "<text" in svg_text
    assert "Editable structural uncertainty" in svg_text
    assert paths[1].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert paths[2].read_bytes().startswith(b"%PDF-")
    assert plt.fignum_exists(fig.number)


def test_save_figure_defaults_close_the_figure_and_return_both_paths(tmp_path: Path) -> None:
    fig, _ = plot_uncertainty_heatmap(np.zeros((5, 7), dtype=np.float32))
    output_base = tmp_path / "default-export"

    paths = tuple(save_figure(fig, output_base))

    assert paths == (
        output_base.with_suffix(".svg"),
        output_base.with_suffix(".png"),
    )
    assert all(path.is_file() for path in paths)
    assert not plt.fignum_exists(fig.number)


@pytest.mark.parametrize(
    ("formats", "dpi"),
    (
        ((), 300),
        (("jpg",), 300),
        (("svg", "svg"), 300),
        (("svg", "png"), 0),
        (("svg", "png"), float("nan")),
    ),
)
def test_save_figure_validates_formats_and_dpi(
    tmp_path: Path,
    formats: tuple[str, ...],
    dpi: float,
) -> None:
    fig, _ = plot_uncertainty_heatmap(np.zeros((5, 7), dtype=np.float32))

    with pytest.raises((TypeError, ValueError)):
        save_figure(fig, tmp_path / "invalid", formats=formats, dpi=dpi)
