"""Publication-oriented Matplotlib defaults for structural-field diagnostics."""

from __future__ import annotations

import matplotlib as mpl

PUBLICATION_RCPARAMS: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "savefig.facecolor": "white",
}


def apply_publication_style() -> None:
    """Apply editable-text, journal-width-safe Matplotlib defaults."""
    # Matplotlib's stubs enumerate every rcParam as a Literal key, while this
    # public mapping intentionally remains extensible and runtime-validated.
    mpl.rcParams.update(PUBLICATION_RCPARAMS)  # type: ignore[arg-type]


__all__ = ["PUBLICATION_RCPARAMS", "apply_publication_style"]
