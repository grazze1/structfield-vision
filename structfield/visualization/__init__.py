"""Publication-oriented visual diagnostics for StructField-Vision."""

from .plots import (
    AXIAL_CMAP,
    SCALAR_CMAPS,
    GateReduction,
    plot_coherence_heatmap,
    plot_frequency_heatmap,
    plot_gate_heatmap,
    plot_orientation_overlay,
    plot_orientation_probability,
    plot_orientation_wheel,
    plot_uncertainty_heatmap,
    save_figure,
)
from .style import PUBLICATION_RCPARAMS, apply_publication_style

__all__ = [
    "AXIAL_CMAP",
    "PUBLICATION_RCPARAMS",
    "SCALAR_CMAPS",
    "GateReduction",
    "apply_publication_style",
    "plot_coherence_heatmap",
    "plot_frequency_heatmap",
    "plot_gate_heatmap",
    "plot_orientation_overlay",
    "plot_orientation_probability",
    "plot_orientation_wheel",
    "plot_uncertainty_heatmap",
    "save_figure",
]
