"""Scientific visualizations for axial structural fields and reliability maps."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeAlias

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from .style import apply_publication_style

ArrayLike: TypeAlias = np.ndarray | torch.Tensor
GateReduction: TypeAlias = Literal["mean", "max"] | int

AXIAL_CMAP = "twilight_shifted"
SCALAR_CMAPS = {
    "uncertainty": "magma",
    "frequency": "cividis",
    "coherence": "cividis",
    "gate": "magma",
}


def _numpy_copy(value: ArrayLike, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise TypeError(f"{name} must be a numpy array or torch.Tensor")
    if array.dtype.kind not in "buif":
        raise TypeError(f"{name} must use a real numeric dtype")
    result = np.array(array, dtype=np.float64, copy=True)
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _scalar_map(
    value: ArrayLike,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    array = _numpy_copy(value, name=name)
    if array.ndim == 4 and array.shape[:2] == (1, 1):
        array = array[0, 0]
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must have shape [H,W], [1,H,W], or [1,1,H,W]")
    tolerance = 1e-7
    if minimum is not None and float(array.min()) < minimum - tolerance:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and float(array.max()) > maximum + tolerance:
        raise ValueError(f"{name} must be at most {maximum}")
    return array


def _image_array(value: ArrayLike) -> np.ndarray:
    array = _numpy_copy(value, name="image")
    if array.ndim == 4:
        if array.shape[0] != 1 or array.shape[1] not in {1, 3}:
            raise ValueError("image BCHW input must have B=1 and C=1 or C=3")
        array = np.moveaxis(array[0], 0, -1)
    elif array.ndim == 3:
        first_is_channel = array.shape[0] in {1, 3}
        last_is_channel = array.shape[-1] in {1, 3}
        if first_is_channel == last_is_channel:
            raise ValueError("3D image shape is ambiguous; use unambiguous CHW or HWC")
        if first_is_channel:
            array = np.moveaxis(array, 0, -1)
    elif array.ndim != 2:
        raise ValueError("image must have shape [H,W], CHW, HWC, or BCHW with B=1")
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if min(array.shape[:2]) < 1:
        raise ValueError("image spatial dimensions must be non-empty")
    if float(array.min()) < -1e-7 or float(array.max()) > 1.0 + 1e-7:
        raise ValueError("image values must lie in [0,1]")
    return np.clip(array, 0.0, 1.0)


def _gate_map(value: ArrayLike, reduction: GateReduction) -> np.ndarray:
    array = _numpy_copy(value, name="gate")
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError("gate batch dimension must equal one for visualization")
        array = array[0]
    if array.ndim == 2:
        result = array
    elif array.ndim == 3:
        if isinstance(reduction, bool):
            raise TypeError("gate reduction must be 'mean', 'max', or a channel index")
        if reduction == "mean":
            result = array.mean(axis=0)
        elif reduction == "max":
            result = array.max(axis=0)
        elif isinstance(reduction, int):
            if not 0 <= reduction < array.shape[0]:
                raise ValueError("gate channel index is out of range")
            result = array[reduction]
        else:
            raise ValueError("gate reduction must be 'mean', 'max', or a channel index")
    else:
        raise ValueError("gate must have shape [H,W], [C,H,W], or [1,C,H,W]")
    if float(result.min()) < -1e-7 or float(result.max()) > 1.0 + 1e-7:
        raise ValueError("gate must lie in [0,1]")
    return np.clip(result, 0.0, 1.0)


def _new_axis(ax: Axes | None, *, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    apply_publication_style()
    if ax is None:
        fig, created_ax = plt.subplots(figsize=figsize, constrained_layout=True)
        return fig, created_ax
    if not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib Axes")
    figure = ax.figure
    if not isinstance(figure, Figure):  # pragma: no cover - Matplotlib invariant
        raise TypeError("ax must belong to a matplotlib Figure")
    return figure, ax


def plot_orientation_wheel(
    *,
    ax: Axes | None = None,
    resolution: int = 256,
    title: str | None = "Axial orientation",
) -> tuple[Figure, Axes]:
    """Plot a cyclic key for axial tangent angles in ``[0, pi)``."""
    if isinstance(resolution, bool) or not isinstance(resolution, int):
        raise TypeError("resolution must be an integer")
    if resolution < 32:
        raise ValueError("resolution must be at least 32")
    fig, axis = _new_axis(ax, figsize=(3.5, 3.1))
    coordinate = np.linspace(-1.0, 1.0, resolution)
    xx, yy = np.meshgrid(coordinate, coordinate)
    radius = np.hypot(xx, yy)
    angle = np.mod(np.arctan2(yy, xx), math.pi)
    rgba = mpl.colormaps[AXIAL_CMAP](angle / math.pi)
    rgba[..., 3] = (radius <= 1.0).astype(float)
    axis.imshow(rgba, origin="lower", extent=(-1, 1, -1, 1), interpolation="bilinear")
    axis.text(1.05, 0.0, "0", ha="left", va="center")
    axis.text(0.0, 1.05, "π/2", ha="center", va="bottom")
    axis.text(-1.05, 0.0, "0", ha="right", va="center")
    axis.set_xlim(-1.18, 1.18)
    axis.set_ylim(-1.18, 1.18)
    axis.set_aspect("equal")
    axis.set_axis_off()
    if title is not None:
        axis.set_title(title)
    return fig, axis


def plot_orientation_overlay(
    image: ArrayLike,
    orientation: ArrayLike,
    *,
    coherence: ArrayLike | None = None,
    ax: Axes | None = None,
    stride: int = 12,
    title: str | None = "Top-1 axial orientation",
) -> tuple[Figure, Axes]:
    """Overlay centered, arrow-free axial tangent segments on an image."""
    if isinstance(stride, bool) or not isinstance(stride, int):
        raise TypeError("stride must be an integer")
    if stride < 2:
        raise ValueError("stride must be at least two pixels")
    image_array = _image_array(image)
    orientation_map = _scalar_map(orientation, name="orientation", minimum=0.0, maximum=math.pi)
    if float(orientation_map.max()) >= math.pi:
        raise ValueError("orientation must lie in [0,pi)")
    if orientation_map.shape != image_array.shape[:2]:
        raise ValueError("orientation and image must share spatial dimensions")
    coherence_map = None
    if coherence is not None:
        coherence_map = _scalar_map(coherence, name="coherence", minimum=0.0, maximum=1.0)
        if coherence_map.shape != orientation_map.shape:
            raise ValueError("coherence and orientation must share spatial dimensions")

    fig, axis = _new_axis(ax, figsize=(3.5, 3.5))
    if image_array.ndim == 2:
        axis.imshow(image_array, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
    else:
        axis.imshow(image_array, vmin=0.0, vmax=1.0, origin="upper")

    height, width = orientation_map.shape
    y_start = min(stride // 2, (height - 1) // 2)
    x_start = min(stride // 2, (width - 1) // 2)
    y_values = np.arange(y_start, height, stride)
    x_values = np.arange(x_start, width, stride)
    yy, xx = np.meshgrid(y_values, x_values, indexing="ij")
    sampled = orientation_map[yy, xx]
    half_length = 0.36 * stride
    dx = half_length * np.cos(sampled)
    dy = -half_length * np.sin(sampled)
    segments = np.stack(
        (
            np.stack((xx - dx, yy - dy), axis=-1),
            np.stack((xx + dx, yy + dy), axis=-1),
        ),
        axis=-2,
    ).reshape(-1, 2, 2)
    colors = mpl.colormaps[AXIAL_CMAP]((sampled / math.pi).reshape(-1))
    if coherence_map is not None:
        colors[:, 3] = 0.2 + 0.8 * coherence_map[yy, xx].reshape(-1)
    collection = LineCollection(segments.tolist(), colors=colors, linewidths=1.0, capstyle="round")
    axis.add_collection(collection)
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)
    axis.set_axis_off()
    if title is not None:
        axis.set_title(title)
    return fig, axis


def plot_orientation_probability(
    probability: ArrayLike,
    *,
    bin_centers: Sequence[float] | None = None,
    ax: Axes | None = None,
    title: str = "Orientation probability",
) -> tuple[Figure, Axes]:
    """Render axial posterior mean as hue and concentration as saturation.

    ``probability`` accepts ``[K,H,W]`` or ``[1,K,H,W]`` and must contain a
    normalized, finite, non-negative distribution. Bin centers default to the
    uniform axial grid ``k*pi/K`` and otherwise use radians in ``[0,pi)``.
    """

    array = _numpy_copy(probability, name="probability")
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or min(array.shape) < 1 or array.shape[0] < 2:
        raise ValueError("probability must have shape [K,H,W] or [1,K,H,W] with K >= 2")
    if float(array.min()) < -1e-7:
        raise ValueError("probability must be non-negative")
    total = array.sum(axis=0)
    if not np.allclose(total, 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("probability must sum to one over orientation channels")
    channels = array.shape[0]
    centers = (
        np.arange(channels, dtype=np.float64) * math.pi / channels
        if bin_centers is None
        else np.asarray(tuple(bin_centers), dtype=np.float64)
    )
    if centers.shape != (channels,) or not np.isfinite(centers).all():
        raise ValueError("bin_centers must contain one finite value per channel")
    if float(centers.min()) < 0.0 or float(centers.max()) >= math.pi or np.any(np.diff(centers) <= 0.0):
        raise ValueError("bin_centers must be strictly increasing in [0,pi)")
    cosine = (array * np.cos(2.0 * centers)[:, None, None]).sum(axis=0)
    sine = (array * np.sin(2.0 * centers)[:, None, None]).sum(axis=0)
    angle = np.mod(0.5 * np.arctan2(sine, cosine), math.pi)
    concentration = np.clip(np.hypot(cosine, sine), 0.0, 1.0)
    rgba = mpl.colormaps[AXIAL_CMAP](angle / math.pi)
    rgba[..., :3] = 1.0 - concentration[..., None] * (1.0 - rgba[..., :3])
    fig, axis = _new_axis(ax, figsize=(3.5, 3.1))
    axis.imshow(rgba, origin="upper", interpolation="nearest")
    axis.set_title(title)
    axis.set_axis_off()
    return fig, axis


def _plot_scalar_heatmap(
    value: ArrayLike,
    *,
    kind: Literal["uncertainty", "frequency", "coherence"],
    ax: Axes | None,
    title: str,
) -> tuple[Figure, Axes]:
    if kind in {"uncertainty", "coherence"}:
        array = _scalar_map(value, name=kind, minimum=0.0, maximum=1.0)
        vmin, vmax = 0.0, 1.0
    else:
        array = _scalar_map(value, name=kind, minimum=0.0)
        vmin = 0.0
        observed_maximum = float(array.max())
        vmax = observed_maximum if observed_maximum > 0.0 else float(np.finfo(np.float32).eps)
    fig, axis = _new_axis(ax, figsize=(3.5, 3.1))
    image_artist = axis.imshow(
        array,
        cmap=SCALAR_CMAPS[kind],
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="nearest",
    )
    colorbar = fig.colorbar(image_artist, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("cycles/pixel" if kind == "frequency" else kind.capitalize())
    axis.set_title(title)
    axis.set_axis_off()
    return fig, axis


def plot_uncertainty_heatmap(
    uncertainty: ArrayLike,
    *,
    ax: Axes | None = None,
    title: str = "Structural uncertainty",
) -> tuple[Figure, Axes]:
    return _plot_scalar_heatmap(uncertainty, kind="uncertainty", ax=ax, title=title)


def plot_frequency_heatmap(
    frequency: ArrayLike,
    *,
    ax: Axes | None = None,
    title: str = "Local frequency",
) -> tuple[Figure, Axes]:
    return _plot_scalar_heatmap(frequency, kind="frequency", ax=ax, title=title)


def plot_coherence_heatmap(
    coherence: ArrayLike,
    *,
    ax: Axes | None = None,
    title: str = "Structural coherence",
) -> tuple[Figure, Axes]:
    return _plot_scalar_heatmap(coherence, kind="coherence", ax=ax, title=title)


def plot_gate_heatmap(
    gate: ArrayLike,
    *,
    reduction: GateReduction = "mean",
    ax: Axes | None = None,
    title: str = "Reliability gate",
) -> tuple[Figure, Axes]:
    """Plot a scalar or channel-reduced downstream reliability gate."""
    array = _gate_map(gate, reduction)
    fig, axis = _new_axis(ax, figsize=(3.5, 3.1))
    image_artist = axis.imshow(
        array,
        cmap=SCALAR_CMAPS["gate"],
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        interpolation="nearest",
    )
    colorbar = fig.colorbar(image_artist, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Gate value")
    axis.set_title(title)
    axis.set_axis_off()
    return fig, axis


def save_figure(
    fig: Figure,
    output_base: str | Path,
    *,
    formats: Sequence[str] = ("svg", "png"),
    dpi: int = 300,
    close: bool = True,
) -> tuple[Path, ...]:
    """Save a single-panel diagnostic with SVG as the editable primary output."""
    if not isinstance(fig, Figure):
        raise TypeError("fig must be a matplotlib Figure")
    if isinstance(formats, (str, bytes, bytearray)):
        raise TypeError("formats must be a sequence of format names")
    normalized = tuple(str(item).lower().lstrip(".") for item in formats)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("formats must be non-empty and unique")
    supported = {"svg", "png", "pdf", "tif", "tiff"}
    if any(item not in supported for item in normalized):
        raise ValueError(f"formats must be drawn from {sorted(supported)}")
    if "svg" not in normalized:
        raise ValueError("formats must include SVG as the editable primary output")
    if isinstance(dpi, bool) or not isinstance(dpi, int):
        raise TypeError("dpi must be an integer")
    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    if not isinstance(close, bool):
        raise TypeError("close must be a bool")

    apply_publication_style()
    base = Path(output_base)
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        fig.canvas.draw()
        for output_format in normalized:
            path = base.with_suffix(f".{output_format}")
            if output_format in {"png", "tif", "tiff"}:
                fig.savefig(path, bbox_inches="tight", dpi=dpi)
            else:
                fig.savefig(path, bbox_inches="tight")
            paths.append(path)
        return tuple(paths)
    finally:
        if close:
            plt.close(fig)


__all__ = [
    "AXIAL_CMAP",
    "SCALAR_CMAPS",
    "GateReduction",
    "plot_coherence_heatmap",
    "plot_frequency_heatmap",
    "plot_gate_heatmap",
    "plot_orientation_overlay",
    "plot_orientation_wheel",
    "plot_uncertainty_heatmap",
    "save_figure",
]
