"""Strict tensor contracts and masked reductions for field metrics.

The helpers in this module intentionally keep metric implementations
state-free and explicit.  A validity mask is geometric/evaluation metadata,
not a learnable weight, and every public metric requires one.
"""

from __future__ import annotations

import torch

_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def validate_floating_bchw(
    name: str,
    value: torch.Tensor,
    *,
    channels: int | None = None,
) -> torch.Tensor:
    """Validate a floating tensor with non-empty ``[B,C,H,W]`` shape.

    Finiteness is intentionally checked only after invalid pixels are removed
    by :func:`mask_invalid`.  A mask is allowed to exclude unavailable values,
    including NaN/Inf sentinels, without letting them contaminate a metric.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must use float16, bfloat16, float32, or float64 dtype")
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape [B,C,H,W]")
    if any(size < 1 for size in value.shape):
        raise ValueError(f"{name} dimensions must be non-empty")
    if channels is not None and value.shape[1] != channels:
        raise ValueError(f"{name} must have {channels} channel(s)")
    return value


def validate_aligned(
    reference_name: str,
    reference: torch.Tensor,
    other_name: str,
    other: torch.Tensor,
    *,
    same_channels: bool = True,
) -> None:
    """Require aligned batch/spatial layout, dtype, and device.

    Set ``same_channels=False`` for comparisons such as top-K predictions
    against a different number of target modes.
    """
    if reference.shape[0] != other.shape[0]:
        raise ValueError(f"{reference_name} and {other_name} must have the same batch size")
    if reference.shape[2:] != other.shape[2:]:
        raise ValueError(f"{reference_name} and {other_name} must have the same spatial shape")
    if same_channels and reference.shape[1] != other.shape[1]:
        raise ValueError(f"{reference_name} and {other_name} must have the same channel count")
    if reference.dtype != other.dtype:
        raise ValueError(f"{reference_name} and {other_name} must use the same dtype")
    if reference.device != other.device:
        raise ValueError(f"{reference_name} and {other_name} must use the same device")


def validate_valid_mask(
    valid_mask: torch.Tensor,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Validate an explicit boolean ``[B,1,H,W]`` mask with a valid pixel."""
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be a torch.Tensor")
    expected_shape = (reference.shape[0], 1, reference.shape[2], reference.shape[3])
    if valid_mask.ndim != 4 or tuple(valid_mask.shape) != expected_shape:
        raise ValueError("valid_mask must have shape [B,1,H,W] aligned with the inputs")
    if valid_mask.dtype is not torch.bool:
        raise TypeError("valid_mask must have bool dtype")
    if valid_mask.device != reference.device:
        raise ValueError("valid_mask and inputs must use the same device")
    if not bool(valid_mask.any()):
        raise ValueError("valid_mask must contain at least one valid pixel")
    return valid_mask


def working_dtype(value: torch.Tensor) -> torch.dtype:
    """Return the metric accumulation dtype for a validated input tensor."""
    if value.dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return value.dtype


def mask_invalid(
    name: str,
    value: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    fill: float = 0.0,
) -> torch.Tensor:
    """Remove invalid pixels before arithmetic and validate the remainder.

    The single-channel mask is broadcast over ``value`` channels.  This must
    be called before subtraction, squaring, logarithms, or other operations
    for which an invalid extreme input could overflow or create NaN.
    """
    safe_value = torch.where(valid_mask.expand_as(value), value, fill)
    if not bool(torch.isfinite(safe_value).all()):
        raise ValueError(f"{name} contains NaN or Inf on a valid pixel")
    return safe_value


def masked_mean(pixel_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Return a safe boolean-masked mean of scalar per-pixel values.

    ``pixel_values`` must be ``[B,1,H,W]`` and already use the desired
    accumulation dtype.  Invalid entries are removed with ``where`` before the
    reduction; multiplying an overflowed invalid value by zero would be unsafe.
    Callers must likewise exclude raw invalid inputs *before* potentially
    overflowing arithmetic that creates ``pixel_values``.
    """
    if not isinstance(pixel_values, torch.Tensor) or not torch.is_floating_point(pixel_values):
        raise TypeError("pixel_values must be a floating-point torch.Tensor")
    if pixel_values.ndim != 4 or pixel_values.shape[1] != 1:
        raise ValueError("pixel_values must have shape [B,1,H,W]")
    expected_shape = (
        pixel_values.shape[0],
        1,
        pixel_values.shape[2],
        pixel_values.shape[3],
    )
    if tuple(valid_mask.shape) != expected_shape:
        raise ValueError("valid_mask must be spatially aligned with pixel_values")
    if valid_mask.dtype is not torch.bool:
        raise TypeError("valid_mask must have bool dtype")
    if valid_mask.device != pixel_values.device:
        raise ValueError("valid_mask and pixel_values must use the same device")

    detached_mask = valid_mask.detach()
    safe_values = torch.where(detached_mask, pixel_values, 0.0)
    if not bool(torch.isfinite(safe_values).all()):
        raise ValueError("metric produced NaN or Inf on a valid pixel")
    return safe_values.sum() / detached_mask.sum().to(dtype=pixel_values.dtype)


__all__ = [
    "mask_invalid",
    "masked_mean",
    "validate_aligned",
    "validate_floating_bchw",
    "validate_valid_mask",
    "working_dtype",
]
