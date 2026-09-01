"""Independent region loss for active V0.2 curvilinear segmentation."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F


def _logits_from_output(
    output: torch.Tensor | Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if not isinstance(output, Mapping):
        raise TypeError("output must be a logits tensor or a mapping")
    if "segmentation" not in output:
        raise KeyError("output mapping must contain 'segmentation'")
    logits = output["segmentation"]
    if not isinstance(logits, torch.Tensor):
        raise TypeError("output['segmentation'] must be a torch.Tensor")
    return logits


def _validate_logits_and_target(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if not torch.is_floating_point(logits):
        raise TypeError("segmentation logits must use a floating-point dtype")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("segmentation logits must have shape [B,1,H,W]")
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a torch.Tensor")
    if not torch.is_floating_point(target):
        raise TypeError("target must use a floating-point dtype")
    if target.shape != logits.shape:
        raise ValueError("target must have the same shape as segmentation logits")
    if target.device != logits.device or target.dtype != logits.dtype:
        raise ValueError("target must share segmentation logits dtype and device")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("segmentation logits contain NaN or Inf")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("target contains NaN or Inf")
    if float(target.detach().min()) < 0.0 or float(target.detach().max()) > 1.0:
        raise ValueError("target must lie in [0,1]")


def _working_pair(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_logits_and_target(logits, target)
    work_dtype = torch.float32 if logits.dtype in {torch.float16, torch.bfloat16} else logits.dtype
    return logits.to(dtype=work_dtype), target.to(dtype=work_dtype)


def _soft_dice_from_working_tensors(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dimensions = (1, 2, 3)
    intersection = (probability * target).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
    score = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - score.mean()


@torch.autocast(device_type="cuda", enabled=False)
@torch.autocast(device_type="cpu", enabled=False)
def soft_dice_loss(
    output: torch.Tensor | Mapping[str, torch.Tensor],
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return batch-mean soft Dice loss from unnormalized logits."""
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    logits = _logits_from_output(output)
    work_logits, work_target = _working_pair(logits, target)
    return _soft_dice_from_working_tensors(work_logits, work_target, eps=eps)


@torch.autocast(device_type="cuda", enabled=False)
@torch.autocast(device_type="cpu", enabled=False)
def curvilinear_segmentation_loss(
    output: torch.Tensor | Mapping[str, torch.Tensor],
    target: torch.Tensor,
    *,
    bce_weight: float = 0.5,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine soft Dice and BCE-with-logits without topology-derived code."""
    if not math.isfinite(bce_weight) or bce_weight < 0.0:
        raise ValueError("bce_weight must be finite and non-negative")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    logits = _logits_from_output(output)
    work_logits, work_target = _working_pair(logits, target)
    dice = _soft_dice_from_working_tensors(work_logits, work_target, eps=eps)
    bce = F.binary_cross_entropy_with_logits(work_logits, work_target)
    total = dice + bce_weight * bce
    return total, {
        "dice_loss": dice.detach(),
        "binary_cross_entropy": bce.detach(),
    }


__all__ = ["curvilinear_segmentation_loss", "soft_dice_loss"]
