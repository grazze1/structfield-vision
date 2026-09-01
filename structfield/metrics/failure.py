"""Stateless ranking metrics for caller-defined structural failures.

These reference metrics deliberately do not create failure labels internally.
The caller supplies an explicit binary ``failure_target`` and a score where a
larger value means more likely failure.  They are evaluation diagnostics, not
a claim that an uncertainty score is calibrated or a final failure model.
"""

from __future__ import annotations

import torch
from torch import nn

from .common import mask_invalid, validate_floating_bchw, validate_valid_mask, working_dtype

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _prepare_binary_inputs(
    scores: torch.Tensor,
    failure_target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_floating_bchw("scores", scores, channels=1)
    mask = validate_valid_mask(valid_mask, reference=scores)

    if not isinstance(failure_target, torch.Tensor):
        raise TypeError("failure_target must be a torch.Tensor")
    if failure_target.ndim != 4 or tuple(failure_target.shape) != tuple(scores.shape):
        raise ValueError("failure_target must have shape [B,1,H,W] aligned with scores")
    if failure_target.device != scores.device:
        raise ValueError("failure_target and scores must use the same device")
    supported_target_dtype = (
        failure_target.dtype is torch.bool
        or failure_target.dtype in _INTEGER_DTYPES
        or failure_target.dtype in _FLOAT_DTYPES
    )
    if not supported_target_dtype:
        raise TypeError(
            "failure_target must have bool, uint8, int8, int16, int32, int64, "
            "float16, bfloat16, float32, or float64 dtype"
        )

    # Mask first so excluded NaN/Inf/extreme sentinels cannot contaminate
    # validation, sorting, or a reduction.
    safe_scores = mask_invalid("scores", scores, mask)
    safe_target = torch.where(mask, failure_target, 0)
    if torch.is_floating_point(safe_target) and not bool(torch.isfinite(safe_target).all()):
        raise ValueError("failure_target contains NaN or Inf on a valid pixel")
    if bool(((safe_target != 0) & (safe_target != 1)).any()):
        raise ValueError("failure_target must contain only explicit 0/1 labels on valid pixels")

    dtype = working_dtype(scores)
    return safe_scores[mask].to(dtype=dtype), safe_target[mask].to(dtype=dtype)


def _ascending_group_counts(
    scores: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return positive/negative counts for each unique score, low to high."""
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    sorted_target = target[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_scores,
        return_inverse=True,
        return_counts=True,
    )
    positive_counts = scores.new_zeros(counts.shape)
    positive_counts.scatter_add_(0, inverse, sorted_target)
    negative_counts = counts.to(dtype=scores.dtype) - positive_counts
    return positive_counts, negative_counts


class FailureDetectionAUROC(nn.Module):
    """AUROC for explicit binary failure labels and higher-is-failure scores.

    The implementation counts positive/negative ranking pairs.  Equal-score
    pairs contribute exactly ``0.5``.  At least one positive and one negative
    valid label are required because AUROC is otherwise undefined.
    """

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(
        self,
        scores: torch.Tensor,
        failure_target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return scalar tie-aware AUROC after masking and flattening."""
        score_values, target_values = _prepare_binary_inputs(scores, failure_target, valid_mask)
        positive_total = target_values.sum()
        negative_total = target_values.new_tensor(target_values.numel()) - positive_total
        if not bool(positive_total > 0.0) or not bool(negative_total > 0.0):
            raise ValueError("AUROC requires at least one positive and one negative valid label")

        positives, negatives = _ascending_group_counts(score_values, target_values)
        negatives_below = torch.cumsum(negatives, dim=0) - negatives
        favorable_pairs = (positives * (negatives_below + 0.5 * negatives)).sum()
        return favorable_pairs / (positive_total * negative_total)


class FailureDetectionAUPRC(nn.Module):
    """Average precision for explicit binary failure labels.

    Tied scores are processed together at each unique descending threshold.
    The returned stepwise area uses the average-precision convention
    ``sum(precision_at_threshold * delta_recall)``.  At least one positive is
    required.  An all-positive valid set is mathematically one up to floating-
    point roundoff; an all-tied set returns the positive prevalence.
    """

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(
        self,
        scores: torch.Tensor,
        failure_target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return scalar grouped-threshold average precision."""
        score_values, target_values = _prepare_binary_inputs(scores, failure_target, valid_mask)
        positive_total = target_values.sum()
        if not bool(positive_total > 0.0):
            raise ValueError("AUPRC requires at least one positive valid label")

        ascending_positives, ascending_negatives = _ascending_group_counts(
            score_values,
            target_values,
        )
        positives = ascending_positives.flip(0)
        negatives = ascending_negatives.flip(0)
        cumulative_positives = torch.cumsum(positives, dim=0)
        cumulative_predictions = torch.cumsum(positives + negatives, dim=0)
        precision = cumulative_positives / cumulative_predictions
        delta_recall = positives / positive_total
        return (precision * delta_recall).sum()


__all__ = ["FailureDetectionAUPRC", "FailureDetectionAUROC"]
