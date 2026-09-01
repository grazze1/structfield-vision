"""Strict/trusted boundary and full-graph validation tests."""

from __future__ import annotations

import pytest
import torch

from structfield import ValidationMode
from structfield.adapters import ConcatStructuralAdapter, ReliabilityGatedAdapter
from structfield.applications import CurvilinearSegmenter


def test_validation_mode_normalization_is_public_and_strict_by_default() -> None:
    assert ValidationMode.normalize("strict") is ValidationMode.strict
    assert ValidationMode.normalize(ValidationMode.trusted) is ValidationMode.trusted
    with pytest.raises(ValueError):
        ValidationMode.normalize("disabled")
    with pytest.raises(TypeError):
        ValidationMode.normalize(True)  # type: ignore[arg-type]


def test_trusted_adapter_skips_values_but_keeps_structural_checks() -> None:
    features = torch.zeros(1, 2, 4, 4)
    structural = torch.zeros(1, 3, 4, 4)
    structural[0, 0, 0, 0] = torch.nan
    adapter = ConcatStructuralAdapter()

    with pytest.raises(ValueError, match="NaN or Inf"):
        adapter(features, structural)
    output = adapter(features, structural, validation_mode=ValidationMode.trusted)
    assert output.features.shape == (1, 5, 4, 4)
    with pytest.raises(ValueError, match="spatial"):
        adapter(
            features,
            torch.zeros(1, 3, 3, 4),
            validation_mode=ValidationMode.trusted,
        )


def test_trusted_application_propagates_mode_and_strict_still_rejects_nan() -> None:
    image = torch.zeros(1, 1, 8, 8)
    image[0, 0, 0, 0] = torch.nan
    strict_model = CurvilinearSegmenter(width=4, adapter_mode="none")
    trusted_model = CurvilinearSegmenter(
        width=4,
        adapter_mode="none",
        validation_mode=ValidationMode.trusted,
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        strict_model(image)
    assert trusted_model(image)["segmentation"].shape == image.shape


def test_trusted_fuse_is_fullgraph_and_matches_strict_eager() -> None:
    features = torch.randn(1, 3, 4, 4)
    delta = torch.randn(1, 3, 4, 4)
    gate = torch.sigmoid(torch.randn(1, 3, 4, 4))
    strict = ReliabilityGatedAdapter.fuse(
        features,
        delta,
        gate,
        validation_mode=ValidationMode.strict,
    )

    def trusted_fuse(
        current: torch.Tensor,
        residual: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        return ReliabilityGatedAdapter.fuse(
            current,
            residual,
            reliability,
            validation_mode=ValidationMode.trusted,
        )

    compiled = torch.compile(trusted_fuse, backend="eager", fullgraph=True)
    actual = compiled(features, delta, gate)
    torch.testing.assert_close(actual, strict)


def test_trusted_application_forward_is_fullgraph_and_numerically_stable() -> None:
    torch.manual_seed(7)
    model = CurvilinearSegmenter(
        width=4,
        adapter_mode="none",
        validation_mode=ValidationMode.trusted,
    ).eval()
    image = torch.rand(1, 1, 8, 8)
    eager = model(image)["segmentation"]
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    actual = compiled(image)["segmentation"]
    torch.testing.assert_close(actual, eager)
