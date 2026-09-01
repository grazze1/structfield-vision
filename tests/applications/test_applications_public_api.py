"""Installed-style public API smoke test for all three V0.3 applications."""

from __future__ import annotations

import torch

from structfield.applications import (
    APPLICATION_SCHEMA_VERSION,
    CurvilinearSegmenter,
    FringeRestorer,
    TextureRestorer,
)


def test_application_models_are_public_and_parameterized_modules() -> None:
    assert APPLICATION_SCHEMA_VERSION == "structfield-applications-v0.3"
    models = (
        FringeRestorer(num_orientations=8, width=4, adapter_mode="none"),
        TextureRestorer(num_orientations=8, width=4, adapter_mode="none"),
        CurvilinearSegmenter(
            num_orientations=8,
            width=4,
            adapter_mode="none",
        ),
    )
    assert all(isinstance(model, torch.nn.Module) for model in models)
    assert all(sum(parameter.numel() for parameter in model.parameters()) > 0 for model in models)
