"""Contracts for the small V0.1 parity inputs frozen during Stage 1."""

from __future__ import annotations

import torch

from tests.fixtures.legacy_v01 import fixture_hashes, fixture_images


def test_legacy_parity_fixtures_are_deterministic() -> None:
    first = fixture_images()
    second = fixture_images()
    assert tuple(first) == ("fringe", "texture", "crossing", "flat", "noise")
    assert fixture_hashes() == fixture_hashes()
    for name in first:
        assert torch.equal(first[name], second[name]), name


def test_legacy_parity_fixture_contract() -> None:
    fixtures = fixture_images()
    assert len(set(fixture_hashes().values())) == len(fixtures)
    for image in fixtures.values():
        assert image.shape == (1, 1, 64, 64)
        assert image.dtype == torch.float32
        assert image.device.type == "cpu"
        assert torch.isfinite(image).all()
        assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0


def test_legacy_parity_fixture_rejects_tiny_images() -> None:
    try:
        fixture_images(16)
    except ValueError as error:
        assert "size >= 32" in str(error)
    else:
        raise AssertionError("fixture_images must reject undersized parity inputs")
