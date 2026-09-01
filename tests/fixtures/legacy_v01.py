"""Small deterministic inputs frozen for the V0.1-to-V0.2 parity audit.

The formulas in this file are project-authored fixture definitions. They do
not contain competition images, external dataset samples, or copied outputs.
"""

from __future__ import annotations

import hashlib
import math

import torch

LEGACY_STRUCTURE_CONFIG: dict[str, object] = {
    "weak_denoise_type": "gaussian",
    "weak_denoise_strength": 0.6,
    "local_radius": 4.0,
    "eps": 1e-3,
    "tensor_sigmas": [1.0, 2.0],
    "gabor_frequencies": [0.05, 0.10, 0.15],
    "gabor_orientations": 16,
    "hessian_sigmas": [1.0, 2.0],
}


def _grid(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-1.0, 1.0, size, dtype=torch.float32)
    return torch.meshgrid(axis, axis, indexing="ij")


def fixture_images(size: int = 64) -> dict[str, torch.Tensor]:
    """Return five finite CPU float32 fixtures with shape ``[1, 1, H, W]``."""
    if size < 32:
        raise ValueError("legacy parity fixtures require size >= 32")

    yy, xx = _grid(size)

    fringe_phase = 17.0 * (xx.square() - 0.65 * yy.square() + 0.25 * xx * yy) + 0.4
    fringe = (0.5 + 0.4 * torch.cos(fringe_phase)).clamp(0.0, 1.0)

    tangent_a = 0.37
    tangent_b = 1.24
    normal_a = tangent_a + math.pi / 2.0
    normal_b = tangent_b + math.pi / 2.0
    phase_a = 2.0 * math.pi * 5.5 * (math.cos(normal_a) * xx + math.sin(normal_a) * yy)
    phase_b = 2.0 * math.pi * 8.0 * (math.cos(normal_b) * xx + math.sin(normal_b) * yy)
    blend = torch.sigmoid(10.0 * (xx + 0.18 * torch.sin(3.0 * yy)))
    texture = (0.5 + 0.4 * ((1.0 - blend) * torch.cos(phase_a) + blend * torch.cos(phase_b))).clamp(0.0, 1.0)

    pixel_axis = torch.arange(size, dtype=torch.float32)
    pixel_yy, pixel_xx = torch.meshgrid(pixel_axis, pixel_axis, indexing="ij")
    crossing = (
        0.5 + 0.22 * torch.cos(2.0 * math.pi * 0.10 * pixel_xx) + 0.22 * torch.cos(2.0 * math.pi * 0.10 * pixel_yy)
    ).clamp(0.0, 1.0)

    flat = torch.full((size, size), 0.5, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(20260830)
    noise = (0.5 + 0.18 * torch.randn((size, size), generator=generator)).clamp(0.0, 1.0)

    return {
        "fringe": fringe[None, None],
        "texture": texture[None, None],
        "crossing": crossing[None, None],
        "flat": flat[None, None],
        "noise": noise[None, None],
    }


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash a contiguous CPU tensor including its dtype and shape contract."""
    tensor = value.detach().to(device="cpu").contiguous()
    header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode()
    return hashlib.sha256(header + tensor.numpy().tobytes(order="C")).hexdigest()


def fixture_hashes(size: int = 64) -> dict[str, str]:
    """Return content hashes for the deterministic fixture tensors."""
    return {name: tensor_sha256(image) for name, image in fixture_images(size).items()}
