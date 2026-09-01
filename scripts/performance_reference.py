#!/usr/bin/env python3
"""Record hardware-specific estimator latency and optional CUDA peak memory."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from structfield import AnalyticStructuralFieldEstimator
from structfield._version import VERSION


def measure(device: torch.device, orientations: int, size: int, repeats: int) -> dict[str, object]:
    estimator = AnalyticStructuralFieldEstimator(num_orientations=orientations).to(device).eval()
    image = torch.rand(1, 1, size, size, device=device, generator=torch.Generator(device=device).manual_seed(1))
    with torch.inference_mode():
        estimator(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            estimator(image)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "num_orientations": orientations,
        "image_shape": [1, 1, size, size],
        "latency_ms_median": statistics.median(samples),
        "latency_ms_samples": samples,
        "cuda_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    records = [measure(device, orientations, size, args.repeats) for orientations in (8, 16, 32) for size in (32, 64)]
    payload = {
        "schema_version": "structfield-performance-reference-v1",
        "software_version": VERSION,
        "disclaimer": "Hardware-specific reference; not a performance commitment.",
        "device": args.device,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
