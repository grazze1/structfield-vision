#!/usr/bin/env python3
"""Generate or check the deterministic v1 benchmark regression baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from structfield import AnalyticStructuralFieldEstimator
from structfield._version import VERSION
from structfield.benchmarks import BENCHMARK_VERSION, CANONICAL_FAMILIES, StructFieldBenchConfig, generate_sample
from structfield.metrics import FrequencyMAE, OrientationDistributionJS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "benchmarks" / "v1.0.0_baseline.json"


def generate() -> dict[str, object]:
    config = StructFieldBenchConfig(height=32, width=32, num_orientations=16, base_seed=100)
    estimator = AnalyticStructuralFieldEstimator(num_orientations=16).eval()
    js_metric = OrientationDistributionJS()
    frequency_metric = FrequencyMAE()
    records: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for family in CANONICAL_FAMILIES:
            sample = generate_sample(family, split="test", index=0, difficulty="medium", config=config)
            field = estimator(sample.image.unsqueeze(0))
            valid = sample.valid_mask.unsqueeze(0)
            count = sample.orientation_count.clamp_min(1).to(sample.dtype)
            target_frequency = sample.component_frequency_gt.sum(0, keepdim=True) / count
            records[family] = {
                "orientation_js": float(
                    js_metric(field.orientation_prob, sample.orientation_distribution_gt.unsqueeze(0), valid)
                ),
                "frequency_mae_cpp": float(
                    frequency_metric(field.frequency_mean, target_frequency.unsqueeze(0), valid)
                ),
                "mean_coherence": float(field.coherence[valid].mean()),
            }
    return {
        "schema_version": "structfield-release-benchmark-v1",
        "software_version": VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "profile": {
            "device": "cpu",
            "height": config.height,
            "width": config.width,
            "num_orientations": config.num_orientations,
            "split": "test",
            "index": 0,
            "base_seed": config.base_seed,
        },
        "tolerances": {
            "orientation_js": 2e-6,
            "frequency_mae_cpp": 2e-6,
            "mean_coherence": 2e-6,
        },
        "families": records,
    }


def check(actual: dict[str, object], expected: dict[str, object]) -> None:
    for key in ("schema_version", "software_version", "benchmark_version", "profile"):
        if actual[key] != expected[key]:
            raise RuntimeError(f"benchmark baseline identity mismatch: {key}")
    tolerances = expected["tolerances"]
    expected_families = expected["families"]
    actual_families = actual["families"]
    if (
        not isinstance(tolerances, dict)
        or not isinstance(expected_families, dict)
        or not isinstance(actual_families, dict)
    ):
        raise RuntimeError("invalid benchmark baseline structure")
    if set(actual_families) != set(expected_families):
        raise RuntimeError("benchmark family set changed")
    for family, expected_metrics in expected_families.items():
        actual_metrics = actual_families[family]
        if not isinstance(expected_metrics, dict) or not isinstance(actual_metrics, dict):
            raise RuntimeError("invalid family metric structure")
        for metric, expected_value in expected_metrics.items():
            actual_value = actual_metrics[metric]
            tolerance = tolerances[metric]
            if not math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=tolerance):
                raise RuntimeError(
                    f"benchmark regression for {family}.{metric}: {actual_value} != {expected_value} ± {tolerance}"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    actual = generate()
    if args.check:
        expected = json.loads(args.output.read_text(encoding="utf-8"))
        check(actual, expected)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
