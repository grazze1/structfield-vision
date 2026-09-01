"""Smoke test for the installed-style benchmark API boundary."""

from __future__ import annotations

from structfield.benchmarks import (
    BENCHMARK_VERSION,
    CANONICAL_DEGRADATIONS,
    CANONICAL_FAMILIES,
    BenchmarkSample,
    DegradationConfig,
    StructFieldBenchConfig,
    apply_degradation,
    generate_sample,
)


def test_public_api_generates_and_degrades_a_versioned_sample() -> None:
    sample = generate_sample(
        "crossing",
        split="validation",
        index=4,
        config=StructFieldBenchConfig(height=32, width=36),
    )
    degraded = apply_degradation(
        sample,
        DegradationConfig(name="gaussian", difficulty="easy"),
        seed=91,
    )
    assert isinstance(degraded, BenchmarkSample)
    assert degraded.metadata.benchmark_version == BENCHMARK_VERSION
    assert degraded.metadata.degradation == "gaussian"
    assert degraded.metadata.sample_id != sample.metadata.sample_id
    assert len(CANONICAL_FAMILIES) == 14
    assert len(CANONICAL_DEGRADATIONS) == 13  # twelve operators plus "none"
