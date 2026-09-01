"""Public API for StructField-Bench-v0.1."""

from .degradations import DIFFICULTIES, DegradationConfig, apply_degradation
from .generator import StructFieldBenchConfig, StructFieldBenchGenerator, generate_sample
from .schema import (
    BENCHMARK_VERSION,
    CANONICAL_DEGRADATIONS,
    CANONICAL_FAMILIES,
    CANONICAL_SPLITS,
    GENERATOR_VERSION,
    BenchmarkDegradation,
    BenchmarkFamily,
    BenchmarkMetadata,
    BenchmarkSample,
    BenchmarkSplit,
)
from .splits import (
    MAX_BASE_SEED,
    MAX_SAMPLES_PER_FAMILY,
    SPLIT_SEED_DOMAINS,
    SeedDomain,
    derive_sample_seed,
    get_seed_domain,
    make_sample_id,
)

__all__ = [
    "BENCHMARK_VERSION",
    "CANONICAL_DEGRADATIONS",
    "CANONICAL_FAMILIES",
    "CANONICAL_SPLITS",
    "DIFFICULTIES",
    "GENERATOR_VERSION",
    "MAX_BASE_SEED",
    "MAX_SAMPLES_PER_FAMILY",
    "SPLIT_SEED_DOMAINS",
    "BenchmarkDegradation",
    "BenchmarkFamily",
    "BenchmarkMetadata",
    "BenchmarkSample",
    "BenchmarkSplit",
    "DegradationConfig",
    "SeedDomain",
    "StructFieldBenchConfig",
    "StructFieldBenchGenerator",
    "apply_degradation",
    "derive_sample_seed",
    "generate_sample",
    "get_seed_domain",
    "make_sample_id",
]
