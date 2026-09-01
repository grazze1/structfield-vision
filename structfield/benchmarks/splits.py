"""Stable, disjoint seed domains for StructField-Bench-v0.1."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from .schema import BENCHMARK_VERSION, CANONICAL_FAMILIES, CANONICAL_SPLITS, BenchmarkFamily, BenchmarkSplit

SPLIT_DOMAIN_SIZE: Final = 1 << 50
MAX_SAMPLES_PER_FAMILY: Final = 1_000_000
FAMILY_COUNT: Final = len(CANONICAL_FAMILIES)
BASE_SEED_STRIDE: Final = FAMILY_COUNT * MAX_SAMPLES_PER_FAMILY
MAX_BASE_SEED: Final = SPLIT_DOMAIN_SIZE // BASE_SEED_STRIDE - 1


@dataclass(frozen=True, slots=True)
class SeedDomain:
    """Half-open integer seed interval assigned to one canonical split."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("a seed domain must satisfy 0 <= start < stop")

    def __contains__(self, seed: object) -> bool:
        return isinstance(seed, int) and not isinstance(seed, bool) and self.start <= seed < self.stop


SPLIT_SEED_DOMAINS = MappingProxyType(
    {
        "train": SeedDomain(0, SPLIT_DOMAIN_SIZE),
        "validation": SeedDomain(SPLIT_DOMAIN_SIZE, 2 * SPLIT_DOMAIN_SIZE),
        "test": SeedDomain(2 * SPLIT_DOMAIN_SIZE, 3 * SPLIT_DOMAIN_SIZE),
    },
)
"""Immutable split-to-domain mapping; intervals are pairwise disjoint."""


def _validate_split(split: str) -> BenchmarkSplit:
    if split not in CANONICAL_SPLITS:
        raise ValueError(f"split must be one of {CANONICAL_SPLITS}")
    return split  # type: ignore[return-value]


def _validate_family(family: str) -> BenchmarkFamily:
    if family not in CANONICAL_FAMILIES:
        raise ValueError(f"family must be one of {CANONICAL_FAMILIES}")
    return family  # type: ignore[return-value]


def _validate_integer(name: str, value: object, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must lie in [0, {maximum}]")
    return value


def get_seed_domain(split: BenchmarkSplit | str) -> SeedDomain:
    """Return the immutable seed domain assigned to ``split``."""
    validated_split = _validate_split(split)
    return SPLIT_SEED_DOMAINS[validated_split]


def derive_sample_seed(
    split: BenchmarkSplit | str,
    family: BenchmarkFamily | str,
    index: int,
    base_seed: int = 0,
) -> int:
    """Derive a stable unique seed without Python's randomized ``hash``.

    The hierarchy is explicit: split domain, base-seed block, canonical family
    offset, then sample index.  Consequently every accepted argument tuple maps
    injectively to a signed-64-bit-safe seed, and split domains cannot overlap.
    """
    validated_split = _validate_split(split)
    validated_family = _validate_family(family)
    validated_index = _validate_integer("index", index, maximum=MAX_SAMPLES_PER_FAMILY - 1)
    validated_base_seed = _validate_integer("base_seed", base_seed, maximum=MAX_BASE_SEED)

    family_offset = CANONICAL_FAMILIES.index(validated_family)
    local_seed = validated_base_seed * BASE_SEED_STRIDE + family_offset * MAX_SAMPLES_PER_FAMILY + validated_index
    seed = SPLIT_SEED_DOMAINS[validated_split].start + local_seed
    if seed not in SPLIT_SEED_DOMAINS[validated_split]:  # pragma: no cover - arithmetic invariant
        raise AssertionError("derived seed escaped its split domain")
    return seed


def make_sample_id(
    split: BenchmarkSplit | str,
    family: BenchmarkFamily | str,
    index: int,
    base_seed: int = 0,
    *,
    difficulty: str | None = None,
) -> str:
    """Return a stable human-readable identifier for a sample variant.

    ``difficulty`` is optional for callers that identify an undegraded base
    scene.  Generators whose difficulty setting changes tensors must pass it so
    distinct variants cannot share an identifier.
    """
    validated_split = _validate_split(split)
    validated_family = _validate_family(family)
    validated_index = _validate_integer("index", index, maximum=MAX_SAMPLES_PER_FAMILY - 1)
    validated_base_seed = _validate_integer("base_seed", base_seed, maximum=MAX_BASE_SEED)
    if difficulty is not None and (not isinstance(difficulty, str) or not difficulty.strip()):
        raise TypeError("difficulty must be None or a non-empty string")
    seed = derive_sample_seed(validated_split, validated_family, validated_index, validated_base_seed)
    identifier = (
        f"{BENCHMARK_VERSION}:{validated_split}:{validated_family}:b{validated_base_seed}:i{validated_index}:s{seed}"
    )
    return identifier if difficulty is None else f"{identifier}:d{difficulty}"
