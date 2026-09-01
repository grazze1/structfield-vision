"""Task-neutral adapters for injecting structural features into backbones."""

from .base import AdapterOutput, StructuralAdapter
from .concat import ConcatStructuralAdapter
from .film import FiLMStructuralAdapter
from .no_prior import NoPriorAdapter
from .reliability_gated import ReliabilityGatedAdapter

__all__ = [
    "AdapterOutput",
    "ConcatStructuralAdapter",
    "FiLMStructuralAdapter",
    "NoPriorAdapter",
    "ReliabilityGatedAdapter",
    "StructuralAdapter",
]
