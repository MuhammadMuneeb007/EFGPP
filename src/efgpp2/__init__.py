"""EFGPP2: modular genotype-phenotype and multi-omics prediction."""

from .schema import Artifact, DataKind, PhenotypeSpec, VariantKey
from .registry import Registry
from .storage import ProjectLayout

__all__ = [
    "Artifact",
    "DataKind",
    "PhenotypeSpec",
    "VariantKey",
    "Registry",
    "ProjectLayout",
]

__version__ = "0.1.0"
