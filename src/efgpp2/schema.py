from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import hashlib
import json
import uuid


class DataKind(str, Enum):
    GENOTYPE = "genotype"
    GWAS = "gwas"
    PHENOTYPE = "phenotype"
    COVARIATES = "covariates"
    ANNOTATION = "annotation"
    VARIANTS = "variants"
    PRS = "prs"
    OMICS = "omics"
    PCA = "pca"
    FEATURES = "features"
    MODEL = "model"
    RESULT = "result"


@dataclass(frozen=True)
class VariantKey:
    chrom: str
    pos: int
    ref: str
    alt: str
    build: str
    rsid: str | None = None

    @property
    def canonical_id(self) -> str:
        chrom = str(self.chrom).replace("chr", "", 1)
        return f"{self.build}:{chrom}:{int(self.pos)}:{self.ref.upper()}:{self.alt.upper()}"


@dataclass
class PhenotypeSpec:
    name: str
    task: str
    target_column: str
    sample_id_column: str = "IID"
    family_id_column: str | None = "FID"
    positive_label: str | int | float | None = 1
    covariates: list[str] = field(default_factory=list)
    genome_build: str | None = None
    ancestry: str | None = None
    description: str | None = None

    def validate(self) -> None:
        if self.task not in {"binary", "continuous", "multiclass", "survival"}:
            raise ValueError(f"Unsupported task: {self.task}")
        if not self.name:
            raise ValueError("Phenotype name must not be empty")
        if not self.target_column:
            raise ValueError("target_column must not be empty")


@dataclass
class Artifact:
    kind: DataKind
    path: str
    name: str
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phenotype: str | None = None
    format: str | None = None
    genome_build: str | None = None
    ancestry: str | None = None
    sample_id_column: str | None = None
    n_samples: int | None = None
    n_variants: int | None = None
    checksum: str | None = None
    source: str | None = None
    parent_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Artifact":
        payload = dict(d)
        payload["kind"] = DataKind(payload["kind"])
        return cls(**payload)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
