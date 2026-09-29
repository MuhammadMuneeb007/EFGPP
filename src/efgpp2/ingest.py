from __future__ import annotations

from pathlib import Path
import shutil
from typing import Literal

from .registry import Registry
from .schema import Artifact, DataKind, sha256_file
from .storage import ProjectLayout


def detect_format(path: str | Path) -> str:
    p = Path(path)
    name = p.name.lower()
    for suffix, fmt in [
        (".pgen", "plink2"),
        (".bed", "plink1"),
        (".bgen", "bgen"),
        (".vcf.gz", "vcf"),
        (".vcf", "vcf"),
        (".parquet", "parquet"),
        (".csv.gz", "csv.gz"),
        (".tsv.gz", "tsv.gz"),
        (".gz", "gzip_table"),
        (".csv", "csv"),
        (".tsv", "tsv"),
        (".txt", "text"),
        (".zarr", "zarr"),
    ]:
        if name.endswith(suffix):
            return fmt
    return p.suffix.lstrip(".") or "unknown"


def validate_plink_prefix(prefix: str | Path) -> dict[str, str]:
    """Validate a PLINK1 or PLINK2 file set and return its component paths."""
    prefix = Path(prefix)
    p1 = {ext: prefix.with_suffix(ext) for ext in [".bed", ".bim", ".fam"]}
    p2 = {ext: prefix.with_suffix(ext) for ext in [".pgen", ".pvar", ".psam"]}

    if all(p.exists() for p in p1.values()):
        return {"format": "plink1", **{k[1:]: str(v) for k, v in p1.items()}}
    if all(p.exists() for p in p2.values()):
        return {"format": "plink2", **{k[1:]: str(v) for k, v in p2.items()}}
    missing1 = [str(p) for p in p1.values() if not p.exists()]
    missing2 = [str(p) for p in p2.values() if not p.exists()]
    raise FileNotFoundError(
        "Not a complete PLINK prefix. "
        f"PLINK1 missing: {missing1}; PLINK2 missing: {missing2}"
    )


def register_file(
    registry: Registry,
    path: str | Path,
    *,
    kind: DataKind,
    name: str | None = None,
    phenotype: str | None = None,
    genome_build: str | None = None,
    ancestry: str | None = None,
    source: str | None = None,
    checksum: bool = False,
    parent_ids: list[str] | None = None,
    metadata: dict | None = None,
) -> Artifact:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    artifact = Artifact(
        kind=kind,
        path=str(p),
        name=name or p.stem,
        phenotype=phenotype,
        format=detect_format(p),
        genome_build=genome_build,
        ancestry=ancestry,
        checksum=sha256_file(p) if checksum and p.is_file() else None,
        source=source,
        parent_ids=parent_ids or [],
        metadata=metadata or {},
    )
    return registry.register(artifact)


def ingest_file(
    layout: ProjectLayout,
    registry: Registry,
    path: str | Path,
    *,
    kind: DataKind,
    mode: Literal["reference", "copy"] = "reference",
    phenotype: str | None = None,
    genome_build: str | None = None,
    ancestry: str | None = None,
    source: str | None = None,
    checksum: bool = False,
    parent_ids: list[str] | None = None,
    metadata: dict | None = None,
) -> Artifact:
    """Register data without duplicating it by default.

    Reference mode is preferred for UK Biobank/HPC-scale assets.
    Copy mode is useful for small public GWAS/covariate/annotation files.
    """
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    final_path = source_path
    if mode == "copy":
        target_dir = layout.raw / kind.value / (phenotype or "shared")
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = target_dir / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, final_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, final_path)

    return register_file(
        registry,
        final_path,
        kind=kind,
        phenotype=phenotype,
        genome_build=genome_build,
        ancestry=ancestry,
        source=source,
        checksum=checksum,
        parent_ids=parent_ids,
        metadata={"ingest_mode": mode, **(metadata or {})},
    )
