from __future__ import annotations

from pathlib import Path

from .ingest import register_file
from .registry import Registry
from .schema import DataKind


def discover_legacy_phenotype(root: str | Path, phenotype: str) -> dict[str, list[Path]]:
    """Discover assets from the original EFGPP phenotype folder layout."""
    base = Path(root) / phenotype
    discovered: dict[str, list[Path]] = {
        "gwas": [],
        "genotype_prefixes": [],
        "covariates": [],
        "phenotypes": [],
    }
    if not base.exists():
        return discovered

    discovered["gwas"] = sorted(base.glob("*.gz"))

    for bed in sorted(base.rglob("*.bed")):
        prefix = bed.with_suffix("")
        if prefix.with_suffix(".bim").exists() and prefix.with_suffix(".fam").exists():
            discovered["genotype_prefixes"].append(prefix)

    discovered["covariates"] = sorted(base.rglob("*.cov"))
    discovered["phenotypes"] = sorted(base.rglob("*.height"))
    return discovered


def register_legacy_phenotype(
    registry: Registry,
    root: str | Path,
    phenotype: str,
    *,
    genome_build: str | None = None,
    ancestry: str | None = None,
) -> dict[str, list[str]]:
    discovered = discover_legacy_phenotype(root, phenotype)
    registered: dict[str, list[str]] = {k: [] for k in discovered}

    for path in discovered["gwas"]:
        a = register_file(
            registry,
            path,
            kind=DataKind.GWAS,
            phenotype=phenotype,
            genome_build=genome_build,
            ancestry=ancestry,
            source="legacy_efgpp",
        )
        registered["gwas"].append(a.artifact_id)

    for prefix in discovered["genotype_prefixes"]:
        bed = prefix.with_suffix(".bed")
        a = register_file(
            registry,
            bed,
            kind=DataKind.GENOTYPE,
            name=prefix.name,
            phenotype=phenotype,
            genome_build=genome_build,
            ancestry=ancestry,
            source="legacy_efgpp",
            metadata={
                "prefix": str(prefix),
                "bim": str(prefix.with_suffix(".bim")),
                "fam": str(prefix.with_suffix(".fam")),
            },
        )
        registered["genotype_prefixes"].append(a.artifact_id)

    for path in discovered["covariates"]:
        a = register_file(
            registry,
            path,
            kind=DataKind.COVARIATES,
            phenotype=phenotype,
            source="legacy_efgpp",
        )
        registered["covariates"].append(a.artifact_id)

    for path in discovered["phenotypes"]:
        a = register_file(
            registry,
            path,
            kind=DataKind.PHENOTYPE,
            phenotype=phenotype,
            source="legacy_efgpp",
        )
        registered["phenotypes"].append(a.artifact_id)

    return registered
