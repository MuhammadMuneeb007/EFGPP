from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

from .schema import PhenotypeSpec


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    return data


def load_phenotype_spec(path: str | Path) -> tuple[PhenotypeSpec, dict[str, Any]]:
    data = load_yaml(path)
    spec = PhenotypeSpec(
        name=data["name"],
        task=data["task"],
        target_column=data["target_column"],
        sample_id_column=data.get("sample_id_column", "IID"),
        family_id_column=data.get("family_id_column", "FID"),
        positive_label=data.get("positive_label", 1),
        covariates=list(data.get("covariates", [])),
        genome_build=data.get("genome_build"),
        ancestry=data.get("ancestry"),
        description=data.get("description"),
    )
    spec.validate()
    return spec, data
