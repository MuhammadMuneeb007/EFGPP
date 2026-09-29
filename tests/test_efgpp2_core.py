from pathlib import Path

import pandas as pd

from efgpp2.dataset import SampleTable, assert_disjoint_ids, merge_sample_features
from efgpp2.gwas import GwasQC, apply_qc, canonicalize_columns
from efgpp2.registry import Registry
from efgpp2.schema import Artifact, DataKind, VariantKey


def test_variant_key():
    key = VariantKey("chr1", 123, "a", "g", "GRCh38", "rs1")
    assert key.canonical_id == "GRCh38:1:123:A:G"


def test_registry_roundtrip(tmp_path: Path):
    reg = Registry(tmp_path / "registry.sqlite")
    artifact = Artifact(
        kind=DataKind.GWAS,
        path="/tmp/gwas.tsv",
        name="gwas1",
        phenotype="migraine",
        genome_build="GRCh37",
    )
    reg.register(artifact)
    loaded = reg.get(artifact.artifact_id)
    assert loaded is not None
    assert loaded.kind == DataKind.GWAS
    assert loaded.phenotype == "migraine"


def test_gwas_alias_and_qc():
    frame = pd.DataFrame(
        {
            "chrom": [1, 1, 1],
            "pos": [10, 11, 12],
            "rsid": ["rs1", "rs2", "rs2"],
            "effect_allele": ["A", "A", "A"],
            "other_allele": ["G", "T", "T"],
            "p_value": [0.01, 0.02, 0.02],
            "info": [0.95, 0.95, 0.95],
            "maf": [0.2, 0.2, 0.2],
        }
    )
    normalized, _ = canonicalize_columns(frame)
    assert {"CHR", "BP", "SNP", "A1", "A2", "P"}.issubset(normalized.columns)
    qc = apply_qc(frame, GwasQC(drop_ambiguous=True))
    assert qc["SNP"].tolist() == ["rs1"]


def test_sample_merge_and_leakage_guard():
    a = SampleTable("covariates", pd.DataFrame({"IID": [1, 2], "age": [30, 40]}))
    b = SampleTable("prs", pd.DataFrame({"IID": [1, 2], "score": [0.1, 0.2]}))
    merged = merge_sample_features([a, b])
    assert merged.shape == (2, 3)
    assert_disjoint_ids([1, 2], [3], [4, 5])
