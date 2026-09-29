from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

from .variants import filter_ambiguous_snps


CANONICAL_ALIASES = {
    "CHR": {"CHR", "CHROM", "CHROMOSOME", "#CHROM"},
    "BP": {"BP", "POS", "POSITION", "BASE_PAIR_LOCATION"},
    "SNP": {"SNP", "RSID", "ID", "VARIANT_ID"},
    "A1": {"A1", "EA", "EFFECT_ALLELE", "ALT"},
    "A2": {"A2", "NEA", "OTHER_ALLELE", "REF"},
    "P": {"P", "PVALUE", "P_VALUE", "PVAL"},
    "BETA": {"BETA", "EFFECT", "EFFECT_SIZE"},
    "OR": {"OR", "ODDS_RATIO"},
    "SE": {"SE", "STANDARD_ERROR"},
    "N": {"N", "SAMPLE_SIZE", "TOTAL_N"},
    "INFO": {"INFO", "INFO_SCORE", "IMPUTATION_INFO"},
    "MAF": {"MAF", "EAF", "AF", "ALLELE_FREQUENCY"},
}


@dataclass
class GwasQC:
    min_maf: float | None = 0.01
    min_info: float | None = 0.8
    require_positive_p: bool = True
    drop_duplicate_snps: bool = True
    drop_ambiguous: bool = True
    notes: list[str] = field(default_factory=list)


def _norm(name: str) -> str:
    return str(name).strip().upper().replace("-", "_").replace(" ", "_")


def canonicalize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename only unambiguous aliases to EFGPP2 canonical GWAS names."""
    inverse: dict[str, list[str]] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        for alias in aliases:
            inverse.setdefault(_norm(alias), []).append(canonical)

    renames: dict[str, str] = {}
    claimed: set[str] = set()
    for raw in frame.columns:
        candidates = inverse.get(_norm(raw), [])
        if len(candidates) != 1:
            continue
        canonical = candidates[0]
        if canonical in claimed:
            continue
        renames[str(raw)] = canonical
        claimed.add(canonical)
    return frame.rename(columns=renames), renames


def read_gwas(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    compression = "gzip" if path.suffix == ".gz" else "infer"
    if "sep" not in kwargs:
        kwargs["sep"] = None
        kwargs["engine"] = "python"
    return pd.read_csv(path, compression=compression, **kwargs)


def apply_qc(frame: pd.DataFrame, qc: GwasQC | None = None) -> pd.DataFrame:
    qc = qc or GwasQC()
    out, _ = canonicalize_columns(frame)

    if qc.require_positive_p and "P" in out:
        out = out[pd.to_numeric(out["P"], errors="coerce") > 0]
    if qc.min_maf is not None and "MAF" in out:
        maf = pd.to_numeric(out["MAF"], errors="coerce")
        out = out[maf > qc.min_maf]
    if qc.min_info is not None and "INFO" in out:
        info = pd.to_numeric(out["INFO"], errors="coerce")
        out = out[info > qc.min_info]
    if qc.drop_duplicate_snps and "SNP" in out:
        out = out.drop_duplicates(subset=["SNP"])
    if qc.drop_ambiguous and {"A1", "A2"}.issubset(out.columns):
        out = filter_ambiguous_snps(out, "A1", "A2")
    return out.reset_index(drop=True)
