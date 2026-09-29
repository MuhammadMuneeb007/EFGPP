from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


AMBIGUOUS = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}


def normalize_chromosome(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    if text.upper() == "MT":
        text = "M"
    return text.upper()


def is_ambiguous_allele_pair(a1: object, a2: object) -> bool:
    return (str(a1).upper(), str(a2).upper()) in AMBIGUOUS


def add_variant_key(
    frame: pd.DataFrame,
    *,
    chrom: str,
    pos: str,
    ref: str,
    alt: str,
    build: str,
    output: str = "variant_key",
) -> pd.DataFrame:
    out = frame.copy()
    c = out[chrom].map(normalize_chromosome)
    p = pd.to_numeric(out[pos], errors="raise").astype("int64")
    r = out[ref].astype("string").str.upper()
    a = out[alt].astype("string").str.upper()
    out[output] = build + ":" + c.astype(str) + ":" + p.astype(str) + ":" + r + ":" + a
    return out


def filter_ambiguous_snps(
    frame: pd.DataFrame, a1: str = "A1", a2: str = "A2"
) -> pd.DataFrame:
    mask = [
        not is_ambiguous_allele_pair(x, y)
        for x, y in zip(frame[a1].tolist(), frame[a2].tolist())
    ]
    return frame.loc[mask].copy()
