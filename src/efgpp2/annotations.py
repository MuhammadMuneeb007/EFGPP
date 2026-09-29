from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd

from .variants import add_variant_key


DEFAULT_KEY_COLUMNS = {
    "chrom": "Chr",
    "pos": "Start",
    "ref": "Ref",
    "alt": "Alt",
}


def merge_annotation_tables(
    paths: Iterable[str | Path],
    *,
    genome_build: str,
    key_columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Merge annotation files by genomic identity, never by row position.

    This replaces the fragile horizontal-concatenation pattern used in the
    CAGI workflow. Row counts can match even when records are ordered
    differently; EFGPP2 therefore joins on build/chromosome/position/ref/alt.
    """
    key_columns = key_columns or DEFAULT_KEY_COLUMNS
    merged: pd.DataFrame | None = None

    for path in paths:
        path = Path(path)
        frame = pd.read_csv(path)
        frame = add_variant_key(
            frame,
            chrom=key_columns["chrom"],
            pos=key_columns["pos"],
            ref=key_columns["ref"],
            alt=key_columns["alt"],
            build=genome_build,
        )
        frame = frame.drop_duplicates(subset=["variant_key"])
        if merged is None:
            merged = frame
            continue

        shared_nonkey = set(merged.columns).intersection(frame.columns) - {
            "variant_key",
            key_columns["chrom"],
            key_columns["pos"],
            key_columns["ref"],
            key_columns["alt"],
        }
        stem = path.stem.replace(".", "_")
        frame = frame.rename(
            columns={c: f"{stem}__{c}" for c in shared_nonkey}
        )
        merged = merged.merge(frame, on="variant_key", how="outer", suffixes=("", "__dup"))

    return merged if merged is not None else pd.DataFrame()
