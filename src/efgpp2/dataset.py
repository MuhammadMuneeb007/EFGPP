from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Iterable
import pandas as pd


@dataclass
class SampleTable:
    name: str
    frame: pd.DataFrame
    sample_id: str = "IID"
    source_artifact_id: str | None = None


def assert_unique_samples(table: SampleTable) -> None:
    if table.sample_id not in table.frame.columns:
        raise KeyError(f"{table.name}: missing sample ID column {table.sample_id}")
    duplicated = table.frame[table.sample_id].duplicated()
    if duplicated.any():
        examples = table.frame.loc[duplicated, table.sample_id].head(5).tolist()
        raise ValueError(f"{table.name}: duplicate sample IDs, e.g. {examples}")


def merge_sample_features(
    tables: Iterable[SampleTable],
    *,
    how: str = "inner",
) -> pd.DataFrame:
    """Merge sample-level features by ID with collision protection."""
    tables = list(tables)
    if not tables:
        return pd.DataFrame()

    for table in tables:
        assert_unique_samples(table)

    canonical_id = tables[0].sample_id
    prepared: list[pd.DataFrame] = []
    used_features: set[str] = set()

    for table in tables:
        frame = table.frame.copy()
        if table.sample_id != canonical_id:
            frame = frame.rename(columns={table.sample_id: canonical_id})
        feature_cols = [c for c in frame.columns if c != canonical_id]
        collisions = used_features.intersection(feature_cols)
        if collisions:
            frame = frame.rename(
                columns={c: f"{table.name}__{c}" for c in collisions}
            )
            feature_cols = [c for c in frame.columns if c != canonical_id]
        used_features.update(feature_cols)
        prepared.append(frame)

    return reduce(
        lambda left, right: left.merge(
            right, on=canonical_id, how=how, validate="one_to_one"
        ),
        prepared,
    )


def assert_disjoint_ids(train_ids, validation_ids, test_ids) -> None:
    train = set(train_ids)
    val = set(validation_ids)
    test = set(test_ids)
    if train & val:
        raise ValueError("Leakage: train and validation sample IDs overlap")
    if train & test:
        raise ValueError("Leakage: train and test sample IDs overlap")
    if val & test:
        raise ValueError("Leakage: validation and test sample IDs overlap")
