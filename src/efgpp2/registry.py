from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .schema import Artifact, DataKind


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    phenotype TEXT,
    path TEXT NOT NULL,
    format TEXT,
    genome_build TEXT,
    ancestry TEXT,
    sample_id_column TEXT,
    n_samples INTEGER,
    n_variants INTEGER,
    checksum TEXT,
    source TEXT,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS lineage (
    parent_id TEXT NOT NULL,
    child_id TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id),
    FOREIGN KEY(parent_id) REFERENCES artifacts(artifact_id),
    FOREIGN KEY(child_id) REFERENCES artifacts(artifact_id)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind);
CREATE INDEX IF NOT EXISTS idx_artifacts_phenotype ON artifacts(phenotype);
"""


class Registry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA_SQL)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def register(self, artifact: Artifact) -> Artifact:
        with self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO artifacts (
                    artifact_id, name, kind, phenotype, path, format,
                    genome_build, ancestry, sample_id_column, n_samples,
                    n_variants, checksum, source, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.artifact_id,
                    artifact.name,
                    artifact.kind.value,
                    artifact.phenotype,
                    artifact.path,
                    artifact.format,
                    artifact.genome_build,
                    artifact.ancestry,
                    artifact.sample_id_column,
                    artifact.n_samples,
                    artifact.n_variants,
                    artifact.checksum,
                    artifact.source,
                    json.dumps(artifact.metadata, sort_keys=True, default=str),
                ),
            )
            for parent in artifact.parent_ids:
                con.execute(
                    "INSERT OR IGNORE INTO lineage(parent_id, child_id) VALUES (?, ?)",
                    (parent, artifact.artifact_id),
                )
        return artifact

    def get(self, artifact_id: str) -> Artifact | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
            if row is None:
                return None
            parents = [
                r["parent_id"]
                for r in con.execute(
                    "SELECT parent_id FROM lineage WHERE child_id = ?", (artifact_id,)
                )
            ]
        return self._row_to_artifact(row, parents)

    def list(
        self,
        *,
        kind: DataKind | str | None = None,
        phenotype: str | None = None,
    ) -> list[Artifact]:
        clauses: list[str] = []
        params: list[object] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind.value if isinstance(kind, DataKind) else str(kind))
        if phenotype is not None:
            clauses.append("phenotype = ?")
            params.append(phenotype)
        sql = "SELECT * FROM artifacts"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at, name"
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
            result: list[Artifact] = []
            for row in rows:
                parents = [
                    r["parent_id"]
                    for r in con.execute(
                        "SELECT parent_id FROM lineage WHERE child_id = ?",
                        (row["artifact_id"],),
                    )
                ]
                result.append(self._row_to_artifact(row, parents))
        return result

    @staticmethod
    def _row_to_artifact(row: sqlite3.Row, parents: Iterable[str]) -> Artifact:
        return Artifact(
            artifact_id=row["artifact_id"],
            name=row["name"],
            kind=DataKind(row["kind"]),
            phenotype=row["phenotype"],
            path=row["path"],
            format=row["format"],
            genome_build=row["genome_build"],
            ancestry=row["ancestry"],
            sample_id_column=row["sample_id_column"],
            n_samples=row["n_samples"],
            n_variants=row["n_variants"],
            checksum=row["checksum"],
            source=row["source"],
            parent_ids=list(parents),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )
