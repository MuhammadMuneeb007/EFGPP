from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectLayout:
    """Filesystem layout for EFGPP2.

    Large scientific files are referenced here; they are not intended to be
    committed to Git. Metadata and lineage live in the registry.
    """

    root: Path

    @classmethod
    def from_path(cls, root: str | Path) -> "ProjectLayout":
        return cls(Path(root).expanduser().resolve())

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def features(self) -> Path:
        return self.root / "data" / "features"

    @property
    def registry_dir(self) -> Path:
        return self.root / ".efgpp2"

    @property
    def registry_path(self) -> Path:
        return self.registry_dir / "registry.sqlite"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def models(self) -> Path:
        return self.root / "models"

    def create(self) -> None:
        for path in [
            self.raw,
            self.interim,
            self.processed,
            self.features,
            self.registry_dir,
            self.reports,
            self.models,
        ]:
            path.mkdir(parents=True, exist_ok=True)
