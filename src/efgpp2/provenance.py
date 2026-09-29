from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from typing import Any
import uuid


@dataclass
class RunRecord:
    operation: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: str | None = None
    status: str = "running"
    command: list[str] | None = None
    tool: str | None = None
    tool_version: str | None = None
    seed: int | None = None
    input_artifact_ids: list[str] = field(default_factory=list)
    output_artifact_ids: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def finish(self, status: str = "success", error: str | None = None) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.status = status
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str))


def basic_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }


def get_tool_version(executable: str, args: list[str] | None = None) -> str | None:
    args = args or ["--version"]
    try:
        result = subprocess.run(
            [executable, *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except Exception:
        return None
    text = (result.stdout or result.stderr or "").strip()
    return text.splitlines()[0] if text else None
