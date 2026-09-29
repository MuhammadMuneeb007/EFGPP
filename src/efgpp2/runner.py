from __future__ import annotations

from pathlib import Path
import shlex
import subprocess

from .prs import CommandPlan
from .provenance import RunRecord, basic_environment


def render_plan(plan: CommandPlan) -> str:
    return "\n".join(shlex.join(command) for command in plan.commands)


def run_plan(
    plan: CommandPlan,
    *,
    cwd: str | Path | None = None,
    dry_run: bool = True,
    provenance_path: str | Path | None = None,
) -> RunRecord:
    record = RunRecord(
        operation=f"run:{plan.tool}",
        tool=plan.tool,
        parameters=plan.metadata,
        environment=basic_environment(),
    )
    try:
        for command in plan.commands:
            record.command = command
            if dry_run:
                continue
            subprocess.run(command, cwd=cwd, check=True)
        record.finish("dry_run" if dry_run else "success")
    except Exception as exc:
        record.finish("failed", error=str(exc))
        if provenance_path is not None:
            record.write_json(provenance_path)
        raise

    if provenance_path is not None:
        record.write_json(provenance_path)
    return record
