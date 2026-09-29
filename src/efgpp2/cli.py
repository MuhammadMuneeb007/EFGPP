from __future__ import annotations

import argparse
import json

from .ingest import ingest_file
from .registry import Registry
from .schema import DataKind
from .storage import ProjectLayout


def _layout(project: str) -> ProjectLayout:
    return ProjectLayout.from_path(project)


def cmd_init(args) -> int:
    layout = _layout(args.project)
    layout.create()
    Registry(layout.registry_path)
    print(f"Initialized EFGPP2 project: {layout.root}")
    print(f"Registry: {layout.registry_path}")
    return 0


def cmd_register(args) -> int:
    layout = _layout(args.project)
    layout.create()
    registry = Registry(layout.registry_path)
    artifact = ingest_file(
        layout,
        registry,
        args.path,
        kind=DataKind(args.kind),
        mode=args.mode,
        phenotype=args.phenotype,
        genome_build=args.genome_build,
        ancestry=args.ancestry,
        source=args.source,
        checksum=args.checksum,
    )
    print(json.dumps(artifact.to_dict(), indent=2))
    return 0


def cmd_list(args) -> int:
    layout = _layout(args.project)
    registry = Registry(layout.registry_path)
    rows = registry.list(kind=args.kind, phenotype=args.phenotype)
    for artifact in rows:
        print(
            "\t".join(
                [
                    artifact.artifact_id,
                    artifact.kind.value,
                    artifact.phenotype or "-",
                    artifact.genome_build or "-",
                    artifact.name,
                    artifact.path,
                ]
            )
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="efgpp2")
    parser.add_argument("--project", default=".", help="EFGPP2 project root")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="Create project directories and registry")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("register", help="Register or ingest a data artifact")
    p.add_argument("kind", choices=[k.value for k in DataKind])
    p.add_argument("path")
    p.add_argument("--mode", choices=["reference", "copy"], default="reference")
    p.add_argument("--phenotype")
    p.add_argument("--genome-build")
    p.add_argument("--ancestry")
    p.add_argument("--source")
    p.add_argument("--checksum", action="store_true")
    p.set_defaults(func=cmd_register)

    p = sub.add_parser("list", help="List registered artifacts")
    p.add_argument("--kind", choices=[k.value for k in DataKind])
    p.add_argument("--phenotype")
    p.set_defaults(func=cmd_list)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
