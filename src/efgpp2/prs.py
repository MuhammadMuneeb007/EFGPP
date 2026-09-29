from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class PRSRunSpec:
    tool: str
    phenotype: str
    gwas_path: str
    genotype_prefix: str
    output_prefix: str
    fold: str | int | None = None
    genome_build: str | None = None
    pca_components: int = 10
    parameters: dict[str, object] = field(default_factory=dict)


@dataclass
class CommandPlan:
    tool: str
    commands: list[list[str]]
    expected_outputs: list[str]
    metadata: dict[str, object] = field(default_factory=dict)


def plink_ct_plan(
    spec: PRSRunSpec,
    *,
    plink: str = "plink",
    p_thresholds: Sequence[float] = (5e-8, 1e-5, 1e-3, 0.05, 0.1, 0.5, 1.0),
    clump_r2: float = 0.1,
    clump_kb: int = 200,
) -> CommandPlan:
    """Build, but do not execute, a PLINK clumping+thresholding PRS plan."""
    out = Path(spec.output_prefix)
    range_file = str(out) + ".ranges"
    snp_p_file = str(out) + ".snp_p.tsv"
    clump_prefix = str(out) + ".clump"

    commands = [
        [
            plink,
            "--bfile", spec.genotype_prefix,
            "--clump", spec.gwas_path,
            "--clump-snp-field", "SNP",
            "--clump-field", "P",
            "--clump-p1", "1",
            "--clump-r2", str(clump_r2),
            "--clump-kb", str(clump_kb),
            "--out", clump_prefix,
        ],
        [
            plink,
            "--bfile", spec.genotype_prefix,
            "--score", spec.gwas_path, "3", "4", "8", "header",
            "--q-score-range", range_file, snp_p_file,
            "--out", str(out),
        ],
    ]
    return CommandPlan(
        tool="plink_ct",
        commands=commands,
        expected_outputs=[range_file, snp_p_file, str(out) + ".*.profile"],
        metadata={
            "p_thresholds": list(p_thresholds),
            "clump_r2": clump_r2,
            "clump_kb": clump_kb,
        },
    )


def prsice2_plan(
    spec: PRSRunSpec,
    *,
    prsice_binary: str = "PRSice",
    prsice_r: str = "PRSice.R",
    lower: float = 1e-5,
    upper: float = 1.0,
    interval: float = 0.1,
) -> CommandPlan:
    """Build a PRSice-2 command using the PRSTools conventions."""
    out = str(Path(spec.output_prefix))
    command = [
        "Rscript", prsice_r,
        "--prsice", prsice_binary,
        "--base", spec.gwas_path,
        "--target", spec.genotype_prefix,
        "--snp", "SNP",
        "--A1", "A1",
        "--A2", "A2",
        "--pvalue", "P",
        "--lower", str(lower),
        "--upper", str(upper),
        "--interval", str(interval),
        "--all-score",
        "--out", out,
    ]
    return CommandPlan(
        tool="prsice2",
        commands=[command],
        expected_outputs=[out + ".all_score", out + ".summary"],
        metadata={"lower": lower, "upper": upper, "interval": interval},
    )
