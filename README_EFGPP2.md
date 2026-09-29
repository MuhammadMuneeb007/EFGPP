# EFGPP2

EFGPP2 is the next version of the Exploratory Framework for Genotype-Phenotype Prediction. It keeps the scientific goals of EFGPP but replaces phenotype-specific research scripts with a reusable data registry, typed artifacts, canonical variant identifiers, modular feature generation, PRS adapters, provenance, and multi-omics-ready storage.

## What EFGPP2 is for

EFGPP2 is designed to support one or many phenotypes using the same pipeline:

1. Register genotype data.
2. Register one or more GWAS summary-statistics datasets.
3. Register phenotype labels and covariates.
4. Register functional annotations and variant-level predictors.
5. Generate PCA, PRS, genotype subsets, annotation-weighted genotype features, and other omics features.
6. Assemble sample-level model matrices by sample ID.
7. Train/evaluate models using leakage-safe folds.
8. Trace every result to the exact source artifacts and parameters.

## Storage strategy

Do not put large genotype or UK Biobank files in Git.

- Genotype: PLINK2 PGEN/PVAR/PSAM, PLINK1 BED/BIM/FAM, or BGEN.
- GWAS: keep the original file; optionally materialize harmonized Parquet.
- Variant annotations: Parquet, ideally partitioned by chromosome.
- Phenotypes/covariates/PRS/PCA: Parquet or TSV for small files.
- Large omics matrices: Zarr or Parquet depending on shape.
- Registry and lineage: SQLite under .efgpp2/registry.sqlite.
- Models/reports: local project directories, not Git by default.

## Quick start

Install in development mode:

    pip install -e ".[dev]"

Initialize a project:

    efgpp2 --project /data/my_efgpp2 init

Register a GWAS without copying it:

    efgpp2 --project /data/my_efgpp2 register gwas /data/gwas/migraine.tsv.gz --phenotype migraine --genome-build GRCh37 --source GWAS-Catalog

Register covariates:

    efgpp2 --project /data/my_efgpp2 register covariates /data/ukb/migraine_covariates.tsv --phenotype migraine

List registered migraine artifacts:

    efgpp2 --project /data/my_efgpp2 list --phenotype migraine

See configs/phenotypes/migraine.example.yaml for a phenotype configuration migrated from the original EFGPP design.

## Core improvements over EFGPP

- Phenotypes are configuration, not hard-coded Python lists.
- Every dataset is a typed artifact with lineage.
- Genome build and ancestry are explicit metadata.
- Variants use canonical build:chrom:pos:ref:alt identities.
- Annotation tables are merged by genomic keys, never by row position.
- Sample-level features are merged by sample ID with one-to-one validation.
- Train/validation/test sample overlap can be checked centrally.
- GWAS columns are canonicalized through aliases rather than destructive global text rewriting.
- PRS commands are represented as reusable plans.
- External-tool execution can emit provenance records.
- Existing EFGPP directories can be discovered and registered without copying the underlying large files.

## Relationship to the other repositories

EFGPP2 integrates design ideas from:

- EFGPP: genotype QC, GWAS-driven SNP selection, PCA, PRS, functional annotations, dataset combinations, ML/DL evaluation.
- GWASPokerforPRS2: normalized internal schemas, cautious canonicalization, provenance, reproducibility, PRS readiness concepts.
- CAGI7_Annotate_All_Missense: broad functional annotation sources, chromosome-wise processing, scalable annotation/feature engineering.
- PRSTools: PRS tool wrappers, clumping/thresholding concepts, PCA/covariate integration, binary/continuous trait handling.

The goal is not to copy those repositories into one folder. EFGPP2 exposes their useful concepts through stable interfaces so additional tools can be plugged in later.
