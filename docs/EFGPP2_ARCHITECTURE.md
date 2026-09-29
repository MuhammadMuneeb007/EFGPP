# EFGPP2 architecture

## 1. Why a second architecture is needed

The original EFGPP proves the central scientific idea: genotype-phenotype prediction can improve when genotype-derived features are combined with GWAS information, annotations, PRS, PCA and covariates. Its current implementation is optimized for a specific experimental analysis: sequential Step*.py scripts, phenotype/GWAS pairs embedded in code, filenames carrying scientific state, and generated feature matrices stored as independent files.

That is useful for a paper workflow but becomes difficult to extend to tens of phenotypes, multiple genome builds, different ancestries, several annotation releases, repeated PRS methods and additional omics.

EFGPP2 separates five things that were previously mixed together:

1. Data identity.
2. Data storage.
3. Scientific transformations.
4. Experiment configuration.
5. Model evaluation.

## 2. Source repositories reviewed

### EFGPP

Useful concepts retained:

- GWAS and target-genotype QC.
- Match GWAS variants to target genotype data.
- P-value-driven SNP subsets.
- Functional annotation integration.
- PCA.
- PLINK, PRSice-2, AnnoPred and LDAK-derived PRS.
- Base dataset generation.
- Dataset combination and model comparison.
- Fold-level evaluation and aggregation.

Main changes:

- Replace phenotype_gwas_pairs embedded in Python with YAML.
- Replace directory/filename inference with registry metadata.
- Replace implicit dataset IDs with artifact IDs and lineage.
- Replace ad-hoc feature merging with key-validated joins.
- Record genome build, ancestry, fold and provenance for every derived output.

### GWASPokerforPRS2

Useful concepts retained:

- Normalized internal data model.
- Canonical column aliases.
- Never invent missing values.
- Avoid destructive normalization.
- Record transformations and provenance.
- PRS-readiness as a validation stage rather than an assumption.

EFGPP2 applies these concepts to all genotype-phenotype artifacts, not just public GWAS discovery.

### CAGI7_Annotate_All_Missense

Useful concepts retained:

- Broad annotation catalogue.
- ANNOVAR-based workflows.
- Per-chromosome processing.
- Polars/pandas-compatible large-table processing.
- Feature groups for pathogenicity and protein-function prediction.

Important correction:

The existing chromosome annotation merge can concatenate files horizontally after checking row counts. Equal row counts do not prove that variants occur in the same order. EFGPP2 joins annotation data using genome build + chromosome + position + reference + alternate allele.

### PRSTools

Useful concepts retained:

- Multiple PRS methods behind a common conceptual workflow.
- Clumping/pruning/threshold parameters.
- Binary vs continuous phenotype handling.
- PCA/covariate integration.
- Tool-specific environments.

EFGPP2 turns each PRS method into an adapter/command plan so a phenotype configuration can request a method without copying a complete script.

## 3. Data layers

### Layer A: immutable/raw

Original scientific assets should be kept unchanged.

Examples:

- UK Biobank BGEN/PGEN/BED data.
- Original GWAS summary statistics.
- Original phenotype exports.
- Covariate source tables.
- Annotation releases.
- Transcriptomics, methylation, proteomics or metabolomics matrices.

Raw files can live on HPC storage, object storage or local disks. EFGPP2 can reference them without copying.

### Layer B: normalized/interim

This layer contains scientifically equivalent but standardized data:

- Canonical GWAS column names.
- Harmonized genome build.
- Normalized chromosome names.
- Variant keys.
- Sample ID mapping.
- QC-filtered genotype sets.
- Harmonized annotation tables.

Every transformation should identify its parents.

### Layer C: model-ready features

Examples:

- Top-N genotype variants.
- P-value-threshold genotype features.
- Annotation-weighted genotype features.
- PRS from one GWAS/method/parameter set.
- PCA.
- Covariate subsets.
- Gene burden features.
- Expression/methylation/proteomic features.
- Cross-phenotype PRS.
- Combined feature sets.

Each feature artifact must carry sample IDs.

### Layer D: experiments/results

Contains:

- Split definitions.
- Model configuration.
- Metrics.
- Predictions.
- Feature importance.
- Calibration.
- Provenance.

## 4. Recommended physical formats

| Data type | Preferred format | Why |
| --- | --- | --- |
| Individual genotype | PLINK2 PGEN/PVAR/PSAM or BGEN | Compact, standard genomic tooling |
| GWAS raw | Original compressed text | Preserve source exactly |
| GWAS normalized | Parquet | Typed columns, fast scans |
| Variant annotations | Parquet partitioned by chromosome | Fast filtering/joining |
| Phenotype | Parquet/TSV | Sample-level table |
| Covariates | Parquet/TSV | Sample-level table |
| PRS/PCA | Parquet | Sample-level derived features |
| Dense omics | Zarr | Chunked n-dimensional arrays |
| Sparse/long omics | Parquet | Fast analytical query |
| Artifact metadata | SQLite initially | Portable local registry |
| Very large shared registry | PostgreSQL later | Concurrent multi-user deployment |

Do not try to store the genotype matrix itself in SQLite/PostgreSQL.

## 5. Canonical identities

### Sample identity

Use a stable sample ID field such as IID. Keep FID separately if required by PLINK. Never rely on dataframe row position to align samples.

Every sample-level artifact should contain:

- sample_id
- phenotype
- cohort
- ancestry if available
- source artifact ID
- fold/split when derived for a specific experiment

### Variant identity

The primary interoperable key is:

    genome_build:chromosome:position:reference:alternate

Example:

    GRCh38:10:123456:A:G

RSID is retained as an attribute, not treated as a globally stable primary key.

This is important because:

- RSIDs can be missing.
- RSIDs can merge/change.
- chromosome-position alone is insufficient for multiallelic sites.
- the same coordinate means different things across genome builds.

## 6. Registry

The SQLite registry stores metadata, not large scientific matrices.

Artifact fields include:

- artifact_id
- kind
- name
- path
- phenotype
- format
- genome_build
- ancestry
- sample_id_column
- n_samples
- n_variants
- checksum
- source
- metadata JSON
- parent artifact IDs

This enables lineage such as:

    raw GWAS
      -> normalized GWAS
      -> QC GWAS
      -> PRS weights
      -> sample PRS
      -> combined feature matrix
      -> model
      -> prediction/metrics

## 7. Phenotype configuration

A phenotype YAML should define:

- task type: binary/continuous/multiclass/survival
- target column
- positive class if relevant
- genotype source
- phenotype source
- covariates
- one or more GWAS sources
- genome build
- ancestry
- feature recipes
- PRS methods
- PCA count
- cross-trait data
- fold strategy
- random seed

This replaces editing scripts for migraine vs depression.

## 8. Pipeline stages

### Stage 1: register

Register all source assets. For large files, use reference mode.

### Stage 2: validate

Validate:

- files exist
- PLINK component sets are complete
- sample IDs are unique
- expected phenotype column exists
- genome build is known where required
- GWAS has enough fields for requested PRS method
- alleles and coordinates are parseable

### Stage 3: harmonize

- canonicalize GWAS columns
- normalize chromosome naming
- assign canonical variant keys
- lift genome build only as an explicit transformation
- align alleles between GWAS and genotype
- handle ambiguous A/T and C/G sites according to declared policy

### Stage 4: QC

Genotype QC should be implemented through PLINK/PLINK2 adapters.

GWAS QC can include:

- MAF threshold
- INFO threshold
- valid positive P values
- duplicate handling
- allele validation
- ambiguous allele policy

Thresholds belong in configuration and provenance.

### Stage 5: generate features

Feature generators should return registered artifacts.

Initial generators:

- genotype Top-N
- genotype P-value thresholds
- annotation-enhanced genotype
- PCA
- covariates
- PLINK clumping+thresholding PRS
- PRSice-2
- AnnoPred
- LDAK

Later adapters:

- LDpred2
- PRS-CS / PRS-CSx
- SBayesR
- lassosum2
- VIPRS
- SCT
- other PRSTools methods

### Stage 6: multi-omics

Treat every omics layer as a typed feature source.

Examples:

- transcriptomics: gene expression per sample
- methylation: CpG or region scores
- proteomics: protein abundance
- metabolomics
- predicted expression
- variant pathogenicity summaries
- gene/pathway burden scores

Require explicit sample ID mapping before combination.

### Stage 7: split and assemble

Create split definitions once, store sample IDs, and reuse the exact same split across all feature sources.

Critical leakage rule:

Any transformation that learns from samples, including scaling, feature selection, PCA fitted from target individuals, imputation statistics or model-based feature selection, must be fit only on training data and then applied to validation/test data.

Population-genetic PCA produced by an external reference strategy can be treated differently, but the strategy must be recorded.

### Stage 8: train/evaluate

Model code should consume a model-ready feature matrix and split definition rather than know whether a column originated from PRS, genotype, annotation or omics.

Store:

- AUROC/AUPRC for binary traits
- R2/RMSE/MAE for continuous traits
- calibration
- confidence intervals
- predictions by sample
- model parameters
- feature groups
- fold IDs

## 9. Cross-phenotype and multi-trait design

EFGPP2 should explicitly support the experiment already explored by EFGPP: using data from related phenotypes.

A feature artifact should therefore contain:

- target phenotype
- source phenotype
- source GWAS
- method

Examples:

- target=migraine, source=migraine, method=PRSice2
- target=migraine, source=depression, method=PRSice2
- target=migraine, source=depression, method=annotation_genotype

This makes cross-trait contribution measurable instead of encoded only in file names.

## 10. Annotation strategy

Create an annotation catalogue where each release records:

- provider/tool
- release/version
- genome build
- source URL or local path
- license
- fields
- chromosome coverage
- checksum

Candidate sources from the CAGI workflow include:

- RefGene / Ensembl / GENCODE
- ClinVar
- gnomAD frequencies and constraint
- dbNSFP
- AlphaMissense
- ESM-derived scores
- conservation scores
- regulatory regions
- protein domains
- GO / Reactome / STRING / UniProt-derived mappings

Do not force every annotation column into the sample-level matrix. Prefer variant-level annotation artifacts, then generate compact biological feature sets.

## 11. Provenance

Every external-tool execution should record:

- operation
- command
- tool and version
- input artifact IDs
- output artifact IDs
- parameters
- seed
- start/end timestamps
- environment
- status/error

This concept is adapted from GWASPokerforPRS2 and is essential for a framework intended to produce publishable results.

## 12. Security and restricted data

Do not commit:

- UK Biobank individual-level data
- participant IDs if they are restricted
- API keys/tokens
- PRSTools GitHub tokens
- large derived matrices containing restricted participant data

Git should contain code/config templates only.

## 13. Suggested migration sequence

1. Keep EFGPP main unchanged as the publication/reproducibility snapshot.
2. Develop EFGPP2 on a separate branch/repository.
3. Register the existing migraine and depression assets with the legacy scanner.
4. Reproduce one original EFGPP configuration exactly.
5. Compare generated fold IDs, PRS, feature dimensions and AUCs with the legacy run.
6. Only after parity, add new annotations and omics.
7. Move PRS adapters from PRSTools one method at a time.
8. Add a workflow engine later if needed: Snakemake, Nextflow or Prefect.

## 14. High-priority next implementation work

The current scaffold provides data identity, registry, ingestion, GWAS QC, annotation-safe joins, feature-table joins, leakage checks, PRS command plans, provenance and legacy discovery.

Next code modules should be implemented in this order:

1. PLINK2 genotype QC adapter.
2. GWAS/genotype allele harmonizer.
3. Split manager storing fold IDs.
4. Parquet materialization with schema validation.
5. PCA adapter.
6. PLINK/PRSice execution and score parsers.
7. AnnoPred/LDAK adapters.
8. Annotation catalogue and chromosome-partitioned Parquet builder.
9. Model-ready feature manifest.
10. ML/DL trainer with group-aware provenance.
11. Multi-omics adapters.
12. Reproduction test for the original migraine experiment.
