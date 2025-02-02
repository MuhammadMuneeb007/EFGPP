# Exploratory Framework for Genotype-Phenotype Prediction

## Initial Setup

1. Place your genotype data
2. Put data in the following format for a specific Phenotype
3. Create a directory/Folder

### Directory Structure

```
EFGPP/
```

## Data Requirements

1. Download Annotations.tsv and place it in the working folder from:
   https://drive.google.com/drive/folders/19PMporIqzUj9IY3FNXbnzR_I-aBy8KmT?usp=sharing

Location: `EFGPP/Annotations.tsv`

### Directory Structure for Phenotypes

```
EFGPP/Phenotype
EFGPP/Phenotype/Phenotype_1.gz (GWAS file 1)
EFGPP/Phenotype/Phenotype_2.gz (GWAS file 2)
EFGPP/Phenotype/Phenotype_1 (Directory containing genotype data)
EFGPP/Phenotype/Phenotype_2 (Directory containing genotype data)
```

### Required Files for Each Phenotype

For Phenotype_1:
```
EFGPP/Phenotype/Phenotype_1/Phenotype.gz (Cleaned GWAS file)
EFGPP/Phenotype/Phenotype_1/Phenotype.bed 
EFGPP/Phenotype/Phenotype_1/Phenotype.bim 
EFGPP/Phenotype/Phenotype_1/Phenotype.fam 
EFGPP/Phenotype/Phenotype_1/Phenotype.cov   (FID,IID,COV1,COV2,...)
EFGPP/Phenotype/Phenotype_1/Phenotype.height (FID,IID,Height)
```

For Phenotype_2:
```
EFGPP/Phenotype/Phenotype_2/Phenotype.gz (Cleaned GWAS file)
EFGPP/Phenotype/Phenotype_2/Phenotype.bed 
EFGPP/Phenotype/Phenotype_2/Phenotype.bim 
EFGPP/Phenotype/Phenotype_2/Phenotype.fam 
EFGPP/Phenotype/Phenotype_2/Phenotype.cov   (FID,IID,COV1,COV2,...)
EFGPP/Phenotype/Phenotype_2/Phenotype.height (FID,IID,Height)
```

### Sample File Formats

Sample `EFGPP/Phenotype/Phenotype_2/Phenotype.height`:
```
IID     FID     Height
5777494 5777494 1
3939107 3939107 1
```

Sample `EFGPP/Phenotype/Phenotype_2/Phenotype.gz`:
```
CHR     BP      SNP     A1      A2      N       SE      P       OR      INFO    MAF
8       101592213       rs62513865      T       C       480359  0.0153  0.3438  1.01461 0.957   1
8       106973048       rs79643588      A       G       480359  0.0136  0.1231  1.02122 0.999   1
8       108690829       rs17396518      T       G       480359  0.008   0.6821  1.00331 0.98    1
8       108681675       rs983166        A       C       480359  0.008   0.2784  0.99144 0.991   1
8       103044620       rs28842593      T       C       480359  0.0112  0.3381  0.98926 0.934   1
8       105176418       rs377046245     I       D       480359  0.0196  0.1311  1.03004 0.994   1
8       103128181       chr8_103128181_I        D       I       480359  0.0108  0.968   1.0004  0.995   1
8       100479917       rs3134156       T       C       480359  0.0108  0.6805  0.99561 0.987   1
8       103144592       rs6980591       A       C       480359  0.0095  0.1579  1.01349 0.994    1
```

## Actual Directory Examples

### Migraine Directory Structure
```
EFGPP/migraine/
├── migraine
│   ├── migraine.bed
│   ├── migraine.bim
│   ├── migraine.cov
│   ├── migraine.fam
│   ├── migraine.gz
│   ├── migraine.height
├── migraine_5
│   ├── migraine.bed
│   ├── migraine.bim
│   ├── migraine.cov
│   ├── migraine.fam
│   ├── migraine.gz
│   ├── migraine.height
├── migraine_5.gz (GWAS files)
├── migraine.gz (GWAS files)
```

### Depression Directory Structure
```
EFGPP/depression/
├── depression_11
│   ├── depression.bed
│   ├── depression.bim
│   ├── depression.cov
│   ├── depression.fam
│   ├── depression.gz
├── depression_11.gz
├── depression_17
│   ├── depression.bed
│   ├── depression.bim
│   ├── depression.cov
│   ├── depression.fam
│   ├── depression.gz
├── depression_17.gz
├── depression_4
│   ├── depression.bed
│   ├── depression.bim
│   ├── depression.cov
│   ├── depression.fam
│   ├── depression.gz
├── depression_4.gz
```

## Processing Steps

### Step 2: Quality Controls
Execute this command for quality controls. Specify the Phenotype and GWAS file as arguments:
```bash
python Step2-GWASAndIndividualGenotypeDataQualityControls.py migraine migraine_5.gz
python Step2-GWASAndIndividualGenotypeDataQualityControls.py Phenotype GWASFile
```

### Step 2: Split Data
This step will split the data using stratified fold:
```python
python Step2-SplitData.py 

phenotype_paths = [
    ('migraine', 'migraine/migraine'),
    ('migraine', 'migraine/migraine_5'),
    ('depression', 'depression/depression_11'),
    ('depression', 'depression/depression_17'),
    ('depression', 'depression/depression_4'),
]
```

### Step 3: P-value Threshold
Perform p-value threshold on the genotype data:
```bash
python Step3-PrepareDataForGenotypeMachineDeepLearning-P-valueThreshold.py migraine migraine_5 0
python Step3-PrepareDataForGenotypeMachineDeepLearning-P-valueThreshold.py Phenotype GWASFile Fold
```

### Step 3.1: Functional Annotations
Append functional annotations with the genotype data:
```bash
python Step3.1-UpdateGenotypeWithAnnotationData.py migraine migraine_5 0
python Step3.1-UpdateGenotypeWithAnnotationData.py Phenotype GWASFile Fold
```

### Step 4: Generate PCA
Generate PCA for genotype data:
```bash
python Step4-GeneratePCA.py migraine migraine_5
python Step4-GeneratePCA.py Phenotype GWASFile
```
### Step 5: Generate PRS

Generate PRS scores using various tools:
```bash
# Using Plink
python Step5-Plink.py Phenotype GWASFile 0 
python Step5-Plink.py migraine migraine_5 0

# Using LDAK-GWAS
python Step5-LDAK-GWAS.py migraine migraine_5 0

# Using PRSice-2
python Step5-PRSice-2.py migraine migraine_5 0

# Using AnnoPred
python Step5-AnnoPredCode.py migraine migraine_5 0
```

**Note**: Before running these commands:
- Download and install: Plink, LDAK, LDAK-GWAS, and PRSice-2
- Installation guide: [PRSTools Installation Guide](https://muhammadmuneeb007.github.io/PRSTools/Introduction.html)

### Step 6: Generate Base Datasets

Generate the core base datasets:
```bash
python Step6-CoreBaseDataGenerator.py Phenotype Fold
python Step6-CoreBaseDataGenerator.py migraine 0 
```

This will create datasets in:
```
EFGPP/migraine/Fold_0/Datasets
EFGPP/migraine/Fold_1/Datasets
EFGPP/migraine/Fold_2/Datasets
EFGPP/migraine/Fold_3/Datasets
EFGPP/migraine/Fold_4/Datasets
```

Configuration options in CoreBaseDataGenerator.py:
```python
self.phenotype_gwas_pairs = [
    ("migraine", "migraine.gz"),
    ("migraine", "migraine_5.gz"),
    # ("depression", "depression_11.gz"),
    # ("depression", "depression_4.gz"),
    # ("depression", "depression_17.gz"),
]

self.models = ["Plink", "PRSice-2", "AnnoPred", "LDAK-GWAS"]
self.pca_components = 10
self.scaling_options = [False]  # Can be [True, False]
self.snp_options = [
    "snps_50", "snps_200", "snps_500", "snps_1000",
    "snps_5000", "snps_10000", "snps_annotated_50",
    "snps_annotated_200", "snps_annotated_500"
]
```
### Step 7: Dataset Analysis

Analyze datasets across folds using these commands:

```bash
# Find unique datasets for a specific fold
python Step7-CoreBaseDataSelection-FindSimilarity.py Phenotype Fold
python Step7-CoreBaseDataSelection-FindSimilarity.py migraine 0

# Find common datasets across all folds
python Step7.1-CoreBaseDataSelection-FindCommon.py Phenotype
python Step7.1-CoreBaseDataSelection-FindCommon.py migraine
```

Results will be stored in: `EFGPP/migraine/Results/UniqueDatasets.txt`
### Step 8: ML/DL Algorithm Application

Apply machine learning and deep learning algorithms using:

```bash
python Step8-CoreBasePredictor.py Phenotype Fold NumberofDataset Algorithm
```

Example commands:
- For machine learning:
    ```bash
    python CoreBasePredictor.py migraine 0 1 ML
    ```
- For deep learning:
    ```bash
    python CoreBasePredictor.py migraine 0 1 DL
    ```


### Step 9: Results Aggregation

Aggregate results across folds per dataset:

```bash
# For machine learning results
python Step9-CoreBasePredictorAggregate.py migraine ResultsML 0.5

# For deep learning results
python Step9-CoreBasePredictorAggregate.py migraine ResultsDL 0.5
```

Results are stored in `EFGPP/migraine/Results/ResultsML/Aggregated`

#### Sample Results Format

```csv
Dataset,Model,ML_Parameters,Train AUC,Validation AUC,Test AUC,...
1,Naive Bayes,{'var_smoothing': 1e-07},0.7487,0.5550,0.5867,...
2,Logistic Regression,{'C': 0.1, 'penalty': 'l2'},0.6540,0.5378,0.5419,...
3,Logistic Regression,{'C': 1.0, 'penalty': 'l2'},0.6483,0.6286,0.6321,...
```

### Step 10: Exploratory Data Analysis

Analyze your results with:

```bash
# For basic exploratory data analysis
python Step10-CoreBaseExploratoryDataAnalysis.py migraine ResultsML 
```

This generates comprehensive insights including:
- Top performing models and their frequency
- Best AUC metrics across:
    - Phenotype-GWAS pairs
    - Dataset types
    - Weight file configurations
    - SNP types
    - Model types

Example visualization:
![EDA Results](EDA_ML_Results.png)

### Step 10.1: Cluster Analysis

For detailed cluster analysis:

```bash
python Step10.1-CoreBaseDataClusterAnalysis.py migraine ResultsML 
```

Example cluster visualization:
![Cluster Results](Cluster_ML_Results.png)


