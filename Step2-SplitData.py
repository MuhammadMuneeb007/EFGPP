import sys
import os
import pandas as pd
import subprocess
import numpy as np
import random
from collections import Counter

# Set random seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

# Use a list of tuples to allow duplicate phenotype names
phenotype_paths = [
    ('migraine', 'migraine/migraine'),
    ('migraine', 'migraine/migraine_5'),
     

    ('depression', 'depression/depression_11'),
    ('depression', 'depression/depression_17'),
    ('depression', 'depression/depression_4'),
]

def force_stratified_split(master_df, strata, n_splits=5):
    """
    Create stratified folds with consistent train sizes.
    Each fold will have approximately (n_splits-1)/n_splits of the data for training.
    """
    splits = {}
    all_indices = np.arange(len(master_df))
    
    # For each unique stratum, pre-divide the indices into n_splits roughly equal parts
    strata_splits = {}
    for stratum in np.unique(strata):
        stratum_indices = np.where(strata == stratum)[0]
        n_samples = len(stratum_indices)
        
        # Shuffle indices
        np.random.seed(seed_value)
        np.random.shuffle(stratum_indices)
        
        # Split into roughly equal parts
        split_size = n_samples // n_splits
        remainder = n_samples % n_splits
        
        current_idx = 0
        strata_splits[stratum] = []
        
        for i in range(n_splits):
            # Add one extra sample if there's remainder
            current_split_size = split_size + (1 if i < remainder else 0)
            split_indices = stratum_indices[current_idx:current_idx + current_split_size]
            strata_splits[stratum].append(split_indices)
            current_idx += current_split_size
    
    # Create folds
    for fold_id in range(n_splits):
        test_indices = []
        val_indices = []
        train_indices = []
        
        # For each stratum
        for stratum in strata_splits:
            # Current part becomes test/val
            holdout_indices = strata_splits[stratum][fold_id]
            
            # Split holdout into test and validation
            n_holdout = len(holdout_indices)
            n_val = n_holdout // 2
            
            val_indices.extend(holdout_indices[:n_val])
            test_indices.extend(holdout_indices[n_val:])
            
            # All other parts become train
            for other_fold in range(n_splits):
                if other_fold != fold_id:
                    train_indices.extend(strata_splits[stratum][other_fold])
        
        # Store splits using FID/IID pairs
        splits[f'Fold_{fold_id}'] = {
            'train': set(zip(master_df.iloc[train_indices]['FID'], master_df.iloc[train_indices]['IID'])),
            'validation': set(zip(master_df.iloc[val_indices]['FID'], master_df.iloc[val_indices]['IID'])),
            'test': set(zip(master_df.iloc[test_indices]['FID'], master_df.iloc[test_indices]['IID']))
        }
        
        # Print split sizes
        print(f"\nFold {fold_id} split sizes:")
        print(f"Train: {len(splits[f'Fold_{fold_id}']['train'])}")
        print(f"Validation: {len(splits[f'Fold_{fold_id}']['validation'])}")
        print(f"Test: {len(splits[f'Fold_{fold_id}']['test'])}")
        
        # Verify no overlap
        train_set = splits[f'Fold_{fold_id}']['train']
        val_set = splits[f'Fold_{fold_id}']['validation']
        test_set = splits[f'Fold_{fold_id}']['test']
        
        assert len(train_set & val_set) == 0, "Overlap between train and validation sets"
        assert len(train_set & test_set) == 0, "Overlap between train and test sets"
        assert len(val_set & test_set) == 0, "Overlap between validation and test sets"
    
    return splits

def process_phenotypes(phenotype_paths, splits):
    """Process each phenotype path using the provided splits."""
    for pheno_name, mainpath in phenotype_paths:
        print(f"\nProcessing phenotype path: {pheno_name} - {mainpath}")
        newfilename = os.path.basename(pheno_name) + "_QC"
        input_file_path = os.path.join(mainpath, f"{newfilename}.fam")
        df = pd.read_csv(input_file_path, sep="\s+", header=None)
        
        output_directory_base = mainpath
        os.makedirs(output_directory_base, exist_ok=True)
        
        # Print phenotype distribution
        pheno_counts = Counter(df[5])
        print(f"Phenotype distribution for {mainpath}:")
        for value, count in pheno_counts.items():
            print(f"Value {value}: {count} samples")
        
        for fold_name, fold_splits in splits.items():
            print(f"\nProcessing {fold_name}")
            fold_directory = os.path.join(output_directory_base, fold_name)
            os.makedirs(fold_directory, exist_ok=True)
            
            # Process each split
            for dataset_name, split_pairs in fold_splits.items():
                # Get data for this split
                dataset = df[df.apply(lambda row: (row[0], row[1]) in split_pairs, axis=1)]
                dataset_file_name = f"{dataset_name}_data"
                
                # Print phenotype distribution for this split
                split_counts = Counter(dataset[5])
                print(f"\nPhenotype distribution for {dataset_name} in {fold_name}:")
                for value, count in split_counts.items():
                    print(f"Value {value}: {count} samples")
                
                # Save split to .fam file
                output_path = os.path.join(fold_directory, f'{dataset_file_name}.fam')
                dataset.to_csv(output_path, sep="\t", header=False, index=False)
                
                # Run PLINK command for split
                plink_command = [
                    './plink',
                    '--bfile', os.path.join(mainpath, newfilename),
                    '--keep', output_path,
                    '--make-bed',
                    '--out', os.path.join(fold_directory, dataset_file_name)
                ]
                subprocess.run(plink_command)
                
                # Process covariates if they exist
                covfile_path = os.path.join(mainpath, f"{newfilename}.cov")
                if os.path.exists(covfile_path):
                    covfile = pd.read_csv(covfile_path, sep="\s+")
                    cov_dataset_data = covfile[covfile.apply(
                        lambda row: (row['FID'], row['IID']) in split_pairs,
                        axis=1
                    )]
                    cov_dataset_data.to_csv(
                        os.path.join(fold_directory, f'{dataset_file_name}.cov'),
                        sep="\t",
                        index=False
                    )
                
                # Run QC on training data
                if dataset_name == 'train':
                    run_plink_qc(fold_directory, dataset_file_name)

def run_plink_qc(fold_directory, dataset_file_name):
    """Run PLINK QC commands on training data."""
    new_dataset_file_name = f"{dataset_file_name}.QC"
    
    # First QC step
    plink_command_1 = [
        './plink',
        '--bfile', os.path.join(fold_directory, dataset_file_name),
        '--maf', '0.01',
        '--hwe', '1e-6',
        '--geno', '0.1',
        '--mind', '0.1',
        '--write-snplist',
        '--make-just-fam',
        '--out', os.path.join(fold_directory, new_dataset_file_name)
    ]
    subprocess.run(plink_command_1, check=True)
    
    # Heterozygosity check
    plink_command_2 = [
        './plink',
        '--bfile', os.path.join(fold_directory, dataset_file_name),
        '--extract', os.path.join(fold_directory, new_dataset_file_name + '.snplist'),
        '--keep', os.path.join(fold_directory, new_dataset_file_name + '.fam'),
        '--het',
        '--out', os.path.join(fold_directory, new_dataset_file_name)
    ]
    subprocess.run(plink_command_2, check=True)
    
    # Run R script for heterozygosity filtering
    r_command = f"Rscript Module1.R {os.path.join(fold_directory)} {dataset_file_name} {new_dataset_file_name} 1"
    print(f"Running: {r_command}")
    os.system(r_command)
    
    # Relatedness check
    plink_command_3 = [
        './plink',
        '--bfile', os.path.join(fold_directory, dataset_file_name),
        '--extract', os.path.join(fold_directory, new_dataset_file_name + '.snplist'),
        '--rel-cutoff', '0.125',
        '--out', os.path.join(fold_directory, new_dataset_file_name)
    ]
    subprocess.run(plink_command_3, check=True)
    
    # Final dataset creation
    plink_command_4 = [
        './plink',
        '--bfile', os.path.join(fold_directory, dataset_file_name),
        '--make-bed',
        '--keep', os.path.join(fold_directory, new_dataset_file_name + '.rel.id'),
        '--extract', os.path.join(fold_directory, new_dataset_file_name + '.snplist'),
        '--exclude', os.path.join(fold_directory, dataset_file_name + '.mismatch'),
        '--a1-allele', os.path.join(fold_directory, dataset_file_name + '.a1'),
        '--out', os.path.join(fold_directory, new_dataset_file_name)
    ]
    subprocess.run(plink_command_4, check=True)

def main():
    # Step 1: Create master dataframe
    print("Creating master dataframe...")
    master_df = pd.DataFrame()
    
    # Track the column names to ensure unique phenotype columns
    pheno_columns = {}
    
    # Read all phenotype files to get complete information
    for pheno_name, mainpath in phenotype_paths:
        newfilename = os.path.basename(pheno_name) + "_QC"
        input_file_path = os.path.join(mainpath, f"{newfilename}.fam")
        df = pd.read_csv(input_file_path, sep="\s+", header=None)
        
        if master_df.empty:
            master_df['FID'] = df[0]
            master_df['IID'] = df[1]
        
        # Create unique column name for each phenotype path
        col_name = f"pheno_{pheno_name}_{os.path.basename(mainpath)}"
        master_df[col_name] = df[5]
        pheno_columns[mainpath] = col_name
    
    # Create combined strata
    strata_cols = [col for col in master_df.columns if col.startswith('pheno_')]
    combined_strata = master_df[strata_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    
    # Step 2: Create stratified splits
    print("\nCreating stratified splits...")
    splits = force_stratified_split(master_df, combined_strata)
    
    # Step 3: Process phenotypes using splits
    print("\nProcessing phenotypes...")
    process_phenotypes(phenotype_paths, splits)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()