import sys
import os
from pathlib import Path
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def process_unique_datasets(phenotype):
    # Initialize empty set for union
    all_datasets = set()
    fold_datasets = []
    
    # Process each fold
    for fold in range(0,5):
        file_path = Path(f"{phenotype}/Fold_{fold}/UniqueDatasets/unique_datasets.txt")
        if file_path.exists():
            with open(file_path, 'r') as f:
                fold_set = set(line.strip() for line in f if line.strip())
                fold_datasets.append(fold_set)
                all_datasets.update(fold_set)
    
    # Print statistics
    print(f"Total unique datasets across all folds: {len(all_datasets)}")
    for i, fold_set in enumerate(fold_datasets):
        print(f"Fold {i} unique datasets: {len(fold_set)}")
    
    # Create results directory if it doesn't exist
    results_dir = Path(f"{phenotype}/Results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save combined unique datasets with natural sort
    output_file = results_dir / "UniqueDatasets.txt"
    with open(output_file, 'w') as f:
        for dataset in sorted(all_datasets, key=natural_sort_key):
            f.write(f"{dataset}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <phenotype_path>")
        sys.exit(1)
    
    process_unique_datasets(sys.argv[1])

