from email.header import Header
import pandas as pd
import numpy as np
import os
import shutil
import re 
import glob
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from typing import List

def find_p_value_threshold(gwasfile: pd.DataFrame, target_snps: int) -> float:
    """
    Use binary search to find p-value threshold that yields target number of SNPs
    """
    p_values = gwasfile['P'].sort_values().values
    left, right = p_values[0], p_values[-1]
    
    while left < right:
        mid = (left + right) / 2
        count = (p_values <= mid).sum()
        
        if count == target_snps:
            return mid
        elif count < target_snps:
            left = mid
        else:
            right = mid
            
        if right - left < 1e-10:  # Convergence threshold
            return mid
    
    return left

def process_plink_commands(pdirectory: str, snp_file: str, traindirec: str) -> None:
    """
    Execute PLINK commands in parallel using OS system calls
    """
    commands = [
        f"./plink --bfile ./{traindirec}/test_data --extract {snp_file} --recodeA --out {pdirectory}/ptest",
        f"./plink --bfile ./{traindirec}/validation_data --extract {snp_file} --recodeA --out {pdirectory}/pval",
        f"./plink --bfile ./{traindirec}/train_data.QC --extract {snp_file} --recodeA --out {pdirectory}/ptrain",
        f"./plink --bfile ./{traindirec}/test_data --extract {snp_file} --make-bed --out {pdirectory}/ptest",
        f"./plink --bfile ./{traindirec}/validation_data --extract {snp_file} --make-bed --out {pdirectory}/pval",
        f"./plink --bfile ./{traindirec}/train_data.QC --extract {snp_file} --make-bed --out {pdirectory}/ptrain"
    ]
    
    # Execute commands
    for cmd in commands:
        os.system(cmd)

def main():
    phenotypename = sys.argv[1]
    gwasfilename = sys.argv[2]
    fold = sys.argv[3]

    if "." in gwasfilename:
        gwasfilenamewithoutextension = gwasfilename.split(".")[0]

    save_path = os.path.join(phenotypename, gwasfilenamewithoutextension)
    traindirec = os.path.join(save_path, f"Fold_{fold}")
    
    # Read GWAS file once
    gwasfile = pd.read_csv(os.path.join(save_path, f"{phenotypename}.txt"), 
                          sep="\s+", 
                          dtype={'P': np.float64})  # Specify dtype for faster loading
    
    print(gwasfile.head())

    # Define target SNP counts
    numberofsnps = [50, 100, 200, 500, 1000, 5000, 10000, 30000, 60000, 5]
    pvalues = []

    # Find p-value thresholds for each target SNP count
    for target_snps in numberofsnps[:-1]:  # Exclude last value as per original code
        threshold = find_p_value_threshold(gwasfile, target_snps)
        pvalues.append(threshold)
        print(f"Found threshold {threshold} for {target_snps} SNPs")

    # Process results
    for i, p_threshold in enumerate(pvalues):
        pdirectory = os.path.join(traindirec, f"snps_{numberofsnps[i]}")
        os.makedirs(pdirectory, exist_ok=True)
        
        # Get SNPs meeting threshold
        snp_file = os.path.join(pdirectory, f"{numberofsnps[i]}.txt")
        mask = gwasfile['P'] <= p_threshold
        gwasfile.loc[mask, 'SNP'].to_csv(snp_file, index=False, header=False)
        
        # Process PLINK commands
        process_plink_commands(pdirectory, snp_file, traindirec)

if __name__ == "__main__":
    main()