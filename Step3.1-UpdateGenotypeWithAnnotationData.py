from email.header import Header
import pandas as pd
import numpy as np
import os
import shutil
import re 
import glob
import sys
 
from typing import List
  
def merge_snp_data(genotype_df, annotation_df):
    # Create a copy of the input dataframes
    merged_data = genotype_df.copy()
    
    # Define columns to exclude from multiplication
    exclude_cols = ['CHR','BP','SNP', 'CM', 'base']
    
    # For each SNP in the annotation data
    for idx, anno_row in annotation_df.iterrows():
        snp_id = anno_row['SNP']
        
        # Get the genotype column for this SNP (if it exists)
        genotype_col = next((col for col in genotype_df.columns if col.split("_")[0] == snp_id), None)
        
        if genotype_col:  # if we found a matching column
            # Multiply genotype with all annotation columns except excluded ones
            for col in annotation_df.columns:
                if col not in exclude_cols:
                    merged_data[f'{snp_id}_{col}_interaction'] = merged_data[genotype_col] * anno_row[col]
    
    return merged_data

phenotypename = sys.argv[1]
gwasfilename = sys.argv[2]
fold = sys.argv[3]

if "." in gwasfilename:
    gwasfilenamewithoutextension = gwasfilename.split(".")[0]

numberofsnps = [50, 100, 200, 500]

#annotations = pd.read_csv("Annotations.tsv", sep="\s+")

def genotype_data_snps(data):
    snps = data.columns[6:]  # Get SNP columns (skip first 6 columns)
    snps = [snp.split("_")[0] for snp in snps]  # Extract SNP names
 
    return snps

def get_common_between_snps_and_annotations(snps, annotations):
    return annotations[annotations["SNP"].isin(snps)]

# Read annotations file
annotations = pd.read_csv("Annotations.tsv", sep="\s+")

for snps in numberofsnps:
    # Process train data
    train = pd.read_csv(f"{phenotypename}/{gwasfilenamewithoutextension}/Fold_{fold}/snps_{snps}/ptrain.raw", sep="\s+") 
    snps_list = genotype_data_snps(train)
    tempannotations = get_common_between_snps_and_annotations(snps_list, annotations)

    merged_train = merge_snp_data(train, tempannotations)
    os.makedirs(f"{phenotypename}/{gwasfilenamewithoutextension}/Fold_{fold}/snps_annotated_{snps}/", exist_ok=True)
    merged_train.to_csv(f"{phenotypename}/{gwasfilenamewithoutextension}/Fold_{fold}/snps_annotated_{snps}/ptrain.raw", sep="\t", index=False)
    
    # Process test data
    test = pd.read_csv(f"{phenotypename}/{gwasfilenamewithoutextension}/Fold_{fold}/snps_{snps}/ptest.raw", sep="\s+")
    print("Test dimensions:", test.shape)
    merged_test = merge_snp_data(test, tempannotations)
    merged_test.to_csv(f"{phenotypename}/{gwasfilenamewithoutextension}/Fold_{fold}/snps_annotated_{snps}/ptest.raw", sep="\t", index=False)
    
    # Process validation data
    val = pd.read_csv(f"{phenotypename}/{gwasfilenamewithoutextension}/Fold_{fold}/snps_{snps}/pval.raw", sep="\s+")
    print("Validation dimensions:", val.shape)
    merged_val = merge_snp_data(val, tempannotations)
    merged_val.to_csv(f"{phenotypename}/{gwasfilenamewithoutextension}/Fold_{fold}/snps_annotated_{snps}/pval.raw", sep="\t", index=False)
    
    # Copy .fam files
    for file_type in ['pval', 'ptest', 'ptrain']:
        shutil.copy(
            f"{phenotypename}/{gwasfilenamewithoutextension}/Fold_{fold}/snps_{snps}/{file_type}.fam",
            f"{phenotypename}/{gwasfilenamewithoutextension}/Fold_{fold}/snps_annotated_{snps}/{file_type}.fam"
        )

    print("Merged dimensions:")
    print("Train:", merged_train.shape)
    print("Test:", merged_test.shape)
    print("Validation:", merged_val.shape)
  
