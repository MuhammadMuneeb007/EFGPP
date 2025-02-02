import sys
import os
import pandas as pd

def create_directory(directory):
    """Function to create a directory if it doesn't exist."""
    if not os.path.exists(directory):  # Check if the directory doesn't exist
        os.makedirs(directory)  # Create the directory if it doesn't exist
    return directory  # Return the created or existing directory

import numpy as np
import random
import os
import tensorflow as tf
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
tf.random.set_seed(seed_value)



phenotype = sys.argv[1]
gwasfilename = sys.argv[2]
gwasfilenamewithoutextension = ""
if "." in gwasfilename:
    gwasfilenamewithoutextension = gwasfilename.split(".")[0]

create_directory(phenotype)
create_directory(phenotype+os.sep+gwasfilenamewithoutextension)

import os
import pandas as pd

gwasfilepath = phenotype + os.sep + gwasfilename 
save_path = phenotype + os.sep + gwasfilenamewithoutextension


# Function to detect the genomic build of the GWAS and update the genotype .bed file.
def detect_genome_build(phenotype):
    # Define the path to the GWAS file for the given phenotype
    GWAS = gwasfilepath
    
    # Read the GWAS file
    df = pd.read_csv(GWAS, compression="gzip", sep="\s+", on_bad_lines='skip')
    df['BP'] = df['BP'].astype(int)

    # Read the bim file for hg19 build
    bimfile = pd.read_csv("genotypes.bim19", sep="\s+", header=None)
    
    # Create a matching column in bimfile and df based on chromosome, base pair location, and alleles
    bimfile["match"] = bimfile[0].astype(str) + "_" + bimfile[3].astype(str) + "_" + bimfile[4].astype(str) + "_" + bimfile[5].astype(str)
    df["match"] = df["CHR"].astype(str) + "_" + df["BP"].astype(str) + "_" + df["A1"].astype(str) + "_" + df["A2"].astype(str)
    
    # Remove duplicates in the match columns
    #df.drop_duplicates(subset='match', inplace=True)
    #bimfile.drop_duplicates(subset='match', inplace=True)
    merged_df = pd.merge(bimfile, df, on='match', how='inner', suffixes=('_bimfile', '_df'))


    # Count the number of variants matching the hg19 build
    hg19variants = len(merged_df)
    print(hg19variants)

    # Re-read the GWAS file to reset the dataframe
    df = pd.read_csv(GWAS, compression="gzip", sep="\s+", on_bad_lines='skip')
    df['BP'] = df['BP'].astype(int)

    # Read the bim file for hg38 build
    bimfile = pd.read_csv("genotypes.bim38", sep="\s+", header=None)
    
    # Create a matching column in bimfile and df based on chromosome, base pair location, and alleles
    bimfile["match"] = bimfile[0].astype(str) + "_" + bimfile[3].astype(str) + "_" + bimfile[4].astype(str) + "_" + bimfile[5].astype(str)
    df["match"] = df["CHR"].astype(str) + "_" + df["BP"].astype(str) + "_" + df["A1"].astype(str) + "_" + df["A2"].astype(str)
    
    # Remove duplicates in the match columns
    #df.drop_duplicates(subset='match', inplace=True)
    #bimfile.drop_duplicates(subset='match', inplace=True)
    merged_df = pd.merge(bimfile, df, on='match', how='inner', suffixes=('_bimfile', '_df'))

    # Count the number of variants matching the hg38 build
    hg38variants = len(merged_df)
    
    print(hg38variants)

    # Return the build with more matching variants
    if hg19variants > hg38variants:
        return "19"
    if hg38variants > hg19variants:
        return "38"
     

# Check the detected genome build and copy the corresponding bim file to the phenotype directory
if detect_genome_build(phenotype) == "19":
    os.system("cp genotypes.bim19 " + save_path + os.sep + phenotype + ".bim")
     
if detect_genome_build(phenotype) == "38":
    os.system("cp genotypes.bim38 " + save_path + os.sep + phenotype + ".bim")

# Copy the genotype .bed and .fam files to the phenotype directory
os.system("cp genotypes.bed " + save_path + os.sep + phenotype + ".bed")
os.system("cp genotypes.fam " + save_path + os.sep + phenotype + ".fam")
#os.system("cp genotypes.bim " + save_path + os.sep + phenotype + ".bim")
os.system("cp  "+ gwasfilepath+" " + save_path + os.sep + phenotype + ".gz")





# Load the phenotype file
data = pd.read_csv("phenotype_file.txt", sep="\s+")


# Define the path to save the list of phenotypes
file_path = 'All_Phenotypes.txt'

# Load the actual phenotype data
actualphenotype = pd.read_csv("phenotype_file.txt", sep="\t")
# Select the covariate columns (first 152 columns)

covariatecolumns = actualphenotype.columns.to_list()[0:152]
#covariatecolumns = actualphenotype.columns.to_list()[0:4]

cov = actualphenotype[covariatecolumns]
del cov['fasting_time']  # Remove the 'fasting_time' column
try:
    del cov['hypertension']  # Remove the 'ID' column
    del cov['depression']  # Remove the 'ID' column
    del cov['migraine']  # Remove the 'ID' column
    
except KeyError:
    print("The column 'ID' does not exist.")

import re

try:
    x = re.sub(r'_\d+', '', phenotype)
    del cov[x]
except KeyError:
    print("The column 'ID' does not exist.")


cov.rename(columns={'ID': 'IID'}, inplace=True)
cov.rename(columns={'sex': 'Sex'}, inplace=True)

# Update the Sex column for PLINK
cov['Sex'] = cov['Sex'].replace({1: 2, 0: 1})

# Create the directory for the phenotype if it doesn't exist
create_directory(phenotype)

# Add 'FID' column with the same values as 'IID'
cov.insert(loc=1, column='FID', value=cov["IID"].values)
# Save the covariate file
cov.to_csv(save_path + os.sep + phenotype + ".cov", sep="\t", index=False)

# Load the actual phenotype data
actualphenotype = pd.read_csv("phenotype_file.txt", sep="\t")
print(actualphenotype.head())

# Create a temporary dataframe for the phenotype data
tempframe = pd.DataFrame()
tempframe["IID"] = actualphenotype["ID"].values
tempframe["FID"] = actualphenotype["ID"].values

# Change the following Line.
#tempframe["Height"] = actualphenotype[sys.argv[2]].values
tempframe["Height"] = actualphenotype[x].values

column_values = tempframe["Height"].unique()

# If the phenotype is binary, replace values 0 and 1 with 1 and 2 as required by PLINK
if set(column_values) == {0, 1}:
    # Change the cases and controls
    print("The column contains only 0 and 1.")
    tempframe.replace({0: 1, 1: 2}, inplace=True)
else:
    print("The column does not contain only 0 and 1.")

# Save the temporary phenotype file
tempframe.to_csv(save_path + os.sep + phenotype + ".heightOLD", sep="\t", index=False)

# Reindex the temporary frame to match the order in the .fam file
tempframe.set_index('IID', inplace=True)
custom_order = pd.read_csv(save_path + os.sep + phenotype + ".fam", sep="\s+", header=None)[0].values
df_reindexed = tempframe.reindex(custom_order)
df_reindexed = df_reindexed.reset_index()
df_reindexed.to_csv(save_path + os.sep + phenotype + ".height", sep="\t", index=False)

# Load and process the covariate file to match the order in the .fam file
cov = pd.read_csv(save_path + os.sep + phenotype + ".cov", sep="\s+")
cov.set_index('IID', inplace=True)
cov.to_csv(save_path + os.sep + phenotype + ".covOLD", sep="\t", index=False)
custom_order = pd.read_csv(save_path + os.sep + phenotype + ".fam", sep="\s+", header=None)[0].values
df_reindexed = cov.reindex(custom_order)
df_reindexed = df_reindexed.reset_index()
df_reindexed.to_csv(save_path + os.sep + phenotype + ".cov", sep="\t", index=False)






