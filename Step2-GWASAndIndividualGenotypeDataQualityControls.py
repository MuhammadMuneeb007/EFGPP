

import sys
import os
import pandas as pd
import subprocess
import numpy as np
import random
import os
import tensorflow as tf
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
tf.random.set_seed(seed_value)


filedirec =   sys.argv[1]
gwasfilename = sys.argv[2]

if "." in gwasfilename:
    gwasfilenamewithoutextension = gwasfilename.split(".")[0]

mainpath = filedirec + os.sep + gwasfilenamewithoutextension


# Set the directory where the files are located
#filedirec = "SampleData1"
#filedirec = "asthma_19"
#filedirec = "migraine_0"

# Define file paths for different data files
BED = mainpath + os.sep + filedirec
BIM = mainpath + os.sep + filedirec+".bim"
FAM = mainpath + os.sep + filedirec+".fam"
COV = mainpath + os.sep + filedirec+".cov"
Height = mainpath + os.sep + filedirec+".height"
GWAS = mainpath + os.sep + filedirec+".gz"
#"""
#Read GWAS data from a compressed file using pandas
df = pd.read_csv(GWAS, compression="gzip", sep="\s+")

# Display the initial number of rows in the dataframe
print("Initial number of SNPs:", len(df))

# Apply quality control steps: Filter SNPs based on Minor Allele Frequency (MAF) and Imputation Information Score (INFO)
df = df.loc[(df['MAF'] > 0.01) & (df['INFO'] > 0.8) & (df['P'] > 0)]
df['BP'] = df['BP'].astype(int)
# Display the number of rows after applying the filters
print("Number of SNPs after quality control:", len(df))

# Remove duplicate SNPs.
df = df.drop_duplicates(subset='SNP')
 
# Display the number of rows after removing duplicate SNPs
print("SNPs in GWAS after removing duplicate SNPs:", len(df))

# Remove ambiguous SNPs with complementary alleles (C/G or A/T) to avoid potential errors
df = df[~((df['A1'] == 'A') & (df['A2'] == 'T') |
          (df['A1'] == 'T') & (df['A2'] == 'A') |
          (df['A1'] == 'G') & (df['A2'] == 'C') |
          (df['A1'] == 'C') & (df['A2'] == 'G'))]

# Display the final number of SNPs after removing ambiguous SNPs
print("Final number of SNPs after removing ambiguous SNPs:", len(df))

# Save the data.
df.to_csv(GWAS,compression="gzip",sep="\t",index=None)

df = pd.read_csv(GWAS,compression= "gzip",sep="\s+")
print(len(df))
print(df.head().to_markdown())
#"""

# ## Match Variants Between GWAS and Individual Genotype Data
# 
# If RSID is present in the GWAS, the following step can be skipped.
# 
# ### Steps for Handling RSIDs in GWAS and Genotype Data
# 
# 1. If RSIDs are not present for SNPs, put `X` in the SNP column in the GWAS file.
# 
# 2. Read the `genotype.bim` file and extract the RSIDs from the genotype data.
# 
# 3. If RSIDs are not present in the genotype data, use HapMap3 or another reference panel to obtain the RSIDs.
# 
# 
# Some PRS tools use different criteria to create unique variants and match them between GWAS and individual genotype data:
# 
# - **CHR:BP:A1:A2**: Some PRS tools use this format to define a unique variant.
# - **RSID**: Some PRS tools use RSID/SNP to define a unique variant.
# - **CHR:BP**: Some PRS tools use this format to define a unique variant.
# 
# We have highlighted which criteria are necessary for each tool.
# 

# In[2]:


bimfile = pd.read_csv(BIM, sep="\s+", header=None)
print("Columns of BIM file:")
print(bimfile.columns)
print("First 10 rows of BIM file:")


#print("Removing SNPs for which even a single row does not contain the required value:", len(df))


# If RSID's are not present for SNPs, put X in the SNP column in the GWAS file.
# Read the genotype.bim file, and extract the RSID from the genotype data.
# If RSID are not present in the genotype data, use HapMap3 or other reference panel to get the RSIDs.
#"""
if (df['SNP'] == 'X').all():
    print("RSIDs are missing!")
    bimfile = pd.read_csv(mainpath+os.sep+filedirec+".bim", sep="\s+", header=None)
    
    # create a unique variant using CHR:BP:A1:A2.
    
    bimfile["match"] = bimfile[0].astype(str)+"_"+bimfile[3].astype(str)+"_"+bimfile[4].astype(str)+"_"+bimfile[5].astype(str)
    df["match"] = df["CHR"].astype(str)+"_"+df["BP"].astype(str)+"_"+df["A1"].astype(str)+"_"+df["A2"].astype(str)
    

  

    df.drop_duplicates(subset='match', inplace=True)
    bimfile.drop_duplicates(subset='match', inplace=True)

    df = df[df['match'].isin(bimfile['match'].values)]
    bimfile = bimfile[bimfile['match'].isin(df['match'].values)]
    df = df[df['match'].isin(bimfile['match'].values)]
    bimfile = bimfile[bimfile['match'].isin(df['match'].values)]
 
    
    df = df.sort_values(by='BP')
    bimfile = bimfile.sort_values(by=3)
    
    print(df.head())
    print(bimfile.head())

    df["SNP"] = bimfile[1].values
    print("match",len(df))


    df.drop_duplicates(subset='match', inplace=True)
    bimfile.drop_duplicates(subset='match', inplace=True)  

    print(len(df))
    print(len(bimfile))
    print(df.head())
    print(bimfile.head())
    
    del df["match"]
    # Just save the modified GWAS file.
    # If bim, file is modified, the genotype data will be considered as corupt by Plink.
    df.to_csv(GWAS,compression="gzip",sep="\t",index=None)   
    print("Total SNPs", len(df))

    pass
else:
    df.drop_duplicates(subset='SNP', inplace=True)
    df.to_csv(GWAS,compression="gzip",sep="\t",index=None)
    print("RSID is present!")
    print("Total SNPs",len(df))
    pass
#"""


# ## Individual genotype data (Target Data) Processing
# 
# Ensure that the phenotype file, FAM file, and covariate file contain an identical number of samples. Remove any missing samples based on your data. Note that the extent of missingness in phenotypes and covariates may vary.
# 
# 
# **Note:** Plink needs to be installed or placed in the same directory as this notebook.
# 
# [Download Plink](https://www.cog-genomics.org/plink/)
# 
# We recommend using Linux. In cases where Windows is required due to package installation issues on Linux, we provide the following guidance:
# 
# 1. For Windows, use `plink`.
# 2. For Linux, use `./plink`.
# 
# 
# 

# ### Remove people with missing Phenotype
# 
# Modify the fam file, make bed file, and modify the covariates files as well.

# In[3]:


# New files to be saved with QC suffix
newfilename = filedirec + "_QC"

# Read information from FAM file
f = pd.read_csv(FAM, header=None, sep="\s+", names=["FID", "IID", "Father", "Mother", "Sex", "Phenotype"])
print("FAM file contents:")
print(f.head())
print("Total number of people in FAM file:", len(f))

# Append the Height phenotype values to FAM file
# Height file is basically the phenotype file.
h = pd.read_csv(Height, sep="\t")
print("Phenotype information is available for:", len(h), "people")
print(len(h))
result = pd.merge(f, h, on=['FID', 'IID'])


# Replace 'Phenotype' column with 'Height' and save to a new PeopleWithPhenotype.txt file
# Ensure that the input Phenotype file has teh header Height.
result["Phenotype"] = result["Height"].values
del result["Height"]

# Remove NA or missing in the phenotype column
result = result.dropna(subset=["Phenotype"])

 
result.to_csv(mainpath + os.sep + "PeopleWithPhenotype.txt", index=False, header=False, sep="\t")


# Use plink to keep only the people with phenotype present
plink_command = [
    './plink',
    '--bfile', mainpath + os.sep + filedirec,
    '--keep', mainpath + os.sep + "PeopleWithPhenotype.txt",
    '--make-bed',
    '--out', mainpath + os.sep + newfilename
]
subprocess.run(plink_command)

# Update the phenotype information in the new FAM file
f = pd.read_csv(mainpath + os.sep + newfilename + ".fam", header=None, sep="\s+",
                names=["FID", "IID", "Father", "Mother", "Sex", "Phenotype"])
f["Phenotype"] = result["Phenotype"].values
f.to_csv(mainpath + os.sep + newfilename + ".fam", index=False, header=False, sep="\t")

# Update the covariate file as well
covfile = mainpath + os.sep + filedirec + '.cov'
covfile = pd.read_csv(covfile, sep="\s+")

print("Covariate file contents:")
print(covfile.head())
print("Total number of people in Covariate file:", len(covfile))

# Match the FID and IID from covariate and height file
covfile = covfile[covfile['FID'].isin(f["FID"].values) & covfile['IID'].isin(f["IID"].values)]
print("Covariate file contents after matching with FAM file:")
print(covfile.head())
print("Total number of people in Covariate file after matching:", len(covfile))
covfile.to_csv(mainpath + os.sep + newfilename + ".cov", index=None, sep="\t")
 


# import sys
# import os
# import pandas as pd
# import subprocess
# import numpy as np
# import random
# import os
# import tensorflow as tf
# seed_value = 42
# random.seed(seed_value)
# np.random.seed(seed_value)
# os.environ['PYTHONHASHSEED'] = str(seed_value)
# tf.random.set_seed(seed_value)


# filedirec =   sys.argv[1]
# gwasfilename = sys.argv[2]

# if "." in gwasfilename:
#     gwasfilenamewithoutextension = gwasfilename.split(".")[0]

# mainpath = filedirec + os.sep + gwasfilenamewithoutextension


# # Set the directory where the files are located
# #filedirec = "SampleData1"
# #filedirec = "asthma_19"
# #filedirec = "migraine_0"

# # Define file paths for different data files
# BED = mainpath + os.sep + filedirec
# BIM = mainpath + os.sep + filedirec+".bim"
# FAM = mainpath + os.sep + filedirec+".fam"
# COV = mainpath + os.sep + filedirec+".cov"
# Height = mainpath + os.sep + filedirec+".height"
# GWAS = mainpath + os.sep + filedirec+".gz"



# import os
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
# import subprocess
# from sklearn.model_selection import KFold, cross_val_score

# # Step 1: Read the Fam file
# # New files to be saved with QC suffix
# newfilename = filedirec + "_QC"

# input_file_path = mainpath+os.sep+newfilename+'.fam'
# df = pd.read_csv(input_file_path,sep="\s+",header=None)

# # Step 2: Create 5 directories for storing fold information
# output_directory_base = mainpath
# os.makedirs(output_directory_base, exist_ok=True)

# # Step 3: Split the data into 5 folds using cross validation 
# fold_column = 5  # fifth column contains phenotypes
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# phenotype_col = 5

# # The following code is for binary phenotype.
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# column_values = df[phenotype_col].unique()
# from sklearn.model_selection import StratifiedKFold, train_test_split

# if set(column_values) == {1, 2}:
#     # Initialize StratifiedKFold for 5 folds
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#     # Step 4: Loop through each fold
#     for fold_id, (train_index, temp_index) in enumerate(skf.split(df, df[phenotype_col])):
#         fold_directory = os.path.join(output_directory_base, f'Fold_{fold_id}')
#         os.makedirs(fold_directory, exist_ok=True)

#         # Split the data for training, validation, and testing
#         train_data = df.iloc[train_index]
#         remaining_data = df.iloc[temp_index]
        
#         # Split the remaining data into 50% validation and 50% test
#         validation_data, test_data = train_test_split(remaining_data, test_size=0.5, stratify=remaining_data[phenotype_col], random_state=42)

#         # Define file names for each split
#         for dataset_name, dataset in [('train', train_data), ('validation', validation_data), ('test', test_data)]:
#             dataset_file_name = f"{dataset_name}_data"

#             # Save the splits to .fam files
#             dataset.to_csv(os.path.join(fold_directory, f'{dataset_name}_data.fam'), sep="\t", header=False, index=False)

#             # Run PLINK commands for the current split
#             plink_command = [
#                 './plink',
#                 '--bfile', mainpath + os.sep + newfilename,
#                 '--keep', os.path.join(fold_directory, f'{dataset_name}_data.fam'),
#                 '--make-bed',
#                 '--out', os.path.join(fold_directory, dataset_file_name)
#             ]
#             subprocess.run(plink_command)

#             # Process the covariate file
#             covfile = mainpath + os.sep + newfilename + '.cov'
#             covfile = pd.read_csv(covfile)

#             # Ensure that the covariate data is consistent with the dataset indices
#             cov_dataset_data = covfile.iloc[dataset.index]

#             # Save the corresponding covariate file for the split
#             cov_dataset_data.to_csv(os.path.join(fold_directory, f'{dataset_name}_data.cov'), sep=",", index=False)
            
             
#             # Quality control on training data only
#             if dataset_name == 'train':  # Apply QC only to the training data
#                 new_dataset_file_name = f"{dataset_name}_data.QC"

#                 # First PLINK command for quality control
#                 plink_command_1 = [
#                     './plink',
#                     '--bfile', os.path.join(fold_directory, dataset_file_name),
#                     '--maf', '0.01',
#                     '--hwe', '1e-6',
#                     '--geno', '0.1',
#                     '--mind', '0.1',
#                     '--write-snplist',
#                     '--make-just-fam',
#                     '--out', os.path.join(fold_directory, new_dataset_file_name)
#                 ]
#                 subprocess.run(plink_command_1, check=True)  # Use check=True to raise an error if the command fails

#                 # Second PLINK command for further pruning and processing
#                 plink_command_2 = [
#                     './plink',
#                     '--bfile', os.path.join(fold_directory, dataset_file_name),
#                     '--extract', os.path.join(fold_directory, new_dataset_file_name + '.snplist'),
#                     '--keep', os.path.join(fold_directory, new_dataset_file_name + '.fam'),
#                     '--het',
#                     '--out', os.path.join(fold_directory, new_dataset_file_name)
#                 ]
#                 subprocess.run(plink_command_2, check=True)

#                 # Invoke R script for additional processing
#                 os.system(f"Rscript Module1.R {os.path.join(fold_directory)} {dataset_file_name} {new_dataset_file_name} 1")
#                 print(f"Rscript Module1.R {os.path.join(fold_directory)} {dataset_file_name} {new_dataset_file_name} 1")

#                 # Code for sex check: Skip if Chromosome X is not available
#                 plink_command_sex_check = [
#                     './plink',
#                     '--bfile', os.path.join(fold_directory, 'train_data'),
#                     '--extract', os.path.join(fold_directory, new_dataset_file_name + '.snplist'),
#                     '--keep', os.path.join(fold_directory, 'train_data.valid.sample'),
#                     '--check-sex',
#                     '--out', os.path.join(fold_directory, 'train_data.QC')
#                 ]
#                 # Uncomment below if you need to perform sex check
#                 # subprocess.run(plink_command_sex_check, check=True)
#                 # os.system(f"Rscript Module1.R {os.path.join(fold_directory)} {train_file_name} {new_train_file_name} 2")

#                 # Third PLINK command for relatedness check (removal of related samples)
#                 plink_command_3 = [
#                     './plink',
#                     '--bfile', os.path.join(fold_directory, dataset_file_name),
#                     '--extract', os.path.join(fold_directory, new_dataset_file_name + '.snplist'),
#                     '--rel-cutoff', '0.125',
#                     '--out', os.path.join(fold_directory, new_dataset_file_name)
#                 ]
#                 subprocess.run(plink_command_3, check=True)

#                 # Fourth PLINK command for removing mismatched samples based on relatedness
#                 plink_command_4 = [
#                     './plink',
#                     '--bfile', os.path.join(fold_directory, dataset_file_name),
#                     '--make-bed',
#                     '--keep', os.path.join(fold_directory, new_dataset_file_name + '.rel.id'),
#                     '--out', os.path.join(fold_directory, new_dataset_file_name),
#                     '--extract', os.path.join(fold_directory, new_dataset_file_name + '.snplist'),
#                     '--exclude', os.path.join(fold_directory, dataset_file_name + '.mismatch'),
#                     '--a1-allele', os.path.join(fold_directory, dataset_file_name + '.a1')
#                 ]
#                 subprocess.run(plink_command_4, check=True)


 

# import os

# # List of file names to check for existence
# files = [
#     "train_data.QC.bed",
#     "train_data.QC.bim",
#     "train_data.QC.fam",
#     "train_data.cov",
#     "test_data.bed",
#     "test_data.bim",
#     "test_data.fam",
#     "test_data.cov"
# ]

# # Directory where the files are expected to be found

# # Print the table header
# print("{:<20} {:<5} {:<5} {:<5} {:<5} {:<5}".format("File Name", "Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4"))

# # Loop through each file name in the list
# for file in files:
#     # Create a list to store the existence status for each fold
#     status = []
#     # Check for each fold from 0 to 4
#     for fold_number in range(5):
#         # Check if the file exists in the specified directory for the given fold
#         #print(os.path.join("./",filedirec, f"Fold_{fold_number}", file))
#         if os.path.exists(filedirec + os.sep + "Fold_" + str(fold_number) + os.sep + file):
#             status.append("yes")
#         else:
#             status.append("no")
    
#     # Print the file name and its status for each fold
#     print("{:<20} {:<5} {:<5} {:<5} {:<5} {:<5}".format(file, *status))


# We will have the following directories if everything works fine.
# 
# ```
# ├── Fold_0
# │   ├── test_data.bed
# │   ├── test_data.bim
# │   ├── test_data.cov
# │   ├── test_data.fam
# │   ├── test_data.log
# │   ├── train_data.a1
# │   ├── train_data.bed
# │   ├── train_data.bim
# │   ├── train_data.cov
# │   ├── train_data.fam
# │   ├── train_data.log
# │   ├── train_data.mismatch
# │   ├── train_data.QC.bed
# │   ├── train_data.QC.bim
# │   ├── train_data.QC.fam
# │   ├── train_data.QC.het
# │   ├── train_data.QC.log
# │   ├── train_data.QC.rel.id
# │   ├── train_data.QC.sexcheck
# │   ├── train_data.QC.snplist
# │   ├── train_data.QC.valid
# │   └── train_data.valid.sample
# ├── Fold_1
#  
# ├── Fold_2
#  
# ├── Fold_3
#  
# ├── Fold_4
# 
# ├── PeopleWithPhenotype.txt
# ├── SampleData1.bed
# ├── SampleData1.bim
# ├── SampleData1.cov
# ├── SampleData1.fam
# ├── SampleData1.gz
# ├── SampleData1.height
# ├── SampleData1_QC.bed
# ├── SampleData1_QC.bim
# ├── SampleData1_QC.cov
# ├── SampleData1_QC.fam
# └── SampleData1_QC.log
# ```

# ## Important Note
# 
# 1. Kindly ensure you have all the files required for the next step after the completion of this step.
# 2. It is better to pass the dataset on which quality controls have already been performed.
# 3. We considered genotype files for which chromosomes 1 to 22 are available, and sex information is present. If you want to use the function in the script to check the sex information, ensure the genotype file includes other chromosomes as well.
# 4. Go through the logs generated by the code if an error occurs. Even if no error occurs, it is always good to ensure that the log file does not produce any errors.
# 

# In[ ]:




