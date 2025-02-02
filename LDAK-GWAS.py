
import os
import pandas as pd
import numpy as np
import sys


filedirec = sys.argv[1]
gwas_file = sys.argv[2]

if "." in gwas_file:
    gwasfilewithoutextension = gwas_file.split(".")[0]

save_path = filedirec + os.sep + gwasfilewithoutextension
foldnumber = sys.argv[3]

gwas_path = save_path + os.sep + filedirec+".gz"

def check_phenotype_is_binary_or_continous(filedirec):
    # Read the processed quality controlled file for a phenotype
    df = pd.read_csv(save_path+os.sep+filedirec+'_QC.fam',sep="\s+",header=None)
    column_values = df[5].unique()
 
    if len(set(column_values)) == 2:
        return "Binary"
    else:
        return "Continous"

print("Phenotype",sys.argv[1])
print("Gwas File",sys.argv[2])
print("Fold Number",sys.argv[3])
print("Save Path",save_path)
print("GWAS Path",gwas_path)

if ".gz"in gwas_file:
    df = pd.read_csv(gwas_path,compression= "gzip",sep="\s+")
else:
    df = pd.read_csv(gwas_path,sep="\s+")

 

if "BETA" in df.columns.to_list():
    # For Continous Phenotype.
    df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]
else:
    df["BETA"] = np.log(df["OR"])
    df["SE"] = df["SE"]/df["OR"]
    df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]
    
print(df.head().to_markdown()) 

df_transformed = pd.DataFrame({
    'Predictor': df['CHR'].astype(str) + ":" + df['BP'].astype(str),
    'A1': df['A1'],
    'A2': df['A2'],
    'n': df['N'],
    'Z': df['BETA']/df['SE'],
    'SNP':df['SNP']
}) 

df.to_csv(save_path + os.sep +filedirec+".txt",sep="\t",index=False)

 
# Remove SNPs where the number of alleles are more than 1
df_transformed = df_transformed[df_transformed['A1'].apply(len) == 1]
df_transformed = df_transformed[df_transformed['A2'].apply(len) == 1]
df_transformed = df_transformed.drop_duplicates(subset=['Predictor'], keep='first')

# Optionally, reset index
df_transformed.reset_index(drop=True, inplace=True)
 
df_transformed.to_csv(save_path + os.sep +filedirec+".ldak",sep="\t",index=False)  

from operator import index
import pandas as pd
import numpy as np
import os
import subprocess
import sys
import pandas as pd
import statsmodels.api as sm
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

def create_directory(directory):
    """Function to create a directory if it doesn't exist."""
    if not os.path.exists(directory):  # Checking if the directory doesn't exist
        os.makedirs(directory)  # Creating the directory if it doesn't exist
    return directory  # Returning the created or existing directory
 
#foldnumber = "0"  # Setting 'foldnumber' to "0"

folddirec = save_path + os.sep + "Fold_" + foldnumber  # Creating a directory path for the specific fold
trainfilename = "train_data"  # Setting the name of the training data file
newtrainfilename = "train_data.QC"  # Setting the name of the new training data file
validationfilename = "validation_data"
newvalidationfilename  =  "validation_data.QC"

testfilename = "test_data"  # Setting the name of the test data file
newtestfilename = "test_data.QC"  # Setting the name of the new test data file

# Number of PCA to be included as a covariate.
numberofpca = ["6"]  # Setting the number of PCA components to be included

# Clumping parameters.
clump_p1 = [1]  # List containing clump parameter 'p1'
clump_r2 = [0.1]  # List containing clump parameter 'r2'
clump_kb = [200]  # List containing clump parameter 'kb'

# Pruning parameters.
p_window_size = [200]  # List containing pruning parameter 'window_size'
p_slide_size = [50]  # List containing pruning parameter 'slide_size'
p_LD_threshold = [0.25]  # List containing pruning parameter 'LD_threshold'
 
pvaluefile = folddirec + os.sep + 'range_list'

# Initializing an empty DataFrame with specified column names
prs_result = pd.DataFrame(columns=["clump_p1", "clump_r2", "clump_kb", "p_window_size", "p_slide_size", "p_LD_threshold",
                                   "pvalue", "numberofpca","numberofvariants","Train_pure_prs", "Train_null_model", "Train_best_model",
                                   "Test_pure_prs", "Test_null_model", "Test_best_model"])
 

import os
import subprocess
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import explained_variance_score

 
 
# Define a global variable to store results
prs_result = pd.DataFrame()
def transform_ldak_data(traindirec, newtrainfilename,numberofpca,ldakmodel,power, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val,Name,pvaluefile):     
    
    os.system("awk "+"\'"+"{print $3,$8}"+"\'"+" ./"+save_path+os.sep+filedirec+".txt >  ./"+traindirec+os.sep+"SNP.pvalue")
    files_to_remove = [
        traindirec + os.sep + ldakmodel +gwas_file+"_ldak_gwas_final",
        traindirec+os.sep+ldakmodel+gwas_file+".effects",
    ]

    # Loop through the files and remove them if they exist
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")
        else:
            print(f"File does not exist: {file_path}")
            
            
    
    command1 = [
        './ldak',
        '--cut-genes', 'highld',
        '--bfile', traindirec+os.sep+newtrainfilename+".clumped.pruned",
        '--genefile', 'highld.txt'
    ]
    subprocess.run(command1)
    
    command2 = [
        './ldak',
        '--calc-cors',traindirec+os.sep+'cors',
        '--bfile', traindirec+os.sep+newtrainfilename+".clumped.pruned"
    ]
    subprocess.run(command2)
    print(" ".join(command2))
    
    
    # Here we need to update the cors.bim file and it should be the same as the GWAS file.
    # LDAK accepts gwas in a specific format and the LDAK
    df = pd.read_csv(save_path + os.sep +filedirec+".ldak",sep="\s+")
    t1 = pd.read_csv(traindirec+os.sep+'cors.cors.bim',sep="\s+",header=None)
    t1[1] = t1[0].astype(str)+":"+t1[3].astype(str)
    t1.to_csv(traindirec+os.sep+'cors.cors.bim',sep="\t",header=False,index=None) 
    
    
    command3 = [
        './ldak',
        '--mega-prs', traindirec+os.sep+ldakmodel+filedirec,
        '--model', ldakmodel,
        '--summary', save_path + os.sep +filedirec+".ldak",
        '--power', str(power),
        '--skip-cv','YES',
        '--cors',  traindirec+os.sep+'cors',
        #'--check-high-LD', 'NO',
        #'--high-LD', 'highld/genes.predictors.used',
        '--allow-ambiguous', 'YES',
        
    ]
    subprocess.run(command3)
    #exit(0)
    
    # Read the original gwas
    df = pd.read_csv(save_path + os.sep +filedirec+".ldak",sep="\s+")
    # Read the effect size for each SNP generated by LDAK.
    
    df2 = pd.read_csv(traindirec+os.sep+ldakmodel+filedirec+".effects",sep="\s+")
    # Get the SNP information as it is required by Plink, because the orginal data have SNPs like rsXXX and 
    # Effects generated by LDAK have SNPs in 2:16937 CHR:POSITION format.
    
    df = df[df["Predictor"].isin(df2["Predictor"].values)]
    df2["SNP"] = df["SNP"].values
    df2.to_csv(traindirec+os.sep+ldakmodel+".effects",index=False,sep="\t")    
    
    numberofcolumns1 = pd.read_csv(traindirec+os.sep+ldakmodel+filedirec+".effects",sep="\s+").shape[1]
    print(numberofcolumns1)
    
    numberofcolumns = numberofcolumns1 - 4

    # Read the effect size.
    # It contains the effect sizes from multiple model.
    temp = pd.read_csv(traindirec + os.sep + ldakmodel + ".effects", sep="\s+")
    # Loop through effect sizes, modify values for binary phenotypes, and save specific columns
    for loop in range(4, numberofcolumns1 - 1):
        # If phenotype is binary, apply the exponential transformation
        if check_phenotype_is_binary_or_continous(filedirec) == "Binary":
            temp.iloc[:, loop] = np.exp(temp.iloc[:, loop])
        else:
            pass  
        
        # Save last, second, and current loop column in specified order
        ordered_columns = [temp.columns[-1], temp.columns[1], temp.columns[loop]]
        temp[ordered_columns].to_csv(
            traindirec + os.sep + ldakmodel +filedirec+"_ldak_gwas_final", 
            sep="\t", 
            index=False
        )
        print(temp.head())
        print(temp[ordered_columns].head())
       
         
        command = [
            "./plink",
            "--bfile", traindirec+os.sep+newtrainfilename,
            ### SNP column = 3, Effect allele column 1 = 4, OR column=9
            "--score", traindirec + os.sep + ldakmodel +filedirec+"_ldak_gwas_final", "1", "2", "3", "header",  
            "--q-score-range", traindirec+os.sep+"range_list",traindirec+os.sep+"SNP.pvalue",
            "--extract", traindirec+os.sep+trainfilename+".valid.snp",
            "--out", Name+os.sep+trainfilename
        ]
        #exit(0)
        subprocess.run(command)
        
        # Calculate the PRS for the test data using the same set of SNPs and also calculate the PCA.
 

        command = [
            "./plink",
            "--bfile", folddirec+os.sep+testfilename,
            ### SNP column = 3, Effect allele column 1 = 4, OR column=9
            "--score", traindirec + os.sep + ldakmodel +filedirec+"_ldak_gwas_final", "1", "2", "3", "header",
            "--q-score-range", traindirec+os.sep+"range_list",traindirec+os.sep+"SNP.pvalue",
            "--extract", traindirec+os.sep+trainfilename+".valid.snp",
            "--out", Name+os.sep+testfilename
        ]
        subprocess.run(command)
 
        command = [
            "./plink",
            "--bfile", folddirec+os.sep+validationfilename,
            ### SNP column = 3, Effect allele column 1 = 4, OR column=9
            "--score", traindirec + os.sep + ldakmodel +filedirec+"_ldak_gwas_final", "1", "2", "3", "header",
            "--q-score-range", traindirec+os.sep+"range_list",traindirec+os.sep+"SNP.pvalue",
            "--extract", traindirec+os.sep+trainfilename+".valid.snp",
            "--out", Name+os.sep+validationfilename
        ]
        subprocess.run(command)

powers = [-0.25]
 
ldakmodels =["lasso","lasso-sparse","ridge","bolt","bayesr","elastic"]
#ldakmodels =["lasso-sparse","ridge","bolt","bayesr","elastic"] 
ldakmodels =["lasso" ] 
 
result_directory = "LDAK-GWAS"
create_directory(folddirec+os.sep+result_directory)
result_directory = folddirec+os.sep+result_directory


for p1_val in p_window_size:
 for p2_val in p_slide_size: 
  for p3_val in p_LD_threshold:
   for c1_val in clump_p1:
    for c2_val in clump_r2:
     for c3_val in clump_kb:
      for p in numberofpca:
       for ldakmodel in ldakmodels:
        for power in powers: 
         transform_ldak_data(folddirec, newtrainfilename, p,ldakmodel,power,str(p1_val), str(p2_val), str(p3_val), str(c1_val), str(c2_val), str(c3_val),result_directory, pvaluefile)




