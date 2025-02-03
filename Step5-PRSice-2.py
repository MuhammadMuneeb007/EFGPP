

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
    df = pd.read_csv(save_path+os.sep+filedirec+'_QC.fam', sep="\s+", header=None)
    column_values = df[5].unique()
    if len(set(column_values)) == 2:
        return "Binary"
    else:
        return "Continous"

 
if ".gz"in gwas_file:
    df = pd.read_csv(gwas_path,compression= "gzip",sep="\s+")
else:
    df = pd.read_csv(gwas_path,sep="\s+")




print("Phenotype",sys.argv[1])
print("Gwas File",sys.argv[2])
print("Fold Number",sys.argv[3])
print("Save Path",save_path)
print("GWAS Path",gwas_path)


if check_phenotype_is_binary_or_continous(filedirec)=="Binary":
    if "BETA" in df.columns.to_list():
        df["OR"] = np.exp(df["BETA"])
        df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'OR', 'INFO', 'MAF']]
    else:
        df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'OR', 'INFO', 'MAF']]
elif check_phenotype_is_binary_or_continous(filedirec)=="Continous":
    if "BETA" in df.columns.to_list():
        df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]
    else:
        df["BETA"] = np.log(df["OR"])
        df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]


df['BP'] = df['BP'].astype(int)
df.to_csv(save_path + os.sep +filedirec+".txt",sep="\t",index=False)
df.to_csv(save_path + os.sep +filedirec+"_PRSice-2.txt",sep="\t",index=False)

print(df.head().to_markdown())
print("Length of DataFrame!",len(df))



print("Length of DataFrame!", len(df))

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
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory
 
folddirec = save_path + os.sep + "Fold_" + foldnumber
trainfilename = "train_data"
newtrainfilename = "train_data.QC"
validationfilename = "validation_data"
newvalidationfilename  =  "validation_data.QC"
testfilename = "test_data"
newtestfilename = "test_data.QC"


numberofpca = ["6"]
clump_p1 = [1]
clump_r2 = [0.1]
clump_kb = [200]
p_window_size = [200]
p_slide_size = [50]
p_LD_threshold = [0.25]


minimumpvalue = str(1e-5)
maximumpvalue = str(1)
interval = str(0.05)
pvalues = np.arange(float(minimumpvalue),float(maximumpvalue),float(interval))


pvaluefile = folddirec + os.sep + 'range_list'

 
import os
import subprocess
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import explained_variance_score
  
 
prs_result = pd.DataFrame()
def transform_prsice_data(traindirec, newtrainfilename,prsicemodel,numberofpca, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val,Name,pvaluefile):
    #perform_clumping_and_pruning_on_individual_data(traindirec, newtrainfilename,p, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val,Name,pvaluefile)
    #calculate_pca_for_traindata_testdata_for_clumped_pruned_snps(traindirec, newtrainfilename,p)

     
    os.system("awk "+"\'"+"{print $3,$8}"+"\'"+" ./"+save_path+os.sep+filedirec+".txt >  ./"+traindirec+os.sep+"SNP.pvalue")
     
    tempphenotype_train = pd.read_table(traindirec+os.sep+newtrainfilename+".clumped.pruned"+".fam", sep="\s+",header=None)
    phenotype = pd.DataFrame()
    phenotype = tempphenotype_train[[0,1,5]]
    phenotype.to_csv(traindirec+os.sep+trainfilename+".PHENO",sep="\t",header=['FID', 'IID', 'PHENO'],index=False)
    print(phenotype.head())
    file_path = traindirec + os.sep + trainfilename + ".eigenvec"
    # Read the file to infer the number of columns
    temp_df = pd.read_csv(file_path, delim_whitespace=True, header=None)
    # Create column names: "FID", "IID", and "PC1", "PC2", ..., "PCN"
    column_names = ["FID", "IID"] + [f"PC{i}" for i in range(1, temp_df.shape[1] - 2 + 1)]
    # Reload the file with proper column names
    pcs_train = pd.read_csv(file_path, delim_whitespace=True, header=None, names=column_names)


    covariate_train = pd.read_table(traindirec+os.sep+trainfilename+".cov",sep="\s+")

    covariate_train.fillna(0, inplace=True)
    covariate_train = covariate_train[covariate_train["FID"].isin(pcs_train["FID"].values) & covariate_train["IID"].isin(pcs_train["IID"].values)]

    covariate_train['FID'] = covariate_train['FID'].astype(str)
    pcs_train['FID'] = pcs_train['FID'].astype(str)
    covariate_train['IID'] = covariate_train['IID'].astype(str)
    pcs_train['IID'] = pcs_train['IID'].astype(str)
    covandpcs_train = pd.merge(covariate_train, pcs_train, on=["FID","IID"])
    covandpcs_train.to_csv(traindirec+os.sep+trainfilename+".COV_PCA",sep="\t",index=False)
    

    binary_phenotype = ""
    stat_col = ""
    if check_phenotype_is_binary_or_continous(filedirec)=="Binary":
        binary_phenotype = 'T'
        stat_col="OR"
    else:
        binary_phenotype = 'F'
        stat_col="BETA"
    
    command = [
    './PRSice',
    '--base', save_path + os.sep +filedirec+"_PRSice-2.txt",
    '--target', traindirec+os.sep+newtrainfilename+".clumped.pruned",
    '--pheno', traindirec+os.sep+trainfilename+".PHENO",
    '--cov', traindirec+os.sep+trainfilename+".COV_PCA",
    '--stat', stat_col,
    '--ld','ref',
    '--all-score',
    '--lower', minimumpvalue, 
    '--upper',maximumpvalue,
    '--interval',interval,
    '--score',prsicemodel,
    '--binary-target', binary_phenotype,
    '--no-clump',
    '--out', Name+os.sep+p+"_"+prsicemodel+"_Train_PRSice_PRS_"+p1_val+"_"+p2_val+"_"+p3_val+"_"+c1_val+"_"+c2_val+"_"+c3_val
    ]
    subprocess.run(command)
    print(" ".join(command))
    print("Train PRSice Done!")

    tempphenotype_test = pd.read_table(traindirec+os.sep+testfilename+".clumped.pruned"+".fam", sep="\s+",header=None)
    phenotype = pd.DataFrame()
    phenotype = tempphenotype_test[[0,1,5]]
    phenotype.to_csv(traindirec+os.sep+testfilename+".PHENO",sep="\t",header=['FID', 'IID', 'PHENO'],index=False)
    
    file_path = traindirec + os.sep + testfilename + ".eigenvec"
    # Read the file to infer the number of columns
    temp_df = pd.read_csv(file_path, delim_whitespace=True, header=None)
    # Create column names: "FID", "IID", and "PC1", "PC2", ..., "PCN"
    column_names = ["FID", "IID"] + [f"PC{i}" for i in range(1, temp_df.shape[1] - 2 + 1)]
    # Reload the file with proper column names
    pcs_test = pd.read_csv(file_path, delim_whitespace=True, header=None, names=column_names)

    covariate_test = pd.read_table(traindirec+os.sep+testfilename+".cov",sep="\s+")
    covariate_test.fillna(0, inplace=True)
    covariate_test = covariate_test[covariate_test["FID"].isin(pcs_test["FID"].values) & covariate_test["IID"].isin(pcs_test["IID"].values)]
    covariate_test['FID'] = covariate_test['FID'].astype(str)
    pcs_test['FID'] = pcs_test['FID'].astype(str)
    covariate_test['IID'] = covariate_test['IID'].astype(str)
    pcs_test['IID'] = pcs_test['IID'].astype(str)
    covandpcs_test = pd.merge(covariate_test, pcs_test, on=["FID","IID"])
    covandpcs_test.to_csv(traindirec+os.sep+testfilename+".COV_PCA",sep="\t",index=False)
    command = [
    './PRSice',
    '--base', save_path + os.sep +filedirec+"_PRSice-2.txt",
    '--target', traindirec+os.sep+testfilename+".clumped.pruned",
    '--pheno', traindirec+os.sep+testfilename+".PHENO",
    '--cov', traindirec+os.sep+testfilename+".COV_PCA",
    '--stat', stat_col,
    '--ld','ref',
    '--all-score',
    '--lower', minimumpvalue, 
    '--upper',maximumpvalue,
    '--interval',interval,
    '--score',prsicemodel,
    '--binary-target', binary_phenotype,
    '--no-clump',
    '--out', Name+os.sep+p+"_"+prsicemodel+"_Test_PRSice_PRS_"+p1_val+"_"+p2_val+"_"+p3_val+"_"+c1_val+"_"+c2_val+"_"+c3_val
    ]
    subprocess.run(command)

    tempphenotype_validation = pd.read_table(traindirec+os.sep+validationfilename+".clumped.pruned"+".fam", sep="\s+",header=None)
    phenotype = pd.DataFrame()
    phenotype = tempphenotype_validation[[0,1,5]]
    phenotype.to_csv(traindirec+os.sep+validationfilename+".PHENO",sep="\t",header=['FID', 'IID', 'PHENO'],index=False)
    
    file_path = traindirec + os.sep + validationfilename + ".eigenvec"
    # Read the file to infer the number of columns
    temp_df = pd.read_csv(file_path, delim_whitespace=True, header=None)
    # Create column names: "FID", "IID", and "PC1", "PC2", ..., "PCN"
    column_names = ["FID", "IID"] + [f"PC{i}" for i in range(1, temp_df.shape[1] - 2 + 1)]
    # Reload the file with proper column names
    pcs_validation = pd.read_csv(file_path, delim_whitespace=True, header=None, names=column_names)

    
    covariate_validation = pd.read_table(traindirec+os.sep+validationfilename+".cov",sep="\s+")
    covariate_validation.fillna(0, inplace=True)
    covariate_validation = covariate_validation[covariate_validation["FID"].isin(pcs_validation["FID"].values) & covariate_validation["IID"].isin(pcs_validation["IID"].values)]
    covariate_validation['FID'] = covariate_validation['FID'].astype(str)
    pcs_validation['FID'] = pcs_validation['FID'].astype(str)
    covariate_validation['IID'] = covariate_validation['IID'].astype(str)
    pcs_validation['IID'] = pcs_validation['IID'].astype(str)
    covandpcs_validation = pd.merge(covariate_validation, pcs_validation, on=["FID","IID"])
    covandpcs_validation.to_csv(traindirec+os.sep+validationfilename+".COV_PCA",sep="\t",index=False)
    command = [
    './PRSice',
    '--base', save_path + os.sep +filedirec+"_PRSice-2.txt",
    '--target', traindirec+os.sep+validationfilename+".clumped.pruned",
    '--pheno', traindirec+os.sep+validationfilename+".PHENO",
    '--cov', traindirec+os.sep+validationfilename+".COV_PCA",
    '--stat', stat_col,
    '--ld','ref',
    '--all-score',
    '--lower', minimumpvalue, 
    '--upper',maximumpvalue,
    '--interval',interval,
    '--score',prsicemodel,
    '--binary-target', binary_phenotype,
    '--no-clump',
    '--out', Name+os.sep+p+"_"+prsicemodel+"_Validation_PRSice_PRS_"+p1_val+"_"+p2_val+"_"+p3_val+"_"+c1_val+"_"+c2_val+"_"+c3_val
    ]
    subprocess.run(command)




result_directory = "PRSice-2"
create_directory(folddirec+os.sep+result_directory)
result_directory = folddirec+os.sep+result_directory 

PRSiceModels =  ['avg']
for p1_val in p_window_size:
 for p2_val in p_slide_size: 
  for p3_val in p_LD_threshold:
   for c1_val in clump_p1:
    for c2_val in clump_r2:
     for c3_val in clump_kb:
      for p in numberofpca:
       for prsicemodel in  PRSiceModels:
        transform_prsice_data(folddirec, newtrainfilename,prsicemodel, p, str(p1_val), str(p2_val), str(p3_val), str(c1_val), str(c2_val), str(c3_val), result_directory, pvaluefile)
