 


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
  
df = pd.read_csv(gwas_path,compression= "gzip",sep="\s+")

 
if "BETA" in df.columns.to_list():
    # For Binary Phenotypes.
    df["OR"] = np.exp(df["BETA"])
    df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'OR', 'INFO', 'MAF']]

else:
    # For Binary Phenotype.
    df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'OR', 'INFO', 'MAF']]



Numberofsamples = df["N"].mean()


column_mapping = {"CHR": "hg19chrc", "SNP": "snpid", "A1": "a1", "A2": "a2", "BP": "bp", "OR": "or", "P": "p"}
new_columns = ["hg19chrc", "snpid", "a1", "a2", "bp", "or", "p"]
transformed_df = df.rename(columns=column_mapping)[new_columns]
transformed_df['hg19chrc'] = transformed_df['hg19chrc'].apply(lambda x: "chr" + str(x))
 
  
transformed_df.to_csv(save_path + os.sep +filedirec+".AnnoPred",sep="\t",index=False)
print("Length of DataFrame!",len(transformed_df))
 

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

# Kindly note that the number of p-values to be considered varies, and the actual p-value depends on the dataset as well.
# We will specify the range list here.
#folddirec = "/path/to/your/folder"  # Replace with your actual folder path
from decimal import Decimal, getcontext
import numpy as np

# Set precision to a high value (e.g., 50)
getcontext().prec = 50
minimumpvalue = 10  # Minimum p-value in exponent
numberofintervals = 20  # Number of intervals to be considered
allpvalues = np.logspace(-minimumpvalue, 0, numberofintervals, endpoint=True)  # Generating an array of logarithmically spaced p-values
count = 1
 
pvaluefile = folddirec + os.sep + 'range_list'

# Initializing an empty DataFrame with specified column names
prs_result = pd.DataFrame(columns=["clump_p1", "clump_r2", "clump_kb", "p_window_size", "p_slide_size", "p_LD_threshold",
                                   "pvalue","datafile", "numberofpca","Train_pure_prs", "Train_null_model", "Train_best_model",
                                   "Test_pure_prs", "Test_null_model", "Test_best_model"])

 

import os
import subprocess
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import explained_variance_score
  


import os
import subprocess
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import MinMaxScaler

def transform_annopred_data(traindirec, newtrainfilename,numberofpca, tier,pvalue,p1_val, p2_val, p3_val, c1_val, c2_val, c3_val,Name,pvaluefile):
    import shutil
    import os

    def remove_all_in_directory(directory_path):
        if not os.path.exists(directory_path):
            print "The directory {} does not exist.".format(directory_path)
            return

        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)

            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print "Failed to remove {}. Reason: {}".format(item_path, e)

        print "All files and directories in {} have been removed.".format(directory_path)

 
    os.system("awk "+"\'"+"{print $3,$8}"+"\'"+" ./"+save_path+os.sep+filedirec+".txt >  ./"+traindirec+os.sep+"SNP.pvalue")

    # Remove the all files in the specific directory.
    remove_all_in_directory(traindirec+os.sep+"AnnoPred_test_output")
    remove_all_in_directory(traindirec+os.sep+"AnnoPred_tmp_test") 

    create_directory(traindirec+os.sep+"AnnoPred_test_output")
    create_directory(traindirec+os.sep+"AnnoPred_tmp_test")
 

    ## AnnoPred overrides the file in the ref directory
    ## So, we file calculate the hertiability using LDSC and calculate LDSC_file for each tire tire0_ldsc_results
    ## And passed to AnnoPred.
    
    munge_command = [
        './munge_sumstats.py',
        '--out', traindirec+os.sep+"AnnoPred_tmp_test"+os.sep+"Curated_GWAS",
        '--merge-alleles', '/data/ascher01/uqmmune1/BenchmarkingPGSTools/ref/Misc/w_hm3.snplist',
        '--N', str(Numberofsamples),
        '--sumstats', save_path+os.sep+filedirec+'.AnnoPred'
    ]
    
    subprocess.call(munge_command)
    # Step 2: Run ldsc.py
    ldsc_command = [
        './ldsc.py',
        '--h2', traindirec+os.sep+"AnnoPred_tmp_test"+os.sep+"Curated_GWAS.sumstats.gz",
        '--ref-ld-chr', '/data/ascher01/uqmmune1/BenchmarkingPGSTools/ref/Annotations/Baseline/baseline.,'
                       '/data/ascher01/uqmmune1/BenchmarkingPGSTools/ref/Annotations/GenoCanyon/GenoCanyon_Func.,'
                       '/data/ascher01/uqmmune1/BenchmarkingPGSTools/ref/Annotations/GenoSkyline/Brain.,'
                       '/data/ascher01/uqmmune1/BenchmarkingPGSTools/ref/Annotations/GenoSkyline/GI.,'
                       '/data/ascher01/uqmmune1/BenchmarkingPGSTools/ref/Annotations/GenoSkyline/Lung.,'
                       '/data/ascher01/uqmmune1/BenchmarkingPGSTools/ref/Annotations/GenoSkyline/Heart.,'
                       '/data/ascher01/uqmmune1/BenchmarkingPGSTools/ref/Annotations/GenoSkyline/Blood.,'
                       '/data/ascher01/uqmmune1/BenchmarkingPGSTools/ref/Annotations/GenoSkyline/Muscle.,'
                       '/data/ascher01/uqmmune1/BenchmarkingPGSTools/ref/Annotations/GenoSkyline/Epithelial.',
        '--out', traindirec+os.sep+'AnnoPred_tmp_test/tier0_ldsc',
        '--overlap-annot',
        # This is the AnnoPred reference set
        '--frqfile-chr', '/data/ascher01/uqmmune1/BenchmarkingPGSTools/ref/Misc/1000G.mac5eur.',
        '--w-ld-chr', '/data/ascher01/uqmmune1/BenchmarkingPGSTools/ref/Misc/weights.',
        '--print-coefficients'
    ]
    subprocess.call(ldsc_command)    
    
    
 
    
    command = [
        "python",
        "AnnoPred.py",
        "--sumstats",save_path + os.sep + filedirec+".AnnoPred",
        "--ref_gt",traindirec+os.sep+newtrainfilename+".clumped.pruned",
        "--val_gt",traindirec+os.sep+newtrainfilename+".clumped.pruned",
        "--coord_out",traindirec+os.sep+"AnnoPred_test_output"+os.sep+"coord_out",
    
        "--N_sample",str(int(Numberofsamples)),
        "--annotation_flag",tier,
        "--P",str(pvalue),
        "--local_ld_prefix",traindirec+os.sep+"AnnoPred_tmp_test"+os.sep+"local_ld",
        "--out",traindirec+os.sep+"AnnoPred_test_output"+os.sep+"test",
        "--temp_dir",traindirec+os.sep+"AnnoPred_tmp_test"
    ]
    print(" ".join(command))
    subprocess.call(command)        
 
    
    data1 = traindirec+os.sep+"AnnoPred_test_output"+os.sep+"test_h2_inf_betas_"+str(pvalue)+".txt"
    data2 = traindirec+os.sep+"AnnoPred_test_output"+os.sep+"test_h2_non_inf_betas_"+str(pvalue)+".txt"
    data3 = traindirec+os.sep+"AnnoPred_test_output"+os.sep+"test_pT_inf_betas_"+str(pvalue)+".txt"
    data4 = traindirec+os.sep+"AnnoPred_test_output"+os.sep+"test_pT_non_inf_betas_"+str(pvalue)+".txt"
 
    
    datafiles = [data1,data2,data3,data4]
    for datafile in datafiles: 
        # Calculate Plink Score.
        try:
            tempgwas = pd.read_csv(traindirec+os.sep+"AnnoPred_test_output"+os.sep+"test_h2_inf_betas_"+str(pvalue)+".txt",sep="\s+" )
        except:
            print("GWAS not generated!")
            return
        
        
        if check_phenotype_is_binary_or_continous(filedirec)=="Binary":
            tempgwas["AnnoPred_inf_beta"] = np.exp(tempgwas["AnnoPred_inf_beta"])
        else:
             pass
            

        tempgwas = tempgwas.rename(columns={"sid": "SNP", "nt1": "A1", "AnnoPred_inf_beta": "BETA"})
        tempgwas[["SNP","A1","BETA"]].to_csv(traindirec+os.sep+"AnnoPred_GWAS.txt",sep="\t",index=False)        
        
        
        command = [
            "./plink",
             "--bfile", traindirec+os.sep+newtrainfilename+".clumped.pruned",
            ### SNP column = 3, Effect allele column 1 = 4, OR column=9
            "--score", traindirec+os.sep+"AnnoPred_GWAS.txt", "1", "2", "3", "header",
            "--q-score-range", traindirec+os.sep+"range_list",traindirec+os.sep+"SNP.pvalue",
            "--extract", traindirec+os.sep+trainfilename+".valid.snp",
            "--out", Name+os.sep+trainfilename
        ]
 
        subprocess.call(command)

        command = [
            "./plink",
            "--bfile", folddirec+os.sep+testfilename+".clumped.pruned",
            ### SNP column = 3, Effect allele column 1 = 4, OR column=9
            "--score", traindirec+os.sep+"AnnoPred_GWAS.txt", "1", "2", "3", "header",
            "--q-score-range", traindirec+os.sep+"range_list",traindirec+os.sep+"SNP.pvalue",
            "--extract", traindirec+os.sep+trainfilename+".valid.snp",
            "--out", Name+os.sep+testfilename
        ]
        subprocess.call(command)

        command = [
            "./plink",
            "--bfile", folddirec+os.sep+validationfilename+".clumped.pruned",
            ### SNP column = 3, Effect allele column 1 = 4, OR column=9
            "--score", traindirec+os.sep+"AnnoPred_GWAS.txt", "1", "2", "3", "header",
            "--q-score-range", traindirec+os.sep+"range_list",traindirec+os.sep+"SNP.pvalue",
            "--extract", traindirec+os.sep+trainfilename+".valid.snp",
            "--out", Name+os.sep+validationfilename
        ]
        subprocess.call(command)
     
   
    
# AnnoPred offers 4 tires of calculating the P 
# tier0: baseline + GenoCanyon + 7 GenoSkyline (Brain, GI, Lung, Heart, Blood, Muscle, Epithelial)
# tier1: baseline + GenoCanyon
# tier2: baseline + GenoCanyon + 7 GenoSkyline_Plus (Immune, Brain, CV, Muscle, GI, Epithelial)
# tier3: baseline + GenoCanyon + 66 GenoSkyline
 

tires = ['tier0','tier1','tier2','tier3']
tires = ['tier0']
tempallpvalues = [allpvalues[-1]]
result_directory = "AnnoPred"
# Nested loops to iterate over different parameter values
create_directory(folddirec+os.sep+result_directory)
result_directory = folddirec+os.sep+result_directory

for p1_val in p_window_size:
 for p2_val in p_slide_size:
  for p3_val in p_LD_threshold:
   for c1_val in clump_p1:
    for c2_val in clump_r2:
     for c3_val in clump_kb:
      for p in numberofpca:
       for t in tires:
        for pvalue in tempallpvalues:
         transform_annopred_data(folddirec, newtrainfilename, p,t,pvalue, str(p1_val), str(p2_val), str(p3_val), str(c1_val), str(c2_val), str(c3_val), result_directory, pvaluefile)

 