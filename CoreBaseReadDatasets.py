import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
#import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def compute_pca(X, n_components=2):
    """Compute PCA and return results with explained variance"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)
    return pca_result, pca.explained_variance_ratio_

def compute_tsne(X, n_components=2):
    """Compute t-SNE and return results with KL divergence"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(X_scaled)
    return tsne_result, tsne.kl_divergence_

def compute_mds(X, n_components=2):
    """Compute MDS and return results with stress"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mds = MDS(n_components=n_components, random_state=42)
    mds_result = mds.fit_transform(X_scaled)
    return mds_result, mds.stress_

def compute_umap(X, n_components=2):
    """Compute UMAP and return results"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    umap_result = reducer.fit_transform(X_scaled)
    return umap_result

def create_scatter(ax, data, y_train, title):
    """Create scatter plot with consistent styling"""
    scatter = ax.scatter(data[:, 0], data[:, 1], c=y_train, cmap='viridis', alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('First Component')
    ax.set_ylabel('Second Component')
    ax.grid(True, linestyle='--', alpha=0.7)
    return scatter

def visualize_dimensionality_reductions(X_train1, X_train2, y_train, output_file="dimensionality_reduction_analysis.png"):
    """
    Create comprehensive visualization of different dimensionality reduction techniques
    
    Parameters:
    -----------
    X_train1 : array-like
        First dataset (with weights)
    X_train2 : array-like
        Second dataset (without weights)
    y_train : array-like
        Labels for the datasets
    output_file : str
        Name of the output file to save the visualization
    """
    
    # Create a figure with 3x2 subplots
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(3, 2)

    # Compute all dimensionality reductions
    pca1, var_ratio1 = compute_pca(X_train1)
    pca2, var_ratio2 = compute_pca(X_train2)
    
    mds1, stress1 = compute_mds(X_train1)
    mds2, stress2 = compute_mds(X_train2)
    
    tsne1, kl_div1 = compute_tsne(X_train1)
    tsne2, kl_div2 = compute_tsne(X_train2)
    
    umap1 = compute_umap(X_train1)
    umap2 = compute_umap(X_train2)

    # Plot PCA
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = create_scatter(ax1, pca1, y_train, 
                            f'PCA - Dataset 1 (with weights)\nExplained Variance: PC1={var_ratio1[0]:.2%}, PC2={var_ratio1[1]:.2%}')
    plt.colorbar(scatter1, ax=ax1, label='Class')

    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = create_scatter(ax2, pca2, y_train,
                            f'PCA - Dataset 2 (without weights)\nExplained Variance: PC1={var_ratio2[0]:.2%}, PC2={var_ratio2[1]:.2%}')
    plt.colorbar(scatter2, ax=ax2, label='Class')

    # Plot t-SNE
    ax3 = fig.add_subplot(gs[1, 0])
    scatter3 = create_scatter(ax3, tsne1, y_train,
                            f't-SNE - Dataset 1 (with weights)\nKL Divergence: {kl_div1:.4f}')
    plt.colorbar(scatter3, ax=ax3, label='Class')

    ax4 = fig.add_subplot(gs[1, 1])
    scatter4 = create_scatter(ax4, tsne2, y_train,
                            f't-SNE - Dataset 2 (without weights)\nKL Divergence: {kl_div2:.4f}')
    plt.colorbar(scatter4, ax=ax4, label='Class')

    # Plot UMAP
    ax5 = fig.add_subplot(gs[2, 0])
    scatter5 = create_scatter(ax5, umap1, y_train,
                            f'UMAP - Dataset 1 (with weights)')
    plt.colorbar(scatter5, ax=ax5, label='Class')

    ax6 = fig.add_subplot(gs[2, 1])
    scatter6 = create_scatter(ax6, umap2, y_train,
                            f'UMAP - Dataset 2 (without weights)')
    plt.colorbar(scatter6, ax=ax6, label='Class')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    # Create combined distribution plot
    plt.figure(figsize=(15, 10))
    
    # Create a DataFrame with all first components
    distribution_data = []
    methods = {
        'PCA': (pca1, pca2),
        'MDS': (mds1, mds2),
        't-SNE': (tsne1, tsne2),
        'UMAP': (umap1, umap2)
    }

    for method, (data1, data2) in methods.items():
        # Dataset 1 (with weights)
        dist_df1 = pd.DataFrame({
            'Component': data1[:, 0],
            'Class': y_train,
            'Method': method,
            'Dataset': 'With Weights'
        })
        # Dataset 2 (without weights)
        dist_df2 = pd.DataFrame({
            'Component': data2[:, 0],
            'Class': y_train,
            'Method': method,
            'Dataset': 'Without Weights'
        })
        distribution_data.append(dist_df1)
        distribution_data.append(dist_df2)
    
    combined_df = pd.concat(distribution_data)
    
    # Create faceted plot
    g = sns.FacetGrid(combined_df, col='Method', row='Dataset', hue='Class', height=4, aspect=1.5)
    g.map(sns.kdeplot, 'Component', alpha=.6)
    g.add_legend(title='Class')
    g.fig.suptitle('Distribution of First Components by Method and Dataset', y=1.02, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"component_distributions_{output_file}", dpi=300, bbox_inches='tight')
    plt.close()

warnings.filterwarnings("ignore")

def generate_pca(phenotype,fold,pca):
    os.system("python Methods1-GeneratePCA.py "+ phenotype + " "+str(fold)+ " " + str(pca))
def train_and_evaluate_model(X_train, y_train, X_test, y_test, X_val, y_val):
    """
    Train a logistic regression model, optimize hyperparameters for AUC, and evaluate it.

    Parameters:
    - X_train (array-like): Training features.
    - y_train (array-like): Training labels.
    - X_test (array-like): Testing features.
    - y_test (array-like): Testing labels.
    - X_val (array-like): Validation features.
    - y_val (array-like): Validation labels.

    Returns:
    - tuple: Train AUC, Test AUC, Validation AUC, and the trained model.
    """
    # Define the logistic regression model
    log_reg = LogisticRegression(solver='saga', max_iter=10000, class_weight='balanced')

    # Define hyperparameters for optimization
    param_grid = {
        'C': np.logspace(-3, 3, 7),  # Regularization strength
        'penalty': ['l1', 'l2']      # L1 and L2 regularization
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate AUC on train, test, and validation sets
    y_train_pred = best_model.predict_proba(X_train)[:, 1]
    y_test_pred = best_model.predict_proba(X_test)[:, 1]
    y_val_pred = best_model.predict_proba(X_val)[:, 1]

    auc_train = roc_auc_score(y_train, y_train_pred)
    auc_test = roc_auc_score(y_test, y_test_pred)
    auc_val = roc_auc_score(y_val, y_val_pred)

    # Return AUC scores and model
    return auc_train, auc_test, auc_val, best_model


def feature_selection_code(train_data_combined, test_data_combined, validation_data_combined, train_labels, test_labels, validation_labels, n_features=10):
 
    # Ensure train_data_combined is a DataFrame
    if not isinstance(train_data_combined, pd.DataFrame):
        train_data_combined = pd.DataFrame(train_data_combined)
        test_data_combined = pd.DataFrame(test_data_combined)
        validation_data_combined = pd.DataFrame(validation_data_combined)
        
  
    # 1. Filter Method: Chi-Square
    chi2_selector = SelectKBest(chi2, k=n_features)
    chi2_selector.fit(abs(train_data_combined), train_labels)
    chi2_selected_features = train_data_combined.columns[chi2_selector.get_support()]

    # 2. Filter Method: Mutual Information
    mi_selector = SelectKBest(mutual_info_classif, k=n_features)
    mi_selector.fit(train_data_combined, train_labels)
    mi_selected_features = train_data_combined.columns[mi_selector.get_support()]

    # 3. Wrapper Method: Recursive Feature Elimination (RFE) with Random Forest
    rfe_model = RandomForestClassifier(random_state=42)
    rfe_selector = RFE(estimator=rfe_model, n_features_to_select=n_features)
    rfe_selector.fit(train_data_combined, train_labels)
    rfe_selected_features = train_data_combined.columns[rfe_selector.get_support()]

    # 4. Embedded Method: Lasso Regression
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(train_data_combined, train_labels)
    lasso_selected_features = train_data_combined.columns[np.abs(lasso.coef_) > 0]

    # 5. Tree-Based Method: Random Forest Feature Importance
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(train_data_combined, train_labels)
    feature_importances = pd.Series(rf_model.feature_importances_, index=train_data_combined.columns)
    tree_selected_features = feature_importances.nlargest(n_features).index

    # Combine features selected by different methods
    all_selected_features = set(
        chi2_selected_features.tolist() +
        mi_selected_features.tolist() +
        rfe_selected_features.tolist() +
        lasso_selected_features.tolist() +
        tree_selected_features.tolist()
    )
 

    # Reduce datasets to selected features
    train_data_selected = train_data_combined[list(all_selected_features)].values
    test_data_selected = test_data_combined[list(all_selected_features)].values
    val_data_selected = validation_data_combined[list(all_selected_features)].values

    return train_data_selected, test_data_selected, val_data_selected, train_labels, test_labels, validation_labels

def add_dataset_info(dataset_name, X_train, X_test, X_val, y_train, y_test, y_val):
    dataset_info.append({
        "dataset_name": dataset_name,
        "X_train_features": X_train.shape[1],
        "X_test_features": X_test.shape[1],
        "X_val_features": X_val.shape[1],
        "y_train_samples": y_train.shape[0],
        "y_test_samples": y_test.shape[0],
        "y_val_samples": y_val.shape[0]
    })



def check_prs_based_gwas_files():
    results = []
    for phenotype, gwas in zip(phenotypes, gwas_files):
        for model in Models:
            for fold in range(0, 5):
                #print(f"Phenotype: {phenotype}, GWAS: {gwas}, Model: {model}, Fold: {fold}")
                try:
                    X = get_me_prs_based_gwas_file(phenotype, gwas, fold, model=model)
                    status = "Success"
                except Exception as e:
                    status = f"Failed: {str(e)}"
                    #print(phenotype, gwas, model, fold, e, "did not work")
                #print(status)
                results.append({
                    'Phenotype': phenotype,
                    'GWAS': gwas,
                    'Model': model,
                    'Fold': fold,
                    'Status': status
                })

    # Create DataFrame and save results
    df = pd.DataFrame(results)
    
    def aggregation_func(x):
        return 'Success' if any(value == 'Success' for value in x) else 'Failed'

    summary = pd.pivot_table(
        df,
        values='Status',
        index=['Phenotype', 'GWAS', 'Model'],
        columns='Fold',
        aggfunc=aggregation_func
    )

    print("\nProcessing Results Summary:")
    print(summary)


def check_pca_data():
    results = []
    for phenotype, gwas in zip(phenotypes, gwas_files):
        for fold in range(0, 5):
            print(f"Phenotype: {phenotype}, GWAS: {gwas}, Fold: {fold}")
            try:
                X_train, X_test, X_val, y_train, y_test, y_val = get_me_just_pca_data(phenotype, gwas, fold)
                print(X_train.head(2))
                status = "Success"
            except Exception as e:
                status = f"Failed: {str(e)}"
                #print(phenotype, gwas, fold, e, "did not work")
            #print(status)
            results.append({
                'Phenotype': phenotype,
                'GWAS': gwas,
                'Fold': fold,
                'Status': status
            })

    # Create DataFrame and save results
    df = pd.DataFrame(results)
    
    def aggregation_func(x):
        return 'Success' if any(value == 'Success' for value in x) else 'Failed'

    summary = pd.pivot_table(
        df,
        values='Status',
        index=['Phenotype', 'GWAS'],
        columns='Fold',
        aggfunc=aggregation_func
    )

    print("\nProcessing Results Summary:")
    print(summary)


def check_prs_data():
    results = []
    for phenotype, gwas in zip(phenotypes, gwas_files):
        for model in Models:
            for fold in range(0, 5):
                print(f"Phenotype: {phenotype}, GWAS: {gwas}, Model: {model}, Fold: {fold}")
                try:
                    X_train, X_test, X_val, y_train, y_test, y_val = get_me_just_prs_data(phenotype, gwas, fold, model=model)
                    print(X_train.head(2))
                    status = "Success"
                except Exception as e:
                    status = f"Failed: {str(e)}"
                    #print(phenotype, gwas, model, fold, e, "did not work")
                #print(status)
                results.append({
                    'Phenotype': phenotype,
                    'GWAS': gwas,
                    'Model': model,
                    'Fold': fold,
                    'Status': status
                })

    # Create DataFrame and save results
    df = pd.DataFrame(results)
    
    def aggregation_func(x):
        return 'Success' if any(value == 'Success' for value in x) else 'Failed'

    summary = pd.pivot_table(
        df,
        values='Status',
        index=['Phenotype', 'GWAS', 'Model'],
        columns='Fold',
        aggfunc=aggregation_func
    )

    print("\nProcessing Results Summary:")
    print(summary)


def check_genotype_data_for_all_snps():
    results = []

    for phenotype, gwas in zip(phenotypes, gwas_files):
        for snp in SNPs:
            for fold in range(0, 1):
                print(f"Phenotype: {phenotype}, GWAS: {gwas}, SNP: {snp}, Fold: {fold}")
                try:
                    X_train, X_test, X_val, y_train, y_test, y_val = get_me_just_genotype_data(phenotype, gwas, fold, numberofsnps=snp,scaling=True)
                    print(X_train.head(2))
                    status = "Success"
                except Exception as e:
                    status = f"Failed: {str(e)}"
                    #print(phenotype, gwas, snp, fold, e, "did not work")
                #print(status)
                results.append({
                    'Phenotype': phenotype,
                    'GWAS': gwas,
                    'SNPs': snp,
                    'Fold': fold,
                    'Status': status
                })

    # Create DataFrame and save results
    df = pd.DataFrame(results)
    
    def aggregation_func(x):
        return 'Success' if any(value == 'Success' for value in x) else 'Failed'

    summary = pd.pivot_table(
        df,
        values='Status',
        index=['Phenotype', 'GWAS', 'SNPs'],
        columns='Fold',
        aggfunc=aggregation_func
    )

    print("\nProcessing Results Summary:")
    print(summary)


def check_covariate_data():
    results = []
    for phenotype, gwas in zip(phenotypes, gwas_files):
    
        for fold in range(0, 5):
            print(f"Phenotype: {phenotype}, GWAS: {gwas},  Fold: {fold}")
            try:
                X_train, X_test, X_val, y_train, y_test, y_val = get_me_just_covariates_data(phenotype, gwas, fold )
                print(X_train.head(2))
                status = "Success"
            except Exception as e:
                status = f"Failed: {str(e)}"
                    
            results.append({
                'Phenotype': phenotype,
                'GWAS': gwas,
                'Fold': fold,
                'Status': status
            })

    # Create DataFrame and save results
    df = pd.DataFrame(results)
    
    def aggregation_func(x):
        return 'Success' if any(value == 'Success' for value in x) else 'Failed'

    summary = pd.pivot_table(
        df,
        values='Status',
        index=['Phenotype', 'GWAS'],
        columns='Fold',
        aggfunc=aggregation_func
    )

    print("\nProcessing Results Summary:")
    print(summary)


def apply_weights(train, test, val, weights, SNPCol, weightCol):
    snps = [col.split("_")[0] for col in train.columns[4:]]
    original_snps = [col for col in train.columns[4:]]
    #weights["OR"] = np.log(weights["OR"])
     
    weights = weights[[SNPCol, weightCol]]

    # Find common SNPs
    common_snps = set(snps).intersection(set(weights[SNPCol]))
    
    # Filter weights for common SNPs
    weights = weights[weights[SNPCol].isin(common_snps)]
    
    # Create a dictionary for quick lookup
    weight_dict = dict(zip(weights[SNPCol], weights[weightCol]))
    
    # Apply weights to the columns
    for col in original_snps:
        snp = col.split("_")[0]
        if snp in weight_dict:
             
            train[col] *= weight_dict[snp]
            test[col] *= weight_dict[snp]
            val[col] *= weight_dict[snp]
          
    return train, test, val


# Dataset 1: Covaariates
def get_me_just_covariates_data(phenotype_file,gwas_file,fold,scaling=False,feature_selection=False):
    gwas_basename = gwas_file.split("/")[-1].split(".")[0]

    # Read and preprocess data (same steps as before)
    train_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/train_data.fam"
    test_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/test_data.fam"
    validation_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/validation_data.fam"
    
    cov_train_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/train_data.cov"
    cov_test_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/test_data.cov"
    cov_validation_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/validation_data.cov"    
    

    # Read the .fam files (phenotype information) and assign proper column names
    train_data = pd.read_csv(train_file, sep="\s+", header=None, names=["FID", "IID"] + [f"SNP_{i}" for i in range(1, len(pd.read_csv(train_file, sep="\s+", header=None).columns)-1)])
    test_data = pd.read_csv(test_file, sep="\s+", header=None, names=["FID", "IID"] + [f"SNP_{i}" for i in range(1, len(pd.read_csv(test_file, sep="\s+", header=None).columns)-1)])
    validation_data = pd.read_csv(validation_file, sep="\s+", header=None, names=["FID", "IID"] + [f"SNP_{i}" for i in range(1, len(pd.read_csv(validation_file, sep="\s+", header=None).columns)-1)])

    # Read the corresponding covariate files (features)
    cov_train_data = pd.read_csv(cov_train_file, sep="\s+")
    cov_test_data = pd.read_csv(cov_test_file, sep="\s+")
    cov_validation_data = pd.read_csv(cov_validation_file, sep="\s+")

    # Sort all data based on FID and IID
    train_data = train_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)
    test_data = test_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)
    validation_data = validation_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)

    cov_train_data = cov_train_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)
    cov_test_data = cov_test_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)
    cov_validation_data = cov_validation_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)

   

 
    # Map phenotype labels to binary (assuming 1 is the negative class, 2 is the positive class)
    train_labels = train_data.iloc[:, -1].map({1: 0, 2: 1})
    test_labels = test_data.iloc[:, -1].map({1: 0, 2: 1})
    validation_labels = validation_data.iloc[:, -1].map({1: 0, 2: 1})


    # Drop only the phenotype column (last column)
    train_features = train_data.drop(columns=[train_data.columns[-1]]) 
    test_features = test_data.drop(columns=[test_data.columns[-1]])
    validation_features = validation_data.drop(columns=[validation_data.columns[-1]])

    # Merge train features and covariates on FID and IID
    train_data_combined = pd.merge(train_features, cov_train_data, 
                                  on=['FID', 'IID'],
                                  how='inner')

    # Merge test features and covariates on FID and IID  
    test_data_combined = pd.merge(test_features, cov_test_data,
                                 on=['FID', 'IID'], 
                                 how='inner')

    # Merge validation features and covariates on FID and IID
    validation_data_combined = pd.merge(validation_features, cov_validation_data,
                                      on=['FID', 'IID'],
                                      how='inner')


    # Replace NaN values with 0
    train_data_combined = train_data_combined.replace(np.nan, 0)
    test_data_combined = test_data_combined.replace(np.nan, 0)
    validation_data_combined = validation_data_combined.replace(np.nan, 0)


   
    # Standardize the data (optional but often recommended)
    if scaling:
        scaler = StandardScaler()
        # Keep FID and IID columns unchanged, scale the rest
        train_data_combined.iloc[:, 2:] = scaler.fit_transform(train_data_combined.iloc[:, 2:]) 
        test_data_combined.iloc[:, 2:] = scaler.transform(test_data_combined.iloc[:, 2:])
        validation_data_combined.iloc[:, 2:] = scaler.transform(validation_data_combined.iloc[:, 2:])
    
     
    return train_data_combined,test_data_combined,validation_data_combined,train_labels,test_labels,validation_labels
 
# Dataset 2: PCA
def get_me_just_pca_data(phenotype_file, gwas_file,fold,pca=10, generate_pca = False,scaling=False, feature_selection=False):
    gwas_basename = gwas_file.split("/")[-1].split(".")[0]

    train_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/train_data.fam"
    test_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/test_data.fam"
    validation_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/validation_data.fam"

    pca_train_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/train_data.eigenvec"
    pca_test_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/test_data.eigenvec"
    pca_validation_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/validation_data.eigenvec"
    
    # Read the .fam files (phenotype information) and assign proper column names
    train_data = pd.read_csv(train_file, sep="\s+", header=None, names=["FID", "IID"] + [f"SNP_{i}" for i in range(1, len(pd.read_csv(train_file, sep="\s+", header=None).columns)-1)])
    test_data = pd.read_csv(test_file, sep="\s+", header=None, names=["FID", "IID"] + [f"SNP_{i}" for i in range(1, len(pd.read_csv(test_file, sep="\s+", header=None).columns)-1)])
    validation_data = pd.read_csv(validation_file, sep="\s+", header=None, names=["FID", "IID"] + [f"SNP_{i}" for i in range(1, len(pd.read_csv(validation_file, sep="\s+", header=None).columns)-1)])

    # Read the corresponding PCA files (features)
    pca_train_data = pd.read_csv(pca_train_file, sep="\s+", names=["FID", "IID"] + [f"PCA_{i}" for i in range(1, len(pd.read_csv(pca_train_file, sep="\s+", header=None).columns)-1)])
    pca_test_data = pd.read_csv(pca_test_file, sep="\s+", names=["FID", "IID"] + [f"PCA_{i}" for i in range(1, len(pd.read_csv(pca_test_file, sep="\s+", header=None).columns)-1)])
    pca_validation_data = pd.read_csv(pca_validation_file, sep="\s+", names=["FID", "IID"] + [f"PCA_{i}" for i in range(1, len(pd.read_csv(pca_validation_file, sep="\s+", header=None).columns)-1)])

    # Sort all data based on FID and IID  
    train_data = train_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)
    test_data = test_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)
    validation_data = validation_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)

    pca_train_data = pca_train_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)
    pca_test_data = pca_test_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)
    pca_validation_data = pca_validation_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)

    
 
    # Map phenotype labels to binary (assuming 1 is the negative class, 2 is the positive class)
    train_labels = train_data.iloc[:, -1].map({1: 0, 2: 1})
    test_labels = test_data.iloc[:, -1].map({1: 0, 2: 1})
    validation_labels = validation_data.iloc[:, -1].map({1: 0, 2: 1})

    # Drop only the phenotype column (last column)
    train_features = train_data.drop(columns=[train_data.columns[-1]])
    test_features = test_data.drop(columns=[test_data.columns[-1]])
    validation_features = validation_data.drop(columns=[validation_data.columns[-1]])

    # Merge train features and PCA on FID and IID
    train_data_combined = pd.merge(train_features, pca_train_data, 
                                  on=['FID', 'IID'],
                                  how='inner')

    # Merge test features and PCA on FID and IID
    test_data_combined = pd.merge(test_features, pca_test_data,
                                 on=['FID', 'IID'], 
                                 how='inner')

    # Merge validation features and PCA on FID and IID
    validation_data_combined = pd.merge(validation_features, pca_validation_data,
                                      on=['FID', 'IID'],
                                      how='inner')
    
    train_data_combined = train_data_combined.replace(np.nan, 0)
    test_data_combined = test_data_combined.replace(np.nan, 0)
    validation_data_combined = validation_data_combined.replace(np.nan, 0)


    if scaling:
        scaler = StandardScaler()
        # Keep FID and IID columns unchanged, scale the rest
        train_data_combined.iloc[:, 2:] = scaler.fit_transform(train_data_combined.iloc[:, 2:])
        test_data_combined.iloc[:, 2:] = scaler.transform(test_data_combined.iloc[:, 2:])
        validation_data_combined.iloc[:, 2:] = scaler.transform(validation_data_combined.iloc[:, 2:])
    
    return train_data_combined, test_data_combined, validation_data_combined, train_labels, test_labels, validation_labels

# Dataset 3: Genotype
def get_me_just_genotype_data(phenotype_file, gwas_file, fold, numberofsnps,scaling=False,feature_selection=False,weightFile=None,SNPCol="SNP", weightCol="OR"):
     
    gwas_basename = gwas_file.split("/")[-1].split(".")[0]


    if numberofsnps.startswith('snps_'):
        train = pd.read_csv(os.path.join(phenotype_file,gwas_basename, f"Fold_{fold}", numberofsnps, 'ptrain.raw'), sep=r"\s+")
        val = pd.read_csv(os.path.join(phenotype_file,gwas_basename, f"Fold_{fold}", numberofsnps, 'pval.raw'), sep=r"\s+")
        test = pd.read_csv(os.path.join(phenotype_file,gwas_basename, f"Fold_{fold}", numberofsnps, 'ptest.raw'), sep=r"\s+")


    train.sort_values(by=["FID", "IID"], inplace=True)
    val.sort_values(by=["FID", "IID"], inplace=True)    
    test.sort_values(by=["FID", "IID"], inplace=True)


    traindirec = phenotype_file+os.sep+gwas_basename + os.sep + f"Fold_{fold}" 
    
    common_columns = train.columns.intersection(test.columns).intersection(val.columns)

    # Select common columns from each DataFrame
    train = train[common_columns]
    test = test[common_columns]
    val = val[common_columns]
 

    del train["PHENOTYPE"]
    del val["PHENOTYPE"]
    del test["PHENOTYPE"]

   

    tempphenotype_train = pd.read_table(traindirec+os.sep+"train_data.QC.fam", sep="\s+",header=None)
    tempphenotype_test = pd.read_table(traindirec+os.sep+"test_data.fam", sep="\s+",header=None)
    tempphenotype_val = pd.read_table(traindirec+os.sep+"validation_data.fam", sep="\s+",header=None)
    
    tempphenotype_train.sort_values(by=[0,1], inplace=True)
    tempphenotype_test.sort_values(by=[0,1], inplace=True)
    tempphenotype_val.sort_values(by=[0,1], inplace=True)


    phenotype_train = pd.DataFrame()
    phenotype_train["Phenotype"] = tempphenotype_train[5].values
    phenotype_train["Phenotype"] = phenotype_train["Phenotype"].replace({1: 0, 2: 1})

    phenotype_test= pd.DataFrame()
    phenotype_test["Phenotype"] = tempphenotype_test[5].values
    phenotype_test["Phenotype"] = phenotype_test["Phenotype"].replace({1: 0, 2: 1})
    
    phenotype_val= pd.DataFrame()
    phenotype_val["Phenotype"] = tempphenotype_val[5].values
    phenotype_val["Phenotype"] = phenotype_val["Phenotype"].replace({1: 0, 2: 1})

     
    y_train = phenotype_train["Phenotype"].values
    y_test = phenotype_test["Phenotype"].values
    y_val = phenotype_val["Phenotype"].values

    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    val.fillna(0, inplace=True)
 
 
    snps = [col.split("_")[0] for col in train.columns[4:]]
    original_snps = [col for col in train.columns[4:]]

    if weightFile is not None:
        train,test,val = apply_weights(train, test, val, weightFile, SNPCol, weightCol)

    X_train = train.copy()
    X_test = test.copy()
    X_val = val.copy()


    if scaling:
        scaler = StandardScaler()
        # Keep FID and IID columns unchanged, scale the rest
        X_train.iloc[:, 2:] = scaler.fit_transform(X_train.iloc[:, 2:])
        X_test.iloc[:, 2:] = scaler.transform(X_test.iloc[:, 2:]) 
        X_val.iloc[:, 2:] = scaler.transform(X_val.iloc[:, 2:])
    
    if feature_selection:
        X_train, X_test, X_val, y_train, y_test, y_val = feature_selection_code(X_train, X_test, X_val, y_train, y_test, y_val)


    return X_train,X_test,X_val,y_train,y_test,y_val    

# Dataset 4: PRS
def get_me_just_prs_data(phenotype_file, gwas_file,fold, model="Plink", min_pvalue=10, num_intervals=200, spacing="log", scaling=False, feature_selection=False):
    if spacing == "log":
        all_pvalues = np.logspace(-min_pvalue, 0, num_intervals, endpoint=True)
    elif spacing == "linear":
        all_pvalues = np.linspace(min_pvalue, 1, num_intervals, endpoint=True)
    else:
        raise ValueError("Invalid spacing value. Use 'log' or 'linear'.")

    gwas_basename = gwas_file.split("/")[-1].split(".")[0]
    range_list_path = os.path.join(phenotype_file, gwas_basename,f"Fold_{fold}", 'range_list')
    with open(range_list_path, 'w') as file:
        for value in all_pvalues:
            file.write(f'pv_{value} 0 {value}\n')

    if model == "LDAK-GWAS":
        #os.system(f"python LDAK-GWAS.py {phenotype_file} {gwas_file} {fold}")
        pass
    
    if model == "AnnoPred":
        #os.system(f"python AnnoPredCode.py {phenotype_file} {gwas_file} {fold}")
        pass
    if model == "Plink":
        #os.system(f"python Plink.py {phenotype_file} {gwas_file} {fold}")
        pass
    if model == "PRSice-2":
        # For PRSice-2, we need to change p-values with in the PRSice-2 script.
        #os.system(f"python PRSice-2.py {phenotype_file} {gwas_file} {fold}")
         
        prs_train_file = None
        prs_test_file = None
        prs_val_file = None

        # Walk through directory to find files
        for root, dirs, files in os.walk(os.path.join(phenotype_file, gwas_file.split("/")[-1].split(".")[0],f"Fold_{fold}", model)):
            for file in files:
               
                if "Train_PRSice_PRS" in file and "all_score" in file:
                    prs_train_file = os.path.join(root, file)
                elif "Test_PRSice_PRS" in file and "all_score" in file:
                    prs_test_file = os.path.join(root, file)
                elif "Validation_PRSice_PRS" in file and "all_score" in file:
                    prs_val_file = os.path.join(root, file)

        
        if not all([prs_train_file, prs_test_file, prs_val_file]):
            raise FileNotFoundError("Could not find all required PRS files")
 
        if os.path.exists(prs_train_file) and os.path.exists(prs_test_file) and os.path.exists(prs_val_file):
        
            prs_train = pd.read_csv(prs_train_file, sep="\s+")
            prs_test = pd.read_csv(prs_test_file, sep="\s+")
            prs_val = pd.read_csv(prs_val_file, sep="\s+")

            prs_train = prs_train.sort_values(by=["FID", "IID"]).reset_index(drop=True)
            prs_test = prs_test.sort_values(by=["FID", "IID"]).reset_index(drop=True)
            prs_val = prs_val.sort_values(by=["FID", "IID"]).reset_index(drop=True)

       

            
            train_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/train_data.fam"
            test_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/test_data.fam"
            validation_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/validation_data.fam"

            train_data = pd.read_csv(train_file, sep="\s+", header=None, names=["FID", "IID"] + [f"SNP_{i}" for i in range(1, len(pd.read_csv(train_file, sep="\s+", header=None).columns)-1)])
            test_data = pd.read_csv(test_file, sep="\s+", header=None, names=["FID", "IID"] + [f"SNP_{i}" for i in range(1, len(pd.read_csv(test_file, sep="\s+", header=None).columns)-1)])
            validation_data = pd.read_csv(validation_file, sep="\s+", header=None, names=["FID", "IID"] + [f"SNP_{i}" for i in range(1, len(pd.read_csv(validation_file, sep="\s+", header=None).columns)-1)])

            
            # Sort all data based on FID and IID  
            train_data = train_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)
            test_data = test_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)
            validation_data = validation_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)

            
        
            # Map phenotype labels to binary (assuming 1 is the negative class, 2 is the positive class)
            train_labels = train_data.iloc[:, -1].map({1: 0, 2: 1})
            test_labels = test_data.iloc[:, -1].map({1: 0, 2: 1})
            validation_labels = validation_data.iloc[:, -1].map({1: 0, 2: 1})

            prs_train_combined = prs_train.copy()
            prs_test_combined = prs_test.copy()
            prs_val_combined = prs_val.copy()

            
            if scaling:
                scaler = StandardScaler()
                # Keep FID and IID columns unchanged, scale the rest
                prs_train_combined.iloc[:, 2:] = scaler.fit_transform(prs_train_combined.iloc[:, 2:])
                prs_test_combined.iloc[:, 2:] = scaler.transform(prs_test_combined.iloc[:, 2:])
                prs_val_combined.iloc[:, 2:] = scaler.transform(prs_val_combined.iloc[:, 2:])

            
            return prs_train_combined, prs_test_combined, prs_val_combined, train_labels, test_labels, validation_labels

    if model == "LDpred-fast":
        #os.system(f"python LDpred-fast.py {phenotype_file} {gwas_file} {fold}")
        pass
         
    prs_train_list = []
    prs_test_list = []
    prs_val_list = []

    for i in all_pvalues:
        try:
            prs_train = pd.read_table(os.path.join(phenotype_file, gwas_file.split("/")[-1].split(".")[0],f"Fold_{fold}",model, f"train_data.pv_{i}.profile"), sep="\s+", usecols=["FID", "IID", "SCORE"])
            prs_test = pd.read_table(os.path.join(phenotype_file, gwas_file.split("/")[-1].split(".")[0],f"Fold_{fold}",model, f"test_data.pv_{i}.profile"), sep="\s+", usecols=["FID", "IID", "SCORE"])
            prs_val = pd.read_table(os.path.join(phenotype_file, gwas_file.split("/")[-1].split(".")[0],f"Fold_{fold}",model,f"validation_data.pv_{i}.profile"), sep="\s+", usecols=["FID", "IID", "SCORE"])
            # Here rename the SCORE column to the pvalue
            prs_train.rename(columns={"SCORE": f"pv_{i}"}, inplace=True)
            prs_test.rename(columns={"SCORE": f"pv_{i}"}, inplace=True)
            prs_val.rename(columns={"SCORE": f"pv_{i}"}, inplace=True)
        
            prs_train_list.append(prs_train)
            prs_test_list.append(prs_test)
            prs_val_list.append(prs_val)
        except:
            pass
    
    prs_train_combined = pd.concat(prs_train_list, axis=1)
    prs_train_combined = prs_train_combined.loc[:,~prs_train_combined.columns.duplicated()]

    prs_test_combined = pd.concat(prs_test_list, axis=1)
    prs_test_combined = prs_test_combined.loc[:,~prs_test_combined.columns.duplicated()]

    prs_val_combined = pd.concat(prs_val_list, axis=1)
    prs_val_combined = prs_val_combined.loc[:,~prs_val_combined.columns.duplicated()]

    prs_train_combined = prs_train_combined.sort_values(by=["FID", "IID"]).reset_index(drop=True)
    prs_test_combined = prs_test_combined.sort_values(by=["FID", "IID"]).reset_index(drop=True)
    prs_val_combined = prs_val_combined.sort_values(by=["FID", "IID"]).reset_index(drop=True)



    prs_train = prs_train.sort_values(by=["FID", "IID"]).reset_index(drop=True)
    prs_test = prs_test.sort_values(by=["FID", "IID"]).reset_index(drop=True)
    prs_val = prs_val.sort_values(by=["FID", "IID"]).reset_index(drop=True)



        
    train_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/train_data.fam"
    test_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/test_data.fam"
    validation_file = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/validation_data.fam"


    train_data = pd.read_csv(train_file, sep="\s+", header=None, names=["FID", "IID"] + [f"SNP_{i}" for i in range(1, len(pd.read_csv(train_file, sep="\s+", header=None).columns)-1)])
    test_data = pd.read_csv(test_file, sep="\s+", header=None, names=["FID", "IID"] + [f"SNP_{i}" for i in range(1, len(pd.read_csv(test_file, sep="\s+", header=None).columns)-1)])
    validation_data = pd.read_csv(validation_file, sep="\s+", header=None, names=["FID", "IID"] + [f"SNP_{i}" for i in range(1, len(pd.read_csv(validation_file, sep="\s+", header=None).columns)-1)])

    
    # Sort all data based on FID and IID  
    train_data = train_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)
    test_data = test_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)
    validation_data = validation_data.sort_values(by=["FID", "IID"]).reset_index(drop=True)

    
 
    # Map phenotype labels to binary (assuming 1 is the negative class, 2 is the positive class)
    train_labels = train_data.iloc[:, -1].map({1: 0, 2: 1})
    test_labels = test_data.iloc[:, -1].map({1: 0, 2: 1})
    validation_labels = validation_data.iloc[:, -1].map({1: 0, 2: 1})


    if scaling:
        scaler = StandardScaler()
        prs_train_combined = scaler.fit_transform(prs_train_combined)
        prs_test_combined = scaler.transform(prs_test_combined)
        prs_val_combined = scaler.transform(prs_val_combined)

    if feature_selection:
        prs_train_combined, prs_test_combined, prs_val_combined, train_labels, test_labels, validation_labels = feature_selection(prs_train_combined, prs_test_combined, prs_val_combined, train_labels, test_labels, validation_labels)

    #print(prs_train_combined.shape, prs_test_combined.shape, prs_val_combined.shape, train_labels.shape, test_labels.shape, validation_labels.shape)

    return prs_train_combined, prs_test_combined, prs_val_combined, train_labels, test_labels, validation_labels
 

def get_me_prs_based_gwas_file(phenotype_file, gwas_file, fold, model="Plink"):
    """Read GWAS file based on different PRS models and return standardized columns."""
    
    gwas_basename = gwas_file.split("/")[-1].split(".")[0]
    
    if model in ["Plink", "PRSice-2"]:
        filename = f"{phenotype_file}/{gwas_basename}/{phenotype_file}_PRSice-2.txt"
        data = pd.read_csv(filename, sep="\s+")
        
    elif model == "LDpred-fast":
        filename = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/{phenotype_file}ldpred_fast_gwas"
        data = pd.read_csv(filename, sep="\s+")
        data = data.rename(columns={'sid': 'SNP', 'nt1': 'A1', 'ldpred_beta': 'OR'})
        
    elif model == "LDAK-GWAS":
        filename = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/lasso{phenotype_file}_ldak_gwas_final"
        data = pd.read_csv(filename, sep="\s+")
        data = data.rename(columns={'SNP': 'SNP', 'A1': 'A1', 'Model6': 'OR'})
        
    elif model == "AnnoPred":
        filename = f"{phenotype_file}/{gwas_basename}/Fold_{fold}/AnnoPred_GWAS.txt"
        data = pd.read_csv(filename, sep="\s+")
        data = data.rename(columns={'SNP': 'SNP', 'A1': 'A1', 'BETA': 'OR'})
    
    else:
        raise ValueError(f"Unknown model: {model}")

    required_columns = ['SNP', 'A1', 'OR']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Expected {required_columns}, got {data.columns}")
        
    return data[required_columns]
 

  
#get_me_just_prs_data("migraine", "migraine.gz", 1, "PRSice-2")
 



# data1 = get_me_prs_based_gwas_file("migraine", "migraine.gz", 0, model="LDAK-GWAS")

# X_train1, X_test, X_val, y_train, y_test, y_val = get_me_just_genotype_data(
#     "migraine", "migraine.gz", 0, "snps_50", weightFile=data1, scaling=False, feature_selection=True
# )
# X_train2, X_test, X_val, _, y_test, y_val = get_me_just_genotype_data(
#     "migraine", "migraine.gz", 0, "snps_50", weightFile=None, scaling=False, feature_selection=False
# )
# visualize_dimensionality_reductions(
#     X_train1,  # Your first dataset (with weights)
#     X_train2,  # Your second dataset (without weights)
#     y_train,   # Your labels
#     output_file="my_visualization.png"  # Optional: specify output filename
# ) 
  

# phenotypes = ["migraine", "migraine", "migraine",
#                 "depression", "depression", "depression",
#                 "hypertension", "hypertension", "hypertension"]
# gwas_files = ["migraine_2.gz", "migraine_5.gz", "migraine.gz",
#                 "depression_11.gz", "depression_17.gz", "depression_4.gz",
#                 "hypertension_0.gz", "hypertension_14.gz", "hypertension_20.gz"]

# SNPs = ["snps_50", "snps_100", "snps_200", "snps_500", "snps_1000", "snps_5000"]
# Models = ["Plink", "PRSice-2", "AnnoPred", "LDAK-GWAS"]
 



# get_me_prs_based_gwas_file("migraine", "migraine.gz", 0, model="LDAK-GWAS")
# get_me_genotype_data("migraine", "migraine.gz", 0, "snps_50",gwasfile, scaling=False, feature_selection=False)
# get_me_prs_data("migraine", "migraine.gz", 0, "Plink", 10, 200, "log", scaling=False, feature_selection=False)



#get_me_prs_based_gwas_file("migraine", "migraine.gz", 0, model="LDAK-GWAS")
# get_me_just_prs_data("migraine", "migraine.gz", 0, "Plink")
# get_me_just_covariates_data("migraine","migraine.gz", 0, scaling=False)
# get_me_just_pca_data("migraine", "migraine.gz", 0,scaling=False)
# get_me_just_genotype_data("migraine", "migraine.gz", 0, "snps_50",scaling=False,weightFile=None)

# parameters = {
#     "phenotype_file":  ["migraine", "migraine", "migraine",
#                 "depression", "depression", "depression",
#                 "hypertension", "hypertension", "hypertension"],
#     "fold": 0,
#     "pca": 10,
#     "gwas_file": ["migraine_2.gz", "migraine_5.gz", "migraine.gz",
#                 "depression_11.gz", "depression_17.gz", "depression_4.gz",
#                 "hypertension_0.gz", "hypertension_14.gz", "hypertension_20.gz"],
#     "model":  ["Plink", "PRSice-2", "AnnoPred", "LDAK-GWAS"],
#     "scaling": [False,True],
#     "weightFile": [None, get_me_prs_based_gwas_file(phenotype_file, gwas_file, fold, model=model)],   
# }



  
 


# # Dataset 5: PCA + PRS 
# def get_me_pca_prs_data(phenotype_file, fold, pca, gwas_file, model="Plink", generate_pca=False, min_pvalue=10, num_intervals=200, spacing="log", scaling=False, feature_selection=False):
#     # Get PCA data
#     train_pca, test_pca, val_pca, train_labels_pca, test_labels_pca, val_labels_pca = get_me_just_pca_data(
#         phenotype_file, fold, pca, generate_pca, scaling=False
#     )
    
#     # Get PRS data
#     train_prs, test_prs, val_prs,_,_,_ = get_me_prs_data(
#         phenotype_file, fold, gwas_file, model, min_pvalue, num_intervals, spacing, scaling=False
#     )
    
#     # Combine the features
#     train_combined = np.hstack((train_pca, train_prs))
#     test_combined = np.hstack((test_pca, test_prs))
#     val_combined = np.hstack((val_pca, val_prs))
    
#     if scaling:
#         scaler = StandardScaler()
#         train_combined = scaler.fit_transform(train_combined)
#         test_combined = scaler.transform(test_combined)
#         val_combined = scaler.transform(val_combined)
    
#     if feature_selection:
#         train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca = feature_selection(
#             train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca
#         )
    
#     return train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca

# # Dataset 6: PRS + Genotype
# def get_me_prs_genotype_data(phenotype_file, fold, pvalue, gwas_file, model="Plink", min_pvalue=10, num_intervals=200, spacing="log", scaling=False, feature_selection=False):
#     # Get genotype data
#     train_geno, test_geno, val_geno, train_labels_geno, test_labels_geno, val_labels_geno = get_me_just_genotype_data(
#         phenotype_file, fold, pvalue, scaling=False
#     )
    
#     # Get PRS data
#     train_prs, test_prs, val_prs,_,_,_ = get_me_prs_data(
#         phenotype_file, fold, gwas_file, model, min_pvalue, num_intervals, spacing, scaling=False
#     )
    
#     # Combine the features
#     train_combined = np.hstack((train_geno, train_prs))
#     test_combined = np.hstack((test_geno, test_prs))
#     val_combined = np.hstack((val_geno, val_prs))
    
#     if scaling:
#         scaler = StandardScaler()
#         train_combined = scaler.fit_transform(train_combined)
#         test_combined = scaler.transform(test_combined)
#         val_combined = scaler.transform(val_combined)
    
#     if feature_selection:
#         train_combined, test_combined, val_combined, train_labels_geno, test_labels_geno, val_labels_geno = feature_selection(
#             train_combined, test_combined, val_combined, train_labels_geno, test_labels_geno, val_labels_geno
#         )
    
#     return train_combined, test_combined, val_combined, train_labels_geno, test_labels_geno, val_labels_geno

# # Dataset 7: PRS + Genotype
# def get_me_prs_covariates_data(phenotype_file, fold, gwas_file, model="Plink", min_pvalue=10, num_intervals=200, spacing="log", scaling=False, feature_selection=False):
#     # Get covariates data
#     train_cov, test_cov, val_cov, train_labels_cov, test_labels_cov, val_labels_cov = get_me_just_covariates_data(
#         phenotype_file, fold, scaling=False
#     )
    
#     # Get PRS data
#     train_prs, test_prs, val_prs,_,_,_ = get_me_prs_data(
#         phenotype_file, fold, gwas_file, model, min_pvalue, num_intervals, spacing, scaling=False
#     )
    
#     # Combine the features
#     train_combined = np.hstack((train_cov, train_prs))
#     test_combined = np.hstack((test_cov, test_prs))
#     val_combined = np.hstack((val_cov, val_prs))
    
#     if scaling:
#         scaler = StandardScaler()
#         train_combined = scaler.fit_transform(train_combined)
#         test_combined = scaler.transform(test_combined)
#         val_combined = scaler.transform(val_combined)
    
#     if feature_selection:
#         train_combined, test_combined, val_combined, train_labels_cov, test_labels_cov, val_labels_cov = feature_selection(
#             train_combined, test_combined, val_combined, train_labels_cov, test_labels_cov, val_labels_cov
#         )
    
#     return train_combined, test_combined, val_combined, train_labels_cov, test_labels_cov, val_labels_cov

# # Dataset 8: PCA + PRS + Genotype
# def get_me_pca_prs_genotype_data(phenotype_file, fold, pca, pvalue, gwas_file, model="Plink", generate_pca=False, min_pvalue=10, num_intervals=200, spacing="log", scaling=False, feature_selection=False):
#     # Get PCA data
#     train_pca, test_pca, val_pca, train_labels_pca, test_labels_pca, val_labels_pca = get_me_just_pca_data(
#         phenotype_file, fold, pca, generate_pca, scaling=False
#     )
    
#     # Get genotype data
#     train_geno, test_geno, val_geno, train_labels_geno, test_labels_geno, val_labels_geno = get_me_just_genotype_data(
#         phenotype_file, fold, pvalue, scaling=False
#     )
    
#     # Get PRS data
#     train_prs, test_prs, val_prs,_,_,_ = get_me_prs_data(
#         phenotype_file, fold, gwas_file, model, min_pvalue, num_intervals, spacing, scaling=False
#     )
    
#     # Combine all features
#     train_combined = np.hstack((train_pca, train_geno, train_prs))
#     test_combined = np.hstack((test_pca, test_geno, test_prs))
#     val_combined = np.hstack((val_pca, val_geno, val_prs))
    
#     if scaling:
#         scaler = StandardScaler()
#         train_combined = scaler.fit_transform(train_combined)
#         test_combined = scaler.transform(test_combined)
#         val_combined = scaler.transform(val_combined)
    
#     if feature_selection:
#         train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca = feature_selection(
#             train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca
#         )
    
#     return train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca

# # Dataset 9: PCA + PRS + Covariates
# def get_me_pca_prs_covariates_data(phenotype_file, fold, pca, gwas_file, model="Plink", generate_pca=False, min_pvalue=10, num_intervals=200, spacing="log", scaling=False, feature_selection=False):
#     # Get PCA data
#     train_pca, test_pca, val_pca, train_labels_pca, test_labels_pca, val_labels_pca = get_me_just_pca_data(
#         phenotype_file, fold, pca, generate_pca, scaling=False
#     )
    
#     # Get covariates data
#     train_cov, test_cov, val_cov, train_labels_cov, test_labels_cov, val_labels_cov = get_me_just_covariates_data(
#         phenotype_file, fold, scaling=False
#     )
    
#     # Get PRS data
#     train_prs, test_prs, val_prs,_,_,_ = get_me_prs_data(
#         phenotype_file, fold, gwas_file, model, min_pvalue, num_intervals, spacing, scaling=False
#     )
    
#     # Combine all features
#     train_combined = np.hstack((train_pca, train_cov, train_prs))
#     test_combined = np.hstack((test_pca, test_cov, test_prs))
#     val_combined = np.hstack((val_pca, val_cov, val_prs))
    
#     if scaling:
#         scaler = StandardScaler()
#         train_combined = scaler.fit_transform(train_combined)
#         test_combined = scaler.transform(test_combined)
#         val_combined = scaler.transform(val_combined)
    
#     if feature_selection:
#         train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca = feature_selection(
#             train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca
#         )
    
#     return train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca

# # Dataset 10: PRS + Genotype + Covariates
# def get_me_prs_genotype_covariates_data(phenotype_file, fold, pvalue, gwas_file, model="Plink", min_pvalue=10, num_intervals=200, spacing="log", scaling=False, feature_selection=False):
#     # Get genotype data
#     train_geno, test_geno, val_geno, train_labels_geno, test_labels_geno, val_labels_geno = get_me_just_genotype_data(
#         phenotype_file, fold, pvalue, scaling=False
#     )
    
#     # Get covariates data
#     train_cov, test_cov, val_cov, train_labels_cov, test_labels_cov, val_labels_cov = get_me_just_covariates_data(
#         phenotype_file, fold, scaling=False
#     )
    
#     # Get PRS data
#     train_prs, test_prs, val_prs,_,_,_ = get_me_prs_data(
#         phenotype_file, fold, gwas_file, model, min_pvalue, num_intervals, spacing, scaling=False
#     )
    
#     # Combine all features
#     train_combined = np.hstack((train_geno, train_cov, train_prs))
#     test_combined = np.hstack((test_geno, test_cov, test_prs))
#     val_combined = np.hstack((val_geno, val_cov, val_prs))
    
#     if scaling:
#         scaler = StandardScaler()
#         train_combined = scaler.fit_transform(train_combined)
#         test_combined = scaler.transform(test_combined)
#         val_combined = scaler.transform(val_combined)
    
#     if feature_selection:
#         train_combined, test_combined, val_combined, train_labels_geno, test_labels_geno, val_labels_geno = feature_selection(
#             train_combined, test_combined, val_combined, train_labels_geno, test_labels_geno, val_labels_geno
#         )
    
#     return train_combined, test_combined, val_combined, train_labels_geno, test_labels_geno, val_labels_geno

# # Dataset 11: PCA + PRS + Genotype + Covariates
# def get_me_pca_prs_genotype_covariates_data(phenotype_file, fold, pca, pvalue, gwas_file, model="Plink", generate_pca=False, min_pvalue=10, num_intervals=200, spacing="log", scaling=False, feature_selection=False):
#     # Get PCA data
#     train_pca, test_pca, val_pca, train_labels_pca, test_labels_pca, val_labels_pca = get_me_just_pca_data(
#         phenotype_file, fold, pca, generate_pca, scaling=False
#     )
    
#     # Get genotype data
#     train_geno, test_geno, val_geno, train_labels_geno, test_labels_geno, val_labels_geno = get_me_just_genotype_data(
#         phenotype_file, fold, pvalue, scaling=False
#     )
    
#     # Get covariates data
#     train_cov, test_cov, val_cov, train_labels_cov, test_labels_cov, val_labels_cov = get_me_just_covariates_data(
#         phenotype_file, fold, scaling=False
#     )
    
#     # Get PRS data
#     train_prs, test_prs, val_prs,_,_,_ = get_me_prs_data(
#         phenotype_file, fold, gwas_file, model, min_pvalue, num_intervals, spacing, scaling=False
#     )
    
#     # Combine all features
#     train_combined = np.hstack((train_pca, train_geno, train_cov, train_prs))
#     test_combined = np.hstack((test_pca, test_geno, test_cov, test_prs))
#     val_combined = np.hstack((val_pca, val_geno, val_cov, val_prs))
    
#     if scaling:
#         scaler = StandardScaler()
#         train_combined = scaler.fit_transform(train_combined)
#         test_combined = scaler.transform(test_combined)
#         val_combined = scaler.transform(val_combined)
    
#     if feature_selection:
#         train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca = feature_selection(
#             train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca
#         )
    
#     return train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca


# # Dataset 12: PCA + Genotype
# def get_me_pca_genotype_data(phenotype_file, fold, pca, pvalue, generate_pca=False, scaling=False, feature_selection=False):
#     # Get PCA data
#     train_pca, test_pca, val_pca, train_labels_pca, test_labels_pca, val_labels_pca = get_me_just_pca_data(
#         phenotype_file, fold, pca, generate_pca, scaling=False
#     )
    
#     # Get genotype data
#     train_geno, test_geno, val_geno, train_labels_geno, test_labels_geno, val_labels_geno = get_me_just_genotype_data(
#         phenotype_file, fold, pvalue, scaling=False
#     )
    
#     # Combine the features
#     train_combined = np.hstack((train_pca, train_geno))
#     test_combined = np.hstack((test_pca, test_geno))
#     val_combined = np.hstack((val_pca, val_geno))
    
#     if scaling:
#         scaler = StandardScaler()
#         train_combined = scaler.fit_transform(train_combined)
#         test_combined = scaler.transform(test_combined)
#         val_combined = scaler.transform(val_combined)
    
#     if feature_selection:
#         train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca = feature_selection(
#             train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca
#         )
    
#     return train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca

# # Dataset 13: PCA + Covariates
# def get_me_pca_covariates_data(phenotype_file, fold, pca, generate_pca=False, scaling=False, feature_selection=False):
#     # Get PCA data
#     train_pca, test_pca, val_pca, train_labels_pca, test_labels_pca, val_labels_pca = get_me_just_pca_data(
#         phenotype_file, fold, pca, generate_pca, scaling=False
#     )
    
#     # Get covariates data
#     train_cov, test_cov, val_cov, train_labels_cov, test_labels_cov, val_labels_cov = get_me_just_covariates_data(
#         phenotype_file, fold, scaling=False
#     )
    
#     # Combine the features
#     train_combined = np.hstack((train_pca, train_cov))
#     test_combined = np.hstack((test_pca, test_cov))
#     val_combined = np.hstack((val_pca, val_cov))
    
#     if scaling:
#         scaler = StandardScaler()
#         train_combined = scaler.fit_transform(train_combined)
#         test_combined = scaler.transform(test_combined)
#         val_combined = scaler.transform(val_combined)
    
#     if feature_selection:
#         train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca = feature_selection(
#             train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca
#         )
    
#     return train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca

# # Dataset 14:  Genotype + Covariates
# def get_me_genotype_covariates_data(phenotype_file, fold, pvalue, scaling=False, feature_selection=False):
#     # Get genotype data
#     train_geno, test_geno, val_geno, train_labels_geno, test_labels_geno, val_labels_geno = get_me_just_genotype_data(
#         phenotype_file, fold, pvalue, scaling=False
#     )
    
#     # Get covariates data
#     train_cov, test_cov, val_cov, train_labels_cov, test_labels_cov, val_labels_cov = get_me_just_covariates_data(
#         phenotype_file, fold, scaling=False
#     )
    
#     # Combine the features
#     train_combined = np.hstack((train_geno, train_cov))
#     test_combined = np.hstack((test_geno, test_cov))
#     val_combined = np.hstack((val_geno, val_cov))
    
#     if scaling:
#         scaler = StandardScaler()
#         train_combined = scaler.fit_transform(train_combined)
#         test_combined = scaler.transform(test_combined)
#         val_combined = scaler.transform(val_combined)
    
#     if feature_selection:
#         train_combined, test_combined, val_combined, train_labels_geno, test_labels_geno, val_labels_geno = feature_selection(
#             train_combined, test_combined, val_combined, train_labels_geno, test_labels_geno, val_labels_geno
#         )
    
#     return train_combined, test_combined, val_combined, train_labels_geno, test_labels_geno, val_labels_geno

# # Dataset 15: PCA + Genotype + Covariates
# def get_me_pca_genotype_covariates_data(phenotype_file, fold, pca, pvalue, generate_pca=False, scaling=False, feature_selection=False):
#     # Get PCA data
#     train_pca, test_pca, val_pca, train_labels_pca, test_labels_pca, val_labels_pca = get_me_just_pca_data(
#         phenotype_file, fold, pca, generate_pca, scaling=False
#     )
    
#     # Get genotype data
#     train_geno, test_geno, val_geno, train_labels_geno, test_labels_geno, val_labels_geno = get_me_just_genotype_data(
#         phenotype_file, fold, pvalue, scaling=False
#     )
    
#     # Get covariates data
#     train_cov, test_cov, val_cov, train_labels_cov, test_labels_cov, val_labels_cov = get_me_just_covariates_data(
#         phenotype_file, fold, scaling=False
#     )
    
#     # Combine all features
#     train_combined = np.hstack((train_pca, train_geno, train_cov))
#     test_combined = np.hstack((test_pca, test_geno, test_cov))
#     val_combined = np.hstack((val_pca, val_geno, val_cov))
    
#     if scaling:
#         scaler = StandardScaler()
#         train_combined = scaler.fit_transform(train_combined)
#         test_combined = scaler.transform(test_combined)
#         val_combined = scaler.transform(val_combined)
    
#     if feature_selection:
#         train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca = feature_selection(
#             train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca
#         )
    
#     return train_combined, test_combined, val_combined, train_labels_pca, test_labels_pca, val_labels_pca







"""
dataset_info = []

 
# Dataset 1: Covariate Data
X_train, X_test, X_val, y_train, y_test, y_val = get_me_just_covariates_data(
    phenotype_file="migraine",
    fold=0,
    scaling=False,
    feature_selection=False
)
add_dataset_info("Dataset 1: Covariate Data (get_me_just_covariates_data)", X_train, X_test, X_val, y_train, y_test, y_val)

# Dataset 2: PCA Data
X_train, X_test, X_val, y_train, y_test, y_val = get_me_just_pca_data(
    phenotype_file="migraine",
    fold=0,
    pca=10,
    generate_pca=True,
    scaling=False,
    feature_selection=False
)
add_dataset_info("Dataset 2: PCA Data (get_me_just_pca_data)", X_train, X_test, X_val, y_train, y_test, y_val)

# Dataset 3: Genotype Data
X_train, X_test, X_val, y_train, y_test, y_val = get_me_just_genotype_data(
    phenotype_file="migraine",
    fold=0,
    pvalue="pv_5.016491878356784e-05",
    scaling=False,
    feature_selection=False,
    weightFile="migraine"+os.sep+"migraine.txt",
    SNPCol="SNP",
    weightCol="BETA"
)
add_dataset_info("Dataset 3: Genotype Data (get_me_just_genotype_data)", X_train, X_test, X_val, y_train, y_test, y_val)

# Dataset 4: PRS Data
X_train, X_test, X_val, y_train, y_test, y_val = get_me_prs_data(
    phenotype_file="migraine",
    fold=0,
    gwas_file="/data/ascher02/uqmmune1/MLPRS/migraine/migraine.gz",
    model="Plink",
    min_pvalue=10,
    num_intervals=200,
    spacing="log",
    scaling=False,
    feature_selection=False
)
add_dataset_info("Dataset 4: PRS Data (get_me_prs_data)", X_train, X_test, X_val, y_train, y_test, y_val)

# Dataset 5: PCA + PRS Data
X_train, X_test, X_val, y_train, y_test, y_val = get_me_pca_prs_data(
    phenotype_file="migraine",
    fold=0,
    pca=10,
    gwas_file="/data/ascher02/uqmmune1/MLPRS/migraine/migraine.gz",
    model="Plink",
    generate_pca=False,
    min_pvalue=10,
    num_intervals=200,
    spacing="log",
    scaling=False,
    feature_selection=False
)
add_dataset_info("Dataset 5: PCA + PRS Data (get_me_pca_prs_data)", X_train, X_test, X_val, y_train, y_test, y_val)

# Dataset 6: PCA + PRS + Covariates Data
X_train, X_test, X_val, y_train, y_test, y_val = get_me_pca_prs_covariates_data(
    phenotype_file="migraine",
    fold=0,
    pca=10,
    gwas_file="/data/ascher02/uqmmune1/MLPRS/migraine/migraine.gz",
    model="Plink",
    generate_pca=False,
    min_pvalue=10,
    num_intervals=200,
    spacing="log",
    scaling=False,
    feature_selection=False
)
add_dataset_info("Dataset 6: PCA + PRS + Covariates Data (get_me_pca_prs_covariates_data)", X_train, X_test, X_val, y_train, y_test, y_val)

# Dataset 7: PRS + Genotype Data
X_train, X_test, X_val, y_train, y_test, y_val = get_me_prs_genotype_data(
    phenotype_file="migraine",
    fold=0,
    pvalue="pv_5.016491878356784e-05",
    gwas_file="/data/ascher02/uqmmune1/MLPRS/migraine/migraine.gz",
    model="Plink",
    min_pvalue=10,
    num_intervals=200,
    spacing="log",
    scaling=False,
    feature_selection=False
)
add_dataset_info("Dataset 7: PRS + Genotype Data (get_me_prs_genotype_data)", X_train, X_test, X_val, y_train, y_test, y_val)

# Dataset 8: PRS + Covariates Data
X_train, X_test, X_val, y_train, y_test, y_val = get_me_prs_covariates_data(
    phenotype_file="migraine",
    fold=0,
    gwas_file="/data/ascher02/uqmmune1/MLPRS/migraine/migraine.gz",
    model="Plink",
    min_pvalue=10,
    num_intervals=200,
    spacing="log",
    scaling=False,
    feature_selection=False
)
add_dataset_info("Dataset 8: PRS + Covariates Data (get_me_prs_covariates_data)", X_train, X_test, X_val, y_train, y_test, y_val)

# Dataset 9: PCA + PRS + Genotype Data
X_train, X_test, X_val, y_train, y_test, y_val = get_me_pca_prs_genotype_data(
    phenotype_file="migraine",
    fold=0,
    pca=10,
    pvalue="pv_5.016491878356784e-05",
    gwas_file="/data/ascher02/uqmmune1/MLPRS/migraine/migraine.gz",
    model="Plink",
    generate_pca=False,
    min_pvalue=10,
    num_intervals=200,
    spacing="log",
    scaling=False,
    feature_selection=False
)
add_dataset_info("Dataset 9: PCA + PRS + Genotype Data (get_me_pca_prs_genotype_data)", X_train, X_test, X_val, y_train, y_test, y_val)

# Dataset 10: PRS + Genotype + Covariates Data
X_train, X_test, X_val, y_train, y_test, y_val = get_me_prs_genotype_covariates_data(
    phenotype_file="migraine",
    fold=0,
    pvalue="pv_5.016491878356784e-05",
    gwas_file="/data/ascher02/uqmmune1/MLPRS/migraine/migraine.gz",
    model="Plink",
    min_pvalue=10,
    num_intervals=200,
    spacing="log",
    scaling=False,
    feature_selection=False
)
add_dataset_info("Dataset 10: PRS + Genotype + Covariates Data (get_me_prs_genotype_covariates_data)", X_train, X_test, X_val, y_train, y_test, y_val)

# Dataset 11: PCA + PRS + Genotype + Covariates Data
X_train, X_test, X_val, y_train, y_test, y_val = get_me_pca_prs_genotype_covariates_data(
    phenotype_file="migraine",
    fold=0,
    pca=10,
    pvalue="pv_5.016491878356784e-05",
    gwas_file="/data/ascher02/uqmmune1/MLPRS/migraine/migraine.gz",
    model="Plink",
    generate_pca=False,
    min_pvalue=10,
    num_intervals=200,
    spacing="log",
    scaling=False,
    feature_selection=False
)
add_dataset_info("Dataset 11: PCA + PRS + Genotype + Covariates Data (get_me_pca_prs_genotype_covariates_data)", X_train, X_test, X_val, y_train, y_test, y_val)

# Dataset 12: PCA + Genotype Data
X_train, X_test, X_val, y_train, y_test, y_val = get_me_pca_genotype_data(
    phenotype_file="migraine",
    fold=0,
    pca=10,
    pvalue="pv_5.016491878356784e-05",
    generate_pca=False,
    scaling=False,
    feature_selection=False
)
add_dataset_info("Dataset 12: PCA + Genotype Data (get_me_pca_genotype_data)", X_train, X_test, X_val, y_train, y_test, y_val)

# Dataset 13: PCA + Covariates Data
X_train, X_test, X_val, y_train, y_test, y_val = get_me_pca_covariates_data(
    phenotype_file="migraine",
    fold=0,
    pca=10,
    generate_pca=False,
    scaling=False,
    feature_selection=False
)
add_dataset_info("Dataset 13: PCA + Covariates Data (get_me_pca_covariates_data)", X_train, X_test, X_val, y_train, y_test, y_val)

# Dataset 14: Genotype + Covariates Data
X_train, X_test, X_val, y_train, y_test, y_val = get_me_genotype_covariates_data(
    phenotype_file="migraine",
    fold=0,
    pvalue="pv_5.016491878356784e-05",
    scaling=False,
    feature_selection=False
)
add_dataset_info("Dataset 14: Genotype + Covariates Data (get_me_genotype_covariates_data)", X_train, X_test, X_val, y_train, y_test, y_val)

# Dataset 15: PCA + Genotype + Covariates Data
X_train, X_test, X_val, y_train, y_test, y_val = get_me_pca_genotype_covariates_data(
    phenotype_file="migraine",
    fold=0,
    pca=10,
    pvalue="pv_5.016491878356784e-05",
    generate_pca=False,
    scaling=False,
    feature_selection=False
)
add_dataset_info("Dataset 15: PCA + Genotype + Covariates Data (get_me_pca_genotype_covariates_data)", X_train, X_test, X_val, y_train, y_test, y_val)


exit(0)

# Example usage
results = []
#X_train, X_test, X_val, y_train, y_test, y_val  = get_me_pca_prs_covariates_data("migraine", fold, 10, "/data/ascher02/uqmmune1/MLPRS/migraine/migraine.gz", model="LDAK-GWAS", generate_pca=False, min_pvalue=10, num_intervals=200, spacing="log", scaling=False, feature_selection=False)



for fold in range(5):  # Simulate 5 runs
    X_train, X_test, X_val, y_train, y_test, y_val  = get_me_prs_data("migraine", fold, "/data/ascher02/uqmmune1/MLPRS/migraine/migraine.gz", model="LDpred-fast", min_pvalue=10, num_intervals=200, spacing="log", scaling=False, feature_selection=False)
 
    auc_train, auc_test, auc_val, model = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, X_val, y_val
    )
    results.append({
        "Fold": fold + 1,
        "Train AUC": auc_train,
        "Test AUC": auc_test,
        "Validation AUC": auc_val
    })
    print(f"Run {fold + 1} - Train AUC: {auc_train:.4f}, Test AUC: {auc_test:.4f}, Validation AUC: {auc_val:.4f}")


# Convert results to pandas DataFrame and print in Markdown format
results_df = pd.DataFrame(results)
print(results_df.to_markdown(index=False))




exit(0)

"""







# phenotypes = [ 
#                 "depression"  ]
# gwas_files = [ 
#                 "depression_11.gz"  ]
 
# Models = [ "PRSice-2", "LDpred-fast", "AnnoPred", "LDAK-GWAS"]
# data = []
# model_names = []
# for phenotype, gwas in zip(phenotypes, gwas_files):
#     for model in Models:
#         for fold in range(0, 1):
#             print(f"Processing {phenotype} ({gwas}) with {model} model for fold {fold}")
#             df = get_me_prs_based_gwas_file(phenotype, gwas, fold, model=model)
#             data.append(df)
#             model_names.append(model)

# # Function to create suffixes for each merge operation
# def merge_with_suffixes(left, right, left_model, right_model):
#     return pd.merge(
#         left, right,
#         on="SNP",
#         how="inner",
#         suffixes=(f'_{left_model}', f'_{right_model}')
#     )

# # Merge dataframes sequentially with proper suffixes
# merged_df = data[0].copy()
# current_model = model_names[0]

# for i in range(1, len(data)):
#     merged_df = merge_with_suffixes(
#         merged_df,
#         data[i],
#         current_model,
#         model_names[i]
#     )
#     current_model = model_names[i]

# # Extract all columns containing "OR"
# or_columns = [col for col in merged_df.columns if 'OR' in col]

# # Compute correlation matrix
# correlation_matrix = merged_df[or_columns].corr()
# print(correlation_matrix)


# or_columns = [col for col in merged_df.columns if 'OR' in col]

# # Create a figure with appropriate size
# plt.figure(figsize=(15, 8))

# # Plot each OR column
# for col in or_columns:
#     # Sort values for a smoother line plot
#     sorted_vals = merged_df[col].sort_values().reset_index(drop=True)
#     plt.plot(range(len(sorted_vals)), sorted_vals, label=col, alpha=0.7)

# # Customize the plot
# plt.xlabel('SNPs (sorted by OR value)')
# plt.ylabel('Odds Ratio (OR)')
# plt.title('Distribution of OR Values Across SNPs for Different Models')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True, alpha=0.3)

# # Adjust layout to prevent label cutoff
# plt.tight_layout()

# # Show the plot
# plt.savefig("OR_values.png")

# # Print summary statistics
# print("\nSummary Statistics for OR values:")
# print(merged_df[or_columns].describe())

# # Create a second plot with boxplots
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=merged_df[or_columns])
# plt.xticks(rotation=45, ha='right')
# plt.title('Boxplot of OR Values by Model')
# plt.ylabel('Odds Ratio (OR)')
# plt.tight_layout()
# plt.savefig("Boxplot.png")


# exit(0)