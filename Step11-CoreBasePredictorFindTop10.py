import pandas as pd
import os
import sys
from tqdm import tqdm
import numpy as np

def aggregate_fold_results(phenotype,directory, dataset_num):
    """
    Aggregate results for a specific dataset across all folds with stability metrics
    """
    all_fold_results = []
    base_path = phenotype
    fold_dirs = [d for d in os.listdir(base_path) if d.startswith('Fold_')]
    
    for fold_dir in fold_dirs:
        fold_num = fold_dir.split('_')[1]
        result_file = f"{base_path}/{fold_dir}/{directory}/dataset_{dataset_num}_results.csv"
        
        if os.path.exists(result_file):
            try:
                fold_result = pd.read_csv(result_file)
                fold_result['Fold'] = fold_num
                all_fold_results.append(fold_result)
            except Exception as e:
                print(f"\nError reading fold {fold_num} results for dataset {dataset_num}: {str(e)}")
                continue
    
    if not all_fold_results:
        print(f"\nNo results found for dataset {dataset_num}")
        return None


    combined_results = pd.concat(all_fold_results, ignore_index=True)

  




    if 'ML_Parameters' not in combined_results.columns:
        # Calculate mean and stability metrics per model configuration
        stability_metrics = combined_results.groupby(['Model' ]).agg({
            'Train AUC': ['mean', 'std'],
            'Validation AUC': ['mean', 'std'],
            'Test AUC': ['mean', 'std'],
            'Phenotype': 'first',
            'Dataset': 'first',
            'Fold': lambda x: f"Folds: {sorted(list(x))}"
        })
    else: 
        # Calculate mean and stability metrics per model configuration
        stability_metrics = combined_results.groupby(['Model', 'ML_Parameters']).agg({
            'Train AUC': ['mean', 'std'],
            'Validation AUC': ['mean', 'std'],
            'Test AUC': ['mean', 'std'],
            'Phenotype': 'first',
            'Dataset': 'first',
            'Fold': lambda x: f"Folds: {sorted(list(x))}"
        })
        
    # Flatten column names
    stability_metrics.columns = [
        f"{col[0]}_{col[1]}" if col[1] != '' else col[0]
        for col in stability_metrics.columns
    ]
    
    mean_results = stability_metrics.reset_index()
    
    # Add number of folds and calculate stability scores
    #mean_results['Folds_Aggregated'] = mean_results['Fold'].apply(lambda x: len(eval(x.replace('Folds: ', ''))))
    mean_results['Train_Stability'] = 1 / (1 + mean_results['Train AUC_std'])
    mean_results['Val_Stability'] = 1 / (1 + mean_results['Validation AUC_std'])
    
    datasetinformation = pd.read_csv(f"{phenotype}/Fold_0/Datasets/dataset_tracking.csv")
    matchingtext = "dataset_"+str(dataset_num)
    datasetinfo = datasetinformation[datasetinformation['Dataset_Name']==matchingtext]
    datasetinfo['Dataset_Name'] = datasetinfo['Dataset_Name'].str.replace("dataset_","",regex=False)

    
    mean_results['Dataset_first'] = mean_results['Dataset_first'].astype(str)
  

    mean_results = pd.merge(
        mean_results,
        datasetinfo,
        left_on='Dataset_first',
        right_on='Dataset_Name',
        how='left'
    )

    # Update the 'Dataset' column with the values from 'Dataset_Name'
    mean_results['Dataset'] = mean_results['Dataset_Name']

    # Drop the redundant 'Dataset_Name' column
    mean_results = mean_results.drop('Dataset_Name', axis=1)



    # Rename mean columns for clarity
    mean_results = mean_results.rename(columns={
        'Train AUC_mean': 'Train AUC',
        'Validation AUC_mean': 'Validation AUC',
        'Test AUC_mean': 'Test AUC'
    })
    
    return mean_results
        
   

def select_best_models(results_df, min_auc=0.5, min_folds=3):
    """
    Select best models using multi-stage filtering with stability metrics
    """
    valid_models = results_df[
        (results_df["Validation AUC"] >= min_auc) &
        (results_df["Train AUC"] >= min_auc) &
        (results_df["Train AUC"] >= results_df["Validation AUC"]) &
        (results_df["Train AUC"] <= 1.0) &
        (results_df["Validation AUC"] <= 1.0)
    ].copy()
    
 




    if valid_models.empty:
        return pd.DataFrame()
    
    # Calculate evaluation metrics
    valid_models['Train_Val_Gap'] = abs(valid_models['Train AUC'] - valid_models['Validation AUC'])
    
    # Normalize all metrics to 0-1 scale
    metrics_to_normalize = [
        'Validation AUC', 'Train_Val_Gap', 'Train_Stability', 'Val_Stability'
    ]
    
    for metric in metrics_to_normalize:
        if metric == 'Validation AUC':
            valid_models[f'{metric}_Norm'] = (valid_models[metric] - min_auc) / (1 - min_auc)
        elif metric in ['Train_Stability', 'Val_Stability']:
            # Stability scores are already normalized by definition
            valid_models[f'{metric}_Norm'] = valid_models[metric]
        else:
            max_val = valid_models[metric].max()
            valid_models[f'{metric}_Norm'] = 1 - (valid_models[metric] / max_val)
    
    # Calculate composite score including stability
    valid_models['Composite_Score'] = (
        0.2 * valid_models['Validation AUC_Norm'] +
        0.2 * valid_models['Train_Val_Gap_Norm'] +
        0.2 * valid_models['Train_Stability_Norm'] +
        0.2 * valid_models['Val_Stability_Norm']
    )
  
    try:
        # Select best model per type based on composite score
        best_models = valid_models.loc[valid_models.groupby('Model')['Composite_Score'].idxmax()]
        best_models['Overall_Rank'] = best_models['Composite_Score'].rank(ascending=False)
        
        return best_models.sort_values('Composite_Score', ascending=False)
    except Exception as e:
        return pd.DataFrame()

def process_dataset(input_file, dataset_num, min_auc, min_folds=3):
    """
    Process single dataset's aggregated results with stability metrics
    """
    print(f"\nProcessing dataset {dataset_num}")

    results = pd.read_csv(input_file)
    print(f"Found {len(results)} models from {len(results['Model'].unique())} model types")
    
    best_models = select_best_models(results, min_auc, min_folds)
    
    if not best_models.empty:
        output_file = input_file + "_selected"
        best_models.to_csv(output_file, index=False)
        
        report_file = input_file + "_performance_report.csv"
        if 'ML_Parameters' in best_models.columns:
            performance_report = best_models[[
                'Model', 'ML_Parameters', 
                'Train AUC', 'Validation AUC',
                'Train_Val_Gap', 'Train_Stability', 'Val_Stability',
                'Composite_Score', 'Overall_Rank' 
            ]]
            performance_report.to_csv(report_file, index=False)
        else:
            performance_report = best_models[[
                'Model',  
                'Train AUC', 'Validation AUC',
                'Train_Val_Gap', 'Train_Stability', 'Val_Stability',
                'Composite_Score', 'Overall_Rank' 
            ]]
            performance_report.to_csv(report_file, index=False)

        
    return best_models

   
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
 
def create_detailed_label(row):
    """Create detailed label based on dataset type"""
    base_label = f"D{row['Dataset']}"
    
    # Helper function to extract SNP count
    def get_snp_info(snp_str):
        if pd.isna(snp_str):
            return 'NA'
        if 'annotated' in str(snp_str).lower():
            return f"SNPs_annotated_{str(snp_str).split('_')[-1]}"
        elif 'snps' in str(snp_str).lower():
            return f"SNPs_{str(snp_str).split('_')[-1]}"
        return 'NA'
    
    if row['Dataset_Type'].lower() == 'genotype':
        # For genotype datasets, extract SNP count and handle weight info
        snp_info = get_snp_info(row.get('snps'))
        weight_status = "W" if row.get('weight_file_present', False) else 'UW'
        gwas_name = os.path.basename(str(row.get('gwas_file', ''))).replace('.gz', '')
        return f"{base_label} | G | {snp_info} | {weight_status} | {gwas_name}"
    
    elif row['Dataset_Type'].lower() == 'prs':
        # For PRS datasets, include model, SNPs and GWAS info
        model_name = str(row.get('model', 'NA'))
        gwas_name = os.path.basename(str(row.get('gwas_file', ''))).replace('.gz', '')
        snp_info = get_snp_info(row.get('snps'))
        return f"{base_label} | PRS | {model_name} | {gwas_name}"
    
    elif row['Dataset_Type'].lower() == 'covariates':
        # For covariate datasets, include feature count if available
        feature_count = row.get('Features', 'NA')
        snp_info = get_snp_info(row.get('snps'))
        return f"{base_label} | C"
    
    elif row['Dataset_Type'].lower() == 'pca':
        # For PCA datasets, include component count and SNPs if available
        pca_components = row.get('pca_components', 'NA')
        snp_info = get_snp_info(row.get('snps'))
        return f"{base_label} | PCA:{pca_components}"
    
    else:
        # Default case for any other dataset types
        snp_info = get_snp_info(row.get('snps'))
        return f"{base_label} | {row['Dataset_Type']}"

def create_publication_plots(phenotype, directory):
    """
    Create and save publication-quality plots for model performance analysis
    """
    # Read results
    results_file = f"{phenotype}/Results/{directory}/ResultsFinal.csv"
    results_df = pd.read_csv(results_file)
    
    # Create output directory
    plot_dir = f"{phenotype}/Results/{directory}/Plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Set publication-ready style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.rcParams['font.family'] = 'sans-serif'
    #plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    # Calculate figure width based on number of datasets
    n_datasets = len(results_df)
    fig_width = max(15, n_datasets * 0.4)
    
    # Create figure
    plt.figure(figsize=(fig_width, 12))  # Increased height for vertical labels
    
    # Setup data
    x = np.arange(len(results_df))
    width = 0.25
    
    # Convert values to numeric
    train_auc = pd.to_numeric(results_df['Train AUC'], errors='coerce')
    val_auc = pd.to_numeric(results_df['Validation AUC'], errors='coerce')
    test_auc = pd.to_numeric(results_df['Test AUC'], errors='coerce')
    
    # Create bars
    plt.bar(x - width, train_auc, width, label='Train AUC', 
            alpha=0.8, edgecolor='black', linewidth=1, color='#ff9999')
    plt.bar(x, val_auc, width, label='Validation AUC', 
            alpha=0.8, edgecolor='black', linewidth=1, color='#66b3ff')
    plt.bar(x + width, test_auc, width, label='Test AUC', 
            alpha=0.8, edgecolor='black', linewidth=1, color='#99ff99')
    
    # Create detailed labels
    labels = [create_detailed_label(row) for _, row in results_df.iterrows()]
    
    # Customize axes
    plt.xlabel('Dataset Information', fontsize=12, fontweight='bold', labelpad=70)  # Increased labelpad
    plt.ylabel('AUC Score', fontsize=12, fontweight='bold')
    plt.title(f'Model Performance Comparison for {phenotype.capitalize()}\nAcross Different Dataset Types', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Customize legend
    legend = plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                       frameon=True, edgecolor='black')
    legend.get_frame().set_linewidth(1)
    
    # Set x-ticks with vertical labels
    plt.xticks(x, labels, rotation=90, ha='center', va='top')
    
    # Adjust label positions for better visibility
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', pad=30)  # Increase padding for x-axis labels
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits
    plt.ylim(0.4, 1.05)
    
    # Add horizontal lines
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=0.7, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figures with extra bottom margin for labels
    plt.savefig(f"{plot_dir}/performance_comparison.png", 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none',
                pad_inches=0.5)  # Added padding
    plt.savefig(f"{plot_dir}/performance_comparison.pdf", 
                format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none',
                pad_inches=0.5)  # Added padding
    plt.close()
    
    print(f"\nEnhanced publication-quality plots saved in: {plot_dir}")
    print("Files created:")
    print("1. performance_comparison.png - High-resolution PNG")
    print("2. performance_comparison.pdf - Vector PDF for publication")

def display_best_model_performances(phenotype,directory):
    """
    Display and save comprehensive performance analysis including all model parameters and metrics
    """
    results_dir = f"{phenotype}/Results/{directory}/Datasets"
    input_files = [f for f in os.listdir(results_dir) if f.endswith('_aggregated_results.csv_selected')]
    dataset_numbers = sorted([int(f.split('_')[1]) for f in input_files])
    
    all_results = []
    
    for dataset_num in dataset_numbers:
        input_file = f"{results_dir}/dataset_{dataset_num}_aggregated_results.csv_selected"
        
        try:
            results = pd.read_csv(input_file)
            if results.empty:
                print(f"\nDataset {dataset_num}: No valid models found")
                continue
                
            best_model = results.loc[results['Composite_Score'].idxmax()]
            result_row = best_model.to_dict()
            result_row['Dataset'] = dataset_num
            all_results.append(result_row)
            
            print(f"\nDataset {dataset_num}:")
            print(f"{best_model['Model']:<20} "
                  f"{best_model['Train AUC']:>8.4f} "
                  f"{best_model['Validation AUC']:>8.4f} "
                  f"{best_model['Test AUC']:>8.4f} "
                  f"{best_model['Train_Val_Gap']:>12.4f} "
                  f"{best_model['Val_Stability']:>12.4f} "
                  f"{best_model['Composite_Score']:>12.4f}")
            
        except Exception as e:
            print(f"\nError processing dataset {dataset_num}: {str(e)}")
            continue
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        key_columns = [
            'Dataset', 
            'Model', 
            'ML_Parameters',
            'Train AUC', 
            'Validation AUC', 
            'Test AUC',
            'Train_Val_Gap',
            'Val_Stability',
            'Train_Stability',
            'Composite_Score',
            'Overall_Rank',
            'Folds_Aggregated',
            'Fold',
            'Phenotype'
        ]
        
        other_columns = [col for col in results_df.columns if col not in key_columns]
        all_columns = key_columns + other_columns
        final_columns = [col for col in all_columns if col in results_df.columns]
        results_df = results_df[final_columns]
        
        os.makedirs(f"{phenotype}/Results/{directory}/", exist_ok=True)
        output_file = f"{phenotype}/Results/{directory}/ResultsFinal.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nFull results saved to: {output_file}")
        
        # Create publication plots
        create_publication_plots(phenotype,directory)
        
        print("\nSaved columns:")
        for col in final_columns:
            print(f"- {col}")
    else:
        print("\nNo results to save")

# Rest of the code (process_all_datasets and main) remains the same
def process_all_datasets(phenotype, directory, start_dataset=1, end_dataset=174):
    """
    Process all datasets from start_dataset to end_dataset
    """
    print(f"\nProcessing datasets {start_dataset} to {end_dataset} for {phenotype}")
    
    output_dir = f"{phenotype}/Results/{directory}/Datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    all_aggregated_results = []

    # read file phenotype/Results/ UniqueDatasets.txt
    # for each line extract line.split("_")[1]
    


    for dataset_num in tqdm(range(start_dataset, end_dataset + 1), 
                           desc="Processing datasets"):
        
        results = aggregate_fold_results(phenotype,directory, dataset_num)
        
        if results is not None:
            output_file = f"{output_dir}/dataset_{dataset_num}_aggregated_results.csv"
            results.to_csv(output_file, index=False)
            all_aggregated_results.append(results)
    
    if all_aggregated_results:
        all_results_df = pd.concat(all_aggregated_results, ignore_index=True)
        summary_file = f"{output_dir}/all_datasets_summary.csv"
        all_results_df.to_csv(summary_file, index=False)
        
        print(f"\nProcessed {len(all_aggregated_results)} datasets successfully")
        print(f"Summary file saved to: {summary_file}")
        
        model_stats = all_results_df.groupby('Model').agg({
            'Train AUC': ['mean', 'std'],
            'Validation AUC': ['mean', 'std'],
            'Test AUC': ['mean', 'std'],
            'Dataset_first': 'count'
        })
        
        stats_file = f"{output_dir}/model_statistics_summary.csv"
        model_stats.to_csv(stats_file)
        print(f"\nDetailed model statistics saved to: {stats_file}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python CoreBasePredictorFindTop10.py <phenotype> <directory>")
        sys.exit(1)
    
    phenotype = sys.argv[1]
    directory = sys.argv[2]
    
    # Clean up previous results directory if it exists
    results_dir = f"{phenotype}/Results/{directory}/Datasets/"
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    results_dir = f"{phenotype}/Results/{directory}/Plots/"
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    

    # Step 1: Process and aggregate all datasets
    process_all_datasets(phenotype,directory)
    
    # Step 2: Process results directory for model selection
    results_dir = f"{phenotype}/Results/{directory}/Datasets"
    print(f"\nSelecting best models for {phenotype}")
    
    input_files = [f for f in os.listdir(results_dir) if f.endswith('_aggregated_results.csv')]
    dataset_numbers = sorted([int(f.split('_')[1]) for f in input_files])
    
    for dataset_num in tqdm(dataset_numbers, desc="Selecting best models"):
        input_file = f"{results_dir}/dataset_{dataset_num}_aggregated_results.csv"
        if os.path.exists(input_file):
            process_dataset(input_file, dataset_num,min_auc=float(sys.argv[3]))
   
    # Step 3: Display final results
    display_best_model_performances(phenotype,directory)

if __name__ == "__main__":
    main()