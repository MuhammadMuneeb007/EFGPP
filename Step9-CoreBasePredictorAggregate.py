import pandas as pd
import os
import sys
import numpy as np

def aggregate_fold_results(phenotype, directory, dataset_num):
    """Aggregate results for a specific dataset across all folds with stability metrics"""
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
            except:
                continue

    if not all_fold_results:
        return None

    combined_results = pd.concat(all_fold_results, ignore_index=True)
    
    group_cols = ['Model', 'ML_Parameters'] if 'ML_Parameters' in combined_results.columns else ['Model']
    
    stability_metrics = combined_results.groupby(group_cols).agg({
        'Train AUC': ['mean', 'std'],
        'Validation AUC': ['mean', 'std'],
        'Test AUC': ['mean', 'std'],
        'Phenotype': 'first',
        'Dataset': 'first',
        'Fold': lambda x: sorted(list(x))
    })
    
    stability_metrics.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0]
                               for col in stability_metrics.columns]
    
    mean_results = stability_metrics.reset_index()
    mean_results['Train_Stability'] = 1 / (1 + mean_results['Train AUC_std'])
    mean_results['Val_Stability'] = 1 / (1 + mean_results['Validation AUC_std'])
    
    datasetinformation = pd.read_csv(f"{phenotype}/Fold_0/Datasets/dataset_tracking.csv")
    datasetinfo = datasetinformation[datasetinformation['Dataset_Name'] == f"dataset_{dataset_num}"]
    datasetinfo['Dataset_Name'] = datasetinfo['Dataset_Name'].str.replace("dataset_", "", regex=False)
    
    mean_results['Dataset_first'] = mean_results['Dataset_first'].astype(str)
    mean_results = pd.merge(
        mean_results,
        datasetinfo,
        left_on='Dataset_first',
        right_on='Dataset_Name',
        how='left'
    )
    
    mean_results['Dataset'] = mean_results['Dataset_Name']
    mean_results = mean_results.drop('Dataset_Name', axis=1)
    
    mean_results = mean_results.rename(columns={
        'Train AUC_mean': 'Train AUC',
        'Validation AUC_mean': 'Validation AUC',
        'Test AUC_mean': 'Test AUC'
    })
    
    return mean_results

def select_best_models(results_df, min_auc=0.5):
    """Select best models using multi-stage filtering with stability metrics"""
    valid_models = results_df[
        (results_df["Validation AUC"] >= min_auc) &
        (results_df["Train AUC"] >= min_auc) &
        (results_df["Train AUC"] >= results_df["Validation AUC"]) &
        (results_df["Train AUC"] <= 1.0) &
        (results_df["Validation AUC"] <= 1.0)
    ].copy()
    
    if valid_models.empty:
        return pd.DataFrame()
    
    valid_models['Train_Val_Gap'] = abs(valid_models['Train AUC'] - valid_models['Validation AUC'])
    
    metrics_to_normalize = ['Validation AUC', 'Train_Val_Gap', 'Train_Stability', 'Val_Stability']
    
    for metric in metrics_to_normalize:
        if metric == 'Validation AUC':
            valid_models[f'{metric}_Norm'] = (valid_models[metric] - min_auc) / (1 - min_auc)
        elif metric in ['Train_Stability', 'Val_Stability']:
            valid_models[f'{metric}_Norm'] = valid_models[metric]
        else:
            max_val = valid_models[metric].max()
            valid_models[f'{metric}_Norm'] = 1 - (valid_models[metric] / max_val)
    
    valid_models['Composite_Score'] = (
        0.2 * valid_models['Validation AUC_Norm'] +
        0.2 * valid_models['Train_Val_Gap_Norm'] +
        0.2 * valid_models['Train_Stability_Norm'] +
        0.2 * valid_models['Val_Stability_Norm']
    )
    
    try:
        best_models = valid_models.loc[valid_models.groupby('Model')['Composite_Score'].idxmax()]
        best_models['Overall_Rank'] = best_models['Composite_Score'].rank(ascending=False)
        return best_models.sort_values('Composite_Score', ascending=False)
    except:
        return pd.DataFrame()

def process_and_save_results(phenotype, directory, datasets, min_auc=0.5):
    """Process all datasets and save final results"""
    all_results = []
    
    for dataset_num in datasets:
        results = aggregate_fold_results(phenotype, directory, dataset_num)
        if results is not None:
            best_models = select_best_models(results, min_auc)
            if not best_models.empty:
                best_model = best_models.loc[best_models['Composite_Score'].idxmax()]
                result_row = best_model.to_dict()
                result_row['Dataset'] = dataset_num
                all_results.append(result_row)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        key_columns = [
            'Dataset', 'Model', 'ML_Parameters', 'Train AUC', 'Validation AUC', 
            'Test AUC', 'Train_Val_Gap', 'Val_Stability', 'Train_Stability',
            'Composite_Score', 'Overall_Rank', 'Fold', 'Phenotype'
        ]
        other_columns = [col for col in results_df.columns if col not in key_columns]
        final_columns = [col for col in key_columns + other_columns if col in results_df.columns]
        
        results_df = results_df[final_columns]
        os.makedirs(f"{phenotype}/Results/{directory}/Aggregated", exist_ok=True)
        results_df.to_csv(f"{phenotype}/Results/{directory}/Aggregated/ResultsFinal.csv", index=False)

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <phenotype> <directory> <min_auc>")
        sys.exit(1)
    
    phenotype = sys.argv[1]
    directory = sys.argv[2]
    min_auc = float(sys.argv[3])
    
    # Read unique datasets
    with open(f"{phenotype}/Results/UniqueDatasets.txt", 'r') as f:
        datasets = [int(x.replace("dataset_", "")) for x in f.read().splitlines()]
    
    process_and_save_results(phenotype, directory, datasets, min_auc)

if __name__ == "__main__":
    main()