import pandas as pd
import os
import logging
from collections import defaultdict
import numpy as np
import pandas as pd
import os
import sys
import argparse
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.metrics import AUC
import numpy as np
import pandas as pd
import os
import sys
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.metrics import AUC
import numpy as np
import tensorflow as tf
import random
import os
import numpy as np
import pandas as pd
import os
import sys
import logging
from typing import List, Dict, Set, Tuple
import itertools
from collections import defaultdict
import random
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.metrics import AUC

import pandas as pd
import os
import logging
from collections import defaultdict

import os
import random
import numpy as np
import tensorflow as tf
import shutil
 
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
 

def load_and_preprocess_data(base_path, dataset_num, fold):
    """Load and preprocess data for a specific fold"""
    fold_dir = os.path.join(base_path, f"Fold_{fold}", "Datasets")
    train_x_path = f"{fold_dir}/dataset_{dataset_num}/dataset_{dataset_num}_X_train.csv"
    train_y_path = f"{fold_dir}/dataset_{dataset_num}/dataset_{dataset_num}_y_train.csv"
    test_x_path = f"{fold_dir}/dataset_{dataset_num}/dataset_{dataset_num}_X_test.csv"
    test_y_path = f"{fold_dir}/dataset_{dataset_num}/dataset_{dataset_num}_y_test.csv"
    val_x_path = f"{fold_dir}/dataset_{dataset_num}/dataset_{dataset_num}_X_val.csv"
    val_y_path = f"{fold_dir}/dataset_{dataset_num}/dataset_{dataset_num}_y_val.csv"
    
    X_train = pd.read_csv(train_x_path).iloc[:, 2:].values
    y_train = pd.read_csv(train_y_path).values.ravel()
    X_test = pd.read_csv(test_x_path).iloc[:, 2:].values
    y_test = pd.read_csv(test_y_path).values.ravel()
    X_val = pd.read_csv(val_x_path).iloc[:, 2:].values
    y_val = pd.read_csv(val_y_path).values.ravel()
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    
    return X_train, y_train, X_test, y_test, X_val, y_val

def get_best_datasets(results_path, n_per_type=2):
    """Get the top performing datasets from results file"""
    try:
        results_df = pd.read_csv(results_path)
        selected_datasets = []
        
        for dataset_type in results_df['Dataset_Type'].unique():
            type_results = results_df[results_df['Dataset_Type'] == dataset_type]
            top_n = type_results.nlargest(n_per_type, 'Validation AUC')
            selected_datasets.extend(top_n['Dataset'].tolist())
        
        best_by_type = results_df[results_df['Dataset'].isin(selected_datasets)].sort_values(
            ['Dataset_Type', 'Validation AUC'], ascending=[True, False]
        )
        
        return selected_datasets, best_by_type

    except Exception as e:
        logging.error(f"Error selecting best datasets: {str(e)}")
        raise

def create_simple_model(input_dim, name):
    """Create a simple neural network"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[AUC(name='auc')]
    )
    
    return model

def create_meta_model(input_dim):
    """Create meta-model to combine predictions"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[AUC(name='auc')]
    )
    
    return model
def train_individual_models(phenotype_dir, datasets, fold):
    """Train individual models for each dataset"""
    individual_models = []
    predictions_train = []
    predictions_val = []
    predictions_test = []
    
    # Get first dataset for labels
    first_data = load_and_preprocess_data(phenotype_dir, datasets[0], fold)
    y_train, y_test, y_val = first_data[1], first_data[3], first_data[5]
    
    # Train model for each dataset
    for i, dataset in enumerate(datasets):
        print(f"\nTraining model for dataset {dataset}")
        
        # Load and preprocess data
        X_train, y_train, X_test, y_test, X_val, y_val = load_and_preprocess_data(
            phenotype_dir, dataset, fold
        )
        
        # Create and train model
        model = create_simple_model(X_train.shape[1], f"model_{dataset}")
        
        # Calculate class weights
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        class_weight = {0: 1.0, 1: n_neg/n_pos}
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_auc',
                patience=10,
                mode='max',
                restore_best_weights=True
            )
        ]
        
        # Train
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks_list,
            class_weight=class_weight,
            verbose=0
        )
        
        # Get predictions
        train_preds = model.predict(X_train, verbose=0)
        val_preds = model.predict(X_val, verbose=0)
        test_preds = model.predict(X_test, verbose=0)
        
        predictions_train.append(train_preds)
        predictions_val.append(val_preds)
        predictions_test.append(test_preds)
        individual_models.append(model)
        
        # Print individual model performance
        print(f"Dataset {dataset} validation AUC: {roc_auc_score(y_val, val_preds):.4f}")
    
    # Combine predictions
    X_meta_train = np.column_stack(predictions_train)
    X_meta_val = np.column_stack(predictions_val)
    X_meta_test = np.column_stack(predictions_test)
    
    return (X_meta_train, y_train, X_meta_test, y_test, X_meta_val, y_val, 
            individual_models)

def get_unique_category_datasets(results_path, metric='Validation AUC', n_per_category=1):
    """
    Get unique categories and select best performing dataset from each category.
    Categories consider annotation status and GWAS file for Genotype and PRS.
    """
    try:
        # Load results file
        results_df = pd.read_csv(results_path)
        categories = defaultdict(list)
        
        def get_annotation_status(snp_str):
            """Helper function to determine annotation status"""
            if pd.isna(snp_str):
                return 'NA'
            snp_str = str(snp_str).lower()
            if 'annotated' in snp_str:
                return 'annotated'
            return 'not_annotated'
        
        def get_gwas_name(gwas_file):
            """Helper function to extract GWAS name from file path"""
            if pd.isna(gwas_file):
                return 'NA'
            return os.path.basename(str(gwas_file)).replace('.gz', '')
        
        # First pass: Categorize all datasets
        for _, row in results_df.iterrows():
            dataset_id = row['Dataset']
            dataset_type = row['Dataset_Type'].lower()
            annotation_status = get_annotation_status(row.get('snps'))
            
            # Build category key based on dataset type and properties
            if dataset_type == 'genotype':
                weight_status = "W" if row.get('weight_file_present', False) else 'UW'
                gwas_name = get_gwas_name(row.get('gwas_file'))
                category_key = f"Genotype_{weight_status}_{annotation_status}_{gwas_name}"
                
            elif dataset_type == 'prs':
                gwas_name = get_gwas_name(row.get('gwas_file'))
                model = row.get('model', 'NA')
                category_key = f"PRS_{gwas_name}"
                
            elif dataset_type == 'covariates':
                features = row.get('Features', 'NA')
                category_key = f"Covariates"
                
            elif dataset_type == 'pca':
                components = row.get('pca_components', 'NA')
                category_key = f"PCA_{components}"
                
            else:
                category_key = f"Other_{dataset_type}"
            
            categories[category_key].append(dataset_id)
        
        # Convert to regular dict and sort datasets within each category
        categories_dict = dict(categories)
        for key in categories_dict:
            categories_dict[key] = sorted(categories_dict[key])
        
        # Second pass: Select best performing dataset from each category
        selected_datasets = []
        for category, datasets in categories_dict.items():
            # Filter results for current category's datasets
            category_results = results_df[results_df['Dataset'].isin(datasets)]
            
            # Select top N performing datasets from this category
            top_n = category_results.nlargest(n_per_category, metric)
            selected_datasets.extend(top_n['Dataset'].tolist())
        
        # Create DataFrame with best performing datasets
        best_datasets = results_df[results_df['Dataset'].isin(selected_datasets)].copy()
        
        # Add category information to best datasets
        best_datasets['Category'] = best_datasets.apply(
            lambda row: next(
                (cat for cat, ds in categories_dict.items() 
                 if row['Dataset'] in ds),
                'Unknown'
            ),
            axis=1
        )
        
        # Sort by category and metric
        best_datasets = best_datasets.sort_values(
            ['Category', metric], 
            ascending=[True, False]
        )
        
        return categories_dict, best_datasets
        
    except FileNotFoundError:
        logging.error(f"Results file not found: {results_path}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"Results file is empty: {results_path}")
        raise
    except Exception as e:
        logging.error(f"Error processing datasets: {str(e)}")
        raise


# Example usage:
parser = argparse.ArgumentParser(description='Train combined model')
parser.add_argument('--phenotype', type=str, help='Phenotype directory')
parser.add_argument('--results_dir', type=str, help='Results directory (e.g., ResultsML)')
args = parser.parse_args()

# Setup paths
results_path = os.path.join(args.phenotype, "Results", args.results_dir, "ResultsFinal.csv")

 
categories, best_datasets = get_unique_category_datasets(results_path)


def select_best_models(results_df, min_auc=0.5):
    """Select best models using multi-stage filtering with stability metrics"""
    # Initial filtering
    valid_models = results_df[
        (results_df["Validation AUC"] >= min_auc) &
        (results_df["Train AUC"] >= min_auc) &
        (results_df["Train AUC"] >= results_df["Validation AUC"]) &
        (results_df["Train AUC"] <= 1.0) &
        (results_df["Validation AUC"] <= 1.0)
    ].copy()
    
    if valid_models.empty:
        print("No models met the initial filtering criteria")
        return None
    
    # Calculate evaluation metrics
    valid_models['Train_Val_Gap'] = abs(valid_models['Train AUC'] - valid_models['Validation AUC'])
    valid_models['Train_Stability'] = 1 / (1 + valid_models['Train AUC'].std())
    valid_models['Val_Stability'] = 1 / (1 + valid_models['Validation AUC'].std())
    
    # Normalize metrics to 0-1 scale
    metrics_to_normalize = ['Validation AUC', 'Train_Val_Gap', 'Train_Stability', 'Val_Stability']
    
    for metric in metrics_to_normalize:
        if metric == 'Validation AUC':
            valid_models[f'{metric}_Norm'] = (valid_models[metric] - min_auc) / (1 - min_auc)
        elif metric in ['Train_Stability', 'Val_Stability']:
            valid_models[f'{metric}_Norm'] = valid_models[metric]
        else:
            max_val = valid_models[metric].max()
            if max_val > 0:
                valid_models[f'{metric}_Norm'] = 1 - (valid_models[metric] / max_val)
            else:
                valid_models[f'{metric}_Norm'] = 0
    
    # Calculate composite score
    valid_models['composite_score'] = (
        0.25 * valid_models['val_auc_norm'] +
        0.25 * valid_models['train_val_gap_norm'] +
        0.25 * valid_models['train_stability_norm'] +
        0.25 * valid_models['val_stability_norm']
    )
    
    try:
        # Select best model per type based on composite score
        best_models = valid_models.loc[valid_models.groupby('Model')['Composite_Score'].idxmax()]
        best_models['Overall_Rank'] = best_models['Composite_Score'].rank(ascending=False)
        
        # Sort and print best models
        best_models_sorted = best_models.sort_values('Composite_Score', ascending=False)
        
        print("\nBest Models Selected:")
        for _, model in best_models_sorted.iterrows():
            print(f"\nModel: {model['Model']}")
            print(f"Composite Score: {model['Composite_Score']:.4f}")
            print(f"Overall Rank: {model['Overall_Rank']:.0f}")
            print(f"Train AUC: {model['Train AUC']:.4f}")
            print(f"Validation AUC: {model['Validation AUC']:.4f}")
            print(f"Test AUC: {model['Test AUC']:.4f}")
            print(f"Train-Val Gap: {model['Train_Val_Gap']:.4f}")
            print(f"Training Stability: {model['Train_Stability']:.4f}")
            print(f"Validation Stability: {model['Val_Stability']:.4f}")
        
        return best_models_sorted
    except Exception as e:
        print(f"Error in selecting best models: {str(e)}")
        return None

def evaluate_combination_with_save(phenotype_dir: str, combo: tuple, 
                                output_file: str, combination_size: int, 
                                n_folds: int = 5) -> None:
    """
    Evaluate a combination across all folds, average results, and save
    """
    fold_results = []
    
    # Process all folds for this combination
    for fold in range(n_folds):
        print(f"Processing combination {combo} - fold {fold}")
        
        # Train models and get predictions
        meta_features = train_individual_models(phenotype_dir, list(combo), fold)
        (X_meta_train, y_train, X_meta_test, y_test, X_meta_val, y_val, 
         individual_models) = meta_features
        
        # Train meta model
        meta_model = create_meta_model(X_meta_train.shape[1])
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        class_weight = {0: 1.0, 1: n_neg/n_pos}
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_auc',
                patience=10,
                mode='max',
                restore_best_weights=True
            )
        ]
        
        meta_model.fit(
            X_meta_train, y_train,
            validation_data=(X_meta_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks_list,
            class_weight=class_weight,
            verbose=0
        )
        
        # Get performance metrics
        train_auc = meta_model.evaluate(X_meta_train, y_train, verbose=0)[1]
        val_auc = meta_model.evaluate(X_meta_val, y_val, verbose=0)[1]
        test_auc = meta_model.evaluate(X_meta_test, y_test, verbose=0)[1]
        
        fold_results.append({
            'Train_AUC': train_auc,
            'Validation_AUC': val_auc,
            'Test_AUC': test_auc
        })
    
    # Calculate average and std across folds
    results_df = pd.DataFrame(fold_results)
    averaged_result = {
        'Combination_Size': combination_size,
        'Datasets': str(list(combo)),
        'Train_AUC_Mean': results_df['Train_AUC'].mean(),
        'Train_AUC_Std': results_df['Train_AUC'].std(),
        'Validation_AUC_Mean': results_df['Validation_AUC'].mean(),
        'Validation_AUC_Std': results_df['Validation_AUC'].std(),
        'Test_AUC_Mean': results_df['Test_AUC'].mean(),
        'Test_AUC_Std': results_df['Test_AUC'].std()
    }
    
    # Save averaged results to file
    result_df = pd.DataFrame([averaged_result])
    if os.path.exists(output_file):
        # If file exists, append without header
        result_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        # If file doesn't exist, create it with header
        result_df.to_csv(output_file, index=False)
    
    print(f"\nCompleted combination {combo}:")
    print(f"Mean Validation AUC: {averaged_result['Validation_AUC_Mean']:.4f} ± {averaged_result['Validation_AUC_Std']:.4f}")
    print(f"Mean Test AUC: {averaged_result['Test_AUC_Mean']:.4f} ± {averaged_result['Test_AUC_Std']:.4f}")

def evaluate_combinations(phenotype_dir: str, datasets: List[int], 
                        combination_size: int, output_file: str) -> None:
    """
    Evaluate all combinations of given size, saving averaged results
    """
    dataset_combinations = list(itertools.combinations(datasets, combination_size))
    print(f"\nStarting evaluation of {len(dataset_combinations)} combinations of size {combination_size}")
    
    for i, combo in enumerate(dataset_combinations, 1):
        print(f"\nProcessing combination {i}/{len(dataset_combinations)}: {combo}")
        evaluate_combination_with_save(phenotype_dir, combo, output_file, combination_size)

def iterative_combination_search(phenotype_dir: str, initial_datasets: List[int], 
                               output_dir: str, max_combo_size: int = 4, 
                               top_k: int = 10, min_auc: float = 0.5):
    """
    Run search saving averaged results for each combination
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'averaged_combinations_results.csv')
    best_combinations = set()
    
    current_size = 2
    current_datasets = initial_datasets
    
    while current_size <= max_combo_size:
        print(f"\nEvaluating {current_size}-way combinations")
        
        # Evaluate combinations
        evaluate_combinations(
            phenotype_dir, 
            current_datasets, 
            current_size,
            output_file
        )
        
        # Read current results to select best combinations
        all_results = pd.read_csv(output_file)
        current_size_results = all_results[all_results['Combination_Size'] == current_size]
        
        # Filter based on minimum AUC if specified
        if min_auc > 0:
            current_size_results = current_size_results[
                current_size_results['Validation_AUC_Mean'] >= min_auc
            ]
        
        # Select top combinations for next iteration
        top_combinations = current_size_results.nlargest(top_k, 'Validation_AUC_Mean')
        
        if top_combinations.empty:
            print(f"No valid {current_size}-way combinations found meeting criteria")
            break
        
        # Print summary of top combinations
        print(f"\nTop {current_size}-way combinations:")
        for _, row in top_combinations.iterrows():
            print(f"\nCombination: {row['Datasets']}")
            print(f"Mean Validation AUC: {row['Validation_AUC_Mean']:.4f} ± {row['Validation_AUC_Std']:.4f}")
            print(f"Mean Test AUC: {row['Test_AUC_Mean']:.4f} ± {row['Test_AUC_Std']:.4f}")
        
        # Get datasets for next iteration
        next_datasets = set()
        for _, row in top_combinations.iterrows():
            combination = eval(row['Datasets'])
            best_combinations.add(tuple(combination))
            next_datasets.update(combination)
        
        current_datasets = list(next_datasets)
        current_size += 1
    
    print("\nAll combinations evaluated. Results saved in:", output_file)
    return pd.read_csv(output_file), best_combinations

def main():
    parser = argparse.ArgumentParser(description='Iterative dataset combination search')
    parser.add_argument('--phenotype', type=str, help='Phenotype directory')
    parser.add_argument('--results_dir', type=str, help='Results directory')
    parser.add_argument('--max_combo_size', type=int, default=5, help='Maximum combination size')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top combinations to keep')
    parser.add_argument('--min_auc', type=float, default=0.5, help='Minimum AUC threshold')
    args = parser.parse_args()
    
    # Get initial datasets
    results_path = os.path.join(args.phenotype, "Results", args.results_dir, "ResultsFinal.csv")
    categories, best_datasets = get_unique_category_datasets(results_path)
    initial_datasets = best_datasets['Dataset'].tolist()
    print(f"Initial datasets: {initial_datasets}")
    print(f"Total initial datasets: {len(initial_datasets)}")
    print(f"Unique categories: {len(categories)}")
    print(f"Best datasets: {len(best_datasets)}")
    print(f"Best datasets by category:")
    for category, datasets in categories.items():
        print(f"{category}: {len(datasets)}")
    
     


    output_dir = os.path.join(args.phenotype, "Results", args.results_dir, "CombinationResults")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to: {output_dir}")



    # Run iterative combination search
    results, best_combinations = iterative_combination_search(
        args.phenotype,
        initial_datasets,
        output_dir,
        args.max_combo_size,
        args.top_k,
        args.min_auc
    )
    
    print("\nSearch complete. Results saved in:", output_dir)
    
    # Print final summary
    print("\nTop combinations by validation AUC:")
    summary_df = pd.read_csv(os.path.join(output_dir, 'best_combinations_summary.csv'))
    top_combos = summary_df.nlargest(5, 'Mean_Val_AUC')
    for _, row in top_combos.iterrows():
        print(f"\nCombination: {row['Combination']}")
        print(f"Size: {row['Size']}")
        print(f"Validation AUC: {row['Mean_Val_AUC']:.4f} ± {row['Std_Val_AUC']:.4f}")
        print(f"Test AUC: {row['Mean_Test_AUC']:.4f} ± {row['Std_Test_AUC']:.4f}")

if __name__ == "__main__":
    main()