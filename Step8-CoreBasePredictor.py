import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from CoreML import train_and_evaluate_machine_learning
from CoreDL import train_and_evaluate_deep_learning

def load_and_preprocess_data(base_path, dataset_num):
    """
    Load and preprocess train, test, and validation data
    """
    # Construct file paths
    train_x_path = f"{base_path}/dataset_{dataset_num}/dataset_{dataset_num}_X_train.csv"
    train_y_path = f"{base_path}/dataset_{dataset_num}/dataset_{dataset_num}_y_train.csv"
    test_x_path = f"{base_path}/dataset_{dataset_num}/dataset_{dataset_num}_X_test.csv"
    test_y_path = f"{base_path}/dataset_{dataset_num}/dataset_{dataset_num}_y_test.csv"
    val_x_path = f"{base_path}/dataset_{dataset_num}/dataset_{dataset_num}_X_val.csv"
    val_y_path = f"{base_path}/dataset_{dataset_num}/dataset_{dataset_num}_y_val.csv"
    
    # Load data
    X_train = pd.read_csv(train_x_path).iloc[:, 2:].values
    y_train = pd.read_csv(train_y_path).values.ravel()
    X_test = pd.read_csv(test_x_path).iloc[:, 2:].values
    y_test = pd.read_csv(test_y_path).values.ravel()
    X_val = pd.read_csv(val_x_path).iloc[:, 2:].values
    y_val = pd.read_csv(val_y_path).values.ravel()
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    
    return X_train, y_train, X_test, y_test, X_val, y_val

def process_dataset(phenotype, fold, dataset_num, base_path, results_path,mlordl):
    """
    Process a single dataset and return results
    """
    try:
        # Load and preprocess data
        X_train, y_train, X_test, y_test, X_val, y_val = load_and_preprocess_data(base_path, dataset_num)
        
        print(f"\nProcessing Dataset {dataset_num}")
        print(f"Input shape: {X_train.shape}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        if mlordl == "ML":
            # Train and evaluate using the imported function
            best_result = train_and_evaluate_machine_learning(
                X_train,X_test,X_val, y_train,y_test,y_val
              
            )
        else:    
            # Train and evaluate using the imported function
            best_result = train_and_evaluate_deep_learning(
                X_train,X_test,X_val, y_train,y_test,y_val
            
            )
        
        # Add additional information to the results
        best_result['Phenotype'] = phenotype
        best_result['Fold'] = fold
        best_result['Dataset'] = dataset_num
        
        # Save results
        output_file = f"{results_path}/dataset_{dataset_num}_results.csv"
        best_result.to_csv(output_file, index=False)
        
        print(f"\nResults saved to: {output_file}")
        print(best_result)
        
        return best_result
        
    except Exception as e:
        print(f"Error processing dataset {dataset_num}: {str(e)}")
        return None

def main():
    # Set paths from command line arguments
    import sys
    if len(sys.argv) != 5:
        print("Usage: python script.py <phenotype> <fold> <dataset_num> <ML or DL>")
        sys.exit(1)
        
    phenotype = sys.argv[1]
    fold = sys.argv[2]
    dataset_num = int(sys.argv[3])
    mlordl = sys.argv[4]

    
    base_path = f"{phenotype}/Fold_{fold}/Datasets"
    results_path = f"{phenotype}/Fold_{fold}/Results" + mlordl
    
    # Create results directory if it doesn't exist
    os.makedirs(results_path, exist_ok=True)
    
    # Process specific dataset
    results = process_dataset(phenotype, fold, dataset_num, base_path, results_path,mlordl)
    
    if results is None:
        print("Failed to process dataset")
        sys.exit(1)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()