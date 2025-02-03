import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
from CoreBaseReadDatasets import *
import json
from datetime import datetime

warnings.filterwarnings('ignore')

class DatasetParameters:
    """Central class for managing dataset parameters"""
    def __init__(self):
        # Define all parameters here
        self.phenotype_gwas_pairs = [
            ("migraine", "migraine.gz"),
            ("depression", "depression_11.gz"),
            ("migraine", "migraine_5.gz"),
            ("depression", "depression_4.gz"),
            ("depression", "depression_17.gz"),
            #("hypertension", "hypertension_0.gz"),
            #("hypertension", "hypertension_20.gz")
        ]
        
        self.models = ["Plink", "PRSice-2", "AnnoPred", "LDAK-GWAS"]
        self.pca_components = 10
        self.scaling_options = [False]  # Can be expanded to [True, False] if needed
        self.snp_options = ["snps_50","snps_200","snps_500","snps_1000","snps_5000","snps_10000",  "snps_annotated_50", "snps_annotated_200", "snps_annotated_500"]
        
    def get_phenotypes(self) -> List[str]:
        """Get unique phenotypes"""
        return list(set(pair[0] for pair in self.phenotype_gwas_pairs))
        
    def get_gwas_files(self) -> List[str]:
        """Get all GWAS files"""
        return list(pair[1] for pair in self.phenotype_gwas_pairs)
    
    def get_parameter_summary(self) -> Dict:
        """Get a summary of all parameters"""
        return {
            "phenotype_gwas_pairs": len(self.phenotype_gwas_pairs),
            "models": len(self.models),
            "pca_components": self.pca_components,
            "scaling_options": len(self.scaling_options),
            "snp_options": len(self.snp_options)
        }

def get_weight_files(fold: int, params: DatasetParameters) -> Dict[str, pd.DataFrame]:
    """Generate weight files for valid phenotype-GWAS pairs"""
    all_weight_files = {}
    for pheno, gwas in params.phenotype_gwas_pairs:
        for model in params.models:
            try:
                key = f"{pheno}_{gwas}_{model}_fold{fold}"
                weight_df = get_me_prs_based_gwas_file(pheno, gwas, fold, model)
                if isinstance(weight_df, pd.DataFrame):
                    all_weight_files[key] = weight_df
                else:
                    print(f"Warning: Weight file for {key} is not a DataFrame")
            except Exception as e:
                print(f"Error getting weight file for {key}: {str(e)}")
    
    return all_weight_files

def calculate_dataset_counts(params: DatasetParameters) -> Dict:
    """Calculate the number of datasets that will be generated based on parameters"""
    num_pairs = len(params.phenotype_gwas_pairs)
    num_models = len(params.models)
    num_snps = len(params.snp_options)
    num_weight_files = num_models * num_pairs + 1  # +1 for None option
    
    counts = {
        'covariate': num_pairs,  # 1 per pair
        'pca': num_pairs,        # 1 per pair
        'genotype': num_pairs * num_snps * num_weight_files,  # SNPs × weight files per pair
        'prs': num_pairs * num_models  # models per pair
    }
    
    return {
        'counts': counts,
        'total': sum(counts.values()),
        'details': {
            'num_pairs': num_pairs,
            'num_models': num_models,
            'num_snps': num_snps,
            'num_weight_files': num_weight_files,
            'per_pair': {
                'covariate': 1,
                'pca': 1,
                'genotype': num_snps * num_weight_files,
                'prs': num_models
            }
        }
    }

def format_parameters(params: Dict) -> str:
    """Format parameters for display"""
    output = "\nDataset Parameters:\n" + "="*50 + "\n"
    
    for key, value in params.items():
        if key == 'weightFile' and isinstance(value, pd.DataFrame):
            output += f"{key}: DataFrame present\n"
        elif isinstance(value, (list, np.ndarray)):
            output += f"{key}: {value if len(value) < 10 else str(value[:5]) + '...'}\n"
        else:
            output += f"{key}: {value}\n"
    
    output += "="*50
    return output

def save_datasets(X_train, X_test, X_val, y_train, y_test, y_val, save_dir: str, 
                 dataset_number: int, dataset_type: str, parameters: Dict):
    """Save datasets with detailed parameter logging"""
    os.makedirs(save_dir, exist_ok=True)
    
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        X_val = pd.DataFrame(X_val)
    
    dataset_prefix = f"dataset_{dataset_number}"
    
    # Save datasets
    X_train.to_csv(os.path.join(save_dir, f'{dataset_prefix}_X_train.csv'), index=False)
    X_test.to_csv(os.path.join(save_dir, f'{dataset_prefix}_X_test.csv'), index=False)
    X_val.to_csv(os.path.join(save_dir, f'{dataset_prefix}_X_val.csv'), index=False)
    pd.Series(y_train).to_csv(os.path.join(save_dir, f'{dataset_prefix}_y_train.csv'), index=False)
    pd.Series(y_test).to_csv(os.path.join(save_dir, f'{dataset_prefix}_y_test.csv'), index=False)
    pd.Series(y_val).to_csv(os.path.join(save_dir, f'{dataset_prefix}_y_val.csv'), index=False)
    
    # Save parameters
    params_for_save = {k: str(v) if isinstance(v, pd.DataFrame) else v for k, v in parameters.items()}
    with open(os.path.join(save_dir, f'{dataset_prefix}_parameters.json'), 'w') as f:
        json.dump(params_for_save, f, indent=4, default=str)
    
    # Display detailed information
    print(f"\n✓ Generated Dataset {dataset_prefix}")
    print(f"Type: {dataset_type}")
    print(f"Location: {save_dir}")
    print(f"Shape: {X_train.shape[1]} features, {X_train.shape[0]} training samples")
    print(format_parameters(parameters))
    
    # Return metadata for tracking
    return {
        'Dataset_Name': dataset_prefix,
        'Dataset_Type': dataset_type,
        'Creation_Time': datetime.now(),
        'Parameters': parameters,
        'Features': X_train.shape[1],
        'Training_Samples': X_train.shape[0]
    }

def track_datasets(datasets_dir: str, dataset_metadata: Dict):
    """Track dataset creation with metadata"""
    tracking_file = os.path.join(datasets_dir, "dataset_tracking.csv")
    
    tracking_data = {
        'Dataset_Name': dataset_metadata['Dataset_Name'],
        'Dataset_Type': dataset_metadata['Dataset_Type'],
        'Creation_Time': dataset_metadata['Creation_Time'].strftime("%Y-%m-%d %H:%M:%S"),  # Convert to string
        'Features': dataset_metadata['Features'],
        'Training_Samples': dataset_metadata['Training_Samples']
    }
    
    # Add all parameters to tracking data
    for key, value in dataset_metadata['Parameters'].items():
        if isinstance(value, (list, pd.DataFrame)):
            tracking_data[key] = str(value)
        else:
            tracking_data[key] = value
    
    tracking_df = pd.DataFrame([tracking_data])
    if os.path.exists(tracking_file):
        existing_df = pd.read_csv(tracking_file)
        # Convert Creation_Time back to datetime when reading
        existing_df['Creation_Time'] = pd.to_datetime(existing_df['Creation_Time'])
        tracking_df['Creation_Time'] = pd.to_datetime(tracking_df['Creation_Time'])
        tracking_df = pd.concat([existing_df, tracking_df], ignore_index=True)
    
    tracking_df.to_csv(tracking_file, index=False)
    return tracking_df


def generate_dataset_summary(tracking_df: pd.DataFrame, save_dir: str):
    """Generate and save a summary of the datasets"""
    # Convert Creation_Time to datetime if it's not already
    tracking_df['Creation_Time'] = pd.to_datetime(tracking_df['Creation_Time'])
    
    summary = {
        'Total_Datasets': len(tracking_df),
        'Dataset_Types': tracking_df['Dataset_Type'].value_counts().to_dict(),
        'Phenotypes': tracking_df['phenotype'].value_counts().to_dict() if 'phenotype' in tracking_df else {},
        'Features_Stats': {
            'Min': tracking_df['Features'].min(),
            'Max': tracking_df['Features'].max(),
            'Mean': tracking_df['Features'].mean(),
            'Median': tracking_df['Features'].median()
        },
        'Creation_Time_Range': {
            'Start': tracking_df['Creation_Time'].min().strftime("%Y-%m-%d %H:%M:%S"),
            'End': tracking_df['Creation_Time'].max().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    summary_text = "\n=== Dataset Generation Summary ===\n\n"
    summary_text += f"Total Datasets Generated: {summary['Total_Datasets']}\n\n"
    
    summary_text += "Datasets by Type:\n"
    for dtype, count in summary['Dataset_Types'].items():
        summary_text += f"  - {dtype}: {count}\n"
    
    summary_text += "\nPhenotype Distribution:\n"
    for pheno, count in summary['Phenotypes'].items():
        summary_text += f"  - {pheno}: {count}\n"
    
    summary_text += "\nFeature Statistics:\n"
    for stat, value in summary['Features_Stats'].items():
        summary_text += f"  - {stat}: {value}\n"
    
    summary_text += "\nGeneration Time Range:\n"
    summary_text += f"  Start: {summary['Creation_Time_Range']['Start']}\n"
    summary_text += f"  End: {summary['Creation_Time_Range']['End']}\n"
    
    with open(os.path.join(save_dir, "dataset_summary.txt"), "w") as f:
        f.write(summary_text)
    
    print("\nDataset Generation Summary:")
    print(summary_text)
    
    return summary
def process_fold_with_parameters(main_phenotype: str, fold: int, base_dir: str, params: DatasetParameters):
    """Process a single fold with detailed parameter tracking"""
    print(f"\nProcessing fold {fold} for {main_phenotype}")
    
    # Setup directories
    fold_dir = os.path.join(base_dir, main_phenotype, f"Fold_{fold}")
    datasets_dir = os.path.join(fold_dir, "Datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Get weight files
    all_weight_files = get_weight_files(fold, params)
    
    #all_weight_files = {}
    

    # Get base y values for main phenotype
    main_gwas = next(gwas for pheno, gwas in params.phenotype_gwas_pairs if pheno == main_phenotype)
    _, _, _, y_train, y_test, y_val = get_me_just_covariates_data(main_phenotype, main_gwas, fold)
    
    dataset_counter = 1
    all_metadata = []
    
    # Process each valid phenotype-GWAS pair
    for pheno, gwas in params.phenotype_gwas_pairs:
        print(f"\nProcessing {pheno} with GWAS {gwas}")
        
       
        X_train, X_test, X_val, _, _, _ = get_me_just_covariates_data(
            pheno, gwas, fold, scaling=params.scaling_options[0]
        )
        
        covariate_params = {
            "dataset_type": "covariates",
            "phenotype": pheno,
            "gwas_file": gwas,
            "fold": fold,
            "scaling": params.scaling_options[0],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata = save_datasets(X_train, X_test, X_val, y_train, y_test, y_val,
                                os.path.join(datasets_dir, f"dataset_{dataset_counter}"),
                                dataset_counter, "covariates", covariate_params)
        all_metadata.append(metadata)
        dataset_counter += 1
      
    
        X_train, X_test, X_val, _, _, _ = get_me_just_pca_data(
            pheno, gwas, fold, scaling=params.scaling_options[0]
        )
        
        pca_params = {
            "dataset_type": "pca",
            "phenotype": pheno,
            "gwas_file": gwas,
            "fold": fold,
            "pca_components": params.pca_components,
            "scaling": params.scaling_options[0],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata = save_datasets(X_train, X_test, X_val, y_train, y_test, y_val,
                                os.path.join(datasets_dir, f"dataset_{dataset_counter}"),
                                dataset_counter, "pca", pca_params)
        all_metadata.append(metadata)
        dataset_counter += 1
   
        
        # 3. Process genotype
        for snps in params.snp_options:
            for weight_file in [None] + list(all_weight_files.values()):
        
                X_train, X_test, X_val, _, _, _ = get_me_just_genotype_data(
                    pheno, gwas, fold, snps, scaling=params.scaling_options[0],
                    weightFile=weight_file
                )
                
                genotype_params = {
                    "dataset_type": "genotype",
                    "phenotype": pheno,
                    "gwas_file": gwas,
                    "fold": fold,
                    "snps": snps,
                    "scaling": params.scaling_options[0],
                    "weight_file_present": weight_file is not None,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                metadata = save_datasets(X_train, X_test, X_val, y_train, y_test, y_val,
                                    os.path.join(datasets_dir, f"dataset_{dataset_counter}"),
                                    dataset_counter, "genotype", genotype_params)
                all_metadata.append(metadata)
                dataset_counter += 1
              
        # 4. Process PRS
        for model in params.models:
        
            X_train, X_test, X_val, _, _, _ = get_me_just_prs_data(
                pheno, gwas, fold, model
            )
            
            prs_params = {
                "dataset_type": "prs",
                "phenotype": pheno,
                "gwas_file": gwas,
                "fold": fold,
                "model": model,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            metadata = save_datasets(X_train, X_test, X_val, y_train, y_test, y_val,
                                    os.path.join(datasets_dir, f"dataset_{dataset_counter}"),
                                    dataset_counter, "prs", prs_params)
            all_metadata.append(metadata)
            dataset_counter += 1
            
    # After all processing is complete, generate tracking and summary
    all_tracking_dfs = []
    for metadata in all_metadata:
        tracking_df = track_datasets(datasets_dir, metadata)
        all_tracking_dfs.append(tracking_df)
    
    if all_tracking_dfs:
        final_tracking_df = pd.concat(all_tracking_dfs, ignore_index=True)
        generate_dataset_summary(final_tracking_df, datasets_dir)
    
    return datasets_dir

def main():
    import sys
    """Main execution function"""
    main_phenotype = sys.argv[1]
    base_dir = "."
    
    # Initialize parameters
    params = DatasetParameters()
    
    # Calculate expected dataset counts using parameters
    counts = calculate_dataset_counts(params)
    count_details = counts['counts']
    details = counts['details']
    per_pair = details['per_pair']
    
    print("\nStarting dataset generation...")
    print("\nParameter Configuration:")
    print("========================")
    param_summary = params.get_parameter_summary()
    for key, value in param_summary.items():
        print(f"- {key}: {value}")
    print("========================\n")
    
    print("This will generate:")
    print(f"- {count_details['covariate']} covariate datasets ({per_pair['covariate']} per pair)")
    print(f"- {count_details['pca']} PCA datasets ({per_pair['pca']} per pair)")
    print(f"- {count_details['genotype']} genotype datasets "
          f"({per_pair['genotype']} per pair = {details['num_snps']} SNPs × "
          f"{details['num_weight_files']} weight files)")
    print(f"- {count_details['prs']} PRS datasets ({per_pair['prs']} per pair = "
          f"{details['num_models']} models)")
    print(f"\nTotal: {counts['total']} datasets across {details['num_pairs']} phenotype-GWAS pairs")
    
    # Process single fold
    for fold in range(int(sys.argv[2]),int(sys.argv[2])+1):  # Process fold 0 only
        print(f"\nProcessing fold {fold}")
        datasets_dir = process_fold_with_parameters(main_phenotype, fold, base_dir, params)
        print(f"\nCompleted processing fold {fold}")
        print(f"Datasets stored in: {datasets_dir}")

if __name__ == "__main__":
    main()