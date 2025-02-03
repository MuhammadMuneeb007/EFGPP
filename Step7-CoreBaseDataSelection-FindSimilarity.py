import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_processing.log'),
        logging.StreamHandler()
    ]
)

class DatasetProcessor:
    def __init__(self, base_folder='migraine', fold_num=0, max_workers=4):
        self.base_folder = base_folder
        self.fold_num = fold_num
        self.fold_path = os.path.join(base_folder, f'Fold_{fold_num}')
        self.datasets_path = os.path.join(self.fold_path, 'Datasets')
        self.unique_datasets_path = os.path.join(self.fold_path, 'UniqueDatasets')
        self.max_workers = max_workers
        
        # Ensure required directory exists
        os.makedirs(self.unique_datasets_path, exist_ok=True)
    
    def load_dataset(self, dataset_num):
        """Load a single dataset"""
        try:
            dataset_path = os.path.join(self.datasets_path, f'dataset_{dataset_num}')
            
            # Load training set
            X_train = pd.read_csv(os.path.join(dataset_path, f'dataset_{dataset_num}_X_train.csv'))
            X_train = X_train.iloc[:, 2:]  # Remove first two columns
            
            # Scale the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            return X_train_scaled
            
        except Exception as e:
            logging.error(f"Error loading dataset {dataset_num}: {str(e)}")
            raise

    def calculate_similarity(self, pair):
        """Calculate similarity between two datasets by loading them on demand"""
        dataset1_num, dataset2_num = pair
        try:
            # Load datasets
            data1 = self.load_dataset(dataset1_num)
            data2 = self.load_dataset(dataset2_num)
            
            # Log if datasets have different dimensions
            if data1.shape[1] != data2.shape[1]:
                logging.info(f"Note: Datasets {dataset1_num} and {dataset2_num} have different number of features: {data1.shape[1]} vs {data2.shape[1]}")
                logging.info("Proceeding with similarity calculation anyway")
            
            # Calculate similarity
            ks_stat = stats.ks_2samp(data1.flatten(), data2.flatten()).statistic
            similarity = 1 - ks_stat
            
            logging.info(f"Similarity between dataset {dataset1_num} and {dataset2_num}: {similarity:.4f}")
            return dataset1_num, dataset2_num, similarity
            
        except Exception as e:
            logging.error(f"Error calculating similarity between datasets {dataset1_num} and {dataset2_num}: {str(e)}")
            return dataset1_num, dataset2_num, 0.0

    def create_similarity_matrix(self, datasets, similarities):
        """Create similarity matrix for given datasets"""
        n = len(datasets)
        matrix = pd.DataFrame(
            np.eye(n),  # Initialize with ones on diagonal (self-similarity = 1)
            columns=datasets,
            index=datasets
        )
        
        # Fill the matrix
        for i in datasets:
            for j in datasets:
                if i != j:
                    sim = similarities.get((i, j)) or similarities.get((j, i), 0.0)
                    matrix.loc[i, j] = sim
                    matrix.loc[j, i] = sim
        
        return matrix

    def find_unique_datasets(self, n_datasets=10, similarity_threshold=0.9999):
        """Identify unique datasets by removing those with high correlation using parallel processing"""
        logging.info("Starting unique dataset identification process")
        
        try:
            # Generate all pairs of datasets to compare
            dataset_pairs = []
            for i in range(2, n_datasets + 1):
                for j in range(1, i):
                    dataset_pairs.append((i, j))
            
            # Calculate similarities in parallel
            similarities = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_pair = {
                    executor.submit(self.calculate_similarity, pair): pair 
                    for pair in dataset_pairs
                }
                
                # Process completed tasks
                for future in as_completed(future_to_pair):
                    dataset1_num, dataset2_num, similarity = future.result()
                    similarities[(dataset1_num, dataset2_num)] = similarity
            
            # Process results to find unique datasets
            unique_datasets = [1]  # Always keep the first dataset
            
            for i in range(2, n_datasets + 1):
                is_unique = True
                for j in unique_datasets:
                    similarity = similarities.get((i, j)) or similarities.get((j, i))
                    if similarity >= similarity_threshold:
                        is_unique = False
                        logging.info(f"Dataset {i} is highly similar to dataset {j} (similarity: {similarity:.4f})")
                        break
                
                if is_unique:
                    unique_datasets.append(i)
                    logging.info(f"Dataset {i} identified as unique")
            
            # Create and save similarity matrix for unique datasets
            similarity_matrix = self.create_similarity_matrix(unique_datasets, similarities)
            
            return unique_datasets, similarity_matrix
            
        except Exception as e:
            logging.error(f"Error in find_unique_datasets: {str(e)}")
            raise
    
    def save_results(self, unique_datasets, similarity_matrix):
        """Save unique datasets list and similarity matrix"""
        try:
            # Save unique datasets list
            datasets_file = os.path.join(self.unique_datasets_path, 'unique_datasets.txt')
            with open(datasets_file, 'w') as f:
                for dataset_num in unique_datasets:
                    f.write(f"dataset_{dataset_num}\n")
            logging.info(f"Unique datasets list saved to {datasets_file}")
            
            # Save similarity matrix
            matrix_file = os.path.join(self.unique_datasets_path, 'similarity_matrix.csv')
            similarity_matrix.to_csv(matrix_file)
            logging.info(f"Similarity matrix saved to {matrix_file}")
            
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            raise

def main():
    try:
        import sys
        # Initialize processor with desired number of worker threads
        processor = DatasetProcessor(base_folder=sys.argv[1], fold_num=int(sys.argv[2]), max_workers=20)
        
        # Find unique datasets and get similarity matrix
        logging.info("Starting dataset analysis")
        
        numberofdatasets = len([d for d in os.listdir(processor.datasets_path) if os.path.isdir(os.path.join(processor.datasets_path, d)) and d.startswith('dataset_')])
        print(processor.datasets_path)
        print(f"Number of datasets: {numberofdatasets}")
    
        unique_datasets, similarity_matrix = processor.find_unique_datasets(n_datasets=int(numberofdatasets))
        
        # Save results
        processor.save_results(unique_datasets, similarity_matrix)
        
        logging.info(f"Process completed successfully. Found {len(unique_datasets)} unique datasets: {unique_datasets}")
        logging.info("\nSimilarity Matrix for unique datasets:")
        logging.info("\n" + str(similarity_matrix))
        
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()