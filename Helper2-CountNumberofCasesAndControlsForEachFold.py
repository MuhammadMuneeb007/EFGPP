import pandas as pd
import sys

# Load the phenotype file path from command-line arguments
phenotype_file = sys.argv[1]

# Initialize an empty list to store the fold data
fold_data = []

# Loop through each fold (0 to 4)
for fold in range(0,5):
    # Construct the file path for train, test, and validation for each fold
    train_file = f"{phenotype_file}/Fold_{fold}/train_data.fam"
    test_file = f"{phenotype_file}/Fold_{fold}/test_data.fam"
    validation_file = f"{phenotype_file}/Fold_{fold}/validation_data.fam"
    
    # Construct the file path for covariates
    train_cov_file = f"{phenotype_file}/Fold_{fold}/train_data.cov"
    test_cov_file = f"{phenotype_file}/Fold_{fold}/test_data.cov"
    validation_cov_file = f"{phenotype_file}/Fold_{fold}/validation_data.cov"
    
    # Read the covariate files
    train_cov_data = pd.read_csv(train_cov_file)
    test_cov_data = pd.read_csv(test_cov_file)
    validation_cov_data = pd.read_csv(validation_cov_file)
    
    # Print the number of rows in the covariate files
    # print(f"Fold {fold}:")
    # print(f"  Train cov file rows: {train_cov_data.shape[0]}")
    # print(f"  Test cov file rows: {test_cov_data.shape[0]}")
    # print(f"  Validation cov file rows: {validation_cov_data.shape[0]}")
    
    # Read the .fam files with space separation and no header
    train_data = pd.read_csv(train_file, sep="\s+", header=None)
    test_data = pd.read_csv(test_file, sep="\s+", header=None)
    validation_data = pd.read_csv(validation_file, sep="\s+", header=None)

    # Assuming phenotype is in the 6th column (index 5), count the number of 1s and 2s
    train_phenotype = train_data[5]
    test_phenotype = test_data[5]
    validation_phenotype = validation_data[5]

    # Count the occurrences of 1s and 2s
    train_count = train_phenotype.value_counts().to_dict()
    test_count = test_phenotype.value_counts().to_dict()
    validation_count = validation_phenotype.value_counts().to_dict()
    
    # Store the fold information in the list as a dictionary
    fold_data.append({
        "Fold": fold,
        "Train 1's": train_count.get(1, 0),
        "Train 2's": train_count.get(2, 0),
        "Test 1's": test_count.get(1, 0),
        "Test 2's": test_count.get(2, 0),
        "Val 1's": validation_count.get(1, 0),
        "Val 2's": validation_count.get(2, 0),
        "Train Cov": train_cov_data.shape[0],
        "Test Cov": test_cov_data.shape[0],
        "Val Cov": validation_cov_data.shape[0]
    })

# Convert the fold data into a DataFrame
df_fold_data = pd.DataFrame(fold_data)

# Print the DataFrame as markdown
markdown = df_fold_data.to_markdown(index=False)

# Output the markdown
print(markdown)
