import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Add argument parser
parser = argparse.ArgumentParser(description='Plot combination results for a specific phenotype')
parser.add_argument('--phenotype', type=str, required=True, help='Phenotype name (e.g., migraine)')
parser.add_argument('--results_dir', type=str, required=True, help='Base results directory')
args = parser.parse_args()

# Construct file paths
results_path = f'{args.phenotype}/Results/{args.results_dir}/CombinationResults/averaged_combinations_results.csv'
output_plot =  f'{args.phenotype}/Results/{args.results_dir}/CombinationResults/best_models_performance.png'

def select_best_models(df, min_auc=0.5):
    # Initial filtering
    valid_models = df[
        (df["Validation_AUC_Mean"] >= min_auc) &
        (df["Train_AUC_Mean"] >= min_auc) &
        (df["Test_AUC_Mean"] >= min_auc) &
        (df["Train_AUC_Mean"] >= df["Validation_AUC_Mean"]) &
        (df["Train_AUC_Mean"] <= 1.0) &
        (df["Validation_AUC_Mean"] <= 1.0)
    ].copy()
    
    if valid_models.empty:
        print("No models met the initial filtering criteria")
        return None
    
    # Calculate stability metrics
    valid_models['Train_Val_Gap'] = abs(valid_models['Train_AUC_Mean'] - valid_models['Validation_AUC_Mean'])
    valid_models['Stability_Score'] = 1 / (1 + valid_models['Train_AUC_Std'] + valid_models['Validation_AUC_Std'])
    
    # Normalize metrics
    valid_models['Val_AUC_Norm'] = (valid_models['Validation_AUC_Mean'] - min_auc) / (1 - min_auc)
    max_gap = valid_models['Train_Val_Gap'].max()
    valid_models['Gap_Norm'] = 1 - (valid_models['Train_Val_Gap'] / max_gap if max_gap > 0 else 0)
    valid_models['Stability_Norm'] = (valid_models['Stability_Score'] - valid_models['Stability_Score'].min()) / \
                                   (valid_models['Stability_Score'].max() - valid_models['Stability_Score'].min())
    
    # Calculate composite score
    valid_models['Composite_Score'] = (
        0.4 * valid_models['Val_AUC_Norm'] +
        0.3 * valid_models['Gap_Norm'] +
        0.3 * valid_models['Stability_Norm']
    )
    
    # Select top 10 models
    best_models = valid_models.nlargest(25, 'Composite_Score')
    
    return best_models



df = pd.read_csv(results_path)
best_models = select_best_models(df,min_auc=0.6)

# After reading the results file, also read the best_datasets file
best_datasets = pd.read_csv(os.path.join(os.path.dirname(results_path), 'best_datasets.csv'))

# Create a mapping of dataset numbers to categories
dataset_category_map = {str(row['Dataset']): row['Category'] for _, row in best_datasets.iterrows()}

# In the plotting section, modify the xticks label creation
def format_dataset_label(dataset_str):
    # Convert "[1,2]" style string to list of strings
    datasets = dataset_str.strip('[]').split(',')
    # Get categories for each dataset number
    categories = [dataset_category_map.get(d.strip(), 'Unknown') for d in datasets]
    # Join categories with ' | '
    return ' | '.join(categories)

if best_models is None or len(best_models) == 0:
    print("No valid models to plot")
    exit()
 
import matplotlib as mpl
mpl.rcParams.update({
     
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# Modify figure creation for improved clarity
fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')

# Define professional color palette and plot parameters
from matplotlib.colors import to_rgba

# Define colors with transparency
colors = [(0.282, 0.470, 0.816, 0.8),    # blue
          (0.933, 0.525, 0.290, 0.8),    # orange
          (0.416, 0.800, 0.392, 0.8)]    # green
bar_width = 0.25
n_groups = len(best_models)
metrics = ['Train_AUC_Mean', 'Validation_AUC_Mean', 'Test_AUC_Mean']
indices = np.arange(n_groups)

# Create bars for each metric
for i, (metric, color) in enumerate(zip(metrics, colors)):
    position = indices + (i * bar_width)
    bars = plt.bar(position, best_models[metric], bar_width,
                  color=color, label=metric.split('_')[0],
                  edgecolor='none')
    
    # Add value labels on top of each bar
    for j, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}',
                ha='center', va='bottom',
                rotation=0, fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', 
                         alpha=0.7, pad=1))

# Enhance plot styling
plt.xlabel('Dataset Combinations', fontsize=12, fontweight='bold')
plt.ylabel('AUC Score', fontsize=12, fontweight='bold')
title = plt.title(f'Performance of Top Dataset Combinations\nfor {args.phenotype.capitalize()}',
                 fontsize=14, fontweight='bold', pad=20)

# Customize ticks with better readability
plt.xticks(indices + bar_width, 
           [format_dataset_label(d) for d in best_models['Datasets']], 
           rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=10)
ax.set_ylim(0.4, 1.0)

# Enhance grid - horizontal only
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, axis='y', zorder=0)

# Update legend styling to include a shadow and nicer padding
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True, borderpad=1)

# Refine spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Adjust layout to accommodate legend
plt.tight_layout()
plt.subplots_adjust(right=0.85)  # Make room for legend

# Save plot with high resolution
plt.savefig(output_plot, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

# Update print header
print(f"\nTop Dataset Combinations for {args.phenotype}:")
for idx, model in best_models.iterrows():
    print(f"\nDataset: {model['Datasets']}")
    print(f"Composite Score: {model['Composite_Score']:.4f}")
    print(f"Train AUC: {model['Train_AUC_Mean']:.4f} ± {model['Train_AUC_Std']:.4f}")
    print(f"Validation AUC: {model['Validation_AUC_Mean']:.4f} ± {model['Validation_AUC_Std']:.4f}")
    print(f"Test AUC: {model['Test_AUC_Mean']:.4f} ± {model['Test_AUC_Std']:.4f}")