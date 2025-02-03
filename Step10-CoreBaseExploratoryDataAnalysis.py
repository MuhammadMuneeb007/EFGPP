import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
import warnings
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def setup_style():
    """Set up plotting style for publication quality figures."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        # Use system default font instead of Arial
        'font.family': 'sans-serif',
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'figure.titlesize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'figure.figsize': [12, 8],
        'figure.autolayout': True,
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    # Publication-quality color palette
    colors = ['#2c3e50', '#e74c3c', '#2ecc71', '#3498db', '#f1c40f', '#9b59b6']
    sns.set_palette(colors)
    return colors

def format_plot(ax, title, xlabel, ylabel, legend=True, rotate_xticks=False):
    """Apply consistent formatting to plots."""
    ax.set_title(title, pad=20, fontweight='bold')
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if rotate_xticks:
        plt.xticks(rotation=45, ha='right')
        
    if legend:
        ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()

def save_plot(output_dir, filename, fig=None):
    """Save plot with consistent settings."""
    if fig is None:
        fig = plt.gcf()
    fig.savefig(
        os.path.join(output_dir, filename),
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.1
    )
    plt.close(fig)

def plot_model_frequency(df, output_dir):
    """1. Plot model frequency distribution."""
    fig, ax = plt.subplots()
    model_counts = df['Model'].value_counts()
    
    sns.barplot(x=model_counts.index, y=model_counts.values, ax=ax)
    format_plot(ax, 'Model Frequency Distribution', 
               'Model Type', 'Count', rotate_xticks=True)
    
    # Add value labels
    for i, v in enumerate(model_counts.values):
        ax.text(i, v, str(v), ha='center', va='bottom')
        
    save_plot(output_dir, '1_model_frequency.png', fig)

def plot_auc_by_phenotype_gwas(df, output_dir):
    """2. Plot AUC scores by phenotype and GWAS file."""
    pivot_data = df.pivot_table(
        values=['Train AUC', 'Validation AUC', 'Test AUC'],
        index=['phenotype', 'gwas_file'],  # Use both as index
        aggfunc='max'
    ).reset_index()
    
    # Create combined labels
    pivot_data['combined_label'] = pivot_data['phenotype'] + '\n' + pivot_data['gwas_file']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(pivot_data))
    width = 0.25
    
    # Plot bars for each metric
    ax.bar(x - width, pivot_data['Train AUC'], width, label='Train AUC')
    ax.bar(x, pivot_data['Validation AUC'], width, label='Validation AUC')
    ax.bar(x + width, pivot_data['Test AUC'], width, label='Test AUC')
    
    # Customize x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_data['combined_label'])
    
    format_plot(ax, 'AUC Scores by Phenotype and GWAS File', 
               'Phenotype - GWAS File', 'AUC Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot(output_dir, '2_phenotype_gwas_barplot.png', fig)

def plot_dataset_type_performance(df, output_dir):
    """3. Plot performance by dataset type."""
    metrics = ['Train AUC', 'Validation AUC', 'Test AUC']
    melted_df = pd.melt(df, 
                        value_vars=metrics,
                        id_vars=['Dataset_Type'])
    
    # Convert labels to sentence case
    melted_df['variable'] = melted_df['variable'].apply(lambda x: x.replace('_', ' ').title())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(x='Dataset_Type', y='value', hue='variable', data=melted_df, ax=ax)
    format_plot(ax, 'Performance by Dataset Type', 
               'Dataset Type', 'AUC Score', rotate_xticks=True)
    save_plot(output_dir, '3_dataset_type_performance.png', fig)

def plot_weight_file_impact(df, output_dir):
    """4. Plot impact of weight file presence."""
    metrics = ['Train AUC', 'Validation AUC', 'Test AUC']
    melted_df = pd.melt(df, 
                        value_vars=metrics,
                        id_vars=['weight_file_present'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='weight_file_present', y='value', hue='variable', data=melted_df, ax=ax)
    format_plot(ax, 'Impact of Weight File on Performance',
               'Weight File Present', 'AUC Score')
    save_plot(output_dir, '4_weight_file_impact.png', fig)

def plot_snp_analysis(df, output_dir):
    """5. Plot performance by SNP type."""
    if 'snps' in df.columns:
        metrics = ['Train AUC', 'Validation AUC', 'Test AUC']
        melted_df = pd.melt(df, 
                           value_vars=metrics,
                           id_vars=['snps'])
        
        # Convert metric labels to sentence case
        melted_df['variable'] = melted_df['variable'].apply(lambda x: x.replace('_', ' ').title())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='snps', y='value', hue='variable', data=melted_df, ax=ax)
        format_plot(ax, 'Performance by SNP type',
                   'SNP type', 'AUC Score', rotate_xticks=True)
        save_plot(output_dir, '5_snp_analysis.png', fig)

def plot_model_performance_comparison(df, output_dir):
    """6. Plot model performance comparison."""
    metrics = ['Train AUC', 'Validation AUC', 'Test AUC']
    melted_df = pd.melt(df, 
                        value_vars=metrics,
                        id_vars=['model'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='model', y='value', hue='variable', data=melted_df, ax=ax)
    format_plot(ax, 'PRS Model Performance Comparison',
               'PRS models', 'AUC Score', rotate_xticks=True)
    save_plot(output_dir, '6_model_performance.png', fig)

def plot_correlation_matrix(df, output_dir):
    """7. Plot correlation matrix of metrics."""
    metrics = ['Train AUC', 'Validation AUC', 'Test AUC', 
              'Train_Val_Gap', 'Composite_Score']
    correlation_matrix = df[metrics].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, 
               fmt='.3f', cmap='coolwarm', center=0, ax=ax)
    format_plot(ax, 'Correlation Between Metrics', '', '', legend=False)
    save_plot(output_dir, '7_correlation_matrix.png', fig)

def plot_stability_analysis(df, output_dir):
    """8. Plot model stability analysis."""
    stability_metrics = ['Train_Stability', 'Val_Stability']
    melted_df = pd.melt(df, 
                        value_vars=stability_metrics,
                        id_vars=['Model'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Model', y='value', hue='variable', data=melted_df, ax=ax)
    format_plot(ax, 'Model Stability Analysis',
               'Model', 'Stability Score', rotate_xticks=True)
    save_plot(output_dir, '8_stability_analysis.png', fig)

def plot_feature_impact(df, output_dir):
    """9. Plot feature impact on performance."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['Train AUC', 'Validation AUC', 'Test AUC']
    
    for i, metric in enumerate(metrics):
        sns.scatterplot(data=df, x='Features', y=metric, 
                       hue='Model', ax=axes[i])
        axes[i].set_xscale('log')
        format_plot(axes[i], f'{metric} vs Features',
                   'Number of Features', metric)
    
    plt.tight_layout()
    save_plot(output_dir, '9_feature_impact.png', fig)

def plot_gap_analysis(df, output_dir):
    """10. Plot gap analysis between train and validation."""
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Train AUC', y='Validation AUC',
                   hue='Model', style='Model', ax=ax)
    
    # Add diagonal line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
    
    format_plot(ax, 'Train vs Validation AUC Gap Analysis',
               'Train AUC', 'Validation AUC')
    save_plot(output_dir, '10_gap_analysis.png', fig)

def plot_pca_analysis(df, output_dir):
    """11. Plot PCA component analysis."""
    if 'pca_components' in df.columns:
        pca_df = df[df['pca_components'].notna()]
        
        if not pca_df.empty:
            fig, ax = plt.subplots()
            metrics = ['Train AUC', 'Validation AUC', 'Test AUC']
            
            for metric in metrics:
                sns.lineplot(data=pca_df, x='pca_components', y=metric,
                           marker='o', label=metric, ax=ax)
            
            format_plot(ax, 'Impact of PCA Components on Performance',
                       'Number of PCA Components', 'AUC Score')
            save_plot(output_dir, '11_pca_analysis.png', fig)

def plot_composite_score_analysis(df, output_dir):
    """12. Plot composite score analysis."""
    metrics = ['Composite_Score', 'Train AUC', 'Validation AUC', 'Test AUC']
    
    # Create correlation matrix for these metrics
    fig, ax = plt.subplots(figsize=(8, 6))
    correlation = df[metrics].corr()
    sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', ax=ax)
    format_plot(ax, 'Composite Score Correlations',
               '', '', legend=False)
    save_plot(output_dir, '12_composite_score_correlation.png', fig)
    
    # Create scatter plot of composite score vs validation AUC
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Composite_Score', y='Validation AUC',
                   hue='Model', style='Model', ax=ax)
    format_plot(ax, 'Composite Score vs Validation AUC',
               'Composite Score', 'Validation AUC')
    save_plot(output_dir, '12_composite_score_scatter.png', fig)

def plot_fold_analysis(df, output_dir):
    """13. Plot fold-wise performance analysis."""
    metrics = ['Train AUC', 'Validation AUC', 'Test AUC']
    melted_df = pd.melt(df, 
                        value_vars=metrics,
                        id_vars=['Fold_', 'Model'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Fold_', y='value', hue='variable', data=melted_df, ax=ax)
    format_plot(ax, 'Performance Across Folds',
               'Fold', 'AUC Score')
    save_plot(output_dir, '13_fold_analysis.png', fig)

def plot_dataset_model_heatmap(df, output_dir):
    """14. Plot dataset type and model combination heatmap."""
    pivot_data = df.pivot_table(
        values='Validation AUC',
        index='Model',
        columns='Dataset_Type',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax)
    format_plot(ax, 'Model Performance by Dataset Type',
               'Dataset Type', 'Model', legend=False)
    save_plot(output_dir, '14_dataset_model_heatmap.png', fig)

def plot_time_analysis(df, output_dir):
    """15. Plot performance over time."""
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics = ['Train AUC', 'Validation AUC', 'Test AUC']
        
        for metric in metrics:
            sns.lineplot(data=df, x='timestamp', y=metric, label=metric, ax=ax)
        
        format_plot(ax, 'Performance Over Time', 'Time', 'AUC Score', rotate_xticks=True)
        save_plot(output_dir, '15_time_analysis.png', fig)

def plot_hyperparameter_impact(df, output_dir):
    """16. Plot hyperparameter impact analysis."""
    if 'ML_Parameters' in df.columns:
        # Convert string parameters to dict if needed
        if df['ML_Parameters'].dtype == 'O':
            df['params_dict'] = df['ML_Parameters'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        
        # Common hyperparameters to analyze
        hyperparams = ['learning_rate', 'n_estimators', 'max_depth', 'C']
        
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)
        
        for i, param in enumerate(hyperparams):
            ax = fig.add_subplot(gs[i//2, i%2])
            
            # Get models that have this parameter
            mask = df['params_dict'].apply(lambda x: param in x if isinstance(x, dict) else False)
            if mask.any():
                plot_df = df[mask].copy()
                plot_df[param] = plot_df['params_dict'].apply(lambda x: x.get(param))
                
                sns.scatterplot(data=plot_df, x=param, y='Validation AUC', 
                              hue='Model', ax=ax)
                ax.set_title(f'Impact of {param} on Validation AUC')
        
        plt.tight_layout()
        save_plot(output_dir, '16_hyperparameter_impact.png', fig)

def plot_model_comparison_violin(df, output_dir):
    """17. Plot detailed model comparison with violin plots."""
    metrics = ['Train AUC', 'Validation AUC', 'Test AUC']
    melted_df = pd.melt(df, value_vars=metrics, id_vars=['Model'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(x='Model', y='value', hue='variable', data=melted_df, 
                  split=True, inner='quartile', ax=ax)
    format_plot(ax, 'Detailed Model Performance Distribution',
               'Model', 'AUC Score', rotate_xticks=True)
    save_plot(output_dir, '17_model_comparison_violin.png', fig)

def plot_features_correlation(df, output_dir):
    """18. Plot feature count correlations."""
    if 'Features' in df.columns:
        feature_metrics = ['Features', 'Train AUC', 'Validation AUC', 'Test AUC', 'Composite_Score']
        correlation = df[feature_metrics].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', ax=ax)
        format_plot(ax, 'Feature Count Correlations', '', '', legend=False)
        save_plot(output_dir, '18_features_correlation.png', fig)

def plot_performance_distribution(df, output_dir):
    """19. Plot overall performance distribution."""
    metrics = ['Train AUC', 'Validation AUC', 'Test AUC']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(metrics):
        sns.histplot(data=df, x=metric, kde=True, ax=axes[i])
        axes[i].set_title(f'{metric} Distribution')
    
    plt.tight_layout()
    save_plot(output_dir, '19_performance_distribution.png', fig)

def plot_summary_statistics(df, output_dir):
    """20. Plot summary statistics."""
    # Calculate summary statistics by model
    summary = df.groupby('Model').agg({
        'Train AUC': ['mean', 'std'],
        'Validation AUC': ['mean', 'std'],
        'Test AUC': ['mean', 'std']
    }).round(3)
    
    # Save summary to CSV
    summary.to_csv(f"{output_dir}/model_summary_statistics.csv")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    summary_plot = summary.xs('mean', axis=1, level=1)
    summary_plot.plot(kind='bar', ax=ax, yerr=summary.xs('std', axis=1, level=1))
    format_plot(ax, 'Model Performance Summary',
               'Model', 'Mean AUC Score', rotate_xticks=True)
    save_plot(output_dir, '20_summary_statistics.png', fig)

def generate_report(df, output_dir):
    """Generate a text report with key findings."""
    report = []
    
    # Overall statistics
    report.append("=== ML Results Analysis Report ===\n")
    report.append(f"Total number of experiments: {len(df)}")
    report.append(f"Number of unique models: {df['Model'].nunique()}")
    report.append(f"Number of dataset types: {df['Dataset_Type'].nunique()}\n")
    
    # Best performing models
    report.append("=== Best Performing Models ===")
    best_models = df.nlargest(3, 'Validation AUC')[['Model', 'Dataset_Type', 'Validation AUC']]
    report.append(best_models.to_string())
    
    # Save report
    with open(f"{output_dir}/analysis_report.txt", 'w') as f:
        f.write('\n'.join(report))

def merge_existing_plots(output_dir, plot_files):
    """
    Merge existing plots into a publication-quality figure with proper spacing and layout.
    
    Parameters:
        output_dir (str): Directory containing the plots
        plot_files (list): List of plot filenames to merge
    """
    if not plot_files:
        return
        
    # Import required libraries if not already imported
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import os
    
    # Set style for publication quality
    plt.style.use('seaborn-v0_8-paper')  # Keeps consistent with original
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.titlesize': 8,  # Reduced from 11 to match original
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300
    })
    
    # Calculate dimensions
    n_cols = 3
    n_rows = (len(plot_files) + 2) // n_cols  # Matches original row calculation
    fig = plt.figure(figsize=(11, 2.5 * n_rows))  # Matches original dimensions
    
    # Create grid with minimal spacing as per original
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    gs.update(wspace=0, hspace=0.1)  # Matches original minimal spacing
    
    # Adjust the figure margins to match original
    plt.subplots_adjust(left=0, right=1, top=0.92, bottom=0, wspace=0, hspace=0)
    
    # Process each plot
    for idx, plot_file in enumerate(sorted(plot_files)):
        try:
            # Read and display image
            img = plt.imread(os.path.join(output_dir, plot_file))
            ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
            ax.imshow(img, aspect='auto')  # Keeps original aspect setting
            ax.axis('off')
            
            # Extract and format title as in original
            title = plot_file.split('_', 1)[1].rsplit('.', 1)[0].replace('_', ' ').title()
            
            # Add subplot title matching original format
            ax.set_title(f"{chr(65 + idx)}) {title}",
                        fontsize=8,
                        fontweight='bold',
                        y=1.0)  # Matches original positioning
            
        except Exception as e:
            print(f"Error processing plot {plot_file}: {str(e)}")
            continue
    
    # Add main title matching original positioning and style
    fig.suptitle('Exploratory Data Analysis Overview',
                fontsize=10,
                fontweight='bold',
                y=0.98)
    
    # Save merged figure in multiple formats with original settings
    for ext in ['pdf', 'png']:
        save_path = os.path.join(output_dir, f'merged_analysis.{ext}')
        try:
            fig.savefig(save_path,
                       dpi=300,
                       bbox_inches='tight',
                       pad_inches=0.05,  # Matches original minimal padding
                       format=ext)
            print(f"Saved merged analysis as {save_path}")
        except Exception as e:
            print(f"Error saving {ext} format: {str(e)}")
    
    plt.close(fig)

def main():
    """Main execution function."""
    if len(sys.argv) != 3:
        print("Usage: python script.py <phenotype> <directory>")
        sys.exit(1)

    phenotype = sys.argv[1]
    directory = sys.argv[2]

    # Set up paths
    input_path = f"{phenotype}/Results/{directory}/Aggregated/ResultsFinal.csv"
    output_dir = f"{phenotype}/Results/{directory}/EDA"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded data from {input_path}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

    # Set up plotting style
    setup_style()

    # Dictionary of plotting functions
    plotting_functions = {
        #'Model Frequency': plot_model_frequency,
        'Phenotype GWAS Analysis': plot_auc_by_phenotype_gwas,
        'Dataset Type Performance': plot_dataset_type_performance,
        'Weight File Impact': plot_weight_file_impact,
        'SNP Analysis': plot_snp_analysis,
        'Model Performance': plot_model_performance_comparison,
        # 'Correlation Matrix': plot_correlation_matrix,
        'Stability Analysis': plot_stability_analysis,
        #'Feature Impact': plot_feature_impact,
        # 'Gap Analysis': plot_gap_analysis,
        # 'PCA Analysis': plot_pca_analysis,
        # 'Composite Score': plot_composite_score_analysis,
        # 'Fold Analysis': plot_fold_analysis,
        # 'Dataset Model Heatmap': plot_dataset_model_heatmap,
   
        # 'Hyperparameter Impact': plot_hyperparameter_impact,
        #'Model Comparison Violin': plot_model_comparison_violin,
        #'Features Correlation': plot_features_correlation,
        #'Performance Distribution': plot_performance_distribution,
        #'Summary Statistics': plot_summary_statistics
    }

    # Generate all plots
    for name, func in plotting_functions.items():
        try:
            print(f"Generating {name} plot...")
            func(df, output_dir)
        except Exception as e:
            print(f"Error generating {name} plot: {str(e)}")
            continue

    # Only merge existing plots, remove combined analysis generation
    try:
        print("Merging existing plots...")
        plot_files = [f for f in os.listdir(output_dir) if f.endswith('.png') and f[0].isdigit()]
        merge_existing_plots(output_dir, plot_files)
        print("Successfully merged plots")
    except Exception as e:
        print(f"Error merging plots: {str(e)}")

    # Clean up combined analysis if it exists
    combined_file = os.path.join(output_dir, 'combined_analysis_publication.png')
    if os.path.exists(combined_file):
        os.remove(combined_file)

    # Generate report
    try:
        generate_report(df, output_dir)
        print("Generated analysis report")
    except Exception as e:
        print(f"Error generating report: {str(e)}")

    print(f"\nAnalysis completed. Results saved in {output_dir}")
    print("\nGenerated files:")
    for file in os.listdir(output_dir):
        print(f"- {file}")

if __name__ == "__main__":
    main()