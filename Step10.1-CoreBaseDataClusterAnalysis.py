import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.gridspec as gridspec
import os
from tqdm import tqdm

def calculate_generalization_score(row):
    """Calculate generalization score based on performance metrics"""
    performances = np.array([row['Train AUC'], row['Validation AUC'], row['Test AUC']])
    mean_perf = np.mean(performances)
    std_perf = np.std(performances)
    train_test_gap = abs(row['Train AUC'] - row['Test AUC'])
    
    # Normalize components
    normalized_std = 1 - (std_perf / 0.1)
    normalized_gap = 1 - (train_test_gap / 0.1)
    
    return (0.4 * mean_perf + 0.3 * normalized_std + 0.3 * normalized_gap)

def get_dataset_features(file_path):
    """Extract statistical features from dataset"""
    try:
        data = pd.read_csv(file_path).iloc[:, 2:]
        features = {
            'mean': np.mean(data.values.flatten()),
            'std': np.std(data.values.flatten()),
            'skew': np.mean(pd.DataFrame(data).skew()),
            'kurtosis': np.mean(pd.DataFrame(data).kurtosis()),
            'q25': np.percentile(data.values.flatten(), 25),
            'q50': np.percentile(data.values.flatten(), 50),
            'q75': np.percentile(data.values.flatten(), 75),
            'iqr': np.percentile(data.values.flatten(), 75) - np.percentile(data.values.flatten(), 25),
            'pca_var_ratio': PCA(n_components=2).fit(data).explained_variance_ratio_
        }
        
        return [
            features['mean'], features['std'], features['skew'],
            features['kurtosis'], features['q25'], features['q50'],
            features['q75'], features['iqr'], 
            features['pca_var_ratio'][0], features['pca_var_ratio'][1]
        ]
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def generate_embeddings(features_scaled, cluster_centers):
    """Generate different embeddings for visualization"""
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    pca_centers = pca.transform(cluster_centers)
    
    # t-SNE
    tsne = TSNE(
        n_components=2, 
        random_state=42, 
        perplexity=min(30, len(features_scaled) - 1)
    )
    tsne_result = tsne.fit_transform(features_scaled)
    
    # Handle t-SNE for cluster centers
    if len(cluster_centers) > 1:
        tsne_centers = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(len(cluster_centers) - 1, 5)
        ).fit_transform(cluster_centers)
    else:
        tsne_centers = pca.transform(cluster_centers)
    
    # UMAP
    umap_reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=min(15, len(features_scaled) - 1)
    )
    umap_result = umap_reducer.fit_transform(features_scaled)
    umap_centers = umap_reducer.transform(cluster_centers)
    
    return {
        'pca': pca_result,
        'tsne': tsne_result,
        'umap': umap_result,
        'cluster_centers_pca': pca_centers,
        'cluster_centers_tsne': tsne_centers,
        'cluster_centers_umap': umap_centers
    }

def create_visualization(embeddings_dict, performance_data, cluster_labels, phenotype, method_dir):
    """Create publication-quality visualization of all analyses"""
    # Convert method directory to method name
    method_name = "Machine Learning" if method_dir == "ResultsML" else "Deep Learning"
    
    # Format phenotype name to title case
    phenotype_name = phenotype.replace('_', ' ').title()
    
    # Set style for publication
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 8,  # Reduced X-axis tick size
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300
    })
    
    fig = plt.figure(figsize=(20, 16))  # Increased height to accommodate main title
    
    # Add main title with phenotype and method
    fig.suptitle(f'Cluster Analysis of {method_name} Model Performance\nPhenotype: {phenotype_name}', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # Plot settings
    marker_size = 100
    alpha = 0.8
    cmap = plt.cm.viridis  # Professional colormap
    
    # Plot titles with bold font
    titles = {
        'pca': 'a) PCA Analysis',
        'tsne': 'b) t-SNE Analysis',
        'umap': 'c) UMAP Analysis',
        'cluster_perf': 'd) Cluster Performance',
        'perf_dist': 'e) Performance Distribution',
        'cluster_dist': 'f) Cluster Size Distribution'
    }
    
    # Create subplots
    axes = {
        'pca': fig.add_subplot(gs[0, 0]),
        'tsne': fig.add_subplot(gs[0, 1]),
        'umap': fig.add_subplot(gs[0, 2]),
        'cluster_perf': fig.add_subplot(gs[1, 0]),
        'perf_dist': fig.add_subplot(gs[1, 1]),
        'cluster_dist': fig.add_subplot(gs[1, 2])
    }
    
    # Plot embeddings with enhanced styling
    for plot_type in ['pca', 'tsne', 'umap']:
        scatter = axes[plot_type].scatter(
            embeddings_dict[plot_type][:, 0], 
            embeddings_dict[plot_type][:, 1],
            c=performance_data['Test AUC'],  # Changed to Test AUC
            cmap=cmap, s=marker_size, alpha=alpha,
            edgecolors='white', linewidth=0.5
        )
        axes[plot_type].scatter(
            embeddings_dict[f'cluster_centers_{plot_type}'][:, 0],
            embeddings_dict[f'cluster_centers_{plot_type}'][:, 1],
            c='red', marker='X', s=300, linewidths=2, 
            edgecolors='black', label='Cluster Centers'
        )
        axes[plot_type].set_title(titles[plot_type], pad=20, fontweight='bold')
        axes[plot_type].set_xlabel('Component 1', labelpad=10)
        axes[plot_type].set_ylabel('Component 2', labelpad=10)
        axes[plot_type].legend(frameon=True, fancybox=True, shadow=True)
        axes[plot_type].grid(True, linestyle='--', alpha=0.3)

    # Enhanced cluster performance plot
    cluster_scores = pd.DataFrame({
        'Cluster': range(len(np.unique(cluster_labels))),
        'Mean_Performance': [performance_data[cluster_labels == i]['Test AUC'].mean() 
                           for i in range(len(np.unique(cluster_labels)))],
        'Std_Performance': [performance_data[cluster_labels == i]['Test AUC'].std() 
                          for i in range(len(np.unique(cluster_labels)))]
    })
    axes['cluster_perf'].errorbar(
        cluster_scores['Cluster'], 
        cluster_scores['Mean_Performance'],
        yerr=cluster_scores['Std_Performance'], 
        fmt='o', capsize=5, capthick=2, 
        elinewidth=2, markersize=10,
        color='darkblue', ecolor='gray'
    )
    axes['cluster_perf'].set_title(titles['cluster_perf'], pad=20, fontweight='bold')
    axes['cluster_perf'].set_xlabel('Cluster ID', labelpad=10)
    axes['cluster_perf'].set_ylabel('Mean Test AUC', labelpad=10)
    axes['cluster_perf'].grid(True, linestyle='--', alpha=0.3)

    # Enhanced performance distribution
    performance_data['cluster'] = cluster_labels
    sns.histplot(
        data=performance_data, 
        x='Test AUC',  # Changed to Test AUC
        bins=30, 
        hue='cluster',
        multiple="stack",
        ax=axes['perf_dist'],
        palette='viridis'
    )
    axes['perf_dist'].set_title(titles['perf_dist'], pad=20, fontweight='bold')
    axes['perf_dist'].set_xlabel('Test AUC', labelpad=10)
    axes['perf_dist'].set_ylabel('Count', labelpad=10)

    # Enhanced cluster distribution
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    bars = axes['cluster_dist'].bar(
        cluster_sizes.index, 
        cluster_sizes.values,
        color=plt.cm.viridis(np.linspace(0, 1, len(cluster_sizes))),
        edgecolor='black',
        linewidth=1
    )
    axes['cluster_dist'].set_title(titles['cluster_dist'], pad=20, fontweight='bold')
    axes['cluster_dist'].set_xlabel('Cluster ID', labelpad=10)
    axes['cluster_dist'].set_ylabel('Number of Datasets', labelpad=10)

    # Enhanced colorbar - moved further right
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])  # Changed from [0.92, 0.15, 0.02, 0.7]
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Test AUC', rotation=270, labelpad=25)
    
    # Adjust layout with space for main title and colorbar
    plt.tight_layout(rect=[0, 0, 0.94, 0.96])  # Changed from [0, 0, 0.94, 1]
    return fig

def save_results(vis_path, dataset_indices, cluster_labels, performance_data, 
                embeddings_dict, original_performance_data):
    """Save analysis results to CSV files"""
    results_df = pd.DataFrame({
        'Dataset': dataset_indices,
        'Cluster': cluster_labels,
        'Generalization_Score': performance_data['generalization_score'],
        'PCA_1': embeddings_dict['pca'][:, 0],
        'PCA_2': embeddings_dict['pca'][:, 1],
        'TSNE_1': embeddings_dict['tsne'][:, 0],
        'TSNE_2': embeddings_dict['tsne'][:, 1],
        'UMAP_1': embeddings_dict['umap'][:, 0],
        'UMAP_2': embeddings_dict['umap'][:, 1]
    })
    
    # Add performance metrics
    results_df = pd.merge(
        results_df,
        original_performance_data[['Dataset', 'Train AUC', 'Validation AUC', 'Test AUC']],
        on='Dataset'
    )
    
    # Calculate cluster statistics
    cluster_stats = results_df.groupby('Cluster').agg({
        'Generalization_Score': ['mean', 'std', 'min', 'max'],
        'Train AUC': 'mean',
        'Validation AUC': 'mean',
        'Test AUC': 'mean',
        'Dataset': 'count'
    }).round(4)
    
    # Save to files
    results_df.to_csv(f'{vis_path}/complete_analysis.csv', index=False)
    cluster_stats.to_csv(f'{vis_path}/cluster_statistics.csv')
    
    return cluster_stats

def analyze_datasets(phenotype, directory):
    """Main analysis pipeline"""
    # Setup paths
    base_path = f"{phenotype}/Fold_0/Datasets"
    vis_path = f"{phenotype}/Results/{directory}/Visualization"
    performance_file = f"{phenotype}/Results/{directory}/Aggregated/ResultsFinal.csv"
    
    # Create visualization directory
    os.makedirs(vis_path, exist_ok=True)
    
    # Load and process performance data
    print("Reading performance metrics...")
    performance_data = pd.read_csv(performance_file)
    performance_data['generalization_score'] = performance_data.apply(
        calculate_generalization_score, axis=1
    )
    
    # Get valid datasets
    print("Loading dataset IDs...")
    with open(f"{phenotype}/Results/UniqueDatasets.txt", 'r') as f:
        datasets = [int(x.replace("dataset_", "")) for x in f.read().splitlines()]
    
    # Extract features
    print("Extracting features from datasets...")
    dataset_features = []
    valid_dataset_indices = []
    for dataset_id in tqdm(datasets):
        path = f"{base_path}/dataset_{dataset_id}/dataset_{dataset_id}_X_train.csv"
        features = get_dataset_features(path)
        if features is not None:
            dataset_features.append(features)
            valid_dataset_indices.append(dataset_id)
    
    # Prepare data for analysis
    dataset_features = np.array(dataset_features)
    dataset_features_scaled = StandardScaler().fit_transform(dataset_features)
    valid_performance_data = performance_data[
        performance_data['Dataset'].isin(valid_dataset_indices)
    ].sort_values('Dataset').reset_index(drop=True)
    
    # Perform clustering and dimensionality reduction
    print("Performing clustering and dimensionality reduction...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(dataset_features_scaled)
    
    # Generate embeddings
    embeddings_dict = generate_embeddings(
        dataset_features_scaled, 
        kmeans.cluster_centers_
    )
    
    # Create and save visualization
    print("Generating visualization...")
    fig = create_visualization(
        embeddings_dict, 
        valid_performance_data, 
        cluster_labels,
        phenotype,      # Add phenotype argument
        directory       # Add directory argument
    )
    plt.savefig(f'{vis_path}/dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    cluster_stats = save_results(
        vis_path,
        valid_dataset_indices, 
        cluster_labels, 
        valid_performance_data, 
        embeddings_dict, 
        performance_data
    )
    
    print(f"Analysis complete! Results saved in: {vis_path}")
    print("\nCluster Statistics Summary:")
    print(cluster_stats)

def main():
    import sys
    phenotype = sys.argv[1]
    directory = sys.argv[2]
    
    analyze_datasets(phenotype, directory)

if __name__ == "__main__":
    main()