"""
DSA 2040 Practical Exam - Section 2: Data Mining
Task 2: Clustering Analysis on Iris Dataset
Author: IRANZI513
Date: 2024

This module implements comprehensive K-Means clustering analysis on the Iris dataset
including elbow curve analysis, optimal cluster determination, and evaluation metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class IrisClustering:
    """
    Comprehensive K-Means clustering analysis for Iris dataset
    """
    
    def __init__(self):
        """Initialize the clustering analyzer"""
        self.scaler = StandardScaler()
        self.kmeans_models = {}
        self.cluster_results = {}
        self.evaluation_metrics = {}
        
    def load_and_prepare_data(self, use_preprocessed=True):
        """
        Load and prepare Iris dataset for clustering
        
        Args:
            use_preprocessed (bool): Whether to use preprocessed data if available
            
        Returns:
            tuple: (features_normalized, true_labels, feature_names, target_names)
        """
        try:
            # Try to load preprocessed data first
            if use_preprocessed:
                try:
                    preprocessed_data = pd.read_csv('iris_preprocessed.csv')
                    features = preprocessed_data.drop(['species', 'species_encoded'], axis=1, errors='ignore')
                    true_labels = preprocessed_data['species'] if 'species' in preprocessed_data.columns else None
                    print("Loaded preprocessed Iris data")
                except FileNotFoundError:
                    print("Preprocessed data not found, loading from sklearn...")
                    use_preprocessed = False
            
            if not use_preprocessed:
                # Load original Iris dataset
                iris = load_iris()
                features = pd.DataFrame(iris.data, columns=iris.feature_names)
                true_labels = pd.Series(iris.target_names[iris.target], name='species')
                feature_names = iris.feature_names
                target_names = iris.target_names
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None, None
        
        # Extract feature and target information
        feature_names = features.columns.tolist()
        target_names = true_labels.unique() if true_labels is not None else ['setosa', 'versicolor', 'virginica']
        
        # Normalize features
        features_normalized = pd.DataFrame(
            self.scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        
        print(f"Dataset loaded: {features.shape[0]} samples, {features.shape[1]} features")
        print(f"Features: {list(feature_names)}")
        print(f"Target classes: {list(target_names)}")
        if true_labels is not None:
            print(f"Class distribution: {true_labels.value_counts().to_dict()}")
        
        return features_normalized, true_labels, feature_names, target_names
    
    def perform_kmeans_clustering(self, features, k=3, random_state=42):
        """
        Perform K-Means clustering
        
        Args:
            features (DataFrame): Normalized feature data
            k (int): Number of clusters
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (cluster_labels, kmeans_model)
        """
        try:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Store model for later use
            self.kmeans_models[k] = kmeans
            
            print(f"K-Means clustering completed with k={k}")
            print(f"Cluster distribution: {np.bincount(cluster_labels)}")
            
            return cluster_labels, kmeans
            
        except Exception as e:
            print(f"Error in K-Means clustering: {e}")
            return None, None
    
    def elbow_curve_analysis(self, features, max_k=10):
        """
        Perform elbow curve analysis to determine optimal number of clusters
        
        Args:
            features (DataFrame): Normalized feature data
            max_k (int): Maximum number of clusters to test
            
        Returns:
            dict: Dictionary with k values and corresponding inertias
        """
        inertias = []
        k_values = range(1, max_k + 1)
        
        print("Performing elbow curve analysis...")
        
        for k in k_values:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(features)
                inertias.append(kmeans.inertia_)
                print(f"k={k}: inertia={kmeans.inertia_:.2f}")
                
            except Exception as e:
                print(f"Error for k={k}: {e}")
                inertias.append(np.nan)
        
        elbow_data = {'k': list(k_values), 'inertia': inertias}
        return elbow_data
    
    def calculate_silhouette_scores(self, features, max_k=10):
        """
        Calculate silhouette scores for different k values
        
        Args:
            features (DataFrame): Normalized feature data
            max_k (int): Maximum number of clusters to test
            
        Returns:
            dict: Dictionary with k values and corresponding silhouette scores
        """
        silhouette_scores = []
        k_values = range(2, max_k + 1)  # Silhouette score requires at least 2 clusters
        
        print("Calculating silhouette scores...")
        
        for k in k_values:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                score = silhouette_score(features, cluster_labels)
                silhouette_scores.append(score)
                print(f"k={k}: silhouette score={score:.3f}")
                
            except Exception as e:
                print(f"Error for k={k}: {e}")
                silhouette_scores.append(np.nan)
        
        silhouette_data = {'k': list(k_values), 'silhouette_score': silhouette_scores}
        return silhouette_data
    
    def evaluate_clustering_performance(self, features, cluster_labels, true_labels=None):
        """
        Evaluate clustering performance using multiple metrics
        
        Args:
            features (DataFrame): Feature data
            cluster_labels (array): Predicted cluster labels
            true_labels (Series): True class labels (if available)
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        metrics = {}
        
        try:
            # Internal validation metrics
            metrics['silhouette_score'] = silhouette_score(features, cluster_labels)
            metrics['inertia'] = self.kmeans_models[len(np.unique(cluster_labels))].inertia_
            
            # External validation metrics (if true labels available)
            if true_labels is not None:
                # Convert true labels to numeric if they're strings
                if true_labels.dtype == 'object':
                    true_labels_encoded = pd.Categorical(true_labels).codes
                else:
                    true_labels_encoded = true_labels
                
                metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels_encoded, cluster_labels)
                metrics['homogeneity_score'] = homogeneity_score(true_labels_encoded, cluster_labels)
                metrics['completeness_score'] = completeness_score(true_labels_encoded, cluster_labels)
                metrics['v_measure_score'] = v_measure_score(true_labels_encoded, cluster_labels)
            
            print("Clustering Performance Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.3f}")
                
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            
        return metrics
    
    def visualize_elbow_curve(self, elbow_data, save_path='elbow_curve.png'):
        """
        Create elbow curve visualization
        
        Args:
            elbow_data (dict): Elbow curve data
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(elbow_data['k'], elbow_data['inertia'], 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=12)
        plt.title('Elbow Curve for Optimal k Selection', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Annotate points
        for i, (k, inertia) in enumerate(zip(elbow_data['k'], elbow_data['inertia'])):
            if not np.isnan(inertia):
                plt.annotate(f'{inertia:.1f}', (k, inertia), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Elbow curve saved to {save_path}")
    
    def visualize_silhouette_analysis(self, silhouette_data, save_path='silhouette_analysis.png'):
        """
        Create silhouette analysis visualization
        
        Args:
            silhouette_data (dict): Silhouette analysis data
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(silhouette_data['k'], silhouette_data['silhouette_score'], 'ro-', 
                linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Silhouette Analysis for Optimal k Selection', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Find and highlight optimal k
        if silhouette_data['silhouette_score']:
            max_score_idx = np.nanargmax(silhouette_data['silhouette_score'])
            optimal_k = silhouette_data['k'][max_score_idx]
            max_score = silhouette_data['silhouette_score'][max_score_idx]
            plt.scatter(optimal_k, max_score, color='red', s=100, zorder=5)
            plt.annotate(f'Optimal k={optimal_k}\nScore={max_score:.3f}', 
                        (optimal_k, max_score), textcoords="offset points", 
                        xytext=(20,20), ha='left', fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Annotate points
        for i, (k, score) in enumerate(zip(silhouette_data['k'], silhouette_data['silhouette_score'])):
            if not np.isnan(score):
                plt.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Silhouette analysis saved to {save_path}")
    
    def visualize_clusters_2d(self, features, cluster_labels, true_labels=None, 
                             save_path='clusters_2d.png'):
        """
        Create 2D visualization of clusters using PCA
        
        Args:
            features (DataFrame): Feature data
            cluster_labels (array): Predicted cluster labels
            true_labels (Series): True class labels (optional)
            save_path (str): Path to save the plot
        """
        # Perform PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2 if true_labels is not None else 1, figsize=(15, 6))
        if true_labels is None:
            axes = [axes]
        
        # Plot predicted clusters
        scatter = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=cluster_labels, cmap='viridis', alpha=0.7, s=50)
        axes[0].set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0].set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[0].set_title('K-Means Clustering Results (PCA Visualization)')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0], label='Cluster')
        
        # Plot true labels if available
        if true_labels is not None:
            # Convert true labels to numeric for coloring
            if true_labels.dtype == 'object':
                true_labels_encoded = pd.Categorical(true_labels).codes
                label_names = true_labels.unique()
            else:
                true_labels_encoded = true_labels
                label_names = [f'Class {i}' for i in range(len(np.unique(true_labels_encoded)))]
            
            scatter2 = axes[1].scatter(features_2d[:, 0], features_2d[:, 1], 
                                     c=true_labels_encoded, cmap='Set1', alpha=0.7, s=50)
            axes[1].set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
            axes[1].set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
            axes[1].set_title('True Class Labels (PCA Visualization)')
            axes[1].grid(True, alpha=0.3)
            
            # Create custom legend
            import matplotlib.patches as mpatches
            legend_elements = [mpatches.Patch(color=plt.cm.Set1(i), label=label_names[i]) 
                             for i in range(len(label_names))]
            axes[1].legend(handles=legend_elements, title='True Classes')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Cluster visualization saved to {save_path}")
        
        # Print PCA information
        print(f"PCA Explained Variance Ratios: {pca.explained_variance_ratio_}")
        print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_):.2%}")
    
    def generate_clustering_report(self, features, cluster_labels, true_labels, 
                                 feature_names, metrics, save_path='clustering_report.txt'):
        """
        Generate comprehensive clustering analysis report
        
        Args:
            features (DataFrame): Feature data
            cluster_labels (array): Predicted cluster labels
            true_labels (Series): True class labels
            feature_names (list): List of feature names
            metrics (dict): Evaluation metrics
            save_path (str): Path to save the report
        """
        report = []
        report.append("=" * 60)
        report.append("IRIS DATASET CLUSTERING ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Dataset information
        report.append("DATASET INFORMATION:")
        report.append(f"  Number of samples: {len(features)}")
        report.append(f"  Number of features: {len(feature_names)}")
        report.append(f"  Features: {', '.join(feature_names)}")
        report.append(f"  Number of clusters found: {len(np.unique(cluster_labels))}")
        report.append("")
        
        # Cluster distribution
        report.append("CLUSTER DISTRIBUTION:")
        cluster_counts = np.bincount(cluster_labels)
        for i, count in enumerate(cluster_counts):
            percentage = (count / len(cluster_labels)) * 100
            report.append(f"  Cluster {i}: {count} samples ({percentage:.1f}%)")
        report.append("")
        
        # Performance metrics
        report.append("CLUSTERING PERFORMANCE METRICS:")
        for metric, value in metrics.items():
            report.append(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
        report.append("")
        
        # Cluster characteristics
        report.append("CLUSTER CHARACTERISTICS:")
        features_with_clusters = features.copy()
        features_with_clusters['cluster'] = cluster_labels
        
        for cluster_id in range(len(np.unique(cluster_labels))):
            cluster_data = features_with_clusters[features_with_clusters['cluster'] == cluster_id]
            report.append(f"  Cluster {cluster_id} (n={len(cluster_data)}):")
            
            for feature in feature_names:
                mean_val = cluster_data[feature].mean()
                std_val = cluster_data[feature].std()
                report.append(f"    {feature}: {mean_val:.3f} ± {std_val:.3f}")
            report.append("")
        
        # Confusion matrix with true labels
        if true_labels is not None:
            report.append("CLUSTER vs TRUE LABEL MAPPING:")
            # Convert true labels to numeric if needed
            if true_labels.dtype == 'object':
                true_labels_encoded = pd.Categorical(true_labels).codes
                label_names = true_labels.unique()
            else:
                true_labels_encoded = true_labels
                label_names = [f'Class {i}' for i in range(len(np.unique(true_labels_encoded)))]
            
            confusion_df = pd.crosstab(cluster_labels, true_labels_encoded, 
                                     rownames=['Cluster'], colnames=['True Label'])
            report.append(str(confusion_df))
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if 'silhouette_score' in metrics:
            if metrics['silhouette_score'] > 0.7:
                report.append("  - Excellent clustering quality (Silhouette Score > 0.7)")
            elif metrics['silhouette_score'] > 0.5:
                report.append("  - Good clustering quality (Silhouette Score > 0.5)")
            else:
                report.append("  - Consider trying different clustering parameters or algorithms")
        
        if 'adjusted_rand_score' in metrics:
            if metrics['adjusted_rand_score'] > 0.8:
                report.append("  - Clusters align very well with true classes")
            elif metrics['adjusted_rand_score'] > 0.5:
                report.append("  - Clusters show moderate alignment with true classes")
            else:
                report.append("  - Clusters do not align well with true classes")
        
        report.append("")
        report.append("=" * 60)
        
        # Save report
        report_text = '\n'.join(report)
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"Clustering report saved to {save_path}")
        print(report_text)
    
    def run_complete_analysis(self, max_k=8):
        """
        Run complete clustering analysis pipeline
        
        Args:
            max_k (int): Maximum number of clusters to analyze
        """
        print("Starting Complete K-Means Clustering Analysis")
        print("=" * 50)
        
        # Load and prepare data
        features, true_labels, feature_names, target_names = self.load_and_prepare_data()
        if features is None:
            print("Failed to load data. Exiting analysis.")
            return
        
        # Elbow curve analysis
        print("\n1. Performing Elbow Curve Analysis...")
        elbow_data = self.elbow_curve_analysis(features, max_k)
        self.visualize_elbow_curve(elbow_data)
        
        # Silhouette analysis
        print("\n2. Performing Silhouette Analysis...")
        silhouette_data = self.calculate_silhouette_scores(features, max_k)
        self.visualize_silhouette_analysis(silhouette_data)
        
        # Determine optimal k (highest silhouette score)
        if silhouette_data['silhouette_score']:
            optimal_k = silhouette_data['k'][np.nanargmax(silhouette_data['silhouette_score'])]
        else:
            optimal_k = 3  # Default to 3 for Iris dataset
        
        print(f"\n3. Performing K-Means Clustering with optimal k={optimal_k}...")
        cluster_labels, kmeans_model = self.perform_kmeans_clustering(features, optimal_k)
        
        if cluster_labels is None:
            print("Clustering failed. Exiting analysis.")
            return
        
        # Evaluate clustering performance
        print("\n4. Evaluating Clustering Performance...")
        metrics = self.evaluate_clustering_performance(features, cluster_labels, true_labels)
        
        # Create visualizations
        print("\n5. Creating Visualizations...")
        self.visualize_clusters_2d(features, cluster_labels, true_labels)
        
        # Generate comprehensive report
        print("\n6. Generating Analysis Report...")
        self.generate_clustering_report(features, cluster_labels, true_labels, 
                                      feature_names, metrics)
        
        print("\nComplete clustering analysis finished successfully!")
        print("Generated files:")
        print("  - elbow_curve.png")
        print("  - silhouette_analysis.png") 
        print("  - clusters_2d.png")
        print("  - clustering_report.txt")


def main():
    """Main function to run the clustering analysis"""
    try:
        # Create clustering analyzer
        analyzer = IrisClustering()
        
        # Run complete analysis
        analyzer.run_complete_analysis(max_k=8)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
