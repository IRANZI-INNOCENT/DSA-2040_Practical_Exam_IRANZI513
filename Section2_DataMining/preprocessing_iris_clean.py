"""
DSA 2040 Practical Exam - Data Preprocessing and Exploration
Task 1: Data preprocessing and exploration on Iris dataset
Author: IRANZI513
Date: August 14, 2025

This script demonstrates comprehensive data preprocessing and exploration
techniques using the Iris dataset from scikit-learn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class IrisDataPreprocessing:
    """
    Comprehensive data preprocessing and exploration for Iris dataset
    """
    
    def __init__(self, use_synthetic_data=False):
        """
        Initialize the preprocessing class
        
        Args:
            use_synthetic_data (bool): If True, generates synthetic data; 
                                     If False, uses real Iris dataset
        """
        self.use_synthetic_data = use_synthetic_data
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        
    def generate_synthetic_iris_data(self, n_samples=150):
        """
        Generate synthetic data similar to Iris dataset
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            tuple: (features DataFrame, target Series)
        """
        print(f"Generating {n_samples} synthetic Iris-like samples...")
        
        np.random.seed(42)  # For reproducibility
        
        # Define approximate parameters for each species based on real Iris data
        species_params = {
            'setosa': {
                'sepal_length': (4.8, 5.2, 0.3),    # mean, max, std
                'sepal_width': (3.2, 3.6, 0.3),
                'petal_length': (1.3, 1.7, 0.2),
                'petal_width': (0.1, 0.3, 0.1)
            },
            'versicolor': {
                'sepal_length': (5.5, 6.5, 0.4),
                'sepal_width': (2.2, 3.2, 0.3),
                'petal_length': (3.5, 4.8, 0.4),
                'petal_width': (1.0, 1.6, 0.2)
            },
            'virginica': {
                'sepal_length': (6.0, 7.5, 0.5),
                'sepal_width': (2.5, 3.5, 0.3),
                'petal_length': (4.8, 6.5, 0.5),
                'petal_width': (1.5, 2.5, 0.3)
            }
        }
        
        # Generate data for each species
        features = []
        targets = []
        
        samples_per_species = n_samples // 3
        
        for i, (species, params) in enumerate(species_params.items()):
            for _ in range(samples_per_species):
                # Generate features with some correlation structure
                sepal_length = np.random.normal(params['sepal_length'][0], params['sepal_length'][2])
                sepal_width = np.random.normal(params['sepal_width'][0], params['sepal_width'][2])
                petal_length = np.random.normal(params['petal_length'][0], params['petal_length'][2])
                petal_width = np.random.normal(params['petal_width'][0], params['petal_width'][2])
                
                # Add some correlation between petal length and width
                petal_width += petal_length * 0.1 + np.random.normal(0, 0.05)
                
                # Ensure positive values
                features.append([
                    max(0.1, sepal_length),
                    max(0.1, sepal_width),
                    max(0.1, petal_length),
                    max(0.1, petal_width)
                ])
                targets.append(species)
        
        # Create DataFrames
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        features_df = pd.DataFrame(features, columns=feature_names)
        targets_series = pd.Series(targets, name='species')
        
        print(f"Generated synthetic data with {len(features_df)} samples")
        print(f"Species distribution: {targets_series.value_counts().to_dict()}")
        
        return features_df, targets_series
    
    def load_dataset(self):
        """
        Load the Iris dataset or generate synthetic data
        
        Returns:
            tuple: (features DataFrame, target Series, feature names, target names)
        """
        print("Loading dataset...")
        
        if self.use_synthetic_data:
            # Generate synthetic data
            features_df, target_series = self.generate_synthetic_iris_data()
            feature_names = features_df.columns.tolist()
            target_names = target_series.unique().tolist()
            
            # Save synthetic data
            synthetic_data = features_df.copy()
            synthetic_data['species'] = target_series
            synthetic_data.to_csv('c:/DSA 2040_Practical_Exam_IRANZI513/Data/synthetic_iris_data.csv', index=False)
            print("Synthetic data saved to: Data/synthetic_iris_data.csv")
            
        else:
            # Load real Iris dataset
            iris = load_iris()
            features_df = pd.DataFrame(iris.data, columns=iris.feature_names)
            target_series = pd.Series(iris.target, name='species')
            
            # Convert numerical targets to species names
            target_names = iris.target_names
            target_series = target_series.map(dict(enumerate(target_names)))
            
            feature_names = iris.feature_names
            
            # Save real data for reference
            real_data = features_df.copy()
            real_data['species'] = target_series
            real_data.to_csv('c:/DSA 2040_Practical_Exam_IRANZI513/Data/iris_data.csv', index=False)
            print("Real Iris data saved to: Data/iris_data.csv")
        
        print(f"Dataset loaded: {features_df.shape[0]} samples, {features_df.shape[1]} features")
        print(f"Features: {feature_names}")
        print(f"Target classes: {target_names}")
        
        return features_df, target_series, feature_names, target_names
    
    def check_missing_values(self, df):
        """
        Check and report missing values in the dataset
        
        Args:
            df (pd.DataFrame): Dataset to check
            
        Returns:
            pd.DataFrame: Missing value report
        """
        print("\n" + "="*60)
        print("MISSING VALUES ANALYSIS")
        print("="*60)
        
        missing_info = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': [df[col].isnull().sum() for col in df.columns],
            'Missing_Percentage': [df[col].isnull().sum() / len(df) * 100 for col in df.columns],
            'Data_Type': [df[col].dtype for col in df.columns]
        })
        
        print("Missing Values Report:")
        print(missing_info)
        
        total_missing = missing_info['Missing_Count'].sum()
        
        if total_missing == 0:
            print("\n✓ No missing values found in the dataset!")
        else:
            print(f"\n⚠ Total missing values: {total_missing}")
            print("Columns with missing values:")
            problematic_cols = missing_info[missing_info['Missing_Count'] > 0]
            print(problematic_cols)
        
        return missing_info
    
    def normalize_features(self, features_df):
        """
        Normalize features using Min-Max scaling
        
        Args:
            features_df (pd.DataFrame): Features to normalize
            
        Returns:
            pd.DataFrame: Normalized features
        """
        print("\n" + "="*60)
        print("FEATURE NORMALIZATION (MIN-MAX SCALING)")
        print("="*60)
        
        print("Original feature ranges:")
        for col in features_df.columns:
            min_val = features_df[col].min()
            max_val = features_df[col].max()
            print(f"{col}: [{min_val:.3f}, {max_val:.3f}]")
        
        # Apply Min-Max scaling
        normalized_features = pd.DataFrame(
            self.scaler.fit_transform(features_df),
            columns=features_df.columns,
            index=features_df.index
        )
        
        print("\nNormalized feature ranges:")
        for col in normalized_features.columns:
            min_val = normalized_features[col].min()
            max_val = normalized_features[col].max()
            print(f"{col}: [{min_val:.3f}, {max_val:.3f}]")
        
        return normalized_features
    
    def encode_target_labels(self, target_series):
        """
        Encode target labels for machine learning models
        
        Args:
            target_series (pd.Series): Target labels
            
        Returns:
            tuple: (encoded labels, label mapping)
        """
        print("\n" + "="*60)
        print("TARGET LABEL ENCODING")
        print("="*60)
        
        print("Original target distribution:")
        print(target_series.value_counts())
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(target_series)
        encoded_series = pd.Series(encoded_labels, index=target_series.index, name='species_encoded')
        
        # Create mapping
        label_mapping = dict(zip(self.label_encoder.classes_, 
                                self.label_encoder.transform(self.label_encoder.classes_)))
        
        print("\nLabel encoding mapping:")
        for original, encoded in label_mapping.items():
            print(f"{original} -> {encoded}")
        
        print("\nEncoded target distribution:")
        print(encoded_series.value_counts().sort_index())
        
        return encoded_series, label_mapping
    
    def compute_summary_statistics(self, features_df, target_series):
        """
        Compute comprehensive summary statistics
        
        Args:
            features_df (pd.DataFrame): Features
            target_series (pd.Series): Target labels
        """
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        # Overall statistics
        print("Overall Feature Statistics:")
        summary_stats = features_df.describe()
        print(summary_stats)
        
        # Statistics by species
        print("\nStatistics by Species:")
        combined_df = features_df.copy()
        combined_df['species'] = target_series
        
        for species in target_series.unique():
            print(f"\n{species.upper()} Statistics:")
            species_data = combined_df[combined_df['species'] == species][features_df.columns]
            print(species_data.describe())
    
    def create_visualizations(self, features_df, target_series):
        """
        Create comprehensive visualizations for data exploration
        
        Args:
            features_df (pd.DataFrame): Features
            target_series (pd.Series): Target labels
        """
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Pairplot
        plt.subplot(3, 3, 1)
        combined_df = features_df.copy()
        combined_df['species'] = target_series
        
        # Create a simplified pairplot using scatter plots
        feature_cols = features_df.columns[:2]  # First two features for visibility
        for i, species in enumerate(target_series.unique()):
            species_data = combined_df[combined_df['species'] == species]
            plt.scatter(species_data[feature_cols[0]], species_data[feature_cols[1]], 
                       label=species, alpha=0.7, s=50)
        
        plt.xlabel(feature_cols[0].replace('_', ' ').title())
        plt.ylabel(feature_cols[1].replace('_', ' ').title())
        plt.title('Pairplot: Sepal Length vs Sepal Width')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Correlation Heatmap
        plt.subplot(3, 3, 2)
        correlation_matrix = features_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Heatmap')
        
        # 3-6. Box plots for outlier detection
        for i, col in enumerate(features_df.columns):
            plt.subplot(3, 3, 3 + i)
            
            # Create box plot by species
            species_data = []
            species_labels = []
            
            for species in target_series.unique():
                mask = target_series == species
                species_data.append(features_df.loc[mask, col])
                species_labels.append(species)
            
            plt.boxplot(species_data, labels=species_labels)
            plt.title(f'Box Plot: {col.replace("_", " ").title()}')
            plt.ylabel('Value')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 7. Distribution plots
        plt.subplot(3, 3, 7)
        for species in target_series.unique():
            mask = target_series == species
            plt.hist(features_df.loc[mask, features_df.columns[0]], 
                    alpha=0.6, label=species, bins=15)
        plt.xlabel(features_df.columns[0].replace('_', ' ').title())
        plt.ylabel('Frequency')
        plt.title('Distribution by Species')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Feature importance (variance)
        plt.subplot(3, 3, 8)
        feature_variance = features_df.var().sort_values(ascending=True)
        plt.barh(range(len(feature_variance)), feature_variance.values)
        plt.yticks(range(len(feature_variance)), 
                  [name.replace('_', ' ').title() for name in feature_variance.index])
        plt.xlabel('Variance')
        plt.title('Feature Variance Analysis')
        plt.grid(True, alpha=0.3)
        
        # 9. Species distribution
        plt.subplot(3, 3, 9)
        species_counts = target_series.value_counts()
        colors = sns.color_palette("husl", len(species_counts))
        plt.pie(species_counts.values, labels=species_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        plt.title('Species Distribution')
        
        plt.tight_layout()
        plt.savefig('c:/DSA 2040_Practical_Exam_IRANZI513/Visualizations/iris_exploration.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        
        print("Visualizations saved to: Visualizations/iris_exploration.png")
        
        # Additional pairplot using seaborn
        plt.figure(figsize=(12, 10))
        sns.pairplot(combined_df, hue='species', diag_kind='hist', 
                    plot_kws={'alpha': 0.7}, diag_kws={'alpha': 0.7})
        plt.suptitle('Comprehensive Pairplot of Iris Features', y=1.02)
        plt.savefig('c:/DSA 2040_Practical_Exam_IRANZI513/Visualizations/iris_pairplot.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        
        print("Pairplot saved to: Visualizations/iris_pairplot.png")
        
        return fig
    
    def identify_outliers(self, features_df, target_series):
        """
        Identify potential outliers using IQR method
        
        Args:
            features_df (pd.DataFrame): Features
            target_series (pd.Series): Target labels
            
        Returns:
            dict: Outlier information by feature
        """
        print("\n" + "="*60)
        print("OUTLIER DETECTION (IQR METHOD)")
        print("="*60)
        
        outlier_info = {}
        
        for col in features_df.columns:
            Q1 = features_df[col].quantile(0.25)
            Q3 = features_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = features_df[(features_df[col] < lower_bound) | 
                                 (features_df[col] > upper_bound)]
            
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(features_df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_indices': outliers.index.tolist(),
                'outlier_values': outliers[col].tolist()
            }
            
            print(f"\n{col.replace('_', ' ').title()}:")
            print(f"  Range: [{lower_bound:.3f}, {upper_bound:.3f}]")
            print(f"  Outliers: {len(outliers)} ({len(outliers)/len(features_df)*100:.1f}%)")
            
            if len(outliers) > 0:
                print(f"  Outlier values: {[f'{v:.3f}' for v in outliers[col].head(5).tolist()]}")
                if len(outliers) > 5:
                    print(f"  ... and {len(outliers) - 5} more")
        
        return outlier_info
    
    def split_data(self, features_df, target_series, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            features_df (pd.DataFrame): Features
            target_series (pd.Series): Target labels
            test_size (float): Proportion of test data
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*60)
        print("DATA SPLITTING (TRAIN/TEST)")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, target_series, 
            test_size=test_size, 
            random_state=random_state,
            stratify=target_series  # Ensure balanced split
        )
        
        print(f"Original dataset: {len(features_df)} samples")
        print(f"Training set: {len(X_train)} samples ({len(X_train)/len(features_df)*100:.1f}%)")
        print(f"Test set: {len(X_test)} samples ({len(X_test)/len(features_df)*100:.1f}%)")
        
        print("\nTraining set class distribution:")
        print(y_train.value_counts().sort_index())
        
        print("\nTest set class distribution:")
        print(y_test.value_counts().sort_index())
        
        return X_train, X_test, y_train, y_test
    
    def run_complete_preprocessing(self):
        """
        Run complete preprocessing pipeline
        
        Returns:
            dict: All preprocessing results
        """
        print("="*80)
        print("IRIS DATASET - COMPREHENSIVE PREPROCESSING AND EXPLORATION")
        print("="*80)
        
        # Load dataset
        features_df, target_series, feature_names, target_names = self.load_dataset()
        
        # Check missing values
        missing_info = self.check_missing_values(features_df)
        
        # Compute summary statistics
        self.compute_summary_statistics(features_df, target_series)
        
        # Create visualizations
        self.create_visualizations(features_df, target_series)
        
        # Identify outliers
        outlier_info = self.identify_outliers(features_df, target_series)
        
        # Normalize features
        normalized_features = self.normalize_features(features_df)
        
        # Encode target labels
        encoded_targets, label_mapping = self.encode_target_labels(target_series)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(normalized_features, encoded_targets)
        
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("Generated Files:")
        print("- Data/iris_data.csv (or synthetic_iris_data.csv)")
        print("- Visualizations/iris_exploration.png")
        print("- Visualizations/iris_pairplot.png")
        print("="*80)
        
        return {
            'original_features': features_df,
            'normalized_features': normalized_features,
            'original_targets': target_series,
            'encoded_targets': encoded_targets,
            'label_mapping': label_mapping,
            'train_test_split': (X_train, X_test, y_train, y_test),
            'missing_info': missing_info,
            'outlier_info': outlier_info,
            'feature_names': feature_names,
            'target_names': target_names
        }

def split_data_function(features_df, target_series, test_size=0.2, random_state=42):
    """
    Standalone function to split data into train/test sets
    
    Args:
        features_df (pd.DataFrame): Features
        target_series (pd.Series): Target labels
        test_size (float): Proportion of test data
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        features_df, target_series,
        test_size=test_size,
        random_state=random_state,
        stratify=target_series
    )

def main():
    """
    Main function to run preprocessing
    """
    # Option 1: Use real Iris dataset
    preprocessor_real = IrisDataPreprocessing(use_synthetic_data=False)
    results_real = preprocessor_real.run_complete_preprocessing()
    
    print("\n" + "#"*80)
    print("ALTERNATIVE: SYNTHETIC DATA GENERATION")
    print("#"*80)
    
    # Option 2: Use synthetic data
    preprocessor_synthetic = IrisDataPreprocessing(use_synthetic_data=True)
    results_synthetic = preprocessor_synthetic.run_complete_preprocessing()

if __name__ == "__main__":
    main()
