"""
DSA 2040 Practical Exam - Section 2: Data Mining
Task 3: Classification and Association Rule Mining on Iris Dataset
Author: IRANZI513
Date: 2024

This module implements comprehensive classification analysis and association rule mining
on the Iris dataset using multiple algorithms and evaluation techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           precision_score, recall_score, f1_score, roc_curve, auc,
                           roc_auc_score)
from sklearn.preprocessing import label_binarize
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

class IrisClassificationMining:
    """
    Comprehensive classification and association rule mining for Iris dataset
    """
    
    def __init__(self):
        """Initialize the classification and mining analyzer"""
        self.scaler = StandardScaler()
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.best_score = 0
        
    def load_and_prepare_data(self):
        """
        Load and prepare Iris dataset for classification
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names, target_names)
        """
        try:
            # Load Iris dataset
            iris = load_iris()
            X = pd.DataFrame(iris.data, columns=iris.feature_names)
            y = iris.target
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Scale the features
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            print(f"Dataset prepared:")
            print(f"  Training set: {X_train_scaled.shape[0]} samples")
            print(f"  Test set: {X_test_scaled.shape[0]} samples")
            print(f"  Features: {list(iris.feature_names)}")
            print(f"  Classes: {list(iris.target_names)}")
            print(f"  Class distribution in training: {np.bincount(y_train)}")
            print(f"  Class distribution in test: {np.bincount(y_test)}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test, iris.feature_names, iris.target_names
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None, None, None, None
    
    def train_multiple_classifiers(self, X_train, y_train):
        """
        Train multiple classification algorithms
        
        Args:
            X_train (DataFrame): Training features
            y_train (array): Training labels
        """
        # Define classifiers
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Support Vector Machine': SVC(random_state=42, probability=True),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5)
        }
        
        print("Training multiple classifiers...")
        
        for name, classifier in classifiers.items():
            try:
                # Train the model
                classifier.fit(X_train, y_train)
                self.models[name] = classifier
                
                # Perform cross-validation
                cv_scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')
                
                self.model_results[name] = {
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"  {name}: CV Accuracy = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
    
    def evaluate_models(self, X_test, y_test, target_names):
        """
        Evaluate all trained models on test data
        
        Args:
            X_test (DataFrame): Test features
            y_test (array): Test labels
            target_names (list): Target class names
        """
        print("\\nEvaluating models on test data...")
        
        for name, model in self.models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Update results
                self.model_results[name].update({
                    'test_accuracy': accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                })
                
                # Track best model
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = name
                
                print(f"  {name}:")
                print(f"    Accuracy: {accuracy:.3f}")
                print(f"    Precision: {precision:.3f}")
                print(f"    Recall: {recall:.3f}")
                print(f"    F1-Score: {f1:.3f}")
                
            except Exception as e:
                print(f"  Error evaluating {name}: {e}")
        
        print(f"\\nBest performing model: {self.best_model} (Accuracy: {self.best_score:.3f})")
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning on selected models
        
        Args:
            X_train (DataFrame): Training features
            y_train (array): Training labels
        """
        print("\\nPerforming hyperparameter tuning...")
        
        # Define parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, None],
                'min_samples_split': [2, 5]
            },
            'Support Vector Machine': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            },
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        }
        
        tuned_models = {}
        
        for model_name, param_grid in param_grids.items():
            if model_name in self.models:
                try:
                    print(f"  Tuning {model_name}...")
                    
                    # Get base model
                    base_model = self.models[model_name]
                    
                    # Perform grid search
                    grid_search = GridSearchCV(
                        base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    
                    # Store tuned model
                    tuned_models[model_name] = grid_search.best_estimator_
                    
                    print(f"    Best parameters: {grid_search.best_params_}")
                    print(f"    Best CV score: {grid_search.best_score_:.3f}")
                    
                except Exception as e:
                    print(f"    Error tuning {model_name}: {e}")
        
        # Update models with tuned versions
        self.models.update(tuned_models)
        print("Hyperparameter tuning completed.")
    
    def create_confusion_matrices(self, y_test, target_names, save_path='confusion_matrices.png'):
        """
        Create confusion matrices for all models
        
        Args:
            y_test (array): True test labels
            target_names (list): Target class names
            save_path (str): Path to save the plot
        """
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, (name, results) in enumerate(self.model_results.items()):
            if i < len(axes) and 'predictions' in results:
                cm = confusion_matrix(y_test, results['predictions'])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=target_names, yticklabels=target_names, ax=axes[i])
                axes[i].set_title(f'{name}\\nAccuracy: {results["test_accuracy"]:.3f}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(len(self.models), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrices saved to {save_path}")
    
    def create_roc_curves(self, y_test, target_names, save_path='roc_curves.png'):
        """
        Create ROC curves for multi-class classification
        
        Args:
            y_test (array): True test labels
            target_names (list): Target class names
            save_path (str): Path to save the plot
        """
        # Binarize the labels for multi-class ROC
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        n_classes = y_test_bin.shape[1]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for class_idx in range(n_classes):
            for name, results in self.model_results.items():
                if 'probabilities' in results and results['probabilities'] is not None:
                    try:
                        # Get probabilities for this class
                        y_scores = results['probabilities'][:, class_idx]
                        
                        # Calculate ROC curve
                        fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_scores)
                        roc_auc = auc(fpr, tpr)
                        
                        # Plot ROC curve
                        axes[class_idx].plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
                        
                    except Exception as e:
                        print(f"Error plotting ROC for {name}, class {class_idx}: {e}")
            
            # Plot diagonal line
            axes[class_idx].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[class_idx].set_xlim([0.0, 1.0])
            axes[class_idx].set_ylim([0.0, 1.05])
            axes[class_idx].set_xlabel('False Positive Rate')
            axes[class_idx].set_ylabel('True Positive Rate')
            axes[class_idx].set_title(f'ROC Curve - {target_names[class_idx]}')
            axes[class_idx].legend(loc="lower right")
            axes[class_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"ROC curves saved to {save_path}")
    
    def visualize_decision_tree(self, feature_names, target_names, save_path='decision_tree.png'):
        """
        Visualize the decision tree
        
        Args:
            feature_names (list): Feature names
            target_names (list): Target class names
            save_path (str): Path to save the plot
        """
        if 'Decision Tree' in self.models:
            plt.figure(figsize=(20, 10))
            plot_tree(self.models['Decision Tree'], 
                     feature_names=feature_names,
                     class_names=target_names,
                     filled=True, rounded=True, fontsize=10)
            plt.title('Decision Tree Visualization')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Decision tree visualization saved to {save_path}")
    
    def prepare_data_for_association_rules(self, X, y, target_names):
        """
        Prepare data for association rule mining by discretizing continuous features
        
        Args:
            X (DataFrame): Feature data
            y (array): Target labels
            target_names (list): Target class names
            
        Returns:
            DataFrame: Prepared data for association rule mining
        """
        print("\\nPreparing data for association rule mining...")
        
        # Create a copy of the data
        data = X.copy()
        data['species'] = [target_names[label] for label in y]
        
        # Discretize continuous features into categorical bins
        discretized_data = pd.DataFrame()
        
        for column in X.columns:
            # Create bins based on quartiles
            bins = pd.qcut(data[column], q=3, labels=[f'{column}_Low', f'{column}_Medium', f'{column}_High'])
            discretized_data[column] = bins
        
        # Add species
        discretized_data['species'] = data['species']
        
        print(f"Data discretized into categorical features:")
        for column in discretized_data.columns:
            print(f"  {column}: {discretized_data[column].unique()}")
        
        return discretized_data
    
    def perform_association_rule_mining(self, discretized_data, min_support=0.1, min_confidence=0.6):
        """
        Perform association rule mining on the discretized data
        
        Args:
            discretized_data (DataFrame): Discretized feature data
            min_support (float): Minimum support threshold
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            tuple: (frequent_itemsets, association_rules)
        """
        print(f"\\nPerforming association rule mining...")
        print(f"  Minimum support: {min_support}")
        print(f"  Minimum confidence: {min_confidence}")
        
        try:
            # Create one-hot encoded data for association rule mining
            one_hot_data = pd.get_dummies(discretized_data)
            
            # Find frequent itemsets
            frequent_itemsets = apriori(one_hot_data, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) == 0:
                print("  No frequent itemsets found. Try lowering min_support.")
                return None, None
            
            print(f"  Found {len(frequent_itemsets)} frequent itemsets")
            
            # Generate association rules
            rules = association_rules(frequent_itemsets, 
                                    metric="confidence", 
                                    min_threshold=min_confidence)
            
            if len(rules) == 0:
                print("  No association rules found. Try lowering min_confidence.")
                return frequent_itemsets, None
            
            print(f"  Found {len(rules)} association rules")
            
            return frequent_itemsets, rules
            
        except Exception as e:
            print(f"  Error in association rule mining: {e}")
            return None, None
    
    def analyze_association_rules(self, rules, save_path='association_rules_analysis.txt'):
        """
        Analyze and display association rules
        
        Args:
            rules (DataFrame): Association rules
            save_path (str): Path to save the analysis
        """
        if rules is None or len(rules) == 0:
            print("No association rules to analyze.")
            return
        
        print("\\nAnalyzing association rules...")
        
        analysis = []
        analysis.append("ASSOCIATION RULE MINING ANALYSIS")
        analysis.append("=" * 50)
        analysis.append("")
        
        # Sort rules by confidence
        rules_sorted = rules.sort_values('confidence', ascending=False)
        
        analysis.append(f"Total number of rules: {len(rules)}")
        analysis.append(f"Average support: {rules['support'].mean():.3f}")
        analysis.append(f"Average confidence: {rules['confidence'].mean():.3f}")
        analysis.append(f"Average lift: {rules['lift'].mean():.3f}")
        analysis.append("")
        
        analysis.append("TOP 10 RULES BY CONFIDENCE:")
        analysis.append("-" * 30)
        
        for i, (idx, rule) in enumerate(rules_sorted.head(10).iterrows()):
            antecedent = ', '.join(list(rule['antecedents']))
            consequent = ', '.join(list(rule['consequents']))
            
            analysis.append(f"Rule {i+1}:")
            analysis.append(f"  {antecedent} => {consequent}")
            analysis.append(f"  Support: {rule['support']:.3f}")
            analysis.append(f"  Confidence: {rule['confidence']:.3f}")
            analysis.append(f"  Lift: {rule['lift']:.3f}")
            analysis.append("")
        
        # Rules involving species
        species_rules = rules[rules['consequents'].astype(str).str.contains('species')]
        if len(species_rules) > 0:
            analysis.append("RULES PREDICTING SPECIES:")
            analysis.append("-" * 25)
            
            for i, (idx, rule) in enumerate(species_rules.sort_values('confidence', ascending=False).iterrows()):
                antecedent = ', '.join(list(rule['antecedents']))
                consequent = ', '.join(list(rule['consequents']))
                
                analysis.append(f"  {antecedent} => {consequent}")
                analysis.append(f"    Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}")
            analysis.append("")
        
        # Save analysis
        analysis_text = '\\n'.join(analysis)
        with open(save_path, 'w') as f:
            f.write(analysis_text)
        
        print(f"Association rules analysis saved to {save_path}")
        print(analysis_text)
    
    def visualize_association_rules(self, rules, save_path='association_rules_viz.png'):
        """
        Create visualizations for association rules
        
        Args:
            rules (DataFrame): Association rules
            save_path (str): Path to save the plot
        """
        if rules is None or len(rules) == 0:
            print("No association rules to visualize.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Support vs Confidence scatter plot
        axes[0,0].scatter(rules['support'], rules['confidence'], alpha=0.6, c=rules['lift'], cmap='viridis')
        axes[0,0].set_xlabel('Support')
        axes[0,0].set_ylabel('Confidence')
        axes[0,0].set_title('Support vs Confidence (colored by Lift)')
        
        # Support histogram
        axes[0,1].hist(rules['support'], bins=20, alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel('Support')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Distribution of Support Values')
        
        # Confidence histogram
        axes[1,0].hist(rules['confidence'], bins=20, alpha=0.7, edgecolor='black')
        axes[1,0].set_xlabel('Confidence')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Distribution of Confidence Values')
        
        # Lift histogram
        axes[1,1].hist(rules['lift'], bins=20, alpha=0.7, edgecolor='black')
        axes[1,1].set_xlabel('Lift')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Distribution of Lift Values')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Association rules visualization saved to {save_path}")
    
    def generate_comprehensive_report(self, target_names, save_path='classification_mining_report.txt'):
        """
        Generate comprehensive classification and mining report
        
        Args:
            target_names (list): Target class names
            save_path (str): Path to save the report
        """
        report = []
        report.append("=" * 70)
        report.append("IRIS DATASET CLASSIFICATION AND ASSOCIATION RULE MINING REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Classification results summary
        report.append("CLASSIFICATION RESULTS SUMMARY:")
        report.append("-" * 35)
        
        # Sort models by test accuracy
        sorted_models = sorted(self.model_results.items(), 
                             key=lambda x: x[1].get('test_accuracy', 0), reverse=True)
        
        report.append(f"{'Model':<25} {'CV Accuracy':<12} {'Test Accuracy':<13} {'F1-Score':<10}")
        report.append("-" * 65)
        
        for name, results in sorted_models:
            cv_acc = results.get('cv_mean', 0)
            test_acc = results.get('test_accuracy', 0)
            f1 = results.get('test_f1', 0)
            report.append(f"{name:<25} {cv_acc:<12.3f} {test_acc:<13.3f} {f1:<10.3f}")
        
        report.append("")
        report.append(f"Best performing model: {self.best_model}")
        report.append(f"Best test accuracy: {self.best_score:.3f}")
        report.append("")
        
        # Model comparison insights
        report.append("MODEL PERFORMANCE INSIGHTS:")
        report.append("-" * 27)
        
        if len(self.model_results) > 0:
            # Find models with highest metrics
            best_cv = max(self.model_results.items(), key=lambda x: x[1].get('cv_mean', 0))
            best_precision = max(self.model_results.items(), key=lambda x: x[1].get('test_precision', 0))
            best_recall = max(self.model_results.items(), key=lambda x: x[1].get('test_recall', 0))
            
            report.append(f"  Highest CV accuracy: {best_cv[0]} ({best_cv[1].get('cv_mean', 0):.3f})")
            report.append(f"  Highest precision: {best_precision[0]} ({best_precision[1].get('test_precision', 0):.3f})")
            report.append(f"  Highest recall: {best_recall[0]} ({best_recall[1].get('test_recall', 0):.3f})")
        
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 15)
        
        if self.best_score > 0.95:
            report.append("  - Excellent classification performance achieved")
            report.append("  - All models perform well on this dataset")
        elif self.best_score > 0.9:
            report.append("  - Very good classification performance")
            report.append(f"  - {self.best_model} is recommended for deployment")
        else:
            report.append("  - Consider feature engineering or different algorithms")
            report.append("  - Collect more training data if possible")
        
        report.append("")
        report.append("  - The Iris dataset is well-suited for classification")
        report.append("  - Association rules can provide additional insights")
        report.append("  - Consider ensemble methods for improved robustness")
        
        report.append("")
        report.append("=" * 70)
        
        # Save report
        report_text = '\\n'.join(report)
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"Comprehensive report saved to {save_path}")
        print(report_text)
    
    def run_complete_analysis(self):
        """
        Run complete classification and association rule mining analysis
        """
        print("Starting Complete Classification and Association Rule Mining Analysis")
        print("=" * 70)
        
        # Load and prepare data
        X_train, X_test, y_train, y_test, feature_names, target_names = self.load_and_prepare_data()
        if X_train is None:
            print("Failed to load data. Exiting analysis.")
            return
        
        # Train multiple classifiers
        print("\\n1. Training Multiple Classification Models...")
        self.train_multiple_classifiers(X_train, y_train)
        
        # Hyperparameter tuning
        print("\\n2. Hyperparameter Tuning...")
        self.hyperparameter_tuning(X_train, y_train)
        
        # Re-train with tuned parameters and evaluate
        print("\\n3. Re-training with Tuned Parameters...")
        self.train_multiple_classifiers(X_train, y_train)
        self.evaluate_models(X_test, y_test, target_names)
        
        # Create visualizations
        print("\\n4. Creating Classification Visualizations...")
        self.create_confusion_matrices(y_test, target_names)
        self.create_roc_curves(y_test, target_names)
        self.visualize_decision_tree(feature_names, target_names)
        
        # Association rule mining
        print("\\n5. Performing Association Rule Mining...")
        
        # Combine train and test data for association rule mining
        X_full = pd.concat([X_train, X_test], ignore_index=True)
        y_full = np.concatenate([y_train, y_test])
        
        discretized_data = self.prepare_data_for_association_rules(X_full, y_full, target_names)
        frequent_itemsets, rules = self.perform_association_rule_mining(discretized_data)
        
        if rules is not None:
            self.analyze_association_rules(rules)
            self.visualize_association_rules(rules)
        
        # Generate comprehensive report
        print("\\n6. Generating Comprehensive Report...")
        self.generate_comprehensive_report(target_names)
        
        print("\\nComplete classification and association rule mining analysis finished!")
        print("Generated files:")
        print("  - confusion_matrices.png")
        print("  - roc_curves.png")
        print("  - decision_tree.png")
        if rules is not None:
            print("  - association_rules_analysis.txt")
            print("  - association_rules_viz.png")
        print("  - classification_mining_report.txt")


def main():
    """Main function to run the classification and association rule mining analysis"""
    try:
        # Create analyzer
        analyzer = IrisClassificationMining()
        
        # Run complete analysis
        analyzer.run_complete_analysis()
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
