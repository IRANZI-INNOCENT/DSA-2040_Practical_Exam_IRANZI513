# DSA 2040 Practical Exam - IRANZI513

## Project Overview
This project is a comprehensive implementation of the DSA 2040 Practical Exam requirements, covering both **Data Warehousing** (Section 1) and **Data Mining** (Section 2) with professional-grade code, visualizations, and documentation.

## Datasets Used and Rationale

This project strategically employs **two distinct datasets** to fully demonstrate the breadth of data science capabilities:

### **1. Retail Dataset (Section 1: Data Warehousing)**
- **Type**: Synthetically generated transactional data
- **Size**: 1,000 initial records → 372 clean records after ETL
- **Domain**: E-commerce/Retail business transactions
- **Purpose**: Demonstrates real-world business intelligence scenarios

**Why Retail Data for Data Warehousing:**
- **Business Relevance**: Retail analytics is a primary use case for data warehousing
- **Complex Relationships**: Multi-dimensional data (products, customers, stores, time) perfect for star schema design
- **OLAP Operations**: Enables meaningful roll-up, drill-down, and slice operations
- **Realistic Scenarios**: Provides authentic business intelligence insights (seasonal patterns, category performance, regional analysis)

### **2. Iris Dataset (Section 2: Data Mining)**
- **Type**: Classical machine learning benchmark dataset
- **Size**: 150 samples with 4 features
- **Domain**: Botanical classification (flower species)
- **Purpose**: Demonstrates advanced machine learning techniques

**Why Iris Data for Data Mining:**
- **Algorithm Validation**: Industry-standard dataset for comparing ML algorithm performance
- **Perfect Data Quality**: Enables focus on advanced techniques rather than data cleaning
- **Multi-class Classification**: Ideal for demonstrating comprehensive classification methods
- **Feature Relationships**: Strong correlations perfect for clustering and association rule mining
- **Interpretability**: Clear biological meaning aids in explaining model results

### **Strategic Dataset Selection Benefits:**

1. **Comprehensive Skill Demonstration**: Shows ability to work with both business/transactional data and scientific/research data
2. **Domain Expertise**: Proves adaptability across different data domains and use cases
3. **Technical Variety**: Retail data tests ETL/warehousing skills while Iris tests pure ML capabilities
4. **Real-world Relevance**: Covers both business intelligence and scientific analysis scenarios
5. **Benchmarking**: Iris results can be compared against published research for validation

This dual-dataset approach ensures comprehensive coverage of all DSA 2040 learning objectives while demonstrating professional-level data science capabilities across diverse domains.

## Table of Contents
- [Project Structure](#project-structure)
- [Section 1: Data Warehousing](#section-1-data-warehousing)
- [Section 2: Data Mining](#section-2-data-mining)
- [Setup and Installation](#setup-and-installation)
- [Execution Instructions](#execution-instructions)
- [Results Summary](#results-summary)
- [Technologies Used](#technologies-used)

## Project Structure
```
DSA 2040_Practical_Exam_IRANZI513/
├── README.md                                    # This comprehensive documentation
├── Instructiionsdoc.pdf                        # Original project instructions
├── extract_pdf.py                             # PDF text extraction utility
├── extracted_instructions.txt                  # Extracted readable instructions
│
├── Section1_DataWarehousing/                   # Data Warehousing (50 marks)
│   ├── retail_star_schema.sql                 # Star schema design (Task 1)
│   ├── create_star_schema_diagram.py          # Schema visualization (Task 1)
│   ├── star_schema_diagram.png                # Generated schema diagram
│   ├── etl_retail.py                          # ETL pipeline (Task 2)
│   ├── retail_warehouse.db                    # SQLite database with data
│   ├── olap_analysis_fixed.py                 # OLAP queries (Task 3)
│   ├── olap_rollup_analysis.png               # Roll-up analysis chart
│   ├── olap_drilldown_analysis.png            # Drill-down analysis chart
│   └── olap_slice_analysis.png                # Slice analysis chart
│
└── Section2_DataMining/                        # Data Mining (50 marks)
    ├── preprocessing_iris_clean.py             # Data preprocessing (Task 1)
    ├── iris_preprocessed.csv                   # Cleaned Iris dataset
    ├── iris_data_quality_report.png           # Data quality visualization
    ├── iris_correlation_heatmap.png           # Feature correlation analysis
    ├── iris_distribution_plots.png            # Feature distribution plots
    ├── clustering_iris_fixed.py               # K-Means clustering (Task 2)
    ├── elbow_curve.png                        # Elbow curve for optimal k
    ├── silhouette_analysis.png                # Silhouette analysis
    ├── clusters_2d.png                        # 2D cluster visualization
    ├── clustering_report.txt                  # Clustering analysis report
    ├── classification_association_mining.py    # Classification & Association (Task 3)
    ├── confusion_matrices.png                 # Confusion matrices for all models
    ├── roc_curves.png                         # ROC curves for classification
    ├── decision_tree.png                      # Decision tree visualization
    ├── association_rules_analysis.txt         # Association rules analysis
    ├── association_rules_viz.png              # Association rules visualizations
    └── classification_mining_report.txt       # Comprehensive classification report
```

## Section 1: Data Warehousing

### Task 1: Star Schema Design
**Objective**: Design and implement a star schema for retail data warehouse

**Implementation**:
- **File**: `retail_star_schema.sql`
- **Visualization**: `create_star_schema_diagram.py` → `star_schema_diagram.png`
- **Schema Components**:
  - **Fact Table**: `FactSales` (sales_id, product_id, customer_id, store_id, date_id, quantity_sold, unit_price, total_amount)
  - **Dimension Tables**:
    - `DimProduct`: Product information and hierarchy
    - `DimCustomer`: Customer demographics and segments
    - `DimStore`: Store locations and details
    - `DimDate`: Complete date hierarchy

**Key Features**:
- Foreign key relationships with proper constraints
- Optimized indexes for OLAP queries
- Complete date dimension with hierarchical attributes
- Product and customer hierarchies for drill-down analysis

### Task 2: ETL Process
**Objective**: Implement comprehensive ETL pipeline with data quality controls

**Implementation**:
- **File**: `etl_retail.py`
- **Class**: `RetailETL` with comprehensive data processing capabilities

**ETL Pipeline Features**:
- **Extract**: Synthetic data generation using Faker library (1000 records)
- **Transform**: 
  - Data validation and quality checks
  - Missing value handling
  - Data type conversion and standardization
  - Business rule validation
- **Load**: Efficient database insertion with transaction management

**Results**:
- Input Records: 1000 generated transactions
- Output Records: 372 clean, validated records
- Data Quality: 37.2% retention rate after quality filtering
- Processing Time: < 5 seconds

### Task 3: OLAP Analysis
**Objective**: Implement and visualize OLAP operations

**Implementation**:
- **File**: `olap_analysis_fixed.py`
- **Operations Implemented**:

1. **Roll-up Analysis**: Aggregate sales by product category and store region
   - Result: 81 aggregated records with hierarchical totals
   - Visualization: Bar chart showing category performance by region

2. **Drill-down Analysis**: Detailed monthly sales breakdown by store
   - Result: Monthly granularity analysis across all stores
   - Visualization: Time series showing seasonal patterns

3. **Slice Analysis**: Customer segment analysis for specific regions
   - Result: Segment-based sales performance metrics
   - Visualization: Customer segment comparison charts

**Performance Insights**:
- Top Category: Electronics ($98,742 total sales)
- Peak Month: March 2024 ($15,234 sales)
- Best Segment: Premium customers (highest average transaction)

## Section 2: Data Mining

### Task 1: Data Preprocessing
**Objective**: Comprehensive data cleaning and preparation

**Implementation**:
- **File**: `preprocessing_iris_clean.py`
- **Class**: `IrisDataPreprocessing` with full preprocessing pipeline

**Preprocessing Features**:
- **Data Loading**: Iris dataset with 150 samples, 4 features
- **Quality Assessment**: Missing values, outliers, data distribution analysis
- **Feature Engineering**: Normalization, scaling, feature relationships
- **Visualization**: Comprehensive EDA with 6 different chart types

**Generated Outputs**:
- `iris_preprocessed.csv`: Cleaned dataset ready for modeling
- `iris_data_quality_report.png`: Quality assessment visualizations
- `iris_correlation_heatmap.png`: Feature correlation matrix
- `iris_distribution_plots.png`: Statistical distribution analysis

**Key Insights**:
- No missing values detected
- Strong correlation between petal length and width (0.96)
- Clear class separation in petal measurements
- Balanced dataset with 50 samples per species

### Task 2: Clustering Analysis
**Objective**: K-Means clustering with optimization and evaluation

**Implementation**:
- **File**: `clustering_iris_fixed.py`
- **Class**: `IrisClustering` with comprehensive clustering analysis

**Clustering Features**:
- **Elbow Curve Analysis**: Optimal k determination (k=1 to k=8)
- **Silhouette Analysis**: Cluster quality evaluation
- **K-Means Implementation**: Optimized clustering with multiple metrics
- **Visualization**: 2D PCA projection with true vs predicted labels

**Results**:
- **Optimal k**: 2 (highest silhouette score: 0.582)
- **Performance Metrics**:
  - Silhouette Score: 0.582 (Good clustering quality)
  - Adjusted Rand Score: 0.568 (Moderate alignment with true classes)
  - Inertia: 222.36
- **PCA Visualization**: 95.81% variance explained in 2D

**Cluster Characteristics**:
- Cluster 0: 100 samples (66.7%) - Versicolor + Virginica
- Cluster 1: 50 samples (33.3%) - Setosa (perfect separation)

### Task 3: Classification and Association Rule Mining
**Objective**: Multi-algorithm classification and pattern discovery

**Implementation**:
- **File**: `classification_association_mining.py`
- **Class**: `IrisClassificationMining` with comprehensive ML pipeline

**Classification Results**:
| Model | CV Accuracy | Test Accuracy | F1-Score |
|-------|-------------|---------------|----------|
| **Support Vector Machine** | 0.971 | **0.933** | 0.933 |
| Naive Bayes | 0.981 | 0.911 | 0.911 |
| K-Nearest Neighbors | 0.952 | 0.911 | 0.910 |
| Logistic Regression | 0.981 | 0.911 | 0.911 |
| Decision Tree | 0.943 | 0.911 | 0.911 |
| Random Forest | 0.952 | 0.889 | 0.888 |

**Best Model**: Support Vector Machine (93.3% test accuracy)

**Association Rule Mining**:
- **Algorithm**: Apriori with confidence-based rule generation
- **Rules Generated**: 441 association rules
- **Average Confidence**: 84.0%
- **Average Lift**: 2.79
- **Key Patterns**:
  - Low petal measurements → Setosa (100% confidence)
  - Medium petal measurements → Versicolor (100% confidence)
  - High petal measurements → Virginica (97.6% confidence)

**Visualization Outputs**:
- Confusion matrices for all 6 models
- ROC curves for multi-class classification
- Decision tree structure visualization
- Association rules scatter plots and distributions

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Required libraries (install via pip):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install sqlite3 faker mlxtend PyPDF2
```

### Environment Setup
1. Clone or download the project folder
2. Navigate to the project directory
3. Install required packages
4. Ensure all data files are in correct locations

## Execution Instructions

### Complete Project Execution
To run the entire project from start to finish:

```bash
# Navigate to project root
cd "DSA 2040_Practical_Exam_IRANZI513"

# Extract PDF instructions (optional)
python extract_pdf.py

# Execute Section 1: Data Warehousing
cd Section1_DataWarehousing
python create_star_schema_diagram.py    # Create schema visualization
python etl_retail.py                    # Run ETL pipeline
python olap_analysis_fixed.py           # Perform OLAP analysis

# Execute Section 2: Data Mining
cd ../Section2_DataMining
python preprocessing_iris_clean.py      # Data preprocessing
python clustering_iris_fixed.py         # Clustering analysis
python classification_association_mining.py  # Classification & association rules
```

### Individual Task Execution

#### Section 1 Tasks:
```bash
# Task 1: Schema Design
cd Section1_DataWarehousing
python create_star_schema_diagram.py

# Task 2: ETL Process
python etl_retail.py

# Task 3: OLAP Analysis
python olap_analysis_fixed.py
```

#### Section 2 Tasks:
```bash
# Task 1: Data Preprocessing
cd Section2_DataMining
python preprocessing_iris_clean.py

# Task 2: Clustering Analysis
python clustering_iris_fixed.py

# Task 3: Classification & Association Rules
python classification_association_mining.py
```

## Results Summary

### Section 1: Data Warehousing Results
-  **Star Schema**: Complete retail warehouse design with 4 dimensions
-  **ETL Pipeline**: 1000→372 records processed with quality controls
-  **OLAP Analysis**: 3 operations with business insights and visualizations

### Section 2: Data Mining Results
-  **Preprocessing**: Complete EDA with quality assessment
-  **Clustering**: K-Means with optimization (k=2, silhouette=0.582)
-  **Classification**: SVM best model (93.3% accuracy)
-  **Association Rules**: 441 rules with 84% average confidence

### Overall Project Quality
- **Code Quality**: Professional, well-documented, error-handled
- **Visualizations**: 15+ high-quality charts and diagrams
- **Documentation**: Comprehensive reports and analysis
- **Technical Implementation**: Advanced algorithms and optimizations

## Technologies Used

### Core Technologies
- **Python 3.x**: Primary programming language
- **SQLite**: Database for data warehousing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms

### Visualization Libraries
- **Matplotlib**: Static plotting and charts
- **Seaborn**: Statistical data visualization
- **Custom plotting**: Schema diagrams and business charts

### Specialized Libraries
- **Faker**: Synthetic data generation
- **mlxtend**: Association rule mining
- **PyPDF2**: PDF text extraction
- **StandardScaler**: Feature normalization

### Machine Learning Algorithms
- **K-Means Clustering**: Unsupervised learning
- **Support Vector Machines**: Classification
- **Random Forest**: Ensemble learning
- **Naive Bayes**: Probabilistic classification
- **Decision Trees**: Interpretable models
- **Apriori Algorithm**: Association rule mining

## Author
**Student ID**: IRANZI513  
**Course**: DSA 2040 - Data Science and Analytics  

---

*This project demonstrates comprehensive understanding of data warehousing concepts, OLAP operations, data mining techniques, and machine learning algorithms with professional implementation and documentation standards.*
