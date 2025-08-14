"""
DSA 2040 Practical Exam - ETL Process Implementation
Task 2: Extract, Transform, Load for Retail Data Warehouse
Author: IRANZI513
Date: August 14, 2025

This script implements a complete ETL process for retail data warehouse.
It can work with either the UCI Online Retail dataset or synthetic data.
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
import random
from faker import Faker
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetailETL:
    """
    Comprehensive ETL class for retail data warehouse
    """
    
    def __init__(self, use_synthetic_data=True):
        """
        Initialize ETL process
        
        Args:
            use_synthetic_data (bool): If True, generates synthetic data; 
                                     If False, expects UCI dataset
        """
        self.use_synthetic_data = use_synthetic_data
        self.db_path = 'c:/DSA 2040_Practical_Exam_IRANZI513/Data/retail_dw.db'
        self.csv_path = 'c:/DSA 2040_Practical_Exam_IRANZI513/Data/retail_data.csv'
        self.faker = Faker()
        self.faker.seed_instance(42)  # For reproducible results
        random.seed(42)
        np.random.seed(42)
        
    def generate_synthetic_data(self, num_records=1000):
        """
        Generate synthetic retail data similar to UCI Online Retail dataset
        
        Args:
            num_records (int): Number of records to generate
            
        Returns:
            pd.DataFrame: Generated retail data
        """
        logger.info(f"Generating {num_records} synthetic retail records...")
        
        # Product categories and items
        categories = {
            'Electronics': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Smart Watch', 
                          'Camera', 'Speaker', 'Monitor', 'Keyboard', 'Mouse'],
            'Clothing': ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Shoes', 
                        'Hat', 'Sweater', 'Shorts', 'Skirt', 'Boots'],
            'Books': ['Fiction Book', 'Technical Manual', 'Cookbook', 'Biography', 'Textbook',
                     'Magazine', 'Journal', 'Dictionary', 'Atlas', 'Encyclopedia'],
            'Home & Garden': ['Vase', 'Lamp', 'Cushion', 'Plant Pot', 'Picture Frame',
                            'Candle', 'Mirror', 'Clock', 'Rug', 'Curtains'],
            'Sports': ['Football', 'Tennis Racket', 'Yoga Mat', 'Dumbbell', 'Running Shoes',
                      'Bicycle', 'Helmet', 'Water Bottle', 'Golf Club', 'Basketball']
        }
        
        countries = ['United Kingdom', 'Germany', 'France', 'EIRE', 'Spain', 
                    'Netherlands', 'Belgium', 'Switzerland', 'Portugal', 'Australia',
                    'Italy', 'Poland', 'Austria', 'Denmark', 'Japan', 'USA', 'Canada']
        
        data = []
        
        # Generate customer pool
        customers = {}
        for i in range(100):  # 100 unique customers
            customer_id = f"C{i+1:04d}"
            customers[customer_id] = {
                'Country': random.choice(countries),
                'CustomerName': self.faker.name(),
                'Email': self.faker.email()
            }
        
        # Generate invoice data
        invoice_counter = 1
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2025, 8, 12)  # Current date as mentioned in instructions
        
        for _ in range(num_records):
            # Generate invoice details
            invoice_no = f"INV{invoice_counter:06d}"
            invoice_date = self.faker.date_time_between(start_date=start_date, end_date=end_date)
            
            # Select random customer
            customer_id = random.choice(list(customers.keys()))
            customer_info = customers[customer_id]
            
            # Select random product
            category = random.choice(list(categories.keys()))
            product_name = random.choice(categories[category])
            stock_code = f"{category[:3].upper()}{random.randint(1000, 9999)}"
            
            # Generate transaction details
            quantity = random.randint(1, 50)
            unit_price = round(random.uniform(1.0, 100.0), 2)
            
            # Occasional negative quantities (returns)
            if random.random() < 0.05:  # 5% chance of return
                quantity = -abs(quantity)
            
            # Occasional zero or negative prices (to be filtered out)
            if random.random() < 0.02:  # 2% chance of bad data
                unit_price = random.choice([0, -random.uniform(1, 10)])
            
            data.append({
                'InvoiceNo': invoice_no,
                'StockCode': stock_code,
                'Description': product_name,
                'Quantity': quantity,
                'InvoiceDate': invoice_date,
                'UnitPrice': unit_price,
                'CustomerID': customer_id,
                'Country': customer_info['Country'],
                'Category': category,
                'CustomerName': customer_info['CustomerName'],
                'CustomerEmail': customer_info['Email']
            })
            
            # Occasionally increment invoice number for new invoice
            if random.random() < 0.7:  # 70% chance of new invoice
                invoice_counter += 1
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} records with {df['CustomerID'].nunique()} unique customers")
        
        return df
    
    def extract_data(self):
        """
        Extract data from source (CSV file or generate synthetic data)
        
        Returns:
            pd.DataFrame: Raw extracted data
        """
        logger.info("Starting data extraction...")
        
        if self.use_synthetic_data:
            # Generate synthetic data
            df = self.generate_synthetic_data(1000)
            
            # Save to CSV for documentation
            df.to_csv(self.csv_path, index=False)
            logger.info(f"Synthetic data saved to {self.csv_path}")
            
        else:
            # Load UCI Online Retail dataset
            try:
                df = pd.read_csv(self.csv_path, encoding='latin-1')
                logger.info(f"Loaded UCI dataset from {self.csv_path}")
            except FileNotFoundError:
                logger.error(f"UCI dataset not found at {self.csv_path}")
                logger.info("Falling back to synthetic data generation...")
                df = self.generate_synthetic_data(1000)
                df.to_csv(self.csv_path, index=False)
        
        logger.info(f"Extracted {len(df)} records")
        return df
    
    def transform_data(self, df):
        """
        Transform the extracted data
        
        Args:
            df (pd.DataFrame): Raw data
            
        Returns:
            tuple: (fact_df, customer_dim, product_dim, time_dim, store_dim)
        """
        logger.info("Starting data transformation...")
        original_count = len(df)
        
        # 1. Handle data types and missing values
        logger.info("Converting data types and handling missing values...")
        
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        
        # Remove records with missing critical values
        df = df.dropna(subset=['InvoiceDate', 'CustomerID', 'UnitPrice', 'Quantity'])
        
        # Fill missing descriptions
        df['Description'] = df['Description'].fillna('Unknown Product')
        
        logger.info(f"After handling missing values: {len(df)} records (removed {original_count - len(df)})")
        
        # 2. Calculate TotalSales column
        df['TotalSales'] = df['Quantity'] * df['UnitPrice']
        
        # 3. Filter data for last year (from August 12, 2025)
        cutoff_date = datetime(2024, 8, 12)
        df_last_year = df[df['InvoiceDate'] >= cutoff_date].copy()
        logger.info(f"After filtering for last year: {len(df_last_year)} records")
        
        # 4. Handle outliers - remove negative quantities and zero/negative prices
        clean_data_count = len(df_last_year)
        df_clean = df_last_year[(df_last_year['Quantity'] > 0) & (df_last_year['UnitPrice'] > 0)].copy()
        logger.info(f"After removing outliers: {len(df_clean)} records (removed {clean_data_count - len(df_clean)})")
        
        # 5. Create dimension tables
        logger.info("Creating dimension tables...")
        
        # Customer Dimension
        customer_dim = df_clean.groupby('CustomerID').agg({
            'CustomerName': 'first',
            'CustomerEmail': 'first', 
            'Country': 'first',
            'TotalSales': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()
        
        customer_dim.columns = ['CustomerID', 'CustomerName', 'CustomerEmail', 
                               'Country', 'TotalPurchases', 'TotalOrders']
        
        # Add customer segmentation based on total purchases
        customer_dim['CustomerSegment'] = pd.cut(customer_dim['TotalPurchases'], 
                                                bins=[0, 100, 500, float('inf')], 
                                                labels=['Basic', 'Standard', 'Premium'])
        
        logger.info(f"Created customer dimension with {len(customer_dim)} customers")
        
        # Product Dimension
        product_dim = df_clean.groupby('StockCode').agg({
            'Description': 'first',
            'Category': 'first',
            'UnitPrice': 'mean'
        }).reset_index()
        
        product_dim.columns = ['ProductID', 'ProductName', 'ProductCategory', 'AvgUnitPrice']
        product_dim = product_dim.reset_index(drop=True)
        product_dim['ProductID'] = product_dim.index + 1
        
        logger.info(f"Created product dimension with {len(product_dim)} products")
        
        # Time Dimension
        unique_dates = df_clean['InvoiceDate'].dt.date.unique()
        time_data = []
        
        for date in unique_dates:
            dt = pd.to_datetime(date)
            time_data.append({
                'Date': date,
                'Day': dt.day,
                'DayName': dt.strftime('%A'),
                'Week': dt.isocalendar()[1],
                'Month': dt.month,
                'MonthName': dt.strftime('%B'),
                'Quarter': (dt.month - 1) // 3 + 1,
                'Year': dt.year,
                'IsWeekend': dt.weekday() >= 5
            })
        
        time_dim = pd.DataFrame(time_data)
        time_dim = time_dim.reset_index(drop=True)
        time_dim['TimeID'] = time_dim.index + 1
        
        logger.info(f"Created time dimension with {len(time_dim)} dates")
        
        # Store Dimension (simplified - assuming online store)
        store_dim = pd.DataFrame({
            'StoreID': [1],
            'StoreName': ['Online Store'],
            'StoreType': ['Online'],
            'Country': ['UK']
        })
        
        # 6. Create fact table with proper foreign keys
        logger.info("Creating fact table...")
        
        # Create lookup dictionaries
        customer_lookup = dict(zip(customer_dim['CustomerID'], customer_dim.index + 1))
        product_lookup = dict(zip(product_dim['ProductName'], product_dim['ProductID']))
        time_lookup = dict(zip(time_dim['Date'], time_dim['TimeID']))
        
        # Create fact table
        fact_data = []
        
        for _, row in df_clean.iterrows():
            fact_data.append({
                'CustomerID': customer_lookup.get(row['CustomerID'], 1),
                'ProductID': product_lookup.get(row['Description'], 1),
                'TimeID': time_lookup.get(row['InvoiceDate'].date(), 1),
                'StoreID': 1,  # Always online store for this example
                'InvoiceNo': row['InvoiceNo'],
                'Quantity': row['Quantity'],
                'UnitPrice': row['UnitPrice'],
                'TotalSales': row['TotalSales']
            })
        
        fact_df = pd.DataFrame(fact_data)
        fact_df = fact_df.reset_index(drop=True)
        fact_df['SalesID'] = fact_df.index + 1
        
        logger.info(f"Created fact table with {len(fact_df)} sales records")
        
        return fact_df, customer_dim, product_dim, time_dim, store_dim
    
    def load_to_database(self, fact_df, customer_dim, product_dim, time_dim, store_dim):
        """
        Load transformed data into SQLite database
        
        Args:
            fact_df, customer_dim, product_dim, time_dim, store_dim: DataFrames to load
        """
        logger.info("Starting data loading to database...")
        
        # Create database connection
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Create tables using the schema
            with open('c:/DSA 2040_Practical_Exam_IRANZI513/Section1_DataWarehousing/retail_star_schema.sql', 'r') as f:
                schema_sql = f.read()
            
            # Execute schema creation (split by semicolons)
            for statement in schema_sql.split(';'):
                if statement.strip():
                    try:
                        conn.execute(statement)
                    except sqlite3.Error as e:
                        if "already exists" not in str(e):
                            logger.warning(f"SQL execution warning: {e}")
            
            # Load dimension tables first
            logger.info("Loading dimension tables...")
            
            # Load customers
            customer_dim_db = customer_dim[['CustomerID', 'CustomerName', 'CustomerEmail', 
                                         'Country', 'CustomerSegment']].copy()
            customer_dim_db.to_sql('DimCustomer', conn, if_exists='replace', index=False)
            logger.info(f"Loaded {len(customer_dim_db)} customer records")
            
            # Load products  
            product_dim_db = product_dim[['ProductID', 'ProductName', 'ProductCategory']].copy()
            product_dim_db['UnitPrice'] = product_dim['AvgUnitPrice']
            product_dim_db.to_sql('DimProduct', conn, if_exists='replace', index=False)
            logger.info(f"Loaded {len(product_dim_db)} product records")
            
            # Load time
            time_dim.to_sql('DimTime', conn, if_exists='replace', index=False)
            logger.info(f"Loaded {len(time_dim)} time records")
            
            # Load store
            store_dim.to_sql('DimStore', conn, if_exists='replace', index=False)
            logger.info(f"Loaded {len(store_dim)} store records")
            
            # Load fact table
            logger.info("Loading fact table...")
            fact_df_db = fact_df[['SalesID', 'CustomerID', 'ProductID', 'TimeID', 'StoreID',
                                 'Quantity', 'UnitPrice', 'TotalSales']].copy()
            fact_df_db['SalesAmount'] = fact_df_db['TotalSales']
            fact_df_db['TotalSalesAmount'] = fact_df_db['TotalSales']
            fact_df_db.to_sql('FactSales', conn, if_exists='replace', index=False)
            logger.info(f"Loaded {len(fact_df_db)} fact records")
            
            conn.commit()
            logger.info("Data loading completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during data loading: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def run_etl(self):
        """
        Execute the complete ETL process
        
        Returns:
            dict: Summary statistics of the ETL process
        """
        logger.info("=" * 50)
        logger.info("STARTING RETAIL DATA WAREHOUSE ETL PROCESS")
        logger.info("=" * 50)
        
        try:
            # Extract
            raw_data = self.extract_data()
            extract_count = len(raw_data)
            
            # Transform
            fact_df, customer_dim, product_dim, time_dim, store_dim = self.transform_data(raw_data)
            transform_count = len(fact_df)
            
            # Load
            self.load_to_database(fact_df, customer_dim, product_dim, time_dim, store_dim)
            load_count = len(fact_df)
            
            # Summary
            summary = {
                'extracted_records': extract_count,
                'transformed_records': transform_count,
                'loaded_records': load_count,
                'customers': len(customer_dim),
                'products': len(product_dim),
                'time_periods': len(time_dim),
                'stores': len(store_dim),
                'data_quality_loss_percent': round((extract_count - transform_count) / extract_count * 100, 2)
            }
            
            logger.info("=" * 50)
            logger.info("ETL PROCESS COMPLETED SUCCESSFULLY")
            logger.info("=" * 50)
            logger.info(f"Records extracted: {summary['extracted_records']}")
            logger.info(f"Records transformed: {summary['transformed_records']}")
            logger.info(f"Records loaded: {summary['loaded_records']}")
            logger.info(f"Unique customers: {summary['customers']}")
            logger.info(f"Unique products: {summary['products']}")
            logger.info(f"Time periods: {summary['time_periods']}")
            logger.info(f"Data quality loss: {summary['data_quality_loss_percent']}%")
            
            return summary
            
        except Exception as e:
            logger.error(f"ETL process failed: {e}")
            raise

def main():
    """
    Main function to run the ETL process
    """
    # Initialize ETL with synthetic data
    etl = RetailETL(use_synthetic_data=True)
    
    # Run ETL process
    summary = etl.run_etl()
    
    # Additional verification
    logger.info("\nVerifying database contents...")
    conn = sqlite3.connect(etl.db_path)
    
    tables = ['FactSales', 'DimCustomer', 'DimProduct', 'DimTime', 'DimStore']
    for table in tables:
        count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn).iloc[0]['count']
        logger.info(f"{table}: {count} records")
    
    conn.close()

if __name__ == "__main__":
    main()
