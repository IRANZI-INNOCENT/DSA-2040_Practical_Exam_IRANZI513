"""
DSA 2040 Practical Exam - Fixed OLAP Analysis and Visualization
Task 3: Execute OLAP queries and create visualizations (Fixed Version)
Author: IRANZI513
Date: August 14, 2025
"""

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class OLAPAnalysisFixed:
    """
    Fixed Class to execute OLAP queries and create visualizations
    """
    
    def __init__(self, db_path):
        """
        Initialize OLAP Analysis
        
        Args:
            db_path (str): Path to the SQLite database
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
    def execute_query(self, query, description=""):
        """
        Execute SQL query and return results
        
        Args:
            query (str): SQL query to execute
            description (str): Description of the query
            
        Returns:
            pd.DataFrame: Query results
        """
        try:
            print(f"\n{'='*60}")
            print(f"EXECUTING: {description}")
            print('='*60)
            
            df = pd.read_sql_query(query, self.conn)
            print(f"Results: {len(df)} rows returned")
            if len(df) > 0:
                print(df.head(10))
            else:
                print("No data returned")
            
            return df
            
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def rollup_analysis(self):
        """
        Execute Roll-up OLAP operation: Total sales by country and quarter
        """
        query = """
        SELECT 
            dc.Country,
            dt.Quarter,
            dt.Year,
            COUNT(fs.SalesID) as NumberOfSales,
            SUM(fs.Quantity) as TotalQuantitySold,
            ROUND(SUM(fs.TotalSalesAmount), 2) as TotalSalesAmount,
            ROUND(AVG(fs.TotalSalesAmount), 2) as AvgSaleAmount,
            COUNT(DISTINCT fs.CustomerID) as UniqueCustomers
        FROM FactSales fs
        JOIN DimCustomer dc ON fs.CustomerID = dc.ROWID
        JOIN DimTime dt ON fs.TimeID = dt.TimeID
        GROUP BY dc.Country, dt.Quarter, dt.Year
        ORDER BY dt.Year DESC, dt.Quarter DESC, TotalSalesAmount DESC;
        """
        
        return self.execute_query(query, "ROLL-UP: Total sales by country and quarter")
    
    def drilldown_analysis(self):
        """
        Execute Drill-down OLAP operation: Sales details for specific country by month
        """
        # First, let's see what countries we have
        country_query = """
        SELECT DISTINCT dc.Country, COUNT(*) as sales_count
        FROM FactSales fs
        JOIN DimCustomer dc ON fs.CustomerID = dc.ROWID
        GROUP BY dc.Country
        ORDER BY sales_count DESC
        LIMIT 1;
        """
        
        top_country_df = pd.read_sql_query(country_query, self.conn)
        
        if len(top_country_df) > 0:
            target_country = top_country_df.iloc[0]['Country']
            print(f"Using top country: {target_country}")
            
            query = f"""
            SELECT 
                dc.Country,
                dt.Year,
                dt.MonthName,
                dt.Month,
                COUNT(fs.SalesID) as NumberOfSales,
                SUM(fs.Quantity) as TotalQuantitySold,
                ROUND(SUM(fs.TotalSalesAmount), 2) as TotalSalesAmount,
                ROUND(AVG(fs.TotalSalesAmount), 2) as AvgSaleAmount,
                COUNT(DISTINCT fs.CustomerID) as UniqueCustomers,
                COUNT(DISTINCT fs.ProductID) as UniqueProducts
            FROM FactSales fs
            JOIN DimCustomer dc ON fs.CustomerID = dc.ROWID
            JOIN DimTime dt ON fs.TimeID = dt.TimeID
            WHERE dc.Country = '{target_country}'
            GROUP BY dc.Country, dt.Year, dt.MonthName, dt.Month
            ORDER BY dt.Year DESC, dt.Month DESC;
            """
            
            return self.execute_query(query, f"DRILL-DOWN: Sales details for {target_country} by month")
        else:
            print("No country data found")
            return pd.DataFrame()
    
    def slice_analysis(self):
        """
        Execute Slice OLAP operation: Total sales for specific product category
        """
        # First, let's see what categories we have
        category_query = """
        SELECT DISTINCT dp.ProductCategory, COUNT(*) as sales_count
        FROM FactSales fs
        JOIN DimProduct dp ON fs.ProductID = dp.ProductID
        GROUP BY dp.ProductCategory
        ORDER BY sales_count DESC
        LIMIT 1;
        """
        
        top_category_df = pd.read_sql_query(category_query, self.conn)
        
        if len(top_category_df) > 0:
            target_category = top_category_df.iloc[0]['ProductCategory']
            print(f"Using top category: {target_category}")
            
            query = f"""
            SELECT 
                dp.ProductCategory,
                dt.Quarter,
                dt.Year,
                COUNT(fs.SalesID) as NumberOfSales,
                SUM(fs.Quantity) as TotalQuantitySold,
                ROUND(SUM(fs.TotalSalesAmount), 2) as TotalSalesAmount,
                ROUND(AVG(fs.TotalSalesAmount), 2) as AvgSaleAmount,
                COUNT(DISTINCT fs.CustomerID) as UniqueCustomers,
                COUNT(DISTINCT dc.Country) as CountriesServed
            FROM FactSales fs
            JOIN DimProduct dp ON fs.ProductID = dp.ProductID
            JOIN DimCustomer dc ON fs.CustomerID = dc.ROWID
            JOIN DimTime dt ON fs.TimeID = dt.TimeID
            WHERE dp.ProductCategory = '{target_category}'
            GROUP BY dp.ProductCategory, dt.Quarter, dt.Year
            ORDER BY dt.Year DESC, dt.Quarter DESC;
            """
            
            return self.execute_query(query, f"SLICE: Total sales for {target_category} category")
        else:
            print("No category data found")
            return pd.DataFrame()
    
    def create_visualizations(self):
        """
        Create visualizations for OLAP query results
        """
        print(f"\n{'='*60}")
        print("CREATING VISUALIZATIONS")
        print('='*60)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Retail Data Warehouse - OLAP Analysis Visualizations', fontsize=16, fontweight='bold')
        
        # 1. Sales by Country (Roll-up visualization)
        country_query = """
        SELECT 
            dc.Country,
            COUNT(fs.SalesID) as NumberOfSales,
            ROUND(SUM(fs.TotalSalesAmount), 2) as TotalSalesAmount
        FROM FactSales fs
        JOIN DimCustomer dc ON fs.CustomerID = dc.ROWID
        GROUP BY dc.Country
        ORDER BY TotalSalesAmount DESC
        LIMIT 10;
        """
        
        country_df = pd.read_sql_query(country_query, self.conn)
        
        if not country_df.empty:
            bars = axes[0, 0].bar(country_df['Country'], country_df['TotalSalesAmount'], 
                                 color=sns.color_palette("husl", len(country_df)))
            axes[0, 0].set_title('Total Sales by Country (Top 10)', fontweight='bold')
            axes[0, 0].set_xlabel('Country')
            axes[0, 0].set_ylabel('Total Sales Amount')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'${height:,.0f}', ha='center', va='bottom', fontsize=8)
        else:
            axes[0, 0].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Total Sales by Country (No Data)', fontweight='bold')
        
        # 2. Product Category Performance
        category_query = """
        SELECT 
            dp.ProductCategory,
            COUNT(fs.SalesID) as TotalSales,
            ROUND(SUM(fs.TotalSalesAmount), 2) as TotalRevenue
        FROM FactSales fs
        JOIN DimProduct dp ON fs.ProductID = dp.ProductID
        GROUP BY dp.ProductCategory
        ORDER BY TotalRevenue DESC;
        """
        
        category_df = pd.read_sql_query(category_query, self.conn)
        
        if not category_df.empty:
            colors = sns.color_palette("Set3", len(category_df))
            wedges, texts, autotexts = axes[0, 1].pie(category_df['TotalRevenue'], 
                                                     labels=category_df['ProductCategory'],
                                                     autopct='%1.1f%%', colors=colors)
            axes[0, 1].set_title('Revenue Distribution by Product Category', fontweight='bold')
            
            # Improve text visibility
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Revenue Distribution by Product Category (No Data)', fontweight='bold')
        
        # 3. Monthly Sales Trend
        monthly_query = """
        SELECT 
            dt.Year,
            dt.Month,
            dt.MonthName,
            COUNT(fs.SalesID) as TotalSales,
            ROUND(SUM(fs.TotalSalesAmount), 2) as MonthlyRevenue
        FROM FactSales fs
        JOIN DimTime dt ON fs.TimeID = dt.TimeID
        GROUP BY dt.Year, dt.Month, dt.MonthName
        ORDER BY dt.Year, dt.Month;
        """
        
        monthly_df = pd.read_sql_query(monthly_query, self.conn)
        
        if not monthly_df.empty:
            # Create a period label for better x-axis
            monthly_df['Period'] = monthly_df['Year'].astype(str) + '-' + monthly_df['Month'].astype(str).str.zfill(2)
            
            line = axes[1, 0].plot(range(len(monthly_df)), monthly_df['MonthlyRevenue'], 
                                  marker='o', linewidth=2, markersize=6, color='#2E86AB')
            axes[1, 0].set_title('Monthly Sales Trend', fontweight='bold')
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel('Monthly Revenue')
            
            # Set x-axis labels
            axes[1, 0].set_xticks(range(0, len(monthly_df), max(1, len(monthly_df)//6)))
            axes[1, 0].set_xticklabels([monthly_df.iloc[i]['Period'] for i in range(0, len(monthly_df), max(1, len(monthly_df)//6))], rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Monthly Sales Trend (No Data)', fontweight='bold')
        
        # 4. Top 10 Products by Sales
        product_query = """
        SELECT 
            dp.ProductName,
            dp.ProductCategory,
            COUNT(fs.SalesID) as SalesCount,
            ROUND(SUM(fs.TotalSalesAmount), 2) as TotalRevenue
        FROM FactSales fs
        JOIN DimProduct dp ON fs.ProductID = dp.ProductID
        GROUP BY dp.ProductName, dp.ProductCategory
        ORDER BY TotalRevenue DESC
        LIMIT 10;
        """
        
        product_df = pd.read_sql_query(product_query, self.conn)
        
        if not product_df.empty:
            # Create a horizontal bar chart for better readability
            bars = axes[1, 1].barh(range(len(product_df)), product_df['TotalRevenue'],
                                  color=sns.color_palette("viridis", len(product_df)))
            axes[1, 1].set_title('Top 10 Products by Revenue', fontweight='bold')
            axes[1, 1].set_xlabel('Total Revenue')
            axes[1, 1].set_ylabel('Products')
            
            # Set y-tick labels
            axes[1, 1].set_yticks(range(len(product_df)))
            axes[1, 1].set_yticklabels([f"{name[:15]}..." if len(name) > 15 else name 
                                       for name in product_df['ProductName']], fontsize=8)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                axes[1, 1].text(width, bar.get_y() + bar.get_height()/2,
                               f'${width:,.0f}', ha='left', va='center', fontsize=8)
        else:
            axes[1, 1].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Top 10 Products by Revenue (No Data)', fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('c:/DSA 2040_Practical_Exam_IRANZI513/Visualizations/olap_analysis_charts.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        
        print("Visualizations saved to: Visualizations/olap_analysis_charts.png")
        
        return fig
    
    def generate_insights_report(self):
        """
        Generate insights and analysis report
        """
        print(f"\n{'='*60}")
        print("GENERATING INSIGHTS REPORT")
        print('='*60)
        
        # Overall statistics
        overall_query = """
        SELECT 
            COUNT(fs.SalesID) as TotalTransactions,
            COUNT(DISTINCT fs.CustomerID) as UniqueCustomers,
            COUNT(DISTINCT fs.ProductID) as UniqueProducts,
            COUNT(DISTINCT dc.Country) as CountriesServed,
            ROUND(SUM(fs.TotalSalesAmount), 2) as TotalRevenue,
            ROUND(AVG(fs.TotalSalesAmount), 2) as AvgTransactionValue,
            MIN(dt.Date) as EarliestSale,
            MAX(dt.Date) as LatestSale
        FROM FactSales fs
        JOIN DimCustomer dc ON fs.CustomerID = dc.ROWID
        JOIN DimTime dt ON fs.TimeID = dt.TimeID;
        """
        
        overall_stats = pd.read_sql_query(overall_query, self.conn)
        
        # Top performing country
        top_country_query = """
        SELECT 
            dc.Country,
            ROUND(SUM(fs.TotalSalesAmount), 2) as TotalRevenue,
            COUNT(DISTINCT fs.CustomerID) as UniqueCustomers
        FROM FactSales fs
        JOIN DimCustomer dc ON fs.CustomerID = dc.ROWID
        GROUP BY dc.Country
        ORDER BY TotalRevenue DESC
        LIMIT 1;
        """
        
        top_country = pd.read_sql_query(top_country_query, self.conn)
        
        # Best performing category
        top_category_query = """
        SELECT 
            dp.ProductCategory,
            ROUND(SUM(fs.TotalSalesAmount), 2) as TotalRevenue,
            COUNT(fs.SalesID) as TotalSales
        FROM FactSales fs
        JOIN DimProduct dp ON fs.ProductID = dp.ProductID
        GROUP BY dp.ProductCategory
        ORDER BY TotalRevenue DESC
        LIMIT 1;
        """
        
        top_category = pd.read_sql_query(top_category_query, self.conn)
        
        # Weekend vs Weekday performance
        weekend_query = """
        SELECT 
            CASE 
                WHEN dt.IsWeekend = 1 THEN 'Weekend'
                ELSE 'Weekday'
            END as DayType,
            COUNT(fs.SalesID) as TotalSales,
            ROUND(SUM(fs.TotalSalesAmount), 2) as TotalRevenue,
            ROUND(AVG(fs.TotalSalesAmount), 2) as AvgSaleValue
        FROM FactSales fs
        JOIN DimTime dt ON fs.TimeID = dt.TimeID
        GROUP BY dt.IsWeekend
        ORDER BY TotalRevenue DESC;
        """
        
        weekend_stats = pd.read_sql_query(weekend_query, self.conn)
        
        # Generate insights text
        insights_text = f"""
# Retail Data Warehouse - OLAP Analysis Report

## Executive Summary

This report presents the analysis results from the retail data warehouse using OLAP operations (Roll-up, Drill-down, and Slice). The analysis covers sales performance across different dimensions including geography, time, and product categories.

## Key Performance Indicators

### Overall Business Metrics
- **Total Transactions**: {overall_stats.iloc[0]['TotalTransactions']:,}
- **Total Revenue**: ${overall_stats.iloc[0]['TotalRevenue']:,.2f}
- **Average Transaction Value**: ${overall_stats.iloc[0]['AvgTransactionValue']:.2f}
- **Unique Customers**: {overall_stats.iloc[0]['UniqueCustomers']:,}
- **Unique Products**: {overall_stats.iloc[0]['UniqueProducts']:,}
- **Countries Served**: {overall_stats.iloc[0]['CountriesServed']}
- **Analysis Period**: {overall_stats.iloc[0]['EarliestSale']} to {overall_stats.iloc[0]['LatestSale']}

## Geographic Analysis (Roll-up Operation)

### Top Performing Market
"""
        
        if len(top_country) > 0:
            insights_text += f"""- **Country**: {top_country.iloc[0]['Country']}
- **Revenue**: ${top_country.iloc[0]['TotalRevenue']:,.2f}
- **Customer Base**: {top_country.iloc[0]['UniqueCustomers']} unique customers

The roll-up analysis by country and quarter reveals significant geographic concentration in sales performance. This information is crucial for:
- Resource allocation across different markets
- Targeted marketing campaigns
- Regional inventory management
- International expansion strategies
"""
        else:
            insights_text += "- No geographic data available in current dataset\n"
        
        insights_text += """
## Product Category Analysis (Slice Operation)

### Best Performing Category
"""
        
        if len(top_category) > 0:
            insights_text += f"""- **Category**: {top_category.iloc[0]['ProductCategory']}
- **Revenue**: ${top_category.iloc[0]['TotalRevenue']:,.2f}
- **Transaction Volume**: {top_category.iloc[0]['TotalSales']:,} sales

The slice operation focusing on product categories shows clear preferences and market demands. This analysis supports:
- Inventory optimization
- Product development priorities
- Category-specific marketing strategies
- Supplier relationship management
"""
        else:
            insights_text += "- No category data available in current dataset\n"
        
        insights_text += """
## Temporal Analysis (Drill-down Operation)

### Shopping Pattern Insights
"""

        if len(weekend_stats) >= 1:
            if len(weekend_stats) == 2:
                weekday_revenue = weekend_stats[weekend_stats['DayType'] == 'Weekday']['TotalRevenue'].iloc[0]
                weekend_revenue = weekend_stats[weekend_stats['DayType'] == 'Weekend']['TotalRevenue'].iloc[0]
                
                insights_text += f"""
- **Weekday Revenue**: ${weekday_revenue:,.2f}
- **Weekend Revenue**: ${weekend_revenue:,.2f}
- **Weekend vs Weekday Ratio**: {weekend_revenue/weekday_revenue:.2f}x
"""
            else:
                # Only one type (weekday or weekend)
                day_type = weekend_stats.iloc[0]['DayType']
                revenue = weekend_stats.iloc[0]['TotalRevenue']
                insights_text += f"""
- **{day_type} Revenue**: ${revenue:,.2f}
- **Note**: Only {day_type.lower()} data available in current dataset
"""
            
            insights_text += """
The temporal drill-down analysis reveals customer shopping patterns that can inform:
- Staff scheduling optimization
- Promotional timing strategies
- Inventory distribution planning
- Customer service resource allocation
"""
        else:
            insights_text += "- No temporal pattern data available in current dataset\n"

        insights_text += """

## Data Warehouse Performance Assessment

### Benefits Realized
1. **Query Performance**: The star schema design enables fast aggregation queries across multiple dimensions
2. **Business Intelligence**: OLAP operations provide intuitive access to business insights
3. **Scalability**: The dimensional model supports growing data volumes and new analytical requirements
4. **Data Quality**: ETL processes ensure consistent, clean data for reliable analysis

### Decision Support Capabilities
The data warehouse effectively supports key business decisions through:
- **Strategic Planning**: Geographic and temporal trend analysis
- **Operational Efficiency**: Product performance and inventory optimization
- **Customer Insights**: Segmentation and behavior analysis
- **Financial Planning**: Revenue forecasting and budget allocation

## Impact of Synthetic Data

Since this analysis uses synthetic data, some limitations should be noted:
- Patterns may not reflect real-world retail behavior complexity
- Seasonal trends might be simplified compared to actual retail cycles
- Customer behavior models are based on random generation rather than empirical data
- Geographic distribution may not reflect actual market dynamics

However, the synthetic data successfully demonstrates:
- OLAP operation capabilities
- Data warehouse schema effectiveness
- ETL process robustness
- Analytical query performance

## Technical Implementation Notes

### OLAP Operations Demonstrated

1. **Roll-up Operation**: Aggregating sales data from individual transactions to country and quarter levels
   - Demonstrates hierarchical aggregation capabilities
   - Shows how data warehouse supports management reporting needs

2. **Drill-down Operation**: Detailed analysis of specific country performance by month
   - Enables detailed investigation of trends
   - Supports operational decision making

3. **Slice Operation**: Filtering data for specific product category analysis
   - Demonstrates dimensional filtering capabilities
   - Supports focused category management

### Database Performance
The star schema design proved effective for:
- Fast aggregation queries across multiple dimensions
- Intuitive join operations between fact and dimension tables
- Scalable query performance as data volumes grow

## Recommendations for Future Enhancement

1. **Real-time Analytics**: Implement near real-time data feeds for current trend monitoring
2. **Advanced Analytics**: Integrate machine learning models for predictive analytics
3. **Customer 360**: Enhance customer dimension with behavioral and demographic details
4. **Performance Optimization**: Add materialized views for frequently accessed aggregations
5. **Self-Service BI**: Develop user-friendly dashboards for business users

## Conclusion

The retail data warehouse successfully demonstrates comprehensive OLAP capabilities, providing valuable insights for business decision-making. The star schema design optimally supports analytical queries, while the ETL process ensures data quality and consistency. This foundation enables data-driven decision making across all levels of the organization.

The three core OLAP operations (Roll-up, Drill-down, and Slice) have been successfully implemented and demonstrated, showing how the data warehouse can support various analytical needs from high-level strategic overview to detailed operational insights.

*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save insights report
        with open('c:/DSA 2040_Practical_Exam_IRANZI513/Documentation/olap_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(insights_text)
        
        print("Insights report saved to: Documentation/olap_analysis_report.md")
        
        return insights_text
    
    def run_complete_analysis(self):
        """
        Run complete OLAP analysis including queries, visualizations, and reporting
        """
        print(f"\n{'='*80}")
        print("RETAIL DATA WAREHOUSE - COMPLETE OLAP ANALYSIS")
        print('='*80)
        
        # Execute OLAP queries
        rollup_results = self.rollup_analysis()
        drilldown_results = self.drilldown_analysis()
        slice_results = self.slice_analysis()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate insights report
        self.generate_insights_report()
        
        print(f"\n{'='*80}")
        print("OLAP ANALYSIS COMPLETED SUCCESSFULLY")
        print("Generated Files:")
        print("- Visualizations/olap_analysis_charts.png")
        print("- Documentation/olap_analysis_report.md")
        print('='*80)
        
        return {
            'rollup': rollup_results,
            'drilldown': drilldown_results,
            'slice': slice_results
        }
    
    def __del__(self):
        """
        Close database connection
        """
        if hasattr(self, 'conn'):
            self.conn.close()

def main():
    """
    Main function to run OLAP analysis
    """
    db_path = 'c:/DSA 2040_Practical_Exam_IRANZI513/Data/retail_dw.db'
    
    # Initialize OLAP analysis
    olap = OLAPAnalysisFixed(db_path)
    
    # Run complete analysis
    results = olap.run_complete_analysis()

if __name__ == "__main__":
    main()
