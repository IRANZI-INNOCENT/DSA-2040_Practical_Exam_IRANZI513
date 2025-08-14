"""
DSA 2040 Practical Exam - Star Schema Diagram Generator
Task 1: Generate visual diagram for retail data warehouse star schema
Author: IRANZI513
Date: August 14, 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_star_schema_diagram():
    """
    Create a visual representation of the retail data warehouse star schema
    """
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    fact_color = '#FFE6E6'  # Light red for fact table
    dim_color = '#E6F3FF'   # Light blue for dimension tables
    border_color = '#333333'
    text_color = '#000000'
    
    # Title
    ax.text(5, 9.5, 'Retail Data Warehouse - Star Schema', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # ============ FACT TABLE (CENTER) ============
    fact_box = FancyBboxPatch((3.5, 4), 3, 2,
                              boxstyle="round,pad=0.1",
                              facecolor=fact_color,
                              edgecolor=border_color,
                              linewidth=2)
    ax.add_patch(fact_box)
    
    # Fact table content
    fact_text = """FactSales (Fact Table)
    
SalesID (PK)
CustomerID (FK)
ProductID (FK)
TimeID (FK)
StoreID (FK)

Measures:
• Quantity
• UnitPrice
• SalesAmount
• DiscountAmount
• TaxAmount
• TotalSalesAmount
• CostOfGoodsSold
• GrossProfit"""
    
    ax.text(5, 5, fact_text, ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # ============ DIMENSION TABLES ============
    
    # DimCustomer (Top Left)
    cust_box = FancyBboxPatch((0.5, 7), 2.5, 1.8,
                              boxstyle="round,pad=0.1",
                              facecolor=dim_color,
                              edgecolor=border_color,
                              linewidth=1.5)
    ax.add_patch(cust_box)
    
    cust_text = """DimCustomer
    
CustomerID (PK)
CustomerName
CustomerEmail
CustomerPhone
CustomerAddress
City, State, Country
CustomerSegment
Gender, DateOfBirth"""
    
    ax.text(1.75, 7.9, cust_text, ha='center', va='center', 
            fontsize=9)
    
    # DimProduct (Top Right)
    prod_box = FancyBboxPatch((7, 7), 2.5, 1.8,
                              boxstyle="round,pad=0.1",
                              facecolor=dim_color,
                              edgecolor=border_color,
                              linewidth=1.5)
    ax.add_patch(prod_box)
    
    prod_text = """DimProduct
    
ProductID (PK)
ProductName
ProductCategory
ProductSubCategory
Brand, Color, Size
UnitPrice
SupplierID
SupplierName"""
    
    ax.text(8.25, 7.9, prod_text, ha='center', va='center', 
            fontsize=9)
    
    # DimTime (Bottom Left)
    time_box = FancyBboxPatch((0.5, 1.2), 2.5, 1.8,
                              boxstyle="round,pad=0.1",
                              facecolor=dim_color,
                              edgecolor=border_color,
                              linewidth=1.5)
    ax.add_patch(time_box)
    
    time_text = """DimTime
    
TimeID (PK)
Date
Day, Week, Month
Quarter, Year
DayName, MonthName
IsWeekend, IsHoliday
FiscalYear
FiscalQuarter"""
    
    ax.text(1.75, 2.1, time_text, ha='center', va='center', 
            fontsize=9)
    
    # DimStore (Bottom Right)
    store_box = FancyBboxPatch((7, 1.2), 2.5, 1.8,
                               boxstyle="round,pad=0.1",
                               facecolor=dim_color,
                               edgecolor=border_color,
                               linewidth=1.5)
    ax.add_patch(store_box)
    
    store_text = """DimStore
    
StoreID (PK)
StoreName
StoreAddress
StoreCity, StoreState
StoreType
StoreSize
ManagerName
OpeningDate"""
    
    ax.text(8.25, 2.1, store_text, ha='center', va='center', 
            fontsize=9)
    
    # ============ RELATIONSHIPS (LINES) ============
    
    # Customer to Fact
    line1 = ConnectionPatch((3, 7.5), (3.8, 5.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=20, fc=border_color, 
                           linewidth=2)
    ax.add_patch(line1)
    
    # Product to Fact
    line2 = ConnectionPatch((7, 7.5), (6.2, 5.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=20, fc=border_color,
                           linewidth=2)
    ax.add_patch(line2)
    
    # Time to Fact
    line3 = ConnectionPatch((3, 2.5), (3.8, 4.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=20, fc=border_color,
                           linewidth=2)
    ax.add_patch(line3)
    
    # Store to Fact
    line4 = ConnectionPatch((7, 2.5), (6.2, 4.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=20, fc=border_color,
                           linewidth=2)
    ax.add_patch(line4)
    
    # ============ LEGEND ============
    fact_patch = mpatches.Patch(color=fact_color, label='Fact Table')
    dim_patch = mpatches.Patch(color=dim_color, label='Dimension Tables')
    
    ax.legend(handles=[fact_patch, dim_patch], 
              loc='upper right', bbox_to_anchor=(1, 1))
    
    # ============ ANNOTATIONS ============
    ax.text(5, 0.5, 'Star Schema Design for Retail Data Warehouse\n' +
                   'Central Fact Table connected to 4 Dimension Tables\n' +
                   'Optimized for OLAP queries and business intelligence',
            ha='center', va='center', fontsize=12, 
            style='italic', color='#555555')
    
    # Save the diagram
    plt.tight_layout()
    plt.savefig('c:/DSA 2040_Practical_Exam_IRANZI513/Visualizations/star_schema_diagram.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('c:/DSA 2040_Practical_Exam_IRANZI513/Documentation/star_schema_diagram.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print("Star schema diagram saved successfully!")
    print("Location: Visualizations/star_schema_diagram.png")
    
    return fig

if __name__ == "__main__":
    # Create the diagram
    fig = create_star_schema_diagram()
    # plt.show()  # Commented out to avoid GUI display issues
