# Data Warehouse Design - Star Schema Explanation

## Overview
This document explains the star schema design for a retail company's data warehouse, as specified in Task 1 of the DSA 2040 Practical Exam.

## Star Schema Design

### Fact Table: FactSales
The central fact table contains all the measurable business events (sales transactions) with the following measures:
- **Quantity**: Number of items sold
- **UnitPrice**: Price per unit
- **SalesAmount**: Total sales before discounts (Quantity × UnitPrice)
- **DiscountAmount**: Discount applied to the sale
- **TaxAmount**: Tax charged on the sale
- **TotalSalesAmount**: Final amount after discounts and taxes
- **CostOfGoodsSold**: Cost to the company for the sold items
- **GrossProfit**: Profit margin (TotalSalesAmount - CostOfGoodsSold)

### Dimension Tables

#### 1. DimCustomer (Customer Dimension)
Contains comprehensive customer information:
- **Basic Info**: CustomerID, Name, Email, Phone
- **Geographic**: Address, City, State, Country, ZipCode
- **Segmentation**: CustomerSegment (Premium, Standard, Basic)
- **Demographics**: Gender, DateOfBirth, RegistrationDate

#### 2. DimProduct (Product Dimension)
Stores detailed product information:
- **Identity**: ProductID, ProductName, ProductDescription
- **Categorization**: ProductCategory, ProductSubCategory
- **Attributes**: Brand, Color, Size, Weight
- **Pricing**: UnitPrice
- **Supply Chain**: SupplierID, SupplierName

#### 3. DimTime (Time Dimension)
Provides comprehensive time-based analysis capabilities:
- **Calendar**: Date, Day, Week, Month, Quarter, Year
- **Named Periods**: DayName, MonthName
- **Business Context**: IsWeekend, IsHoliday
- **Fiscal Periods**: FiscalYear, FiscalQuarter

#### 4. DimStore (Store Dimension)
Contains store/channel information:
- **Identity**: StoreID, StoreName
- **Location**: Address, City, State, Country, ZipCode
- **Characteristics**: StoreType (Online, Physical, Hybrid), StoreSize
- **Management**: ManagerName, OpeningDate

## Why Star Schema Over Snowflake Schema

**I chose the star schema over the snowflake schema for the following reasons:**

1. **Query Performance**: Star schema provides faster query performance because it requires fewer joins. With dimension tables directly connected to the fact table, OLAP queries execute more efficiently, which is crucial for business intelligence and reporting applications.

2. **Simplicity and Usability**: The denormalized structure of star schema is more intuitive for business users and report developers. They can easily understand the relationships and write queries without navigating through multiple levels of normalized tables.

3. **Business Requirements Alignment**: Given the retail company's need for quick analytical queries (total sales by category per quarter, customer demographics analysis, inventory trends), the star schema's optimized structure for aggregation queries makes it the ideal choice.

## Supported Business Intelligence Queries

This star schema design efficiently supports the key business requirements:

### 1. Total Sales by Product Category per Quarter
```sql
SELECT dp.ProductCategory, dt.Quarter, dt.Year, 
       SUM(fs.TotalSalesAmount) as TotalSales
FROM FactSales fs
JOIN DimProduct dp ON fs.ProductID = dp.ProductID
JOIN DimTime dt ON fs.TimeID = dt.TimeID
GROUP BY dp.ProductCategory, dt.Quarter, dt.Year;
```

### 2. Customer Demographics Analysis
```sql
SELECT dc.Country, dc.CustomerSegment, 
       COUNT(*) as CustomerCount,
       AVG(fs.TotalSalesAmount) as AvgSalesPerCustomer
FROM FactSales fs
JOIN DimCustomer dc ON fs.CustomerID = dc.CustomerID
GROUP BY dc.Country, dc.CustomerSegment;
```

### 3. Inventory Trends Analysis
```sql
SELECT dp.ProductCategory, dt.MonthName, dt.Year,
       SUM(fs.Quantity) as TotalQuantitySold
FROM FactSales fs
JOIN DimProduct dp ON fs.ProductID = dp.ProductID
JOIN DimTime dt ON fs.TimeID = dt.TimeID
GROUP BY dp.ProductCategory, dt.MonthName, dt.Year;
```

## Performance Optimizations

The schema includes several performance optimizations:
- **Indexes** on all foreign keys in the fact table
- **Selective indexes** on frequently filtered dimension attributes
- **Appropriate data types** to minimize storage and improve query speed
- **Denormalized dimension tables** to reduce join complexity

## Scalability Considerations

The design accommodates future growth through:
- **Flexible product categorization** with category and subcategory fields
- **Comprehensive time dimension** supporting both calendar and fiscal year analysis
- **Extensible customer segmentation** allowing for additional classification schemes
- **Multi-channel support** through the store dimension's StoreType attribute
