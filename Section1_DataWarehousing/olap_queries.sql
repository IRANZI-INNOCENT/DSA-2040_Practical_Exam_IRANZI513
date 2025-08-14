-- DSA 2040 Practical Exam - OLAP Queries and Analysis
-- Task 3: OLAP-style SQL queries for retail data warehouse
-- Author: IRANZI513
-- Date: August 14, 2025

-- =================================================================
-- OLAP QUERIES FOR RETAIL DATA WAREHOUSE ANALYSIS
-- =================================================================

-- These queries demonstrate OLAP operations on the retail data warehouse
-- including Roll-up, Drill-down, and Slice operations

-- =================================================================
-- QUERY 1: ROLL-UP - Total sales by country and quarter
-- =================================================================

-- This query aggregates sales data to a higher level (country and quarter)
-- demonstrating the roll-up OLAP operation

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
JOIN DimCustomer dc ON fs.CustomerID = dc.CustomerID
JOIN DimTime dt ON fs.TimeID = dt.TimeID
GROUP BY dc.Country, dt.Quarter, dt.Year
ORDER BY dt.Year DESC, dt.Quarter DESC, TotalSalesAmount DESC;

-- =================================================================
-- QUERY 2: DRILL-DOWN - Sales details for United Kingdom by month
-- =================================================================

-- This query provides detailed breakdown for a specific country (UK)
-- by month, demonstrating the drill-down OLAP operation

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
JOIN DimCustomer dc ON fs.CustomerID = dc.CustomerID
JOIN DimTime dt ON fs.TimeID = dt.TimeID
WHERE dc.Country = 'United Kingdom'
GROUP BY dc.Country, dt.Year, dt.MonthName, dt.Month
ORDER BY dt.Year DESC, dt.Month DESC;

-- =================================================================
-- QUERY 3: SLICE - Total sales for Electronics category
-- =================================================================

-- This query filters data for a specific product category (Electronics)
-- demonstrating the slice OLAP operation

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
JOIN DimCustomer dc ON fs.CustomerID = dc.CustomerID
JOIN DimTime dt ON fs.TimeID = dt.TimeID
WHERE dp.ProductCategory = 'Electronics'
GROUP BY dp.ProductCategory, dt.Quarter, dt.Year
ORDER BY dt.Year DESC, dt.Quarter DESC;

-- =================================================================
-- ADDITIONAL ANALYTICAL QUERIES
-- =================================================================

-- Query 4: Top 10 customers by total sales
SELECT 
    dc.CustomerID,
    dc.CustomerName,
    dc.Country,
    dc.CustomerSegment,
    COUNT(fs.SalesID) as TotalOrders,
    SUM(fs.Quantity) as TotalItemsPurchased,
    ROUND(SUM(fs.TotalSalesAmount), 2) as TotalSpent,
    ROUND(AVG(fs.TotalSalesAmount), 2) as AvgOrderValue
FROM FactSales fs
JOIN DimCustomer dc ON fs.CustomerID = dc.CustomerID
GROUP BY dc.CustomerID, dc.CustomerName, dc.Country, dc.CustomerSegment
ORDER BY TotalSpent DESC
LIMIT 10;

-- Query 5: Product category performance analysis
SELECT 
    dp.ProductCategory,
    COUNT(fs.SalesID) as TotalSales,
    SUM(fs.Quantity) as TotalQuantitySold,
    ROUND(SUM(fs.TotalSalesAmount), 2) as TotalRevenue,
    ROUND(AVG(fs.TotalSalesAmount), 2) as AvgSaleValue,
    COUNT(DISTINCT fs.CustomerID) as UniqueCustomers,
    COUNT(DISTINCT fs.ProductID) as UniqueProducts,
    ROUND(SUM(fs.TotalSalesAmount) * 100.0 / 
          (SELECT SUM(TotalSalesAmount) FROM FactSales), 2) as RevenueSharePercent
FROM FactSales fs
JOIN DimProduct dp ON fs.ProductID = dp.ProductID
GROUP BY dp.ProductCategory
ORDER BY TotalRevenue DESC;

-- Query 6: Monthly sales trend analysis
SELECT 
    dt.Year,
    dt.MonthName,
    dt.Month,
    COUNT(fs.SalesID) as TotalSales,
    ROUND(SUM(fs.TotalSalesAmount), 2) as MonthlyRevenue,
    COUNT(DISTINCT fs.CustomerID) as UniqueCustomers,
    ROUND(AVG(fs.TotalSalesAmount), 2) as AvgSaleValue
FROM FactSales fs
JOIN DimTime dt ON fs.TimeID = dt.TimeID
GROUP BY dt.Year, dt.MonthName, dt.Month
ORDER BY dt.Year, dt.Month;

-- Query 7: Weekend vs Weekday sales comparison
SELECT 
    CASE 
        WHEN dt.IsWeekend = 1 THEN 'Weekend'
        ELSE 'Weekday'
    END as DayType,
    COUNT(fs.SalesID) as TotalSales,
    ROUND(SUM(fs.TotalSalesAmount), 2) as TotalRevenue,
    ROUND(AVG(fs.TotalSalesAmount), 2) as AvgSaleValue,
    COUNT(DISTINCT fs.CustomerID) as UniqueCustomers
FROM FactSales fs
JOIN DimTime dt ON fs.TimeID = dt.TimeID
GROUP BY dt.IsWeekend
ORDER BY TotalRevenue DESC;

-- =================================================================
-- PERFORMANCE VERIFICATION QUERIES
-- =================================================================

-- Query to verify data integrity
SELECT 
    'FactSales' as TableName,
    COUNT(*) as RecordCount
FROM FactSales
UNION ALL
SELECT 
    'DimCustomer' as TableName,
    COUNT(*) as RecordCount
FROM DimCustomer
UNION ALL
SELECT 
    'DimProduct' as TableName,
    COUNT(*) as RecordCount
FROM DimProduct
UNION ALL
SELECT 
    'DimTime' as TableName,
    COUNT(*) as RecordCount
FROM DimTime
UNION ALL
SELECT 
    'DimStore' as TableName,
    COUNT(*) as RecordCount
FROM DimStore;
