-- DSA 2040 Practical Exam - Data Warehouse Design
-- Task 1: Star Schema for Retail Data Warehouse
-- Author: IRANZI513
-- Date: August 14, 2025

-- =================================================================
-- STAR SCHEMA DESIGN FOR RETAIL DATA WAREHOUSE
-- =================================================================

-- This star schema consists of:
-- 1. FactSales (Fact Table)
-- 2. DimCustomer (Customer Dimension)
-- 3. DimProduct (Product Dimension) 
-- 4. DimTime (Time Dimension)
-- 5. DimStore (Store Dimension)

-- =================================================================
-- DIMENSION TABLES
-- =================================================================

-- Customer Dimension Table
CREATE TABLE DimCustomer (
    CustomerID INTEGER PRIMARY KEY,
    CustomerName TEXT NOT NULL,
    CustomerEmail TEXT,
    CustomerPhone TEXT,
    CustomerAddress TEXT,
    City TEXT,
    State TEXT,
    Country TEXT,
    ZipCode TEXT,
    CustomerSegment TEXT,  -- Premium, Standard, Basic
    RegistrationDate DATE,
    DateOfBirth DATE,
    Gender TEXT
);

-- Product Dimension Table
CREATE TABLE DimProduct (
    ProductID INTEGER PRIMARY KEY,
    ProductName TEXT NOT NULL,
    ProductCategory TEXT NOT NULL,  -- Electronics, Clothing, Books, etc.
    ProductSubCategory TEXT,
    Brand TEXT,
    ProductDescription TEXT,
    UnitPrice DECIMAL(10,2),
    ProductColor TEXT,
    ProductSize TEXT,
    ProductWeight DECIMAL(8,2),
    SupplierID INTEGER,
    SupplierName TEXT
);

-- Time Dimension Table
CREATE TABLE DimTime (
    TimeID INTEGER PRIMARY KEY,
    Date DATE NOT NULL,
    Day INTEGER,
    DayName TEXT,
    Week INTEGER,
    Month INTEGER,
    MonthName TEXT,
    Quarter INTEGER,
    Year INTEGER,
    IsWeekend BOOLEAN,
    IsHoliday BOOLEAN,
    FiscalYear INTEGER,
    FiscalQuarter INTEGER
);

-- Store Dimension Table
CREATE TABLE DimStore (
    StoreID INTEGER PRIMARY KEY,
    StoreName TEXT NOT NULL,
    StoreAddress TEXT,
    StoreCity TEXT,
    StoreState TEXT,
    StoreCountry TEXT,
    StoreZipCode TEXT,
    StoreType TEXT,  -- Online, Physical, Hybrid
    StoreSize TEXT,  -- Small, Medium, Large
    ManagerName TEXT,
    OpeningDate DATE
);

-- =================================================================
-- FACT TABLE
-- =================================================================

-- Sales Fact Table (Central fact table containing all measures)
CREATE TABLE FactSales (
    SalesID INTEGER PRIMARY KEY,
    CustomerID INTEGER NOT NULL,
    ProductID INTEGER NOT NULL,
    TimeID INTEGER NOT NULL,
    StoreID INTEGER NOT NULL,
    
    -- Measures (Facts)
    Quantity INTEGER NOT NULL,
    UnitPrice DECIMAL(10,2) NOT NULL,
    SalesAmount DECIMAL(12,2) NOT NULL,  -- Quantity * UnitPrice
    DiscountAmount DECIMAL(10,2) DEFAULT 0,
    TaxAmount DECIMAL(10,2) DEFAULT 0,
    TotalSalesAmount DECIMAL(12,2) NOT NULL,  -- SalesAmount - DiscountAmount + TaxAmount
    CostOfGoodsSold DECIMAL(12,2),
    GrossProfit DECIMAL(12,2),  -- TotalSalesAmount - CostOfGoodsSold
    
    -- Foreign Key Constraints
    FOREIGN KEY (CustomerID) REFERENCES DimCustomer(CustomerID),
    FOREIGN KEY (ProductID) REFERENCES DimProduct(ProductID),
    FOREIGN KEY (TimeID) REFERENCES DimTime(TimeID),
    FOREIGN KEY (StoreID) REFERENCES DimStore(StoreID)
);

-- =================================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- =================================================================

-- Indexes on fact table foreign keys for faster joins
CREATE INDEX idx_factsales_customerid ON FactSales(CustomerID);
CREATE INDEX idx_factsales_productid ON FactSales(ProductID);
CREATE INDEX idx_factsales_timeid ON FactSales(TimeID);
CREATE INDEX idx_factsales_storeid ON FactSales(StoreID);

-- Indexes on dimension table attributes for faster filtering
CREATE INDEX idx_dimproduct_category ON DimProduct(ProductCategory);
CREATE INDEX idx_dimcustomer_country ON DimCustomer(Country);
CREATE INDEX idx_dimtime_quarter ON DimTime(Quarter);
CREATE INDEX idx_dimtime_year ON DimTime(Year);
CREATE INDEX idx_dimstore_type ON DimStore(StoreType);

-- =================================================================
-- SAMPLE OLAP QUERIES SUPPORTED BY THIS SCHEMA
-- =================================================================

/*
1. Total sales by product category per quarter:
SELECT dp.ProductCategory, dt.Quarter, dt.Year, 
       SUM(fs.TotalSalesAmount) as TotalSales
FROM FactSales fs
JOIN DimProduct dp ON fs.ProductID = dp.ProductID
JOIN DimTime dt ON fs.TimeID = dt.TimeID
GROUP BY dp.ProductCategory, dt.Quarter, dt.Year
ORDER BY dt.Year, dt.Quarter, dp.ProductCategory;

2. Customer demographics analysis:
SELECT dc.Country, dc.CustomerSegment, 
       COUNT(*) as CustomerCount,
       AVG(fs.TotalSalesAmount) as AvgSalesPerCustomer
FROM FactSales fs
JOIN DimCustomer dc ON fs.CustomerID = dc.CustomerID
GROUP BY dc.Country, dc.CustomerSegment;

3. Inventory trends analysis:
SELECT dp.ProductCategory, dt.MonthName, dt.Year,
       SUM(fs.Quantity) as TotalQuantitySold,
       COUNT(DISTINCT fs.CustomerID) as UniqueCustomers
FROM FactSales fs
JOIN DimProduct dp ON fs.ProductID = dp.ProductID
JOIN DimTime dt ON fs.TimeID = dt.TimeID
GROUP BY dp.ProductCategory, dt.MonthName, dt.Year
ORDER BY dt.Year, dt.Month, dp.ProductCategory;
*/

-- =================================================================
-- END OF SCHEMA DEFINITION
-- =================================================================
