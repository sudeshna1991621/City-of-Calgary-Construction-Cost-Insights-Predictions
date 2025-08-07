# City of Calgary – Project Cost Prediction & Trends Dashboard

## **Purpose**
A Power BI solution analyzing over a decade of building permit data for Calgary to deliver actionable insights 
for executives, planners, and analysts. The report combines KPI tracking, community‑level analysis, and 
machine learning–based project cost prediction.

---

## **Data Preparation**
- Cleaned and transformed permit data in **Python** (Pandas) and Power BI.
- Imputed missing values (e.g., *TotalSqFt*) using multiple statistical methods.
- Created calculated columns and measures in **DAX** for KPIs, model evaluation, and data quality tracking.
- Integrated **XGBoost regression model** for cost prediction, exporting results and feature importance back into Power BI.

---

## **Key DAX Measures**
```DAX
-- Total Estimated Cost
TotalEstCost =
SUM('BP_RAW'[EstimatedProjectCost])

-- Average Permit Cost
AvgPermitCost =
DIVIDE([TotalEstCost], COUNT('BP_RAW'[PermitNum]))

-- MostActiveCommunity
MostActiveCommunity = 
CALCULATE (
    SELECTCOLUMNS (
        TOPN (
            1, 
            VALUES('BP_RAW'[CommunityName]), 
            CALCULATE(COUNT('BP_RAW'[PermitNum])), 
            DESC
        ), 
        "Community", 'BP_RAW'[CommunityName]
    )
)


-- R² Score for Model Performance
R2_Score =
VAR SST =
    VAR MeanActual = AVERAGE('BP_FINAL'[EstProjectCost])
    RETURN SUMX('BP_FINAL', ('BP_FINAL'[EstProjectCost] - MeanActual)^2)
VAR SSR =
    SUMX('BP_FINAL', ('BP_FINAL'[EstProjectCost] - 'BP_FINAL'[cost_xgb])^2)
RETURN
1 - DIVIDE(SSR, SST)

-- RMSE for Model Performance
RMSE_XGB =
VAR MSE =
    AVERAGEX(
        'BP_FINAL',
        ('BP_FINAL'[EstProjectCost] - 'BP_FINAL'[cost_xgb]) ^ 2
    )
RETURN
SQRT(MSE)
```

---

## **Report Pages**
1. **Overview Dashboard** – KPI cards (*Total Permits, Total Estimated Cost, Average Permit Cost, Most Active Community*), yearly trends, permit type distribution, and slicers.
2. **Community Insights** – Map by community, top 10 communities by permit count and cost, and community‑level summary tables.
3. **Predictive Insights** – Missing Values bar chart, Feature Importance bar chart, and Error Distribution histogram.

---

## **Tools & Technologies**
- **Power BI** (DAX, bookmarks, slicers, map visuals)
- **Python** (Pandas, Scikit‑learn, XGBoost)
- **Excel** (data audit & export)

---

## **Insights**
- After cleaning, **378K permits** analyzed; **Residential Improvements** dominate with **183K permits**.
- **XGBoost** achieved **R² = 0.85** and **RMSE ≈ $45K**, showing strong prediction accuracy.
- Commercial/Multi-Family projects have higher errors; specialized models could improve estimates.
- Large missing **TotalSqFt** imputed; better data capture needed at application stage.
- **Downtown** leads with **$8.8B** in permits; **Cornerstone** and **Livingston** are fast-growing communities.
- **Single Family permits** lead in cost and count; 2024 had the highest permit volume.

---

## **Recommendations**
- Build tailored models for commercial/multi-family permits.
- Improve TotalSqFt data collection at application stage.
- Focus planning on high-growth communities like Cornerstone.
