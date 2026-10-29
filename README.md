# Customer RFM & Cohort Analytics Engine

An end-to-end Data Science and Unsupervised Machine Learning platform analyzing customer transactional behavior, acquisition cohorts, retention lifecycles, and RFM (Recency, Frequency, Monetary) metrics. Leverages Python, Pandas, Seaborn, and Scikit-Learn K-Means Clustering to segment customer tiers and forecast long-term lifetime value intent.

# 🛒 E-Commerce Customer Segmentation & RFM Cohort Analytics Engine

An end-to-end Data Science and Unsupervised Machine Learning platform analyzing customer transactional behavior, acquisition cohorts, retention lifecycles, and RFM (Recency, Frequency, Monetary) metrics. Leverages Python, Pandas, Seaborn, and Scikit-Learn K-Means Clustering to segment customer tiers and evaluate lifetime value intent.

---

## 📋 Project Overview

Understanding customer purchasing behavior and retention trends is vital for revenue optimization and targeted marketing. This project processes transactional e-commerce data to build a complete customer analytics pipeline. By combining **Cohort Analysis** (retention rates over lifecycle months) with **RFM Segmentation** (quantile scoring and unsupervised K-Means clustering), this system translates raw transactional line items into actionable business intelligence.

---

## 🎯 Key Objectives

* **Data Cleaning & Standardization:** Filter out cancellations, zero-price anomalies, missing customer identifiers, and duplicate records.
* **Cohort Analysis:** Track acquisition cohorts month-over-month to evaluate retention rates and average product purchasing volume over time.
* **RFM Score Computation:** Quantify **Recency** (days since last purchase), **Frequency** (total distinct orders), and **Monetary Value** (total revenue contributed).
* **Quantile & ML Segmentation:** Classify customers into heuristic tiers (**Gold, Silver, Bronze** / **Champions, At Risk, Lost**) and validate them against **Unsupervised K-Means Clustering**.
* **Visual Analytics & Diagnostics:** Generate intuitive heatmaps, distribution plots, and normalized **Snake Plots** to evaluate feature dominance per cluster.

---

## 🛠️ Dataset Architecture & Features

The platform ingests transactional data containing line-item records across international transactions.

| Feature Name | Raw Data Type | Transformed Type | Description |
| :--- | :--- | :--- | :--- |
| **`InvoiceNo`** | String / Object | String | Unique transaction code (prefixed with `'C'` for cancellations). |
| **`StockCode`** | String / Object | Categorical | Item/product identification code. |
| **`Description`** | String / Object | String | Product description cleaned of excess whitespace. |
| **`Quantity`** | Integer / Float | Integer | Number of units purchased per line item (filtered for $> 0$). |
| **`InvoiceDate`** | String / Object | Datetime64 | Date and time stamp of transaction completion. |
| **`UnitPrice`** | Float | Float | Unit price of item in currency units (filtered for $> 0.0$). |
| **`CustomerID`** | Float / Object | String / Integer | Unique customer identifier (rows with missing IDs dropped). |
| **`Country`** | String / Object | Categorical | Country where purchase originated. |
| **`TotalAmount`** | *Engineered* | Float | Line-item total spend calculated as $\text{Quantity} \times \text{UnitPrice}$. |
| **`CohortMonth`** | *Engineered* | Datetime64 | First acquisition month for a given `CustomerID`. |
| **`CohortIndex`** | *Engineered* | Integer | Lifecycle month index elapsed since first purchase ($1, 2, \dots, n$). |

---

## 🧠 Methodology, Dataset & EDA Summary

1. **Preprocessing & Anomaly Detection:** Dropped null `CustomerID` records, removed order cancellations (invoices starting with `'C'`), and filtered out zero/negative prices or quantities to isolate legitimate buying transactions.
2. **Feature Engineering:** Calculated overall item expenditure (`TotalAmount`), assigned acquisition baseline dates per customer (`CohortMonth`), and computed monthly offset indices (`CohortIndex`).
3. **Distribution Inspection (Skewness Diagnostic):**
   * Raw Recency, Frequency, and Monetary metrics exhibited extreme right-skewness.
   * Applied a natural **logarithmic transformation** ($\log(X)$) to normalize feature variance prior to distance-based modeling.

---

## 🤖 Machine Learning & Modeling Pipeline

The segmentation pipeline combines traditional quantile scoring with unsupervised machine learning:

1. **Standardization:** Normalized log-transformed features using `StandardScaler` ($\mu = 0, \sigma = 1$) to ensure equal Euclidean distance weighting across RFM dimensions.
2. **Hyperparameter Selection (Elbow Method & Silhouette):** Evaluated inertia curves and silhouette coefficients across $k \in [2, 7]$ to select $k = 3$ as the optimal cluster baseline.
3. **K-Means Model Fitting:** Fitted `KMeans(n_clusters=3, random_state=42)` on the scaled dataset to assign automated cluster boundaries (`K_Cluster`).

---

## 📊 Model Performance & Evaluation

* **Optimal Cluster Resolution ($k = 3$):**
  * **Cluster 0 (Moderate Active / Core Customers):** Average Recency ~55.4 days, Moderate Frequency (~3.4 orders), Steady Monetary value (~$1,318.4)[cite: 9].
  * **Cluster 1 (VIP / High-Value Champions):** Low Recency (~13.9 days), High Order Frequency (~12.7 orders), Highest Spend Contribution (~$7,351.6)[cite: 9].
  * **Cluster 2 (Hibernating / Churned Customers):** High Recency (~162.3 days), Low Frequency (~1.3 orders), Minimal Spend (~$334.1)[cite: 9].
* **Feature Relative Importance:**
  Calculated the proportional gap between cluster/tier averages relative to overall population averages:
  $$\text{Relative Importance} = \frac{\text{Mean}_{\text{Cluster}}}{\text{Mean}_{\text{Population}}} - 1$$
  * *Heatmap Insights:* High positive relative importance for Frequency and Monetary value confirmed Cluster 1 as the primary driver of enterprise revenue.

---

## 💻 Data Science Implementation & Stack

* **Language:** Python 3.x
* **Core Libraries:**
  * **Data Manipulation:** `pandas`, `numpy`
  * **Visualization & Plotting:** `matplotlib`, `seaborn`
  * **Machine Learning:** `scikit-learn` (`StandardScaler`, `KMeans`, `silhouette_score`)
  * **Date & Time Operations:** `datetime`

---

## ⚠️ Challenges & Future Improvements

* **Challenges Addressed:**
  * Handling extreme right-skewness in transaction monetary amounts without losing valuable high-spender variance.
  * Mitigating scale distortion in K-Means distance matrices through log-standardization.
* **Future Enhancements:**
  * **Predictive Churn Modeling:** Implement supervised classifiers (Random Forest / XGBoost) to predict customer churn probability within 30–90 days.
  * **Customer Lifetime Value (CLV) Forecasting:** Integrate BG/NBD and Gamma-Gamma probabilistic models to predict future transaction count and spend.
  * **Product Recommendation Integration:** Combine RFM clusters with Market Basket Analysis (Apriori algorithm) for targeted cross-selling.

---

## ✅ Project Deliverables Checklist

* [x] **Step 1:** Data Loading & Cleaning Pipeline (`load_and_clean_data`)
* [x] **Step 2A:** Cohort Feature Engineering (`CohortMonth`, `CohortIndex`)
* [x] **Step 2B:** RFM Metrics Aggregation & Quantile Segmentation
* [x] **Step 3:** Exploratory Data Analysis & Log-Transformation Diagnostics
* [x] **Step 4:** Cohort Retention Rates & Average Order Quantity Heatmaps
* [x] **Step 5:** K-Means ML Clustering Pipeline, Snake Plots & Relative Importance Heatmaps
