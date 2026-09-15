# Level 1 — Task 2: Customer Segmentation Analysis

**Domain:** Data Analytics  
**Program:** Oasis Infobyte Summer Internship Program (OIBSIP)  
**Repository:** `OIBSIP_DataAnalytics_Task2`  
**Author:** Aryan  

---

## 1. Objective
The objective of this project is to apply unsupervised machine learning (**K-Means Clustering**) combined with the industry-standard **RFM (Recency, Frequency, Monetary) analytical framework** on e-commerce transaction data. The goal is to mathematically segment the customer base into distinct behavioral cohorts, enabling the business to transition from generic mass marketing to targeted, high-ROI retention and upselling strategies.

---

## 2. Steps Performed
1. **Data Ingestion & Integrity Cleansing:**
   - Ingested 5,871 raw customer transaction records (`data/customer_transactions.csv`).
   - Filtered 15 records with missing customer identifiers and 20 invalid negative/zero quantity transactions representing returns or cancellations.
2. **Customer-Level Descriptive Analysis:**
   - Calculated customer-level summary metrics: Average Purchase Value, Purchase Frequency, and Customer Lifetime Value (CLV).
3. **RFM Metric Engineering:**
   - Extracted three behavioral features for each unique customer:
     - **Recency ($R$):** Days elapsed since the customer's most recent completed order.
     - **Frequency ($F$):** Total count of unique completed purchase orders.
     - **Monetary ($M$):** Cumulative revenue contributed across all transactions.
4. **Mathematical Transformation & Standardization:**
   - Applied logarithmic transformation (`np.log1p`) to mitigate positive skewness in transactional distributions.
   - Normalized features using `StandardScaler` to ensure equal variance weighting for Euclidean distance computations in K-Means.
5. **Cluster Optimization (Elbow & Silhouette):**
   - Evaluated cluster counts from $K=2$ to $K=8$.
   - Analyzed Within-Cluster Sum of Squares (WCSS / Elbow Method) and Silhouette Coefficients to determine the optimal partition ($K=4$).
6. **Persona Profiling & Cluster Evaluation:**
   - Mapped clusters to actionable commercial personas based on mean RFM metrics.
   - Generated 2D scatter plots (Recency vs. Monetary, Frequency vs. Monetary) and volume vs. value comparative distributions.
   - Formulated customized marketing playbooks for each segment.

---

## 3. Tools Used
- **Programming Language:** Python 3.12+
- **Machine Learning & Preprocessing:** `scikit-learn` (`KMeans`, `StandardScaler`, `silhouette_score`)
- **Data Analysis & Manipulation:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Interactive Computing:** Jupyter Notebook (`.ipynb`)
- **Version Control:** Git & GitHub

---

## 4. Outcome in Brief
- **Mathematical Cluster Partition ($K=4$):** The customer base of 420 accounts was successfully partitioned into 4 distinct, actionable behavioral segments:
  1. **Champions (VIP - 14.5% of Customers):** High recency (mean 25 days), frequent buyers (mean 13.6 orders), and substantial spending (mean $37,950 spend). This small cohort drives **51.7% of total company revenue**.
  2. **Loyal Customers (30.7% of Customers):** Consistent buyers (mean 90 days recency, 8.1 orders, $13,627 spend) accounting for **39.1% of revenue**.
  3. **Potential Loyalists / New (12.1% of Customers):** Recently active (mean 67 days recency) but early in their lifecycle (mean 2.3 orders, $2,325 spend), contributing **2.6% of revenue**.
  4. **At-Risk / Inactive (42.6% of Customers):** Long dormancy (mean 263 days recency, 1.8 orders, $1,632 spend), contributing only **6.5% of revenue**.
- **Commercial Takeaway & Marketing Playbook:**
  - *Champions:* Prioritize relationship retention through exclusive VIP perks, early product drops, and concierge service rather than deep discounting.
  - *Loyalists:* Deploy milestone-based loyalty rewards and bundle incentives to elevate them into Champions.
  - *Potential Loyalists:* Trigger an automated 3-part educational email sequence with a 15% incentive for repeat purchases within 21 days.
  - *At-Risk Customers:* Launch automated "We Miss You" win-back campaigns featuring personalized product recommendations and time-limited reactivation vouchers.

---

## 5. Repository Structure & How to Run
```
OIBSIP_DataAnalytics_Task2/
│
├── data/
│   ├── customer_transactions.csv      # Raw transaction history
│   └── rfm_segmented_customers.csv    # Exported RFM database with cluster labels
├── plots/
│   ├── 01_rfm_distributions.png
│   ├── 02_elbow_method_and_silhouette.png
│   ├── 03_cluster_scatter_rfm.png
│   └── 04_customer_count_and_monetary_share.png
├── Customer_Segmentation_RFM.ipynb    # Fully executed interactive Jupyter Notebook
├── customer_segmentation.py           # Standalone Python script
└── README.md                          # Project documentation
```

### Execution Instructions:
```bash
# Run standalone script:
python customer_segmentation.py

# Launch interactive notebook:
jupyter notebook Customer_Segmentation_RFM.ipynb
```