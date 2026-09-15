# Task 2: Customer Segmentation Analysis

**Internship Track:** Data Analytics  
**Program:** Oasis Infobyte Summer Internship Program (OIBSIP)  
**Level:** Level 1 — Task 2  
**Folder Format:** `OIBSIP/DataAnalytics-L1-CustomerSegmentation`  
**Author:** Aryan  

---

## 📌 Project Objective
The goal of this project is to apply unsupervised machine learning (**K-Means Clustering**) integrated with the **RFM (Recency, Frequency, Monetary) analytical model** on customer transaction data. This enables the business to partition customers into distinct behavioral segments and deploy targeted marketing strategies to increase customer retention and gross revenue.

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python 3.12+
- **Machine Learning & Preprocessing:** `scikit-learn` (`KMeans`, `StandardScaler`, `silhouette_score`)
- **Data Manipulation:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Interactive Environment:** Jupyter Notebook (`.ipynb`)

---

## 📋 Feature Checklist Verification
- [x] **Data Loading & Inspection:** Loaded 5,871 transactions; successfully identified and handled 15 missing customer records and 20 negative/zero return anomalies.
- [x] **Descriptive Statistics:** Computed customer-level Average Purchase Value ($1,443.78 mean), Purchase Frequency (mean 5.8 orders), and Customer Lifetime Value ($10,675.28 mean).
- [x] **Feature Selection (RFM):** Computed Recency ($R$, days since last order), Frequency ($F$, count of unique orders), and Monetary ($M$, total cumulative spend).
- [x] **Data Normalization:** Transformed features with `log1p` and normalized with `StandardScaler` to satisfy K-Means assumptions.
- [x] **Elbow Method & Silhouette Score:** Explored $K=2$ through $K=8$; identified optimal cluster count at $K=4$.
- [x] **Cluster Scatter Plots:** Visualized cluster boundaries for Recency vs. Monetary and Frequency vs. Monetary.
- [x] **Cluster Profiling:** Calculated mean and median metrics per cluster and assigned commercial persona labels:
  - *Champions (VIP)*
  - *Loyal Customers*
  - *Potential Loyalists (New)*
  - *At Risk / Inactive*
- [x] **Volume vs. Value Analysis:** Bar charts depicting customer headcounts and revenue shares per segment.
- [x] **Actionable Marketing Insights:** Segment-specific playbooks to maximize retention, conversion, and reactivation.

---

## 👥 Customer Segment Persona Profiles

| Segment | Recency (Mean) | Frequency (Mean) | Monetary Spend (Mean) | % Customer Base | % Gross Revenue |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Champions (VIP)** | 25 days | 13.6 orders | $37,950 | 14.5% | 51.7% |
| **Loyal Customers** | 90 days | 8.1 orders | $13,627 | 30.7% | 39.1% |
| **Potential Loyalists (New)** | 67 days | 2.3 orders | $2,325 | 12.1% | 2.6% |
| **At Risk / Inactive** | 263 days | 1.8 orders | $1,632 | 42.6% | 6.5% |

---

## 🎯 Targeted Marketing Playbook

1. **Champions (VIP):**
   - *Strategy:* Personalized recognition, private previews, concierge customer service, and exclusive loyalty perks.
   - *Goal:* Maximize lifetime engagement and foster organic word-of-mouth advocacy.
2. **Loyal Customers:**
   - *Strategy:* Milestone tier-ups (e.g. "Spend $150 more this quarter to reach VIP"), bundle deals, and loyalty points multipliers.
   - *Goal:* Elevate high-frequency buyers into top-tier VIPs.
3. **Potential Loyalists (New):**
   - *Strategy:* Automated 3-part onboarding email sequence with product walkthroughs, related recommendations, and a 15% incentive for orders within 21 days.
   - *Goal:* Convert one-off buyers into routine repeat shoppers.
4. **At Risk / Inactive:**
   - *Strategy:* "We Miss You" win-back campaigns featuring personalized product recommendations from their prior purchase history and limited-time reactivation voucher codes.
   - *Goal:* Recover dormant accounts before complete churn occurs.

---

## 📂 Project Structure
```
OIBSIP/DataAnalytics-L1-CustomerSegmentation/
│
├── data/
│   ├── customer_transactions.csv      # Raw transaction history
│   └── rfm_segmented_customers.csv    # Exported RFM database with cluster labels
├── plots/
│   ├── 01_rfm_distributions.png
│   ├── 02_elbow_method_and_silhouette.png
│   ├── 03_cluster_scatter_rfm.png
│   └── 04_customer_count_and_monetary_share.png
├── Customer_Segmentation_RFM.ipynb    # Executed interactive Jupyter Notebook
├── customer_segmentation.py           # Automated standalone Python script
└── README.md                          # Comprehensive documentation
```

---

## 🚀 How to Run the Project

1. **Navigate to the Task Directory:**
   ```bash
   cd OIBSIP/DataAnalytics-L1-CustomerSegmentation
   ```

2. **Install Required Libraries:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

3. **Run the Script:**
   ```bash
   python customer_segmentation.py
   ```

4. **Launch the Notebook:**
   ```bash
   jupyter notebook Customer_Segmentation_RFM.ipynb
   ```

---

## 🎥 Video Walkthrough Guide (for Oasis Infobyte Submission)
- **Duration:** 3–5 minutes
- **Title Card (First 2 Seconds):**
  - **Full Name:** Aryan
  - **Assigned Track:** Data Analytics
  - **Task Title:** Level 1 — Task 2: Customer Segmentation Analysis
- **Walkthrough Agenda:**
  1. Introduction to the RFM framework and business goal.
  2. Data preprocessing: handling missing IDs, anomalies, and log scaling.
  3. Explaining the Elbow Method and Silhouette Score for selecting $K=4$.
  4. Interpreting the 2D cluster scatter plots and the customer profiling table.
  5. Presenting the marketing playbooks for Champions, Loyalists, New Customers, and At-Risk groups.

---

## 📱 LinkedIn Post Template
```
Thrilled to present Task 2 (Level 1) of my Data Analytics Internship with @Oasis Infobyte! 🎯

📈 Task 2: Customer Segmentation Analysis using RFM & K-Means Clustering
Segmenting customers allows organizations to transition from generic mass marketing to hyper-personalized communication that accelerates revenue and customer lifetime value (CLV).

Key Highlights:
✅ Data cleaning & anomalous transaction filtering
✅ RFM (Recency, Frequency, Monetary) metric derivation
✅ Feature transformation (Log1p & StandardScaler)
✅ Optimal cluster selection using Elbow Method & Silhouette Coefficient (K=4)
✅ Persona mapping: Champions, Loyal Customers, Potential Loyalists, and At-Risk customers
✅ Actionable marketing strategies and campaign triggers for each segment

GitHub Repository: [Insert Your GitHub Repo Link Here]
Demo Video: [Insert Video Link Here]

Special thanks to @Oasis Infobyte for this enriching project experience!

#oasisinfobyte #dataanalytics #machinelearning #kmeans #clustering #rfm #customersegmentation #python #scikitlearn #internship
```