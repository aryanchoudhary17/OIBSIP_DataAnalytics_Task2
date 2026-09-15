"""
OASIS INFOBYTE — SIP (Summer Internship Program)
Track: Data Analytics
Level: 1 | Task 2: Customer Segmentation Analysis (RFM & K-Means)
Author: Aryan
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.titlesize': 16
})

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "customer_transactions.csv")
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 70)
    print("OASIS INFOBYTE SIP — TASK 2: CUSTOMER SEGMENTATION ANALYSIS")
    print("=" * 70)

    # 1. Load Dataset & Data Cleaning
    print("\n[Step 1] Loading Dataset & Handling Missing/Inconsistent Data...")
    df_raw = pd.read_csv(data_path)
    print(f"Raw shape: {df_raw.shape}")
    print("Missing values before cleaning:\n", df_raw.isnull().sum())

    # Filter out missing CustomerIDs, non-positive quantities and prices
    df = df_raw.dropna(subset=['CustomerID']).copy()
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
    print(f"Cleaned shape: {df.shape} ({len(df_raw) - len(df)} anomalous rows filtered)")

    # 2. Descriptive Statistics (Purchase Value, Frequency, CLV)
    print("\n" + "=" * 70)
    print("[Step 2] Descriptive Customer Statistics")
    print("=" * 70)
    customer_spend = df.groupby('CustomerID')['TotalSpend'].sum()
    customer_freq = df.groupby('CustomerID')['InvoiceNo'].nunique()
    customer_avg_ticket = customer_spend / customer_freq

    desc_stats = pd.DataFrame({
        'Average Purchase Value ($)': customer_avg_ticket.describe(),
        'Purchase Frequency (Invoices)': customer_freq.describe(),
        'Customer Lifetime Value ($)': customer_spend.describe()
    })
    print(desc_stats.round(2))

    # 3. RFM Feature Engineering
    print("\n[Step 3] Computing RFM Metrics (Recency, Frequency, Monetary)...")
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSpend': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    print("RFM Head:")
    print(rfm.head())

    # Visualizing RFM Distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sns.histplot(rfm['Recency'], kde=True, ax=axes[0], color='#2b5c8f', bins=20)
    axes[0].set_title('Recency Distribution (Days)')
    axes[0].set_xlabel('Days Since Last Purchase')
    
    sns.histplot(rfm['Frequency'], kde=True, ax=axes[1], color='#3b9a62', bins=15)
    axes[1].set_title('Frequency Distribution (Invoices)')
    axes[1].set_xlabel('Unique Orders Count')
    
    sns.histplot(rfm['Monetary'], kde=True, ax=axes[2], color='#c2593f', bins=20)
    axes[2].set_title('Monetary Value Distribution ($)')
    axes[2].set_xlabel('Total Spend ($)')
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "01_rfm_distributions.png")
    fig.savefig(p1, dpi=300)
    plt.close()
    print(f"-> Saved: {p1}")

    # 4. Data Standardization
    print("\n[Step 4] Transforming & Standardizing RFM Features...")
    # Log-transform to handle right-skewness in RFM distributions
    rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    # 5. Elbow Method & Silhouette Analysis for Optimal K
    print("\n[Step 5] Applying Elbow Method & Silhouette Analysis...")
    wcss = []
    silhouette_scores = []
    k_range = range(2, 9)

    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans_temp.fit(rfm_scaled)
        wcss.append(kmeans_temp.inertia_)
        silhouette_scores.append(silhouette_score(rfm_scaled, kmeans_temp.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(k_range, wcss, marker='o', linewidth=2.5, color='#1f77b4')
    axes[0].set_title('Elbow Method for Optimal K', fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    axes[0].grid(True, linestyle='--')

    axes[1].plot(k_range, silhouette_scores, marker='s', linewidth=2.5, color='#2ca02c')
    axes[1].set_title('Silhouette Score by Cluster Count', fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].grid(True, linestyle='--')

    plt.tight_layout()
    p2 = os.path.join(plots_dir, "02_elbow_method_and_silhouette.png")
    fig.savefig(p2, dpi=300)
    plt.close()
    print(f"-> Saved: {p2}")

    # Select Optimal K = 4
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=15)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # 6. Cluster Profiling & Archetype Mapping
    cluster_means = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Count'})

    # Dynamically assign meaningful segment labels based on RFM profile
    # Champions: High Monetary, High Frequency, Low Recency
    # Loyalists: Moderate-High Freq & Monetary, moderate Recency
    # Potential Loyalists / Recent: Low Recency, Lower Freq
    # At-Risk / Hibernating: High Recency, Lower Freq
    cluster_scores = {}
    for c in range(optimal_k):
        m = cluster_means.loc[c, 'Monetary']
        f = cluster_means.loc[c, 'Frequency']
        r = cluster_means.loc[c, 'Recency']
        cluster_scores[c] = (m * f) / (r + 1)
    
    sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
    cluster_label_map = {
        sorted_clusters[0][0]: 'Champions (VIP)',
        sorted_clusters[1][0]: 'Loyal Customers',
        sorted_clusters[2][0]: 'Potential Loyalists (New)',
        sorted_clusters[3][0]: 'At Risk / Inactive'
    }
    rfm['Segment'] = rfm['Cluster'].map(cluster_label_map)
    print("\nCluster Profiling Table:")
    print(rfm.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).round(2))

    # 7. Cluster Visualizations (Scatter Plots)
    print("\n[Step 7] Generating Cluster Scatter Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    segment_colors = {
        'Champions (VIP)': '#27ae60',
        'Loyal Customers': '#2980b9',
        'Potential Loyalists (New)': '#f39c12',
        'At Risk / Inactive': '#c0392b'
    }

    sns.scatterplot(
        data=rfm, x='Recency', y='Monetary', hue='Segment', palette=segment_colors,
        alpha=0.85, s=70, ax=axes[0], edgecolor='k'
    )
    axes[0].set_title("Customer Clusters: Recency vs. Monetary Spend", fontweight='bold')
    axes[0].set_xlabel("Recency (Days since last order)")
    axes[0].set_ylabel("Monetary ($ Total Spend)")

    sns.scatterplot(
        data=rfm, x='Frequency', y='Monetary', hue='Segment', palette=segment_colors,
        alpha=0.85, s=70, ax=axes[1], edgecolor='k'
    )
    axes[1].set_title("Customer Clusters: Frequency vs. Monetary Spend", fontweight='bold')
    axes[1].set_xlabel("Frequency (Number of Orders)")
    axes[1].set_ylabel("Monetary ($ Total Spend)")

    plt.tight_layout()
    p3 = os.path.join(plots_dir, "03_cluster_scatter_rfm.png")
    fig.savefig(p3, dpi=300)
    plt.close()
    print(f"-> Saved: {p3}")

    # 8. Customer Count & Monetary Share by Cluster
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    segment_summary = rfm.groupby('Segment').agg(
        Customer_Count=('CustomerID', 'count'),
        Total_Revenue=('Monetary', 'sum')
    ).reset_index()

    sns.barplot(
        data=segment_summary, x='Segment', y='Customer_Count',
        palette=segment_colors, ax=axes[0], edgecolor='black', hue='Segment', legend=False
    )
    axes[0].set_title("Customer Count per Segment", fontweight='bold')
    axes[0].set_ylabel("Number of Customers")
    axes[0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(segment_summary['Customer_Count']):
        axes[0].text(i, v + 3, str(v), ha='center', fontweight='bold')

    sns.barplot(
        data=segment_summary, x='Segment', y='Total_Revenue',
        palette=segment_colors, ax=axes[1], edgecolor='black', hue='Segment', legend=False
    )
    axes[1].set_title("Total Revenue Contribution by Segment", fontweight='bold')
    axes[1].set_ylabel("Total Revenue ($)")
    axes[1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(segment_summary['Total_Revenue']):
        axes[1].text(i, v + 2000, f"${v:,.0f}", ha='center', fontweight='bold')

    plt.tight_layout()
    p4 = os.path.join(plots_dir, "04_customer_count_and_monetary_share.png")
    fig.savefig(p4, dpi=300)
    plt.close()
    print(f"-> Saved: {p4}")

    # Export Segmented CSV
    segmented_csv_path = os.path.join(script_dir, "data", "rfm_segmented_customers.csv")
    rfm.to_csv(segmented_csv_path, index=False)
    print(f"-> Saved segmented customer database to: {segmented_csv_path}")

    # 9. Targeted Marketing Action Plan
    print("\n" + "=" * 70)
    print("RECOMMENDED MARKETING ACTION PLAN PER SEGMENT")
    print("=" * 70)
    print("""
1. Champions (VIP):
   - Characteristics: Bought recently, order very frequently, highest monetary contribution.
   - Strategy: Exclusive loyalty perks, early access to new product drops, personalized concierge support.
   - Objective: Retention, brand advocacy, and premium upsells.

2. Loyal Customers:
   - Characteristics: Steady purchase history with solid lifetime spend and low churn risk.
   - Strategy: Tiered milestone reward programs, bundle incentives, and referral reward bonuses.
   - Objective: Increase Average Order Value (AOV) and elevate them into Champions.

3. Potential Loyalists (New):
   - Characteristics: High recency (bought recently) but lower frequency and moderate spend.
   - Strategy: Welcoming onboarding drip campaigns, second-purchase discount vouchers, product education.
   - Objective: Cultivate repeat purchasing habit within 30 days of initial acquisition.

4. At Risk / Inactive:
   - Characteristics: Long days since last transaction; previously active but disengaging.
   - Strategy: Re-engagement campaigns ('We Miss You' offers), feedback survey incentives, time-limited winback promo.
   - Objective: Reactivate lapsed accounts before customer attrition becomes permanent.
    """)
    print("=" * 70)
    print("Task 2 completed successfully!")

if __name__ == '__main__':
    main()