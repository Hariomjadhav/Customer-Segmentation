import os
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# PAGE CONFIG
st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide"
)

st.title("Customer Segmentation Using Machine Learning")

st.write(
    "Customer Segmentation using K-Means "
    "based on customer characteristics and purchasing behaviour."
)

# LOAD DATASET
df = pd.read_csv("Mall_Customers.csv")

st.success("Dataset Loaded Successfully!")

# DATASET OVERVIEW
st.header("1. Customer Dataset Analysis")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Average Age", round(df["Age"].mean(), 2))
col3.metric(
    "Average Income",
    round(df["Annual Income (k$)"].mean(), 2)
)

st.dataframe(df, width="stretch")

# CUSTOMER CHARACTERISTICS
st.header("2. Customer Characteristics")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df["Age"], bins=15, ax=axes[0])
axes[0].set_title("Customer Age Distribution")

sns.countplot(data=df, x="Gender", ax=axes[1])
axes[1].set_title("Gender Distribution")

st.pyplot(fig)
plt.close(fig)

# FEATURE SELECTION
features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

data = df[features]

# SCALING
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# KMEANS FUNCTION
def run_kmeans(k):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(scaled_data)

    score = silhouette_score(scaled_data, labels)

    return model, labels, score


# K=3 AND K=4
model3, labels3, score3 = run_kmeans(3)
model4, labels4, score4 = run_kmeans(4)

# EXPERIMENT
st.header("3. K-Means Clustering Experiment")

selected_k = st.select_slider(
    "Change Number of Clusters (K)",
    options=[3, 4],
    value=3
)

if selected_k == 3:
    selected_model = model3
    selected_labels = labels3
    selected_score = score3
else:
    selected_model = model4
    selected_labels = labels4
    selected_score = score4

st.success(f"Currently Selected Clusters: K = {selected_k}")

result = df.copy()
result["Cluster"] = selected_labels

st.metric("Silhouette Score", round(selected_score, 3))

# CLUSTER VISUALIZATION
st.header(f"4. Customer Segmentation (K={selected_k})")

fig, ax = plt.subplots(figsize=(9, 5))

sns.scatterplot(
    data=result,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=80,
    ax=ax
)

ax.set_title(f"Customer Groups - K={selected_k}")

st.pyplot(fig)
plt.close(fig)

# CUSTOMER GROUP ANALYSIS
st.header("5. Customer Group Characteristics")

summary = result.groupby("Cluster").agg(
    Total_Customers=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending=("Spending Score (1-100)", "mean")
).round(2)

st.dataframe(summary, width="stretch")

# K=3 VS K=4 COMPARISON
st.header("6. K=3 vs K=4 Comparison")

col1, col2 = st.columns(2)

with col1:
    st.subheader("K = 3")
    st.metric("Silhouette Score", round(score3, 3))
    st.write("Number of Customer Groups: 3")

with col2:
    st.subheader("K = 4")
    st.metric("Silhouette Score", round(score4, 3))
    st.write("Number of Customer Groups: 4")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for k, labels, ax in [
    (3, labels3, axes[0]),
    (4, labels4, axes[1])
]:

    sns.scatterplot(
        x=df["Annual Income (k$)"],
        y=df["Spending Score (1-100)"],
        hue=labels,
        palette="viridis",
        s=60,
        ax=ax
    )

    ax.set_title(f"Customer Segmentation K={k}")

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# PURCHASING BEHAVIOUR
st.header("7. Purchasing Behaviour Analysis")

fig, ax = plt.subplots(figsize=(9, 5))

sns.boxplot(
    data=result,
    x="Cluster",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    legend=False,
    ax=ax
)

ax.set_title("Spending Behaviour Across Customer Groups")

st.pyplot(fig)
plt.close(fig)

# FINAL INFORMATION
st.divider()

st.subheader("Project Information")

st.write("Project: Customer Segmentation Using Machine Learning")
st.write("Algorithm: K-Means Clustering")
st.write("Language: Python")
st.write("Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit")

st.success("Customer Segmentation Experiment Completed Successfully!")