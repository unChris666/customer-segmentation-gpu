import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_data  # Cache the data for faster loading
def load_data():
    return pd.read_csv("Dataset_clustering.csv")

df = load_data()

# Title of the app
st.title("Customer Clustering Visualization")

# Sidebar for user inputs
st.sidebar.header("Filters")
cluster_filter = st.sidebar.multiselect("Select Clusters", df["Cluster"].unique())

# Filter data based on user selection
if cluster_filter:
    df_filtered = df[df["Cluster"].isin(cluster_filter)]
else:
    df_filtered = df

# Display the dataset
st.subheader("Filtered Dataset")
st.write(df_filtered)

# Scatter Plot: Total Spending vs Order Frequency
st.subheader("Scatter Plot: Total Spending vs Order Frequency")
fig, ax = plt.subplots()
sns.scatterplot(data=df_filtered, x="total_spending", y="order_frequency", hue="Cluster", ax=ax)
st.pyplot(fig)

# Stacked Bar Chart: Beverage Preferences by Cluster
st.subheader("Beverage Preferences by Cluster")
beverage_cols = ["Alcoholic Beverages_pct", "Juices_pct", "Soft Drinks_pct", "Water_pct"]
beverage_means = df_filtered.groupby("Cluster")[beverage_cols].mean().T
st.bar_chart(beverage_means)

# Regional Distribution by Cluster
st.subheader("Regional Distribution by Cluster")
region_counts = df_filtered.groupby(["Region", "Cluster"]).size().unstack()
st.bar_chart(region_counts)

# Discount Sensitivity Comparison (Box Plot)
st.subheader("Discount Sensitivity by Cluster")
fig, ax = plt.subplots()
sns.boxplot(data=df_filtered, x="Cluster", y="discount_sensitivity", ax=ax)
st.pyplot(fig)
