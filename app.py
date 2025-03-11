import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix

# Load data
cluster_data = pd.read_csv("clusters.csv")
model_results = pd.read_csv("supervised_results.csv")

st.title("🔍 Final Project: Customer Behavior Analysis")

# Tab 1: Unsupervised Results
with st.expander("📊 Unsupervised Clustering"):
    st.subheader("Customer Segments (KMeans)")
    fig = px.scatter(
        cluster_data, 
        x="total_spending", 
        y="order_frequency", 
        color="Cluster",
        title="Spending vs. Frequency by Cluster"
    )
    st.plotly_chart(fig)

    # Cluster Metrics Table
    st.subheader("Cluster Aggregations (Min/Avg/Max)")
    st.dataframe(
        cluster_data.groupby("Cluster").agg({
            "total_spending": ["min", "mean", "max"],
            "order_frequency": ["min", "mean", "max"],
            "discount_sensitivity": ["min", "mean", "max"]
        }),
        use_container_width=True
    )

# Tab 2: Supervised Results
with st.expander("🎯 Supervised Model Performance"):
    st.subheader("Classification Report")
    st.write(classification_report(model_results["actual"], model_results["predicted"]))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(model_results["actual"], model_results["predicted"])
    fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"))
    st.plotly_chart(fig)
