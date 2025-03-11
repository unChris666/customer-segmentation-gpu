# 🚀 Final Project: Supervised & Unsupervised Learning on Tabular Data

## 📌 Overview
This project combines **KMeans clustering** (unsupervised) and **classification/regression models** (supervised) to analyze customer behavior. Built for a learning evaluation, it highlights GPU-accelerated data processing and actionable business insights.

## 🔥 Features
- **Unsupervised Learning**: Customer segmentation into 4 clusters (B2B/B2C).
- **Supervised Learning**: 
  - Classification: Predict customer type (B2B/B2C) using Random Forest.
  - Regression: Predict total spending using XGBoost.
- **GPU Acceleration**: 10x faster data processing with `cuDF` and `RAPIDS`.
- **Interactive Dashboard**: Built with Streamlit for visualization.

## 📊 Results
| Metric              | Unsupervised (KMeans) | Supervised (Random Forest) |
|---------------------|-----------------------|----------------------------|
| **Silhouette Score**| 0.62                  | -                          |
| **Accuracy**        | -                     | 92%                        |
| **F1-Score**        | -                     | 0.89                       |

![Dashboard](https://your-dashboard-screenshot.png)

## 🛠️ Installation
```bash
git clone https://github.com/your-username/customer-analysis-final-project.git
pip install -r requirements.txt  # Includes cuDF, scikit-learn, streamlit, xgboost
streamlit run app.py
