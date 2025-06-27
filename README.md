# E-commerse-Purchse-Prediction
A machine learning pipeline to predict whether an e-commerce order will be completed or cancelled using transaction-level features. Includes a Streamlit app for real-time predictions.
# 🛒 E-Commerce Purchase Prediction

Predict whether a retail transaction will end in **completion or cancellation** using historical e-commerce data.

---

## 📦 Dataset

- **Source**: [Kaggle - Online Retail Dataset](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- 500,000+ transactions from a UK-based online retailer (2010–2011)

---

## 🎯 Objective

Build a machine learning pipeline to:
- Analyze patterns in cancelled vs. completed orders
- Engineer customer, product, and time-based features
- Train a **stacked ensemble model (XGBoost + Random Forest)**
- Deploy a prediction UI using **Streamlit**

---

## 🔍 Features Used

| Feature            | Description                                 |
|--------------------|---------------------------------------------|
| Recency            | Days since customer’s last purchase         |
| Frequency          | Total number of past purchases              |
| Monetary           | Total money spent                           |
| AvgOrderValue      | Average spend per order                     |
| PurchaseStd        | Std deviation in past spending              |
| CancelRate         | Historical cancellation rate                |
| Hour, Weekday, Month | Time of purchase                          |
| ProductDiversity   | Unique products in invoice                  |
| IsNegativeQuantity | Return flag                                 |
| IsHighValue        | Top quartile order spender flag             |
| AbsQuantity        | Absolute quantity purchased                 |
| One-hot: Country_, MainCategory_ | Encoded categorical info     |

---

## 📊 Model Performance

- **Model**: `StackingClassifier (XGBoost + RF)`
- **Evaluation Metrics**:
  - Accuracy: **100%**
  - AUC: **1.00**
  - F1-Score: **1.00**

![Confusion Matrix](confusion%20matrix.png)

✅ The model classifies both completed and cancelled orders **perfectly** on the test set.
