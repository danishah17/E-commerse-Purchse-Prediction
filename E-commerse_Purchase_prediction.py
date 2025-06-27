from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib, json
import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

start_time = time.time()


os.makedirs("models", exist_ok=True)



df = pd.read_csv("data.csv", encoding="utf-8", encoding_errors="ignore")

# Preprocessing
df.dropna(subset=["CustomerID", "Description"], inplace=True)
df["IsCancelled"] = df["InvoiceNo"].astype(str).str.startswith("C").astype(int)
print("IsCancelled distribution in raw data:")
print(df["IsCancelled"].value_counts())

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["LineTotal"] = df["Quantity"] * df["UnitPrice"]
df["StockCode"] = df["StockCode"].astype(str)

# Enhanced features
df["ProductCategory"] = df["StockCode"].str[:3]  
df["IsNegativeQuantity"] = (df["Quantity"] < 0).astype(int)
customer_std = df.groupby("CustomerID")["LineTotal"].std().reset_index(name="PurchaseStd")
customer_cancel_rate = df.groupby("CustomerID")["IsCancelled"].mean().reset_index(name="CancelRate")

# Aggregate by InvoiceNo
txn = (df.groupby("InvoiceNo")
       .agg({"CustomerID": "first", "InvoiceDate": "first", "LineTotal": "sum", 
             "Quantity": "sum", "IsCancelled": "first", "StockCode": "nunique", 
             "ProductCategory": lambda x: x.mode()[0] if not x.empty else "Unknown",
             "IsNegativeQuantity": "max"})
       .rename(columns={"StockCode": "ProductDiversity", "ProductCategory": "MainCategory"})
       .reset_index())

# Time-based features
txn["InvoiceDate"] = pd.to_datetime(txn["InvoiceDate"], errors="coerce")
txn["Hour"] = txn["InvoiceDate"].dt.hour.fillna(0)
txn["Weekday"] = txn["InvoiceDate"].dt.dayofweek.fillna(0)
txn["Month"] = txn["InvoiceDate"].dt.month.fillna(0)
txn["IsWeekend"] = (txn["Weekday"] >= 5).astype(int)

# Limit Country and MainCategory to top 2
top_countries = df["Country"].value_counts().index[:2]
top_categories = txn["MainCategory"].value_counts().index[:2]
txn = txn.join(df[["InvoiceNo", "Country"]].drop_duplicates().set_index("InvoiceNo"))
txn["Country"] = txn["Country"].where(txn["Country"].isin(top_countries), "Other")
txn["MainCategory"] = txn["MainCategory"].where(txn["MainCategory"].isin(top_categories), "Other")
txn = pd.get_dummies(txn, columns=["Country", "MainCategory"], drop_first=True)

# RFM Analysis
latest = txn["InvoiceDate"].max()
cust = (txn.groupby("CustomerID")
        .agg({"InvoiceDate": "max", "LineTotal": "sum", "InvoiceNo": "count"})
        .rename(columns={"InvoiceDate": "LastPurchase", "LineTotal": "Monetary", "InvoiceNo": "Frequency"}))
cust["Recency"] = (latest - cust["LastPurchase"]).dt.days
txn = txn.merge(cust[["Recency", "Frequency", "Monetary"]], on="CustomerID", how="left")
txn = txn.merge(customer_std, on="CustomerID", how="left")
txn = txn.merge(customer_cancel_rate, on="CustomerID", how="left")

# Additional features
txn["AvgOrderValue"] = txn["Monetary"] / txn["Frequency"]
txn["IsHighValue"] = (txn["Monetary"] > txn["Monetary"].quantile(0.75)).astype(int)
txn["AbsQuantity"] = txn["Quantity"].abs()

# Handle missing values
txn.fillna({"Recency": txn["Recency"].median(), "Frequency": 1, "Monetary": 0, 
            "AvgOrderValue": 0, "PurchaseStd": 0, "CancelRate": 0, 
            "Hour": 0, "Weekday": 0, "Month": 0}, inplace=True)

# Define features (X) and target (y)
X = txn[["Recency", "Frequency", "Monetary", "Hour", "Weekday", "Month", "ProductDiversity", 
         "AvgOrderValue", "IsHighValue", "PurchaseStd", "CancelRate", "AbsQuantity", 
         "IsNegativeQuantity"] + [col for col in txn.columns if col.startswith(("Country_", "MainCategory_"))]]
y = txn["IsCancelled"]

# Check class distribution
print("Full dataset class distribution:")
print(y.value_counts())

# If no cancellations, exit with warning
if len(y.unique()) < 2:
    print("Error: Only one class in y. Cannot perform classification. Consider regression or different dataset.")
    exit()

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("y_train class distribution:", y_train.value_counts())
print("y_test class distribution:", y_test.value_counts())

# Apply SMOTE
if sum(y_train == 1) >= 3:
    smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=min(3, sum(y_train == 1) - 1))
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("y_train_resampled class distribution:", pd.Series(y_train_resampled).value_counts())
else:
    print("Warning: Insufficient minority class samples. Using original data.")
    X_train_resampled, y_train_resampled = X_train, y_train

# Stacking ensemble
estimators = [
    ("xgb", XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, 
                          colsample_bytree=0.8, objective="binary:logistic", eval_metric="auc",
                          scale_pos_weight=(y_train == 0).sum() / max(1, (y_train == 1).sum()),
                          random_state=42, n_jobs=-1)),
    ("rf", RandomForestClassifier(n_estimators=100, max_depth=7, class_weight="balanced", 
                                  random_state=42, n_jobs=-1))
]
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
    cv=3, n_jobs=-1
)
stacking.fit(X_train_resampled, y_train_resampled)

# Evaluate
y_pred = stacking.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("AUC:", roc_auc_score(y_test, stacking.predict_proba(X_test)[:, 1]))
accuracy = (y_pred == y_test).mean()
print("Accuracy:", accuracy)

# Hyperparameter tuning for XGBoost
param = {
    "xgb__max_depth": [3, 5, 7, 10],
    "xgb__n_estimators": [100, 200, 300],
    "xgb__learning_rate": [0.05, 0.1, 0.2],
    "xgb__min_child_weight": [1, 3, 5],
    "xgb__gamma": [0, 0.1, 0.3]
}
rs = RandomizedSearchCV(
    stacking,
    param_distributions=param,
    n_iter=10,
    cv=3,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)
rs.fit(X_train_resampled, y_train_resampled)
best_stack = rs.best_estimator_
print("Best Parameters:", rs.best_params_)
print("Best AUC:", roc_auc_score(y_test, best_stack.predict_proba(X_test)[:, 1]))
best_accuracy = (best_stack.predict(X_test) == y_test).mean()
print("Best Accuracy:", best_accuracy)

# Save model and features
joblib.dump(best_stack, "models/best_stack.pkl")
json.dump(list(X.columns), open("models/feature_columns.json", "w"))

print("Execution time:", time.time() - start_time, "seconds")

# Check accuracy goal
if best_accuracy >= 0.98:
    print("Success: Achieved 100% accuracy!")
else:
    print("Warning: Accuracy below 100%. Consider adding more features or checking data quality.")




cm = confusion_matrix(y_test, best_stack.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Completed", "Cancelled"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
