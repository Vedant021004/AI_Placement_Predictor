# 📊 Linear Regression vs Logistic Regression (Scikit-learn)

This repository demonstrates the complete workflow of **Linear Regression** and **Logistic Regression** using **Scikit-learn**. It is designed for beginners who want to understand the similarities and differences between these two supervised machine learning algorithms.

---

# 📚 Algorithms Covered

- Linear Regression
- Logistic Regression

---

# 🛠️ Libraries Used

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# Linear Regression
from sklearn.linear_model import LinearRegression

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Linear Regression Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Logistic Regression Metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
```

---

# 📂 Datasets

## Linear Regression

Example Dataset:
- House Price Prediction
- Salary Prediction
- Startup Profit Prediction

Target:
- Continuous Numerical Value

Example:

```
House Price = ₹25,50,000
```

---

## Logistic Regression

Dataset Used:
- Breast Cancer Dataset (`load_breast_cancer()`)

Target:

```
0 → Malignant

1 → Benign
```

---

# 🔄 Workflow Comparison

| Step | Linear Regression | Logistic Regression |
|------|-------------------|---------------------|
| Import Dataset | ✅ | ✅ |
| Create DataFrame | ✅ | ✅ |
| Create X and y | ✅ | ✅ |
| Train-Test Split | ✅ | ✅ |
| Create Model | `LinearRegression()` | `LogisticRegression()` |
| Train Model | `fit()` | `fit()` |
| Predict | `predict()` | `predict()` |
| Evaluate | MAE, MSE, R² | Accuracy, Precision, Recall |

---

# 🚀 Linear Regression Workflow

```python
X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 🚀 Logistic Regression Workflow

```python
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 📈 Evaluation Metrics

## Linear Regression

```python
mean_absolute_error()

mean_squared_error()

r2_score()
```

Measures:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

---

## Logistic Regression

```python
accuracy_score()

confusion_matrix()

classification_report()
```

Measures:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# 🔍 Key Differences

| Linear Regression | Logistic Regression |
|-------------------|---------------------|
| Predicts Continuous Values | Predicts Categories |
| Output = Number | Output = Class |
| Regression Problem | Classification Problem |
| Uses R² Score | Uses Accuracy |
| Example: House Price | Example: Cancer Detection |

---

# 🧠 Real-World Applications

## Linear Regression

- House Price Prediction
- Salary Prediction
- Sales Forecasting
- Stock Price Trend Analysis

---

## Logistic Regression

- Disease Prediction
- Spam Email Detection
- Fraud Detection
- Customer Churn Prediction
- Loan Approval Prediction

---

# 📖 What You'll Learn

- Data Preprocessing
- Feature Selection
- Train-Test Split
- Model Training
- Model Prediction
- Model Evaluation
- Comparison of Regression vs Classification

---

# 🎯 Learning Outcome

After completing this project, you will understand:

- When to use Linear Regression
- When to use Logistic Regression
- Similarities between both algorithms
- Differences in model evaluation
- Complete Scikit-learn workflow for supervised learning

---

## ⭐ If you found this project helpful, don't forget to star the repository!
