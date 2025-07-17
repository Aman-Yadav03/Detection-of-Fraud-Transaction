# 🕵️‍♀️ Fraud Detection Using Machine Learning

This repository contains a comprehensive machine learning pipeline to detect fraudulent transactions using traditional classification models. The dataset includes millions of transaction records, and this project explores preprocessing, balancing, modeling, and evaluation strategies to tackle class imbalance and improve fraud detection performance.

---

## 📁 Dataset

- The dataset used is a financial transactions dataset (`Fraud.csv`)
- Total records: 6,362,620
- Target column: `isFraud` (binary classification)

---

## 🚀 Workflow Summary

The pipeline includes the following key steps:

1. **Data Exploration**
   - Understand distribution using `.describe()`, `.head()`, and correlation matrix
   - Check for null values

2. **Data Cleaning**
   - Dropped unnecessary columns (`oldbalanceOrg`, `oldbalanceDest`) due to multicollinearity
   - Handled outliers using IQR-based filtering (percentages printed per feature)

3. **Feature Engineering**
   - Converted object columns (`type`, `nameOrig`, `nameDest`) to numeric using Label Encoding

4. **Handling Class Imbalance**
   - Applied `SMOTE` oversampling to balance classes (resulting in 1:1 fraud-to-non-fraud ratio)

5. **Train-Test Split**
   - 70% training, 30% testing using `train_test_split()`

6. **Feature Scaling**
   - Standardized features using `StandardScaler`

---

## 🧠 Models Trained

| Model                   | Accuracy (Approx.) |
|------------------------|--------------------|
| Logistic Regression     | ✅ Measured         |
| SVM (Linear & RBF)      | ✅ Measured         |
| Gaussian Naive Bayes    | ✅ Measured         |
| Decision Tree           | ✅ Measured         |
| Random Forest (n=50-200)| ✅ Measured & Tuned |

Each model was trained and evaluated using accuracy, confusion matrices, and classification reports.

---

## 📊 Evaluation & Cross-Validation

- Used `cross_val_score` and `cross_val_predict` to evaluate:
  - **Random Forest**
  - **Decision Tree**

- Plotted **Confusion Matrices** side-by-side
- Displayed **Classification Reports** for precision, recall, and F1-score

---

## 🔍 Insights

- `amount`, `newbalanceOrig`, and `newbalanceDest` contain significant outliers
- Categorical variables like `type` are strongly correlated with fraud
- SMOTE balancing improves model fairness but can increase training time
- Random Forest showed strong performance after tuning

---

## 📦 Requirements

Install the required Python libraries with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost