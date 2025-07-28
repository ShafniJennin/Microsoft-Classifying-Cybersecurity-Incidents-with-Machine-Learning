# Microsoft: Classifying Cybersecurity Incidents with Machine Learning

## 🚀 Project Overview

This project focuses on building a machine learning model to classify cybersecurity incidents as **True Positive (TP)**, **Benign Positive (BP)**, or **False Positive (FP)** using the GUIDE dataset. This model enhances the efficiency of **Security Operation Centers (SOCs)** by enabling accurate, automated triage of incident alerts.

---

## 🎯 Problem Statement

You are working as a Data Scientist at Microsoft to:
- Automate incident triage classification for SOCs.
- Predict the incident grade using machine learning models.
- Provide actionable recommendations by supporting guided response systems.

---

## 💼 Business Use Cases

- **SOCs Automation** – Speed up triage processes to focus on real threats.
- **Incident Response** – Enable quick decision-making using model output.
- **Threat Intelligence** – Incorporate historical responses for precision.
- **Enterprise Security** – Improve security posture by reducing false positives.

---

## 🧠 Skills Applied

- Data Cleaning & Preprocessing
- Feature Engineering
- Machine Learning Classification (Logistic Regression, XGBoost, etc.)
- Model Evaluation (Macro-F1 Score, Precision, Recall)
- Handling Imbalanced Datasets
- MITRE ATT&CK Cybersecurity Concepts

---

## 🧰 Technologies Used

- Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM
- Matplotlib, Seaborn
- Jupyter Notebooks
- GUIDE Dataset (CSV)
- Git & GitHub

---

## 📊 Dataset Overview

- **Source**: GUIDE Dataset (Provided by Microsoft)
- **Format**: CSV (`train.csv`, `test.csv`)
- **Classes**: TP (True Positive), BP (Benign Positive), FP (False Positive)
- **Levels**: Evidence → Alert → Incident
- **Features**: 45 structured fields (categorical, numerical, metadata)

---

## 🧪 Project Approach

### 1. Data Understanding & EDA
- Analyze class imbalance
- Visualize feature distributions
- Inspect correlations and anomalies

### 2. Data Preprocessing
- Handle missing values
- Encode categorical features (One-Hot, Label)
- Normalize numerical columns
- Feature engineering (timestamp breakdowns, new flags, etc.)

### 3. Model Training
- Baseline: Logistic Regression
- Advanced Models: XGBoost, LightGBM, Random Forest
- Imbalanced handling: Class Weights, SMOTE
- Hyperparameter tuning using GridSearchCV / RandomizedSearchCV

### 4. Evaluation
- Metrics: **Macro-F1 Score**, **Precision**, **Recall**
- Cross-validation to avoid overfitting
- Feature importance using SHAP, permutation

---

## 🧾 Evaluation Metrics

| Metric       | Description                            |
|--------------|----------------------------------------|
| **Macro-F1** | Balanced performance across TP, BP, FP |
| **Precision**| Minimize false alarms                  |
| **Recall**   | Maximize detection of true threats     |

---
---

## 📌 Project Results

- Developed and tuned a classifier achieving **macro-F1 > 0.80**
- Reduced false positives by over 25% vs baseline
- Identified most influential features for each triage class
- Built reusable preprocessing and training pipeline

---

## 📦 Deliverables

- ✅ Cleaned and structured dataset
- ✅ Reproducible source code for preprocessing, modeling, evaluation
- ✅ Trained model file (`best_model.pkl`)
- ✅ Evaluation metrics (Macro-F1, Precision, Recall)
- ✅ Technical documentation and final presentation

---

## 🔍 Future Improvements

- Integrate model into a SOC dashboard (Streamlit/FastAPI)
- Use ensemble learning or deep learning (LSTM/Transformer)
- Real-time streaming with Apache Kafka
- Link predictions with MITRE ATT&CK framework responses
