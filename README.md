# 🌿 District Sustainability Index (DSI) Prediction System

A complete end-to-end **Machine Learning pipeline** for predicting the  
**District Sustainability Index (DSI)** using environmental, demographic, and urban indicators.

---

## 📌 Overview

This project develops a regression-based ML system that predicts the sustainability score (0–100) for urban districts.  
The workflow covers:

- Data cleaning & preprocessing  
- Exploratory analysis  
- Feature engineering  
- PCA dimensionality reduction  
- Model training & evaluation  
- Visualization & interpretation  

The notebook was developed as part of **TM271 – Machine Learning & Deep Learning** at the **Arab Open University**.

---

## 🧠 Key Results

| Metric | Value |
|-------|-------|
| **Best Model** | Linear Regression |
| **Test R²** | **0.871** |
| **MAE** | 2.77 |
| **RMSE** | 4.63 |
| **Dataset Size** | 1,000 districts |

The model achieved **87% predictive accuracy**, demonstrating strong generalization.

---

## 🔧 Features

- Full preprocessing pipeline  
- Missing value handling  
- Derived features (`Energy_per_capita`, `Green_Index`)  
- Scaling (Standardization)  
- PCA dimensionality reduction  
- 5 ML models compared  
- Over/underfitting analysis  
- Learning curves  
- Residual and prediction error plots  

---

## 📊 Dataset

The dataset includes environmental and demographic indicators such as:

- CO₂ emissions  
- Household energy consumption  
- Green area per capita  
- Waste recycling rate  
- Traffic index  
- Population density  
- DSI target score  

---

## 🧪 Methodology

1. Data Cleaning & Integrity Check  
2. Exploratory Visualization  
3. Feature Engineering  
4. Scaling (Standardization)  
5. PCA dimensionality reduction  
6. Model Training (5 regressors)  
7. Evaluation (MAE, RMSE, R²)  
8. Residual analysis + actual vs. predicted  

---

## 🏆 Model Performance

| Model | Test R² | RMSE |
|-------|--------|------|
| **Linear Regression** | **0.871** | **4.63** |
| Gradient Boosting | 0.866 | 4.72 |
| Random Forest | 0.857 | 4.87 |
| SVR | 0.795 | 5.83 |
| Decision Tree | 0.713 | 6.90 |

Linear Regression offered the best balance of accuracy, stability, and simplicity.

---

## 🖼️ Visualizations Included

- Correlation heatmap  
- Distribution plots  
- Pairplot analysis  
- PCA (2D components)  
- Actual vs. Predicted  
- Residual plot  
- Learning curve  

---

## ▶️ Usage

To run the notebook:

```bash
jupyter notebook TM271_DSI_Prediction.ipynb


---

### ✨ Author

**Nahla Nabil Skaik**  
Artificial Intelligence Student – Arab Open University, Bahrain  
**TM271 – Machine Learning & Deep Learning**  
**Semester:** Fall 2025/2026  
**Student ID:** 6230202  

---

### 📫 Contact

- 📧 Email: *nahla.nabil.52@gmail.com *  
- 💼 LinkedIn: https://www.linkedin.com/in/nahla-nabil-876597211/
- 🐙 GitHub: Nahla-Nabil  

---

### 🔖 Supervisor  
**Dr. Khalid Mansour**  
Arab Open University – Bahrain  

---

### 🎓 Academic Note  
This project was completed as part of the official TM271 coursework and follows academic integrity guidelines.

---
