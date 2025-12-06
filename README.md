# 🏙️ District Sustainability Index (DSI) Prediction

![Banner](https://img.shields.io/badge/Project-TM271_DSI_Prediction-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Regression-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📋 Overview
An end-to-end machine learning system to predict the **District Sustainability Index (DSI)** based on environmental, demographic, and infrastructure indicators. This project was developed as part of the TM271 course under the GCC Sustainability Innovation Lab.

**Key Features:**
- ✅ Complete EDA with insightful visualizations
- ✅ Advanced missing value handling (Mean, KNN, Drop)
- ✅ Feature engineering & PCA dimensionality reduction
- ✅ Multiple ML models trained & evaluated
- ✅ Model generalization analysis & learning curves
- ✅ Best model saved for deployment

---

## 📁 Project Structure
TM271-DSI-Prediction/
│
├── notebooks/
│ └── TM271_DSI_Prediction.ipynb # Main Jupyter notebook
│
├── data/
│ └── tm271data.csv # Original dataset
│
├── models/
│ └── best_dsi_model.pkl # Saved best model
│
├── images/ # Generated visualizations
│ ├── correlation_heatmap.png
│ ├── co2_vs_dsi.png
│ ├── distributions.png
│ ├── pairplot.png
│ ├── variance_comparison.png
│ ├── pca_visualization.png
│ ├── feature_importance.png
│ ├── actual_vs_predicted.png
│ └── residual_plot.png
│
├── reports/
│ └── Project_Report_Arabic.pdf # Comprehensive project report
│
├── README.md # This file
└── requirements.txt # Python dependencies

text

---

## 📊 Dataset Description

**File:** `tm271data.csv`  
**Size:** 57.29 KB | **Rows:** 1000 | **Columns:** 9

### Features:
| Feature | Description | Type |
|---------|-------------|------|
| `district_id` | District identifier | Object |
| `district_name` | District name | Object |
| `CO2_emission_kilotons` | CO₂ emissions in kilotons | Numeric |
| `Average_energy_consumption_kWh_per_household` | Average household energy use | Numeric |
| `Green_area_per_capita_m2` | Green space per capita in m² | Numeric |
| `Waste_recycling_rate_pct` | Waste recycling rate (%) | Numeric |
| `Population_density_people_per_km2` | Population density | Numeric |
| `Traffic_index_0_100` | Traffic congestion index (0-100) | Numeric |
| `DSI_target_0_100` | **Target:** District Sustainability Index (0-100) | Numeric |

---

## 🔬 Methodology

### 1. **Exploratory Data Analysis (EDA)**
- Missing value analysis (5.4% missing overall)
- Correlation heatmap & pairplots
- Distribution histograms for all features
- **Key Findings:**
  - Strong correlation between CO₂ and energy consumption (0.73)
  - Green area positively impacts DSI (0.76 correlation)
  - Traffic congestion negatively affects sustainability (-0.81 correlation)

### 2. **Data Preprocessing**
- Removed non-predictive columns (`district_id`, `district_name`)
- **Missing value handling:** Mean imputation (preserves all records)
- **Feature engineering:**
  - `Energy_per_capita` = Energy consumption / Population density
  - `Green_Index` = Green area / Population density
- **Scaling:** Standardization (mean=0, std=1)
- **Dimensionality Reduction:** PCA (68.32% variance retained with 2 components)

### 3. **Modeling**
**Models Implemented:**
1. Linear Regression
2. Decision Tree Regressor
3. Support Vector Regressor (SVR)
4. Random Forest Regressor
5. Gradient Boosting Regressor

**Evaluation Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score (Coefficient of Determination)

### 4. **Results**
| Model | MAE | RMSE | R² Score | Rank |
|-------|-----|------|----------|------|
| **Linear Regression** | **2.7660** | **4.6315** | **0.8712** | 🥇 **Best** |
| Gradient Boosting | 3.0920 | 4.7271 | 0.8658 | 🥈 |
| Random Forest | 3.4556 | 4.8737 | 0.8574 | 🥉 |
| SVR | 3.7770 | 5.8297 | 0.7959 | 4 |
| Decision Tree | 5.1917 | 6.9041 | 0.7138 | 5 |

---

## 🎯 Key Insights

### 🏆 **Best Performing Model: Linear Regression**
- **MAE:** 2.7660 → Average prediction error < 3 DSI points
- **R²:** 0.8712 → 87% variance explained
- **Generalization:** Excellent (Train-Test gap: 0.0504)

### 🌿 **Sustainability Drivers Identified:**
1. **Energy Efficiency** directly reduces carbon footprint
2. **Green Spaces** significantly improve sustainability scores
3. **Traffic Management** is crucial for urban sustainability
4. **Population Density** requires balanced resource allocation

---

## 📈 Visualizations

| Visualization | Purpose |
|---------------|---------|
| ![Correlation Heatmap](images/correlation_heatmap.png) | Relationships between features |
| ![CO2 vs DSI](images/co2_vs_dsi.png) | Negative impact of emissions on sustainability |
| ![PCA](images/pca_visualization.png) | Data structure in reduced dimensions |
| ![Actual vs Predicted](images/actual_vs_predicted.png) | Model prediction accuracy |

*(All visualizations are available in the `images/` folder)*

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook/Lab

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/TM271-DSI-Prediction.git
cd TM271-DSI-Prediction

# Install dependencies
pip install -r requirements.txt
Run the Project
Open notebooks/TM271_DSI_Prediction.ipynb in Jupyter

Run all cells sequentially

Check generated visualizations in images/ folder

Requirements
txt
pandas==2.0.0
numpy==1.24.0
matplotlib==3.7.0
seaborn==0.12.0
scikit-learn==1.3.0
jupyter==1.0.0
joblib==1.2.0
📝 Usage Example
python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('models/best_dsi_model.pkl')

# Prepare new data (example)
new_data = pd.DataFrame({
    'CO2_emission_kilotons': [500.0],
    'Average_energy_consumption_kWh_per_household': [8000.0],
    'Green_area_per_capita_m2': [30.0],
    'Waste_recycling_rate_pct': [40.0],
    'Population_density_people_per_km2': [600.0],
    'Traffic_index_0_100': [50.0],
    'Energy_per_capita': [13.33],
    'Green_Index': [0.05]
})

# Predict DSI score
predicted_dsi = model.predict(new_data)
print(f"Predicted DSI Score: {predicted_dsi[0]:.2f}")
📄 Project Report
A comprehensive Arabic project report is available in reports/Project_Report_Arabic.pdf including:

Complete methodology

Detailed results analysis

Future recommendations

Academic context

🔮 Future Improvements
Model Enhancement

Implement XGBoost/LightGBM

Hyperparameter tuning with Optuna

Ensemble methods stacking

Feature Expansion

Air quality indicators

Public transportation metrics

Renewable energy usage data

Deployment

Web application with Streamlit

REST API using FastAPI

Real-time monitoring dashboard

👥 Contributors
Nahla Nabil Skaik (@nahla-skaik) - Main Developer

Dr. Khalid Mansour - Project Supervisor

Institution: GCC Sustainability Innovation Lab
Course: TM271 - Machine Learning Project

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
GCC Sustainability Innovation Lab for dataset provision

Dr. Khalid Mansour for academic supervision

Open-source ML community for invaluable resources

📞 Contact
For questions or collaboration:

Email: your.nahla.nabil.52@gmail.com 

LinkedIn: Nahla Nabil

GitHub: @Nahla-Nabil
⭐ Show Your Support
If you find this project useful, please give it a star! ⭐

Last Updated: December 
