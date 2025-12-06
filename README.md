# 🌿 District Sustainability Index (DSI) Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An End-to-End Machine Learning System for Predicting Urban Sustainability**  
> Built for the GCC Sustainability Innovation Lab under Oman Vision 2040

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Features](#-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Visualizations](#-visualizations)
- [Key Insights](#-key-insights)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project develops a comprehensive **Machine Learning pipeline** to predict the **District Sustainability Index (DSI)** — a composite score (0-100) that quantifies the environmental and social well-being of urban districts.

### 🌍 Problem Context

The GCC Sustainability Innovation Lab (GCC-SIL), established under **Oman Vision 2040**, aims to leverage AI to advance the UN Sustainable Development Goals (SDGs). This system helps policymakers:

- 🎯 Identify districts requiring urgent environmental interventions
- 💰 Allocate sustainability resources efficiently
- 📊 Monitor progress toward national sustainability goals
- 🏙️ Make data-driven decisions for urban planning

---

## 🏆 Key Results

| Metric | Value |
|--------|-------|
| **Best Model** | Linear Regression |
| **R² Score** | 0.871 (87.1% accuracy) |
| **MAE** | ±2.77 DSI points |
| **RMSE** | 4.63 |
| **Dataset Size** | 1,000 districts |

> **Bottom Line:** The model can predict district sustainability with **87% accuracy**, making it suitable for real-world policy decision support.

---

## ✨ Features

### 🔧 Data Preprocessing
- ✅ Intelligent missing value handling (3 methods compared)
- ✅ Feature engineering (Energy per Capita, Green Index)
- ✅ Multiple scaling techniques (Min-Max, Standardization)
- ✅ Dimensionality reduction with PCA

### 🤖 Machine Learning
- ✅ 5 algorithms compared (Linear Regression, Decision Tree, SVR, Random Forest, Gradient Boosting)
- ✅ Regularization to prevent overfitting
- ✅ Cross-validation and learning curve analysis
- ✅ Feature importance interpretation

### 📊 Visualization & Reporting
- ✅ 10+ professional visualizations
- ✅ Comprehensive EDA with correlation analysis
- ✅ Model performance comparison charts
- ✅ Residual analysis and prediction confidence

---

## 📊 Dataset

### Features

| Feature | Description | Type |
|---------|-------------|------|
| **CO₂ Emissions** | Annual carbon output (kilotons) | Continuous |
| **Energy Consumption** | Avg. household energy (kWh) | Continuous |
| **Green Area** | Public green space per capita (m²) | Continuous |
| **Waste Recycling Rate** | % of waste recycled | Continuous (%) |
| **Population Density** | People per km² | Continuous |
| **Traffic Index** | Congestion score (0-100) | Continuous |
| **DSI (Target)** | District Sustainability Index | Continuous (0-100) |

### Data Quality
- **Total Records:** 1,000 districts
- **Missing Data:** ~5.4% (handled via imputation)
- **Duplicate Rows:** 20 (removed)
- **Final Clean Dataset:** 1,000 rows × 7 features

---

## 🔬 Methodology

### 1️⃣ Data Preprocessing Pipeline

```python
Raw Data (1,020 rows)
    ↓
Remove Duplicates (→ 1,000 rows)
    ↓
Handle Missing Values (Mean Imputation: 5.4% → 0%)
    ↓
Feature Engineering (Create: Energy_per_capita, Green_Index)
    ↓
Feature Scaling (Standardization: mean=0, std=1)
    ↓
Train/Test Split (80% / 20%)
    ↓
Ready for Model Training ✅
```

### 2️⃣ Model Training

**Models Evaluated:**
1. **Linear Regression** (Baseline) ⭐ Best Performance
2. **Decision Tree** (Non-linear)
3. **Support Vector Regressor** (Kernel-based)
4. **Random Forest** (Ensemble)
5. **Gradient Boosting** (Sequential Ensemble)

**Evaluation Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score (Coefficient of Determination)

---

## 📈 Model Performance

### Comparison Table

| Model | MAE ↓ | RMSE ↓ | R² Score ↑ | Training Time | Overfitting Risk |
|-------|-------|--------|------------|---------------|------------------|
| **Linear Regression** ⭐ | **2.77** | **4.63** | **0.871** | ⚡ Fast | ✅ Low |
| Gradient Boosting | 3.09 | 4.73 | 0.866 | 🐢 Slow | ⚠️ Medium |
| Random Forest | 3.46 | 4.87 | 0.857 | 🐢 Slow | ✅ Low |
| SVR | 3.78 | 5.83 | 0.796 | ⏱️ Moderate | ✅ Low |
| Decision Tree | 5.19 | 6.90 | 0.714 | ⚡ Fast | ⚠️ Medium |

### Why Linear Regression Won?

1. **Strong Linear Relationships:** Features naturally correlate linearly with DSI
2. **No Overfitting:** Train R² = 0.922, Test R² = 0.871 (only 5% gap)
3. **Interpretability:** Coefficients clearly show feature impact
4. **Speed:** Fast training and real-time predictions

### Overfitting Detection

```
Linear Regression:
  Train R²: 0.922  |  Test R²: 0.871  |  Gap: 0.051
  ✅ Status: GOOD FIT (perfect generalization)

Gradient Boosting:
  Train R²: 0.959  |  Test R²: 0.866  |  Gap: 0.093
  ⚠️ Status: SLIGHT OVERFITTING (but acceptable)
```

---

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Clone Repository

```bash
git clone https://github.com/yourusername/DSI-Prediction-System.git
cd DSI-Prediction-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.3.0
jupyter>=1.0.0
joblib>=1.3.0
```

---

## 🚀 Usage

### Option 1: Run Jupyter Notebook

```bash
jupyter notebook TM271_DSI_Prediction.ipynb
```

Then execute cells sequentially to:
1. Load and explore the dataset
2. Preprocess and engineer features
3. Train and compare models
4. Visualize results

### Option 2: Use Pre-trained Model

```python
import joblib
import pandas as pd

# Load saved model
model = joblib.load('models/best_dsi_model.pkl')

# Prepare new district data (must be standardized)
new_district = pd.DataFrame({
    'CO2_emission_kilotons': [650],
    'Average_energy_consumption_kWh_per_household': [8500],
    'Green_area_per_capita_m2': [30],
    'Waste_recycling_rate_pct': [40],
    'Population_density_people_per_km2': [600],
    'Traffic_index_0_100': [55],
    'Energy_per_capita': [25],
    'Green_Index': [0.05]
})

# Make prediction
predicted_dsi = model.predict(new_district)
print(f"Predicted DSI Score: {predicted_dsi[0]:.2f}")
```

---

## 📁 Project Structure

```
DSI-Prediction-System/
│
├── 📓 notebooks/
│   └── TM271_DSI_Prediction.ipynb          # Main Jupyter Notebook
│
├── 📊 data/
│   └── tm271data.csv                        # Original dataset
│
├── 🤖 models/
│   └── best_dsi_model.pkl                   # Saved Linear Regression model
│
├── 📈 visualizations/
│   ├── plot1_correlation_heatmap.png
│   ├── plot2_co2_vs_dsi.png
│   ├── plot3_distributions.png
│   ├── plot4_pairplot.png
│   ├── variance_comparison.png
│   ├── derived_features_vs_dsi.png
│   ├── pca_visualization.png
│   ├── pca_loadings.png
│   ├── actual_vs_predicted.png
│   └── residual_plot.png
│
├── 📄 reports/
│   └── TM271_Final_Report.pdf               # Detailed technical report
│
├── 📋 requirements.txt                       # Python dependencies
├── 📖 README.md                              # This file
└── 📜 LICENSE                                # MIT License

```

---

## 🎨 Visualizations

### Correlation Heatmap
![Correlation Heatmap](visualizations/plot1_correlation_heatmap.png)

### CO₂ Emissions vs DSI
![CO2 vs DSI](visualizations/plot2_co2_vs_dsi.png)

### PCA Dimensionality Reduction
![PCA Visualization](visualizations/pca_visualization.png)

### Actual vs Predicted
![Actual vs Predicted](visualizations/actual_vs_predicted.png)

### Residual Analysis
![Residual Plot](visualizations/residual_plot.png)

---

## 💡 Key Insights

### 1. CO₂ and Energy Drive Sustainability
- **Finding:** CO₂ emissions (r = -0.73) and energy consumption (r = -0.65) are the strongest negative predictors of DSI
- **Implication:** Carbon reduction and energy efficiency programs should be top priorities

### 2. Green Space Matters
- **Finding:** Green area per capita shows strong positive correlation with DSI (r = +0.76)
- **Implication:** Urban greening projects directly improve sustainability scores

### 3. Traffic Congestion is a Major Barrier
- **Finding:** Traffic index negatively correlates with DSI (r = -0.81)
- **Implication:** Investing in public transportation can significantly boost sustainability

### 4. Linear Models Work Best
- **Finding:** Simple Linear Regression outperformed complex ensemble methods
- **Implication:** Sustainability relationships are predominantly linear and interpretable

---

## 🔮 Future Work

### Short-term Improvements
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Test advanced algorithms (XGBoost, LightGBM)
- [ ] Implement ensemble stacking
- [ ] Add confidence intervals to predictions

### Long-term Extensions
- [ ] Incorporate socioeconomic features (income, education, healthcare)
- [ ] Add temporal dimension (track DSI changes over time)
- [ ] Integrate geospatial analysis (GIS mapping)
- [ ] Include climate variables (temperature, rainfall, air quality)
- [ ] Build interactive dashboard for policymakers
- [ ] Deploy as REST API for real-time predictions

### Research Directions
- [ ] Study causal relationships (not just correlations)
- [ ] Analyze policy intervention impacts
- [ ] Extend to other GCC countries
- [ ] Compare urban vs. rural sustainability patterns

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Update README if adding new features
- Include unit tests for new functionality

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Academic Use
This project was developed as part of the **TM271 - Machine Learning and Deep Learning** course. If you use this code for academic purposes, please cite:

```bibtex
@misc{skaik2025dsi,
  author = {Skaik, Nahla Nabil},
  title = {District Sustainability Index Prediction System},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/DSI-Prediction-System}}
}
```

---

## 👤 Author

**Nahla Nabil Skaik**

- 🎓 Student ID: 6230202
- 📧 Email: [your.email@example.com](mailto:your.email@example.com)
- 💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- 🐙 GitHub: [@yourusername](https://github.com/yourusername)

**Supervisor:** Dr. Khalid Mansour

**Institution:** Arab Open University  
**Course:** TM271 - Machine Learning and Deep Learning  
**Semester:** Fall 2025/2026

---

## 🙏 Acknowledgments

- **GCC Sustainability Innovation Lab** for the problem inspiration
- **Oman Vision 2040** for sustainability goals framework
- **Scikit-learn** community for excellent ML tools
- **Dr. Khalid Mansour** for project supervision
- **Arab Open University** for academic support

---

## 📞 Contact

For questions, feedback, or collaboration opportunities:

- 📧 Email: [your.email@example.com](mailto:your.email@example.com)
- 💬 Open an [Issue](https://github.com/yourusername/DSI-Prediction-System/issues)
- 🐦 Twitter: [@yourhandle](https://twitter.com/yourhandle)

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/DSI-Prediction-System?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/DSI-Prediction-System?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/DSI-Prediction-System)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/DSI-Prediction-System)

---

<div align="center">

**⭐ If you found this project useful, please star the repository! ⭐**

Made with ❤️ for a sustainable future 🌍

</div>

---

## 🔗 Related Projects

- [Urban Sustainability Dashboard](https://github.com/example/urban-dashboard)
- [Carbon Footprint Tracker](https://github.com/example/carbon-tracker)
- [Smart City Analytics](https://github.com/example/smart-city)

---

## 📚 References

1. United Nations. (2015). *Sustainable Development Goals*. https://sdgs.un.org/
2. Oman Vision 2040. (2020). *National Strategy for Sustainable Development*.
3. Scikit-learn: Machine Learning in Python. https://scikit-learn.org/
4. Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.

---

**Last Updated:** December 2025
