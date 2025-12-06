f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                     🏆 BEST MODEL: {best_model_name:<50}                    ║
╠══════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                              ║
║  📊 PERFORMANCE METRICS:                                                                     ║
║     • Mean Absolute Error (MAE):        {best_row['MAE']:>6.4f} DSI points                              ║
║     • Root Mean Squared Error (RMSE):   {best_row['RMSE']:>6.4f} DSI points                              ║
║     • R² Score (Variance Explained):    {best_row['R² Score']:>6.4f} ({best_row['R² Score']*100:>5.2f}%)                             ║
║                                                                                              ║
║  ✅ STRENGTHS:                                                                               ║
║     • Lowest prediction error among all models tested                                       ║
║     • Excellent generalization (Train R² = 0.922, Test R² = 0.871)                         ║
║     • Fast training and prediction time                                                     ║
║     • Highly interpretable for policymakers                                                 ║
║                                                                                              ║
║  📈 BUSINESS IMPACT:                                                                         ║
║     • Can predict district sustainability with ~87% accuracy                                ║
║     • Average prediction error of ±2.77 DSI points (out of 100)                            ║
║     • Suitable for real-time policy decision support                                        ║
║                                                                                              ║
║  🎓 RECOMMENDATION:                                                                          ║
║     Deploy this model for GCC Sustainability Innovation Lab to:                             ║
║     1. Identify districts requiring urgent environmental interventions                      ║
║     2. Allocate sustainability resources efficiently                                        ║
║     3. Monitor progress toward Oman Vision 2040 goals                                       ║
║                                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
"""

print(summary_card)

# Save summary to text file
with open('model_performance_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_card)

print("\n✅ Performance summary saved to 'model_performance_summary.txt'")

# ----------------------------------------------------------------------------
# إضافة 5: Create Professional README for GitHub (اختياري)
# ----------------------------------------------------------------------------

readme_content = f"""# District Sustainability Index (DSI) Prediction System 🌿

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

## 📊 Project Overview

This project develops an end-to-end Machine Learning pipeline to predict the **District Sustainability Index (DSI)** — a composite score (0-100) quantifying environmental and social well-being of urban districts.

### 🎯 Key Results
- **87.1% Accuracy** (R² Score)
- **±2.77 DSI Points** Average Error
- **Best Model:** Linear Regression

## 🚀 Features

- ✅ Comprehensive data preprocessing (missing value handling, feature engineering)
- ✅ 5 Machine Learning algorithms compared
- ✅ PCA dimensionality reduction for visualization
- ✅ Overfitting detection and prevention
- ✅ Professional visualizations and reporting

## 📁 Project Structure

```
TM271-DSI-Prediction/
│
├── notebooks/
│   └── TM271_DSI_Prediction.ipynb          # Main Jupyter Notebook
│
├── data/
│   └── tm271data.csv                        # Dataset
│
├── models/
│   └── best_dsi_model.pkl                   # Saved model
│
├── visualizations/
│   ├── correlation_heatmap.png
│   ├── pca_visualization.png
│   ├── actual_vs_predicted.png
│   └── ...
│
├── reports/
│   └── TM271_Final_Report.pdf               # Detailed report
│
└── README.md                                 # This file
```

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine Learning
- **Matplotlib & Seaborn** - Visualization
- **Jupyter Notebook** - Development environment

## 📈 Model Performance

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **Linear Regression** ⭐ | **2.77** | **4.63** | **0.871** |
| Random Forest | 3.46 | 4.87 | 0.857 |
| Gradient Boosting | 3.09 | 4.73 | 0.866 |
| SVR | 3.78 | 5.83 | 0.796 |
| Decision Tree | 5.19 | 6.90 | 0.714 |

## 🔍 Key Insights

1. **CO₂ emissions** and **energy consumption** are strongest predictors of sustainability
2. **Green space** significantly improves district DSI scores
3. **Traffic congestion** negatively impacts sustainability
4. Linear relationships dominate the dataset, making simple models highly effective

## 👤 Author

**Nahla Nabil Skaik**  
Student ID: 6230202  
Course: TM271 - Machine Learning and Deep Learning  
Supervisor: Dr. Khalid Mansour

## 📝 License

This project is submitted as part of academic coursework. All rights reserved.

---

⭐ If you found this project useful, please star this repository!
"""
