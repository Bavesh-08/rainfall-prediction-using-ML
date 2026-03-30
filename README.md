# 🌧️ Rainfall Prediction System

## Overview

A machine learning-powered rainfall prediction system that leverages meteorological data to accurately forecast rainfall occurrence. Built with **Random Forest Classifier**, this project combines data preprocessing, exploratory data analysis, and hyperparameter tuning to deliver reliable weather predictions.

---

## 📊 Project Highlights

| Metric | Value |
|--------|-------|
| **Dataset Size** | 366 days of weather data |
| **Model Accuracy** | 72.34% on test set |
| **Precision (Rainfall)** | 69% |
| **Recall (Rainfall)** | 78% |
| **Cross-Validation Score** | 82.38% |
| **Model Type** | Random Forest Classifier |

---

## 🎯 Objective

To predict whether rainfall will occur on a given day based on various meteorological features such as:
- Atmospheric pressure
- Temperature metrics (max, min, dew point)
- Humidity levels
- Cloud coverage
- Wind direction and speed
- Sunshine hours

---

## 📁 Dataset Structure

### Features (Input Variables)
- **day**: Day number (1-31)
- **pressure**: Atmospheric pressure (hPa)
- **dewpoint**: Dew point temperature (°C)
- **humidity**: Relative humidity (%)
- **cloud**: Cloud coverage (%)
- **sunshine**: Hours of sunshine
- **winddirection**: Wind direction (degrees)
- **windspeed**: Wind speed (km/h)

### Target Variable
- **rainfall**: Binary classification (Yes=1, No=0)
  - Class Distribution: 249 days with rain, 117 days without (imbalanced)

### Data Characteristics
- **Total Records**: 366
- **Missing Values**: Handled (wind direction & wind speed)
- **Data Type**: Mixed (int, float, object)

---

## 🔧 Methodology

### 1. **Data Preprocessing**
- Loaded rainfall dataset containing 366 days of meteorological records
- Identified and handled missing values:
  - Wind direction: Filled with mode (80.0)
  - Wind speed: Filled with median
- Converted categorical target variable (yes/no) to binary (1/0)

### 2. **Exploratory Data Analysis (EDA)**
- Analyzed distribution of all numerical features using histograms and KDE plots
- Created correlation heatmap to identify feature relationships
- Detected multicollinearity among temperature features (maxtemp, temparature, mintemp)
- Generated boxplots to identify outliers and data spread

### 3. **Feature Engineering**
- Removed highly correlated features (maxtemp, temparature, mintemp) to reduce redundancy
- Retained 8 features after feature selection
- Final feature set optimized for model performance

### 4. **Handling Class Imbalance**
- Identified significant class imbalance (249:117 ratio)
- Applied **downsampling** on majority class to balance dataset
- Final balanced dataset: 234 samples (117 with rain + 117 without rain)
- Shuffled and reset index for proper data distribution

### 5. **Model Training**
- **Algorithm**: Random Forest Classifier
- **Train-Test Split**: 80-20 (187 training, 47 test samples)
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Parameter Search Space**:
  - n_estimators: [50, 100, 200]
  - max_features: ['sqrt', 'log2']
  - max_depth: [None, 10, 20, 30]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]

### 6. **Best Model Configuration**
```
RandomForestClassifier(
    n_estimators=50,
    max_features='log2',
    min_samples_split=5,
    random_state=42
)
```

### 7. **Model Evaluation**

#### Cross-Validation Results
```
Fold Scores: [0.711, 0.895, 0.811, 0.811, 0.892]
Mean CV Score: 82.38%
```

#### Test Set Performance
```
                Precision  Recall  F1-Score  Support
Class 0 (No Rain)   0.76      0.67     0.71       24
Class 1 (Rain)      0.69      0.78     0.73       23

Overall Accuracy: 72.34%
```

#### Confusion Matrix
```
                Predicted No Rain  Predicted Rain
Actual No Rain       16                8
Actual Rain           5               18
```

---

## 📈 Key Findings

✅ **Strengths**
- Good recall for rainfall prediction (78%) - catches most rainy days
- Balanced precision-recall trade-off (F1-score: 0.73)
- Robust cross-validation performance (82.38%)
- Successfully handled class imbalance through downsampling

⚠️ **Considerations**
- 24% misclassification rate on test set
- 8 false positives (predicted rain but didn't occur)
- Model could benefit from additional features (e.g., seasonal indicators, previous day weather)

---

## 🛠️ Technologies & Libraries

```python
# Data Processing
pandas            # Data manipulation and analysis
numpy             # Numerical computations

# Visualization
matplotlib        # Basic plotting
seaborn          # Statistical data visualization

# Machine Learning
scikit-learn     # ML algorithms and utilities
  - RandomForestClassifier
  - GridSearchCV
  - cross_val_score
  - train_test_split
  - classification_report
  - confusion_matrix

# Model Serialization
pickle           # Model persistence
```

---

## 📋 Requirements

```
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

---

## 🚀 Quick Start Guide

### 1. **Installation**
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 2. **Data Preparation**
```python
import pandas as pd
data = pd.read_csv('Rainfall.csv')
# Data cleaning and preprocessing handled in the notebook
```

### 3. **Train the Model**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier(random_state=42)
# GridSearchCV optimizes hyperparameters automatically
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### 4. **Make Predictions**
```python
predictions = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2%}")
```

---

## 📊 Results Summary

| Metric | Value |
|--------|-------|
| Best CV Score | 82.38% |
| Test Accuracy | 72.34% |
| Precision (Rain) | 69% |
| Recall (Rain) | 78% |
| F1-Score (Rain) | 0.73 |
| True Positives | 18 |
| True Negatives | 16 |
| False Positives | 8 |
| False Negatives | 5 |

---

## 🔍 Model Insights

### Feature Importance (Inferred)
Based on random forest architecture, the following features likely contribute most to predictions:
- Cloud coverage (strong predictor of rainfall)
- Humidity levels (high correlation with rain)
- Atmospheric pressure (inverse relationship with storms)
- Sunshine hours (negative indicator of rain)

### Decision Logic
- **Higher humidity + High cloud coverage** → Higher probability of rainfall
- **Low atmospheric pressure** → More unstable conditions → Higher rain probability
- **Low sunshine hours** → Likely overcast conditions → Higher rain probability

---

## 💡 Future Improvements

1. **Feature Engineering**
   - Add seasonal indicators (month-based features)
   - Create lag features (previous day's weather)
   - Add interaction terms between features

2. **Model Enhancement**
   - Experiment with ensemble methods (Gradient Boosting, XGBoost)
   - Try different resampling techniques (SMOTE, stratified sampling)
   - Implement time-series cross-validation

3. **Data Collection**
   - Expand dataset to multiple years
   - Include additional meteorological variables
   - Geographic diversification

4. **Deployment**
   - Create REST API for predictions
   - Build real-time prediction dashboard
   - Integrate with weather data APIs

---

## 📝 Code Structure

```
rainfall_prediction/
├── README.md                 # This file
├── rainfall_notebook.ipynb   # Full analysis notebook
├── Rainfall.csv             # Input dataset
├── rainfall_model.pkl       # Trained model (serialized)
└── requirements.txt         # Python dependencies
```

---



## 🎓 Learning Outcomes

This project demonstrates:
- ✅ End-to-end machine learning workflow
- ✅ Data preprocessing and cleaning techniques
- ✅ Exploratory data analysis (EDA) best practices
- ✅ Handling class imbalance with downsampling
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Model evaluation and interpretation
- ✅ Cross-validation for robust assessment
- ✅ Python scikit-learn library usage

---

## 🌟 Highlights

> **72.34% Accuracy** in predicting rainfall with a well-balanced Random Forest Classifier trained on 1 year of meteorological data.

> **82.38% Cross-Validation Score** demonstrates model robustness and reduced overfitting.

> **78% Recall** means the model successfully identifies most days when rainfall will occur.

---

**Created**: 2026 | **Dataset Size**: 366 days | **Model Version**: Random Forest v1.0
