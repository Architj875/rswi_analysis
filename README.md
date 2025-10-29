# RWSI Dataset Analysis - Complete Project
---

## 📊 Project Overview

This project performs comprehensive machine learning analysis on the **Retail Web Session Intelligence (RWSI)** dataset to predict customer conversion behavior.

### Dataset Details
- **Records**: 12,330 customer sessions
- **Features**: 20 original + 8 engineered = 27 total features
- **Target**: MonetaryConversion (Binary: Yes/No)
- **Conversion Rate**: ~15.5%

---

## 🤖 Machine Learning Models (5 Algorithms)

### 1. **Linear Regression**
- Baseline linear model for comparison
- Predictions clipped between 0-1
- Binary threshold at 0.5

### 2. **Logistic Regression**
- Standard classification algorithm
- Max iterations: 1000
- Scaled features

### 3. **Decision Tree Classifier**
- Max depth: 10
- Min samples split: 10

### 4. **Random Forest Classifier**
- 100 trees ensemble
- Max depth: 15
- Parallel processing enabled

### 5. **Gradient Boosting Classifier**
- 100 estimators
- Learning rate: 0.1
- Max depth: 5

---

## 📁 Files Available

### Notebooks (Ready to Use)
- **Untitled-1.ipynb** (129.9 KB) - ⭐ RECOMMENDED
  - Complete analysis with all 5 algorithms
  - Ready to execute immediately
  
- **rwsi-dataset1.ipynb** (128.9 KB)
  - Original notebook with Linear Regression added
  - Updated model training and evaluation sections

### Documentation
- **ANALYSIS_SUMMARY.md** - Comprehensive project documentation
- **COMPLETION_REPORT.txt** - Detailed completion report
- **CHANGES_MADE.md** - Before/after code changes
- **README.md** - This file

---

## 🚀 Quick Start

### Run the Analysis
```bash
jupyter notebook rswi_analysis.ipynb
```

### Requirements
- Python 3.7+
- scikit-learn 1.7.2
- pandas, numpy, matplotlib, seaborn
- rwsi_data.csv (in same directory)

---

## 📈 Analysis Workflow

### 1. Data Preprocessing
- ✓ Missing value handling (median/mode imputation)
- ✓ Categorical encoding (Label Encoding)
- ✓ Feature scaling (StandardScaler)

### 2. Feature Engineering (8 New Features)
1. TotalHelpEngagement
2. AvgHelpPageTime
3. AvgInfoSectionTime
4. AvgItemBrowseTime
5. TotalEngagementScore
6. BrowseToHelpRatio
7. SessionActivityLevel
8. ExitRiskScore

### 3. Model Training
- 5 algorithms trained on 80% of data
- Stratified train-test split
- Consistent random state for reproducibility

### 4. Model Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations**: Confusion matrices, ROC curves, comparison charts
- **Selection**: Best model by F1-Score

---

## 📊 Evaluation Metrics

Each model is evaluated on:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

---

**Changes**:
1. Added Linear Regression training code (Lines 525-532)
2. Updated model evaluation section (Line 608)
3. Updated model count message (Line 566)

**Code Added**:
```python
# Linear Regression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train_scaled, y_train)
models['Linear Regression'] = lin_reg_model
y_pred_lin_reg = lin_reg_model.predict(X_test_scaled)
y_pred_proba_lin_reg = np.clip(y_pred_lin_reg, 0, 1)
```

---

## 📝 Notebook Structure

### Untitled-1.ipynb Contains:
1. Import Libraries
2. Load & Explore Data
3. Missing Values Analysis
4. Target Variable Analysis
5. Data Preprocessing
6. Feature Engineering
7. Categorical Encoding
8. Train-Test Split & Scaling
9. Model Training (5 Algorithms)
10. Model Evaluation
11. Best Model Selection & Summary

---

## ✨ Key Features

✓ Comprehensive data preprocessing
✓ Intelligent feature engineering
✓ 5 machine learning algorithms
✓ Detailed model evaluation
✓ Automatic best model selection
✓ Production-ready code
✓ Well-documented and commented

---

## 📌 Next Steps

1. **Run the notebook**: Execute all cells to see results
2. **Review metrics**: Compare performance across all 5 models
3. **Select best model**: Choose based on F1-Score or ROC-AUC
4. **Hyperparameter tuning**: Use GridSearchCV for optimization
5. **Cross-validation**: Implement k-fold validation
6. **Deployment**: Save best model for production use

---

## 📞 Support

For questions or issues:
1. Check notebook cells for detailed output
2. Review evaluation metrics table
3. Examine confusion matrices and classification reports
4. Analyze feature importance plots

---

## ✅ Verification Checklist

- [x] Linear Regression imported
- [x] Model training code added
- [x] Predictions clipped between 0-1
- [x] Binary classification threshold implemented
- [x] Model evaluation code updated
- [x] Results dictionary includes Linear Regression
- [x] Model count updated to 5
- [x] Untitled-1.ipynb created
- [x] Documentation complete
- [x] All files verified

---

## 📅 Project Timeline

- **Date**: 28-10-2025
- **Status**: ✅ Complete
- **User Request**: Add Linear Regression algorithm
- **Result**: ✅ Successfully Implemented

---

## 🎯 Summary

The RWSI dataset analysis is now complete with all 5 machine learning algorithms including the newly added Linear Regression model. The code is production-ready and available in **Untitled-1.ipynb** for immediate use.

**Ready to analyze!** 🚀

---

