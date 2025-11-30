# 🏠 California Housing Price Prediction – End-to-End ML Pipeline

This project performs an end-to-end machine learning workflow using the California Housing Dataset. It includes data loading, exploratory data analysis (EDA), preprocessing, feature engineering, model building, hyperparameter tuning, evaluation, and model deployment. The goal is to build a reliable regression model that predicts median house values for California districts.

---

## 📁 Project Structure
├── datasets/
│ └── housing.tgz # Auto-downloaded dataset
├── House_predictions.ipynb # Main notebook (full workflow)
└── california_housing_model.pkl # Saved final model


---

## 📊 1. Dataset Overview

The dataset contains housing statistics for California districts, including:
- Median income  
- Housing median age  
- Total rooms & total bedrooms  
- Population & households  
- Latitude & longitude  
- Median house value (target)  
- Ocean proximity (categorical feature)

---

## 🔍 2. Exploratory Data Analysis (EDA)

The notebook performs:
- `.head()`, `.info()`, `.describe()`  
- Category counts for `ocean_proximity`  
- Histograms for distributions  
- Identification of missing values and skew  

Key findings:
- Median income strongly correlates with house prices  
- Ocean proximity is unevenly distributed  
- Room and population-related features are highly skewed  
- Missing values exist in `total_bedrooms`

---

## 🧹 3. Data Preprocessing

Steps completed:
- Train-test split using **StratifiedShuffleSplit** (based on income category)
- Median imputation for missing values (`SimpleImputer`)
- Standardization using `StandardScaler`
- One-hot encoding for `ocean_proximity`
- Full preprocessing pipeline created using **ColumnTransformer**

---

## 🧪 4. Feature Engineering

The following engineered features were added to improve predictive power:
- `rooms_per_household`
- `bedrooms_per_room`
- `population_per_household`

These help reduce noise in raw counts and capture meaningful ratios.

---

## 🤖 5. Model Training

Trained regression models:
- **Linear Regression** (baseline)
- **Decision Tree Regressor** (overfitted)
- **Random Forest Regressor** (best performer)

Evaluation was done with cross-validation RMSE.  
Random Forest delivered the most reliable performance.

---

## 🔧 6. Hyperparameter Tuning

Used **RandomizedSearchCV** to optimize Random Forest parameters:
- `n_estimators`
- `max_features`
- `bootstrap`

This improved performance over baseline RandomForest settings.

---

## 🧾 7. Final Evaluation

The tuned model was tested on the **final test set**, computing:
- Final RMSE  
- 95% confidence interval for the prediction error  

This confirms model generalization accuracy.

---

## 💾 8. Saving & Loading the Model

The final model is saved using joblib:

```python```

joblib.dump(final_model, "california_housing_model.pkl")


## 🚀 9. Key Insights

1) Median income is the most influential predictor.

2) Engineered ratio features notably improve model performance.

3) Decision Trees overfit easily; Random Forest generalizes better.

4) Preprocessing (imputation, scaling, encoding) is crucial.

5) Hyperparameter tuning boosts model accuracy and reliability.



## ✅ Conclusion

1)This project demonstrates a complete Machine Learning pipeline for a real-world dataset, covering:

2) Data ingestion

3) Exploratory data analysis

4) Preprocessing & feature engineering

5) Model selection & training

6) Cross-validation

7) Hyperparameter tuning

8) Final evaluation

9) Model saving for deployment
