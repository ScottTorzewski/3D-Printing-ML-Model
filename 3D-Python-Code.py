# Separate Python Code for Github Statistics

# Import libraries

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
import shap

# Check for Duplicates, Null Values, Outliers

# Check for Duplicates in Each Column
duplicates = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Check for Null Values in Each Column
null_values = data.isnull().sum()
print("Null values in each column:\n", null_values)

# Check for Outliers in Each Column
def detect_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    return outliers

numeric_cols = data.select_dtypes(include=np.number)
outliers = detect_outliers_iqr(numeric_cols)
print("Outliers detected in each column:\n", outliers)
