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

# Import and review data

data = pd.read_csv('data.csv')
data.info()
data.head(10)

# Modeling Preparation

# Convert Categorical Calues to Numerical Values
pd.set_option('future.no_silent_downcasting', True)
data['infill_pattern'] = data['infill_pattern'].replace({'grid': 0, 'honeycomb': 1}).astype(int)
data['material'] = data['material'].replace({'abs': 0, 'pla': 1}).astype(int)
data.head(10)

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

# Visualizing Distribution of Outlier Print Speed
plt.figure(figsize=(8, 6))
sns.histplot(data['print_speed'], bins=20, kde=True, color="skyblue")
plt.title("Distribution of Print Speed")
plt.xlabel("Print Speed")
plt.ylabel("Frequency")
plt.show()

# EDA

# Correlation Heatmap to Show how Strongly Variables are Related to Each Other
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Scatterplot Matrix to Visualize Relationships Between Multiple Variables
sns.pairplot(data, hue="material", diag_kind="kde")
plt.show()

# Boxplot to visualize spread and outliers
plt.figure(figsize=(12, 6))
target_vars = ['roughness', 'tension_strenght', 'elongation']

for i, col in enumerate(target_vars, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=data[col], color='lightblue')
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)

plt.tight_layout()
plt.show()

# Dsitributions of target variables by material
for col in ['roughness', 'tension_strenght', 'elongation']:
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=data, x=col, hue="material", fill=True, common_norm=False)
    plt.title(f'Distribution of {col} by Material Type')
    plt.show()

# Modeling Roughness
X = data.drop(columns=['roughness', 'infill_pattern', 'infill_density', 'wall_thickness'])
y = data[['roughness']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
standard_scaler = StandardScaler()
X_train = pd.DataFrame(standard_scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(standard_scaler.transform(X_test), columns=X_test.columns)

# Train Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Model Engineering
# Train XGBRegressor with Hyperparameters
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.001,
    reg_lambda=1.5
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Calculate Metrics
def print_metrics(model_name, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} Regression Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}\n")

print_metrics("Linear", y_test, y_pred_linear)
print_metrics("XGBoost", y_test, y_pred_xgb)

# Plot Roughness
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear, color='blue', alpha=0.6, label='Linear Regression')
plt.scatter(y_test, y_pred_xgb, color='green', alpha=0.6, label='XGBoost Regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Ideal Fit')

plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Regression Models: Roughness Prediction')
plt.legend()
plt.grid(True)
plt.show()

# Modeling Elongation
X2 = data.drop(columns=['elongation', 'nozzle_temperature', 'material', 'fan_speed', 'bed_temperature', 'print_speed'])
y2 = data[['elongation']]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=100)
standard_scaler = StandardScaler()
X_train2 = pd.DataFrame(standard_scaler.fit_transform(X_train2), columns=X_train2.columns)
X_test2 = pd.DataFrame(standard_scaler.transform(X_test2), columns=X_test2.columns)

# algorithms
linear_model = LinearRegression()
ridge_model = Ridge()
lasso_model = Lasso(max_iter=10000)

# Linear Regression
linear_model.fit(X_train2, y_train2)
y_pred_linear = linear_model.predict(X_test2)

# Model Engneering
# Ridge Regression with Hyperparameter Tuning
ridge_params = {'alpha': np.logspace(-3, 3, 20)}
ridge_cv = GridSearchCV(ridge_model, ridge_params, cv=5, scoring='r2')
ridge_cv.fit(X_train2, y_train2)
best_ridge = ridge_cv.best_estimator_
y_pred_ridge = best_ridge.predict(X_test2)

# Lasso Regression with Hyperparameter Tuning
lasso_params = {'alpha': np.logspace(-3, 3, 20)}
lasso_cv = GridSearchCV(lasso_model, lasso_params, cv=5, scoring='r2')
lasso_cv.fit(X_train2, y_train2)
best_lasso = lasso_cv.best_estimator_
y_pred_lasso = best_lasso.predict(X_test2)

print(f"Best Ridge Alpha: {ridge_cv.best_params_['alpha']}")
print(f"Best Lasso Alpha: {lasso_cv.best_params_['alpha']}\n")

# Calculate Metrics
def print_metrics(model_name, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} Regression Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}\n")

print_metrics("Linear", y_test2, y_pred_linear)
print_metrics("Ridge", y_test2, y_pred_ridge)
print_metrics("Lasso", y_test2, y_pred_lasso)

plt.figure(figsize=(10, 6))
plt.scatter(y_test2, y_pred_linear, color='blue', alpha=0.6, label='Linear Regression')
plt.scatter(y_test2, y_pred_ridge, color='green', alpha=0.6, label='Ridge Regression')
plt.scatter(y_test2, y_pred_lasso, color='orange', alpha=0.6, label='Lasso Regression')
plt.plot([y_test2.min(), y_test2.max()], [y_test2.min(), y_test2.max()], color='red', lw=2, label='Ideal Fit')

plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Regression Models: Elongation Prediction')
plt.legend()
plt.grid(True)
plt.show()

# Modeling Tensile Strength
X3 = data.drop(columns=['tension_strenght', 'nozzle_temperature', 'material', 'fan_speed', 'bed_temperature', 'print_speed'])
y3 = data[['tension_strenght']]

X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=100)
standard_scaler = StandardScaler()
X_train3 = pd.DataFrame(standard_scaler.fit_transform(X_train3), columns=X_train3.columns)
X_test3 = pd.DataFrame(standard_scaler.transform(X_test3), columns=X_test3.columns)

# Algorithms
LR_model3 = LinearRegression()
ridge_model = Ridge()
lasso_model = Lasso(max_iter=5000)

# Linear Regression
LR_model3.fit(X_train3, y_train3)
y_train_pred3 = LR_model3.predict(X_train3)
y_test_pred3 = LR_model3.predict(X_test3)

# Model Engineering
# Ridge Regression with Hyperparameter Tuning
ridge_params = {'alpha': np.logspace(-3, 3, 100)}
ridge_cv = GridSearchCV(ridge_model, ridge_params, cv=5, scoring='r2')
ridge_cv.fit(X_train3, y_train3)
best_ridge_alpha = ridge_cv.best_params_['alpha']
ridge_best = Ridge(alpha=best_ridge_alpha)
ridge_best.fit(X_train3, y_train3)
y_train_ridge = ridge_best.predict(X_train3)
y_test_ridge = ridge_best.predict(X_test3)

# Lasso Regression with Hyperparameter Tuning
lasso_params = {'alpha': np.logspace(-3, 3, 100)}
lasso_cv = GridSearchCV(lasso_model, lasso_params, cv=5, scoring='r2')
lasso_cv.fit(X_train3, y_train3)
best_lasso_alpha = lasso_cv.best_params_['alpha']
lasso_best = Lasso(alpha=best_lasso_alpha, max_iter=5000)
lasso_best.fit(X_train3, y_train3)
y_train_lasso = lasso_best.predict(X_train3)
y_test_lasso = lasso_best.predict(X_test3)

# Calculate Metrics
def compute_metrics(y_train, y_train_pred, y_test, y_test_pred):
    return {
        "Training MAE": mean_absolute_error(y_train, y_train_pred),
        "Testing MAE": mean_absolute_error(y_test, y_test_pred),
        "Training MSE": mean_squared_error(y_train, y_train_pred),
        "Testing MSE": mean_squared_error(y_test, y_test_pred),
        "Training RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Testing RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "Training R²": r2_score(y_train, y_train_pred),
        "Testing R²": r2_score(y_test, y_test_pred),
    }

linear_metrics = compute_metrics(y_train3, y_train_pred3, y_test3, y_test_pred3)
ridge_metrics = compute_metrics(y_train3, y_train_ridge, y_test3, y_test_ridge)
lasso_metrics = compute_metrics(y_train3, y_train_lasso, y_test3, y_test_lasso)

print(f"Best Ridge Alpha: {best_ridge_alpha:.6f}")
print(f"Best Lasso Alpha: {best_lasso_alpha:.6f}\n")

print("Linear Regression Metrics:")
for k, v in linear_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nRidge Regression Metrics:")
for k, v in ridge_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nLasso Regression Metrics:")
for k, v in lasso_metrics.items():
    print(f"{k}: {v:.4f}")

# Plot Tensile Strength
plt.figure(figsize=(18, 6))
plt.scatter(y_test3, y_test_pred3, color='blue', label='Linear Regression')
plt.scatter(y_test3, y_test_ridge, color='green', label='Ridge Regression')
plt.scatter(y_test3, y_test_lasso, color='orange', label='Lasso Regression')
plt.plot([y_test3.min(), y_test3.max()], [y_test3.min(), y_test3.max()], color='red', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Regression Models: Tensile Strength')
plt.legend()
plt.grid(True)
plt.show()

# Observations and Conclusions

# SHAP 
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X_train.iloc[0])
shap.summary_plot(shap_values, X_train)

# Feature Importance
xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=10)
plt.title("XGBoost Feature Importance")
plt.show()

feature_importance = np.abs(linear_model.coef_).flatten()  
feature_names = X_train.columns  
sorted_idx = np.argsort(feature_importance)[::+1]

plt.barh(feature_names[sorted_idx], feature_importance[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Linear Regression Feature Importance")
plt.show()

# Cross Validation
def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    print(f"Cross-validation R² scores: {scores}")
    print(f"Mean R² score: {scores.mean():.4f}")

cross_validate_model(linear_model, X_train, y_train)
cross_validate_model(xgb_model, X_train, y_train)
