import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

# Load the dataset
df = pd.read_csv(r'C:\Users\DELL\Desktop\MAJ\Agri-Management\Trainset.csv')

# Drop unwanted column
df.drop('ElectricalConductivity(ds/m)', axis=1, inplace=True)

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['Season', 'Crop'])

# Normalize continuous features
for col in ['Area', 'Rainfall', 'Temperature', 'pH', 'Nitrogen(kg/ha)']:
    df[col] = df[col] / df[col].max()

# Define features (X) and target variable (y)
X = df.drop(columns=['Production'])
y = df['Production']

# Split data into training and validation sets
train_inputs, val_inputs, train_targets, val_targets = train_test_split(X, y, test_size=0.2, random_state=8)

# Initialize and train Decision Tree Regressor
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(train_inputs, train_targets)

# Predictions and evaluation for Decision Tree
pred_dtr = dtr.predict(val_inputs)
dtr_train_accuracy = dtr.score(train_inputs, train_targets)
dtr_val_accuracy = dtr.score(val_inputs, val_targets)
rms_dtr = sqrt(mean_squared_error(val_targets, pred_dtr))
va_dtr = rms_dtr / df['Production'].max()

print(f"Decision Tree Training Accuracy (R²): {dtr_train_accuracy * 100:.2f}%")
print(f"Decision Tree Validation Accuracy (R²): {dtr_val_accuracy * 100:.2f}%")
print(f"Decision Tree RMSE: {rms_dtr}")
print(f"Decision Tree RMSE Normalized: {va_dtr}")

# Initialize and train Random Forest Regressor
rfc = RandomForestRegressor(n_estimators=30, max_depth=15, min_samples_split=6, bootstrap=True, random_state=0, n_jobs=-1)
rfc.fit(train_inputs, train_targets)

# Evaluate Random Forest
rfc_train_accuracy = rfc.score(train_inputs, train_targets)
rfc_val_accuracy = rfc.score(val_inputs, val_targets)
pred_rfc = rfc.predict(val_inputs)
rms_rfc = sqrt(mean_squared_error(val_targets, pred_rfc))
va_rfc = rms_rfc / df['Production'].max()

print(f"Random Forest Training Accuracy (R²): {rfc_train_accuracy * 100:.2f}%")
print(f"Random Forest Validation Accuracy (R²): {rfc_val_accuracy * 100:.2f}%")
print(f"Random Forest RMSE: {rms_rfc}")
print(f"Random Forest RMSE Normalized: {va_rfc}")

# Initialize and train XGBoost Regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=0, n_jobs=-1)
xgb_model.fit(train_inputs, train_targets)

# Evaluate XGBoost
xgb_train_accuracy = xgb_model.score(train_inputs, train_targets)
xgb_val_accuracy = xgb_model.score(val_inputs, val_targets)
pred_xgb = xgb_model.predict(val_inputs)
rms_xgb = sqrt(mean_squared_error(val_targets, pred_xgb))
va_xgb = rms_xgb / df['Production'].max()

print(f"XGBoost Training Accuracy (R²): {xgb_train_accuracy * 100:.2f}%")
print(f"XGBoost Validation Accuracy (R²): {xgb_val_accuracy * 100:.2f}%")
print(f"XGBoost RMSE: {rms_xgb}")
print(f"XGBoost RMSE Normalized: {va_xgb}")
