from ibtracs import Ibtracs
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from scipy.stats import boxcox
from scipy.special import inv_boxcox

def calculate_wind_field_area(ne, se, sw, nw):
    """
    Calculate wind field area intelligently:
    - All 4 quadrants: use elliptical approximation (captures asymmetry)
    - <4 quadrants: sum circular sectors (no data inference)
    
    Returns area in km²
    """
    # Collect valid radii
    radii_dict = {'ne': ne, 'se': se, 'sw': sw, 'nw': nw}
    valid_radii = {}
    
    for name, r in radii_dict.items():
        if pd.isna(r):
            continue
        try:
            r_val = float(r)
            if r_val >= 0:
                valid_radii[name] = r_val
        except (ValueError, TypeError):
            continue
    
    if len(valid_radii) == 0:
        return np.nan
    
    # Case 1: All 4 quadrants → ellipse (captures asymmetry)
    if len(valid_radii) == 4:
        semi_major = (valid_radii['ne'] + valid_radii['sw']) / 2
        semi_minor = (valid_radii['nw'] + valid_radii['se']) / 2
        area_nm2 = np.pi * semi_major * semi_minor
        return area_nm2 * 3.434  # nm² to km²
    
    # Case 2: <4 quadrants → sum of circular sectors
    # Each sector is (π/4) * r²
    sum_sector_area_nm2 = (np.pi / 4) * sum(r**2 for r in valid_radii.values())
    return sum_sector_area_nm2 * 3.434


# Load and process data
print("Loading hurricane data...")
hurricane_data = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\ibtracs_filtered_R64.csv", low_memory=False)

hurricane_data['basin'] = hurricane_data['USA_ATCF_ID'].str.slice(0, 2)
# Filter for atlantic basin
hurricane_data = hurricane_data[hurricane_data['basin'] == 'AL']
# Format columns
hurricane_data['Month'] = pd.to_datetime(hurricane_data['ISO_TIME']).dt.month

hurricane_data['LAT'] = pd.to_numeric(hurricane_data['LAT'], errors='coerce')
hurricane_data['LON'] = pd.to_numeric(hurricane_data['LON'], errors='coerce')
hurricane_data['USA_WIND'] = pd.to_numeric(hurricane_data['USA_WIND'], errors='coerce')
hurricane_data['USA_PRES'] = pd.to_numeric(hurricane_data['USA_PRES'], errors='coerce')
hurricane_data['STORM_SPEED'] = pd.to_numeric(hurricane_data['STORM_SPEED'], errors='coerce')
hurricane_data['USA_R64_NE'] = pd.to_numeric(hurricane_data['USA_R64_NE'], errors='coerce')
hurricane_data['USA_R64_SE'] = pd.to_numeric(hurricane_data['USA_R64_SE'], errors='coerce')
hurricane_data['USA_R64_SW'] = pd.to_numeric(hurricane_data['USA_R64_SW'], errors='coerce')
hurricane_data['USA_R64_NW'] = pd.to_numeric(hurricane_data['USA_R64_NW'], errors='coerce')

print("Calculating wind field areas...")
hurricane_data['wind_field_area_km2'] = hurricane_data.apply(lambda row: calculate_wind_field_area(
    row['USA_R64_NE'], row['USA_R64_SE'], row['USA_R64_SW'], row['USA_R64_NW']), axis=1)
hurricane_data['mean_wind_radius_km'] = hurricane_data[['USA_R64_NE', 'USA_R64_SE', 'USA_R64_SW', 'USA_R64_NW']].mean(axis=1)

#remove 0 area and 0 mean wind radius
hurricane_data = hurricane_data[(hurricane_data['wind_field_area_km2'] > 0) & (hurricane_data['mean_wind_radius_km'] > 0)]
predict_column = 'mean_wind_radius_km'  # Target variable: mean wind radius in km

feature_columns = ['LAT', 'LON', 'USA_WIND', 'USA_PRES', 'STORM_SPEED', 'Month' ]

# Drop rows with NaN in feature or target columns
print(f"Total rows: {len(hurricane_data)}")
hurricane_data_clean = hurricane_data.dropna(subset=feature_columns + [predict_column])
print(f"Clean rows (no NaN): {len(hurricane_data_clean)}")

#check distribution of target variable
# plt.figure(figsize=(8, 5))
# plt.hist(np.log(hurricane_data_clean[predict_column]), bins=30, color='skyblue', edgecolor='black')
# plt.show()
# #try box cox transformation
# from scipy import stats
# hurricane_data_clean['log_mean_wind_radius_km'], fitted_lambda = stats.boxcox(hurricane_data_clean[predict_column] + 1)  # add 1 to avoid log(0)
# plt.figure(figsize=(8, 5))
# plt.hist(hurricane_data_clean['log_mean_wind_radius_km'], bins=30, color='lightgreen', edgecolor='black')
# plt.show()

# quit()
X = hurricane_data_clean[feature_columns]
y = hurricane_data_clean[predict_column]
#box cox transformation
y_transformed, fitted_lambda = boxcox(y)  # add 1 to avoid log(0)

# Split data into training and testing sets
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline model
print("\n" + "="*60)
print("BASELINE XGBOOST")
print("="*60)

xgb_baseline = xgb.XGBRegressor(
    n_estimators=100,
    random_state=42,
    verbosity=0,
    early_stopping_rounds=10
)

xgb_baseline.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
y_pred_transformed = xgb_baseline.predict(X_test_scaled)
y_pred_baseline = inv_boxcox(y_pred_transformed, fitted_lambda)  # Inverse boxcox-transform
mse_baseline = mean_squared_error(inv_boxcox(y_test, fitted_lambda), y_pred_baseline)
r2_baseline = r2_score(inv_boxcox(y_test, fitted_lambda), y_pred_baseline)

print(f"Mean Squared Error: {mse_baseline:.2f}")
print(f"R² Score: {r2_baseline:.4f}")

# Feature importance
importances = xgb_baseline.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
print(feature_importance_df)

# Plot baseline
plt.figure(figsize=(10, 6))
plt.scatter(inv_boxcox(y_test, fitted_lambda), y_pred_baseline, alpha=0.6, edgecolors='k', s=50)
plt.plot([inv_boxcox(y_test, fitted_lambda).min(), inv_boxcox(y_test, fitted_lambda).max()], [inv_boxcox(y_test, fitted_lambda).min(), inv_boxcox(y_test, fitted_lambda).max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual R64 Area (km²)', fontsize=12)
plt.ylabel('Predicted R64 Area (km²)', fontsize=12)
plt.title(f'Baseline XGBoost\nR² = {r2_baseline:.4f}, RMSE = {np.sqrt(mse_baseline):.2f}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Bayesian Optimization
print("\n" + "="*60)
print("BAYESIAN OPTIMIZATION")
print("="*60)

# Define search space
space = [
    Integer(3, 12, name='max_depth'),
    Real(0.001, 0.3, name='learning_rate'),
    Real(0.5, 1.0, name='subsample'),
    Real(0.5, 1.0, name='colsample_bytree'),
]

# Define objective function
# BUG 1: In Bayesian optimization objective function
# Currently evaluates on TRANSFORMED y_test, should be on ORIGINAL scale
@use_named_args(space)
def objective(max_depth, learning_rate, subsample, colsample_bytree):
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=20
    )
    
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
    y_pred = model.predict(X_test_scaled)
    y_pred_original = inv_boxcox(y_pred, fitted_lambda)  # FIX: Transform back
    y_test_original = inv_boxcox(y_test, fitted_lambda)  # FIX: Transform back
    r2 = r2_score(y_test_original, y_pred_original)  # FIX: Evaluate on original scale
    
    return -r2

# Run Bayesian optimization
print("Searching for optimal hyperparameters...")
result = gp_minimize(
    objective,
    space,
    n_calls=30,  # Number of iterations
    random_state=42,
    verbose=1,
    n_initial_points=10
)

print(f"\nBest R² found: {-result.fun:.4f}")
print("Best parameters:")
best_params = {
    'max_depth': result.x[0],
    'learning_rate': result.x[1],
    'subsample': result.x[2],
    'colsample_bytree': result.x[3]
}
for key, val in best_params.items():
    print(f"  {key}: {val}")

# Train final model with best parameters
print("\nTraining final model with optimized parameters...")
xgb_best = xgb.XGBRegressor(
    n_estimators=200,
    random_state=42,
    verbosity=0,
    early_stopping_rounds=20,
    **best_params
)

xgb_best.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
y_pred_best_transformed = xgb_best.predict(X_test_scaled)
y_pred_best = inv_boxcox(y_pred_best_transformed, fitted_lambda)  # Transform back
y_test_original = inv_boxcox(y_test, fitted_lambda)  # Transform back

# BUG 2: Final model evaluation - MSE and R² on ORIGINAL scale
mse_best = mean_squared_error(y_test_original, y_pred_best)  # FIX: Original scale
r2_best = r2_score(y_test_original, y_pred_best)  # FIX: Original scale

print(f"\nFinal Model Results (on original scale):")
print(f"Mean Squared Error: {mse_best:.2f}")
print(f"R² Score: {r2_best:.4f}")
print(f"Improvement: +{(r2_best - r2_baseline)*100:.2f}%")

# BUG 3: Plot final results - use original scale
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred_best, alpha=0.6, edgecolors='k', s=50, color='steelblue')  # FIX
plt.plot([y_test_original.min(), y_test_original.max()], 
         [y_test_original.min(), y_test_original.max()], 'r--', lw=2, label='Perfect Prediction')  # FIX
plt.xlabel('Actual Mean Wind Radius (km)', fontsize=12)
plt.ylabel('Predicted Mean Wind Radius (km)', fontsize=12)
plt.title(f'Box-Cox Bayesian XGBoost\nR² = {r2_best:.4f}, RMSE = {np.sqrt(mse_best):.2f}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
