from ibtracs import Ibtracs
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, roc_auc_score, classification_report
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from scipy.stats import boxcox, yeojohnson
from scipy.special import inv_boxcox
import joblib

def safe_inv_boxcox(y_transformed, lambda_param, offset=0.1):
    """Inverse Box-Cox transform and remove offset, handling NaN values"""
    y_original = inv_boxcox(y_transformed, lambda_param)
    y_original = np.maximum(y_original - offset, 0)  # Ensure non-negative
    # Replace any NaN with 0
    y_original = np.where(np.isnan(y_original), 0, y_original)
    return y_original
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

def calculate_mean_wind_radius(ne, se, sw, nw):
    """
    Calculate mean wind radius from available quadrant radii.
    Returns mean radius in km
    """
    radii = []
    for r in [ne, se, sw, nw]:
        if pd.isna(r):
            continue
        try:
            r_val = float(r)
            if r_val >= 0:
                radii.append(r_val)
        except (ValueError, TypeError):
            continue
    
    if len(radii) == 0:
        return np.nan
    if len(radii) < 4:
        while len(radii) < 4:
            radii.append(0) #maybe impute later
    
    return np.mean(radii)
def load_and_process_data():
    # Load and process data
    print("Loading hurricane data...")
    hurricane_data = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\ibtracs.ALL.list.v04r01.csv", usecols=["ISO_TIME", "LAT", "LON", "WMO_WIND", "WMO_PRES","USA_ATCF_ID", "USA_LAT", "USA_LON", 
                                                                                                    "USA_WIND", "USA_PRES", "STORM_SPEED", "USA_R64_NE", "USA_R64_SE", "USA_R64_SW", "USA_R64_NW",
                                                                                                    "USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW",
                                                                                                    "USA_R50_NE", "USA_R50_SE", "USA_R50_SW", "USA_R50_NW",
                                                                                                    'DIST2LAND', 'STORM_DIR'],low_memory=False)
    #hurricane_data = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\ibtracs_filtered_R64.csv", low_memory=False)

    hurricane_data['basin'] = hurricane_data['USA_ATCF_ID'].str.slice(0, 2)
    # Filter for atlantic basin
    hurricane_data = hurricane_data[hurricane_data['basin'] == 'AL']
    # Format columns
    hurricane_data['Month'] = pd.to_datetime(hurricane_data['ISO_TIME']).dt.month
    hurricane_data['Year'] = pd.to_datetime(hurricane_data['ISO_TIME']).dt.year
    hurricane_data['Day'] = pd.to_datetime(hurricane_data['ISO_TIME']).dt.day

    hurricane_data['LAT'] = pd.to_numeric(hurricane_data['LAT'], errors='coerce')
    hurricane_data['LON'] = pd.to_numeric(hurricane_data['LON'], errors='coerce')
    hurricane_data['USA_WIND'] = pd.to_numeric(hurricane_data['USA_WIND'], errors='coerce')
    hurricane_data['USA_PRES'] = pd.to_numeric(hurricane_data['USA_PRES'], errors='coerce')
    hurricane_data['STORM_SPEED'] = pd.to_numeric(hurricane_data['STORM_SPEED'], errors='coerce')
    hurricane_data['USA_R50_NE'] = pd.to_numeric(hurricane_data['USA_R50_NE'], errors='coerce')
    hurricane_data['USA_R50_SE'] = pd.to_numeric(hurricane_data['USA_R50_SE'], errors='coerce')
    hurricane_data['USA_R50_SW'] = pd.to_numeric(hurricane_data['USA_R50_SW'], errors='coerce')
    hurricane_data['USA_R50_NW'] = pd.to_numeric(hurricane_data['USA_R50_NW'], errors='coerce')

    print("Calculating wind field areas...")
    hurricane_data['wind_field_area_km2'] = hurricane_data.apply(lambda row: calculate_wind_field_area(
        row['USA_R50_NE'], row['USA_R50_SE'], row['USA_R50_SW'], row['USA_R50_NW']), axis=1)
    hurricane_data['mean_wind_radius_km'] = hurricane_data.apply(lambda row: calculate_mean_wind_radius(
        row['USA_R50_NE'], row['USA_R50_SE'], row['USA_R50_SW'], row['USA_R50_NW']), axis=1)


    #remove 0 area and 0 mean wind radius pre 2004
    hurricane_data_for_ml = hurricane_data[ (hurricane_data['Year'] >= 2004)]
    print(f"Data rows after 2004: {len(hurricane_data)}")
    return hurricane_data_for_ml

def create_classifier_model(hurricane_data_for_ml):

    #set nan values to 0 for area and mean wind radius 
    hurricane_data_for_ml['mean_wind_radius_km'] = hurricane_data_for_ml['mean_wind_radius_km'].fillna(0)

    #make flag column for 0 mean wind radius
    hurricane_data_for_ml['has_R50'] = hurricane_data_for_ml['mean_wind_radius_km'] > 0

    print(hurricane_data_for_ml[hurricane_data_for_ml['has_R50'] == True].shape)
    print(hurricane_data_for_ml[hurricane_data_for_ml['has_R50'] == False].shape)
                        
    #create classifier for zero mean wind radius
    predict_column = 'has_R50'  # Target variable: mean wind radius in km

    feature_columns = ['LAT', 'LON', 'USA_WIND', 'USA_PRES', 'STORM_SPEED', 'Month' , 'Year', 'Day', 'DIST2LAND', 'STORM_DIR']


    hurricane_data_for_ml_clean = hurricane_data_for_ml.dropna(subset=feature_columns + [predict_column])

    X = hurricane_data_for_ml_clean[feature_columns]
    y = hurricane_data_for_ml_clean[predict_column]


    # Split data into training and testing sets


    # Split data into training and testing sets
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # BASELINE CLASSIFIER with CV
    print("\n" + "="*60)
    print("BASELINE XGBOOST CLASSIFIER (with 5-Fold Cross-Validation)")
    print("="*60)

    xgb_classifier_baseline = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
        eval_metric='logloss',
    )



    # Cross-validation scores (accuracy and AUC) - NO eval_set
    cv_scores_accuracy = cross_val_score(
        xgb_classifier_baseline, X_train_scaled, y_train, 
        cv=kfold, scoring='accuracy', n_jobs=-1
    )

    cv_scores_auc = cross_val_score(
        xgb_classifier_baseline, X_train_scaled, y_train, 
        cv=kfold, scoring='roc_auc', n_jobs=-1
    )

    print(f"CV Accuracy Scores: {cv_scores_accuracy}")
    print(f"Mean CV Accuracy: {cv_scores_accuracy.mean():.4f} (+/- {cv_scores_accuracy.std():.4f})")
    print(f"\nCV AUC Scores: {cv_scores_auc}")
    print(f"Mean CV AUC: {cv_scores_auc.mean():.4f} (+/- {cv_scores_auc.std():.4f})")

    # Train on full training set for final test evaluation
    xgb_classifier_baseline.fit(
        X_train_scaled, y_train, 
        eval_set=[(X_test_scaled, y_test)], 
        verbose=False
    )

    y_pred_baseline = xgb_classifier_baseline.predict(X_test_scaled)
    y_pred_proba_baseline = xgb_classifier_baseline.predict_proba(X_test_scaled)[:, 1]


    train_score_baseline = xgb_classifier_baseline.score(X_train_scaled, y_train)
    test_score_baseline = xgb_classifier_baseline.score(X_test_scaled, y_test)
    auc_baseline = roc_auc_score(y_test, y_pred_proba_baseline)

    print(f"\nTest Set Performance:")
    print(f"Train Accuracy: {train_score_baseline:.4f}")
    print(f"Test Accuracy: {test_score_baseline:.4f}")
    print(f"Overfitting Gap: {(train_score_baseline - test_score_baseline):.4f}")
    print(f"Test ROC AUC: {auc_baseline:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_baseline))

    print("\n" + "="*60)
    print("DIAGNOSTIC CHECK")
    print("="*60)
    print(f"Class distribution (train):")
    print(y_train.value_counts())
    print(f"\nClass distribution (test):")
    print(y_test.value_counts())
    print(f"\nClass imbalance ratio: {y_train.value_counts()[1] / y_train.value_counts()[0]:.2f}")

    print(f"\nPrediction distribution (baseline):")
    print(f"Predicted 0: {(y_pred_baseline == 0).sum()}")
    print(f"Predicted 1: {(y_pred_baseline == 1).sum()}")

    print(f"\nProbability statistics:")
    print(f"Min probability: {y_pred_proba_baseline.min():.4f}")
    print(f"Max probability: {y_pred_proba_baseline.max():.4f}")
    print(f"Mean probability: {y_pred_proba_baseline.mean():.4f}")

    # COMPARISON PLOT with error bars
    plt.figure(figsize=(12, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.bar(['Baseline'], [cv_scores_accuracy.mean()], 
            yerr=[cv_scores_accuracy.std()], capsize=10, 
            alpha=0.7, color='skyblue', edgecolor='black', linewidth=1.5)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('CV Accuracy (5-Fold)', fontsize=12)
    plt.ylim([0.95, 1.0])
    plt.text(0, cv_scores_accuracy.mean() + cv_scores_accuracy.std() + 0.001, 
            f'{cv_scores_accuracy.mean():.4f}', ha='center', fontsize=11, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # AUC subplot
    plt.subplot(1, 2, 2)
    plt.bar(['Baseline'], [cv_scores_auc.mean()], 
            yerr=[cv_scores_auc.std()], capsize=10, 
            alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
    plt.ylabel('AUC', fontsize=12)
    plt.title('CV AUC (5-Fold)', fontsize=12)
    plt.ylim([0.95, 1.0])
    plt.text(0, cv_scores_auc.mean() + cv_scores_auc.std() + 0.001, 
            f'{cv_scores_auc.mean():.4f}', ha='center', fontsize=11, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    booster = xgb_classifier_baseline.get_booster()

    importance = booster.get_score(importance_type='gain')

    importance_df = (
        pd.DataFrame(importance.items(), columns=['feature', 'importance'])
        .sort_values(by='importance', ascending=False)
    )

    # Map feature names correctly
    feature_map = {f"f{i}": name for i, name in enumerate(feature_columns)}
    importance_df['feature'] = importance_df['feature'].map(feature_map)

    #export model
    joblib.dump(xgb_classifier_baseline, r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\models\xgb_baseline_classifier_model.joblib")

def create_regression_model(hurricane_data_for_ml, plot=True):


    hurricane_data_for_ml = hurricane_data_for_ml[ (hurricane_data_for_ml['mean_wind_radius_km'] > 0)]


    predict_column = 'mean_wind_radius_km'  # Target variable: mean wind radius in km

    feature_columns = ['LAT', 'LON', 'USA_WIND', 'USA_PRES', 'STORM_SPEED', 'Month' , 'Year', 'Day', 'DIST2LAND', 'STORM_DIR']

    # Drop rows with NaN in feature or target columns
    print(f"Total rows: {len(hurricane_data_for_ml)}")
    hurricane_data_for_ml_clean = hurricane_data_for_ml.dropna(subset=feature_columns + [predict_column])
    print(f"Clean rows (no NaN): {len(hurricane_data_for_ml_clean)}")



    X = hurricane_data_for_ml_clean[feature_columns]
    y = hurricane_data_for_ml_clean[predict_column]
    #box cox transformation
    offset = 0.1  # small constant to ensure positivity
    y_offset = y + offset
    y_transformed, fitted_lambda = boxcox(y_offset)  # add 1 to avoid log(0)

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

    y_pred_baseline = safe_inv_boxcox(y_pred_transformed, fitted_lambda, offset=0.1)  # Inverse boxcox-transform

    mse_baseline = mean_squared_error(safe_inv_boxcox(y_test, fitted_lambda, offset=0.1), y_pred_baseline)
    r2_baseline = r2_score(safe_inv_boxcox(y_test, fitted_lambda, offset=0.1), y_pred_baseline)

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
    if plot:
            
        # Plot baseline
        plt.figure(figsize=(10, 6))
        plt.scatter(safe_inv_boxcox(y_test, fitted_lambda, offset=0.1), y_pred_baseline, alpha=0.6, edgecolors='k', s=50)
        plt.plot([safe_inv_boxcox(y_test, fitted_lambda, offset=0.1).min(), safe_inv_boxcox(y_test, fitted_lambda, offset=0.1).max()], [safe_inv_boxcox(y_test, fitted_lambda, offset=0.1).min(), safe_inv_boxcox(y_test, fitted_lambda, offset=0.1).max()], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual R50 Area (km²)', fontsize=12)
        plt.ylabel('Predicted R50 Area (km²)', fontsize=12)
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
        Integer(50, 300, name='n_estimators')
    ]

    # Define objective function
    # BUG 1: In Bayesian optimization objective function
    # Currently evaluates on TRANSFORMED y_test, should be on ORIGINAL scale
    @use_named_args(space)
    def objective(max_depth, learning_rate, subsample, colsample_bytree, n_estimators):
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
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
        y_pred_original = safe_inv_boxcox(y_pred, fitted_lambda, offset=0.1)  # FIX: Transform back
        y_test_original = safe_inv_boxcox(y_test, fitted_lambda, offset=0.1)  # FIX: Transform back
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
    y_pred_best = safe_inv_boxcox(y_pred_best_transformed, fitted_lambda, offset=0.1)  # Transform back
    y_test_original = safe_inv_boxcox(y_test, fitted_lambda, offset=0.1)  # Transform back

    # BUG 2: Final model evaluation - MSE and R² on ORIGINAL scale
    mse_best = mean_squared_error(y_test_original, y_pred_best)  # FIX: Original scale
    r2_best = r2_score(y_test_original, y_pred_best)  # FIX: Original scale

    print(f"\nFinal Model Results (on original scale):")
    print(f"Mean Squared Error: {mse_best:.2f}")
    print(f"R² Score: {r2_best:.4f}")
    print(f"Improvement: +{(r2_best - r2_baseline)*100:.2f}%")
    print(f"Feature Importances:")
    importances_best = xgb_best.feature_importances_
    feature_importance_best_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances_best
    }).sort_values(by='Importance', ascending=False)
    print(feature_importance_best_df)

    if plot:
        
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

    #save model

    model_dir = r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\models"
    model_path = os.path.join(model_dir, "xgb_best_model.joblib")
    joblib.dump(xgb_best, model_path)
    #export lambda
    import pickle
    lambda_path = os.path.join(model_dir, "xgb_fitted_lambda.pkl")
    with open(lambda_path, 'wb') as f:
        pickle.dump(fitted_lambda, f)
    return xgb_best, fitted_lambda
# hurricane_data_for_ml = load_and_process_data()
# create_regression_model(hurricane_data_for_ml, plot=False)
# create_classifier_model(hurricane_data_for_ml)


aslak_data= pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\Aslak_data_with_tide_and_travelspeed.csv", low_memory=False)
#only predict prior to 2004
aslak_data['Year'] = pd.to_datetime(aslak_data['lf_ISO_TIME']).dt.year
post_2004_data = aslak_data[aslak_data['Year'] >= 2004]
post_2004_data['R50_mean_at_landfall'] = post_2004_data.apply(lambda row: calculate_mean_wind_radius(row['r50_ne_at_landfall'],row['r50_se_at_landfall'],
                                                                                                      row['r50_sw_at_landfall'], row['r50_nw_at_landfall']), axis=1)
post_2004_data['R50_mean_at_landfall'] = post_2004_data['R50_mean_at_landfall'].fillna(0)
pre_2004_data = aslak_data[aslak_data['Year'] < 2004]


import pickle
# Load the trained model, scaler, and lambda

model_classifier = joblib.load(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\models\xgb_baseline_classifier_model.joblib")

#predict which rows have R50


model_regressor = joblib.load(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\models\xgb_best_model.joblib")

#load lambda
with open(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\models\xgb_fitted_lambda.pkl", "rb") as f:
    fitted_lambda = pickle.load(f)
xgb_regressor = model_regressor

pre_2004_data = pre_2004_data.copy()
rename_dict = {
    'lf_lat': 'LAT',
    'lf_lon': 'LON',
    'lf_wind': 'USA_WIND',
    'lf_pressure': 'USA_PRES',
}
pre_2004_data['Month'] = pd.to_datetime(pre_2004_data['lf_ISO_TIME']).dt.month
pre_2004_data['Year'] = pd.to_datetime(pre_2004_data['lf_ISO_TIME']).dt.year
pre_2004_data['Day'] = pd.to_datetime(pre_2004_data['lf_ISO_TIME']).dt.day
pre_2004_data = pre_2004_data.rename(columns=rename_dict)
feature_columns = ['LAT', 'LON', 'USA_WIND', 'USA_PRES', 'STORM_SPEED', 'Month' , 'Year', 'Day', 'DIST2LAND', 'STORM_DIR']

pre_2004_data = pre_2004_data.dropna(subset=feature_columns)
scaler = StandardScaler()

X_classify = pre_2004_data[feature_columns].copy()
X_classify_scaled = scaler.fit_transform(X_classify)

# Predict which rows have R50
y_pred_classify = model_classifier.predict(X_classify_scaled)
pre_2004_data['has_R50'] = y_pred_classify
pre_2004_data['has_R50'] = pre_2004_data['has_R50'].map({1: True, 0: False})
pre_2004_data_to_predict = pre_2004_data[pre_2004_data['has_R50'] == True]
pre_2004_data_no_R50 = pre_2004_data[pre_2004_data['has_R50'] == False]
pre_2004_data_no_R50['R50_mean_at_landfall'] = 0






X_regression = pre_2004_data_to_predict[feature_columns].copy()
X_regression_scaled = scaler.fit_transform(X_regression)
# Make predictions on scaled data
y_pred_transformed = xgb_regressor.predict(X_regression_scaled)

y_pred_real = safe_inv_boxcox(y_pred_transformed, fitted_lambda, offset=0.1)
pre_2004_data_to_predict['R50_mean_at_landfall'] = y_pred_real
pre_2004_data = pd.concat([pre_2004_data_to_predict, pre_2004_data_no_R50])
pre_2004_data = pre_2004_data[pre_2004_data['ND'] > 0]

total_data = pd.concat([pre_2004_data, post_2004_data])
rename_back_dict = {
    'LAT': 'lf_lat',
    'LON': 'lf_lon',
    'USA_WIND': 'lf_wind',
    'USA_PRES': 'lf_pressure',
}
total_data = total_data.rename(columns=rename_back_dict)
total_data = total_data.sort_values(by=['lf_ISO_TIME'])
total_data.to_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\Aslak_data_with_predicted_R50_at_landfall.csv", index=False)


