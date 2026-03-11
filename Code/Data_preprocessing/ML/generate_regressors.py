
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
import sys
sys.path.append(r"./Speciale/Code")
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from scipy.stats import boxcox

import joblib
import generate_training_data
import pickle

from Data_preprocessing.ML.ml_utils import safe_inv_boxcox, FEATURE_COLUMNS, CONSTRUCTED_FEATURES



training_data = generate_training_data.get_training_data()

random_session = 43
def create_regressor(radius):
    model_dir = r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\ML\regressors"
    model_path = os.path.join(model_dir, f"model_{radius}.joblib")
    # if os.path.exists(model_path):
    #     print(f"Regressor for {radius} already exists. Loading from file.")
    #     fitted_model = joblib.load(model_path)
    #     lambda_path = os.path.join(model_dir, f"fitted_lambda_{radius}.pkl")
    #     with open(lambda_path, 'rb') as f:
    #         fitted_lambda = pickle.load(f)
    #     return fitted_model, fitted_lambda

    training_data_filtered = training_data[ (training_data[radius + '_MEAN_RADIUS'] > 0)]

    predict_column = radius + '_MEAN_RADIUS'  # Target variable: mean wind radius in km

    training_data_clean = training_data_filtered.dropna(subset=FEATURE_COLUMNS + [predict_column])
    training_data_clean["pressure_relative"] = 1023.25 - training_data_clean["USA_PRES"]
    training_data_clean["wind_pressure_ratio"] = (
        training_data_clean["USA_WIND"] /
        (training_data_clean["pressure_relative"] + 1)
    )
    feature_columns = FEATURE_COLUMNS + CONSTRUCTED_FEATURES
    feature_columns.remove('USA_PRES')
    #print max and min of new features to check for outliers

    X = training_data_clean[feature_columns]
    y = training_data_clean[predict_column]
    #plot predict distribution
    plt.figure(figsize=(8, 6))
    plt.hist(y, bins=30, alpha=0.7, label='Actual')
    plt.xlabel('Mean Wind Radius (km)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {predict_column}')
    plt.legend()
    plt.savefig(os.path.join(model_dir, f"{radius}_distribution.png"))
    plt.close()
    #box cox transformation
    offset = 0.1  # small constant to ensure positivity
    y_offset = y + offset
    y_transformed, fitted_lambda = boxcox(y_offset)  # add 1 to avoid log(0)

    groups = training_data_clean["USA_ATCF_ID"]  # Group by storm ID to prevent data leakage

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_session)

    train_idx, test_idx = next(
        gss.split(X, y_transformed, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y_transformed[train_idx]
    y_test = y_transformed[test_idx]


    # Baseline model
    print("\n" + "="*60)
    print("BASELINE XGBOOST")
    print("="*60)

    xgb_baseline = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1,
        reg_alpha=0.1,
        random_state=random_session,
        verbosity=0,
        early_stopping_rounds=30
    )

    xgb_baseline.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred_transformed = xgb_baseline.predict(X_test)

    y_pred_baseline = safe_inv_boxcox(y_pred_transformed, fitted_lambda, offset=0.1)  # Inverse boxcox-transform

    mae = mean_absolute_error(safe_inv_boxcox(y_test, fitted_lambda, offset=0.1), y_pred_baseline)
    r2_baseline = r2_score(safe_inv_boxcox(y_test, fitted_lambda, offset=0.1), y_pred_baseline)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2_baseline:.4f}")

    # Feature importance
    importances = xgb_baseline.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:")
    print(feature_importance_df)

    # Bayesian Optimization
    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION")
    print("="*60)

    # Define search space
    space = [
        Integer(3, 15, name='max_depth'),
        Real(0.001, 0.3, name='learning_rate', prior='log-uniform'),
        Real(0.5, 1.0, name='subsample'),
        Real(0.5, 1.0, name='colsample_bytree'),
        Integer(50, 3000, name='n_estimators'),
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
            random_state=random_session,
            verbosity=0,
            early_stopping_rounds=20
        )
        
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = model.predict(X_test)
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
        random_state=random_session,
        verbose=1,
        n_initial_points=10
    )

    print(f"\nBest R² found: {-result.fun:.4f}")
    print("Best parameters:")
    best_params = {
        'max_depth': result.x[0],
        'learning_rate': result.x[1],
        'subsample': result.x[2],
        'colsample_bytree': result.x[3],
        'n_estimators': result.x[4]
    }
    for key, val in best_params.items():
        print(f"  {key}: {val}")

    # Train final model with best parameters
    print("\nTraining final model with optimized parameters...")
    xgb_best = xgb.XGBRegressor(
        random_state=random_session,
        verbosity=0,
        early_stopping_rounds=20,
        **best_params
    )
    gkf = GroupKFold(n_splits=5)
    scores = []

    for train_idx_cv, test_idx_cv in gkf.split(X, y_transformed, groups):

        X_train_cv = X.iloc[train_idx_cv]
        X_test_cv = X.iloc[test_idx_cv]

        y_train_cv = y_transformed[train_idx_cv]
        y_test_cv = y_transformed[test_idx_cv]

        xgb_best.fit(X_train_cv, y_train_cv,
                eval_set=[(X_test_cv, y_test_cv)],
                verbose=False)

        y_pred = xgb_best.predict(X_test_cv)

        y_pred_original = safe_inv_boxcox(y_pred, fitted_lambda, offset=0.1)
        y_test_original = safe_inv_boxcox(y_test_cv, fitted_lambda, offset=0.1)

        scores.append((r2_score(y_test_original, y_pred_original), mean_absolute_error(y_test_original, y_pred_original)))
    #print cv mean and std of results
    # Calculate mean and std of R² scores
    r2_scores = [s[0] for s in scores]
    mae_scores = [s[1] for s in scores]
    print(f"\nR² Summary: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"R² Range: {np.min(r2_scores):.4f} (min) to {np.max(r2_scores):.4f} (max)")
    print(f"MAE Summary: {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}\n")

    print(f"Feature Importances:")
    importances_best = xgb_best.feature_importances_
    feature_importance_best_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances_best
    }).sort_values(by='Importance', ascending=False)
    print(feature_importance_best_df)
    #plot predicted vs actual
    # plt.figure(figsize=(8, 6))
    # plt.scatter(y_test_original, y_pred_original, alpha=0.5)
    # plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--')
    # plt.xlabel('Actual Mean Wind Radius (km)')
    # plt.ylabel('Predicted Mean Wind Radius (km)')
    # plt.title(f'Predicted vs Actual Mean Wind Radius for {radius}')
    # plt.show()


    #save model

    joblib.dump(xgb_best, model_path)
    #export lambda

    lambda_path = os.path.join(model_dir, f"fitted_lambda_{radius}.pkl")
    with open(lambda_path, 'wb') as f:
        pickle.dump(fitted_lambda, f)
    return xgb_best, fitted_lambda

# radii_list = ['R34', 'R50', 'R64',]
# for radius in radii_list:
#     print(f"\n\nCreating regressor for {radius}...")
#     create_regressor(radius)
# quit()
def create_regressor_no_boxcox(radius):

    model_dir = r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\ML\regressors"
    model_path = os.path.join(model_dir, f"model_{radius}.joblib")
    if os.path.exists(model_path):
        print(f"Regressor for {radius} already exists. Loading from file.")
        fitted_model = joblib.load(model_path)
        return fitted_model

    training_data_filtered = training_data[(training_data[radius + '_MEAN_RADIUS'] > 0)]

    predict_column = radius + '_MEAN_RADIUS'

    training_data_clean = training_data_filtered.dropna(subset=FEATURE_COLUMNS + [predict_column])

    training_data_clean["pressure_relative"] = 1023.25 - training_data_clean["USA_PRES"]

    training_data_clean["wind_pressure_ratio"] = (
        training_data_clean["USA_WIND"] /
        (training_data_clean["pressure_relative"] + 1)
    )

    feature_columns = FEATURE_COLUMNS + CONSTRUCTED_FEATURES
    feature_columns.remove("USA_PRES")

    X = training_data_clean[feature_columns]
    y = training_data_clean[predict_column]

    # Plot target distribution
    plt.figure(figsize=(8, 6))
    plt.hist(y, bins=30, alpha=0.7, label="Actual")
    plt.xlabel("Mean Wind Radius (km)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {predict_column}")
    plt.legend()
    plt.savefig(os.path.join(model_dir, f"{radius}_distribution.png"))
    plt.close()

    groups = training_data_clean["USA_ATCF_ID"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_session)

    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Baseline model
    print("\n" + "="*60)
    print("BASELINE XGBOOST")
    print("="*60)

    xgb_baseline = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1,
        reg_alpha=0.1,
        random_state=random_session,
        verbosity=0,
        early_stopping_rounds=30
    )

    xgb_baseline.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred_baseline = xgb_baseline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_baseline)
    r2_baseline = r2_score(y_test, y_pred_baseline)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2_baseline:.4f}")

    # Feature importance
    importances = xgb_baseline.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importances:")
    print(feature_importance_df)

    # Bayesian Optimization
    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION")
    print("="*60)

    space = [
        Integer(3, 15, name="max_depth"),
        Real(0.001, 0.3, name="learning_rate", prior="log-uniform"),
        Real(0.5, 1.0, name="subsample"),
        Real(0.5, 1.0, name="colsample_bytree"),
        Integer(50, 3000, name="n_estimators"),
    ]

    @use_named_args(space)
    def objective(max_depth, learning_rate, subsample, colsample_bytree, n_estimators):

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_session,
            verbosity=0,
            early_stopping_rounds=20
        )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)

        return -r2

    print("Searching for optimal hyperparameters...")

    result = gp_minimize(
        objective,
        space,
        n_calls=30,
        random_state=random_session,
        verbose=1,
        n_initial_points=10
    )

    print(f"\nBest R² found: {-result.fun:.4f}")

    best_params = {
        "max_depth": result.x[0],
        "learning_rate": result.x[1],
        "subsample": result.x[2],
        "colsample_bytree": result.x[3],
        "n_estimators": result.x[4]
    }

    for key, val in best_params.items():
        print(f"{key}: {val}")

    print("\nTraining final model with optimized parameters...")

    xgb_best = xgb.XGBRegressor(
        random_state=random_session,
        verbosity=0,
        early_stopping_rounds=20,
        **best_params
    )

    gkf = GroupKFold(n_splits=5)

    scores = []

    for train_idx_cv, test_idx_cv in gkf.split(X, y, groups):

        X_train_cv = X.iloc[train_idx_cv]
        X_test_cv = X.iloc[test_idx_cv]

        y_train_cv = y.iloc[train_idx_cv]
        y_test_cv = y.iloc[test_idx_cv]

        xgb_best.fit(
            X_train_cv,
            y_train_cv,
            eval_set=[(X_test_cv, y_test_cv)],
            verbose=False
        )

        y_pred = xgb_best.predict(X_test_cv)

        scores.append((
            r2_score(y_test_cv, y_pred),
            mean_absolute_error(y_test_cv, y_pred)
        ))

    print(f"Cross-validated scores: {scores}")
    
    # Calculate mean and std of R² scores
    r2_scores = [s[0] for s in scores]
    mae_scores = [s[1] for s in scores]
    print(f"\nR² Summary: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"R² Range: {np.min(r2_scores):.4f} (min) to {np.max(r2_scores):.4f} (max)")
    print(f"MAE Summary: {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}\n")

    print("Feature Importances:")

    importances_best = xgb_best.feature_importances_

    feature_importance_best_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importances_best
    }).sort_values(by="Importance", ascending=False)

    print(feature_importance_best_df)

    # Save model
    joblib.dump(xgb_best, model_path)

    return xgb_best

radii_list = ['R34', 'R50', 'R64',]
for radius in radii_list:
    print(f"\n\nCreating regressor for {radius}...")
    create_regressor_no_boxcox(radius)
