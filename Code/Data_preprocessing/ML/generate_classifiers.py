
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r"./Speciale/Code")
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from Data_preprocessing.ML.ml_utils import FEATURE_COLUMNS, CONSTRUCTED_FEATURES
from sklearn.model_selection import GroupShuffleSplit

import joblib
import generate_training_data


training_data = generate_training_data.get_training_data()
def create_classifier(radius):
    path = r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\ML\classifiers\\" + radius + "_classifier_model.joblib"
    if os.path.exists(path):
        print(f"Classifier for {radius} already exists. Loading from file.")
        fitted_model = joblib.load(path)
        return fitted_model

    training_data['has_' + radius] = training_data[radius + '_MEAN_RADIUS'] > 0

    print(training_data[training_data['has_' + radius] == True].shape)
    print(training_data[training_data['has_' + radius] == False].shape)
                        
    #create classifier for zero mean wind radius
    predict_column = 'has_' + radius  # Target variable: mean wind radius in km



    training_data_clean = training_data.dropna(subset=FEATURE_COLUMNS + [predict_column])
    training_data_clean["pressure_relative"] = 1023.25 - training_data_clean["USA_PRES"]
    training_data_clean["wind_pressure_ratio"] = (
        training_data_clean["USA_WIND"] /
        (training_data_clean["pressure_relative"] + 1)
    )
    feature_columns = FEATURE_COLUMNS + CONSTRUCTED_FEATURES
    feature_columns.remove('USA_PRES')
    X = training_data_clean[feature_columns]
    y = training_data_clean[predict_column]

    # Use groups from cleaned data to match X and y indices
    groups = training_data_clean['USA_ATCF_ID'].values
    
    # Try to find a split where both classes are present in training set
    train_idx, test_idx = None, None
    for attempt in range(10):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42 + attempt)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        # Check if both classes are present in training set
        if len(np.unique(y.iloc[train_idx])) == 2:
            break
    
    # If no suitable split found, fall back to stratified split
    if train_idx is None or len(np.unique(y.iloc[train_idx])) != 2:
        print("Warning: Using stratified split as fallback (group distribution couldn't guarantee both classes)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        # Split data into training and testing sets
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]



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
        xgb_classifier_baseline, X_train, y_train, 
        cv=kfold, scoring='accuracy', n_jobs=-1
    )

    cv_scores_auc = cross_val_score(
        xgb_classifier_baseline, X_train, y_train, 
        cv=kfold, scoring='roc_auc', n_jobs=-1
    )

    print(f"CV Accuracy Scores: {cv_scores_accuracy}")
    print(f"Mean CV Accuracy: {cv_scores_accuracy.mean():.4f} (+/- {cv_scores_accuracy.std():.4f})")
    print(f"\nCV AUC Scores: {cv_scores_auc}")
    print(f"Mean CV AUC: {cv_scores_auc.mean():.4f} (+/- {cv_scores_auc.std():.4f})")

    # Train on full training set for final test evaluation
    xgb_classifier_baseline.fit(
        X_train, y_train, 
        eval_set=[(X_test, y_test)], 
        verbose=False
    )

    y_pred_baseline = xgb_classifier_baseline.predict(X_test)
    y_pred_proba_baseline = xgb_classifier_baseline.predict_proba(X_test)[:, 1]


    train_score_baseline = xgb_classifier_baseline.score(X_train, y_train)
    test_score_baseline = xgb_classifier_baseline.score(X_test, y_test)
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
    
    # Calculate imbalance ratio only if both classes are present
    if 0 in y_train.value_counts().index and 1 in y_train.value_counts().index:
        imbalance_ratio = y_train.value_counts()[1] / y_train.value_counts()[0]
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
    else:
        print(f"\nWarning: Not all classes present in training data")

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
    print("\nFeature Importance (Gain):")
    print(importance_df)

    # Map feature names correctly
    feature_map = {f"f{i}": name for i, name in enumerate(feature_columns)}
    importance_df['feature'] = importance_df['feature'].map(feature_map)

    #export model
    joblib.dump(xgb_classifier_baseline, path)

radii_list = ['R34', 'R50', 'R64',]
for radius in radii_list:
    print(f"\n\nCreating classifier for {radius}...")
    create_classifier(radius)
