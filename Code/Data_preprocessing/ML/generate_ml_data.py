import pandas as pd
import numpy as np
import joblib
import sys
import pickle
sys.path.append(r"./Speciale/Code")
from Data_preprocessing.ML.ml_utils import safe_inv_boxcox, FEATURE_COLUMNS, CONSTRUCTED_FEATURES, calculate_mean_wind_radius


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

def estimate_mean_wind_radius(df, radius):

    working_frame = df.copy()
    #rename columns to match feature names
    rename_dict = {'lf_wind':'USA_WIND', 'lf_pressure':'USA_PRES', 'lf_lat':'LAT', 'lf_lon':'LON'}
    working_frame.rename(columns=rename_dict, inplace=True)
    working_frame[radius + '_MEAN_RADIUS'] = working_frame.apply(
        lambda row: calculate_mean_wind_radius(
            row[radius + '_NE'],
            row[radius + '_SE'],
            row[radius + '_SW'],
            row[radius + '_NW']
        ), axis=1
    )
    #first split into pre and post 2004 if Radius is R34 do 2001 instead
    if radius == 'R34':
        split_year = 2001
    else:
        split_year = 2004
    working_frame["pressure_relative"] = 1023.25 - working_frame["USA_PRES"]
    working_frame["wind_pressure_ratio"] = (
        working_frame["USA_WIND"] /
        (working_frame["pressure_relative"] + 1)
    )
    feature_columns = FEATURE_COLUMNS + CONSTRUCTED_FEATURES
    feature_columns.remove('USA_PRES')

    df_preYear = working_frame[working_frame['Year'] < split_year]
    df_postYear = working_frame[working_frame['Year'] >= split_year]
    #data exists only post 2004 so calculations here are valid NAN is 0's 
    df_postYear[radius + '_MEAN_RADIUS'] = df_postYear[radius + '_MEAN_RADIUS'].fillna(0)


    #predict for pre2004 using classifiers

    #first classify if radius should be 0 or not
    model = joblib.load(f'C:\\Users\\123ti\\Documents\\Speciale_git\\Speciale\\Code\\Data_preprocessing\\ML\\classifiers\\{radius}_classifier_model.joblib')
    X_preYear = df_preYear[feature_columns]
    y_pred_class = model.predict(X_preYear)
    df_preYear['PREDICTED_' + radius + '_CLASS'] = y_pred_class
    df_preYear['PREDICTED_' + radius + '_CLASS'] = df_preYear['PREDICTED_' + radius + '_CLASS'].replace({0: False, 1: True})
    df_preYear_zero = df_preYear[df_preYear['PREDICTED_' + radius + '_CLASS'] == False]
    df_preYear_nonzero = df_preYear[df_preYear['PREDICTED_' + radius + '_CLASS'] == True]

    #set predicted zero radius to 0
    df_preYear_zero[radius + '_MEAN_RADIUS'] = 0
    #for non zero radius predict using regression model
    reg_model = joblib.load(f'C:\\Users\\123ti\\Documents\\Speciale_git\\Speciale\\Code\\Data_preprocessing\\ML\\regressors\\model_{radius}.joblib')

    X_preYear_nonzero = df_preYear_nonzero[feature_columns]
    #apply lambda transformation
    y_pred_original = reg_model.predict(X_preYear_nonzero)
    df_preYear_nonzero[radius + '_MEAN_RADIUS'] = y_pred_original
    #combine pre year dataframes
    df_preYear = pd.concat([df_preYear_zero, df_preYear_nonzero], ignore_index=False)
    df_preYear.drop(columns=['PREDICTED_' + radius + '_CLASS'], inplace=True)
    df_final = pd.concat([df_preYear, df_postYear], ignore_index=False)
    #sort by time
    df_final.sort_values(by=df_final['lf_ISO_TIME'].name, inplace=True)
    df_final = df_final[['Unique_ID', radius + '_MEAN_RADIUS']]
    return df_final

def estimate_mean_radius_for_all(df):
    radius_types = ['R34', 'R50', 'R64']
    frames = []
    for radius in radius_types:
        print(f"Estimating mean wind radius for {radius}...")
        frames.append(estimate_mean_wind_radius(df, radius))
    result_frame = frames[0]
    for i in range(1, len(frames)):
        result_frame = pd.merge(result_frame, frames[i], on='Unique_ID', how='left')
    return result_frame




