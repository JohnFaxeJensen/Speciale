import os
import pandas as pd
import numpy as np

import sys
sys.path.append(r"./Speciale/Code")

from Data_preprocessing.generate_tide import generate_tide_data
from Data_preprocessing.generate_region_temps import generate_temp_data
from Data_preprocessing.generate_ibtracks_data import generate_travel_speed_data, generate_ibtracs_data
from Data_preprocessing.ML.generate_ml_data import estimate_mean_radius_for_all
from Data_preprocessing.generate_surge import generate_surge_data, get_inspected_surge_data




def generate_csv_data(researcher = 'Aslak'):
    path = rf'C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\{researcher}_data_with_all_features.csv'
    
    #check if file already exists
    if os.path.exists(path):
        print(f"Data file for {researcher} already exists at {path}.")
        df_existing = pd.read_csv(path)
        return df_existing
    #if 'Aslak', then use aslak dataset
    if researcher == 'Aslak':
        df = pd.read_excel(r'C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\Aslak_data.xls', sheet_name = 'ATD of ICAT')
    if researcher == 'Weinkle':
        df = pd.read_excel(r'C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\Aslak_data.xls', sheet_name = 'ATD of Weinkle')
    #use the orignal aslak data to merge temp data into
    if researcher != 'Aslak' and researcher != 'Weinkle':
        raise ValueError("researcher must be either 'Aslak' or 'Weinkle'")
    #create unique id for merging
    numbers = np.arange(len(df))
    df['Unique_ID'] = ['ID_' + str(num) for num in numbers]
    #drop bad data:
    subset = ['basedamage','lf_wind', 'lf_pressure']
    df = df.dropna(subset=subset)
    #change string types to datetime
    df['lf_ISO_TIME'] = pd.to_datetime(df['lf_ISO_TIME'], format="%Y-%m-%d %H:%M:%S")
    df['Year'] = df['lf_ISO_TIME'].dt.year
    df['Month'] = df['lf_ISO_TIME'].dt.month
    df['Day'] = df['lf_ISO_TIME'].dt.day

    #first add tide data
    print('pre_tide merge: ',df.shape)
    df_for_for_tide = df[['Unique_ID','lf_lat', 'lf_lon', 'lf_ISO_TIME']]
    tide_data = generate_tide_data(df_for_for_tide)
    print('tide data shape: ',tide_data.shape)
    print(tide_data.columns)
    df = pd.merge(df, tide_data, how='left', left_on=['Unique_ID'], right_on=['Unique_ID'], suffixes=('', '_tide'))
    columns_with_suffix = [col for col in df.columns if col.endswith('_tide')]
    df = df.drop(columns_with_suffix, axis=1)
    print('post_tide merge: ',df.shape)
    #add temp data
    hadisst_frame, icoads_frame = generate_temp_data(df)

    #merge hadisst data
    print('pre_hadisst merge: ',df.shape)
    df = pd.merge(df, hadisst_frame, how='left', left_on=['Year', 'Month'], right_on=['Year', 'Month'], suffixes=('', '_hadisst'))
    columns_with_suffix = [col for col in df.columns if col.endswith('_hadisst')]
    df = df.drop(columns_with_suffix, axis=1)
    print('post_hadisst merge: ',df.shape)
    #merge icoads data
    print('pre_icoads merge: ',df.shape)
    df = pd.merge(df, icoads_frame, how='left', left_on=['Year', 'Month'], right_on=['Year', 'Month'], suffixes=('', '_icoads'))
    columns_with_suffix = [col for col in df.columns if col.endswith('_icoads')]
    df = df.drop(columns_with_suffix, axis=1)
    print('post_icoads merge: ',df.shape)
    #merge global temp data
    print('pre_global_temp merge: ',df.shape)
    df_global_ocean_anomaly = pd.read_csv('./Speciale/temp_data/data_global_ocean_temp.csv', skiprows=3)
    df = pd.merge(df, df_global_ocean_anomaly, how='left', left_on='Year', right_on='Year')
    df.rename({'Anomaly': 'Anomaly_global'}, axis=1, inplace=True)
    print('post_global_temp merge: ',df.shape)
    #merge ibtracs data for travelspeed
    relevant_columns_for_travelspeed = ['ATCF_ID', 'lf_ISO_TIME', 'lf_lat', 'lf_lon', 'Unique_ID']
    travelspeed_df = generate_travel_speed_data(df[relevant_columns_for_travelspeed])
    print("Travelspeed data shape:", travelspeed_df.shape)
    df = pd.merge(df, travelspeed_df, how='left', left_on=['Unique_ID'], right_on=['Unique_ID'], suffixes=('', '_travelspeed'))
    columns_with_suffix = [col for col in df.columns if col.endswith('_travelspeed')]
    df = df.drop(columns_with_suffix, axis=1)
    print("post travelspeed shape:", df.shape)
    ibtracs_data = generate_ibtracs_data(df[relevant_columns_for_travelspeed])
    print("Ibtracs data shape:", ibtracs_data.shape)

    df = pd.merge(df, ibtracs_data, how='left', left_on=['Unique_ID'], right_on=['Unique_ID'], suffixes=('', '_ibtracs'))
    columns_with_suffix = [col for col in df.columns if col.endswith('_ibtracs')]
    df = df.drop(columns_with_suffix, axis=1)  
    print("post ibtracs shape:", df.shape)
    #convert feature columns to numeric
    df['lf_wind'] = pd.to_numeric(df['lf_wind'], errors='coerce')
    df['lf_pressure'] = pd.to_numeric(df['lf_pressure'], errors='coerce')
    df['lf_lat'] = pd.to_numeric(df['lf_lat'], errors='coerce')
    df['lf_lon'] = pd.to_numeric(df['lf_lon'], errors='coerce')
    df['STORM_SPEED_ms'] = pd.to_numeric(df['STORM_SPEED_ms'], errors='coerce')
    df['DIST2LAND_m'] = pd.to_numeric(df['DIST2LAND_m'], errors='coerce')
    df['STORM_DIR'] = pd.to_numeric(df['STORM_DIR'], errors='coerce')
    df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
    df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    radius_data = estimate_mean_radius_for_all(df)
    print("Radius data shape:", radius_data.shape)
    df = pd.merge(df, radius_data, how='left', left_on=['Unique_ID'], right_on=['Unique_ID'], suffixes=('', '_radius'))
    columns_with_suffix = [col for col in df.columns if col.endswith('_radius')]
    df = df.drop(columns_with_suffix, axis=1)
    print("post radius shape:", df.shape)

    #try with the risk factor columns too
    relevant_states = df['lf_state'].unique().tolist()

    

    risk_df = pd.read_csv("C:\\Users\\123ti\\Downloads\\National_Risk_Index_County_Hurricane_Expected_Annual_Loss_Rating_985662790059011280.csv")
    risk_df = risk_df[risk_df['State Name Abbreviation'].isin(relevant_states)]
    risk_df = risk_df[['State Name Abbreviation', 'Hurricane - Hazard Type Risk Index Score']]
    #calculate mean risk score per state
    state_risk_scores = risk_df.groupby('State Name Abbreviation')['Hurricane - Hazard Type Risk Index Score'].mean().reset_index()
    state_risk_scores.rename(columns={'State Name Abbreviation': 'lf_state', 'Hurricane - Hazard Type Risk Index Score': 'Hurricane_Risk_Score'}, inplace=True)
    
    #merge risk scores into main df
    print('pre_risk_score merge: ',df.shape)
    df = pd.merge(df, state_risk_scores, how='left', left_on=['lf_state'], right_on=['lf_state'], suffixes=('', '_risk'))
    columns_with_suffix = [col for col in df.columns if col.endswith('_risk')]
    df = df.drop(columns_with_suffix, axis=1)
    print('post_risk_score merge: ',df.shape)
    #merge storm surge data 
    print('pre_surge_data merge: ',df.shape)
    # Try UTF-8 first, fall back to latin-1
    surge_data = get_inspected_surge_data()
    print("Surge data shape:", surge_data.shape)
    df = pd.merge(df, surge_data, how='left', left_on=['Unique_ID'], right_on=['Unique_ID'], suffixes=('', '_surge'))
    columns_with_suffix = [col for col in df.columns if col.endswith('_surge')]
    df = df.drop(columns_with_suffix, axis=1)
    print('post_surge_data merge: ',df.shape)
    #final csv output
    df.to_csv(path, index=False)
    return df

generate_csv_data()