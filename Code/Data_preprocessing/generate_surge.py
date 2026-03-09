
import pandas as pd
import os
from datetime import datetime
import re
import requests
import numpy as np
import sys
sys.path.append(r"./Speciale/Code")

from Data_preprocessing.surge_preprocess.convert_datums import convert_datums
from Data_preprocessing.sea_level_rise import compute_epoch_fix
from Data_preprocessing.generate_tide import generate_tidal_ranges

def get_tide_instrument_error(year):
    one_foot_in_meters = 0.3048
    """Measurement error based on era and technology"""
    if year < 1950:
        return one_foot_in_meters  # ±1 foot (early manual gauges)
    elif year < 1980:
        return 0.5*one_foot_in_meters   # ±10 cm (analog electronic)
    else:
        return 0.1   # ±5 cm (modern digital)


def estimate_surge_and_uncertainty(row):
    
    uncertainty = 0
    uncertainty += get_tide_instrument_error(row['Year'])**2
    if row['Surge_observed']:
        return  row['Surge_m'], np.sqrt(uncertainty)
    else:
        converted =  not pd.isna(row['Converted_MSL'])
        if converted:
            storm_tide = row['Converted_MSL']
        else: 
            storm_tide = row['Storm_Tide_m']
        print(storm_tide)
        surge_estimate = storm_tide - 0.5 * row['Tidal_Range_peak']
        print(surge_estimate)
        if (row['Year'] > 2001 and row['Datum'] in ['NAVD88', 'NGVD29']) or row['Year'] < 1983:
            surge_estimate += row['Offset_to_1983_2001_epoch_m']
            uncertainty += row['Error_Offset_m']**2
        print(f"Surge estimate after epoch correction: {surge_estimate}, with uncertainty: {np.sqrt(uncertainty)}")
        #add extra uncertainties
        if not pd.isna(row['Converted_uncertainty_MSL']):
            uncertainty += row['Converted_uncertainty_MSL']**2
        tidal_diff_percentage = np.array([0,0,0.023448276,0.255715495,0.100917431,0,0,0,0.531958763,0.161637931,0.050701187,0.131818182,0.02739726,0.203870968]) #from excel file
        # Calculate statistics
        if row['Datum'] != 'gauge':
            std_diff = np.std(tidal_diff_percentage)
            uncertainty += (std_diff * storm_tide)**2
        # add datum uncertainty if datum is unknown
        if row['Datum'] == 'Unknown':
            uncertainty += (0.33 * row['Tidal_Range_peak'])**2  # assume 50% of tidal range as additional uncertainty for unknown datums¨
        #maybe add extra location based uncertainty. 
        return surge_estimate, np.sqrt(uncertainty)

def get_inspected_data_full_conversion(select_out_of_range=False, convert_datum=True, above_normal_as_surge=False):
    path = r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\generated_data\surge_data_manual_check.xlsx"

    df_existing = pd.read_excel(path)
    if above_normal_as_surge:
        #convert storm tides marked as 'Above Normal' datum to surges
        for idx, row in df_existing.iterrows():
            if row['Datum'] == 'Above Normal':
                if pd.isna(row['Surge_m']) and not pd.isna(row['Storm_Tide_m']):
                    df_existing.at[idx, 'Surge_m'] = row['Storm_Tide_m']
    else:
        #convert 'Above Normal' to 'Unknown' datum because we cannot be sure of the datum
        df_existing['Datum'] = df_existing['Datum'].replace('Above Normal', 'Unknown')
    valid_manual_checks = ['yes', 'surge only', 'gauge']
    if select_out_of_range:
        valid_manual_checks.append('out of range')
    df_existing = df_existing[df_existing['Manual check'].isin(valid_manual_checks)]
    #add tidal range data
    tidal_ranges_df = generate_tidal_ranges()
    #merge on Unique_ID
    df_existing = df_existing.merge(tidal_ranges_df[['Unique_ID', 'Tidal_Range_peak']], left_on='Unique_ID', right_on='Unique_ID', how='left')
    if convert_datum:
        converted_df = convert_datums()
        df_existing = df_existing.merge(converted_df[['Unique_ID', 'Converted_value', 'Converted_uncertainty']], on='Unique_ID', how='left')
        df_existing.rename(columns={'Converted_value': 'Converted_MSL', 'Converted_uncertainty': 'Converted_uncertainty_MSL'}, inplace=True)
    if not convert_datum:
        df_existing['Converted_MSL'] = np.nan
        df_existing['Converted_uncertainty_MSL'] = np.nan
    df_existing['Datum'] = df_existing['Datum'].replace(pd.NA, 'Unknown')
    epoch_correction_df = compute_epoch_fix()
    df_existing = df_existing.merge(epoch_correction_df[['Unique_ID', 'Offset_to_1983_2001_epoch_m', 'Error_Offset_m']], on='Unique_ID', how='left')
    #add mask that indicates whether surge value is observed or calculated based on storm tide and tidal range
    surge_observed_mask = ~df_existing['Surge_m'].isna()
    df_existing['Surge_observed'] = surge_observed_mask
    unique_ids = df_existing['Unique_ID'].values
    surge_guesses = []
    uncertainties = []
    for idx, row in df_existing.iterrows():
        surge_estimate, surge_uncertainty = estimate_surge_and_uncertainty(row)
        surge_guesses.append(surge_estimate)
        uncertainties.append(surge_uncertainty)
    
    return unique_ids,surge_guesses,uncertainties

def get_inspected_data_raw_scenarios():
    path = r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\generated_data\surge_data_manual_check.xlsx"
    df_existing = pd.read_excel(path)
    valid_manual_checks = ['yes', 'surge only', 'gauge']
    df_existing = df_existing[df_existing['Manual check'].isin(valid_manual_checks)]
    tidal_ranges_df = generate_tidal_ranges()
    df_existing = df_existing.merge(tidal_ranges_df[['Unique_ID', 'Tidal_Range_peak']], left_on='Unique_ID', right_on='Unique_ID', how='left')
    #1st scenario: use surge and storm tide as the same variable
    ids = []
    surge_values_first = []
    uncertainties_first = []
    for idx, row in df_existing.iterrows():
        ids.append(row['Unique_ID'])
        uncertainties_first.append(get_tide_instrument_error(row['Year']))
        if pd.isna(row['Surge_m']):
            surge_values_first.append(row['Storm_Tide_m'])
        else:
            surge_values_first.append(row['Surge_m'])
    df_first = pd.DataFrame({
        'Unique_ID': ids,
        'Surge_m': surge_values_first,
        'Uncertainty_m': uncertainties_first
    })
    # second scenario: try to assume that measurements happens at peak tide
    ids = []
    surge_values_second = []
    uncertainties_second = []
    for idx, row in df_existing.iterrows():
        ids.append(row['Unique_ID'])
        uncertainties_second.append(get_tide_instrument_error(row['Year']))
        if pd.isna(row['Surge_m']):
            surge_values_second.append(row['Storm_Tide_m'] - 0.5*row['Tidal_Range_peak'])
        else:
            surge_values_second.append(row['Surge_m'])
    df_second = pd.DataFrame({
        'Unique_ID': ids,
        'Surge_m': surge_values_second,
        'Uncertainty_m': uncertainties_second
    })
    merged_df = pd.merge(df_first, df_second, on='Unique_ID', suffixes=('_storm_tide_as_surge', '_subtract_peak_tide'))
    return merged_df

def generate_surge_data():
    path = r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\generated_data\surge_data_all_scenarios.csv"
    if os.path.exists(path):
        print(f"Surge data file already exists at {path}.")
        return pd.read_csv(path)
    else:
        #generate surge data and uncertainties for all scenarios and save to csv
        #first scenario: 'above normal' is treated as surge, and conversion is done for known datums
        unique_ids_first, surge_guesses_first, uncertainties_first = get_inspected_data_full_conversion(select_out_of_range=False, convert_datum=True, above_normal_as_surge=True)
        df_surge_first = pd.DataFrame({
            'Unique_ID': unique_ids_first,
            'Surge_m': surge_guesses_first,
            'Uncertainty_m': uncertainties_first
        })
        #second scenario treat above normal as unknown datum 
        unique_ids_second, surge_guesses_second, uncertainties_second = get_inspected_data_full_conversion(select_out_of_range=False, convert_datum=True, above_normal_as_surge=False)
        df_surge_second = pd.DataFrame({
            'Unique_ID': unique_ids_second,
            'Surge_m': surge_guesses_second,
            'Uncertainty_m': uncertainties_second
        })
        merged = df_surge_first.merge(df_surge_second, on='Unique_ID', suffixes=('_full_conversion_AN_as_surge', '_full_conversion_AN_as_unknown'))
        #next two scenarios are merely to test the roboustness of the conversion in regards to if it even 
        #add anything to the model parameters or that the data itself is mostly driven by surge and not trends
        #coming from slr corrections, datum corrections or unknown datums. which just infuses more variability to
        #the model parameters
        raw_scenarios = get_inspected_data_raw_scenarios()

        merged = merged.merge(raw_scenarios, on='Unique_ID', how='left')
        merged.to_csv(path, index=False)

        return merged