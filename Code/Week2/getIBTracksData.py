from ibtracs import Ibtracs
import os
import pandas as pd
import numpy as np
from geopy import distance
import matplotlib.pyplot as plt


# raw_hurricane_data = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\ibtracs.ALL.list.v04r01.csv", usecols=["ISO_TIME", "LAT", "LON", "WMO_WIND", "WMO_PRES","USA_ATCF_ID", "USA_LAT", "USA_LON", 
#                                                                                                    "USA_WIND", "USA_PRES", "STORM_SPEED", "USA_R64_NE", "USA_R64_SE", "USA_R64_SW", "USA_R64_NW",
#                                                                                                    "USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW",
#                                                                                                    "USA_R50_NE", "USA_R50_SE", "USA_R50_SW", "USA_R50_NW",
#                                                                                                    'DIST2LAND', 'STORM_DIR'],low_memory=False)
# hurricane_data = pd.read_csv(r"./Speciale/Hurricane_data/Aslak_data_with_tide.csv")
# unique_ids = hurricane_data['ATCF_ID'].unique()
# filtered_hurricane_data = raw_hurricane_data[raw_hurricane_data['USA_ATCF_ID'].isin(unique_ids)]
# #save to csv
# filtered_hurricane_data.to_csv(r"\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\IBTrACS_filtered_data.csv", index=False)

# quit()

def safe_inv_boxcox(y_transformed, lambda_param, offset=0.1):
    """Inverse Box-Cox transform and remove offset, handling NaN values"""
    y_original = inv_boxcox(y_transformed, lambda_param)
    y_original = np.maximum(y_original - offset, 0)  # Ensure non-negative
    # Replace any NaN with 0
    y_original = np.where(np.isnan(y_original), 0, y_original)
    return y_original
def calculate_travelspeed_and_radii(hurricane_id, landfall_time, landfall_lat, landfall_lon, Filtered_IBTrACS_data, radii_types=['r34', 'r50', 'r64']):
    """
    Calculate travel speed and wind radii at landfall for multiple radius types.
    
    Parameters:
    -----------
    radii_types : list of str
        Wind radii to extract: 'r34', 'r50', 'r64', etc.
        
    Returns:
    --------
    dict with speeds and radii for all requested types
    """
    times = Filtered_IBTrACS_data['ISO_TIME']
    lats = Filtered_IBTrACS_data['LAT'].values
    lons = Filtered_IBTrACS_data['LON'].values
    storm_speeds = Filtered_IBTrACS_data['STORM_SPEED'].values
    dist_2_land = Filtered_IBTrACS_data['DIST2LAND'].values
    storm_dirs = Filtered_IBTrACS_data['STORM_DIR'].values

    if hurricane_id == "AL051965":
        landfall_time = pd.to_datetime("1965-09-29 20:00:00")
    else:
        landfall_time = pd.to_datetime(landfall_time)
    
    times_before = times[times < landfall_time]
    times_after = times[times > landfall_time]
    
    
    if len(times_before) == 0 or len(times_after) == 0:
        print(f"Not enough data before/after landfall for {hurricane_id}")
        return None

    before_time = times_before.max()
    after_time = times_after.min()
    
    
    before_index = np.where(times == before_time)[0][0]
    after_index = np.where(times == after_time)[0][0]
    before_lat = lats[before_index]
    before_lon = lons[before_index]
    after_lat = lats[after_index]
    after_lon = lons[after_index]


    time_delta_before = landfall_time - before_time
    time_delta_after = after_time - landfall_time
    
    closest_index = before_index if time_delta_before <= time_delta_after else after_index
    
    ibtracs_speed_at_landfall = storm_speeds[closest_index]
    dist2land_at_landfall = dist_2_land[closest_index]
    storm_dir_at_landfall = storm_dirs[closest_index]
    landfall_geo = (landfall_lat, landfall_lon)
    before_geo = (before_lat, before_lon)
    after_geo = (after_lat, after_lon)
    
    
    delta_dist_before = distance.distance(landfall_geo, before_geo).m
    delta_dist_after = distance.distance(landfall_geo, after_geo).m
    delta_time_before = abs((landfall_time - before_time).total_seconds())
    delta_time_after = abs((after_time - landfall_time).total_seconds())

    speed_before = delta_dist_before / delta_time_before
    speed_after = delta_dist_after / delta_time_after
    speed_at_landfall = distance.distance(before_geo, after_geo).m / (delta_time_before + delta_time_after)
    
    # Initialize results dictionary
    results = {
        'speed_before': speed_before,
        'speed_after': speed_after,
        'speed_at_landfall': speed_at_landfall,
        'STORM_SPEED': ibtracs_speed_at_landfall,
        'DIST2LAND': dist2land_at_landfall,
        'STORM_DIR': storm_dir_at_landfall
    }

    # Extract radii for all requested types
    for radii_type in radii_types:
        radii_type_upper = radii_type.upper()
        try:
            ne = Filtered_IBTrACS_data[f'USA_{radii_type_upper}_NE'].values[closest_index]
            se = Filtered_IBTrACS_data[f'USA_{radii_type_upper}_SE'].values[closest_index]
            sw = Filtered_IBTrACS_data[f'USA_{radii_type_upper}_SW'].values[closest_index]
            nw = Filtered_IBTrACS_data[f'USA_{radii_type_upper}_NW'].values[closest_index]
            
            results[f'{radii_type}_ne'] = ne
            results[f'{radii_type}_se'] = se
            results[f'{radii_type}_sw'] = sw
            results[f'{radii_type}_nw'] = nw
        except KeyError:
            print(f"Warning: {radii_type_upper} columns not found in IBTrACS data")
    
    return results


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


def calculate_mean_radius(ne, se, sw, nw):
    """Calculate mean radius from 4 quadrants"""
    try:
        ne, se, sw, nw = float(ne), float(se), float(sw), float(nw)
        if any(x < 0 for x in [ne, se, sw, nw]):
            return np.nan
        return np.mean([ne, se, sw, nw])
    except (ValueError, TypeError):
        return np.nan
# Main execution
hurricane_data = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\Aslak_data_with_tide.csv")
relevant_data = hurricane_data[['ATCF_ID', 'lf_ISO_TIME', 'lf_wind', 'lf_pressure', 'lf_lat', 'lf_lon']]
Ibtracs_data = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\IBTrACS_filtered_data.csv")
Ibtracs_data['ISO_TIME'] = pd.to_datetime(Ibtracs_data['ISO_TIME'], errors='coerce')

# Define which radii to extract
RADII_TYPES = ['r34', 'r50', 'r64']

# Initialize lists for all outputs
speeds_before = []
speeds_after = []
speeds_at_landfall = []
Ibtracs_speeds_at_landfall = []
storm_dirs_at_landfall = []
dist2land_at_landfall = []
radii_data = {rt: {'ne': [], 'se': [], 'sw': [], 'nw': [], 'area': [], 'max': [], 'mean': []} for rt in RADII_TYPES}

# Process each hurricane
Ibtracs_speeds_at_landfall = []
radii_data = {rt: {'ne': [], 'se': [], 'sw': [], 'nw': [], 'area': [], 'max': [], 'mean': []} for rt in RADII_TYPES}

# Process each hurricane
for index, row in relevant_data.iterrows():
    hurricane_id = row['ATCF_ID']
    landfall_time = row['lf_ISO_TIME']
    landfall_lat = row['lf_lat']
    landfall_lon = row['lf_lon']
    
    
    filtered_IBTrACS_data = Ibtracs_data[Ibtracs_data['USA_ATCF_ID'] == hurricane_id]
    results = calculate_travelspeed_and_radii(hurricane_id, landfall_time, landfall_lat, landfall_lon, 
                                               filtered_IBTrACS_data, RADII_TYPES)
    
    if results is None:
        speeds_before.append(None)
        speeds_after.append(None)
        speeds_at_landfall.append(None)
        Ibtracs_speeds_at_landfall.append(None)
        storm_dirs_at_landfall.append(None)
        dist2land_at_landfall.append(None)
        for rt in RADII_TYPES:
            radii_data[rt]['ne'].append(None)
            radii_data[rt]['se'].append(None)
            radii_data[rt]['sw'].append(None)
            radii_data[rt]['nw'].append(None)
            radii_data[rt]['area'].append(None)
            radii_data[rt]['mean'].append(None)
        continue
    
    speeds_before.append(results['speed_before'])
    speeds_after.append(results['speed_after'])
    speeds_at_landfall.append(results['speed_at_landfall'])
    Ibtracs_speeds_at_landfall.append(results['STORM_SPEED'])
    storm_dirs_at_landfall.append(results['STORM_DIR'])
    dist2land_at_landfall.append(results['DIST2LAND'])
    
    # Extract and calculate area, max, and mean for each radii type
    for rt in RADII_TYPES:
        ne = results.get(f'{rt}_ne')
        se = results.get(f'{rt}_se')
        sw = results.get(f'{rt}_sw')
        nw = results.get(f'{rt}_nw')
        
        radii_data[rt]['ne'].append(ne)
        radii_data[rt]['se'].append(se)
        radii_data[rt]['sw'].append(sw)
        radii_data[rt]['nw'].append(nw)
        
        area = calculate_wind_field_area(ne, se, sw, nw)
        mean_rad = calculate_mean_radius(ne, se, sw, nw)
        
        radii_data[rt]['area'].append(area)
        radii_data[rt]['mean'].append(mean_rad)


    

# Add speed columns
hurricane_data['travel_speed_before_landfall_m_s'] = speeds_before
hurricane_data['travel_speed_after_landfall_m_s'] = speeds_after
hurricane_data['travel_speed_at_landfall_m_s'] = speeds_at_landfall
hurricane_data['STORM_SPEED'] = Ibtracs_speeds_at_landfall
hurricane_data['DIST2LAND'] = dist2land_at_landfall
hurricane_data['STORM_DIR'] = storm_dirs_at_landfall

# Add radii and area columns
for rt in RADII_TYPES:
    hurricane_data[f'{rt}_ne_at_landfall'] = radii_data[rt]['ne']
    hurricane_data[f'{rt}_se_at_landfall'] = radii_data[rt]['se']
    hurricane_data[f'{rt}_sw_at_landfall'] = radii_data[rt]['sw']
    hurricane_data[f'{rt}_nw_at_landfall'] = radii_data[rt]['nw']
    hurricane_data[f'{rt}_area_at_landfall'] = radii_data[rt]['area']
    hurricane_data[f'{rt}_mean_at_landfall'] = radii_data[rt]['mean']

#handle NaN values after 2004, where NAN means mean radius 0
hurricane_data['r64_mean_at_landfall'] = np.where(hurricane_data['r64_mean_at_landfall'].isna() & (hurricane_data['lf_ISO_TIME'] >= '2004-01-01'), 0, hurricane_data['r64_mean_at_landfall'])
hurricane_data['r50_mean_at_landfall'] = np.where(hurricane_data['r50_mean_at_landfall'].isna() & (hurricane_data['lf_ISO_TIME'] >= '2004-01-01'), 0, hurricane_data['r50_mean_at_landfall'])
# Save updated data
hurricane_data.to_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\Aslak_data_with_tide_and_travelspeed.csv", index=False)

rename_dict = {
    'lf_lat': 'LAT',
    'lf_lon': 'LON',
    'lf_wind': 'USA_WIND',
    'lf_pressure': 'USA_PRES',
}
hurricane_data = hurricane_data.rename(columns=rename_dict)
hurricane_data['Month'] = pd.to_datetime(hurricane_data['lf_ISO_TIME']).dt.month
hurricane_data['Year'] = pd.to_datetime(hurricane_data['lf_ISO_TIME']).dt.year
hurricane_data['Day'] = pd.to_datetime(hurricane_data['lf_ISO_TIME']).dt.day

feature_columns = ['LAT', 'LON', 'USA_WIND', 'USA_PRES', 'STORM_SPEED', 'Month', 'Year', 'Day','DIST2LAND', 'STORM_DIR']
#Lets try to visualize basedamage vs mean wind radius at landfall
#predict r64 mean radius from ML
import joblib
import pickle
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
no_r64_data = hurricane_data[hurricane_data['r64_mean_at_landfall'].isna()]
classifier_model = joblib.load(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\models\xgb_baseline_classifier_model.joblib")

# Prepare features with EXACT column names
X_no_r64 = no_r64_data[feature_columns].copy()


# Rename to match training column names


# Ensure column order matches training

X_no_r64 = X_no_r64[feature_columns]


# CRITICAL: Scale features using the SAME scaler
X_no_r64_scaled = scaler.fit_transform(X_no_r64)

classifier_predictions = classifier_model.predict(X_no_r64_scaled)
# Filter data predicted as having r64 > 0
print("Predicted classes for missing r64 mean data:", classifier_predictions)
no_r64_data['has_r64'] = classifier_predictions
#rename 1=True, 0=False
no_r64_data['has_r64'] = no_r64_data['has_r64'].map({1: True, 0: False})

classified_to_have_r64 = no_r64_data[no_r64_data['has_r64'] == True]
classified_to_not_have_r64 = no_r64_data[no_r64_data['has_r64'] == False]
classified_to_not_have_r64['r64_mean_at_landfall'] = 0
hurricane_data.update(classified_to_not_have_r64)
# Predict damage class based on 

from scipy.special import inv_boxcox

# Load the trained model, scaler, and lambda
model = joblib.load(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\models\xgb_best_model.joblib")

#load lambda
with open(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\models\xgb_fitted_lambda.pkl", "rb") as f:
    fitted_lambda = pickle.load(f)
xgb_best = model




X_regression = classified_to_have_r64[feature_columns].copy()

# Make predictions on scaled data
y_pred_transformed = xgb_best.predict(X_regression)

# CRITICAL: Inverse Box-Cox transform to get back to original scale
y_pred_original = safe_inv_boxcox(y_pred_transformed, fitted_lambda, offset=0.1)


# Add predictions to dataframe
classified_to_have_r64_idx = classified_to_have_r64.index
hurricane_data.update(pd.DataFrame({'r64_mean_at_landfall': y_pred_original}, index=classified_to_have_r64_idx))
hurricane_data = hurricane_data[hurricane_data['ND'] > 0]
norm_damage = hurricane_data['ND'].values
mean_r64 = hurricane_data['r64_mean_at_landfall'].values
plt.scatter(mean_r64, np.log(norm_damage))
plt.xlabel('Mean R64 at Landfall (km)')
plt.ylabel('Normalized Damage')
plt.title('Normalized Damage vs Mean R64 at Landfall')
plt.grid()
plt.show()

hurricane_data.to_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\Aslak_data_with_tide_and_travelspeed_and_ML_r64.csv", index=False)
