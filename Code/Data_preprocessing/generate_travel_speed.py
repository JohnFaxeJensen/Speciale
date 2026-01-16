import pandas as pd
import numpy as np
from geopy import distance
import os

def find_closest_time_indices(times, target_time):
    
    times_before = times[times < target_time]
    times_after = times[times > target_time]    
    if len(times_before) == 0 or len(times_after) == 0:
        print(f"Not enough data before/after landfall for ")
        return None
    
    before_time = times_before.max()
    after_time = times_after.min()
    
    before_index = np.where(times == before_time)[0][0]
    after_index = np.where(times == after_time)[0][0]



    time_delta_before = target_time - before_time
    time_delta_after = after_time - target_time
    
    closest_index = before_index if time_delta_before <= time_delta_after else after_index

    return before_index, after_index, closest_index


def calculate_travelspeed(hurricane_id, landfall_time, landfall_lat, landfall_lon, Filtered_IBTrACS_data):

    times = Filtered_IBTrACS_data['ISO_TIME']
    times = pd.to_datetime(times)
    lats = Filtered_IBTrACS_data['LAT'].values
    lons = Filtered_IBTrACS_data['LON'].values
    storm_speeds = Filtered_IBTrACS_data['STORM_SPEED'].values
    dist_2_land = Filtered_IBTrACS_data['DIST2LAND'].values
    storm_dirs = Filtered_IBTrACS_data['STORM_DIR'].values

    if hurricane_id == "AL051965":
        landfall_time = pd.to_datetime("1965-09-29 20:00:00")
    else:
        landfall_time = pd.to_datetime(landfall_time)
    

    before_index, after_index, closest_index = find_closest_time_indices(times, landfall_time)
    
    before_time = times.iloc[before_index]
    after_time = times.iloc[after_index]

    before_lat = lats[before_index]
    before_lon = lons[before_index]
    after_lat = lats[after_index]
    after_lon = lons[after_index]

    
    ibtracs_speed_at_landfall = storm_speeds[closest_index]
    ibtracs_speed_at_landfall = float(ibtracs_speed_at_landfall) * 0.514444 #convert knots to m/s
    dist2land_at_landfall = dist_2_land[closest_index]
    dist2land_at_landfall = float(dist2land_at_landfall) * 1000 #convert from km to m
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
        'lf_speed_before': speed_before,
        'lf_speed_after': speed_after,
        'lf_speed': speed_at_landfall,
        'STORM_SPEED_ms': ibtracs_speed_at_landfall, #convert knots to m/s
        'DIST2LAND_m': dist2land_at_landfall, #convert from km to m
        'STORM_DIR': storm_dir_at_landfall
    }


    
    return results


def generate_travel_speed(hurricane_data, Ibtracs_data):

    relevant_data = hurricane_data[['ATCF_ID', 'lf_ISO_TIME', 'lf_lat', 'lf_lon']].copy()
    
    # Initialize lists for all outputs
    speeds_before = []
    speeds_after = []
    speeds_at_landfall = []
    ibtracs_speeds_at_landfall = []
    storm_dirs_at_landfall = []
    dist2land_at_landfall = []
    
    # Process each hurricane
    for index, row in relevant_data.iterrows():
        hurricane_id = row['ATCF_ID']
        landfall_time = row['lf_ISO_TIME']
        landfall_lat = row['lf_lat']
        landfall_lon = row['lf_lon']
        
        
        filtered_IBTrACS_data = Ibtracs_data[Ibtracs_data['USA_ATCF_ID'] == hurricane_id]
        results = calculate_travelspeed(hurricane_id, landfall_time, landfall_lat, landfall_lon, 
                                                   filtered_IBTrACS_data)
        
        if results is None:
            speeds_before.append(None)
            speeds_after.append(None)
            speeds_at_landfall.append(None)
            ibtracs_speeds_at_landfall.append(None)
            storm_dirs_at_landfall.append(None)
            dist2land_at_landfall.append(None)
            continue
        
        speeds_before.append(results['lf_speed_before'])
        speeds_after.append(results['lf_speed_after'])
        speeds_at_landfall.append(results['lf_speed'])
        ibtracs_speeds_at_landfall.append(results['STORM_SPEED_ms'])
        storm_dirs_at_landfall.append(results['STORM_DIR'])
        dist2land_at_landfall.append(results['DIST2LAND_m'])
    
    # Add speed columns to dataframe
    hurricane_data['lf_speed_before_m'] = speeds_before
    hurricane_data['lf_speed_after_m'] = speeds_after
    hurricane_data['lf_speed_m'] = speeds_at_landfall
    hurricane_data['STORM_SPEED_ms'] = ibtracs_speeds_at_landfall
    hurricane_data['DIST2LAND_m'] = dist2land_at_landfall
    hurricane_data['STORM_DIR'] = storm_dirs_at_landfall
    
    return hurricane_data

def generate_travel_speed_data(df):
    path = r"C:/Users/123ti/Documents/Speciale_git/Speciale/Code/Data_preprocessing/generated_data/travelspeed_data.csv"
    if os.path.exists(path):
        print("Travel speed data already exists. Loading from file.")
        return pd.read_csv(path)
    valid_IDs = df['ATCF_ID'].unique()
    # Paths
    ibtracks_data = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\ibtracs.ALL.list.v04r01.csv", low_memory=False)
    ibtracks_data_filtered = ibtracks_data[ibtracks_data['USA_ATCF_ID'].isin(valid_IDs)]
    
    df_with_travelspeed = generate_travel_speed(df, ibtracks_data_filtered)
    df_with_travelspeed = df_with_travelspeed.drop_duplicates()
    df_with_travelspeed.to_csv(path, index=False)
    return df_with_travelspeed



