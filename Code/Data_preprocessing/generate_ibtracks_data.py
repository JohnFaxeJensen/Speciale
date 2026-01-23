import pandas as pd
import numpy as np
from geopy import distance
import os

def find_closest_time_indices(times, target_time):
    
    times_before = times[times < target_time]
    times_after = times[times > target_time]    
    if len(times_before) == 0 or len(times_after) == 0:
        print(f"Not enough data before/after landfall for {target_time}")
        return None
    
    before_time = times_before.max()
    after_time = times_after.min()
    
    before_index = np.where(times == before_time)[0][0]
    after_index = np.where(times == after_time)[0][0]



    time_delta_before = target_time - before_time
    time_delta_after = after_time - target_time
    
    closest_index = before_index if time_delta_before <= time_delta_after else after_index

    return before_index, after_index, closest_index


def extract_ibtracs_data_at_closest_index(hurricane_id, landfall_time, Filtered_IBTrACS_data, radius_types=['R34', 'R50', 'R64']):
    """
    Extract IBTrACS data at the closest time observation to landfall.
    
    Extracts: STORM_SPEED_ms, DIST2LAND_m, STORM_DIR, and wind radii (R34/R50/R64)
    
    Parameters:
    -----------
    hurricane_id : str
        Hurricane identifier
    landfall_time : datetime
        Landfall time
    Filtered_IBTrACS_data : pd.DataFrame
        IBTrACS data for this hurricane
    radius_types : list of str
        Wind radius types to extract (R34, R50, R64, etc.)
        
    Returns:
    --------
    dict with IBTrACS data at closest index, or None if extraction failed
    """
    times = Filtered_IBTrACS_data['ISO_TIME']
    times = pd.to_datetime(times)

    if hurricane_id == "AL051965":
        landfall_time = pd.to_datetime("1965-09-29 20:00:00")
    else:
        landfall_time = pd.to_datetime(landfall_time)

    indices = find_closest_time_indices(times, landfall_time)
    if indices is None:
        return None
    
    before_index, after_index, closest_index = indices
    
    results = {}
    
    # Extract basic IBTrACS data
    if 'STORM_SPEED' in Filtered_IBTrACS_data.columns:
        storm_speed = float(Filtered_IBTrACS_data['STORM_SPEED'].values[closest_index]) * 0.514444  # knots to m/s
        results['STORM_SPEED_ms'] = storm_speed
    
    if 'DIST2LAND' in Filtered_IBTrACS_data.columns:
        dist2land = float(Filtered_IBTrACS_data['DIST2LAND'].values[closest_index]) * 1000  # km to m
        results['DIST2LAND_m'] = dist2land
    
    if 'STORM_DIR' in Filtered_IBTrACS_data.columns:
        storm_dir = Filtered_IBTrACS_data['STORM_DIR'].values[closest_index]
        results['STORM_DIR'] = storm_dir
    
    # Extract wind radii for all requested types
    for radii_type in radius_types:
        radii_type_upper = radii_type.upper()
        try:
            ne = Filtered_IBTrACS_data[f'USA_{radii_type_upper}_NE'].values[closest_index]
            se = Filtered_IBTrACS_data[f'USA_{radii_type_upper}_SE'].values[closest_index]
            sw = Filtered_IBTrACS_data[f'USA_{radii_type_upper}_SW'].values[closest_index]
            nw = Filtered_IBTrACS_data[f'USA_{radii_type_upper}_NW'].values[closest_index]
            
            results[f'{radii_type}_NE'] = ne
            results[f'{radii_type}_SE'] = se
            results[f'{radii_type}_SW'] = sw
            results[f'{radii_type}_NW'] = nw
        except KeyError:
            pass  # Column doesn't exist, skip this radii type
    
    return results


def calculate_travelspeed(hurricane_id, landfall_time, landfall_lat, landfall_lon, Filtered_IBTrACS_data):
    """
    Calculate travel speed at landfall based on before/after positions.
    
    Returns:
    --------
    dict with travel speed calculations, or None if calculation failed
    """
    times = Filtered_IBTrACS_data['ISO_TIME']
    times = pd.to_datetime(times)
    lats = Filtered_IBTrACS_data['LAT'].values
    lons = Filtered_IBTrACS_data['LON'].values

    if hurricane_id == "AL051965":
        landfall_time = pd.to_datetime("1965-09-29 20:00:00")
    else:
        landfall_time = pd.to_datetime(landfall_time)
    
    indices = find_closest_time_indices(times, landfall_time)
    if indices is None:
        return None
    
    before_index, after_index, closest_index = indices
    
    before_time = times.iloc[before_index]
    after_time = times.iloc[after_index]

    before_lat = lats[before_index]
    before_lon = lons[before_index]
    after_lat = lats[after_index]
    after_lon = lons[after_index]

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
    
    results = {
        'lf_speed_before_ms': speed_before,
        'lf_speed_after_ms': speed_after,
        'lf_speed_ms': speed_at_landfall,
    }
    
    return results


def generate_travel_speed(hurricane_data, Ibtracs_data):
    """
    Add travel speed columns to hurricane data.
    
    Adds: lf_speed_before_m, lf_speed_after_m, lf_speed_m
    """
    relevant_data = hurricane_data[['ATCF_ID', 'lf_ISO_TIME', 'lf_lat', 'lf_lon']].copy()
    
    # Initialize lists for all outputs
    speeds_before = []
    speeds_after = []
    speeds_at_landfall = []
    
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
            continue
        
        speeds_before.append(results['lf_speed_before_ms'])
        speeds_after.append(results['lf_speed_after_ms'])
        speeds_at_landfall.append(results['lf_speed_ms'])
    
    # Add speed columns to dataframe
    hurricane_data['lf_speed_before_ms'] = speeds_before
    hurricane_data['lf_speed_after_ms'] = speeds_after
    hurricane_data['lf_speed_ms'] = speeds_at_landfall
    
    return hurricane_data


def generate_ibtracs(hurricane_data, Ibtracs_data, radius_types=['R34', 'R50', 'R64']):
    """
    Add IBTrACS data columns to hurricane data at closest time observation.
    
    Adds: STORM_SPEED_ms, DIST2LAND_m, STORM_DIR, and wind radii columns
    """
    relevant_data = hurricane_data[['ATCF_ID', 'lf_ISO_TIME']].copy()
    
    # Initialize lists for all outputs
    ibtracs_speeds = []
    dist2land_values = []
    storm_dirs = []
    
    # Initialize radii lists
    radii_data = {rt: {'NE': [], 'SE': [], 'SW': [], 'NW': []} for rt in radius_types}
    
    # Process each hurricane
    for index, row in relevant_data.iterrows():
        hurricane_id = row['ATCF_ID']
        landfall_time = row['lf_ISO_TIME']
        
        filtered_IBTrACS_data = Ibtracs_data[Ibtracs_data['USA_ATCF_ID'] == hurricane_id]
        results = extract_ibtracs_data_at_closest_index(hurricane_id, landfall_time, 
                                                        filtered_IBTrACS_data, radius_types)
        
        if results is None:
            ibtracs_speeds.append(None)
            dist2land_values.append(None)
            storm_dirs.append(None)
            for rt in radius_types:
                radii_data[rt]['NE'].append(None)
                radii_data[rt]['SE'].append(None)
                radii_data[rt]['SW'].append(None)
                radii_data[rt]['NW'].append(None)
            continue
        
        ibtracs_speeds.append(results.get('STORM_SPEED_ms'))
        dist2land_values.append(results.get('DIST2LAND_m'))
        storm_dirs.append(results.get('STORM_DIR'))
        
        for rt in radius_types:
            radii_data[rt]['NE'].append(results.get(f'{rt}_NE'))
            radii_data[rt]['SE'].append(results.get(f'{rt}_SE'))
            radii_data[rt]['SW'].append(results.get(f'{rt}_SW'))
            radii_data[rt]['NW'].append(results.get(f'{rt}_NW'))
    
    # Add columns to dataframe
    hurricane_data['STORM_SPEED_ms'] = ibtracs_speeds
    hurricane_data['DIST2LAND_m'] = dist2land_values
    hurricane_data['STORM_DIR'] = storm_dirs
    
    # Add radii columns
    for rt in radius_types:
        hurricane_data[f'{rt}_NE'] = radii_data[rt]['NE']
        hurricane_data[f'{rt}_SE'] = radii_data[rt]['SE']
        hurricane_data[f'{rt}_SW'] = radii_data[rt]['SW']
        hurricane_data[f'{rt}_NW'] = radii_data[rt]['NW']

    
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
def generate_ibtracs_data(df):
    path = r"C:/Users/123ti/Documents/Speciale_git/Speciale/Code/Data_preprocessing/generated_data/ibtracs_data.csv"
    if os.path.exists(path):
        print("IBTrACS data already exists. Loading from file.")
        return pd.read_csv(path)
    valid_IDs = df['ATCF_ID'].unique()
    # Paths
    ibtracks_data = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\ibtracs.ALL.list.v04r01.csv", low_memory=False)
    ibtracks_data_filtered = ibtracks_data[ibtracks_data['USA_ATCF_ID'].isin(valid_IDs)]
    
    df_with_ibtracs = generate_ibtracs(df, ibtracks_data_filtered)
    df_with_ibtracs = df_with_ibtracs.drop_duplicates()
    df_with_ibtracs.to_csv(path, index=False)
    return df_with_ibtracs



