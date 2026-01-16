
import os
import pandas as pd
import numpy as np


feature_columns = ['LAT', 'LON', 'USA_WIND', 'USA_PRES', 'STORM_SPEED_ms', 'Month' , 'Year', 'Day', 'DIST2LAND_m', 'STORM_DIR']

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


def get_training_data():
    if os.path.exists(r"C:/Users/123ti/Documents/Speciale_git/Speciale/Code/Data_preprocessing/ML/raw_training_data.csv"):
        print("Training data already exists. Loading from file.")
        return pd.read_csv(r"C:/Users/123ti/Documents/Speciale_git/Speciale/Code/Data_preprocessing/ML/raw_training_data.csv")
    radius_types = {
        'R50': ['USA_R50_NE', 'USA_R50_SE', 'USA_R50_SW', 'USA_R50_NW'],
        'R64': ['USA_R64_NE', 'USA_R64_SE', 'USA_R64_SW', 'USA_R64_NW'],
        'R34': ['USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW']
    }

    cols = ["ISO_TIME", "LAT", "LON","USA_ATCF_ID", "USA_LAT", "USA_LON", 
            "USA_WIND", "USA_PRES", "STORM_SPEED",
            'DIST2LAND', 'STORM_DIR']
    cols += [col for sublist in radius_types.values() for col in sublist]
    print("Loading hurricane data...")
    hurricane_data = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\ibtracs.ALL.list.v04r01.csv", usecols=cols,low_memory=False)
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
    hurricane_data['STORM_DIR'] = pd.to_numeric(hurricane_data['STORM_DIR'], errors='coerce')
    hurricane_data['STORM_SPEED'] = pd.to_numeric(hurricane_data['STORM_SPEED'], errors='coerce')
    hurricane_data['DIST2LAND'] = pd.to_numeric(hurricane_data['DIST2LAND'], errors='coerce')
    #change units from knots to m/s
    hurricane_data['STORM_SPEED_ms'] = hurricane_data['STORM_SPEED'] * 0.514444
    #change units from km to m
    hurricane_data['DIST2LAND_m'] = hurricane_data['DIST2LAND'] * 1000
    hurricane_data.drop(['STORM_SPEED', 'DIST2LAND'], axis=1, inplace=True)


    
    for radius in radius_types.values():
        hurricane_data[radius[0]] = pd.to_numeric(hurricane_data[radius[0]], errors='coerce')
        hurricane_data[radius[1]] = pd.to_numeric(hurricane_data[radius[1]], errors='coerce')
        hurricane_data[radius[2]] = pd.to_numeric(hurricane_data[radius[2]], errors='coerce')
        hurricane_data[radius[3]] = pd.to_numeric(hurricane_data[radius[3]], errors='coerce')
    
    #remove pre2004 data due to different measurement standards
    hurricane_data = hurricane_data[hurricane_data['Year'] >= 2004]
    for radius_name, radius_cols in radius_types.items():
        hurricane_data[f'{radius_name}_AREA'] = hurricane_data.apply(
            lambda row: calculate_wind_field_area(row[radius_cols[0]], row[radius_cols[1]], row[radius_cols[2]], row[radius_cols[3]]),
            axis=1
        )
        hurricane_data[f'{radius_name}_AREA'] = hurricane_data[f'{radius_name}_AREA'].fillna(0)
        hurricane_data[f'{radius_name}_MEAN_RADIUS'] = hurricane_data.apply(
            lambda row: calculate_mean_wind_radius(row[radius_cols[0]], row[radius_cols[1]], row[radius_cols[2]], row[radius_cols[3]]),
            axis=1
        )
        hurricane_data[f'{radius_name}_MEAN_RADIUS'] = hurricane_data[f'{radius_name}_MEAN_RADIUS'].fillna(0)
    #drop rows with missing feature data
    hurricane_data = hurricane_data.dropna(subset=feature_columns)
    hurricane_data.to_csv(r"C:/Users/123ti/Documents/Speciale_git/Speciale/Code/Data_preprocessing/ML/raw_training_data.csv", index=False)
    
    return hurricane_data

data = get_training_data()

