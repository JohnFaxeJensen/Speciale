import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4
    

# Load the grid from one of your FES netCDF files
nc_file = r"C:\Users\123ti\Documents\Speciale_git\Speciale\Pyfes_data\load_tide\2n2_fes2022.nc"
ds = netCDF4.Dataset(nc_file)
lats_global = ds['lat'][:]
lons_global = ds['lon'][:]  # 0 to 360
ds.close()
def closest_grid(array, value):
    """Return the closest value in array to the given value."""
    return array[np.abs(array - value).argmin()]

def bounding_box(lat, lon, delta_lat=0.5, delta_lon=0.5):
    """
    Returns a bounding box aligned to the FES grid.
    
    lat: latitude of point (-90 to 90)
    lon: longitude of point (-180 to 180 or 0 to 360)
    delta_lat, delta_lon: half-width of box
    """
    # Convert longitude to 0-360
    lon = lon % 360

    # Latitude min/max with bounds check
    lat_min = max(-90, lat - delta_lat)
    lat_max = min(90, lat + delta_lat)
    
    lat_min = closest_grid(lats_global, lat_min)
    lat_max = closest_grid(lats_global, lat_max)
    
    # Longitude min/max with wrap-around
    lon_min = (lon - delta_lon) % 360
    lon_max = (lon + delta_lon) % 360

    lon_min = closest_grid(lons_global, lon_min)
    lon_max = closest_grid(lons_global, lon_max)

    return (lon_min, lat_min, lon_max, lat_max)


def generate_tide_data(lat_lon_time_df):
    path = r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\generated_data\tide_data_lf.csv"
    #check if data exists
    if os.path.exists(path):
        print("Tide data already exists. Loading from file.")
        return pd.read_csv(path)
    import pyfes


    data = pd.read_excel(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\Aslak_data.xls", sheet_name='ATD of ICAT', engine='xlrd')
    data['lf_ISO_TIME'] = pd.to_datetime(data['lf_ISO_TIME'], format="%Y-%m-%d %H:%M:%S")


    os.chdir(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Pyfes_data")
    def simulate_tide_at_landfall(lat, lon, time):
        bbox = bounding_box(lat, lon, delta_lat=1, delta_lon=1)
        lat = closest_grid(lats_global, lat)
        lon = closest_grid(lons_global, lon % 360)
        

        cfg = pyfes.load_config(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Pyfes_data\fes2022.yaml", bbox=bbox)

        date = np.array([time.to_datetime64()])


        lons = np.full(date.shape, lon % 360)
        lats = np.full(date.shape, lat)

        tide, lp, flag_tide = pyfes.evaluate_tide(
            cfg['tide'], date, lons, lats, num_threads=1
        )
        load, load_lp, flag_load = pyfes.evaluate_tide(
            cfg['radial'], date, lons, lats, num_threads=1
        )
        return tide + lp + load + load_lp
    
    def add_tide_column(df):
        #df should have 'lf_lat', 'lf_lon', 'lf_ISO_TIME' columns
        copy_df = df.copy()
        tide_values = []
        for index, row in copy_df.iterrows():
            lat = row['lf_lat']
            lon = row['lf_lon']
            datetime = row['lf_ISO_TIME']
            tide_level = simulate_tide_at_landfall(lat, lon, datetime)
            tide_values.append(tide_level[0]*0.01)  #convert from cm to m
        copy_df['Tide_Level_lf'] = tide_values
        return copy_df

    new_data = add_tide_column(lat_lon_time_df)
    new_data.to_csv(path, index=False)
    return new_data

def simulate_range_at_peak(lat, lon, time):
    import pyfes
    bbox = bounding_box(lat, lon, delta_lat=1, delta_lon=1)
    print(lat, lon, time)
    print(bbox)
    lat = closest_grid(lats_global, lat)
    lon = closest_grid(lons_global, lon % 360)
    print(f"Closest grid point: lat={lat}, lon={lon}")
    
    cfg = pyfes.load_config(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Pyfes_data\fes2022.yaml", bbox=bbox)

    #take the range within 48 hours before and after landfall
    date = np.array([time.to_datetime64()])
    dates = np.arange(
        date[0] - np.timedelta64(48, 'h'), date[0] + np.timedelta64(48, 'h'), np.timedelta64(20, 'm')
    )


    lons = np.full(dates.shape, lon % 360)
    lats = np.full(dates.shape, lat)

    tide, lp, flag_tide = pyfes.evaluate_tide(
        cfg['tide'], dates, lons, lats, num_threads=1
    )
    load, load_lp, flag_load = pyfes.evaluate_tide(
        cfg['radial'], dates, lons, lats, num_threads=1
    )
    total_tide = tide + lp + load + load_lp
    tidal_range = np.max(total_tide) - np.min(total_tide)
    return tidal_range*0.01  #convert from cm to m

def generate_tidal_ranges():
    path = r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\generated_data\tidal_ranges_peak.csv"
    if os.path.exists(path):
        print("Tidal range data already exists. Loading from file.")
        return pd.read_csv(path)
    #this function is used to calculate tidal range for tide observations close to landfall
    manual_checked_data = pd.read_excel(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\generated_data\surge_data_manual_check.xlsx")
    print(manual_checked_data.shape)
    #remove data points with no Lat_db and Lon_db
    copy = manual_checked_data.copy()
    relevant_columns = ['Unique_ID','Lat_db', 'Lon_db', 'lf_ISO_TIME']
    copy = copy[relevant_columns]
    os.chdir(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Pyfes_data")

    tide_ranges = []
    for index, row in copy.iterrows():
        lat = row['Lat_db']
        lon = row['Lon_db']
        print(f"Processing index {index} with lat={lat}, lon={lon}")
        if (
            pd.isna(lat)
            or pd.isna(lon)
            or (isinstance(lat, str) and lat.strip().lower() in {"", "nan"})
            or (isinstance(lon, str) and lon.strip().lower() in {"", "nan"})
        ):
            print(f"Index {index} has missing/invalid lat/lon. Skipping.")
            tide_range = np.nan
        else:
            datetime = row['lf_ISO_TIME']
            tide_range = simulate_range_at_peak(lat, lon, datetime)
        tide_ranges.append(tide_range)
    copy['Tidal_Range_peak'] = tide_ranges
    copy.to_csv(path, index=False)
    return copy

#generate_tidal_ranges()