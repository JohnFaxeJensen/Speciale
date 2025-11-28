import os
import pyfes

import netCDF4
import numpy as np
import pandas as pd
data = pd.read_excel("./Speciale/Aslak_data.xls", sheet_name='ATD of ICAT', engine='xlrd')
data['lf_ISO_TIME'] = pd.to_datetime(data['lf_ISO_TIME'], format="%Y-%m-%d %H:%M:%S")

print(data.head())

nc_file = r"./Speciale/Pyfes_data/load_tide/2n2_fes2022.nc"
ds = netCDF4.Dataset(nc_file)
lats = ds['lat'][:]
lons = ds['lon'][:]
ds.close()


# Load the grid from one of your FES netCDF files
nc_file = r"./Speciale/Pyfes_data/load_tide/2n2_fes2022.nc"
ds = netCDF4.Dataset(nc_file)
lats = ds['lat'][:]
lons = ds['lon'][:]  # 0 to 360
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
    
    lat_min = closest_grid(lats, lat_min)
    lat_max = closest_grid(lats, lat_max)
    
    # Longitude min/max with wrap-around
    lon_min = (lon - delta_lon) % 360
    lon_max = (lon + delta_lon) % 360

    lon_min = closest_grid(lons, lon_min)
    lon_max = closest_grid(lons, lon_max)

    return (lon_min, lat_min, lon_max, lat_max)


os.chdir(r"./Speciale/Pyfes_data")
def simulate_tide_at_landfall(lat, lon, time):
    bbox = bounding_box(lat, lon, delta_lat=5, delta_lon=5)
    print(time)
    

    cfg = pyfes.load_config("fes2022.yaml", bbox=bbox)

    date = np.array([time.to_datetime64()])


    lons = np.full(date.shape, lon)
    lats = np.full(date.shape, lat)

    tide, lp, flag_tide = pyfes.evaluate_tide(
        cfg['tide'], date, lons, lats, num_threads=1
    )
    load, load_lp, flag_load = pyfes.evaluate_tide(
        cfg['radial'], date, lons, lats, num_threads=1
    )
    print(tide)
    return tide + lp + load
def add_tide_column(df):
    tide_values = []
    for index, row in df.iterrows():
        lat = row['lf_lat']
        lon = row['lf_lon']
        datetime = row['lf_ISO_TIME']
        tide_level = simulate_tide_at_landfall(lat, lon, datetime)
        tide_values.append(tide_level)
    df['Tide_Level'] = tide_values
    return df
add_tide_column(data.head(10))



