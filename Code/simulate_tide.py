import os
import pyfes

import netCDF4
import numpy as np
import pandas as pd

nc_file = r"./Speciale/Pyfes_data/load_tide/2n2_fes2022.nc"
ds = netCDF4.Dataset(nc_file)
lats = ds['lat'][:]
lons = ds['lon'][:]
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




lat_station = 24.553
lon_station = -81.808  # Key West

bbox = bounding_box(lat_station, lon_station, delta_lat=5, delta_lon=5)
print("Bounding box aligned to FES grid:", bbox)


path = r"./Speciale/Pyfes_data"
os.chdir(path)

cfg = pyfes.load_config("fes2022.yaml", bbox=bbox)

date = np.datetime64('1926-09-14T00:00:00')

dates = np.arange(
    date, date + np.timedelta64(10, 'D'), np.timedelta64(1, 'h')
)
lons = np.full(dates.shape, lon_station)
lats = np.full(dates.shape, lat_station)

tide, lp, flag_tide = pyfes.evaluate_tide(
    cfg['tide'], dates, lons, lats, num_threads=1
)
load, load_lp, flag_load = pyfes.evaluate_tide(
    cfg['radial'], dates, lons, lats, num_threads=1
)
print(f'Tide evaluation flags: tide={flag_tide}, load={flag_load}')
# Convert numpy.datetime64 array to pandas datetime
dates_pd = pd.to_datetime(dates)

print(
    f'{"DateTime":>20s} {"Latitude":>10s} {"Longitude":>10s} '
    f'{"Short_tide":>10s} {"LP_tide":>10s} {"Pure_Tide":>10s} '
    f'{"Geo_Tide":>10s} {"Rad_Tide":>10s}'
)
print('=' * 100)
for ix, dt in enumerate(dates_pd):
    print(
        f'{dt:%Y-%m-%d %H:%M} {lats[ix]:>10.3f} {lons[ix]:>10.3f} '
        f'{tide[ix]:>10.3f} {lp[ix]:>10.3f} {tide[ix] + lp[ix]:>10.3f} '
        f'{tide[ix] + lp[ix] + load[ix]:>10.3f} {load[ix]:>10.3f}'
    )
#compare with measurements
key_west = pd.read_csv(r"Key_west_measurements.csv", header=None, names=['Year', 'Month', 'Day', 'Hour', 'WaterLevel'])
key_west['DateTime'] = pd.to_datetime(key_west[['Year', 'Month', 'Day', 'Hour']])
#convert mm to meters
key_west['WaterLevel'] = key_west['WaterLevel'] / 1000
key_west.set_index('DateTime', inplace=True)

#filter dates
key_west = key_west[(key_west.index >= dates[0]) & (key_west.index < dates[-1])]

#calculate deviations from mean
key_west['Deviation'] = key_west['WaterLevel'] - key_west['WaterLevel'].mean()

geocentric = (tide + lp + load) / 100
#pure_tide = (tide + lp) / 100
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(key_west.index, key_west['Deviation'], label='Measured Deviation', color='blue')
plt.plot(dates_pd, geocentric, label='Simulated Deviation', color='orange')
#plt.plot(dates_pd, pure_tide, label='Pure Tide Deviation', color='green', linestyle='--')
plt.xlabel('DateTime')
plt.ylabel('Water Level Deviation')
plt.title(f'Measured Water Level Deviation at Key West. Interval: {dates_pd[0]} to {dates_pd[-1]}')
plt.legend()
plt.grid(True)
plt.show()


