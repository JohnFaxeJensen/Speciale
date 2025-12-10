import rioxarray as rio

import xarray as xr

import numpy as np

import matplotlib.pyplot as plt

import glob

from tqdm import trange, tqdm

from pyproj import Transformer

import os
import sys
import rasterio
import logging
import pandas as pd

# console_handler = logging.StreamHandler()
# formatter = logging.Formatter("%(levelname)s:%(message)s")
# console_handler.setFormatter(formatter)
# logger = logging.getLogger("rasterio")
# logger.addHandler(console_handler)
# logger.setLevel(logging.DEBUG)
# logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)
# os.environ["PROJ_DEBUG"] = "2"
# os.makedirs('./Speciale/USpop_m5_tif',exist_ok=True)
# with rasterio.Env(CPL_DEBUG=True):
# files = sorted(glob.glob('./Speciale/USA_HistoricalPopulationDataset/pop_m5_*/w001001.adf'))

# for file in tqdm(files):

#     # Load ADF grid
#     ds = rio.open_rasterio(file, band_as_variable=True).band_1

#     # Clip to your bounding box
#     xdsc = ds.rio.clip_box(
#         minx=-0.6e6,
#         miny=-3e6,
#         maxx=2e6,
#         maxy=0.5e6,
#     )

#     # Parse folder name → year
#     year = os.path.basename(os.path.dirname(file)).split("_")[-1]
#     if int(year)<1890:
#         continue
#     outfile = f'./Speciale/USpop_m5_tif/pop_m5_{year}.tif'
#     ds = rasterio.open(file)
#     print(ds.bounds)
#     print(ds.crs)
#     print(ds.width, ds.height)
#     # Save as COG
#     xdsc.rio.to_raster(
#         outfile,
#         driver='COG',
#         compress='LZW',
#         predictor=2,
#         windowed=True,
#     )

#     print("Created:", outfile)

# Cache for opened raster datasets to avoid repeated file I/O
_raster_cache = {}

def get_raster_dataset(year):
    """Get cached raster dataset for a given year."""
    if year not in _raster_cache:
        tif_file = f'./Speciale/USpop_m5_tif/pop_m5_{year}.tif'
        _raster_cache[year] = rasterio.open(tif_file)
    return _raster_cache[year]

# Pre-open the reference dataset for coordinate system info
ds = get_raster_dataset(1980)

def get_population_density(lon, lat, year):
    ds = get_raster_dataset(year)
    transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    row, col = ds.index(x, y)
    if 0 <= row < ds.height and 0 <= col < ds.width:
        value = ds.read(1)[row, col]
        return value
    else:
        return np.nan

def exponential_interpolation(value1, value2, year1, year2, target_year):
    return value1 * (value2 / value1) ** ((target_year - year1) / (year2 - year1))
def linear_interpolation(value1, value2, year1, year2, target_year):
    return value1 + ((target_year - year1) / (year2 - year1))*(value2 - value1)


def get_population_density_at_year(lon, lat, year):
    decadal_years = [1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
    if(int(year) in decadal_years):
        pop_dens = get_population_density(lon, lat, year)
        if(np.isnan(pop_dens) or pop_dens<0):
            pop_dens = 0
        return pop_dens

    if int(year) < 2010:
        lower_year = max([y for y in decadal_years if y < year])
        upper_year = min([y for y in decadal_years if y > year])
        lower_value = get_population_density(lon, lat, lower_year)
        if(np.isnan(lower_value) or lower_value<0):
            lower_value = 0
        upper_value = get_population_density(lon, lat, upper_year)
        if(np.isnan(upper_value) or upper_value<0):
            upper_value = 0
        if(lower_value < 1 or upper_value < 1):
            lin_value = linear_interpolation(lower_value, upper_value, lower_year, upper_year, year)
            return lin_value
        return exponential_interpolation(lower_value, upper_value, lower_year, upper_year, year)
    value_2000 = get_population_density(lon, lat, 2000)
    value_2010 = get_population_density(lon, lat, 2010)
    if np.isnan(value_2000) or value_2000 < 0:
        value_2000 = 0
    if np.isnan(value_2010) or value_2010 < 0:
        value_2010 = 0

    # If no data, population likely tiny -> return 0
    if value_2000 == 0 or value_2010 == 0:
        return value_2010


    # exp extrapolation rate from 2000 to 2010
    if(value_2000 > 1 or value_2010 > 1): #Don't do exp if very low values
        r = (value_2010 / value_2000) ** (1/10) - 1
        if(r < -0.03): #cap decline rate to -3% per year
            r = -0.03
        if(r>0.03): #cap growth rate to 3% per year to avoid extreme extrapolations
            r = 0.03
        return value_2010 * (1 + r) ** (year - 2010)
    else: #linear extrapolation
        return linear_interpolation(value_2000, value_2010, 2000, 2010, year)




def compute_population_in_radius(ds, landfall_xy, radius, year):
    """
    ds = rasterio dataset open for a specific year
    landfall_xy = (x0, y0) in raster CRS (meters)
    radius = meters
    """
    decadal_years = [1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
    
    x0, y0 = landfall_xy
    res = ds.res[0]  # pixel size (should be 1000 m)
    # Determine which rows/cols might be inside the circle
    # Bounding box of circle
    minx = x0 - radius
    maxx = x0 + radius
    miny = y0 - radius
    maxy = y0 + radius

    # Convert bounding box to pixel indices
    row_min, col_min = ds.index(minx, maxy)
    row_max, col_max = ds.index(maxx, miny)
    # Clamp to dataset bounds
    row_min = max(row_min, 0)
    col_min = max(col_min, 0)
    row_max = min(row_max, ds.height - 1)
    col_max = min(col_max, ds.width - 1)
    # Read relevant window
    window = ((row_min, row_max), (col_min, col_max))
    data = ds.read(1, window=window)
    data = np.where(data < 0, 0, data)  # Replace negative values with 0

    # Build coordinate grid for pixel centers
    rows = np.arange(row_min, row_max)
    cols = np.arange(col_min, col_max)

    rows_2d, cols_2d = np.meshgrid(rows, cols, indexing='ij')
    xs, ys = rasterio.transform.xy(ds.transform, rows_2d, cols_2d)
    xs = np.array(xs).reshape(rows_2d.shape)
    ys = np.array(ys).reshape(rows_2d.shape)
    # Compute distances from landfall
    dist = np.sqrt((xs - x0)**2 + (ys - y0)**2)

    # Mask inside circle
    mask = dist <= radius

    # Transform grid points back to lon/lat
    transformer = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xs, ys)

    # Efficiently compute population for target year using array operations
    # Determine which years we need to load
    if int(year) in decadal_years:
        # Directly use the data for that year
        ds_year = get_raster_dataset(year)
        pop_data = ds_year.read(1, window=window)
        pop_grid = np.where(pop_data < 0, 0, pop_data)
    elif int(year) < 2010:
        # Interpolate between two decadal years
        lower_year = max([y for y in decadal_years if y < year])
        upper_year = min([y for y in decadal_years if y > year])
        
        ds_lower = get_raster_dataset(lower_year)
        ds_upper = get_raster_dataset(upper_year)
        
        lower_data = ds_lower.read(1, window=window)
        upper_data = ds_upper.read(1, window=window)
        
        lower_data = np.where(lower_data < 0, 0, lower_data)
        upper_data = np.where(upper_data < 0, 0, upper_data)
        
        # Vectorized interpolation
        low_mask = (lower_data < 1) | (upper_data < 1)
        pop_grid = np.where(
            low_mask,
            lower_data + ((year - lower_year) / (upper_year - lower_year)) * (upper_data - lower_data),
            lower_data * (upper_data / np.maximum(lower_data, 1e-10)) ** ((year - lower_year) / (upper_year - lower_year))
        )
    else:
        # Extrapolate from 2000 and 2010
        ds_2000 = get_raster_dataset(2000)
        ds_2010 = get_raster_dataset(2010)
        
        data_2000 = ds_2000.read(1, window=window)
        data_2010 = ds_2010.read(1, window=window)
        
        data_2000 = np.where(data_2000 < 0, 0, data_2000)
        data_2010 = np.where(data_2010 < 0, 0, data_2010)
        
        # Handle zero cases
        zero_mask = (data_2000 == 0) | (data_2010 == 0)
        
        # For non-zero, calculate growth rate
        low_val_mask = (data_2000 <= 1) | (data_2010 <= 1)
        
        # Calculate exponential growth rate, capped at ±3%
        r = (data_2010 / np.maximum(data_2000, 1e-10)) ** (1/10) - 1
        r = np.clip(r, -0.03, 0.03)
        
        # Exponential extrapolation
        exp_result = data_2010 * (1 + r) ** (year - 2010)
        
        # Linear extrapolation
        lin_result = data_2000 + ((year - 2000) / 10) * (data_2010 - data_2000)
        
        # Choose method based on data values
        pop_grid = np.where(
            zero_mask,
            data_2010,
            np.where(low_val_mask, lin_result, exp_result)
        )

    # Sum only the population inside the circle
    total_population = np.sum(pop_grid[mask])

    return total_population

hurricane_data = pd.read_csv('./Speciale/Hurricane_data/Aslak_data_with_tide.csv')
hurricane_data_clean = hurricane_data[hurricane_data['basedamage'] > 0]
basedamage_list = hurricane_data_clean['basedamage'].values
years_list = hurricane_data_clean['lf_ISO_TIME'].values
years_list = [int(yt.split('-')[0]) for yt in years_list]
lats_list = hurricane_data_clean['lf_lat'].values
lons_list = hurricane_data_clean['lf_lon'].values
WPC_list = hurricane_data_clean['WPC'].values

test_year = years_list[-3]
test_lon = lons_list[-3]
test_lat = lats_list[-3]
test_basedamage = basedamage_list[-3]
test_WPC = WPC_list[-3]
print(f"Test case - Year: {test_year}, Lon: {test_lon}, Lat: {test_lat}, Basedamage: {test_basedamage}, WPC: {test_WPC}")

#loop until basedamage = WPC*population_in_radius
def estimate_area(landfall_lat, landfall_lon, year, basedamage, WPC):
    transformer = Transformer.from_crs("EPSG:4326", "ESRI:102003", always_xy=True)
    landfall_x, landfall_y = transformer.transform(landfall_lon, landfall_lat)

    radius = 500000  # Start with 500 km
    max_iterations = 20
    tolerance = 0.05  # 5% tolerance
    i = 0
    for _ in range(max_iterations):
        print(f"Iteration {i+1}: Testing radius {radius} meters")
        i += 1
        population_in_radius = compute_population_in_radius(ds, (landfall_x, landfall_y), radius, year)
        estimated_damage = WPC * population_in_radius
        print(f"  Estimated damage: {estimated_damage}, Target basedamage: {basedamage}, Population in radius: {population_in_radius}")
        if abs(estimated_damage - basedamage) / basedamage <= tolerance:
            break  # Within tolerance

        if estimated_damage < basedamage:
            radius *= 1.2  # Increase radius
        else:
            radius *= 0.8  # Decrease radius

    return radius, population_in_radius

area = estimate_area(test_lat, test_lon, test_year, test_basedamage, test_WPC)
print(f"Estimated area radius: {area[0]} meters, Population in area: {area[1]}")