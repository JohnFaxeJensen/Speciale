import os

import mat73
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt



# Function to compute trend and uncertainty using Monte Carlo
def compute_trend_with_error(HRrel_ts, SE_ts):
    # Define linear model for curve fitting
    def linear_model(t, a, b):
        return a * t + b
    popt, pcov = scipy.optimize.curve_fit(linear_model, np.arange(len(HRrel_ts)), HRrel_ts, sigma=SE_ts, absolute_sigma=True)
    slope_mean = popt[0]
    slope_std = np.sqrt(pcov[0, 0])
    #plot the data with error bars and the fitted line
    # plt.errorbar(np.arange(len(HRrel_ts)), HRrel_ts, yerr=SE_ts, fmt='o', label='Data with error bars')
    # plt.plot(np.arange(len(HRrel_ts)), linear_model(np.arange(len(HRrel_ts)), *popt), 'r-', label='Fitted line')
    # plt.xlabel('Time (arbitrary units)')
    # plt.ylabel('Relative Sea Level (m)')
    # plt.title(f'Trend: {slope_mean:.4f} m/year ± {slope_std:.4f} m/year')
    # plt.legend()
    # plt.show()
    # quit()
    
    return slope_mean, slope_std




# Global cache for data
_data_cache = {'main': None, 'se': None}

def load_data():
    """Load data once and cache it"""
    global _data_cache
    if _data_cache['main'] is None:
        _data_cache['main'] = mat73.loadmat(r'C:\Users\123ti\Documents\Speciale_git\Speciale\temp_data\KSSLfin.mat')
        _data_cache['se'] = mat73.loadmat(r'C:\Users\123ti\Documents\Speciale_git\Speciale\temp_data\KSSL_SEfin.mat')
    return _data_cache['main'], _data_cache['se']

def get_time_indices(year):
    """Get time indices and delta_years based on measurement year"""
    data_main, _ = load_data()
    tt = data_main['tt']
    
    if year >= 1900 and year < 1983:
        s = np.where((tt >= year) & (tt <= 1983))[0]
        delta_years = 1983 - year
    elif year > 2001 and year <= 2021:
        s = np.where((tt >= 2001) & (tt <= year))[0]
        delta_years = year - 2001
    else:
        raise ValueError(f"Year {year} outside valid range (1900-2021)")
    
    return s, delta_years

def get_trend_at_location(lat, lon, year, box_size=1):
    """
    Get trend at specific location using a box around the point
    Accounts for changing grid cell area with latitude via cosine weighting
    
    box_size: degrees (default 1.0° = ~111 km)
    """
    if year >= 1983 and year <= 2001:
        return 0, 0  # no adjustment needed for current epoch
    data_main, data_se = load_data()
    s, delta_years = get_time_indices(year)
    
    HR = data_main['SL_multi'][:, s]
    GIA = data_main['GIA_Field_KS'][:, s]
    HRrel = HR + GIA

    Lcg = data_main['LALT']  # [lon, lat]
    t = data_main['tt'][s]
    
    # ...existing code...
    # Find all grid points in box around location
    lat_mask = np.abs(Lcg[:, 1] - lat) <= box_size 
    lon_mask = np.abs(Lcg[:, 0] - lon) <= box_size

    region_indices = np.where(lat_mask & lon_mask)[0]

    if len(region_indices) == 0:
        raise ValueError(f"No grid points found near ({lat}, {lon})")
    
    # Latitude-weighted average (accounts for changing cell area)
    lat_weights = np.cos(np.radians(Lcg[region_indices, 1]))
    HRrel_region = np.average(HRrel[region_indices, :], axis=0, weights=lat_weights)

    # Same weighting for error
    SE_HR = data_se['SE_SL_multi'][:, s]
    SE_GIA_base = data_se['SE_GIA']
    SE_GIA = np.outer(SE_GIA_base, t)
    SE_GIA = SE_GIA - SE_GIA[:, -1:]

    SE_HRrel = np.sqrt(SE_HR**2 + SE_GIA**2)
    SE_HRrel_region = np.average(SE_HRrel[region_indices, :], axis=0, weights=lat_weights)

    slope_mean, slope_std = compute_trend_with_error(HRrel_region, SE_HRrel_region)
    if year > 2001:
        slope_mean = -slope_mean  # reverse sign for future trends
    return slope_mean * delta_years, slope_std * delta_years


def compute_and_plot_trends(year_start=1900, year_end=2021):
    """
    Compute trends for all grid cells and plot as scatter map
    """
    data_main = mat73.loadmat(r'C:\Users\123ti\Documents\Speciale_git\Speciale\temp_data\KSSLfin.mat')
    
    s = np.where((data_main['tt'] >= year_start) & (data_main['tt'] <= year_end))[0]
    
    HR = data_main['SL_multi'][:, s]
    GIA = data_main['GIA_Field_KS'][:, s]
    HRrel = HR + GIA
    Lcg = data_main['LALT']
    t = data_main['tt'][s]
    
    # Compute trend for each grid point
    trends = np.zeros(HRrel.shape[0])
    for i in range(HRrel.shape[0]):
        slope, _, _, _, _ = scipy.stats.linregress(t, HRrel[i, :])
        trends[i] = slope * 1000  # mm/year
    
    # Plot
    plt.figure(figsize=(15, 8))
    scatter = plt.scatter(Lcg[:, 0], Lcg[:, 1], c=trends, cmap='RdBu_r', s=10, vmin=-5, vmax=5)
    plt.colorbar(scatter, label='Trend (mm/year)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Sea Level Rise Trends {year_start}-{year_end}')
    plt.show()

def compute_epoch_fix():
    save_path = r'C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\generated_data\trends_to_convert_tide_levels.csv'
    if os.path.exists(save_path):
        print("Epoch correction data already exists. Loading from file.")
        return pd.read_csv(save_path)
    storm_tide_data  = pd.read_excel(r'C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\generated_data\surge_data_manual_check.xlsx')
    #clean data for no Lat_db and Lon_db
    storm_tide_data = storm_tide_data.dropna(subset=['Lat_db', 'Lon_db'])

    lf_ISO_TIMEs = storm_tide_data['lf_ISO_TIME'].values
    lats = storm_tide_data['Lat_db'].values
    lons = storm_tide_data['Lon_db'].values
    years = storm_tide_data['Year'].values

    lf_times = []
    trends = []
    for zip_id_lat_lon_year in zip(lf_ISO_TIMEs, lats, lons, years):
        lf_ISO_TIME, lat, lon, year = zip_id_lat_lon_year
        offset, error_offset = get_trend_at_location(lat, lon, year)
        lf_times.append(lf_ISO_TIME)
        trends.append((offset, error_offset))
    trend_df = pd.DataFrame(lf_times, columns=['lf_ISO_TIME'])
    trend_df['Offset_to_1983_2001_epoch_m'] = [t[0] for t in trends]
    trend_df['Error_Offset_m'] = [t[1] for t in trends]

    trend_df.to_csv(save_path, index=False)
    return trend_df
