"""
Centralized utilities for ML modules to avoid circular imports.
Contains shared constants, functions, and utilities used across classifiers and regressors.
"""

import numpy as np
import pandas as pd
from scipy.special import inv_boxcox


# ============================================================
# FEATURE COLUMNS - Used for all models
# ============================================================
FEATURE_COLUMNS = [
    'LAT', 'LON', 'USA_WIND', 'USA_PRES', 'STORM_SPEED_ms', 
    'Month', 'Year', 'Day', 'DIST2LAND_m', 'STORM_DIR'
]


# ============================================================
# BOX-COX TRANSFORMATION UTILITIES
# ============================================================
def safe_inv_boxcox(y_transformed, lambda_param, offset=0.1):
    """
    Inverse Box-Cox transform and remove offset, handling NaN values.
    
    Parameters:
    -----------
    y_transformed : array-like
        Box-Cox transformed values
    lambda_param : float
        Lambda parameter from Box-Cox transformation
    offset : float, default=0.1
        Offset that was added before Box-Cox transformation
        
    Returns:
    --------
    array-like
        Back-transformed values in original scale (non-negative)
    """
    y_original = inv_boxcox(y_transformed, lambda_param)
    y_original = np.maximum(y_original - offset, 0)  # Ensure non-negative
    # Replace any NaN with 0
    y_original = np.where(np.isnan(y_original), 0, y_original)
    return y_original


def calculate_wind_field_area(ne, se, sw, nw):
    """
    Calculate wind field area intelligently:
    - All 4 quadrants: use elliptical approximation (captures asymmetry)
    - <4 quadrants: sum circular sectors (no data inference)
    
    Parameters:
    -----------
    ne, se, sw, nw : float or nan
        Radii in nautical miles for NE, SE, SW, NW quadrants
        
    Returns:
    --------
    float
        Area in km²
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
    
    Parameters:
    -----------
    ne, se, sw, nw : float or nan
        Radii in nautical miles for NE, SE, SW, NW quadrants
        
    Returns:
    --------
    float
        Mean radius in km, or np.nan if no valid radii
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
            radii.append(0)  # Pad with zeros if less than 4 quadrants
    
    return np.mean(radii)
