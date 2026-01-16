import pandas as pd
import numpy as np
from generate_training_data import calculate_mean_wind_radius
feature_columns = ['LAT', 'LON', 'USA_WIND', 'USA_PRES', 'STORM_SPEED_ms', 'Month' , 'Year', 'Day', 'DIST2LAND_m', 'STORM_DIR']


def estimate_mean_wind_radius(df, radius):

    df[radius + '_MEAN_RADIUS'] = df.apply(
        lambda row: calculate_mean_wind_radius(
            row['USA_' +radius + '_NE'],
            row['USA_' +radius + '_SE'],
            row['USA_' +radius + '_SW'],
            row['USA_' +radius + '_NW']
        ), axis=1
    )
    df_pre2004 = df[df['YEAR'] < 2004]
    df_post2004 = df[df['YEAR'] >= 2004]




