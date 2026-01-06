import pandas as pd


def merge_temp_data(df_hurricane, df_temp, merge_col_temp):
    #extract year from date in hurricane data
    df_hurricane['Year'] = pd.to_datetime(df_hurricane['lf_ISO_TIME']).dt.year
    merged_hurricane = pd.merge(df_hurricane, df_temp, how='left', left_on='Year', right_on=merge_col_temp)
    merged_hurricane.rename({'Anomaly': 'Anomaly_global'}, axis=1, inplace=True)
    return merged_hurricane


def merge_temp_data_monthly(df_hurricane, df_temp, merge_col_temp):
    #extract year and month from date in hurricane data
    df_hurricane['Year'] = pd.to_datetime(df_hurricane['lf_ISO_TIME']).dt.year
    df_hurricane['Month'] = pd.to_datetime(df_hurricane['lf_ISO_TIME']).dt.month
    merged_hurricane = pd.merge(df_hurricane, df_temp, how='left', left_on=['Year', 'Month'], right_on=merge_col_temp)
    return merged_hurricane
