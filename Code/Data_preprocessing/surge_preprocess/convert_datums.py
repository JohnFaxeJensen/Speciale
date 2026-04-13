#this script is used to convert datums for the tide observations in the excel file
import os
from datetime import datetime
import re
from matplotlib import path
import requests
import pandas as pd


def call_datum_api(current_datum, height, lat, lon, region,target_metric = 'LMSL', absolute_string=''):
    #construct api call to convert all unknown datums to a common datum, e.g. NAVD88
    #this is a placeholder function and should be implemented with actual API calls
    #Find appropiate region to call:
    api =  f"https://vdatum.noaa.gov/vdatumweb/api/convert?s_v_frame={current_datum}&s_y={lat}&s_x={lon}&t_v_frame={target_metric}&units=meters&s_z={height}&s_v_unit=m&t_v_unit=m&region={region}"
    if current_datum == 'NGVD29':
        api = f"https://vdatum.noaa.gov/vdatumweb/api/convert?s_v_frame={current_datum}&s_y={lat}&s_x={lon}&t_v_frame={target_metric}&units=meters&s_z={height}&s_v_unit=m&t_v_unit=m&region={region}&s_h_frame=NAD27&t_h_frame=IGS14"
    if region == 'chesapeak_delaware' or region == 'wgom'  and current_datum !='NAVD88' and current_datum != 'NGVD29':
        api = f"https://vdatum.noaa.gov/vdatumweb/api/convert?s_v_frame={current_datum}&s_y={lat}&s_x={lon}&t_v_frame={target_metric}&units=meters&s_z={height}&s_v_unit=m&t_v_unit=m&region={region}&s_h_frame=IGS14&t_h_frame=IGS14"
    if absolute_string != '':
        api = absolute_string
    response = requests.get(api)
    if response.status_code == 200:
        data = response.json()
        return data, api
# manual_tweaks = {
#     'ID_8' : [29.224353380920125, -90.65386626746059],
#     'ID_115' : [30.109960897443557,-84.20482042792997],
#     'ID_239': [29.190595333223435, -89.2709359372307], #this one is quite far away, but the tidal range is quite low so prop okay:)
#     'ID_173': [29.175299739135266,-90.61787279040159],
#     'ID_226': [29.7688952695177, -93.18726130026378],
#     'ID_234': [29.54785677157314, -94.38455920351721]
# }
manual_tweaks = {
    # convert format: 9/21/1909  12:00:00 AM to datetime
    pd.to_datetime('9/21/1909  12:00:00 AM'): [29.224353380920125, -90.65386626746059],
    pd.to_datetime('6/9/1966  8:00:00 PM'): [30.109960897443557,-84.20482042792997],
    pd.to_datetime('8/29/2012  7:00:00 AM'): [29.190595333223435, -89.2709359372307], #this one is quite far away, but the tidal range is quite low so prop okay:)
    pd.to_datetime('8/26/1992  6:30:00 AM'): [29.175299739135266,-90.61787279040159],
    pd.to_datetime('9/24/2005  8:30:00 AM'): [29.7688952695177, -93.18726130026378],
    pd.to_datetime('9/13/2008  9:00:00 AM'): [29.54785677157314, -94.38455920351721]
}
def convert_datums():
    path = r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\generated_data\surge_data_manual_check" 
    additional_path = "_converted"
    if os.path.exists(path + additional_path + '.csv'):
        print("Converted surge data already exists. Loading from file.")
        return pd.read_csv(path + additional_path + '.csv')
    else:
        df_existing = pd.read_excel(path + '.xlsx')
        df_to_convert = df_existing[~df_existing['Datum'].isin(['Unknown', 'MSL', 'Above Normal'])]
        df_to_convert = df_to_convert[df_to_convert['Datum'].notna()]
        df_to_convert = df_to_convert[['lf_ISO_TIME', 'Datum', 'Storm_Tide_m', 'Lat_db', 'Lon_db']]
        heights = []
        uncertainties = []
        source_frames = []
        target_frames = []
        for idx, row in df_to_convert.iterrows():
            lf_time = row['lf_ISO_TIME']
            datum = row['Datum']
            height = row['Storm_Tide_m']
            lat = row['Lat_db']
            lon = row['Lon_db']
            if lf_time in manual_tweaks:
                lat, lon = manual_tweaks[lf_time]
            regions = {'Contiguous United States': 'contiguous', 'Chesapeake_Delaware': 'chesapeak_delaware', 'Western_Gulf_of_Mexico': 'wgom'}
            if lat >= 36.0 and lat <= 39.466012 and lon >= -77.036871 and lon <= -75.000000:
                region = 'chesapeak_delaware'
            elif lat >= 25.8371 and lat <= 30 and lon >= -97.4344 and lon <= -81.3844:
                region = 'wgom'
            elif lat >= 24.396308 and lat <= 49.384358 and lon >= -125.0 and lon <= -66.93457:
                region = 'contiguous'
            if datum == 'Unknown' or pd.isna(datum) or datum == 'MSL':
                continue
            result,api = call_datum_api(datum, height, lat, lon, region)
            if 't_z' in result and 'uncertainty' in result:
                heights.append(result['t_z'])
                uncertainties.append(result['uncertainty'])
                source_frames.append(result['s_h_frame'])
                target_frames.append(result['t_h_frame'])
                continue
            else:
                if 'errorCode' in result and result['errorCode'] == 412 and datum == 'NGVD29':
                    absolute_string = f"https://vdatum.noaa.gov/vdatumweb/api/convert?s_v_frame={datum}&s_y={lat}&s_x={lon}&t_v_frame=LMSL&units=meters&s_z={height}&s_v_unit=m&t_v_unit=m&region={region}&s_h_frame=NAD27"
                    result,api =call_datum_api(datum, height, lat, lon, region, absolute_string=absolute_string)
                    if 't_z' in result and 'uncertainty' in result:
                        heights.append(result['t_z'])
                        uncertainties.append(result['uncertainty'])
                        source_frames.append(result['s_h_frame'])
                        target_frames.append(result['t_h_frame'])
                        continue
                if 'errorCode' in result and result['message'] == 'For West Gulf Coast Region, Target Horizontal Frame should be IGS14 for Tidal':
                    absolute_string = f"https://vdatum.noaa.gov/vdatumweb/api/convert?s_v_frame={datum}&s_y={lat}&s_x={lon}&t_v_frame=LMSL&units=meters&s_z={height}&s_v_unit=m&t_v_unit=m&region={region}&t_h_frame=IGS14"
                    result,api =call_datum_api(datum, height, lat, lon, region, absolute_string=absolute_string)
                    if 't_z' in result and 'uncertainty' in result:
                        heights.append(result['t_z'])
                        uncertainties.append(result['uncertainty'])
                        source_frames.append(result['s_h_frame'])
                        target_frames.append(result['t_h_frame'])
                        continue
                if 'errorCode' in result and datum == 'NGVD29':
                    absolute_string = f"https://vdatum.noaa.gov/vdatumweb/api/convert?s_v_frame={datum}&s_y={lat}&s_x={lon}&t_v_frame=LMSL&units=meters&s_z={height}&s_v_unit=m&t_v_unit=m&region=contiguous&s_h_frame=NAD27"
                    result,api =call_datum_api(datum, height, lat, lon, region, absolute_string=absolute_string)
                    if 't_z' in result and 'uncertainty' in result:
                        heights.append(result['t_z'])
                        uncertainties.append(result['uncertainty'])
                        source_frames.append(result['s_h_frame'])
                        target_frames.append(result['t_h_frame'])
                        continue
                if 'errorCode' in result and region == 'wgom':
                    absolute_string = f"https://vdatum.noaa.gov/vdatumweb/api/convert?s_v_frame={datum}&s_y={lat}&s_x={lon}&t_v_frame=LMSL&units=meters&s_z={height}&s_v_unit=m&t_v_unit=m&region=contiguous"
                    result,api =call_datum_api(datum, height, lat, lon, region, absolute_string=absolute_string)
                    if 't_z' in result and 'uncertainty' in result:
                        heights.append(result['t_z'])
                        uncertainties.append(result['uncertainty'])
                        source_frames.append(result['s_h_frame'])
                        target_frames.append(result['t_h_frame'])
                        continue

                else:
                    print(f"new error for {lf_time}: {result}")
                    print(datum, height, lat, lon, region)
                    heights.append(pd.NA)
                    uncertainties.append(pd.NA)
                    source_frames.append(pd.NA)
                    target_frames.append(pd.NA)
                    continue
                print(f"Error for {lf_time}: {result}")
                heights.append(pd.NA)
                uncertainties.append(pd.NA)
                source_frames.append(pd.NA)
                target_frames.append(pd.NA)
                
        df_to_convert['Converted_value'] = heights
        df_to_convert['Converted_uncertainty'] = uncertainties
        df_to_convert['Source_frame'] = source_frames
        df_to_convert['Target_frame'] = target_frames
        df_to_convert.to_csv(path + additional_path + '.csv', index=False)
        return df_to_convert
    
