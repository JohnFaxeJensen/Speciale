from ibtracs import Ibtracs
import os
import pandas as pd
import numpy as np
from geopy import distance
import matplotlib.pyplot as plt
I = Ibtracs() # missing some ids

# raw_hurricane_data = pd.read_csv(r"C:\Users\123ti\Downloads\ibtracs.ALL.list.v04r01.csv", usecols=["ISO_TIME", "LAT", "LON", "WMO_WIND", "WMO_PRES","USA_ATCF_ID", "USA_LAT", "USA_LON", "USA_WIND", "USA_PRES" ],low_memory=False)
# hurricane_data = pd.read_csv(r"./Speciale/Hurricane_data/Aslak_data_with_tide.csv")
# unique_ids = hurricane_data['ATCF_ID'].unique()
# filtered_hurricane_data = raw_hurricane_data[raw_hurricane_data['USA_ATCF_ID'].isin(unique_ids)]
# #save to csv
# filtered_hurricane_data.to_csv(r"./Speciale/Hurricane_data/IBTrACS_filtered_data.csv", index=False)

def calculate_travelspeed(hurricane_id, landfall_time, landfall_lat, landfall_lon, Filtered_IBTrACS_data):
    times = Filtered_IBTrACS_data['ISO_TIME']
    lats = Filtered_IBTrACS_data['LAT'].values
    lons = Filtered_IBTrACS_data['LON'].values

    # print(type(times[0]))
    if(hurricane_id == "AL051965"):
        landfall_time = pd.to_datetime("1965-09-29 20:00:00") #fix known error in data seen on Ibtracs
    else:
        landfall_time = pd.to_datetime(landfall_time)
    # print("Landfall time:", landfall_time_np)
    
    # Find times before and after landfall
    times_before = times[times < landfall_time]
    times_after = times[times > landfall_time]
    # Get the closest before (max) and closest after (min)
    if len(times_before) == 0 or len(times_after) == 0:
        print("Not enough data before/after landfall time to calculate speed.")
        print('hurricane id:', hurricane_id)
        print(times)

        return None, None, None
    
    before_time = times_before.max()
    after_time = times_after.min()

    # Get the actual timestamps
    closest_times = [before_time, after_time]
    # print("Closest times:", closest_times, "landfall time:", landfall_time)
    # print("Before time:", before_time)
    # print("After time:", after_time)
    # Get the corresponding latitudes and longitudes
    before_index = np.where(times == before_time)[0][0]
    after_index = np.where(times == after_time)[0][0]
    before_lat = lats[before_index]
    before_lon = lons[before_index]
    after_lat = lats[after_index]
    after_lon = lons[after_index]
    landfall_geo = (landfall_lat, landfall_lon)
    before_geo = (before_lat, before_lon)
    after_geo = (after_lat, after_lon)
    # print("Before lat, lon:", before_geo)
    # print("Landfall lat, lon:", landfall_geo)
    # print("After lat, lon:", after_geo)
    # Calculate distances
    delta_dist_before = distance.distance(landfall_geo, before_geo).m
    delta_dist_after = distance.distance(landfall_geo, after_geo).m
    # Calculate time differences in seconds
    delta_time_before = abs((landfall_time - before_time).total_seconds())
    delta_time_after = abs((after_time - landfall_time).total_seconds())

    # Calculate speeds in m/s
    speed_before = delta_dist_before / delta_time_before
    speed_after = delta_dist_after / delta_time_after
    #calculate speed at landfall as derivative
    speed_at_landfall = (distance.distance(before_geo, after_geo).m) / (delta_time_before + delta_time_after) #assume constant acceleration
    if speed_before > 20 or speed_after > 20 or speed_at_landfall > 20:
        print(hurricane_id, "Unrealistic speed calculated, check data:")
        print("Speeds (m/s):", speed_before, speed_after, speed_at_landfall)
        print("Times:", before_time, landfall_time, after_time)
        print("Locations:", before_geo, landfall_geo, after_geo)
    return speed_before, speed_after, speed_at_landfall

 
hurricane_data = pd.read_csv(r"./Speciale/Hurricane_data/Aslak_data_with_tide.csv")
relevant_data = hurricane_data[['ATCF_ID', 'lf_ISO_TIME', 'lf_wind', 'lf_pressure' , 'lf_lat', 'lf_lon']]
Ibtracs_data = pd.read_csv(r"./Speciale/Hurricane_data/IBTrACS_filtered_data.csv")
Ibtracs_data['ISO_TIME'] = pd.to_datetime(Ibtracs_data['ISO_TIME'],errors='coerce')

speeds_before = []
speeds_after = []
speeds_at_landfall = []
i = 0
for index, row in relevant_data.iterrows():
    hurricane_id = row['ATCF_ID']
    landfall_time = row['lf_ISO_TIME']
    landfall_lat = row['lf_lat']
    landfall_lon = row['lf_lon']
    test_time = landfall_time
    test_lat = landfall_lat
    test_lon = landfall_lon
    test_ID = hurricane_id
    filtered_IBTrACS_data = Ibtracs_data[Ibtracs_data['USA_ATCF_ID'] == hurricane_id]
    speed_before, speed_after, speed_at_landfall = calculate_travelspeed(hurricane_id, landfall_time, landfall_lat, landfall_lon, filtered_IBTrACS_data)
    speeds_before.append(speed_before)
    speeds_after.append(speed_after)
    speeds_at_landfall.append(speed_at_landfall)


hurricane_data['travel_speed_before_landfall_m_s'] = speeds_before
hurricane_data['travel_speed_after_landfall_m_s'] = speeds_after
hurricane_data['travel_speed_at_landfall_m_s'] = speeds_at_landfall
hurricane_data.to_csv(r"./Speciale/Hurricane_data/Aslak_data_with_tide_and_travelspeed.csv", index=False)

basedamage = hurricane_data['basedamage']
normBasedamage = hurricane_data['ND']
plt.scatter(np.log(basedamage), hurricane_data['travel_speed_after_landfall_m_s'], label='Travel Speed After Landfall', alpha=0.5)
#plt.scatter(np.log(basedamage), hurricane_data['travel_speed_before_landfall_m_s'], label='Travel Speed Before Landfall', alpha=0.5)
#plt.scatter(np.log(basedamage), hurricane_data['travel_speed_at_landfall_m_s'], label='Travel Speed At Landfall', alpha=0.5)
#plt.scatter(np.log(normBasedamage), hurricane_data['travel_speed_after_landfall_m_s'], label='Travel Speed After Landfall', alpha=0.5)
#plt.scatter(np.log(normBasedamage), hurricane_data['travel_speed_before_landfall_m_s'], label='Travel Speed Before Landfall', alpha=0.5)
#plt.scatter(np.log(normBasedamage), hurricane_data['travel_speed_at_landfall_m_s'], label='Travel Speed At Landfall', alpha=0.5)

plt.xlabel("ln(Base Damage)")
plt.ylabel("Travel Speed (m/s)")
plt.legend()
plt.show()
