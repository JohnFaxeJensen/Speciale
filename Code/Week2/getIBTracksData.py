from ibtracs import Ibtracs
import os
import pandas as pd
I = Ibtracs()

tc = I.get_storm_from_atcfid('AL092017')
print(vars(tc).keys())

for t, vmax,pressure, lat, lon in zip(tc.time, tc.wind, tc.mslp, tc.lat, tc.lon):
    print(t, vmax, pressure)


# os.chdir(r"C:\Users\123ti\Downloads")
# df = pd.read_csv("ibtracs.ALL.list.v04r01.csv", low_memory=False)
# print(df.columns)
# df_columns = pd.read_csv("ibtracs.ALL.list.v04r01.csv", nrows=0)
# print(df_columns.columns.tolist())