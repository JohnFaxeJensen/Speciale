import os
import pyfes

import netCDF4
import numpy as np
import pandas as pd

data = pd.read_excel("./Aslak_data.xls", sheet_name='ATD of ICAT', engine='xlrd')

dates = pd.to_datetime(data['lf_ISO_TIME']).values
print(dates)

