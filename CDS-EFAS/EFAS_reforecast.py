#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tool to download EFAS reforecast from CDS"""

import os
import zipfile
import cdsapi
import pandas as pd
from cdo import *
from pandas.tseries.offsets import Day
cdo = Cdo()

TGTDIR = "/work_big/users/davini/EFAS"
KIND = 'control' # 'control' or 'ensemble'
VERSION = 4

# create the target directory if it does not exist
if not os.path.exists(TGTDIR):
    os.makedirs(TGTDIR)

# create biweekly loop with pandas
START_DATE = "1999-01-03"
END_DATE = "1999-01-31"
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='7D')
total_date_range = date_range.union(date_range + Day(4))


for date in total_date_range:

    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')

    outfile = os.path.join(TGTDIR,
                           f'EFAS_reforecast_{year}{month}{day}_{KIND}.zip')

    request =  {
        'format': 'netcdf4.zip',
        'system_version': f'version_{VERSION}_0',
        'variable': 'river_discharge_in_the_last_6_hours',
        'model_levels': 'surface_level',
        'hyear': year,
        'hmonth': month,
        'leadtime_hour': [str(i) for i in range(0, 1105, 6)],
        'hday': day,
            }

    # set the product type
    if KIND == 'control':
        request['product_type'] = 'control_forecast'
    elif KIND == 'ensemble':
        request['product_type'] = 'ensemble_perturbed_forecasts'

    print(request)

    # make the request
    if not os.path.exists(outfile):
        c = cdsapi.Client()
        c.retrieve(
            'efas-reforecast',
            request,
            outfile)

    # Unzip the file
    outdir = outfile.replace('.zip', '')
    with zipfile.ZipFile(outfile, 'r') as zip_ref:
        zip_ref.extractall(outdir)
    #os.remove(outfile)

    cdo.settaxis(f'{year}-{month}-{day},00:00:00,6hour',
                 input = f"-selname,dis06 {outdir}/mars_data_0.nc",
                 output = f"{outdir}/EFAS_reforecast_{year}{month}{day}_{KIND}.nc")



