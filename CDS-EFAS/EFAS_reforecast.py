#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tool to download EFAS reforecast from CDS"""

import os
import zipfile
import cdsapi
import pandas as pd
import xarray as xr
from cdo import *
from pandas.tseries.offsets import Day
cdo = Cdo()

TGTDIR = "/work_big/users/davini/EFAS"
TMPDIR = "/work_big/users/davini/EFAS/tmp"
KIND = 'ensemble' # 'control' or 'ensemble'
VERSION = 4 # version of the EFAS reforecast, only 4 tested so far
CLEAN = True # remove the temporary files

# create the target directory if it does not exist
os.makedirs(TGTDIR, exist_ok=True)
os.makedirs(TMPDIR, exist_ok=True)

# create biweekly loop with pandas
START_DATE = "1999-01-03"
END_DATE = "1999-01-31"
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='7D')
total_date_range = date_range.union(date_range + Day(4))

# loop over the dates
for date in total_date_range:

    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')
    print(f"Downloading EFAS reforecast {KIND} for {year}-{month}-{day}")

    if KIND == 'control':
        target_file = f"{TGTDIR}/EFAS_reforecast_{year}{month}{day}_{KIND}.nc"
    elif KIND == 'ensemble':
        target_file = f"{TGTDIR}/EFAS_reforecast_{year}{month}{day}_{KIND}_10.nc"  
    zip_file = f"{TMPDIR}/EFAS_reforecast_{year}{month}{day}_{KIND}.zip"

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

    if os.path.exists(target_file):
        print(f"File {target_file} already exists, skipping download")
    else:

        if os.path.exists(zip_file):
            print(f"File {zip_file} already exists, skipping download")
        else:
            print("Launcing the CDSAPI request...")
            c = cdsapi.Client()
            c.retrieve(
                'efas-reforecast',
                request,
                zip_file)
            
        # Unzip the file
        print(f"Unzipping the file {zip_file}...")
        outdir = zip_file.replace('.zip', '')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(outdir)
        if CLEAN:
            os.remove(zip_file)

        # this works only for the control
        if KIND == 'control':
            # compress and set the time axis
            print(f"Compressing and setting the time axis of the file {target_file}...")
            cdo.settaxis(f'{year}-{month}-{day},00:00:00,6hour',
                        input = f"{outdir}/mars_data_0.nc",
                        output = target_file,
                        options = '-f nc4 -z zip')
            
        else:
            print("Ensemble files not processable by CDO, splitting with Xarray")
            dataset = xr.open_dataset(f"{outdir}/mars_data_0.nc")
            for ensemble in dataset.number.values:
                print(f"Ensemble {ensemble}")
                ensname = f"{outdir}/mars_data_ens{ensemble}.nc"
                dataset_ens = dataset.sel(number=ensemble).to_netcdf(ensname)
                str_ensemble = str(ensemble).zfill(2)
                target_file = f"{TGTDIR}/EFAS_reforecast_{year}{month}{day}_{KIND}_{str_ensemble}.nc"

                cdo.settaxis(f'{year}-{month}-{day},00:00:00,6hour',
                            input = ensname,
                            output = target_file,
                            options = '-f nc4 -z zip')
                if CLEAN:
                    os.remove(ensname)

        if CLEAN:
            os.remove(f"{outdir}/mars_data_0.nc")
            os.rmdir(outdir)

    






