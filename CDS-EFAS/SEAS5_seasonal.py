#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tool to download SEAS5 reforecast from CDS"""

import os
import glob
import logging
import cdsapi
import pandas as pd
import xarray as xr
from cdo import *
from functions import chunks
cdo = Cdo()

logging.warning("Launching the SEAS5 seasonal downloader...")

TGTDIR = "/work_big/users/davini/SEAS5/seasonal-v1"
TMPDIR = "/work_big/users/davini/SEAS5/tmp"
CLEAN = True # remove the temporary files

# create the target directory if it does not exist
os.makedirs(TGTDIR, exist_ok=True)
os.makedirs(TMPDIR, exist_ok=True)

# create biweekly loop with pandas
START_DATE = "1999-01-01"
END_DATE = "1999-02-01"
NENS = 25 # number of ensemble members
DELTA = 6 # delta between the leadtimes in hours
CHUNKS_DOWNLOAD = 5160 # number of leadtimes to download at once
MAXLEADTIME = 5160 # maximum leadtime in hours of 215 days
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
variables = ['mean_sea_level_pressure', 'total_precipitation']


# loop over the dates
for date in date_range:

    for variable in variables:

        year = date.strftime('%Y')
        month = date.strftime('%m')
        day = date.strftime('%d')
        LAST = NENS - 1
        target_file = f"{TGTDIR}/SEAS5_{variable}_{year}{month}{day}_{LAST}.nc"
        if os.path.exists(target_file):
            print(f"File {target_file} already exists, skipping download")
            continue

        # full leadtime
        logging.warning(f"Downloading SEAS5 seasonal for {variable} for {year}-{month}-{day}")
        steps = [str(i) for i in range(DELTA, MAXLEADTIME, DELTA)]
        for chunk in chunks(steps, n=CHUNKS_DOWNLOAD):
            target_file = f"{TMPDIR}/SEAS5_{variable}_{year}{month}{day}_{chunk[0]}_{LAST}.nc"
            if os.path.exists(target_file):
                logging.warning(f"File {target_file} already exists, skipping download")
                continue

            single_file = f"{TMPDIR}/SEAS5_{variable}_{year}{month}{day}_{chunk[0]}.nc"

            # first check if the zip file is corrupted
            #if os.path.exists(zip_file):
            #    try:
            #        the_zip_file = zipfile.ZipFile(zip_file)
            #    except zipfile.BadZipFile:
            #        logging.error('Corrupted/incomplete zip file')
            #        os.remove(zip_file)

            # download the file
            if os.path.exists(single_file):
                logging.warning(f"File {single_file} already exists, skipping download")
            else:
                request =  {
                'format': 'netcdf',
                'originating_centre': 'ecmwf',
                'system': '51',
                'year': year,
                'variable': variable,
                'month': month,
                'day': '01',
                'leadtime_hour': chunk,
                #'area': [lat[1], lon[0], lat[0], lon[1]]
                #'leadtime_hour': [str(i) for i in range(0, 5161, 24)]
                    }
                logging.warning("Launching the CDSAPI request...")
                logging.warning(request)
                c = cdsapi.Client()
                c.retrieve(
                    'seasonal-original-single-levels',
                    request,
                    single_file)
                

            logging.warning("Ensemble files not processable by CDO, splitting with Xarray")
            dataset = xr.open_dataset(single_file)
            for ensemble in dataset.number.values:
                logging.warning(f"Ensemble {ensemble}")
                #ensname = f"{TMPDIR}/temp_data_ens{ensemble}_{chunk[0]}.nc"
                str_ensemble = str(ensemble).zfill(2)
                target_file = f"{TMPDIR}/SEAS5_{variable}_{year}{month}{day}_{chunk[0]}_{str_ensemble}.nc"
                dataset_ens = dataset.sel(number=ensemble).to_netcdf(target_file)
                #target_file = f"{TMPDIR}/SEAS5_{variable}_{year}{month}{day}_{chunk[0]}_{str_ensemble}.nc"

                #delay = pd.Timedelta(int(chunk[0])/24, unit="d")
                #reference_time = pd.Timestamp(f'{year}-{month}-{day}') + pd.Timedelta(delay, unit="d")
                #cdo.settaxis(f"{reference_time.strftime('%Y-%m-%d')},00:00:00,1day",
                #            input = ensname,
                #            output = target_file,
                #            options = '-f nc4 -z zip')
                #if CLEAN:
                #    os.remove(ensname)

            if CLEAN:
                os.remove(single_file)
            
        # merge the files
        for ens in range(NENS):
            logging.warning("Merging multiple for chunks for ensemble member %s", ens)
            str_ensemble = str(ens).zfill(2)
            files = f"{TMPDIR}/SEAS5_{variable}_{year}{month}{day}_*_{str_ensemble}.nc"
            cdo.merge(input = files, output = f"{TGTDIR}/SEAS5_{variable}_{year}{month}{day}_{str_ensemble}.nc")
            if CLEAN:
                for file in glob.glob(files):
                    os.remove(file)
