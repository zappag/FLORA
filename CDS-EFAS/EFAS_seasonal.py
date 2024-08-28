#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tool to download EFAS reforecast from CDS"""

import os
import zipfile
import glob
import logging
import cdsapi
import pandas as pd
import xarray as xr
from cdo import *
from functions import chunks, fixed_region, mask_efas
cdo = Cdo()

logging.warning("Launching the EFAS seasonal downloader...")

TGTDIR = "/work_big/users/davini/EFAS/seasonal-v3"
TMPDIR = "/work_big/users/davini/EFAS/tmp_regions"
KIND = 'seasonal' # 'control' or 'ensemble'
VERSION = 5 # version of the EFAS reforecast, only 4 tested so far
CLEAN = True # remove the temporary files
MASK = False # to mask everything

# create the target directory if it does not exist
os.makedirs(TGTDIR, exist_ok=True)
os.makedirs(TMPDIR, exist_ok=True)

# create biweekly loop with pandas
START_DATE = "1999-01-01"
END_DATE = "2022-01-01"
BOXSIZE = .0 #how  many degrees do you want to extend the boundaries
NENS = 25 # number of ensemble members
DELTA = 24 # delta between the leadtimes in hours
CHUNKS_DOWNLOAD = 215 # number of leadtimes to download at once
MAXLEADTIME = 215*DELTA # maximum leadtime in hours of 215 days
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
REGIONS = ['Panaro', 'Timis', 'Lagen', 'Aragon', 'Reno']


# loop over the dates
for date in date_range:

    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')
    LAST = NENS - 1
    for region in REGIONS:
        target_file = f"{TGTDIR}/EFAS{VERSION}_reforecast_{region}_{year}{month}{day}_{KIND}_{LAST}.nc"
        if os.path.exists(target_file):
            print(f"File {target_file} already exists, skipping download")
            continue
    
        # full leadtime
        logging.warning(f"Downloading EFAS{VERSION} reforecast {KIND} for {year}-{month}-{day} in {region}")
        lat, lon = fixed_region(name=region, delta=BOXSIZE)
        print(lat, lon)
        steps = [str(i) for i in range(DELTA, MAXLEADTIME, DELTA)]
        for chunk in chunks(steps, n=CHUNKS_DOWNLOAD):
            target_file = f"{TMPDIR}/EFAS{VERSION}_reforecast__{region}_{year}{month}{day}_{KIND}_{chunk[0]}_{LAST}.nc"
            if os.path.exists(target_file):
                logging.warning(f"File {target_file} already exists, skipping download")
                continue

            zip_file = f"{TMPDIR}/EFAS_reforecast_{region}_{year}{month}{day}_{KIND}_{chunk[0]}.zip"

            # first check if the zip file is corrupted
            if os.path.exists(zip_file):
                try:
                    the_zip_file = zipfile.ZipFile(zip_file)
                except zipfile.BadZipFile:
                    logging.error('Corrupted/incomplete zip file')
                    os.remove(zip_file)

            # download the file
            if os.path.exists(zip_file):
                logging.warning(f"File {zip_file} already exists, skipping download")
            else:
                request =  {
                'data_format': 'netcdf4',
                'download_format': 'zip',
                'system_version': f'version_{VERSION}_0',
                'variable': f'river_discharge_in_the_last_{DELTA}_hours',
                'model_levels': 'surface_level',
                'hyear': year,
                'hmonth': month,
                'leadtime_hour': chunk,
                'area': [lat[1], lon[0], lat[0], lon[1]]
                #'leadtime_hour': [str(i) for i in range(0, 5161, 24)]
                    }
                logging.warning("Launcing the CDSAPI request...")
                logging.warning(request)
                c = cdsapi.Client()
                c.retrieve(
                    'efas-seasonal-reforecast',
                    request,
                    zip_file)
                
            # Unzip the file
            logging.warning(f"Unzipping the file {zip_file}...")
            outdir = zip_file.replace('.zip', '')
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(outdir)
                
            logging.warning("Ensemble files not processable by CDO, splitting with Xarray")
            dataset = xr.open_dataset(f"{outdir}/data_0.nc")
            if MASK:
                dataset = mask_efas(dataset, REGIONS)
            for ensemble in dataset.number.values:
                logging.warning(f"Ensemble {ensemble}")
                ensname = f"{outdir}/mars_data_ens{ensemble}_{chunk[0]}.nc"
                dataset_ens = dataset.sel(number=ensemble).to_netcdf(ensname)
                str_ensemble = str(ensemble).zfill(2)
                target_file = f"{TMPDIR}/EFAS{VERSION}_reforecast_{region}_{year}{month}{day}_{KIND}_{chunk[0]}_{str_ensemble}.nc"

                delay = pd.Timedelta(int(chunk[0])/DELTA, unit="d")
                reference_time = pd.Timestamp(f'{year}-{month}-{day}') + pd.Timedelta(delay, unit="d")
                cdo.settaxis(f"{reference_time.strftime('%Y-%m-%d')},00:00:00,{DELTA}hours",
                            input = ensname,
                            output = target_file,
                            options = '-f nc4 -z zip')
                if CLEAN:
                    os.remove(ensname)

            if CLEAN:
                os.remove(f"{outdir}/data_0.nc")
                #os.rmdir(outdir)
                os.remove(zip_file)
            
        # merge the files
        for ens in range(NENS):
            logging.warning("Merging multiple for chunks for ensemble member %s", ens)
            str_ensemble = str(ens).zfill(2)
            files = f"{TMPDIR}/EFAS{VERSION}_reforecast_{region}_{year}{month}{day}_{KIND}_*_{str_ensemble}.nc"
            WRITEDIR = os.path.join(TGTDIR, region)
            os.makedirs(WRITEDIR, exist_ok=True)
            cdo.merge(input = files, output = f"{WRITEDIR}/EFAS{VERSION}_reforecast_{region}_{year}{month}{day}_{KIND}_{str_ensemble}.nc")
            if CLEAN:
                for file in glob.glob(files):
                    os.remove(file)

        






