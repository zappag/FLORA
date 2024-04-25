#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tool to download EFAS reforecast from CDS"""

import os
import zipfile
import glob
import cdsapi
import pandas as pd
import xarray as xr
from cdo import *
cdo = Cdo()

TGTDIR = "/work_big/users/davini/EFAS/seasonal"
TMPDIR = "/work_big/users/davini/EFAS/tmp"
KIND = 'seasonal' # 'control' or 'ensemble'
VERSION = 5 # version of the EFAS reforecast, only 4 tested so far
CLEAN = True # remove the temporary files

# create the target directory if it does not exist
os.makedirs(TGTDIR, exist_ok=True)
os.makedirs(TMPDIR, exist_ok=True)

# create biweekly loop with pandas
START_DATE = "1999-01-01"
END_DATE = "1999-12-31"
NENS = 25
DELTA = 24 # delta between the leadtimes in hours
CHUNKS_DOWNLOAD = 10 # number of leadtimes to download at once
MAXLEADTIME = 215*DELTA # maximum leadtime in hours of 215 days
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
regions = ['Panaro', 'Timis', 'Lagen', 'Aragon']

# help functions
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_regions(name):
    """Get lat-lon bounds of a region"""
    if name == 'Panaro':
        return (42.2,44.8), (10.5,11.5)
    if name == 'Timis':
        return  (45.47,45.85), (20.87,21.82)
    if name == 'Lagen':
        return (61.02,62.14), (8.21, 10.8)
    if name == 'Aragon':
        return (42.30458,43.044), (-1.97765,-1.4743)
    else:
        raise ValueError(f"Region {name} not found")

def mask_efas(field, regions):
    """Create a mask for the EFAS reforecast"""
    for region in regions:
        lat, lon = get_regions(region)
        if region == regions[0]:
            mask = ((field.longitude >= lon[0]) & (field.longitude <= lon[1]) & (field.latitude >= lat[0]) & (field.latitude <= lat[1]))
        else:
            mask = mask | ((field.longitude >= lon[0]) & (field.longitude <= lon[1]) & (field.latitude >= lat[0]) & (field.latitude <= lat[1]))

    return field.where(mask)


# loop over the dates
for date in date_range:

    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')
    print(f"Downloading EFAS{VERSION} reforecast {KIND} for {year}-{month}-{day}")

    LAST = NENS - 1
    target_file = f"{TGTDIR}/EFAS{VERSION}_reforecast_{year}{month}{day}_{KIND}_{LAST}.nc"
    if os.path.exists(target_file):
        print(f"File {target_file} already exists, skipping download")
        continue
    
    # full leadtime
    steps = [str(i) for i in range(DELTA, MAXLEADTIME, DELTA)]
    for chunk in chunks(steps, n=CHUNKS_DOWNLOAD):
        print(chunk[0])
        target_file = f"{TMPDIR}/EFAS{VERSION}_reforecast_{year}{month}{day}_{KIND}_{chunk[0]}_{LAST}.nc"
        if os.path.exists(target_file):
            print(f"File {target_file} already exists, skipping download")
            continue

        zip_file = f"{TMPDIR}/EFAS_reforecast_{year}{month}{day}_{KIND}_{chunk[0]}.zip"

        # first check if the zip file is corrupted
        if os.path.exists(zip_file):
            try:
                the_zip_file = zipfile.ZipFile(zip_file)
            except zipfile.BadZipFile:
                print('Corrupted/incomplete zip file')
                os.remove(zip_file)

        # download the file
        if os.path.exists(zip_file):
            print(f"File {zip_file} already exists, skipping download")
        else:
            request =  {
            'format': 'netcdf4.zip',
            'system_version': f'version_{VERSION}_0',
            'variable': f'river_discharge_in_the_last_{DELTA}_hours',
            'model_levels': 'surface_level',
            'hyear': year,
            'hmonth': month,
            'leadtime_hour': chunk
            #'leadtime_hour': [str(i) for i in range(0, 5161, 24)]
                }
            print("Launcing the CDSAPI request...")
            print(request)
            c = cdsapi.Client()
            c.retrieve(
                'efas-seasonal-reforecast',
                request,
                zip_file)
            
        # Unzip the file
        print(f"Unzipping the file {zip_file}...")
        outdir = zip_file.replace('.zip', '')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(outdir)
            
        print("Ensemble files not processable by CDO, splitting with Xarray")
        dataset = xr.open_dataset(f"{outdir}/mars_data_0.nc")
        dataset = mask_efas(dataset, regions)
        for ensemble in dataset.number.values:
            print(f"Ensemble {ensemble}")
            ensname = f"{outdir}/mars_data_ens{ensemble}_{chunk[0]}.nc"
            dataset_ens = dataset.sel(number=ensemble).to_netcdf(ensname)
            str_ensemble = str(ensemble).zfill(2)
            target_file = f"{TMPDIR}/EFAS{VERSION}_reforecast_{year}{month}{day}_{KIND}_{chunk[0]}_{str_ensemble}.nc"

            delay = pd.Timedelta(int(chunk[0])/DELTA, unit="d")
            reference_time = pd.Timestamp(f'{year}-{month}-{day}') + pd.Timedelta(delay, unit="d")
            cdo.settaxis(f"{reference_time.strftime('%Y-%m-%d')},00:00:00,{DELTA}hours",
                        input = ensname,
                        output = target_file,
                        options = '-f nc4 -z zip')
            if CLEAN:
                os.remove(ensname)

        if CLEAN:
            os.remove(f"{outdir}/mars_data_0.nc")
            os.rmdir(outdir)
            os.remove(zip_file)
        
    # merge the files
    for ens in range(NENS):
        str_ensemble = str(ens).zfill(2)
        files = f"{TMPDIR}/EFAS{VERSION}_reforecast_{year}{month}{day}_{KIND}_*_{str_ensemble}.nc"
        cdo.merge(input = files, output = f"{TGTDIR}/EFAS{VERSION}_reforecast_{year}{month}{day}_{KIND}_{str_ensemble}.nc")
        if CLEAN:
            for file in glob.glob(files):
                os.remove(file)

    






