#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tool to download EFAS reforecast from CDS"""

import os
import zipfile
import glob
import logging
import argparse
import cdsapi
import pandas as pd
import xarray as xr
from cdo import *
from functions import chunks, fixed_region, mask_efas
cdo = Cdo()

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
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


# Set up command line argument parsing
parser = argparse.ArgumentParser(description='EFAS reforecast downloader.')
parser.add_argument('--year', type=int, required=True, help='The year to start the forecast (e.g. 2016)')
parser.add_argument('--region', type=str, required=True, help='The region for which to download the reforecast (e.g. Panaro)')
args = parser.parse_args()

# Extract year and region from the command line arguments
year = args.year
region = args.region
year1 = year + 2

# create biweekly loop with pandas
START_DATE = f"{year}-01-01"
END_DATE = f"{year1}-01-01"
BOXSIZE = .0 #how  many degrees do you want to extend the boundaries
NENS = 25 # number of ensemble members
DELTA = 24 # delta between the leadtimes in hours
CHUNKS_DOWNLOAD = 215 # number of leadtimes to download at once
MAXLEADTIME = 215*DELTA # maximum leadtime in hours of 215 days
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')

# Check if the region provided is in the predefined list
REGIONS = ['Panaro', 'Timis', 'Lagen', 'Aragon', 'Reno']
if region not in REGIONS:
    raise ValueError(f"Region '{region}' is not a valid region. Choose from {REGIONS}")

# loop over the dates
for date in date_range:

    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')
    LAST = NENS - 1
    WRITEDIR = os.path.join(TGTDIR, region)
    os.makedirs(WRITEDIR, exist_ok=True)
    target_file = f"{WRITEDIR}/EFAS{VERSION}_reforecast_{region}_{year}{month}{day}_{KIND}_{LAST}.nc"
    if os.path.exists(target_file):
        print(f"File {target_file} already exists, skipping download")
        continue

    # full leadtime
    logging.warning(f"Downloading EFAS{VERSION} reforecast {KIND} for {year}-{month}-{day} in {region}")
    lat, lon = fixed_region(name=region, delta=BOXSIZE)
    logging.warning("Region %s has boundaries -> lon %s and lat -> %s", region, lon, lat)
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
        outdir = zip_file.replace('.zip', '')
        netcdf_file = glob.glob(f"{outdir}/data*.nc")[0]
        try:
            logging.warning("NetCDF unzipped already found as %s...", netcdf_file)
            check = xr.open_dataset(netcdf_file)
        except (ValueError, OSError):
            logging.warning(f"Unzipping the file {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(outdir)
            
        logging.warning("Ensemble files not processable by CDO, splitting with Xarray")
        dataset = xr.open_dataset(netcdf_file)
        if MASK:
            dataset = mask_efas(dataset, REGIONS)
        for ensemble in range(NENS):
            str_ensemble = str(ensemble).zfill(2)
            target_file = f"{TMPDIR}/EFAS{VERSION}_reforecast_{region}_{year}{month}{day}_{KIND}_{chunk[0]}_{str_ensemble}.nc"
            logging.warning('Checking status of %s ...', target_file)
            DISSEMBLE = True
            if os.path.exists(target_file):
                try:
                    check = xr.open_dataset(target_file)
                    if not len(check.forecast_period) == 0:
                        DISSEMBLE=False
                except ValueError:
                    pass

            if DISSEMBLE:
                logging.warning("Ensemble %s saving from xarray...", ensemble)
                ensname = f"{outdir}/mars_data_ens{ensemble}_{chunk[0]}.nc"
                dataset_ens = dataset.sel(number=ensemble).to_netcdf(ensname)
                delay = pd.Timedelta(int(chunk[0])/DELTA, unit="d")
                reference_time = pd.Timestamp(f'{year}-{month}-{day}') + pd.Timedelta(delay, unit="d")
                logging.warning("Ensemble %s time axis conversion...", ensemble)
                if os.path.exists(target_file):
                    os.remove(target_file)
                cdo.settaxis(f"{reference_time.strftime('%Y-%m-%d')},00:00:00,{DELTA}hours",
                            input = ensname,
                            output = target_file,
                            options = '-f nc4 -z zip')
                if CLEAN:
                    os.remove(ensname)
            else:
                logging.warning("Ensemble %s saving already found %s", ensemble, target_file)
            

        if CLEAN:
            os.remove(netcdf_file)
            os.rmdir(outdir)
            os.remove(zip_file)
        
    # merge the files
    for ens in range(NENS):
        logging.warning("Merging multiple for chunks for ensemble member %s", ens)
        str_ensemble = str(ens).zfill(2)
        files = f"{TMPDIR}/EFAS{VERSION}_reforecast_{region}_{year}{month}{day}_{KIND}_*_{str_ensemble}.nc"
        final_file = f"{WRITEDIR}/EFAS{VERSION}_reforecast_{region}_{year}{month}{day}_{KIND}_{str_ensemble}.nc"
        logging.warning("%s to %s", files, final_file)
        if len(files) > 1:
            cdo.merge(input = files, output = final_file,
                        options = '-f nc4')
            if CLEAN:
                for file in glob.glob(files):
                    os.remove(file)
        else:
            logging.warning("Only one file, moving the file...")
            os.rename(files[0], final_file)








