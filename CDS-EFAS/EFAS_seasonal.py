#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tool to download EFAS reforecast from CDS"""

import os
import zipfile
import glob
import time
import logging
import argparse
import cdsapi
import warnings
import pandas as pd
import xarray as xr

from cdo import Cdo
from download_functions import chunks, fixed_region, mask_efas
cdo = Cdo()

warnings.filterwarnings("ignore", category=UserWarning, module='xarray')
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.warning("Launching the EFAS/SEAS seasonal downloader...")

KIND = 'seasonal'


# Set up command line argument parsing
def parse_args():
    """parsing the stuff for running"""
    parser = argparse.ArgumentParser(description='EFAS/SEAS reforecast downloader.')
    parser.add_argument("mode", type=str, help="Download mode: EFAS or SEAS5")
    parser.add_argument('--year', type=int, required=True,
                        help='The year to start the forecast (e.g. 2016)')
    parser.add_argument('--region', type=str, required=True,
                        help='The region for which to download the reforecast (e.g. Panaro)')
    parser.add_argument('-c', '--clean', action="store_true",
                        help='clean worthless files')
    parser.add_argument('-m', '--mask', action="store_true",
                        help='Mask domain (deprecated)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Extract year and region from the command line arguments
    year = args.year
    region = args.region
    mode = args.mode
    mask = args.mask # to mask everything
    clean = args.clean #remove the temporary files
    year1 = year + 1


    TGTDIR = f"/work_big/users/davini/{mode}/seasonal-v4"
    TMPDIR = f"/work_big/users/davini/{mode}/tmp_regions"
    # create the target directory if it does not exist
    os.makedirs(TGTDIR, exist_ok=True)
    os.makedirs(TMPDIR, exist_ok=True)

    if mode == "EFAS":
        DELTA = 24 # delta between the leadtimes in hours
        VERSION = 5 # version of the EFAS reforecast
        variables = [f'river_discharge_in_the_last_{DELTA}_hours']
        modename = f'{mode}{VERSION}'
        URL = 'https://ewds.climate.copernicus.eu/api'
        DATASET_NAME = 'efas-seasonal-reforecast'
        CHUNKS_DOWNLOAD = 22 # number of leadtimes to download at once


    elif mode == "SEAS5":
        DELTA = 6 # delta between the leadtimes in hours
        variables = ['mean_sea_level_pressure']
        modename = mode
        URL = 'https://cds.climate.copernicus.eu/api'
        DATASET_NAME = 'seasonal-original-single-levels'
        CHUNKS_DOWNLOAD = 5160 # number of leadtimes to download at once
    else:
        raise ValueError(f'Unknown model {mode}')

    START_DATE = f"{year}-01-01"
    END_DATE = f"{year1}-01-01"
    BOXSIZE = .0 #how  many degrees do you want to extend the boundaries
    NENS = 25 # number of ensemble members
    KIND = 'seasonal'
    MAXLEADTIME = 5160 # maximum leadtime in hours of 215 days
    

    MAX_RETRIES = 100
    WAIT_TIME = 120
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')

    # Check if the region provided is in the predefined list
    REGIONS = ['Panaro', 'Timis', 'Lagen', 'Aragon', 'Reno', 'Euro', 'Global', 'Turia']
    if region not in REGIONS:
        raise ValueError(f"Region '{region}' is not a valid region. Choose from {REGIONS}")

    # loop over the dates
    for date in date_range:

        for variable in variables:
            if mode == "SEAS5":
                namedefinition = f'{region}_{variable}'
            else:
                namedefinition = region

            year = date.strftime('%Y')
            month = date.strftime('%m')
            day = date.strftime('%d')
            LAST = NENS - 1
            WRITEDIR = os.path.join(TGTDIR, region, year)
            os.makedirs(WRITEDIR, exist_ok=True)
            target_file = f"{WRITEDIR}/{modename}_reforecast_{namedefinition}_{year}{month}{day}_{KIND}_{LAST}.nc"
            if os.path.exists(target_file):
                logging.warning(f"File {target_file} already exists, skipping download")
                continue

            # full leadtime
            logging.warning(f"Downloading {modename} reforecast {KIND} for {variable} for {year}-{month}-{day} in {region}")
            lat, lon = fixed_region(name=region, delta=BOXSIZE)
            logging.warning("Region %s has boundaries -> lon %s and lat -> %s", region, lon, lat)
            steps = [str(i) for i in range(DELTA, MAXLEADTIME, DELTA)]
            for chunk in chunks(steps, n=CHUNKS_DOWNLOAD):
                target_file = f"{TMPDIR}/{modename}_reforecast_{namedefinition}_{year}{month}{day}_{KIND}_{chunk[0]}_{LAST}.nc"
                logging.warning('Target file is %s', target_file)
                if os.path.exists(target_file):
                    logging.warning("File %s already exists, skipping download",  target_file)
                    continue

                download_file = f"{TMPDIR}/NEW_{mode}_reforecast_{namedefinition}_{year}{month}{day}_{KIND}_{chunk[0]}"

                if mode == "SEAS5":
                    download_file = download_file + '.nc'
                else:
                    download_file = download_file + '.zip'


                # first check if the zip file is corrupted
                if mode == "EFAS":
                    if os.path.exists(download_file):
                        try:
                            the_zip_file = zipfile.ZipFile(download_file)
                        except zipfile.BadZipFile:
                            logging.error('Corrupted/incomplete zip file')
                            os.remove(download_file)

                # download the file
                if os.path.exists(download_file):
                    logging.warning("File %s already exists, skipping download", download_file)
                else:
                    if mode == "EFAS":
                        request =  {
                        'data_format': 'netcdf',
                        'download_format': 'zip',
                        'system_version': [f'version_{VERSION}_0'],
                        'variable': [f'river_discharge_in_the_last_{DELTA}_hours'],
                        'model_levels': 'surface_level',
                        'hyear': year,
                        'hmonth': month,
                        'leadtime_hour': chunk,
                        'area': [lat[1], lon[0], lat[0], lon[1]]
                            }
                    elif mode == "SEAS5":
                        request =  {
                        'format': 'netcdf',
                        'originating_centre': 'ecmwf',
                        'system': '51',
                        'year': year,
                        'variable': variable,
                        'month': month,
                        'day': '01',
                        'leadtime_hour': chunk,
                        'area': [lat[1], lon[0], lat[0], lon[1]]
                            }
                    else:
                        raise ValueError(f'Unknown model {mode}')
                    
                    logging.warning("Launching the CDSAPI request...")
                    logging.warning(request)
                    c = cdsapi.Client(url=URL, timeout=600, sleep_max=10)
                    retries = 0
                    while retries < MAX_RETRIES:
                        try:
                            logging.warning("Attempt %s  of %s", retries + 1, MAX_RETRIES)
                            
                            mycall = c.retrieve(DATASET_NAME, request).download(download_file)
                            
                            logging.warning("Download successful on attempt %s",  retries + 1)
                            break
                            
                        except Exception as e:
                            retries += 1
                            # Catch and check if it's a protocol-related error
                            if 'protocol' in str(e).lower():
                                logging.error("Protocol error on attempt %d. Retrying in %d seconds...", retries, WAIT_TIME)
                                time.sleep(WAIT_TIME)  # Wait before retrying
                            elif 'bad request' in str(e).lower():
                                logging.error("Bad request error on attempt %d. Retrying in %d seconds...", retries, WAIT_TIME)
                                time.sleep(WAIT_TIME)  # Wait before retrying
                            elif 'broken' in str(e).lower():
                                logging.error("Broken connection error on attempt %d. Retrying in %d seconds...", retries, WAIT_TIME)
                                time.sleep(WAIT_TIME)

                            else:
                                # If the error is not related to the protocol, re-raise the exception
                                raise e
                    if retries >= MAX_RETRIES:
                        raise KeyError('Cannot proceed, number of maximum tentative achieved!')

                    
                # Unzip the file
                if mode == "EFAS":
                    outdir = download_file.replace('.zip', '')
                    netcdf_file = glob.glob(f"{outdir}/data*.nc")
                    logging.warning('Find file %s', netcdf_file)
                    try:
                        if netcdf_file:
                            check = xr.open_dataset(netcdf_file[0])
                            logging.warning("NetCDF unzipped already found as %s...", netcdf_file[0])
                        else:
                            raise FileNotFoundError("No NetCDF file found, proceeding to unzip.")
                    except (ValueError, OSError, IndexError, FileNotFoundError):
                        logging.warning("Unzipping the file %s ...", download_file)
                        with zipfile.ZipFile(download_file, 'r') as zip_ref:
                            zip_ref.extractall(outdir)
                        netcdf_file = glob.glob(f"{outdir}/data*.nc")[0]
                else:
                    outdir = download_file.replace('.nc', '')
                    os.makedirs(outdir, exist_ok=True)
                    netcdf_file = download_file
                    
                logging.warning("Ensemble files not processable by CDO, splitting with Xarray")
                logging.warning("Opening netcdf %s file...", netcdf_file)
                dataset = xr.open_dataset(netcdf_file)
                if mask:
                    dataset = mask_efas(dataset, REGIONS)
                for ensemble in range(NENS):
                    str_ensemble = str(ensemble).zfill(2)
                    target_file = f"{TMPDIR}/{modename}_reforecast_{namedefinition}_{year}{month}{day}_{KIND}_{chunk[0]}_{str_ensemble}.nc"
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
                        dataset_ens = dataset.sel(number=ensemble,
                                                latitude=slice(lat[1],lat[0]),
                                                longitude=slice(lon[0],lon[1]))
                        # special treatment for SEAS5 time axis
                        if mode == "SEAS5":
                            dataset_ens = dataset_ens.drop_vars('forecast_period')
                            dataset_ens = dataset_ens.rename({'valid_time': 'forecast_period'})#.set_index(forecast_period='forecast_period')
                            dataset_ens = dataset_ens.transpose("forecast_period", "forecast_reference_time", "latitude", "longitude")
                        dataset_ens.to_netcdf(ensname, unlimited_dims="forecast_period")

                        # boxname = f"{outdir}/mars_data_ens{ensemble}_{chunk[0]}_boxed.nc"
                        # logging.warning("Boxing %s ...", ensemble)
                        # if os.path.exists(boxname):
                        #     os.remove(boxname)
                        # cdo.sellonlatbox(f'{lon[0]},{lon[1]},{lat[0]},{lat[1]}',
                        #                  input = ensname,
                        #                  output = boxname,
                        #                  options = '-f nc4')
                        
                        logging.warning("Ensemble %s time axis conversion...", ensemble)
                        if mode == "EFAS":
                            delay = pd.Timedelta(int(chunk[0])/DELTA, unit="d")       
                        else:
                            delay = pd.Timedelta(int(chunk[0]), unit="h")
                        reference_time = pd.Timestamp(f'{year}-{month}-{day}') + delay
                        logging.warning("Reference time will be %s ...", reference_time)

                        if os.path.exists(target_file):
                            os.remove(target_file)
                        cdo.settaxis(f"{reference_time.strftime('%Y-%m-%d,%H:%M:%S')},{DELTA}hours",
                                    input = ensname,
                                    output = target_file,
                                    options = '-f nc4 -z zip')
                        if clean:
                            os.remove(ensname)
                            #os.remove(boxname)
                    else:
                        logging.warning("Ensemble %s saving already found %s", ensemble, target_file)
                
                if clean:
                    if os.path.exists(netcdf_file):
                        os.remove(netcdf_file)
                    if os.path.exists(outdir):
                        os.rmdir(outdir)
                    if os.path.exists(download_file):
                        os.remove(download_file)
                
            # merge the files
            for ens in range(NENS):
                logging.warning("Merging multiple for chunks for ensemble member %s", ens)
                str_ensemble = str(ens).zfill(2)
                files = f"{TMPDIR}/{modename}_reforecast_{namedefinition}_{year}{month}{day}_{KIND}_*_{str_ensemble}.nc"
                final_file = f"{WRITEDIR}/{modename}_reforecast_{namedefinition}_{year}{month}{day}_{KIND}_{str_ensemble}.nc"
                files = glob.glob(files)
                logging.warning("%s to %s", files, final_file)
                if len(files) > 1:
                    cdo.mergetime(input = files, output = final_file,
                                options = '-f nc4')
                    if clean:
                        for file in files:
                            os.remove(file)
                else:
                    logging.warning("Only one file, moving the file...")
                    os.rename(files[0], final_file)


    logging.warning('Everything completed, it is time to get a life!')







