#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tool to download EFAS reforecast from CDS"""

import os
import zipfile
import glob
import time
import logging
import argparse
import warnings
import cdsapi
import pandas as pd
import xarray as xr

from cdo import Cdo
from download_functions import chunks, fixed_region
cdo = Cdo()

warnings.filterwarnings("ignore", category=UserWarning, module='xarray')
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.warning("Launching the EFAS5/SEAS5 seasonal downloader v2...")

# from Nov-24 longitudes are been moved to 0:360
# use this flag to roll back to -180:180
ROLL_LONGITUDE = True

# Set up command line argument parsing
def parse_args():
    """parsing the stuff for running"""
    parser = argparse.ArgumentParser(description='EFAS5/SEAS5 reforecast downloader.')
    parser.add_argument("mode", type=str, help="Download mode: EFAS or SEAS5")
    parser.add_argument('-y', '--year', type=int, required=True,
                        help='The year to start the forecast (e.g. 2016)')
    parser.add_argument('-r', '--region', type=str, required=True,
                        help='The region for which to download the reforecast (e.g. Panaro)')
    parser.add_argument('-m', '--month', type=int, required=False,
                        help='The month to start the forecast download (e.g. 1)', default=None)
    parser.add_argument('-c', '--clean', action="store_true",
                        help='clean worthless files')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Extract year and region from the command line arguments
    year = args.year
    region = args.region
    month = args.month
    mode = args.mode
    clean = args.clean #remove the temporary files

    TGTDIR = f"/work_big/users/clima/davini/{mode}/seasonal-v5"
    TMPDIR = f"/work_big/users/clima/davini/{mode}/tmp_regions"
    # create the target directory if it does not exist
    os.makedirs(TGTDIR, exist_ok=True)
    os.makedirs(TMPDIR, exist_ok=True)

    if mode == "EFAS5":
        DELTA = 24 # delta between the leadtimes in hours
        VERSION = 5 # version of the EFAS reforecast
        variables = [f'river_discharge_in_the_last_{DELTA}_hours']
        URL = 'https://ewds.climate.copernicus.eu/api'
        DATASET_NAME = 'efas-seasonal-reforecast'
        CHUNKS_DOWNLOAD = 22 # number of leadtimes to download at once
    elif mode == "SEAS5":
        DELTA = 6 # delta between the leadtimes in hours: for SEAS5 this is adjusted to 12 hours for 3d data
        variables = ['geopotential', 'mean_sea_level_pressure']
        URL = 'https://cds.climate.copernicus.eu/api'
        CHUNKS_DOWNLOAD = 5160 # number of leadtimes to download at once
    else:
        raise ValueError(f'Unknown model {mode}')

    # create a loop for the entire year if month is not defined
    if month is None:
        START_DATE = f"{year}-01-01"
        END_DATE = f"{year+1}-01-01"
        date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='MS', inclusive='left')
    else:
        year1 = year if month < 12 else year + 1
        month1 = month + 1 if month < 12 else 1
        date_range = pd.date_range(start=f"{year}-{month}-01", end=f"{year1}-{month1}-01", freq='MS', inclusive='left')

    BOXSIZE = .0 #how  many degrees do you want to extend the boundaries
    NENS = 25 # number of ensemble members
    MAXLEADTIME = 5160 # maximum leadtime in hours of 215 days

    # options for safe donwload
    MAX_RETRIES = 100
    WAIT_TIME = 120

    # Check if the region provided is in the predefined list
    REGIONS = ['Panaro', 'Timis', 'Lagen', 'Aragon', 'Reno', 'Euro', 'Global', 'Turia']
    if region not in REGIONS:
        raise ValueError(f"Region '{region}' is not a valid region. Choose from {REGIONS}")

    # loop over the dates
    for date in date_range:

        for variable in variables:
            if mode == "SEAS5":
                # 3d vs 2s case for SEAS5
                if variable in ['geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind']:
                    DATASET_NAME = 'seasonal-original-pressure-levels'
                    DELTA = 12 # delta between the leadtimes in hours
                else:
                    DATASET_NAME = 'seasonal-original-single-levels'
                    DELTA = 6 # delta between the leadtimes in hours

            namedefinition = f'{region}_{variable}'

            year = date.strftime('%Y')
            month = date.strftime('%m')
            day = date.strftime('%d')
            LAST = NENS - 1
            WRITEDIR = os.path.join(TGTDIR, region, year)
            os.makedirs(WRITEDIR, exist_ok=True)
            target_file = f"{WRITEDIR}/{mode}_reforecast_{namedefinition}_{year}{month}{day}_*.nc"
            if len(glob.glob(target_file)) == NENS:
                logging.warning("All files already exist %s, skipping...", target_file)
                continue
            
            logging.warning("Downloading target file %s", target_file)

            # full leadtime
            logging.warning("Downloading %s reforecast for %s for %s-%s-%s in %s", mode, variable, year, month, day, region)
            lat, lon = fixed_region(name=region, delta=BOXSIZE)
            logging.warning("Region %s has boundaries -> lon %s and lat -> %s", region, lon, lat)
            steps = [str(i) for i in range(DELTA, MAXLEADTIME, DELTA)]
            for chunk in chunks(steps, n=CHUNKS_DOWNLOAD):
                target_file = f"{TMPDIR}/{mode}_reforecast_{namedefinition}_{year}{month}{day}_{chunk[0]}_{LAST}.nc"
                logging.warning('Target file is %s', target_file)
                if os.path.exists(target_file):
                    logging.warning("File %s already exists, skipping download",  target_file)
                    continue

                download_file = f"{TMPDIR}/{mode}_reforecast_{namedefinition}_{year}{month}{day}_{chunk[0]}"

                if mode == "SEAS5":
                    download_file = download_file + '.nc'
                else:
                    download_file = download_file + '.zip'


                # first check if the zip file is corrupted
                if mode == "EFAS5":
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
                    if mode == "EFAS5":
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
                        if DATASET_NAME == 'seasonal-original-pressure-levels':
                            request['pressure_level'] = [925, 850, 700, 500, 300]
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
                                retries = MAX_RETRIES
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
                if mode == "EFAS5":
                    outdir = download_file.replace('.zip', '')
                    netcdf_file = glob.glob(f"{outdir}/data*.nc")
                    logging.warning('Find file %s', netcdf_file)
                    try:
                        if netcdf_file:
                            netcdf_file = netcdf_file[0]
                            check = xr.open_dataset(netcdf_file)
                            logging.warning("NetCDF unzipped already found as %s...", netcdf_file)
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
                for ensemble in range(NENS):
                    str_ensemble = str(ensemble).zfill(2)
                    target_file = f"{TMPDIR}/{mode}_reforecast_{namedefinition}_{year}{month}{day}_{chunk[0]}_{str_ensemble}.nc"
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
                        if ROLL_LONGITUDE:
                            dataset["longitude"] = ((dataset["longitude"] + 180) % 360) - 180
                            dataset = dataset.sortby("longitude")
                        dataset_ens = dataset.sel(number=ensemble,
                                                latitude=slice(lat[1],lat[0]),
                                                longitude=slice(lon[0],lon[1]))
                        # special treatment for SEAS5 time axis
                        if mode == "SEAS5":
                            dataset_ens = dataset_ens.drop_vars('forecast_period')
                            dataset_ens = dataset_ens.rename({'valid_time': 'forecast_period'})#.set_index(forecast_period='forecast_period')
                            if DATASET_NAME == 'seasonal-original-pressure-levels':
                                dataset_ens = dataset_ens.transpose("forecast_period", "forecast_reference_time", "pressure_level", "latitude", "longitude").squeeze()
                            else:
                                dataset_ens = dataset_ens.transpose("forecast_period", "forecast_reference_time", "latitude", "longitude")
                        dataset_ens.to_netcdf(ensname, unlimited_dims="forecast_period")

                        logging.warning("Ensemble %s time axis conversion...", ensemble)
                        if mode == "EFAS5":
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
                    else:
                        logging.warning("Ensemble %s saving already found %s", ensemble, target_file)
                
                # clean up
                if clean:
                    for file_path in [netcdf_file, outdir, download_file]:
                        if os.path.exists(file_path):
                            if os.path.isdir(file_path):
                                os.rmdir(file_path)
                            else:
                                os.remove(file_path)
                
            # merge the files, if more than one is available
            for ens in range(NENS):
                logging.warning("Merging multiple for chunks for ensemble member %s", ens)
                str_ensemble = str(ens).zfill(2)
                files = f"{TMPDIR}/{mode}_reforecast_{namedefinition}_{year}{month}{day}_*_{str_ensemble}.nc"
                final_file = f"{WRITEDIR}/{mode}_reforecast_{namedefinition}_{year}{month}{day}_{str_ensemble}.nc"
                files = glob.glob(files)
                logging.warning("%s to %s", files, final_file)
                if len(files) > 1:
                    cdo.mergetime(input = files, output = final_file,
                                options = '-f nc4 -z zip')
                    if clean:
                        for file in files:
                            os.remove(file)
                elif len(files) == 1:
                    logging.warning("Only one file, moving the file...")
                    os.rename(files[0], final_file)
                else:
                    raise FileNotFoundError(f"No files found for {files}")
                


    logging.warning('Everything completed, it is time to get a life!')







