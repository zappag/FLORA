#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tool to download SEAS5 reforecast from MARS"""

import subprocess
import logging
import argparse
import os
import time
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import xarray as xr
import numpy as np
from cdo import *
from download_functions import fixed_region
cdo = Cdo()

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.warning("Launching the MARS seasonal downloader...")

# paths
TGTDIR = "/work_big/users/clima/davini/SEAS5/mars-v1/"
TMPDIR = "/work_big/users/clima/davini/SEAS5/tmp_mars"

# where MARS client is
MARSPATH = '/home/davini/opt/bin/mars'
MAX_RETRIES = 60 # wait 6 hours
SLEEP_TIME = 300

MAPPING = {
    'var228': 'total_precipitation'
}

# Set up command line argument parsing
def parse_args():
    """parsing the stuff for running"""
    parser = argparse.ArgumentParser(description='MARS reforecast downloader')
    parser.add_argument('-y', '--year', type=int, required=True,
                        help='The year to start the forecast (e.g. 2016)')
    parser.add_argument('-r', '--region', type=str, required=True,
                        help='The region for which to download the reforecast (e.g. Euro)')
    parser.add_argument('-c', '--clean', action="store_true",
                        help='clean worthless files')
    parser.add_argument('-e', '--ensemble', type=str,
                        help='Ensemble to be downloaded (Default all 25 ensemble)')
    parser.add_argument('-m', '--month', type=int,
                        help='which month download')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    year = args.year
    region = args.region
    year1 = year + 1
    clean = args.clean #remove the temporary files
    month = args.month
    month = str(month).zfill(2)


    # create the target directory if it does not exist
    os.makedirs(TGTDIR, exist_ok=True)
    os.makedirs(TMPDIR, exist_ok=True)

    START_DATE = f"{year}-{month}-01"
    if month:
        END_DATE = f"{year}-{month}-01"
    else:
        END_DATE = f"{year1}-01-01"


    NENS = 25 # number of ensemble members
    DELTA = 6 # delta between the leadtimes in hours
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='MS', inclusive='left')
    MAXLEADTIME = 5160
    #MAXLEADTIME = 24
    if args.ensemble:
        numbers = [args.ensemble]
    else:
        numbers = range(NENS)
    parameters = ['228.128']
    lat, lon = fixed_region(name=region)

    # creating the string for the mars request
    steps = list(range(0, MAXLEADTIME+1, DELTA))
    STEPS_STRING = '/'.join(map(str, steps))

    # loop on dates and ensembles
    for date in date_range:
        for number in numbers:
            for parameter in parameters:
                logging.warning('Processing date %s for ensemble %s and parameter %s', date, number, parameter)

                # useful variables
                varname = 'var' + parameter.split('.', maxsplit=1)[0]
                FINALDIR = os.path.join(TGTDIR, region, str(year))
                os.makedirs(FINALDIR, exist_ok=True)
                str_number = str(number).zfill(2)
                startdate = date.strftime('%Y%m%d')

                # target filename
                file_target=os.path.join(FINALDIR, f"SEAS5_reforecast_{region}_{MAPPING[varname]}_{startdate}_seasonal_{str_number}.nc")
                if os.path.exists(file_target):
                    logging.warning('%s already exist, skipping!', file_target)
                    continue

                # real download
                logging.warning('Downloading date %s for ensemble %s and parameter %s', date, number, parameter)
        
                # Define the dictionary for jinja
                filename =  os.path.join(TMPDIR, f"SEAS5_{startdate}_{parameter}_ens{number}.grib")
                request = f"SEAS5_{startdate}_{parameter}_ens{number}.req"
                data = {
                    "steps": STEPS_STRING,
                    "param": parameter,
                    "number": number,
                    "startdate": startdate,
                    "output": filename,
                    "area": f'{lat[1]}/{lon[0]}/{lat[0]}/{lat[1]}'
                }

                # download the file
                if os.path.exists(filename):
                    logging.warning("File %s already exists, skipping download", filename)
                else:
                    # Load the Jinja2 template file
                    file_loader = FileSystemLoader('.')  # '.' means current directory
                    env = Environment(loader=file_loader)
                    template = env.get_template('mars.j2')

                    # Render the template with the dictionary values
                    output = template.render(data)

                    # Write the output to the new file 'request.req'
                    with open(request, 'w', encoding='utf8') as f:
                        f.write(output)

                    logging.warning("Template processed and saved as %s", request)

                    # Run the command 'mars request.req' using subproces
                    retries = 0
                    while retries < MAX_RETRIES:
                        try:
                            logging.warning("Attempt %s  of %s", retries + 1, MAX_RETRIES)
                            result = subprocess.run([MARSPATH, request], check=True,
                                                    text=True, capture_output=True)
                            print(result.stdout)
                            if clean:
                                os.remove(request)
                            break
                        except subprocess.CalledProcessError as e:
                            logging.error("An error occurred while executing the command:")
                            logging.error(e.stderr)
                            logging.warning('Waiting for % seconds', SLEEP_TIME)
                            time.sleep(SLEEP_TIME)
                            retries += 1
                        if retries >= MAX_RETRIES:
                            raise KeyError('Cannot proceed, number of maximum tentative achieved!')


                # call cdo to convert to regular grid netcdf
                file_regular=os.path.join(TMPDIR, f"SEAS5_{startdate}_{parameter}_{str_number}_regular.nc")
                if not os.path.exists(file_regular):
                    logging.warning("Converting from grib unstructured to regular netcdf grid...")
                    cdo.setgridtype("regular", input=filename, output=file_regular, options='-f nc')
                if clean:
                    os.remove(filename)
                
                logging.warning("Rearranging the data via xarray...")
                xfield = xr.open_dataset(file_regular)
                # renaming variable to be compatible with CDS syntax
                refactor = xfield.rename({'lon': 'longitude', 'lat': 'latitude', 'time': 'forecast_period'})
                refactor["longitude"] = refactor["longitude"] - 360

                # fix variable name according to CDS syntax
                refactor = refactor.rename_vars({varname: MAPPING[varname]})
                #selected = refactor.sel(latitude=slice(lat[1],lat[0]), longitude=slice(lon[0],lon[1]))[MAPPING[varname]]
                selected = refactor[MAPPING[varname]]

                if MAPPING[varname] == 'total_precipitation':
                    logging.warning('Need to decumulate param %s, i.e. %s ...', varname, MAPPING[varname])
                    zeros = xr.zeros_like(selected.isel(forecast_period=0))
                    selected = xr.concat([zeros, selected.diff(dim='forecast_period')],
                                         dim='forecast_period').transpose('forecast_period', ...)

                var_encoding = {
                    'dtype': 'float32',
                    'zlib': True,
                    'complevel': 1,
                    '_FillValue': np.nan
                }
                
                file_target=os.path.join(FINALDIR, 
                                         f"SEAS5_reforecast_{region}_{MAPPING[varname]}_{startdate}_seasonal_{str_number}.nc")
                logging.warning("Saving the data to NeetCDF to %s", file_target)
                selected.to_netcdf(file_target, encoding={selected.name: var_encoding})
                if clean:
                    os.remove(file_regular)







