#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tool to download SEAS5 reforecast from MARS"""

import subprocess
import logging
import os
from jinja2 import Environment, FileSystemLoader
import xarray as xr
from cdo import *
cdo = Cdo()

MARSPATH = '/home/davini/opt/bin/mars'
TGTDIR = "/work_big/users/davini/SEAS5/seasonal-v2"
TMPDIR = "/work_big/users/davini/SEAS5/tmp"
CLEAN = False # remove the temporary files

# create the target directory if it does not exist
os.makedirs(TGTDIR, exist_ok=True)
os.makedirs(TMPDIR, exist_ok=True)

# create biweekly loop with pandas
# START_DATE = "1999-01-01"
# END_DATE = "1999-02-01"
# NENS = 25 # number of ensemble members
# DELTA = 6 # delta between the leadtimes in hours
# CHUNKS_DOWNLOAD = 5160 # number of leadtimes to download at once
# MAXLEADTIME = 5160 # maximum leadtime in hours of 215 days
# date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
# variables = ['mean_sea_level_pressure', 'total_precipitation']

dates = ['20000801']
MAXLEADTIME = 5160
numbers = range(25)
parameters = ['228.128']

steps = list(range(0, MAXLEADTIME+1, 6))
steps_string = '/'.join(map(str, steps))

for startdate in dates:
    for number in numbers:
        for parameter in parameters:

            # Define the dictionary with values
            filename =  os.path.join(TMPDIR, f"SEAS5_{startdate}_{parameter}_ens{number}.grib")
            request = f"SEAS5_{startdate}_{parameter}_ens{number}.req"
            data = {
                "steps": steps_string,
                "param": parameter,
                "number": number,
                "startdate": startdate,
                "output": filename
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

                logging.warning("Template processed and saved as 'mars.req'")

                # Run the command 'mars request.req' using subprocess
                try:
                    logging.warning('Running request...')
                    result = subprocess.run([MARSPATH, request], check=True,
                                            text=True, capture_output=True)
                    print(result.stdout)
                    if CLEAN:
                        os.remove(request)
                except subprocess.CalledProcessError as e:
                    logging.error("An error occurred while executing the command:")
                    logging.error(e.stderr)

            str_number = str(number).zfill(2)
            filetmp2=os.path.join(TMPDIR, f"SEAS5_{startdate}_{parameter}_{str_number}_tmp2.nc")
            filetmp3=os.path.join(TMPDIR, f"SEAS5_{startdate}_{parameter}_{str_number}_tmp3.nc")
            filetgt=os.path.join(TGTDIR, f"SEAS5_{startdate}_{parameter}_{str_number}.nc")
            if not os.path.exists(filetmp2):
                logging.warning("Converting to regular netcdf grid...")
                cdo.setgridtype("regular", input=filename, output=filetmp2, options='-f nc')
            if CLEAN:
                os.remove(filename)

            if parameter == '228.128':
                logging.warning('Need to decumulate param %s ...', parameter)
                xfield = xr.open_dataset(filetmp2)['var228']
                zeros = xr.zeros_like(xfield.isel(time=0))
                deltas = xr.concat([zeros, xfield.diff(dim='time')], dim='time').transpose('time', ...)
                deltas.to_netcdf(filetmp3)
                if CLEAN:
                    os.remove(filetmp2)
            else:
                os.rename(filetmp2, filetmp3)

            logging.warning('Selecting 60W to 60E box and compressing...')
            cdo.sellonlatbox('-60,60,0,90', input=filetmp3, output=filetgt, options='-f nc4 -z zip')
            if CLEAN:
                os.remove(filetmp3)







