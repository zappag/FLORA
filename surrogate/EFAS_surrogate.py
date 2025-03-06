#!/usr/bin/env python3

"""Create surrogate timeseries """

import os
import sys
import argparse
import logging
import xarray as xr
import pandas as pd
from surrogate_functions import load_and_filter_file, find_matching_files
from surrogate_functions import get_surrogate_details, save_to_netcdf, target_filename

# Define parameters
start_date = '1999-01-01'
end_date = '2023-12-31'
lead_months = 7

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.warning("Launching the EFAS/SEAS seasonal downloader...")

def parse_args():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Parse parameters for surrogate file generation")

    # Required positional argument for region
    parser.add_argument("mode", type=str, help="EFAS5/SEAS5 surrogate series creator")

    # Optional arguments for ensemble and surrogate kind
    parser.add_argument("-e", "--ensemble", type=str,
                        help="Ensemble name or number", required=True)
    parser.add_argument("-r", "--region", type=str,
                        help="Ensemble name or number (default: '22')")
    parser.add_argument("-s", "--surrogate", type=str, choices=['monthly', 'trimestral', 'quadrimestral'],
                        default ="monthly",
                        help="Surrogate kind mode: choose from 'monthly', 'trimestral', or 'quadrimestral'")
    parser.add_argument("-v", "--variable", type=str, help="variable to be surrogated",
                        default='dis24')

    # Parse the arguments
    args = parser.parse_args()

    # Check if region is provided
    if not args.region:
        print("Error: Region is a required argument.")
        sys.exit(1)

    return args


if __name__ == "__main__":
    args = parse_args()

    mode = args.mode
    region = args.region
    surrogate_kind = args.surrogate
    variable = args.variable
    ensemble_name = str(args.ensemble).zfill(2)

    ecmwf_conversion = {
        'dis24': 'river_discharge_in_the_last_24_hours',
        'msl': 'mean_sea_level_pressure',
        'z': 'geopotential'
    }
    matchvariable = ecmwf_conversion.get(variable)
    if matchvariable is None:
        raise KeyError(f'Cannot find ECMWF matching for {variable}, please edit the table')

    source = f'/work_big/users/clima/davini/{mode}/seasonal-v5'
    target = f'/work_big/users/clima/davini/{mode}/surrogate-v3'
    os.makedirs(target, exist_ok=True)

    logging.warning(f'Running for {variable} on {region} for {ensemble_name} in {surrogate_kind} mode')

    # get info on the different surrogate options
    discard_month, variants, pd_delta = get_surrogate_details(surrogate_kind=surrogate_kind, lead_months=lead_months)

    logging.warning(f'Discarding {discard_month} month for {len(variants)} variants with delta of {pd_delta}')

    # real offset
    offset_list = [pd.DateOffset(months=i) for i in variants]

    for offset in offset_list:
        date_list = pd.date_range(start=start_date, end=end_date, freq=pd_delta).tolist()
        date_list = [date + offset for date in date_list]
        offset_string = (date_list[0] + pd.DateOffset(months=discard_month)).strftime('%b')
        output_filename = target_filename(path=target, region=region, mode=mode,
                                          variable=variable, surrogate_kind=surrogate_kind,
                                          ensemble=ensemble_name, offset_string=offset_string)
        if os.path.exists(output_filename):
            logging.warning(f"File {output_filename} is already there, skipping!")
            continue

        logging.warning(f"Offset: {offset}")
        surrogate = []
        default_lon = None
        default_lat = None
        for date in date_list:
            logging.warning(f"Date: {date}")
            matched_files = find_matching_files(date=date, mode=mode, region=region, variable=matchvariable,
                                                ensemble_name=ensemble_name, path=source)
            logging.warning(f"Matching files for {date}: {matched_files}")
            selected_data = load_and_filter_file(filepath=matched_files[0],
                                            discard_months=discard_month, lead_months=lead_months)
            # store values for specific longitude and latitude
            if default_lon is None:
                default_lon = selected_data.longitude.values
            if default_lat is None:
                default_lat = selected_data.latitude.values
        
            if selected_data is not None:
                logging.info(f"Shape: {selected_data[variable].shape}")
                mindate =  pd.to_datetime(selected_data['forecast_period'].min().item()).strftime('%Y-%m-%d %H:%M')
                maxdate =  pd.to_datetime(selected_data['forecast_period'].max().item()).strftime('%Y-%m-%d %H:%M')
                logging.info(f"Startdate: {mindate}, Enddate: {maxdate}")
                #print(selected_data.latitude.values[0:4])
                #selected_data = selected_data.assign_coords(longitude=np.round(selected_data.longitude, 6), latitude=np.round(selected_data.latitude, 6))
                if default_lon[0] != selected_data.longitude.values[0] or default_lat[0] != selected_data.latitude.values[0]:
                    logging.warning("Not the same longitude and latitude, forcing assignement!")
                    selected_data = selected_data.assign_coords(longitude=default_lon, latitude=default_lat)
                surrogate.append(selected_data)
            else:
                logging.error(f"No data for {date}")
        pack = xr.concat(surrogate, dim='forecast_period')
        pack = pack.rename({"longitude": "lon"})
        pack = pack.rename({"latitude": "lat"})

        
        logging.warning(f"Saving surrogate timeseries to {output_filename}...")
        offset_string = (date_list[0] + pd.DateOffset(months=discard_month)).strftime('%b')
        save_to_netcdf(dataset=pack, filename=output_filename)
        #pack.to_netcdf(f'{target}/surrogate_test_{region}_{offset.kwds['months']}_{surrogate_kind}.nc')
        time_diffs = pack.forecast_period.diff(dim='forecast_period')
        check = (time_diffs == time_diffs[0]).all().item()
        if not check:
            raise ValueError("Not contininous temporal axis")
        logging.warning(f'Tstep is continuous {check}')
