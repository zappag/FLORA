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
source = '/work_big/users/davini/EFAS/seasonal-v4'
target = '/work_big/users/davini/EFAS/surrogate-v1'
lead_months = 7
os.makedirs(target, exist_ok=True)

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.warning("Launching the EFAS/SEAS seasonal downloader...")

def parse_args():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Parse parameters for surrogate file generation")

    # Required positional argument for region
    parser.add_argument("region", type=str, help="Region name")

    # Optional arguments for ensemble and surrogate kind
    parser.add_argument("-e", "--ensemble", type=str, default="22",
                        help="Ensemble name or number (default: '22')")
    parser.add_argument("-m", "--mode", type=str, choices=['monthly', 'trimestral', 'quadrimestral'],
                        default ="monthly",
                        help="Surrogate kind mode: choose from 'monthly', 'trimestral', or 'quadrimestral'")

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
    ensemble_name = str(args.ensemble).zfill(2)

    logging.warning(f'Running {region} for {ensemble_name} in {mode} mode')

    # get info on the different surrogate options
    discard_month, variants, pd_delta = get_surrogate_details(surrogate_kind=mode, lead_months=lead_months)

    logging.warning(f'Discarding {discard_month} month for {len(variants)} variants with delta of {pd_delta}')

    # real offset
    offset_list = [pd.DateOffset(months=i) for i in variants]

    for offset in offset_list:
        date_list = pd.date_range(start=start_date, end=end_date, freq=pd_delta).tolist()
        date_list = [date + offset for date in date_list]
        offset_string = (date_list[0] + pd.DateOffset(months=discard_month)).strftime('%b')
        output_filename = target_filename(target, region, mode, ensemble_name, offset_string)
        if os.path.exists(output_filename):
            logging.warning(f"File {output_filename} is already there, skipping!")
            continue

        logging.warning(f"Offset: {offset}")
        surrogate = []
        for date in date_list:
            logging.warning(f"Date: {date}")
            matched_files = find_matching_files(date, region, ensemble_name, source)
            logging.warning(f"Matching files for {date}: {matched_files}")
            selected_data = load_and_filter_file(filepath=matched_files[0],
                                            discard_months=discard_month, lead_months=lead_months)
            logging.warning(f"Shape: {selected_data['dis24'].shape}")
            mindate =  pd.to_datetime(selected_data['forecast_period'].min().item()).strftime('%Y-%m-%d')
            maxdate =  pd.to_datetime(selected_data['forecast_period'].max().item()).strftime('%Y-%m-%d')
            logging.warning(f"Startdate: {mindate}, Enddate: {maxdate}")
            if selected_data is not None:
                surrogate.append(selected_data)
        pack = xr.concat(surrogate, dim='forecast_period')
        
        logging.warning(f"Saving surrogate timeseries to {output_filename}...")
        offset_string = (date_list[0] + pd.DateOffset(months=discard_month)).strftime('%b')
        save_to_netcdf(dataset=pack, filename=output_filename)
        #pack.to_netcdf(f'{target}/surrogate_test_{region}_{offset.kwds['months']}_{surrogate_kind}.nc')
        time_diffs = pack.forecast_period.diff(dim='forecast_period')
        check = (time_diffs == time_diffs[0]).all().item()
        if not check:
            raise ValueError("Not contininous temporal axis")
        logging.warning(f'Tstep is continuous {check}')
