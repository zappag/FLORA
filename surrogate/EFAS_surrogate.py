#!/usr/bin/env python3

"""Create surrogate timeseries """

import os
import sys
import argparse
import xarray as xr
import pandas as pd
from surrogate_functions import load_and_filter_file, find_matching_files

# Define parameters
start_date = '1999-01-01'
end_date = '2001-12-31'
path = '/work_big/users/davini/EFAS/seasonal-v4'
target = '/work_big/users/davini/EFAS/surrogate'
os.makedirs(target, exist_ok=True)

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

    surrogate_kind = args.mode
    region = args.region
    ensemble_name = args.ensemble

    lead_months = 7
    if surrogate_kind == 'monthly':
        discard_month = 1
        variants = [0, 1, 2, 3, 4, 5]
        pd_delta = '6MS'   
    elif surrogate_kind == "trimestral":
        discard_month = 3
        variants = [0, 1, 2, 3]
        pd_delta = '4MS'
    elif surrogate_kind == "quadrimestral":
        discard_month = 4
        variants = [0, 1, 2]
        pd_delta = '3MS'
    else:
        raise ValueError(f'Wrong surrogate month selected {surrogate_kind}')

    # real script
    offset_list = [pd.DateOffset(months=i) for i in variants]

    for offset in offset_list:
        date_list = pd.date_range(start=start_date, end=end_date, freq=pd_delta).tolist()
        date_list = [date + offset for date in date_list]
        surrogate = []
        for date in date_list:
            matched_files = find_matching_files(date, region, ensemble_name, path)
            print(f"Matching files for {date}: {matched_files}")
            selected_data = load_and_filter_file(filepath=matched_files[0], 
                                            discard_months=discard_month, lead_months=lead_months)
            print(selected_data['dis24'].shape)
            if selected_data is not None:
                surrogate.append(selected_data)
        pack = xr.concat(surrogate, dim='forecast_period')
        pack.to_netcdf(f'{target}/surrogate_test_{region}_{offset.kwds['months']}_{surrogate_kind}.nc')
        time_diffs = pack.forecast_period.diff(dim='forecast_period')
        check = (time_diffs == time_diffs[0]).all().item()
        if not check:
            raise ValueError("Not contininous temporal axis")
        print(f'Tstep is continuous {check}')