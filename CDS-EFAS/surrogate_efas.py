#!/usr/bin/env python3

import glob
import os

import xarray as xr
import pandas as pd


def load_and_filter_file(filepath, discard_months, lead_months):
    """Load a single NetCDF file, discard initial months, and select lead months."""
    ds = xr.open_dataset(filepath)

    # define a real time axis in numpy
    ds['forecast_period'] = pd.to_datetime(ds['forecast_period'].astype(int).astype(str), format="%Y%m%d")

    # drop reference time
    ds = ds.drop_vars('forecast_reference_time')

    # removing one day because of the missing initial timestamp
    start_date = ds['forecast_period'].min().values + pd.DateOffset(months=discard_months) - pd.DateOffset(day=1)
    
    # Define the ending point based on lead months
    end_date = start_date + pd.DateOffset(months=lead_months-discard_months) - pd.DateOffset(days=1)

    selected_data = []

    print(start_date)
    print(end_date)

    selection = ds.sel(forecast_period=slice(start_date, end_date))

    if selection.forecast_period.size > 0:
        selected_data.append(selection)

    if selected_data:
        return xr.concat(selected_data, dim='forecast_period')
    
    return None
    
def find_matching_files(date, region, ensemble_name, path):
    """Generate file path pattern and find matching files for a given date."""
    year = date.strftime('%Y')
    yearmonth = date.strftime('%Y%m')
    file_pattern = os.path.join(path, region, year, f"EFAS5_reforecast_{region}_{yearmonth}01_seasonal_{ensemble_name}.nc")
    matched_files = glob.glob(file_pattern)
    return matched_files