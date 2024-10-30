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

def save_to_netcdf(dataset, path, region, surrogate_kind, ensemble):
    # Extract the first timestep and get its month abbreviation
    first_timestamp = dataset['forecast_period'].values[0]  # Assumes 'time' coordinate exists
    first_month_abbr = pd.to_datetime(first_timestamp).strftime('%b')  # E.g., 'Jan' for January

    # Generate structured filename
    #offset_months = offset.kwds['months']
    filename = f"{path}/EFAS5_surrogate_{region}_{surrogate_kind}_{first_month_abbr}_{ensemble}.nc"

    # Set compression options for variables
    compression = {"zlib": True, "complevel": 1} 
    encoding = {var: compression for var in dataset.data_vars}  # Apply to all variables in dataset

    # Save to NetCDF with compression and structured filename
    dataset.to_netcdf(filename, encoding=encoding)
    print(f"Data saved to {filename}")

def get_surrogate_details(surrogate_kind, lead_months=7):
    # Define discard_month, variants, and pd_delta based on surrogate_kind
    if surrogate_kind == 'monthly':
        discard_month = 1
    elif surrogate_kind == 'trimestral':
        discard_month = 3
    elif surrogate_kind == 'quadrimestral':
        discard_month = 4
    else:
        raise ValueError(f'Invalid surrogate kind: {surrogate_kind}')

    # Determine variants based on discard_month
    variants = list(range(lead_months - discard_month))
    pd_delta = str(len(variants)) + 'MS'

    return discard_month, variants, pd_delta
