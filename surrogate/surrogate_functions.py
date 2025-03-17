#!/usr/bin/env python3

import glob
import os

import xarray as xr
import pandas as pd


def load_and_filter_file(filepath, discard_months, lead_months):
    """
    Load a single NetCDF file, discard initial months, and select lead months.

    Args:
        filepath (str): Path to the NetCDF file.
        discard_months (int): Number of initial months to discard from the dataset.
        lead_months (int): Number of lead months to select from the dataset after discarding initial months.
    Returns:
        xarray.Dataset or None: The filtered dataset containing the selected lead months, or None if no data is selected.
    """

    ds = xr.open_dataset(filepath)

    # define a real time axis in numpy
    if isinstance(ds['forecast_period'].values[0], float):
        ds['forecast_period'] = pd.to_datetime(ds['forecast_period'].astype(int).astype(str), format="%Y%m%d")

    #get offset from data
    time_diff = ds['forecast_period'].diff(dim='forecast_period')[0].item()
    time_delta = pd.Timedelta(time_diff)
    date_offset = pd.tseries.frequencies.to_offset(time_delta)

    # drop reference time
    if 'forecast_reference_time' in ds:
        ds = ds.drop_vars('forecast_reference_time')

    # removing one timestamp because of the missing initial timestamp
    start_date = ds['forecast_period'].min().values + pd.DateOffset(months=discard_months) - date_offset
    # Define the ending point based on lead months
    end_date = start_date + pd.DateOffset(months=lead_months - discard_months) - date_offset
    selected_data = []
    selection = ds.sel(forecast_period=slice(start_date, end_date))

    if selection.forecast_period.size > 0:
        selected_data.append(selection)

    if selected_data:
        return xr.concat(selected_data, dim='forecast_period')
    
    return None
    
def find_matching_files(date, mode, region, variable, ensemble_name, path):
    """
    Generate file path pattern and find matching files for a given date.

    Args:
        date (datetime): The date for which to find matching files.
        mode (str): The mode of the forecast (e.g., 'SEAS5', 'EFAS5').
        region (str): The region for which the forecast is generated.
        variable (str): The variable of interest (e.g., '2m_temperature', 'total_precipitation').
        ensemble_name (str): The name of the ensemble.
        path (str): The base path where the files are stored.

    Returns:
        list: A list of file paths that match the generated pattern.

    Raises:
        FileNotFoundError: If no files matching the pattern are found.
    """
    """Generate file path pattern and find matching files for a given date."""
    year = date.strftime('%Y')
    yearmonth = date.strftime('%Y%m')
    modenodigit = ''.join([i for i in mode if not i.isdigit()])
    file_pattern = os.path.join(path, region, year,
                                f"{modenodigit}5_reforecast_{region}_{variable}_{yearmonth}01_{ensemble_name}.nc")
    print(file_pattern)
    matched_files = glob.glob(file_pattern)
    if not matched_files:
        raise FileNotFoundError(f"No files found for {file_pattern}")
    return matched_files

def target_filename(path, region, mode, variable, surrogate_kind, ensemble, offset_string):
    """
    Create a target filename structure based on the provided parameters.

    Parameters:
    path (str): The base directory path where the file will be saved.
    region (str): The region identifier for the filename.
    mode (str): The mode identifier, which may contain digits.
    variable (str): The variable name to be included in the filename.
    surrogate_kind (str): The type of surrogate to be included in the filename.
    ensemble (str): The ensemble identifier for the filename.
    offset_string (str): The offset string to be included in the directory and filename.

    Returns:
    str: The complete path to the target file with the constructed filename.
    """
    """Create target filename structure"""

    final_directory = os.path.join(path, surrogate_kind, region, offset_string)
    os.makedirs(final_directory, exist_ok=True)
    modenodigit = ''.join([i for i in mode if not i.isdigit()])
    filename = os.path.join(final_directory,
                            f"{modenodigit}5_surrogate_{region}_{variable}_{surrogate_kind}_{offset_string}_{ensemble}.nc")

    return filename


def save_to_netcdf(dataset, filename):
    """
    Save a dataset to a NetCDF file with compression.

    Parameters:
    dataset (xarray.Dataset): The dataset to be saved.
    filename (str): The path and name of the file to save the dataset to.

    Returns:
    None

    Notes:
    - The function applies zlib compression with a compression level of 1 to all variables in the dataset.
    - The filename should include the .nc extension to indicate a NetCDF file.
    """
    """Save NetCDF file"""
    # Extract the first timestep and get its month abbreviation
    #first_timestamp = dataset['forecast_period'].values[0]  # Assumes 'time' coordinate exists
    #first_month_abbr = pd.to_datetime(first_timestamp).strftime('%b')  # E.g., 'Jan' for January


    # Set compression options for variables
    compression = {"zlib": True, "complevel": 1}
    encoding = {var: compression for var in dataset.data_vars}  # Apply to all variables in dataset

    # Save to NetCDF with compression and structured filename
    dataset.to_netcdf(filename, encoding=encoding)
    print(f"Data saved to {filename}")

def get_surrogate_details(surrogate_kind, lead_months=7):
    """
    Define discard_month, variants, and pd_delta based on surrogate_kind.

    Parameters:
    surrogate_kind (str): The type of surrogate, can be 'monthly', 'trimestral', or 'quadrimestral'.
    lead_months (int, optional): The number of lead months. Default is 7.

    Returns:
    tuple: A tuple containing:
        - discard_month (int): The number of months to discard based on the surrogate kind.
        - variants (list): A list of integers representing the variants based on discard_month.
        - pd_delta (str): A string representing the period delta in months.
    
    Raises:
    ValueError: If an invalid surrogate_kind is provided.
    """

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
