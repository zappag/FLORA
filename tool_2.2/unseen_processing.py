import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import dask
import cartopy.crs as ccrs
import pandas as pd
from datetime import timedelta



# target region selection
#########################
reg='Turia'##############
#########################


# define coordinates of the grid point nearest the basin's hydrological station (GRDC/ARPA)

if reg == 'Panaro_spil':
           plat,plon,plon_360=44.54111300000298, 11.024999986972926,11.024999986972926 #Spilamberto coords
if reg == 'Panaro_bom':
           plat,plon,plon_360=44.72445000000296, 11.041666653786793,11.041666653786793 # Bomporto coords             
if reg == 'Timis':
           plat,plon,plon_360=45.64113500000286, 21.175000076617426,plon-360 # GRDC @ sag
if reg == 'Aragon':
           plat,plon,plon_360 = 42.34535,358.352339,plon-360  # GRDC @ caparroso       
if reg == 'Lagen':
           data=data.where(data.lat>61.0,drop=True) 
           plat,plon,plon_360=61.34144900000118, 10.274999980348907,plon-360 # GRDC @ losna
if reg == 'Turia':
           plat,plon=39.51777,359.49594
           plon_360=plon-360 #GRDC @ la presa


# 1. Load surrogates on a dictionary

if reg=='Panaro_bom' or reg=='Panaro_spil':
    reg_broader='Panaro'
    path = f'/work_big/users/clima/davini/EFAS5/surrogate-v3/trimestral/{reg_broader}'
    
else:
    path = f'/work_big/users/clima/davini/EFAS5/surrogate-v3/trimestral/{reg}'
    


def load_surrogates(base_path):
    """
    
    Load surrogate series into a dictionary as ensembles of 25 members divided by start month 
    (April, May, Jun, Jul)

    """

    result = {}
    
    #subdirrectories containing init. month
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(base_path, subdir)
        files = [f for f in os.listdir(subdir_path) if f.endswith('.nc')]
        
        # to ensure consistent order
        files.sort()
        
        # load all files in subdir.
        datasets = []
        for file in files:
            file_path = os.path.join(subdir_path, file)
            ds = xr.open_dataset(file_path)
            datasets.append(ds)
        
        # combine all files in the subdir along a dummy time axis (renamed to 'member' below)
        combined_ds = xr.concat(datasets, dim='forecast_reference_time')
        
        # store in a dictionary
        result[subdir] = combined_ds
    
    return result


# 2. preprocessing of surrogates: select grid point closest to hydro.station, resample to annual maxima, rename dimensions


def preprocess_surrogates(surr_dict, plat, plon_360):
    """
    Preprocess surrogates by selecting the grid point closest to the available hydro station,
    resampling to annual maxima, and renaming dimensions.
    
    Input:
    - surr_dict: Dictionary of surrogates.
    - plat, plon_360: coordinates of the hydro station (longitude defined in east-west format).
    
    returns:
    - surr_dict_point: Surrogates at a single grid point closest to hydro station.
    - surr_dict_point_amax: annual maxima of surrogates at a single grid point.
    - surr_dict_amax: Annual maxima of full 2D surrogates (for every latitude and longitude).
    """
    
    # daily single grid point
    surr_dict_point = {k: ds.sel(lat=plat, lon=plon_360, method='nearest').rename({'forecast_reference_time': 'member', 'forecast_period': 'time'}) for k, ds in surr_dict.items()}
    
    # optional: cut to end of 2023
    surr_dict_point = {k: ds.where(ds.time.dt.year < 2024) for k, ds in surr_dict_point.items()}
    
    # annual maxima single grid point
    surr_dict_point_amax = {k: ds.resample(time='1Y').max() for k, ds in surr_dict_point.items()}
    
    # annual maxima full 2D data
    surr_dict_amax = {k: ds.resample(forecast_period='1Y').max().rename({'forecast_reference_time': 'member', 'forecast_period': 'time'}) for k, ds in surr_dict.items()}
    
    return surr_dict_point, surr_dict_point_amax, surr_dict_amax


# 3. select the largest extremes. Example: selecting the top 25 values from streamflow values at selected grid point.

def select_top_values(surr_dict_point, surr_dict_point_amax):
    """
    creates pooled datasets for daily and annual maxima values and selects the top 25 values.
    
    Input:
    - surr_dict_point: surrogates at a single grid point
    - surr_dict_point_amax: annual maxima of surrogates at a single grid point
    
    returns:
    - selected_values_daily: largest, 'most extreme' 25 daily values at selected grid point (considering all daily values)
    - selected_values_amax: largest, 'most extreme' 25 annual maxima values
    """
    
    # create pooled datasets (all events pooled together)
    pooled_daily = np.concatenate([ds.to_array().values.flatten() for ds in surr_dict_point.values()])
    pooled_amax = np.concatenate([ds.to_array().values.flatten() for ds in surr_dict_point_amax.values()])
    
    # select top 25 values in descending order
    non_nan_array_d = pooled_daily[~np.isnan(pooled_daily)]
    selected_values_daily = np.sort(non_nan_array_d)[-25:][::-1]
    
    non_nan_array_a = pooled_amax[~np.isnan(pooled_amax)]
    selected_values_amax = np.sort(non_nan_array_a)[-25:][::-1]
    
    return selected_values_daily, selected_values_amax


def compare_selected_values(selected_values_daily, selected_values_amax):
    """
    Check which events do not overlap between daily and amax selection.
    For example, events that are contiguous across two or three days will be selected in the daily pool of top extremes, 
    but not in the annual maxima top extremes, by definition.

    """
    
    # Check which values don't overlap between daily and annual maxima selected events
    for val in selected_values_daily:
        if val not in selected_values_amax:
            print(f'{val} in selected daily, not in amax')
    print('-----------------------')
    for val in selected_values_amax:
        if val not in selected_values_daily:
            print(f'{val} in selected amax, not in daily')


# 4. trace back selected values to original surrogates, the member of the ensemble, and the timestep of that event


def find_matching_surrogate(surr_dict, selected_values):
    """
    
    Traces back the value of streamflow in selected extremes to the surrogate series that generates that value.
    The surrogate start month, ensemble member and timestep of extreme event are extracted and matched to 
    the selected extremes.
    
    """


    matching_results = {}
    # for every surrogate init. month 
    for month, dataset in surr_dict.items():
        for var_name in dataset.data_vars:
            var_data = dataset[var_name].values # streamflow values
            time_axis = dataset['time'].values
            member_axis = dataset['member'].values
            
            for value in selected_values:
                if np.any(var_data == value):# check if a match exists 
                    matching_indices = np.where(var_data == value)
                    
                    if value not in matching_results:
                        matching_results[value] = []
                    
                    for time_idx, member_idx in zip(*matching_indices):
                        # move the time one day backward due to efas convention (refers to streamflow in the last 24hours)
                        adjusted_time = time_axis[time_idx] - np.timedelta64(1, 'D')
                        
                        matching_results[value].append({
                            "month": month,
                            "variable": var_name,
                            "time": adjusted_time,
                            "member": member_axis[member_idx]
                        })
    
    return matching_results




# 5. save output to .csv by creating a Pandas dataframe with 
# - starting month of surrogate
# - ensemble member
# - daily timestep of selected extreme event
# - value of Q at selected grid point, at the time of selected extreme
           

def extract_surrogate_info(matching_results):
    """
    Extracts surrogate information from matching results into lists.
    
    returns

    - value_list: List of values (Q).
    - mem_list: List of ensemble members.
    - init_list: List of initialization months (start month of surrogate).
    - timestep_list: List of time steps of selected events, daily.
    """
    
    value_list = []
    mem_list = []
    init_list = []
    timestep_list = []
    
    for value, matches in matching_results.items():
        for match in matches:
            value_list.append(value)
            mem_list.append(match['member'])
            init_list.append(match['month'])
            timestep_list.append(match['time'])
    
    return value_list, mem_list, init_list, timestep_list


def create_dataframes_and_save(matching_results_daily, matching_results_amax, reg):
    """
    Creates dataframes from matching results and saves them to .csv
    
    Input:
    - matching_results_daily: matching results for daily values
    - matching_results_amax: matching results for annual maxima
    - reg: selected region (for file naming)
    """
    
    # extract surrogate information
    dvalue, dmem, dinit, dtimestep = extract_surrogate_info(matching_results_daily)
    avalue, amem, ainit, atimestep = extract_surrogate_info(matching_results_amax)
    
    # create a dataframe
    top_25_daily = pd.DataFrame({
        'value': dvalue,
        'surrogate_type': dinit,
        'surrogate_mem': dmem,
        'time_of_event': dtimestep
    })
    
    top_25_amax = pd.DataFrame({
        'value': avalue,
        'surrogate_type': ainit,
        'surrogate_mem': amem,
        'time_of_event': atimestep
    })
    
    # save to .csv
    top_25_amax.to_csv(f'/work/users/clima/bianco/top_25_events_from_amax_{reg}.csv')
    top_25_daily.to_csv(f'/work/users/clima/bianco/top_25_events_from_daily_{reg}.csv')


###############################à


# example usage

reg=reg 

#1 - loading
base_path = path
surr_dict = load_surrogates(base_path)

#2 - processing
surr_dict_point, surr_dict_point_amax, surr_dict_amax = preprocess_surrogates(surr_dict, plat, plon_360)

#3 - selection
selected_values_daily, selected_values_amax = select_top_values(surr_dict_point, surr_dict_point_amax)
compare_selected_values(selected_values_daily, selected_values_amax)

#4 - tracing
matching_results_daily = find_matching_surrogate(surr_dict_point, selected_values_daily)
matching_results_amax = find_matching_surrogate(surr_dict_point, selected_values_amax)

#5 - saving
create_dataframes_and_save(matching_results_daily, matching_results_amax, reg)


