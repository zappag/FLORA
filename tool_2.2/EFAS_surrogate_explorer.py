#!/home/tino/miniconda3/envs/efas/bin/python
# Read surrogate tiemseries for Panaro and makes some figures
# must be run in conda activate efas
# 
# Original program by Giuseppe Zappa. Copied from FLORA on 07 Jan 2025.
# Revisideted by Tino Manzato Feb-March 2025

import sys
# Add your local "bin" dir:
# This is the new path for gihub local directory:
# before it was in program/tools_official/
BINDIR="/home/tino/documents/work/Translate/github/FLORA/tool_2.2/"


sys.path.append(BINDIR)
import region_details
import importlib
importlib.reload(region_details)
from region_details import get_region_details  # region_details.py library must be in BINDIR
print(sys.path)

import xarray as xr
import cartopy.crs as ccrs
import matplotlib
matplotlib.use('Agg')  # Using backend without GUI for multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from minisom import MiniSom
from shapely.geometry import Point
import geopandas as gpd
from scipy.stats import linregress
from scipy.stats import skew
import csv
import time
import math
import multiprocessing as mp
import subprocess    # to be used instead of os() because not compatible with multiprocessing
from pathlib import Path


# SORT OF USER CONFIGURATION
start_time = time.time()
# SORT OF USER CONFIGURATION

# trimestral surrogates have 4 months (Apr May Jun Jul) and 00-24 members
#month="Jul"
#member="1"

DATADIR="/home/tino/documents/work/Translate/data/"
EFAS_tino=DATADIR+"EFAS_historical/"   # historical EFAS input data dir

# Name Pattern of EFAS data, * reflects different years
watershed= 'panaro'   # can be any other river defined in region_details.py
tres='day'
disname='dis24'   # River discharge in the last 24 hours
window_size = 31  # was 45  # Smoothing to highlight mean seasonal cycle

# path of efas input data masked for watershed
wshed_fname=f"{EFAS_tino}/postpro/{tres}/watersheds/efas_{watershed}_masked.nc"  # from preprocessed file

# path of maximum discharge in watershed
#wshed_dismax_fname=os.path.dirname(wshed_fname)+f"/basinmax/efas_{watershed}.nc"

# directory of HydroBasins shapefile. Needed only by function read_watershed_shapefiles()... (not stricly necessary)
#shape_dir=f"{EFAS_tino}/hybas_lake_eu_lev01-12_v1c"  

# starting and last year to be analysed (EFAS historical)
#sy=1992
#ly=2023

# region details
winfo=get_region_details(watershed)
tlon=winfo['tlon']
tlat=winfo['tlat']
tlabel=winfo['tlabel']
hblevel=winfo['shape_file_level']
bbox=winfo['bounding_box'] # lonS, latW, lonN, latE
#atmbox=winfo['era_box'] 

# watershed central point
wspoint = Point(tlon, tlat)
print(f"Single point: {tlabel}", wspoint)

# temporary output EFAS timeseries at single point
#fname_gp=f"{EFAS_tino}/postpro/{tres}/single_gp/efas_{watershed}_{tlabel}_{tres}.nc"
fname_gp_dir=Path(f"{EFAS_tino}/postpro/{tres}/single_gp/")
fname_gp_dir.mkdir(parents=True, exist_ok=True)
fname_gp= f"{fname_gp_dir}efas_{watershed}_{tlabel}_{tres}.nc"

command= f"ls -lt {fname_gp}" 
print(command)
os.system(command)
# file for annual maximum discharge:
fname_gp_ymax = f"{EFAS_tino}/postpro/{tres}/single_gp/efas_{watershed}_{tlabel}_{tres}_annual_maxima.nc"

EFAS_surro=f"{DATADIR}EFAS_surrogate/surrogate-v3/trimestral/"  # watershed/Jul or Jun or May/

command= f"ls -lt {EFAS_surro}" 
print(command)
os.system(command)

# path for statistics of surrogate seasonal cycle:
## work\Translate\data\EFAS_surrogate\seasonal_cycle\panaro\day
cycle_dir = f"{DATADIR}EFAS_historical/postpro/day/watersheds/stats/"

# FLAGS
plot2d_day=True
ReadExisting= False  # True: reads any preprocessed data written to disk. False: EFAS data is read from the netcdf files
HazardDef= "gp"   # Was "wmax" in original  # gridpoint (gp), watershed max (wmax), economic (eco). gp is based on tlon,tlat. 



# path to directory with figures
figdir=f"{BINDIR}figures/{watershed}/{tres}/"
# path to directory with figures
figdir_sur=f"{BINDIR}figures/{watershed}/{tres}/surrogate/"
os.makedirs(figdir, exist_ok=True)
os.makedirs(figdir_sur, exist_ok=True)


print("--- %s seconds ---" % (time.time() - start_time))
print("done block 1")


def fit_line(x, y):
    # given one dimensional x and y vectors - return x and y for fitting a line on top of the regression
    # inspired by the numpy manual - https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html 
    x = x.to_numpy() # convert into numpy arrays
    y = y.to_numpy() # convert into numpy arrays

    A = np.vstack([x, np.ones(len(x))]).T # sent the design matrix using the intercepts
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    return m, c


def save_results_txt(output_file) :
    print(result_param)

    #output_file = "output.txt"
    # Scrittura su file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')  # Usa il tab come delimitatore per ASCII leggibile
        
        # Scrivi l'intestazione (usa le chiavi del primo dizionario interno)
        headers = list(result_param[0].keys())
        writer.writerow(headers)
        for key, row in result_param.items():
            writer.writerow([row[h] for h in headers])


def circular_rolling(da, window_size):
    # Makes a moving average of +-window_size//2 after attaching the reflected tails to the timeseries:
    #  
    '''
    # Append the start of the DataArray to the end to create a circular buffer
    circular_data = np.concatenate([da, da[:window_size]])
    # Create a new DataArray with the circular buffer
    circular_da = xr.DataArray(circular_data, dims=da.dims, coords=da.coords)
    # Apply the rolling operation
    result = circular_da.rolling({da.dims[0]: window_size}, center=True).mean()
    # Slice off the extra values at the end
    result = result[:len(da)]
    return result
    '''
    
    # Definisce metà finestra (arrotondata per eccesso se dispari)
    half_window = (window_size // 2) + (window_size % 2)
    # Estende la serie prendendo gli ultimi e primi giorni e specchiandoli
    start_pad = da.isel(dayofyear=slice(-half_window, None)).isel(dayofyear=slice(None, None, -1))
    end_pad = da.isel(dayofyear=slice(0, half_window)).isel(dayofyear=slice(None, None, -1))

    # Concatena i dati per creare un'estensione riflessiva
    extended_means = xr.concat([start_pad, da, end_pad], dim='dayofyear')

    # Applica la media mobile
    smoothed_extended = extended_means.rolling(dayofyear=window_size, center=True).mean()
    # Ritaglia per ottenere solo i giorni dell'anno originali (dal 1 al 365/366)
    smoothed_da = smoothed_extended.isel(dayofyear=slice(half_window, -half_window))

    return smoothed_da


'''
# This code will draw the watershed figure. 
# To run it you need the watershed shape files in shape_dir=f"{EFAS_tino}/hybas_lake_eu_lev01-12_v1c" 
# Read watershed shapefiles 
def read_watershed_shapefiles(shape_dir, level, wspoint):
    """
    Read watershed shapefiles and extract the shapefile entry that contains a specific point.

    Parameters:
    shape_dir (str): The directory where the shapefiles are located.
    level (str): The level of the shapefiles to read.
    wspoint (Point): The point to analyze.

    Returns:
    GeoDataFrame: A GeoDataFrame containing the shapefile entry that contains the point.
    """
    # Construct the shapefile path
    shapef_path = os.path.join(shape_dir, f"hybas_lake_eu_lev{level}_v1c.shp")

    # Read the shapefile
    shape_gdf = gpd.read_file(shapef_path)

    # Find the shapefile entries that contain the point
    contains_point = shape_gdf['geometry'].contains(wspoint)

    # Extract the shapefile entry that contains the point
    shape_basin_gdf = shape_gdf[contains_point]

    return shape_basin_gdf


def xarray_in_shapefile(xds,varname,latname,lonname,shapef,operation):
    
    # compute average precipitation (ds_emo) for grid points within shape_basin_gdf
    df = xds[varname].to_dataframe()
    lats = df.index.get_level_values(latname)
    lons = df.index.get_level_values(lonname)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(lons,lats),crs='EPSG:4326')

    # created join dataframe between data and shapefile
    gdf_joined = gpd.sjoin(gdf, shapef, predicate='within')

     # Group by time and region, and sum the values
    if operation=='mean':
        results = gdf_joined.groupby(['time', 'index_right'])[varname].mean()
    elif operation=='sum':
        results = gdf_joined.groupby(['time', 'index_right'])[varname].sum()
    elif operation=='max':
        results = gdf_joined.groupby(['time', 'index_right'])[varname].max()

    # Convert the series to a DataFrame and pivot it
    results_df=pd.DataFrame()
    results_df[varname] = results.reset_index().pivot(index='time', columns='index_right', values=varname)
    results_xds=results_df.to_xarray()

    return results_df, results_xds

# read shape file
shape_basin_gdf=read_watershed_shapefiles(shape_dir, hblevel, wspoint)
shape_basin_gdf.plot()
print("done block 2")
'''



start_time = time.time()
# Read EFAS data for single grid point, and save it to .nc for more rapid access in future executions
# It would make sense to have a function that reads the hazard: either as a single grid point, watershed max, or economic
def read_and_store_time_series(fname_pat, lat, lon, output_file, ReadExisting):
        
    """
    Read time series data from a specified grid point and store it on disk.
    
    Parameters:
    fname_pat (str or list of str): File path pattern or list of file paths.
    lat_index (int): Index of the latitude grid point.
    lon_index (int): Index of the longitude grid point.
    output_file (str): Output file name to store the time series data (default: 'time_series_data.nc').
    check_existing (bool): Whether to check if the output file already exists (default: True).
    """
    # Check if the output file already exists
    if ReadExisting and os.path.exists(output_file):
        print("File existing", ReadExisting, output_file)
        # If the output file exists, read it straight away
        time_series = xr.open_dataset(output_file)
        print("after reading existing timeserie--- %s seconds ---" % (time.time() - start_time))
    else:
        # If the output file does not exist, proceed with extracting data from the dataset
        # Open the multi-file dataset
        print("Creating not exiting file", output_file)
        dataset = xr.open_mfdataset(fname_pat, combine='by_coords')
        dataset.load()
        dataset.close()
        print("afer reading input nc--- %s seconds ---" % (time.time() - start_time))
        # Extract time series data for the specified grid point
        time_series = dataset.sel(lat=lat,lon=lon,method='nearest')
        time_series.load()
        print("after extracting gp nearest values timeserie--- %s seconds ---" % (time.time() - start_time))
        # Store the extracted time series data on disk
        time_series.to_netcdf(output_file)
        os.system(f"ls -lt {output_file}")
        print("after saving non-existing timeserie--- %s seconds ---" % (time.time() - start_time))
    
    return time_series



# read hazard based on hazard definition
print(wshed_fname, fname_gp)

if HazardDef=="gp":
    xds_hazard = read_and_store_time_series(wshed_fname, tlat, tlon, fname_gp, ReadExisting)
    #print(f"trying to read {fname_gp_ref}")
    #xds_hazard_ref = read_and_store_time_series(wshed_fname, tlat, tlon, fname_gp_ref, ReadExisting)
'''
elif HazardDef=="wmax":
    # if maximum discharge exists, read it, otherwise compute it
    if os.path.exists(wshed_dismax_fname) and ReadExisting:
        xds_hazard=xr.open_dataset(wshed_dismax_fname)
    else:
        # read EFAS data``
        xds=xr.open_mfdataset(wshed_fname,combine='by_coords')
        xds_hazard=xds.max(dim=['lat', 'lon'])

        print(f"trying to save {wshed_dismax_fname}")
        # save to file
        xds_hazard.to_netcdf(wshed_dismax_fname)'
'''
oldest= xds_hazard.time.min().values
recent= xds_hazard.time.max().values
print(f"Oldest date: {oldest}. Most recent date: {recent}")
print("end of cell--- %s seconds ---" % (time.time() - start_time))
# discharge values (convert to numpy array)
# Why this line???
#ds_hazard=xds_hazard[disname].values
#ds_hazard_ref=xds_hazard_ref[disname].values


# PLOT HISTORICAL EFAS:
# Plot timeseries adding a linear fit:

# FITTING the timeseries:
# Convert the time index to numerical values (e.g., ordinal)
time_numeric = xds_hazard.coords["time"].values.astype('datetime64[s]').astype(int)
#disname_numeric = xds_hazard[disname]
# Perform linear regression on the numerical time values and data
mask = ~np.isnan(time_numeric) & ~np.isnan(xds_hazard[disname])
print(mask)
#slope, intercept, r_value, p_value, std_err = linregress(time_numeric[mask], casalecchio_ds.streamflow[mask])
slope, intercept, r_value, p_value, std_err = linregress(time_numeric[mask], xds_hazard[disname].sel(time=mask.time))
#slope, intercept, r_value, p_value, std_err = linregress(time_numeric[mask], xds_hazard[mask])
print(f"All days:\nm={slope}, q={intercept}, R={r_value}, p-value={p_value}, StdErr={std_err}")
# Compute the fitted values using the regression parameters
fitted_values = slope * time_numeric + intercept
# converting slope from seconds to years:
slope = slope*3600*24*365.2425


# plot timeseries using xarray 
fig,axs=plt.subplots(3,1,figsize=(8,10))
time_start_plot="1992-01-01"  # or better oldest?
time_end_plot="2023-12-31"    # or better recent?
xds_hazard[disname].sel(time=slice(time_start_plot,time_end_plot)).plot(ax=axs[0])
# In this case we have always fitted_values for all xds.times 
axs[0].plot(xds_hazard.time, fitted_values, color="r", linestyle='dashed')
#axs[0].set_title(f"{HazardDef} {disname} in {watershed}")
if HazardDef == "gp" :
    gp_name = tlabel
else :
    gp_name = HazardDef
axs[0].set_title(f"{gp_name} {disname} in {watershed} daily mean")
axs[0].set_xlabel("")


# Second figure: Adding the timeseries of the annual maxima (one value per year):
year_max = xds_hazard[disname].resample(time='1YE').max(dim='time').chunk({'time': -1})  # saves only maximum RD per year, not original dates
print("Annual Maxima=", year_max.values)
year_max_mean_value = year_max.values.mean()
year_max_std_value = year_max.values.std()
year_max_StdErr_value = 2*year_max_std_value/math.sqrt(len(year_max.values))
print(f"Average value of annual maxima= {year_max_mean_value}, StD= {year_max_std_value}, 2*StdErr= {year_max_StdErr_value}")
# years = year_max.astype('datetime64[Y]').item().year  # da errore
years = year_max['time'].dt.year.values[:-1]  # .tolist()  BE CAREFUL: in EFAS historical there is a point in 2024 and hence last year must be removed!
# exact dates of each annual maxima:
date_ymax = xds_hazard[disname].resample(time='1YE').apply(lambda x: x.idxmax(dim='time'))[:-1] # By Paolo Davini!
# Remove 12h to center the maximum ad midday (not 00 of the following day):
date_ymax = date_ymax - np.timedelta64(12,'h')
date_ymax_list = date_ymax['time'].dt.date.values[:-1].tolist()
#date_ymax_abs =  xds_hazard[disname].apply(lambda x: x.idxmax(dim='time'))
# date of absolute mximum:
date_ymax_abs =  xds_hazard[disname].idxmax(dim='time').values - np.timedelta64(12,'h')
# Absolute maximum:
year_max_abs = xds_hazard[disname].max(dim='time').values  #.chunk({'time': -1})
# Year with absolute maximum:
years_abs = date_ymax_abs.astype('datetime64[Y]').astype(int) + 1970 
print(f'Maximum of all timeseries= {year_max_abs} on {date_ymax_abs}')
#date_ymax = xds_hazard[disname].resample(time='1Y').idxmax(dim='time')   #  DataArrayResample of xarray not supports .idxmax() method !
#date_ymax = xds_hazard[disname].groupby('time.year').apply(lambda group: group.time.values[group.argmax(dim='time')])  
#           xds_hazard[disname].groupby('time.year').reduce(lambda arr: arr.time[arr.argmax(dim='time')])  # reduce is not working properly
#date_ymax = xds_hazard[disname].resample(time='1Y').map(find_max_date) # save the dates of the annual maxima.
#date_ymax = []
#date_ymax = xr.DataArray(date_ymax, coords={'time': year_max.time}, dims=['time'])  # convert to DataArray
#date_ymax = xds_hazard[disname].resample(time='1Y').reduce(find_max_date)
#date_ymax = xds_hazard[disname].resample(time='1Y').map(find_max_date)
#date_ymax = year_max.groupby('time.year').map(find_max_date)

print(date_ymax.values, "\n", year_max.values[:-1])
annual_maxima = xr.Dataset(
    {
    'dis24': (["time"], year_max.values[:-1]) 
    },
    coords= {
    'time': date_ymax.values,  # take other coordinates date_ymax
    "lon": year_max.lon,       # Altre coordinate
    "lat": year_max.lat #,
    #"spatial_ref": year_max.spatial_ref
    })
# Impostiamo esplicitamente i valori mancanti per le date non incluse
annual_maxima["dis24"] = annual_maxima["dis24"].where(~np.isnan(annual_maxima["dis24"]))
#
#time_origin = "1900-01-01"  # Origine per le unità temporali
#time_units = f"days since {time_origin}"
#time_numeric = (pd.to_datetime(annual_maxima["time"].values) - pd.to_datetime(time_origin)).days
# Aggiornare la coordinata temporale con i numeri e aggiungere l'attributo `units`
#annual_maxima = annual_maxima.assign_coords(
#    time=("time", time_numeric, {"units": time_units, "calendar": "gregorian"})
#)
print(f"Annual_max_XR= {annual_maxima}")
print(f"trying to save {fname_gp_ymax}")
# save to file
annual_maxima.to_netcdf(fname_gp_ymax, unlimited_dims={'time':True}, mode='w')
#annual_maxima.to_netcdf(fname_gp_ymax)


'''
# FITTING the timeseries:
# Convert the time index to numerical values (e.g., ordinal)
#slope, intercept, r_value, p_value, std_err = linregress(time_numeric[mask], casalecchio_ds.streamflow[mask])
slope_ymax, intercept_ymax, r_value_ymax, p_value_ymax, std_err_ymax = linregress(years, year_max.values[:-1])
#slope, intercept, r_value, p_value, std_err = linregress(time_numeric[mask], xds_hazard[mask])
print(f"Only annual maxima: \nm={slope_ymax}, q={intercept_ymax}, R={r_value_ymax}, p-value={p_value_ymax}, StdErr={std_err_ymax}")
# Compute the fitted values using the regression parameters
fitted_values_ymax = slope_ymax * years + intercept_ymax
print("Fit_ymax=")
print(fitted_values_ymax, "HERE!")

# years is numeric, date_max are dates:
axs[1].bar(years, year_max.values[:-1])
#axs[1].plot(date_ymax, year_max.values[:-1])
axs[1].plot(years, fitted_values_ymax, color="r", linestyle='dashed')
# In this case we have always fitted_values for all xds.times 
#axs[1].plot(xds_hazard.time, fitted_values, color="r", linestyle='dashed')
#axs[0].set_title(f"{HazardDef} {disname} in {watershed}")
if HazardDef == "gp" :
    gp_name = tlabel
else :
    gp_name = HazardDef
axs[1].set_title(f"{gp_name} {disname} in {watershed} annual maxima")
axs[1].set_xlabel("")
axs[1].legend([f"m={round(slope_ymax, 1)} $m^{3}s^{{-1}}y^{{-1}}$, R={round(r_value_ymax, 2)}, p-value={round(p_value_ymax, 3)}", "Max-of-Year"],  prop={'size': 8}, loc='upper left')
'''

# Redoing the same using real MaxDate instead than mid-of-year bars:
# FITTING the timeseries:
# Convert the time index to numerical values (e.g., ordinal)
date_ymax_numeric = date_ymax.coords["time"].values.astype('datetime64[s]').astype(int)
#slope, intercept, r_value, p_value, std_err = linregress(time_numeric[mask], casalecchio_ds.streamflow[mask])
slope_ymax, intercept_ymax, r_value_ymax, p_value_ymax, std_err_ymax = linregress(date_ymax_numeric, year_max.values[:-1])
#slope_ymax, intercept_ymax, r_value_ymax, p_value_ymax, std_err_ymax = linregress(date_ymax_list, year_max.values[:-1])
#slope, intercept, r_value, p_value, std_err = linregress(time_numeric[mask], xds_hazard[mask])
print(f"Only annual maxima: \nm={slope_ymax}, q={intercept_ymax}, R={r_value_ymax}, p-value={p_value_ymax}, StdErr={std_err_ymax}")
# Compute the fitted values using the regression parameters
fitted_values_ymax = slope_ymax * date_ymax_numeric + intercept_ymax
# converting slope from seconds to years:
slope_ymax = slope_ymax*3600*24*365.2425
print("Fit_ymax=")
print(fitted_values_ymax, "HERE!")

# years is numeric, date_max are dates:
#axs[1].bar(years, year_max.values[:-1])
axs[1].vlines(annual_maxima.time, ymin=0, ymax=annual_maxima.dis24, linewidth=2)
#axs[1].plot(date_ymax, year_max.values[:-1])
axs[1].plot(annual_maxima.time, fitted_values_ymax, color="r", linestyle='dashed')
# In this case we have always fitted_values for all xds.times 
#axs[1].plot(xds_hazard.time, fitted_values, color="r", linestyle='dashed')
#axs[0].set_title(f"{HazardDef} {disname} in {watershed}")
if HazardDef == "gp" :
    gp_name = tlabel
else :
    gp_name = HazardDef
axs[1].set_title(f"{gp_name} {disname} in {watershed} annual maxima")
axs[1].set_xlabel("")
#axs[1].legend([f"m={round(slope_ymax, 1)} $m^{3}s^{{-1}}y^{{-1}}$, R={round(r_value_ymax, 2)}, p-value={round(p_value_ymax, 3)}", "Max-of-Year"],  prop={'size': 8}, loc='upper left')
axs[1].legend(["Max of each Year", f"m={round(slope_ymax, 1)} $m^{3}s^{{-1}}y^{{-1}}$, R={round(r_value_ymax, 2)}, p-value={round(p_value_ymax, 3)}"],  prop={'size': 8}, loc='upper left')
#handles, labels = plt.gca().get_legend_handles_labels()  # Ottieni handles e labels
#axs[1].legend(handles[::-1], labels[::-1])

# saving historical values for later:
histo_annual_maxima_times = years   # annual_maxima.time
histo_annual_maxima = annual_maxima.dis24

# Third figure select year with maximum discharge
print(f"input len= {len(xds_hazard[disname])}, {len(xds_hazard[disname])/365} years")
max_year1 = xds_hazard[disname].idxmax(dim='time').values
print("Historical maximum: ", max_year1.astype('datetime64[D]'), type(max_year1.astype('datetime64[D]')))
date_max=max_year1.astype('datetime64[D]')
value_max=xds_hazard[disname].sel(time=date_max).values   # .astype(np.float32)
print(xds_hazard[disname].idxmax(dim='time'), value_max)
#print(type(date_max), type(value_max))

#print(max_year1, max_year1.astype('datetime64[Y]'),max_year1.astype('datetime64[M]'),max_year1.astype('datetime64[D]'))
max_year = max_year1.astype('datetime64[Y]').item().year

# set time_start_plot
time_start_plot=f"{max_year}-01-01"
time_end_plot=f"{max_year}-12-31"
xds_hazard[disname].sel(time=slice(time_start_plot,time_end_plot)).plot(ax=axs[2])
axs[2].set_title("Most extreme year")

# save figure
if HazardDef == "gp" :
    gp_name = tlabel
else :
    gp_name = HazardDef
#figname=f"{figdir}/{gp_name}_timeseries_fitted.png"
figname=f"{figdir}/{gp_name}_timeseries_fitted.pdf"
plt.savefig(figname)
plt.close()


# here must define the dictionary where to save numerical results:
result_param = {}  # reset the result dictionary, so the first line is EFAS historical

#Saving results:
result_param[0] = {'member': 'histo', 'annual_max': year_max_abs, 'annual_date': date_ymax_abs, 'annual_year': years_abs, 'R_all': r_value, 'm_all': slope, 'p-value_all': p_value, 'R_ymax': r_value_ymax, 'm_ymax': slope_ymax, 'p-value_ymax': p_value_ymax}
print(result_param)


# HISTORICAL EFAS CYCLE:

# Compute statistics of seasonal cycle and save to file
# check if output exists
# Move this definition in the initial configuration part!
#window_size = 31  # was 45  # Smoothing to highlight mean seasonal cycle
percentile_value=50 # percentile of extremes values within 1 week window across years
#stats_clim_fname=f"{EFASDIR_HIST}/postpro/{tres}/watersheds/stats/efas_{watershed}_seasonalcycle_win{window_size}_pctx{percentile_value}.nc"
stats_clim_fname=f"{cycle_dir}efas_{watershed}_seasonalcycle_win{window_size}_pctx{percentile_value}.nc"

if os.path.exists(stats_clim_fname) and ReadExisting and 2>4:
    xds_stats_clim=xr.open_dataset(stats_clim_fname)
    # daily_means=xds_stats_clim['daily_means']
    # smoothed_means=xds_stats_clim['smoothed_means']
    # daily_max=xds_stats_clim['daily_max']
    # smoothed_max=xds_stats_clim['smoothed_max']
    # monthly_max_p=xds_stats_clim['week_max_p']
    # smoothed_week_max=xds_stats_clim['smoothed_week_max']
else:
    # mean seasonality
    daily_means = xds_hazard[disname].groupby('time.dayofyear').mean(dim='time')
    daily_max = xds_hazard[disname].groupby('time.dayofyear').max(dim='time')
    # Tino adds daily quantiles:
    daily_p50 = xds_hazard[disname].groupby('time.dayofyear').quantile(q=50 / 100, dim='time')
    daily_p90 = xds_hazard[disname].groupby('time.dayofyear').quantile(q=90 / 100, dim='time')


    week_max = xds_hazard[disname].resample(time='1W').max(dim='time').chunk({'time': -1})
    week_max_p50 = week_max.groupby('time.week').quantile(q=50 / 100, dim='time')
    week_max_p90 = week_max.groupby('time.week').quantile(q=90 / 100, dim='time')
    week_max_max = week_max.groupby('time.week').max(dim='time')

    monthly_max = xds_hazard[disname].resample(time='1M').max(dim='time').chunk({'time': -1})
    monthly_max_p50 = monthly_max.groupby('time.month').quantile(q=50 / 100, dim='time')
    monthly_max_p90 = monthly_max.groupby('time.month').quantile(q=90 / 100, dim='time')
    monthly_max_max = monthly_max.groupby('time.month').max(dim='time')

    # roolling average
    week_size=int(np.round(window_size/7))
    #smoothed_means = daily_means.rolling(dayofyear=window_size, center=True).mean()
    #smoothed_max = daily_max.rolling(dayofyear=window_size, center=True).mean()
    #smoothed_p50 = daily_p50.rolling(dayofyear=window_size, center=True).mean()
    #smoothed_p90 = daily_p90.rolling(dayofyear=window_size, center=True).mean()
    smoothed_means = circular_rolling(daily_means, window_size)
    smoothed_max = circular_rolling(daily_max, window_size)
    smoothed_p50 = circular_rolling(daily_p50, window_size)
    smoothed_p90 = circular_rolling(daily_p90, window_size)
    print(len(smoothed_means), len(smoothed_p90))  # 366

    # save to file
    xds_stats_clim=xr.Dataset()
    xds_stats_clim['daily_means']=daily_means
    xds_stats_clim['smoothed_means']=smoothed_means
    xds_stats_clim['smoothed_p50']=smoothed_p50
    xds_stats_clim['smoothed_p90']=smoothed_p90
    xds_stats_clim['daily_max']=daily_max
    xds_stats_clim['smoothed_max']=smoothed_max

    xds_stats_clim['monthly_max']=monthly_max
    xds_stats_clim['week_max_p50']=week_max_p50
    xds_stats_clim['week_max_p90']=week_max_p90
    xds_stats_clim['week_max_max']=week_max_max
    xds_stats_clim['monthly_max_p50']=monthly_max_p50
    xds_stats_clim['monthly_max_p90']=monthly_max_p90
    xds_stats_clim['monthly_max_max']=monthly_max_max
    
    # remove file if exists
    if os.path.exists(stats_clim_fname):
        os.remove(stats_clim_fname)
    xds_stats_clim.to_netcdf(stats_clim_fname)
    print("saved", stats_clim_fname)


# plot stats

# day corresponding to first day of each month
# length of months in days
mons = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_lengths = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
# Calcola il giorno dell'anno del primo giorno di ogni mese
# accumulate
month_starts = np.cumsum(month_lengths) - month_lengths
#print(len(xds_stats_clim['daily_means']), len(xds_stats_clim['smoothed_p90']))  # 366

plt.figure
fig, axs = plt.subplots(2, 1, figsize=(7, 10))  # 2 rows, 1 column
xds_stats_clim['daily_means'].plot(ax=axs[0], label='Daily mean')
xds_stats_clim['smoothed_means'].plot(ax=axs[0], label=f'$\pm${window_size//2}d mean')
xds_stats_clim['smoothed_p50'].plot(ax=axs[0], label=f'$\pm${window_size//2}d median')
xds_stats_clim['smoothed_p90'].plot(ax=axs[0], label=f'$\pm${window_size//2}d 90%')
xds_stats_clim['smoothed_max'].plot(ax=axs[0], label=f'$\pm${window_size//2}d Max')

axs[0].set_title(f'{gp_name} seasonal cycle: mean and percentiles with $\pm${(window_size//2)} days smooth')
axs[0].set_xticks(month_starts)
axs[0].set_xticklabels([])
#axs[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
for i, mo in enumerate(mons):
    axs[0].text(month_starts[i]+15, -0.05, mo, ha='center', va='center', transform=axs[0].get_xaxis_transform())
#axs[0].legend(['Daily mean', 'Smoothed mean'])
axs[0].legend()
axs[0].set_xlabel('')
axs[0].set_ylabel('Discharge [m$^3$/s]')
# remove day of year from plot

# tx=1
# xds_stats_clim['week_max_p50'].plot(ax=axs[tx])
# xds_stats_clim['week_max_p90'].plot(ax=axs[tx])
# xds_stats_clim['week_max_max'].plot(ax=axs[tx])
# axs[tx].set_title(f'Percentiles of week maxima')
# #axs[tx].set_xticks(np.arange(1, 13, 1))
# #axs[tx].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# axs[tx].legend(['50#tile', '90%tile', 'max'])
# axs[tx].set_xlabel('')
# axs[tx].set_ylabel('Discharge (m3/s)')

tx=1
xds_stats_clim['monthly_max_max'].plot(ax=axs[tx])
xds_stats_clim['monthly_max_p90'].plot(ax=axs[tx])
xds_stats_clim['monthly_max_p50'].plot(ax=axs[tx])

axs[tx].set_title(f'{gp_name} seasonal cycle of monthly extremes')
axs[tx].set_xticks(np.arange(1, 13, 1))
axs[tx].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
#axs[tx].set_xticklabels([])
#for i, mo in enumerate(months):
#    axs[tx].text(np.arange(0.5, 12.5, 1)[i], -0.05, mo, ha='center', va='center', transform=axs[tx].get_xaxis_transform())

axs[tx].legend(['max','90% ', '50%'])
axs[tx].set_xlabel('')
axs[tx].set_ylabel('Discharge [m$^3$/s]')
# legend

# save figure
if HazardDef == "gp" :
    gp_name = tlabel
else :
    gp_name = HazardDef
figname=f"{figdir}/{gp_name}_seasonal_cycle.pdf"
plt.savefig(figname)
plt.close()
print("saved", figname)

#print(stop)

#****************************************************************#
#               START WORKING ON EFAS SURROGATES:                #
#****************************************************************#


start_time = time.time()
# In this cell only surrogate members are read. The original EFAS historical are done in the previous cell.

# With Giuseppe we decide to cut the surrogates before 1/1/2000 and after 31/12/2023
#Apr_start = np.datetime64("1999-04-01") + np.timedelta64(9131,'D')
#Jul_end = np.datetime64("1999-07-01") + np.timedelta64(9131,'D')
surrogate_start = np.datetime64("2000-01-01")  # np.datetime64("1999-07-01")
surrogate_end = np.datetime64("2023-12-31")

def process_month(month, EFAS_surro, watershed, ReadExisting):
  # Here start the Big loop on each surrogate timeseries:
  print("*********************************\n Processing month", month, "****************\n")

  # surrogate PATHS:
  wshed_fname_sur_dir=f"{EFAS_surro}/{watershed.capitalize()}/{month}/"
  #fname_gp_sur=f"{EFAS_surro}/{watershed.capitalize()}/{month}/efas_surrogate_{watershed}_{tlabel}_{month}_{member}_masked.nc"
  fname_gp_sur_dir=f"{EFAS_surro}/{watershed.capitalize()}/{month}/"

  # Read EFAS surrogate data for single grid point, and save it to .nc for more rapid access in future executions
  # It would make sense to have a function that reads the hazard: either as a single grid point, watershed max, or economic
  def read_and_store_time_series(fname_pat, lat, lon, output_file, ReadExisting):
        
    """
    Read time series data from a specified grid point and store it on disk.
    
    Parameters:
    fname_pat (str or list of str): File path pattern or list of file paths.
    lat_index (int): Index of the latitude grid point.
    lon_index (int): Index of the longitude grid point.
    output_file (str): Output file name to store the time series data (default: 'time_series_data.nc').
    check_existing (bool): Whether to check if the output file already exists (default: True).
    """
    print("I'm here 0\n")
    # Check if the output file already exists
    if ReadExisting and os.path.exists(output_file):
        print("I'm here 0.5\n")
        print("File existing", ReadExisting, output_file)
        # If the output file exists, read it straight away
        time_series = xr.open_dataset(output_file)
        if "surrogate" in output_file :
            if 'forecast_period' in time_series.dims :
                print('renaming forecast_period in time')
                time_series = time_series.rename({'forecast_period': 'time'})
                #time_series = time_series.rename_vars({'forecast_period': 'time'})
            if 'forecast_reference_time' in time_series.dims and time_series.dims['forecast_reference_time'] == 1:
                time_series = time_series.squeeze(dim='forecast_reference_time')
            if 'forecast_reference_time' in time_series.coords:
                time_series = time_series.drop_vars('forecast_reference_time')
    else:
        print(f"trying to read {fname_pat}\n and write {output_file}")
        # If the output file does not exist, proceed with extracting data from the dataset
        if "reforecast" in output_file :
            print("Non implemented for now!")
        elif "surrogate" in output_file :
            #print("Input passed=", fname_pat, "why rebuilding the name?")
            #fname_single_sur=f"{EFAS_surro}/{watershed.capitalize()}/{month}/EFAS5_surrogate_{watershed.capitalize()}_dis24_trimestral_{month}_{mem}.nc"
            #print(fname_single_sur)
            fname_single_sur = fname_pat
            print(f"Trying to read {fname_single_sur} and write {output_file}")
            if os.path.exists(fname_single_sur) :
                #command= f"ls -lt {fname_single_sur}" 
                #print(command)
                #os.system(command)
                print(f"File {fname_single_sur} exists")
                subprocess.run(["ls", "-lt", fname_single_sur], check=True, timeout=5)
                dataset = xr.open_mfdataset(fname_single_sur, combine='by_coords')  # Apri il singolo file
                dataset.load()
                dataset.close()
                print("afer reading input nc--- %s seconds ---" % (time.time() - start_time))
                #print(dataset)
            else :
                print(f"File {fname_single_sur} does not exists!")
                 

        else :         
            # Open the multi-file dataset
            dataset = xr.open_mfdataset(fname_pat, combine='by_coords')
        #print(dataset)
        
        


        # Extract time series data for the specified grid point
        # Note that reforecast EFAS have latitude and longitude, while hystorical EFAS has lat and lon!
        if ("surrogate" in output_file) or ("reforecast" in output_file) :
            print("latitude")
            #time_series = dataset.sel(latitude=lat,longitude=lon,method='nearest')   # for Aragon
            time_series = dataset.sel(lat=lat,lon=lon,method='nearest')   # for Panaro
            time_series.load()
            print("after extracting gp nearest values timeserie--- %s seconds ---" % (time.time() - start_time))
            if 'forecast_period' in time_series.dims :
                # rename the dimension time to time for compatibility with historical EFAS
                print('renaming forecast_period in time')
                time_series = time_series.rename({'forecast_period': 'time'})
                #time_series = time_series.rename_vars({'forecast_period': 'time'})
            if 'forecast_reference_time' in time_series.dims and time_series.dims['forecast_reference_time'] == 1:
                time_series = time_series.squeeze(dim='forecast_reference_time')
            if 'forecast_reference_time' in time_series.coords:
                time_series = time_series.drop_vars('forecast_reference_time')
            print("after renaming/dropping different coordinates timeserie--- %s seconds ---" % (time.time() - start_time))
            #print(time_series)
        else :
            print("lat")
            time_series = dataset.sel(lat=lat,lon=lon,method='nearest')
            time_series.load()
            dataset.close()
            print("after extracting gp nearest values timeserie--- %s seconds ---" % (time.time() - start_time))

        # Store the extracted time series data on disk
        time_series.to_netcdf(output_file)
        #os.system(f"ls -lt {output_file}")
        print(f"File {output_file} exists")
        subprocess.run(["ls", "-lt", output_file], check=True, timeout=5)
        print("after saving non-existing timeserie--- %s seconds ---" % (time.time() - start_time))
    print("I'm here 1\n")
    return time_series

  for member in range(0,25):
  #for member in range(0,1):
    member = str(member).zfill(2)
    #if len(member) == 1 :
    #    member="0"+member
    print("Working on", month, member)
    wshed_fname_sur = f"{wshed_fname_sur_dir}/EFAS5_surrogate_{watershed.capitalize()}_dis24_trimestral_{month}_{member}.nc"
    fname_gp_sur = f"{fname_gp_sur_dir}/efas_surrogate_{watershed}_{tlabel}_{month}_{member}_masked.nc"


    # read hazard based on hazard definition
    print(wshed_fname)
    if HazardDef=="gp":
        # Decomment to analyze the EFAS historical also here:
        #xds_hazard = read_and_store_time_series(wshed_fname, tlat, tlon, fname_gp, ReadExisting, -99)
        ##xds_hazard_ref = read_and_store_time_series(wshed_fname_ref, tlat, tlon, fname_gp_ref, ReadExisting)
        xds_hazard_sur = read_and_store_time_series(wshed_fname_sur, tlat, tlon, fname_gp_sur, ReadExisting)
        print("I'm here 2\n")
    '''
    elif HazardDef=="wmax":
        # if maximum discharge exists, read it, otherwise compute it
        #if os.path.exists(wshed_dismax_fname) and ReadExisting:
        #    xds_hazard=xr.open_dataset(wshed_dismax_fname)
        if os.path.exists(wshed_dismax_fname_sur) and ReadExisting:       
            xds_hazard_sur=xr.open_dataset(wshed_dismax_fname_sur)        
        else:
            # read EFAS data``
            #xds=xr.open_mfdataset(wshed_fname,combine='by_coords')
            #xds_hazard=xds.max(dim=['lat', 'lon'])
            #print(f"trying to save {wshed_dismax_fname}")
            # save to file
            #xds_hazard.to_netcdf(wshed_dismax_fname)

            ### read EFAS reforecast data``
            ##xds_ref=xr.open_mfdataset(wshed_fname_ref,combine='by_coords')
            ##xds_hazard_ref=xds_ref.max(dim=['lat', 'lon'])
            # read EFAS reforecast surrogate data``
            print(f"trying to read {wshed_fname_sur}")
            xds_sur=xr.open_mfdataset(wshed_fname_sur,combine='by_coords')
            xds_hazard_sur=xds_sur.max(dim=['lat', 'lon'])   # For Aragon is latitude and longitude!!!
            #xds_hazard_sur=xds_sur.max(dim=['latitude', 'longitude'])   # For Aragon is latitude and longitude!!!
    '''
    oldest_sur= xds_hazard_sur.time.min().values
    recent_sur= xds_hazard_sur.time.max().values    
    print(f"Original oldest date: {oldest_sur}. Most recent date: {recent_sur}")
    xds_hazard_sur = xds_hazard_sur.sel(time=slice(surrogate_start, surrogate_end))
    oldest_sur= xds_hazard_sur.time.min().values
    recent_sur= xds_hazard_sur.time.max().values    
    print(f"After filtering: oldest date: {oldest_sur}. Most recent date: {recent_sur}")
    xds_hazard_sur.load()
    
    print("before starting figures timeserie--- %s seconds ---" % (time.time() - start_time))
    # Starting to elaborate the time series:
    
    # FITTING the timeseries:
    # Convert the time index to numerical values (e.g., ordinal)
    time_numeric = xds_hazard_sur.coords["time"].values.astype('datetime64[s]').astype(int)
    print("after converting time in numeric--- %s seconds ---" % (time.time() - start_time))
    # Perform linear regression on the numerical time values and data
    mask = ~np.isnan(time_numeric) & ~np.isnan(xds_hazard_sur[disname])
    print(mask)
    slope, intercept, r_value, p_value, std_err = linregress(time_numeric[mask], xds_hazard_sur[disname].sel(time=mask.time))
    print(f"All days:\nm={slope}, q={intercept}, R={r_value}, p-value={p_value}, StdErr={std_err}")
    # Compute the fitted values using the regression parameters
    fitted_values = slope * time_numeric + intercept
    # converting slope from seconds to years:
    slope = slope*3600*24*365.2425    

    #print(f"trying to save {wshed_dismax_fname_sur}")
    # save figure:
    # plot timeseries using xarray 
    fig,axs=plt.subplots(3,1,figsize=(8,10))
    time_start_plot="2000-01-01"  #"1999-08-01"
    time_end_plot="2023-12-31"
    xds_hazard_sur[disname].sel(time=slice(time_start_plot,time_end_plot)).plot(ax=axs[0])
    axs[0].plot(xds_hazard_sur.time, fitted_values, color="r", linestyle='dashed')
    if HazardDef == "gp" :
        gp_name = tlabel
    else :
        gp_name = HazardDef
    axs[0].set_title(f"{gp_name} {disname} in {watershed.capitalize()} daily mean m={month}/{member}")
    axs[0].set_xlabel("")

    # select year with maximum discharge
    #max_year1 = xds_hazard_sur[disname].idxmax(dim='time').values
    #print("Historical maximum: ", max_year1.astype('datetime64[Y]').item().year, max_year1.astype('datetime64[Y]').item().month, max_year1.astype('datetime64[Y]').item().day)

    # Second figure: Adding the timeseries of the annual maxima (one value per year):
    year_max = xds_hazard_sur[disname].resample(time='1YE').max(dim='time').chunk({'time': -1})  # saves only maximum RD per year, sot original dates
    print("Annual Maxima=", year_max.values)
    year_max_mean_value = year_max.values.mean()
    year_max_std_value = year_max.values.std()
    year_max_StdErr_value = 2*year_max_std_value/math.sqrt(len(year_max.values))
    print(f"Average value of surrogate annual maxima= {year_max_mean_value}, StD= {year_max_std_value}, 2*StdErr= {year_max_StdErr_value}")
    # years = year_max.astype('datetime64[Y]').item().year  # da errore
    years = year_max['time'].dt.year.values  # .tolist()
    print("years", years)
    # exact dates of each annual maximum:
    date_ymax = xds_hazard_sur[disname].resample(time='1YE').apply(lambda x: x.idxmax(dim='time'))  # By Paolo Davini!
    print("date_ymax", date_ymax)
    # Removing 12h to center at midday instead of 00 of following day:
    date_ymax = date_ymax - np.timedelta64(12,'h')
    date_ymax_list = date_ymax['time'].dt.date.values.tolist()
    #date_ymax_abs =  xds_hazard_sur[disname].apply(lambda x: x.idxmax(dim='time'))
    # Date of absolute maximum:
    date_ymax_abs =  xds_hazard_sur[disname].idxmax(dim='time').values - np.timedelta64(12,'h')
    # Asbolute maximum:
    year_max_abs = xds_hazard_sur[disname].max(dim='time').values  #.chunk({'time': -1})
    # Year of absolute maximum:
    years_abs = date_ymax_abs.astype('datetime64[Y]').astype(int) + 1970
    print(f'Maximum of all timeseries= {year_max_abs} on {date_ymax_abs}')

    print(date_ymax.values, "\n", year_max.values)
    annual_maxima = xr.Dataset(
        {
        'dis24': (["time"], year_max.values) 
        },
        coords= {
        'time': date_ymax.values,  # take other coordinates date_ymax
        "lon": year_max.lon,       # Altre coordinate
        "lat": year_max.lat #,
        #"spatial_ref": year_max.spatial_ref
        })



    print(f"Annual_max_XR= {annual_maxima}")

    # save to file
    fname_gp_ymax_sur = fname_gp_ymax_sur = f"{EFAS_surro}/{watershed.capitalize()}/{month}/efas_surrogate_{watershed}_{tlabel}_{month}_{member}_masked_annual_maxima.nc"
    print(f"Saving {fname_gp_ymax_sur}")
    # Keeps giving permission errors! First remove file:
    # remove file if exists
    #if os.path.exists(fname_gp_ymax_sur):
    #        os.remove(fname_gp_ymax_sur)
    if os.path.exists(fname_gp_ymax_sur):
        try:
            os.remove(fname_gp_ymax_sur)
            print(f"Removed file: {fname_gp_ymax_sur}")
        except Exception as e:
            print(f"Error removing file {fname_gp_ymax_sur}: {e}")
    annual_maxima.to_netcdf(fname_gp_ymax_sur, unlimited_dims={'time':True}, mode="w")
    #annual_maxima.to_netcdf(fname_gp_ymax)

    # Redoing the same using real MaxDate instead than mid-of-year bars:
    # FITTING the timeseries:
    # Convert the time index to numerical values (e.g., ordinal)
    date_ymax_numeric = date_ymax.coords["time"].values.astype('datetime64[s]').astype(int)
    #slope, intercept, r_value, p_value, std_err = linregress(time_numeric[mask], casalecchio_ds.streamflow[mask])
    slope_ymax, intercept_ymax, r_value_ymax, p_value_ymax, std_err_ymax = linregress(date_ymax_numeric, year_max.values)
    #slope_ymax, intercept_ymax, r_value_ymax, p_value_ymax, std_err_ymax = linregress(date_ymax_list, year_max.values[:-1])
    #slope, intercept, r_value, p_value, std_err = linregress(time_numeric[mask], xds_hazard[mask])
    print(f"Only annual maxima: \nm={slope_ymax}, q={intercept_ymax}, R={r_value_ymax}, p-value={p_value_ymax}, StdErr={std_err_ymax}")
    # Compute the fitted values using the regression parameters
    fitted_values_ymax = slope_ymax * date_ymax_numeric + intercept_ymax
    # converting slope from seconds to years:
    slope_ymax = slope_ymax*3600*24*365.2425
    print("Fit_ymax=")
    print(fitted_values_ymax, "HERE!")

    # years is numeric, date_max are dates:
    #axs[1].bar(years, year_max.values[:-1])
    axs[1].vlines(annual_maxima.time, ymin=0, ymax=annual_maxima.dis24, linewidth=2)
    #axs[1].plot(date_ymax, year_max.values[:-1])
    axs[1].plot(annual_maxima.time, fitted_values_ymax, color="r", linestyle='dashed')
    if HazardDef == "gp" :
        gp_name = tlabel
    else :
        gp_name = HazardDef
    axs[1].set_title(f"{gp_name} {disname} in {watershed} annual maxima m={month}/{member}")
    axs[1].set_xlabel("")
    #axs[1].legend([f"m={round(slope_ymax, 1)} $m^{3}s^{{-1}}y^{{-1}}$, R={round(r_value_ymax, 2)}, p-value={round(p_value_ymax, 3)}", "Max-of-Year"],  prop={'size': 8}, loc='upper left')
    axs[1].legend(["Max of each Year", f"m={round(slope_ymax, 1)} $m^{3}s^{{-1}}y^{{-1}}$, R={round(r_value_ymax, 2)}, p-value={round(p_value_ymax, 3)}"],  prop={'size': 8}, loc='upper left')
    #handles, labels = plt.gca().get_legend_handles_labels()  # Ottieni handles e labels
    #axs[1].legend(handles[::-1], labels[::-1])

    # Third figure select year with maximum discharge
    print(f"input len= {len(xds_hazard[disname])}, {len(xds_hazard[disname])/365} years")
    max_year1_sur = xds_hazard_sur[disname].idxmax(dim='time').values
    print("Historical maximum: ", max_year1_sur.astype('datetime64[D]'), type(max_year1_sur.astype('datetime64[D]')))
    date_max=max_year1_sur.astype('datetime64[D]')
    value_max=xds_hazard_sur[disname].sel(time=date_max).values   # .astype(np.float32)
    print(xds_hazard_sur[disname].idxmax(dim='time'), value_max)
    #print(type(date_max), type(value_max))

    #print(max_year1, max_year1.astype('datetime64[Y]'),max_year1.astype('datetime64[M]'),max_year1.astype('datetime64[D]'))
    max_year = max_year1_sur.astype('datetime64[Y]').item().year


    # set time_start_plot
    time_start_plot=f"{max_year}-01-01"
    time_end_plot=f"{max_year}-12-31"
    xds_hazard_sur[disname].sel(time=slice(time_start_plot,time_end_plot)).plot(ax=axs[2])
    #xds_hazard_sur[disname].plot(ax=axs[1])
    axs[2].set_title(f"Most extreme year in {gp_name} {watershed.capitalize()} m={month}/{member}")

    # save figure
    if HazardDef == "gp" :
        gp_name = tlabel
    else :
        gp_name = HazardDef
    #figname=f"{figdir_sur}/EFAS_surrogate_{gp_name}_timeseries_fitted_{month}_{member}.png"
    figname=f"{figdir_sur}/EFAS_surrogate_{gp_name}_timeseries_fitted_{month}_{member}.pdf"
    plt.savefig(figname)
    plt.close()
    print(f"wrote {figname}")


    # ACHTUNG: WHY SAVING xds_hazard_sur ON wshed_DISMAX_fname_sur instead of wshed_fname_sur?
    # BTW, it seems that wshed_dismax_fname_sur are never used before...
    # save to nc file
    #wshed_dismax_fname_sur=f"{wshed_dismax_fname_sur_dir}efas_{watershed}_{month}_{member}.nc"
    #print(f"Saving {wshed_dismax_fname_sur}")
    #xds_hazard_sur.to_netcdf(wshed_dismax_fname_sur)

    # discharge values (convert to numpy array)
    #ds_hazard_sur=xds_hazard_sur[disname].values   # why doing that?


    #Saving results:
    following_key = max(result_param.keys()) + 1
    result_param[following_key] = {'member': str(month+member), 'annual_max': year_max_abs, 'annual_date': date_ymax_abs, 'annual_year': years_abs, 'R_all': r_value, 'm_all': slope, 'p-value_all': p_value, 'R_ymax': r_value_ymax, 'm_ymax': slope_ymax, 'p-value_ymax': p_value_ymax}




    '''
    # Repeating with bars in the middle figure:
    #
    # FITTING the timeseries:
    # Convert the time index to numerical values (e.g., ordinal)
    time_numeric = xds_hazard_sur.coords["time"].values.astype('datetime64[s]').astype(int)
    # Perform linear regression on the numerical time values and data
    mask = ~np.isnan(time_numeric) & ~np.isnan(xds_hazard_sur[disname])
    print(mask)
    slope, intercept, r_value, p_value, std_err = linregress(time_numeric[mask], xds_hazard_sur[disname].sel(time=mask.time))
    print(f"All days:\nm={slope}, q={intercept}, R={r_value}, p-value={p_value}, StdErr={std_err}")
    # Compute the fitted values using the regression parameters
    fitted_values = slope * time_numeric + intercept

    # save figure:
    # plot timeseries using xarray 
    fig,axs=plt.subplots(3,1,figsize=(8,10))
    time_start_plot="2000-01-01"  # "1999-08-01"
    time_end_plot="2023-12-31"
    xds_hazard_sur[disname].sel(time=slice(time_start_plot,time_end_plot)).plot(ax=axs[0])
    axs[0].plot(xds_hazard_sur.time, fitted_values, color="r", linestyle='dashed')
    if HazardDef == "gp" :
        gp_name = tlabel
    else :
        gp_name = HazardDef
    axs[0].set_title(f"{gp_name} {disname} in {watershed.capitalize()} daily mean m={month}/{member}")
    axs[0].set_xlabel("")

    # select year with maximum discharge
    #max_year1 = xds_hazard_sur[disname].idxmax(dim='time').values
    #print("Historical maximum: ", max_year1.astype('datetime64[Y]').item().year, max_year1.astype('datetime64[Y]').item().month, max_year1.astype('datetime64[Y]').item().day)

    # Second figure: Adding the timeseries of the annual maxima (one value per year):
    
    #
    # FITTING the timeseries:
    # Convert the time index to numerical values (e.g., ordinal)
    date_ymax_numeric = date_ymax.coords["time"].values.astype('datetime64[s]').astype(int)
    #slope, intercept, r_value, p_value, std_err = linregress(time_numeric[mask], casalecchio_ds.streamflow[mask])
    #slope_ymax, intercept_ymax, r_value_ymax, p_value_ymax, std_err_ymax = linregress(date_ymax_numeric, year_max.values[:-1])
    slope_ymax, intercept_ymax, r_value_ymax, p_value_ymax, std_err_ymax = linregress(years, year_max.values)
    #slope, intercept, r_value, p_value, std_err = linregress(time_numeric[mask], xds_hazard[mask])
    print(f"Only annual maxima: \nm={slope_ymax}, q={intercept_ymax}, R={r_value_ymax}, p-value={p_value_ymax}, StdErr={std_err_ymax}")
    # Compute the fitted values using the regression parameters
    #fitted_values_ymax = slope_ymax * date_ymax_numeric + intercept_ymax
    fitted_values_ymax = slope_ymax * years + intercept_ymax
    ## converting slope from seconds to years:
    ##slope_ymax = slope_ymax*3600*24*365.2425
    print("Fit_ymax=")
    print(fitted_values_ymax, "HERE!")

    # years is numeric, date_max are dates:
    axs[1].bar(years, year_max.values)
    #axs[1].vlines(annual_maxima.time, ymin=0, ymax=annual_maxima.dis24, linewidth=2)
    #axs[1].plot(date_ymax, year_max.values[:-1])
    #axs[1].plot(annual_maxima.time, fitted_values_ymax, color="r", linestyle='dashed')
    axs[1].plot(years, fitted_values_ymax, color="r", linestyle='dashed')
    if HazardDef == "gp" :
        gp_name = tlabel
    else :
        gp_name = HazardDef
    axs[1].set_title(f"{gp_name} {disname} in {watershed} annual maxima m={month}/{member}")
    axs[1].set_xlabel("")
    #axs[1].legend([f"m={round(slope_ymax, 1)} $m^{3}s^{{-1}}y^{{-1}}$, R={round(r_value_ymax, 2)}, p-value={round(p_value_ymax, 3)}", "Max-of-Year"],  prop={'size': 8}, loc='upper left')
    axs[1].legend(["Max of each Year", f"m={round(slope_ymax, 1)} $m^{3}s^{{-1}}y^{{-1}}$, R={round(r_value_ymax, 2)}, p-value={round(p_value_ymax, 3)}"],  prop={'size': 8}, loc='upper left')
    #handles, labels = plt.gca().get_legend_handles_labels()  # Ottieni handles e labels
    #axs[1].legend(handles[::-1], labels[::-1])

    # Third figure select year with maximum discharge
    print(f"input len= {len(xds_hazard[disname])}, {len(xds_hazard[disname])/365} years")
    max_year1_sur = xds_hazard_sur[disname].idxmax(dim='time').values
    print("Historical maximum: ", max_year1_sur.astype('datetime64[D]'), type(max_year1_sur.astype('datetime64[D]')))
    date_max=max_year1_sur.astype('datetime64[D]')
    value_max=xds_hazard_sur[disname].sel(time=date_max).values   # .astype(np.float32)
    print(xds_hazard_sur[disname].idxmax(dim='time'), value_max)
    #print(type(date_max), type(value_max))

    #print(max_year1_sur, max_year1_sur.astype('datetime64[Y]'),max_year1_sur.astype('datetime64[M]'),max_year1_sur.astype('datetime64[D]'))
    max_year = max_year1_sur.astype('datetime64[Y]').item().year


    # set time_start_plot
    time_start_plot=f"{max_year}-01-01"
    time_end_plot=f"{max_year}-12-31"
    xds_hazard_sur[disname].sel(time=slice(time_start_plot,time_end_plot)).plot(ax=axs[2])
    #xds_hazard_sur[disname].plot(ax=axs[1])
    axs[2].set_title(f"Most extreme year in {gp_name} {watershed.capitalize()} m={month}/{member}")

    # save figure
    if HazardDef == "gp" :
        gp_name = tlabel
    else :
        gp_name = HazardDef
    #figname=f"{figdir_sur}/EFAS_surrogate_{gp_name}_timeseries_fitted_{month}_{member}.png"
    figname=f"{figdir_sur}/EFAS_surrogate_{gp_name}_timeseries_fitted_bars_{month}_{member}.pdf"
    plt.savefig(figname)
    print(f"wrote {figname}")

    # end repeated figure
    '''


    # Adding in the surrogate loop also the seasonal cycle figure:
    # Compute statistics of seasonal cycle and save to file
    # check if output exists

    # Parameters defined above for the historical EFAS:
    #window_size = 45  # Smoothing to highlight mean seasonal cycle
    #percentile_value=50 # percentile of extremes values within 1 week window across years
    #stats_clim_fname=f"{EFASDIR_HIST}/postpro/{tres}/watersheds/stats/efas_{watershed}_seasonalcycle_win{window_size}_pctx{percentile_value}.nc"
    stats_clim_fname_sur=f"{cycle_dir}efas_{watershed}_seasonalcycle_win{window_size}_pctx{percentile_value}_{month}_{member}.nc"

    if os.path.exists(stats_clim_fname_sur) and ReadExisting :
        xds_stats_clim_sur=xr.open_dataset(stats_clim_fname_sur)
        # daily_means=xds_stats_clim['daily_means']
        # smoothed_means=xds_stats_clim['smoothed_means']
        # daily_max=xds_stats_clim['daily_max']
        # smoothed_max=xds_stats_clim['smoothed_max']
        # monthly_max_p=xds_stats_clim['week_max_p']
        # smoothed_week_max=xds_stats_clim['smoothed_week_max']
    else:
        # mean seasonality
        daily_means_sur = xds_hazard_sur[disname].groupby('time.dayofyear').mean(dim='time')
        daily_max_sur = xds_hazard_sur[disname].groupby('time.dayofyear').max(dim='time')
        # Tino adds daily quantiles:
        daily_p50_sur = xds_hazard_sur[disname].groupby('time.dayofyear').quantile(q=50/100, dim='time')
        daily_p90_sur = xds_hazard_sur[disname].groupby('time.dayofyear').quantile(q=90/100, dim='time')


        week_max_sur = xds_hazard_sur[disname].resample(time='1W').max(dim='time').chunk({'time': -1})
        week_max_p50_sur = week_max_sur.groupby('time.week').quantile(q=50/100, dim='time')
        week_max_p90_sur = week_max_sur.groupby('time.week').quantile(q=90/100, dim='time')
        week_max_max_sur = week_max_sur.groupby('time.week').max(dim='time')

        monthly_max_sur = xds_hazard_sur[disname].resample(time='1M').max(dim='time').chunk({'time': -1})
        monthly_max_p50_sur = monthly_max_sur.groupby('time.month').quantile(q=50/100, dim='time')
        monthly_max_p90_sur = monthly_max_sur.groupby('time.month').quantile(q=90/100, dim='time')
        monthly_max_max_sur = monthly_max_sur.groupby('time.month').max(dim='time')

        # roolling average
        week_size=int(np.round(window_size/7))
        #smoothed_means_sur = daily_means_sur.rolling(dayofyear=window_size, center=True).mean()
        #smoothed_max_sur = daily_max_sur.rolling(dayofyear=window_size, center=True).mean()
        smoothed_means_sur = circular_rolling(daily_means_sur, window_size)
        smoothed_max_sur = circular_rolling(daily_max_sur, window_size)
        smoothed_p50_sur = circular_rolling(daily_p50_sur, window_size)
        smoothed_p90_sur = circular_rolling(daily_p90_sur, window_size)
        print(len(smoothed_means), len(smoothed_p90))  # 366


        # save to file
        xds_stats_clim_sur=xr.Dataset()
        xds_stats_clim_sur['daily_means']=daily_means_sur
        xds_stats_clim_sur['smoothed_means']=smoothed_means_sur
        xds_stats_clim_sur['smoothed_p50']=smoothed_p50_sur
        xds_stats_clim_sur['smoothed_p90']=smoothed_p90_sur
        xds_stats_clim_sur['daily_max']=daily_max_sur
        xds_stats_clim_sur['smoothed_max']=smoothed_max_sur
        xds_stats_clim_sur['monthly_max']=monthly_max_sur
        xds_stats_clim_sur['week_max_p50']=week_max_p50_sur
        xds_stats_clim_sur['week_max_p90']=week_max_p90_sur
        xds_stats_clim_sur['week_max_max']=week_max_max_sur
        xds_stats_clim_sur['monthly_max_p50']=monthly_max_p50_sur
        xds_stats_clim_sur['monthly_max_p90']=monthly_max_p90_sur
        xds_stats_clim_sur['monthly_max_max']=monthly_max_max_sur
        
        # remove file if exists
        #if os.path.exists(stats_clim_fname_sur):
        #    os.remove(stats_clim_fname_sur)
        if os.path.exists(stats_clim_fname_sur):
            try:
                os.remove(stats_clim_fname_sur)
                print(f"Removed file: {stats_clim_fname_sur}")
            except Exception as e:
                print(f"Error removing file {stats_clim_fname_sur}: {e}")
        xds_stats_clim_sur.to_netcdf(stats_clim_fname_sur)
        print(f"Wrote {stats_clim_fname_sur}")
        # HERE SAVE THE SEASONAL CYCLE DATA
        # f""/home/manzato/work/programs/postpro/{tres}/watersheds/stats/efas_{watershed}_seasonalcycle_win{window_size}_pctx{percentile_value}_{month}_{member}.nc"
        # /home/manzato/work/programs/postpro/day/watersheds/stats/efas_panaro_seasonalcycle_win45_pctx50_Apr_00.nc o efas_reno_seasonalcycle_win45_pctx50_Apr_00.nc
        # Also historical EFFAS are there:    efas_panaro_seasonalcycle_win45_pctx50.nc  and   efas_reno_seasonalcycle_win45_pctx50.nc
        # Each file have the 11 vars above for 365 days o year

    # plot surrogate stats


    # day corresponding to first day of each month
    # length of months in days
    mons = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_lengths = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    # Calcola il giorno dell'anno del primo giorno di ogni mese
    # accumulate
    month_starts = np.cumsum(month_lengths) - month_lengths
    #print(len(xds_stats_clim['daily_means']), len(xds_stats_clim['smoothed_p90']))  # 366

    plt.figure
    fig, axs = plt.subplots(2, 1, figsize=(7, 10))  # 2 rows, 1 column
    xds_stats_clim_sur['daily_means'].plot(ax=axs[0], label='Daily mean')
    xds_stats_clim_sur['smoothed_means'].plot(ax=axs[0], label=f'$\pm${window_size//2}d mean')
    xds_stats_clim_sur['smoothed_p50'].plot(ax=axs[0], label=f'$\pm${window_size//2}d median')
    xds_stats_clim_sur['smoothed_p90'].plot(ax=axs[0], label=f'$\pm${window_size//2}d 90%')
    xds_stats_clim_sur['smoothed_max'].plot(ax=axs[0], label=f'$\pm${window_size//2}d Max')

    axs[0].set_title(f'{gp_name} seasonal cycle: mean and percentiles with $\pm${(window_size//2)} days smooth {month}/{member}')
    axs[0].set_xticks(month_starts)
    axs[0].set_xticklabels([])
    #axs[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    for i, mo in enumerate(mons):
        axs[0].text(month_starts[i]+15, -0.05, mo, ha='center', va='center', transform=axs[0].get_xaxis_transform())
    #axs[0].legend(['Daily mean', 'Smoothed mean'])
    axs[0].legend()
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Discharge [m$^3$/s]')



    tx=1
    xds_stats_clim_sur['monthly_max_max'].plot(ax=axs[tx])
    xds_stats_clim_sur['monthly_max_p90'].plot(ax=axs[tx])
    xds_stats_clim_sur['monthly_max_p50'].plot(ax=axs[tx])

    axs[tx].set_title(f'{gp_name} seasonal cycle of monthly extremes {month}/{member}')
    axs[tx].set_xticks(np.arange(1, 13, 1))
    axs[tx].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    axs[tx].legend(['max','90%', '50%'])
    axs[tx].set_xlabel('')
    axs[tx].set_ylabel('Discharge [m$^3$/s]')
    # legend

    # save figure
    if HazardDef == "gp" :
        gp_name = tlabel
    else :
        gp_name = HazardDef
    figname=f"{figdir_sur}/{gp_name}_seasonal_cycle_surrogate_{month}_{member}.pdf"
    plt.savefig(figname)
    plt.close()
    print("Saved", figname)

  # END Process_month:
  save_results_txt(f"{figdir_sur}/EFAS_surrogate_{gp_name}_timeseries_fit_results_{month}.txt")
  print(f"Fit results saved in {figdir_sur}/EFAS_surrogate_{gp_name}_timeseries_fit_results_{month}.txt")
  # This function will run in parallel for each month
  

# Running on 4 different CPU cores the four months at the same time!
if __name__ == "__main__":
    months = ["Apr", "May", "Jun", "Jul"]
    efas_path = EFAS_surro
    watershed = "Panaro"
    read_existing = True

    with mp.Pool(processes=4) as pool:
        pool.starmap(process_month, [(month, efas_path, watershed, read_existing) for month in months])



#*******************************************************************************************#
#               COMPARING HISTORICAL EFAS with STATISTICS of EFAS SURROGATES:                #
#*******************************************************************************************#

#*****************************
# Histogram of R and SLOPES m:

# READING Results from the 4 txt months and put together:
results_df = pd.DataFrame()

for m in ["Apr", "May", "Jun", "Jul"] :
#for m in ["Apr", "May"] :

    filein = f"{figdir_sur}EFAS_surrogate_{gp_name}_timeseries_fit_results_{m}.txt"
    print(f"Reading {filein} ")
    if os.path.exists(filein) :
        df=pd.read_csv(os.path.join(filein), delimiter="\t", parse_dates=[2])
        df.info()
        # Add a new column like the first one, but casting from string to datetime:
        #df['annual_date'] = pd.to_datetime(df['annual_date UTC'], utc=True)
        #df.info()
        #all_result_param =xr.open_dataset(filein)
        #print(df)
        if m == "Apr" :
            results_df = df
        else :
            df = df.iloc[1:].reset_index(drop=True)  # removing the first entry, which is always the historical EFAS
            results_df = pd.concat([results_df, df], ignore_index=True)
    else:
        print(f"File {filein} does not exist!")

#print(results_df)

print("\nAnalysis of linear fit on the daily mean RD of each surrogate:")
R_mean = results_df["R_all"][1:].mean()
R_std = results_df["R_all"][1:].std()
TwoSE = 2*R_std/math.sqrt(100)
print(f"Mean correlation for 100 surrogate Daily-mean-RD +-2SE = {R_mean} +- {TwoSE}")
print(f"Historical EFAS correlation was {results_df["R_all"][0]}")
m_mean = results_df["m_all"][1:].mean()
m_std = results_df["m_all"][1:].std()
TwoSE = 2*m_std/math.sqrt(100)
print(f"Mean slope for 100 surrogate Daily-mean-RD +-2SE= {m_mean} +- {TwoSE}")
print(f"Historical EFAS slope was {results_df["m_all"][0]}")

print("\nAnalysis of linear fit on the absolute maximum of each surrogate:")
R_mean = results_df["R_ymax"][1:].mean()
R_std = results_df["R_ymax"][1:].std()
TwoSE = 2*R_std/math.sqrt(100)
print(f"Mean correlation for 100 surrogate Annual-Max-RD +-2SE = {R_mean} +- {TwoSE}")
print(f"Historical EFAS correlation was {results_df["R_ymax"][0]}")
m_mean = results_df["m_ymax"][1:].mean()
m_std = results_df["m_ymax"][1:].std()
TwoSE = 2*m_std/math.sqrt(100)
print(f"Mean slope for 100 surrogate Annual-Max-RD +-2SE = {m_mean} +- {TwoSE}")
print(f"Historical EFAS slope was {results_df["m_ymax"][0]}")

annual_max_mean = results_df["annual_max"][1:].mean() 
annual_max_std = results_df["annual_max"][1:].std()
TwoSE = 2*annual_max_std/math.sqrt(100)
annual_max_max = results_df["annual_max"][1:].max()
print(f"Mean RD for 100 surrogate Annual-Max-RD  +-2SE = {annual_max_mean} +- {TwoSE}")
print(f"MAX RD for 100 surrogate Annual-Max-RD= {annual_max_max}")
print(f"Historical EFAS maximum was {results_df["annual_max"][0]}")


plt.figure(figsize=(10, 6))
plt.vlines(results_df["annual_date"][1:], ymin=0, ymax=results_df["annual_max"][1:],  linestyle='-', label="Surrogate Annual Max", linewidth=2)
plt.vlines(results_df["annual_date"][0], ymin=0, ymax=results_df["annual_max"][0],  linestyle='-', label="Historical EFAS", color="red", linewidth=2)

# Personalizzazione del grafico
plt.title("100 surrogate annual maxima & historical maximum", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Discharge [m$^3$/s]", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
#plt.legend(loc="upper right", fontsize=12)
#plt.legend(loc="upper left", fontsize=12)
plt.legend(loc="best", fontsize=12)
plt.xticks(rotation=45)  # Rotazione delle date per leggibilità

# save figure
if HazardDef == "gp" :
    gp_name = tlabel
else :
    gp_name = HazardDef
#figname=f"{figdir}/{gp_name}_annual_maximum_surrogate_100_members.png"
figname=f"{figdir}/{gp_name}_annual_maximum_surrogate_100_members.pdf"
plt.savefig(figname)
print("Plotted", figname)

#print(results_df["annual_date"].values)
print(f"Earliest maximum: {results_df["annual_date"].values.min()}. Most recent maximum: {results_df["annual_date"].values.max()}")
print(results_df[results_df["annual_date"]==results_df["annual_date"].values.max()])

# adding distro of m and R together:
fig,axs=plt.subplots(2,1,figsize=(8,10))
# Adding a distribution of m slopes:
#plt.figure(figsize=(8, 5))

# Estrazione dei dati di slope
data_example = results_df["m_ymax"][1:]  # surrogate_maxima.sel(time=str(year_example))["dis24"].values.flatten()

# Istogramma manuale
counts, bins, patches = axs[0].hist(data_example, bins=20, color="lightblue", edgecolor="black", alpha=0.7)

# Calcolo dei quantili e statistiche principali
q25, q75 = np.percentile(data_example, [25, 75])
mean_val = np.mean(data_example)
max_val = np.max(data_example)
std_val = np.std(data_example, ddof=1)
skew_val = skew(data_example)
print(f"slope m: mean={mean_val}, median={np.percentile(data_example, 50)}, std={std_val}, skewness={skew_val}")

# Aggiunta delle linee verticali
axs[0].axvline(results_df["m_ymax"][0], color='red', linestyle='-', linewidth=2, label=f"EFAS historical")
#plt.axvline(q25, color='cyan', linestyle='--', label="25$^{th}$: "+f"{q25:.2f}")
#plt.axvline(q75, color='blue', linestyle='--', label="75$^{th}$: "+f"{q75:.2f}")
axs[0].axvline(mean_val, color='blue', linestyle='-', label=f"Mean: {mean_val:.2f}")
#plt.axvline(max_val, color='black', linestyle=':', label=f"Max: {max_val:.2f}")
axs[0].axvline(mean_val - std_val, color='green', linestyle='-.', label=f"Mean - 1SD: {(mean_val - std_val):.2f}")
axs[0].axvline(mean_val + std_val, color='green', linestyle='-.', label=f"Mean + 1SD: {(mean_val + std_val):.2f}")

# Personalizzazione del grafico
axs[0].set_xlabel("Slope of linear fit m []")
axs[0].set_ylabel("Frequecy")
axs[0].set_title(f"Distribution of 100 surrogate m slopes in {gp_name}")
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.6)

# Estrazione dei dati di slope
data_example = results_df["R_ymax"][1:]  # surrogate_maxima.sel(time=str(year_example))["dis24"].values.flatten()

# Istogramma manuale
counts, bins, patches = axs[1].hist(data_example, bins=20, color="lightblue", edgecolor="black", alpha=0.7)

# Calcolo dei quantili e statistiche principali
q25, q75 = np.percentile(data_example, [25, 75])
mean_val = np.mean(data_example)
max_val = np.max(data_example)
std_val = np.std(data_example, ddof=1)
skew_val = skew(data_example)
print(f"correlations R: mean={mean_val}, median={np.percentile(data_example, 50)}, std={std_val}, skewness={skew_val}")

# Aggiunta delle linee verticali
axs[1].axvline(results_df["R_ymax"][0], color='red', linestyle='-', linewidth=2, label=f"EFAS historical")
#plt.axvline(q25, color='cyan', linestyle='--', label="25$^{th}$: "+f"{q25:.2f}")
#plt.axvline(q75, color='blue', linestyle='--', label="75$^{th}$: "+f"{q75:.2f}")
axs[1].axvline(mean_val, color='blue', linestyle='-', label=f"Mean: {mean_val:.2f}")
#plt.axvline(max_val, color='black', linestyle=':', label=f"Max: {max_val:.2f}")
axs[1].axvline(mean_val - std_val, color='green', linestyle='-.', label=f"Mean - 1SD: {(mean_val - std_val):.2f}")
axs[1].axvline(mean_val + std_val, color='green', linestyle='-.', label=f"Mean + 1SD: {(mean_val + std_val):.2f}")

if HazardDef == "gp" :
    gp_name = tlabel
else :
    gp_name = HazardDef
#figname=f"{figdir}/{gp_name}_m_distro_surrogate_100_members.png"
figname=f"{figdir}/{gp_name}_m_and_R_distro_for_annual_maximum_surrogate_100_members.pdf"

# Personalizzazione del grafico
axs[1].set_xlabel("Correlation of linear fit R []")
axs[1].set_ylabel("Frequecy")
axs[1].set_title(f"Distribution of 100 surrogate R correlations in {gp_name}")
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.6)

# save figure
plt.savefig(figname)
plt.close()
print("Plotted", figname)


#************************************************
# Here start the block to compare RD Intensities:

# This block is relatively slow: it takes some half minute to run.

# First reading the historical EFAS maxima:
histo_maxima = xr.open_mfdataset(fname_gp_ymax)
histo_maxima.close()
print(histo_maxima)
print(histo_maxima.time)


# Analyzing the 2000-2023 (24 years) annual maximum for 100 surrogates series: 

#name_gp_ymax_sur = f"{EFAS_surro}/{watershed.capitalize()}/{m}/efas_surrogate_{watershed}_{tlabel}_{m}_00_masked_annual_maxima.nc"
mon = ["Apr", "May", "Jun", "Jul"]
ensemble_data = []   # simple array

for m in mon:
    file_list = []
    for mem in range(0, 25):
        mem = str(mem).zfill(2)
        fname = f"{EFAS_surro}/{watershed.capitalize()}/{m}/efas_surrogate_{watershed}_{tlabel}_{m}_{mem}_masked_annual_maxima.nc"

        if os.path.exists(fname):
            file_list.append(fname)
        else:
            print(fname, "does not exist!")

    # Loading 25 files of this month in parallel and with a single concat!
    if file_list:
        datasets = xr.open_mfdataset(file_list, combine="nested", concat_dim="ensemble", parallel=True)
        ensemble_data.append(datasets)    # should not work for huge datasets
        datasets.load()
        datasets.close()  # otherwise the nc files remains locked and give permission errors


 # Concat only once:
surrogate_maxima = xr.concat(ensemble_data, dim="ensemble")
       
#surrogate_maxima.load()
surrogate_maxima = surrogate_maxima.compute()

print(surrogate_maxima)


# Need to look at 100 maxima per each year:

print("total cases=", surrogate_maxima.groupby("time.year").count()["dis24"])

#years = surrogate_maxima["time"].dt.year.values
#mean_annual_maxima = surrogate_maxima.groupby("time.year").mean(dim=["ensemble"], skipna=True)
#mean_annual_maxima =surrogate_maxima.resample(time="1Y").mean(dim="ensemble", skipna=True)
mean_annual_maxima = surrogate_maxima.groupby("time.year").mean(dim=["time", "ensemble"], skipna=True)
#print("mean_annual_maxima", mean_annual_maxima.shape)
#print("25 years?", surrogate_maxima["time"].dt.year.values)
print("24 years?", mean_annual_maxima["year"].values)

mean_annual_maxima_da = mean_annual_maxima["dis24"]
print("mean_annual_maxima_da",mean_annual_maxima_da.shape)
unique_years = mean_annual_maxima_da["year"].values # Convert to DataArray
print(unique_years)

max_annual_maxima = surrogate_maxima.groupby("time.year").max(dim=["time", "ensemble"], skipna=True)
#max_annual_maxima = surrogate_maxima.groupby("time.year").max(dim=["ensemble"], skipna=True)
max_annual_maxima_da = max_annual_maxima["dis24"]

std_annual_maxima = surrogate_maxima.groupby("time.year").std(dim=["time", "ensemble"], skipna=True)
#std_annual_maxima = surrogate_maxima.groupby("time.year").std(dim=["ensemble"], skipna=True)
std_annual_maxima_da = std_annual_maxima["dis24"]

percentile_95_annual_maxima = surrogate_maxima.groupby("time.year").quantile(0.95, dim=["time", "ensemble"], skipna=True)
percentile_95_annual_maxima_da = percentile_95_annual_maxima["dis24"]

#mean_annual_maxima = surrogate_maxima.groupby("time.year").apply(lambda x: x.mean(dim="ensemble"))
#print(mean_annual_maxima.mean().values)
#print("Qui", surrogate_maxima[disname].resample(time='1YE').mean(dim='time'))
#mean_annual_maxima_da = mean_annual_maxima.to_array().isel(variable=0)  # Converte in DataArray
print("mean=", mean_annual_maxima_da)
print("max=", max_annual_maxima_da)
#surrogate_maxima.sel(['time'].dt.year.values[:-1]

'''
plt.figure(figsize=(10, 5))
plt.plot(unique_years, max_annual_maxima_da, marker='o', linestyle='-', color='r', label="max 100 annual-maxima")
plt.plot(unique_years, percentile_95_annual_maxima_da, marker='o', linestyle='-', color='orange', label="95$^{th}$ 100 annual-maxima")
plt.plot(unique_years, mean_annual_maxima_da, marker='o', linestyle='-', color='b', label="mean 100 annual-maxima")
plt.vlines(unique_years, ymin=mean_annual_maxima_da-std_annual_maxima_da, ymax=mean_annual_maxima_da+std_annual_maxima_da, linestyle='-', color='c', label="$\pm$1SD 100 annual-maxima")
#plt.vlines(histo_annual_maxima_times[6:], ymin=0, ymax=histo_annual_maxima[6:], linewidth=2, color="grey", label="historical annual-max")
plt.plot(histo_annual_maxima_times[8:], histo_annual_maxima[8:], marker='+', linestyle='-', linewidth=2, color="grey", label="historical annual-max")
#lt.plot(years, mean_annual_maxima.mean().values, marker='o', linestyle='-', color='b', label="Valore Medio")
# Etichette e titolo

plt.grid(True)
plt.title("100 surrogate annual-maxima statistics & historical maximum", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Discharge [m$^3$/s]", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
#plt.legend(loc="upper center", fontsize=10, framealpha=0.5)
plt.legend(loc="upper center",  bbox_to_anchor=(0.6,1.01), fontsize=10, framealpha=0.5)
#plt.legend(loc="upper left",  fontsize=10, framealpha=0.5)
plt.xticks(rotation=45)  # Rotazione delle date per leggibilità

# save figure
if HazardDef == "gp" :
    gp_name = tlabel
else :
    gp_name = HazardDef
figname=f"{figdir}/{gp_name}_annual_maximum_surrogate_statistics.pdf"
plt.savefig(figname)
print("Plotted", figname)
'''



# SD is very low compared to the difference between Max and mean.
# Comparing SD with interquantile difference:
q25 = surrogate_maxima.groupby("time.year").quantile(0.25, dim=["time", "ensemble"], skipna=True)
q75 = surrogate_maxima.groupby("time.year").quantile(0.75, dim=["time", "ensemble"], skipna=True)
iqr = q75 - q25
print("InterQuantile 75-25=", iqr["dis24"])
iqr_da = iqr["dis24"]

median_annual_maxima = surrogate_maxima.groupby("time.year").quantile(0.50, dim=["time", "ensemble"], skipna=True)
median_annual_maxima_da = median_annual_maxima["dis24"]


plt.figure(figsize=(10, 5))
plt.plot(unique_years, max_annual_maxima_da, marker='o', linestyle='-', color='r', label="max 100 annual-maxima")
plt.plot(unique_years, percentile_95_annual_maxima_da, marker='o', linestyle='-', color='orange', label="95$^{th}$ 100 annual-maxima")
plt.plot(unique_years, median_annual_maxima_da, marker='o', linestyle='-', color='b', label="median 100 annual-maxima")
plt.vlines(unique_years, ymin=q25["dis24"], ymax=q75["dis24"], linestyle='-', linewidth=3, color='c',  alpha=0.5, label="25$^{th}$-75$^{th}$ 100 annual-maxima")
#plt.vlines(histo_annual_maxima_times[6:], ymin=0, ymax=histo_annual_maxima[6:], linewidth=2, color="grey", label="historical annual-max")
plt.plot(histo_annual_maxima_times[8:], histo_annual_maxima[8:], marker='+', linestyle='-', linewidth=2, color="grey", label="historical annual-max")
#lt.plot(years, mean_annual_maxima.mean().values, marker='o', linestyle='-', color='b', label="Valore Medio")
# Etichette e titolo

plt.grid(True)
plt.title("100 surrogate annual-maxima statistics & historical maximum", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Discharge [m$^3$/s]", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
#plt.legend(loc="upper center", fontsize=10, framealpha=0.5)
plt.legend(loc="upper center",  bbox_to_anchor=(0.6,1.01), fontsize=10, framealpha=0.5)
#plt.legend(loc="upper left",  fontsize=10, framealpha=0.5)
plt.xticks(rotation=45)  # Rotazione delle date per leggibilità

# save figure
if HazardDef == "gp" :
    gp_name = tlabel
else :
    gp_name = HazardDef
figname=f"{figdir}/{gp_name}_annual_maximum_surrogate_statistics_quantile.pdf"
plt.savefig(figname)
print("Plotted", figname)


# Verifica per il 2019:
#import numpy as np
#import matplotlib.pyplot as plt

# Year with historical maximum
year_example = 2019  

# Estrazione dei dati per l'anno selezionato
data_example = surrogate_maxima.sel(time=str(year_example))["dis24"].values.flatten()

# Rimozione dei valori NaN
data_example = data_example[~np.isnan(data_example)]

# Creazione della figura
plt.figure(figsize=(8, 5))

# Istogramma manuale
counts, bins, patches = plt.hist(data_example, bins=20, color="lightblue", edgecolor="black", alpha=0.7)

# Calcolo dei quantili e statistiche principali
q25, q75 = np.percentile(data_example, [25, 75])
mean_val = np.mean(data_example)
max_val = np.max(data_example)
std_val = np.std(data_example, ddof=1)
skew_val = skew(data_example)
print(f"{year_example} annual max RD: mean={mean_val}, median={np.percentile(data_example, 50)}, std={std_val}, skewness={skew_val}")


# Aggiunta delle linee verticali
plt.axvline(q25, color='cyan', linestyle='--', label="25$^{th}$: "+f"{q25:.2f}")
plt.axvline(q75, color='blue', linestyle='--', label="75$^{th}$: "+f"{q75:.2f}")
plt.axvline(mean_val, color='red', linestyle='-', label=f"Mean: {mean_val:.2f}")
plt.axvline(max_val, color='black', linestyle=':', label=f"Max: {max_val:.2f}")
plt.axvline(mean_val - std_val, color='orange', linestyle='-.', label=f"Mean - 1SD: {(mean_val - std_val):.2f}")
plt.axvline(mean_val + std_val, color='orange', linestyle='-.', label=f"Mean + 1SD: {(mean_val + std_val):.2f}")

# Personalizzazione del grafico
plt.xlabel("Discharge [m$^³$/s]")
plt.ylabel("Frequecy")
plt.title(f"Distribution of 100 surrogate maxima during year {year_example}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
#plt.show()

# save figure
if HazardDef == "gp" :
    gp_name = tlabel
else :
    gp_name = HazardDef
figname=f"{figdir}/{gp_name}_annual_maximum_surrogate_statistics_{year_example}.pdf"
plt.savefig(figname)
print("Plotted", figname)


#**********************************
# Statistics of the seasona cycles:

# Reading statistics describing all seasonal cycles.

# First reading the historical EFAS maxima:
# Be carefull: station name is missing in file name of seasonal cycle!!!
cycle_fname = f"{cycle_dir}efas_{watershed}_seasonalcycle_win{window_size}_pctx50.nc"
# efas_panaro_seasonalcycle_win31_pctx50.nc

histo_cycle = xr.open_mfdataset(cycle_fname)
histo_cycle.close()
print(histo_cycle)
print(histo_cycle.time)


# Analyzing the 2000-2023 (24 years) annual maximum for 100 surrogates series: 
mon = ["Apr", "May", "Jun", "Jul"]
ensemble_data = []   # simple array

for m in mon:
    file_list = []
    for mem in range(0, 25):
        mem = str(mem).zfill(2)
        fname = f"{cycle_dir}efas_{watershed}_seasonalcycle_win{window_size}_pctx50_{m}_{mem}.nc"
        #f"{EFAS_surro}/{watershed.capitalize()}/{m}/efas_surrogate_{watershed}_{tlabel}_{m}_{mem}_masked_annual_maxima.nc"

        if os.path.exists(fname):
            file_list.append(fname)
        else:
            print(fname, "does not exist!")

    # Loading 25 files of this month in parallel and with a single concat!
    if file_list:
        datasets = xr.open_mfdataset(file_list, combine="nested", concat_dim="ensemble", parallel=True)
        ensemble_data.append(datasets)    # should not work for huge datasets
        datasets.load()
        datasets.close()

 # Concat only once:
surrogate_cycles = xr.concat(ensemble_data, dim="ensemble")
       
#surrogate_maxima.load()
surrogate_cycles = surrogate_cycles.compute()


print(surrogate_cycles)

# Need to look at 100 timeseries per each day:

#print("total cases=", surrogate_cycles.groupby("time.day").count()["smoothed_means"])
print("total cases=", surrogate_cycles.groupby("dayofyear").count()["smoothed_means"])
#mean_annual_cycle = surrogate_cycles.groupby("dayofyear").mean(dim=["smoothed_means", "ensemble"], skipna=True)
mean_annual_cycle = surrogate_cycles["smoothed_means"].mean(dim="ensemble", skipna=True)
print(mean_annual_cycle)
q25 = surrogate_cycles["smoothed_means"].quantile(0.25, dim="ensemble", skipna=True)
q75 = surrogate_cycles["smoothed_means"].quantile(0.75, dim="ensemble", skipna=True)

mean_annual_cycle_max = surrogate_cycles["smoothed_max"].mean(dim="ensemble", skipna=True)
q25_max = surrogate_cycles["smoothed_max"].quantile(0.25, dim="ensemble", skipna=True)
q75_max = surrogate_cycles["smoothed_max"].quantile(0.75, dim="ensemble", skipna=True)

mean_annual_cycle_p90 = surrogate_cycles["smoothed_p90"].mean(dim="ensemble", skipna=True)
q25_p90 = surrogate_cycles["smoothed_p90"].quantile(0.25, dim="ensemble", skipna=True)
q75_p90 = surrogate_cycles["smoothed_p90"].quantile(0.75, dim="ensemble", skipna=True)

mean_annual_cycle_p50 = surrogate_cycles["smoothed_p50"].mean(dim="ensemble", skipna=True)
q25_p50 = surrogate_cycles["smoothed_p50"].quantile(0.25, dim="ensemble", skipna=True)
q75_p50 = surrogate_cycles["smoothed_p50"].quantile(0.75, dim="ensemble", skipna=True)


# plot surrogate stats

# day corresponding to first day of each month
# length of months in days
mons = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_lengths = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
# Calcola il giorno dell'anno del primo giorno di ogni mese
# accumulate
month_starts = np.cumsum(month_lengths) - month_lengths
#print(len(xds_stats_clim['daily_means']), len(xds_stats_clim['smoothed_p90']))  # 366

plt.figure
fig, axs = plt.subplots(2, 1, figsize=(7, 10))  # 2 rows, 1 column

histo_cycle["smoothed_means"].plot(ax=axs[0], linestyle='solid', color="blue", label=f'historical $\pm${window_size//2}d mean')
histo_cycle["smoothed_max"].plot(ax=axs[0], linestyle='solid', color="purple", label=f'historical $\pm${window_size//2}d max')
histo_cycle["smoothed_p90"].plot(ax=axs[0], linestyle='solid', color="red", label=f'historical $\pm${window_size//2}d 90%')
histo_cycle["smoothed_p50"].plot(ax=axs[0], linestyle='solid', color="green", label=f'historical $\pm${window_size//2}d median')

#xds_stats_clim_sur['daily_means'].plot(ax=axs[0], label='Daily mean')
mean_annual_cycle.plot(ax=axs[0], linestyle='dashed', color="blue", label=f'mean of 100 $\pm${window_size//2}d mean')
# Aggiunta delle linee verticali
#axs[0].axvline(q25, color='cyan', linestyle='--', label="25$^{th}$: "+f"{q25:.2f}")
#axs[0].axvline(q75, color='blue', linestyle='--', label="75$^{th}$: "+f"{q75:.2f}")
#axs[0].fill_between(mean_annual_cycle.dayofyear, q25, q75, color="lightblue", alpha=0.5, label="IQR (25-75%) mean")
axs[0].fill_between(mean_annual_cycle.dayofyear, q25, q75, color="lightblue", alpha=0.5, label=None)

mean_annual_cycle_max.plot(ax=axs[0], linestyle='dashed', color="purple", label=f'mean of 100 $\pm${window_size//2}d max')
#axs[0].fill_between(mean_annual_cycle_max.dayofyear, q25_max, q75_max, color="plum", alpha=0.5, label="IQR (25-75%) 50%")
axs[0].fill_between(mean_annual_cycle_max.dayofyear, q25_max, q75_max, color="plum", alpha=0.5, label=None)

mean_annual_cycle_p90.plot(ax=axs[0], linestyle='dashed', color="red", label=f'mean of 100 $\pm${window_size//2}d 90%')
#axs[0].fill_between(mean_annual_cycle_p90.dayofyear, q25_p90, q75_p90, color="lightcoral", alpha=0.5, label="IQR (25-75%) 50%")
axs[0].fill_between(mean_annual_cycle_p90.dayofyear, q25_p90, q75_p90, color="lightcoral", alpha=0.5, label=None)

mean_annual_cycle_p50.plot(ax=axs[0], linestyle='dashed', color="green", label=f'mean of 100 $\pm${window_size//2}d median')
#axs[0].fill_between(mean_annual_cycle_p50.dayofyear, q25_p50, q75_p50, color="palegreen", alpha=0.5, label="IQR (25-75%) median")
axs[0].fill_between(mean_annual_cycle_p50.dayofyear, q25_p50, q75_p50, color="palegreen", alpha=0.5, label=None)

# add a fuzzy filled field to have a grey box in the legend
axs[0].fill_between([], [], [], color="gray", alpha=0.5, label="IQR (25-75%) surrogates")

axs[0].set_title(f'{gp_name} daily seasonal cycle: smoothed mean and perc. for hist. and surrogates')
axs[0].set_xticks(month_starts)
axs[0].set_xticklabels([])
#axs[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
for i, mo in enumerate(mons):
    axs[0].text(month_starts[i]+15, -0.05, mo, ha='center', va='center', transform=axs[0].get_xaxis_transform())
#axs[0].legend(['Daily mean', 'Smoothed mean'])
axs[0].legend(prop={'size': 8})
axs[0].set_xlabel('')
axs[0].set_ylabel('Discharge [m$^3$/s]')

# Second figure:
tx=1

histo_cycle["monthly_max_max"].plot(ax=axs[tx], linestyle='solid', color="blue", label=f'historical monthly max')
histo_cycle["monthly_max_p90"].plot(ax=axs[tx], linestyle='solid', color="Orange", label=f'historical monthly 90%')
histo_cycle["monthly_max_p50"].plot(ax=axs[tx], linestyle='solid', color="green", label=f'historical monthly 50%')

surrogate_cycles["monthly_max_max"].mean(dim="ensemble", skipna=True).plot(ax=axs[tx], linestyle='dashed', color="blue", label=f'mean surrogate max')
q25_mon_max = surrogate_cycles["monthly_max_max"].quantile(0.25, dim="ensemble", skipna=True)
q75_mon_max = surrogate_cycles["monthly_max_max"].quantile(0.75, dim="ensemble", skipna=True)
axs[tx].fill_between(np.arange(1, 13, 1), q25_mon_max, q75_mon_max, color="lightblue", alpha=0.5, label=None)

surrogate_cycles["monthly_max_p90"].mean(dim="ensemble", skipna=True).plot(ax=axs[tx], linestyle='dashed', color="Orange", label=f'mean surrogate 90%')
q25_mon_max = surrogate_cycles["monthly_max_p90"].quantile(0.25, dim="ensemble", skipna=True)
q75_mon_max = surrogate_cycles["monthly_max_p90"].quantile(0.75, dim="ensemble", skipna=True)
axs[tx].fill_between(np.arange(1, 13, 1), q25_mon_max, q75_mon_max, color="peachpuff", alpha=0.5, label=None)

surrogate_cycles["monthly_max_p50"].mean(dim="ensemble", skipna=True).plot(ax=axs[tx], linestyle='dashed', color="green", label=f'mean surrogate 50%')
q25_mon_max = surrogate_cycles["monthly_max_p50"].quantile(0.25, dim="ensemble", skipna=True)
q75_mon_max = surrogate_cycles["monthly_max_p50"].quantile(0.75, dim="ensemble", skipna=True)
axs[tx].fill_between(np.arange(1, 13, 1), q25_mon_max, q75_mon_max, color="palegreen", alpha=0.5, label=None)

# add a fuzzy filled field to have a grey box in the legend
axs[tx].fill_between([], [], [], color="gray", alpha=0.5, label="IQR (25-75%) surrogates")

axs[tx].set_title(f'{gp_name} hist. and mean-value surrogates of unsmoothed monthly extremes')
axs[tx].set_xticks(np.arange(1, 13, 1))
axs[tx].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
#axs[tx].legend(['max','90%', '50%'])
axs[tx].legend(prop={'size': 8})
axs[tx].set_xlabel('')
axs[tx].set_ylabel('Discharge [m$^3$/s]')
# legend


# save figure
if HazardDef == "gp" :
    gp_name = tlabel
else :
    gp_name = HazardDef
figname=f"{figdir}/{gp_name}_seasonal_cycle_surrogate_statistics.pdf"
plt.savefig(figname)
print("Saved", figname)



# Adding a new figure with unsmoothed daily mean values to estimate maximum spread:

plt.figure(figsize=(8, 5))

surrogate_cycles["daily_means"].quantile(0.50, dim="ensemble", skipna=True).plot(linestyle='solid', color="blue", alpha=0.7, label=f'median of unsmooth surrogate mean')
q25_mon_max = surrogate_cycles["daily_means"].quantile(0.25, dim="ensemble", skipna=True)
q75_mon_max = surrogate_cycles["daily_means"].quantile(0.75, dim="ensemble", skipna=True)
plt.fill_between(histo_cycle["daily_means"].dayofyear, q25_mon_max, q75_mon_max, color="lightblue", alpha=0.5, label="IQR (25-75%) surrogate mean")
surrogate_cycles["daily_means"].quantile(0.95, dim="ensemble", skipna=True).plot(linestyle='solid', color="orange", alpha=0.7, label=f'95% of 100 surrogate mean')
surrogate_cycles["daily_means"].max(dim="ensemble", skipna=True).plot(linestyle='solid', color="red", alpha=0.7, label=f'max of 100 surrogate mean')
histo_cycle["daily_means"].plot(linestyle='solid', color="grey", label=f'historical unsmoothed daily mean')

plt.title(f'{gp_name} daily mean seasonal cycle: hist. and perc. of unsmoothed surrogates')
plt.xticks(month_starts)
#plt.set_xticklabels([])
plt.gca().set_xticklabels([])
#axs[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
shift = -5
if watershed == "panaro" :
    shift = -1
if watershed == "reno" :
    shift = -3
for i, mo in enumerate(mons):
   plt.text(month_starts[i]+15, shift, mo, ha='center', va='center')  # , transform=get_xaxis_transform())
plt.legend(prop={'size': 8})
plt.xlabel('')
plt.ylabel('Discharge [m$^3$/s]')
#plt.show()

figname=f"{figdir}/{gp_name}_seasonal_cycle_daily-mean_surrogate_statistics.pdf"
plt.savefig(figname)
print("Saved", figname)


# Adding a new figure with unsmoothed daily mean values to estimate maximum spread:
# SON QUA

plt.figure(figsize=(8, 5))

surrogate_cycles["daily_max"].quantile(0.50, dim="ensemble", skipna=True).plot(linestyle='solid', color="blue", alpha=0.7, label=f'median of unsmooth surrogate max')
q25_mon_max = surrogate_cycles["daily_max"].quantile(0.25, dim="ensemble", skipna=True)
q75_mon_max = surrogate_cycles["daily_max"].quantile(0.75, dim="ensemble", skipna=True)
plt.fill_between(histo_cycle["daily_max"].dayofyear, q25_mon_max, q75_mon_max, color="lightblue", alpha=0.5, label="IQR (25-75%) surrogate max")
surrogate_cycles["daily_max"].quantile(0.95, dim="ensemble", skipna=True).plot(linestyle='solid', color="orange", alpha=0.7, label=f'95% of 100 surrogate max')
surrogate_cycles["daily_max"].max(dim="ensemble", skipna=True).plot(linestyle='solid', color="red", alpha=0.7, label=f'max of 100 surrogate max')
histo_cycle["daily_max"].plot(linestyle='solid', color="grey", label=f'historical unsmoothed daily max')

plt.title(f'{gp_name} daily max seasonal cycle: hist. and perc. of unsmoothed surrogates')
plt.xticks(month_starts)
#plt.set_xticklabels([])
plt.gca().set_xticklabels([])
#axs[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
shift = -100
if watershed == "panaro" :
    shift = -50
if watershed == "reno" :
    shift = -25
for i, mo in enumerate(mons):
   plt.text(month_starts[i]+15, shift, mo, ha='center', va='center')  # , transform=get_xaxis_transform())
plt.legend(prop={'size': 8})
plt.xlabel('')
plt.ylabel('Discharge [m$^3$/s]')
#plt.show()

figname=f"{figdir}/{gp_name}_seasonal_cycle_daily-max_surrogate_statistics.pdf"
plt.savefig(figname)
print("Saved", figname)




print("The END!")