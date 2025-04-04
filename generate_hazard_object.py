import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import dask 
import cartopy.crs as ccrs 
import pandas as pd 
from datetime import timedelta 
from scipy.stats import genextreme 
import xesmf as xe


 
# target region selection
#########################
reg='Turia'##############
#########################

# region-specific parameters

if reg=='Panaro_bom':
      latmin,latmax,lonmin,lonmax=44,45.1,10.55,11.5
      plat, plon, plon_360 =44.72445000000296, 11.041666653786793,11.041666653786793
if reg =='Turia':
      latmin,latmax,lonmin,lonmax=39,41,-2,0
      plat, plon = 39.51777,359.49594
      plon_360 = plon-360



# 1. fit GEV ditribution on 2D pooled annual maxima (UNSEEN)


def analyze_gev_over_grid(surr_dict_amax_2D_masked_pooled, max_return_period=None, outlier_threshold_factor=20):

    """
    fits a GEV distribution on each grid point for all events (timestep) in the input dataset.

    Input: a dictionary containing annual maxima events (time, latitude, lontitude) mapped to different keys 
    representing different surrogate series. 

     
    """

    def process_point(data):
        """
        processes a single grid point 
        """
        data = np.asarray(data)
        #ignore nans over land
        data_clean = data[~np.isnan(data)]

        #after removing nans, the data should have enough values
        if data_clean.size == 0 or np.nanmax(data_clean) < 2:
            print('not enough valid data, converted to nan')
            return np.full_like(data, np.nan)

        #check for unrealistic outliers based on the distance between the max value and the 99th percentile
        max_val = np.nanmax(data_clean)
        percentile_99 = np.nanpercentile(data_clean, 99)

        if max_val > outlier_threshold_factor * percentile_99:
            print('unrealistic outlier')
            return np.full_like(data, np.nan)  # Mask as outlier

        try:
            gev_shape,gev_loc,gev_scale = genextreme.fit(data_clean)

            if gev_scale <= 0:
                print('scale param. too low, converting to nan')
                return np.full_like(data, np.nan)

            cdf_values = genextreme.cdf(data_clean, c=gev_shape, loc=gev_loc, scale=gev_scale)
            return_periods = 1 / (1 - cdf_values)

            # copy data structure of input
            output_array = np.full_like(data, np.nan)
            # add output data to output array
            output_array[~np.isnan(data)] = return_periods 

            # OPTIONAL: cap the return period below a threshold of 1000 (unrealistically high values are turned to nan)
            #output_array[output_array > max_return_period] = np.nan
            return output_array

        except Exception as e:
            print(f"GEV fit failed, returning nans: {e}")
            return np.full_like(data, np.nan)

    #apply the function to every grid point
    rp_array = xr.apply_ufunc(
        process_point,
        surr_dict_amax_2D_masked_pooled,
        input_core_dims=[['pooled_time']],
        output_core_dims=[['pooled_time']],
        exclude_dims=set(), 
        dask='parallelized',
        output_dtypes=[surr_dict_amax_2D_masked_pooled.dtype],
        
    )
    rp_array['pooled_time'] = surr_dict_amax_2D_masked_pooled['pooled_time']# keep dates as original file
    return rp_array



# 2. Load flood hazard maps at different return periods (Dottori et al. 2016).
# Note: the data should be downloaded from https://data.jrc.ec.europa.eu/dataset/1d128b6c-a4ee-4858-9e34-6210707f3c81
# and converted from .tif to .nc format. 

def load_floodmaps(ds): 
    """
    load JRC flood maps from files separated by return period [10,20,30,40,50,75,100,200,500]

    """
    
    filename = os.path.basename(ds.encoding['source'])
    rp = int(filename.split('_RP')[1].split('_')[0])
    
    # add return_period coordinate
    ds = ds.expand_dims(return_period=[rp])
    
    return ds

dirpath=dirpath

# locate all .nc files converted from .tif to .nc (9 files corresponding to 9 RP)
file_pattern = os.path.join(dirpath, "*.nc")

#open all files in a single dataset with a return period coordinate
fmaps = xr.open_mfdataset(file_pattern, 
                          combine='nested', 
                          concat_dim='return_period', 
                          preprocess=load_floodmaps)

# sort dataset by return_period
fmaps = fmaps.sortby('return_period')

# cut to region of interest
fmaps=fmaps.where((fmaps.lat>latmin)&(fmaps.lat<latmax)&(fmaps.lon>lonmin)&(fmaps.lon<lonmax),drop=True)


# 3. Regrid the return period dataset (2D) to the high-resolution grid of flood hazard maps

def regrid_to_hr(fmaps,file_to_regrid):
    """
    regrid return period dataset to high-resolution grid of flood hazard maps
    Input: 
    -flood hazard maps, loaded onto a single dataset with return periods as axis
    - file to regrid, i.e. gridded return periods for each UNSEEN event
    """


    grid_out=fmaps
    lats = grid_out.lat.values
    lons = grid_out.lon.values

    # # creating the destination grid

    grid_lats = np.linspace(lats.max(), lats.min(), len(grid_out.lat)) # descending lats -----> this prevents errors with misaligned arrays
    grid_lons = np.linspace(lons.min(), lons.max(), len(grid_out.lon)) # ascending lons

    new_grid = xr.Dataset({'latitude':(['latitude'],grid_lats), 'longitude':(['longitude'],grid_lons)})
    regridder = xe.Regridder(filetoregrid,grid_out, 'nearest_s2d')
 
    return_period_regrid = regridder(filetoregrid)
    
    #optional: save file 
    return_period_regrid.to_netcdf(ps.path.join(writepath,f'return_period_regrid_{reg}.nc'))

    return return_period_regrid


# 4. interpolate flood heights


def interpolate_flood_height_loop(fmaps, return_period_regrid):

    """
    performs a linear interpolation between the flood heights at different T in "fmaps"
    and the gridded return periods of each UNSEEN event in "return_period_regrid"

    returns a gridded dataset containing estimated flood heights for each UNSEEN event
    output unit: meter

    """

    # arrays for interpolation: 
    flood_heights = fmaps['flood_height'].values#(return_period, lat, lon) -- array of known height from JRC maps
    return_periods = fmaps['return_period'].values#(return_period,) -- array of T [10,20,30,40,50,75,100,200,500]
    rp_regrid = return_period_regrid['return_period'].values #(time, lat, lon) -- 2D T array, calculated with GEV fits

    #initialize array of interpolated heights
    fheight = np.full_like(rp_regrid, np.nan, dtype=np.float32)

    #loop over lats and lons
    for i in range(rp_regrid.shape[1]):  
        for j in range(rp_regrid.shape[2]): 
            if not np.isnan(rp_regrid[:, i, j]).all():# if a grid point is nan (ie land), skip inteprolation

                fheight[:, i, j] = np.interp( #using np.interp
                    rp_regrid[:, i, j],  # T at this grid point (for all timesteps)
                    return_periods,      # known return periods from fmaps
                    flood_heights[:, i, j], # heights at this grid point (for all return periods)
                    left=np.nan,# make it nan when over boundaries 
                    right=np.nan,
                )

    # dataset of interpolated flood heights at different timesteps (UNSEEN events)
    interpolated_fheight = return_period_regrid.copy()
    interpolated_fheight['fheight'] = (('time', 'lat', 'lon'), fheight)

    # OPTIONAL save file
    
    interpolated_fheight.to_netcdf(os.path.join(writepath,f'interpolated_flood_heights_{reg}.nc'))

    return interpolated_fheight






# Example Usage


reg = reg 
dirpath = dirpath # path to flood hazard maps data  
writepath = writepath # path to write/save output files

rp_output = analyze_gev_over_grid(surr_dict_amax_2D_masked_pooled) # input array with maxima pooled across surrogates

return_period_regrid=regrid_to_hr(fmaps, filetoregrid)

interpolated_fheight_unseen = interpolate_flood_height_loop(fmaps, return_period_regrid)


   

