import xarray as xr
import numpy as np  
import matplotlib.pyplot as plt  
import os  
import dask  
import pandas as pd  
from datetime import timedelta   
from shapely.geometry import Polygon
import time
# import xesmf as xe

# climada packages

import climada
from climada.hazard import Hazard, Centroids
from climada.entity import LitPop
from climada.entity import Impactfunc, ImpactFuncSet
import climada.entity.exposures.litpop as lp
from climada_petals.entity.impact_funcs.river_flood import ImpRiverFlood, flood_imp_func_set
from climada.engine import Impact, ImpactCalc
from climada_petals.util.constants import RIVER_FLOOD_REGIONS_CSV

  
# target region selection
#########################
reg='Turia'


# region-specific parameters

if reg=='Panaro_bom': 
      latmin,latmax,lonmin,lonmax=44,45.1,10.55,11.5 
      plat, plon, plon_360 =44.72445000000296, 11.041666653786793,11.041666653786793 
if reg =='Turia': 
      latmin,latmax,lonmin,lonmax=39,41,-2,0 
      plat, plon = 39.51777,359.49594 
      plon_360 = plon-360 
 

 #1 Generate the Climada hazard object from gridded interpolated flood heights associated with UNSEEN events


def generate_climada_hazard(heights):

    """
    Generates a hazard object as required by CLIMADA for impact calculation.
    
    Input: array with interpolated flood heights (hazard metric)
    Returns: hazard object

    Additional parameters for hazard object generation:

    - n_cen: number of centroids in the hazard intensity matrix as defined from the latitudes and longitudes of the input data
    - event_names: a list of event names that could be numbers, or strings. 
    - fraction: array of ones with dimensions number of events, number of centroids. Can be prescribed to have different values
    - frequency: array with prescribed frequency of each event. For synthethis or probabilistic  events, the frequency is typically set as 1 for all events as a common assumption.

    """

    lat=heights.lat.values
    lon=heights.lon.values

    n_cen = lat.size*lon.size
    n_ev=heights.time.size

    event_names = np.arange(0,heights.fheight.shape[0],1)

    intensity = np.nan_to_num(h_selected.fheight.values, nan=0).reshape(n_ev, -1)
    intensity = sparse.csr_matrix(intensity)

    fraction = sparse.csr_matrix(np.ones((n_ev, n_cen)))
    from numpy import meshgrid

    lon_2d, lat_2d = meshgrid(lon, lat)
    centroids = Centroids(lat=lat_2d.flatten(), lon=lon_2d.flatten())

    # create hazard object

    haz = Hazard(haz_type='RF',# two letter acronym for climada hazard type river flooding
             
             intensity=intensity, # intensity of events at centroids
             fraction=fraction, # fraction of affected exposures for each event at each centroid
             centroids=centroids,  # default crs used
             units='m',# refers to units of the intensity
             event_id=np.arange(n_ev, dtype=int),#  id of each event 
             event_name=event_names, 
             orig=np.zeros(n_ev, bool),# flag indicating historical events (True) or probabilistic (False)
             frequency=np.ones(n_ev)/n_ev,) 
             
                
    haz.check()

    return haz




# 2. Load exposure data using climada's LitPop module


def load_exposure(reg):

    """
    Load exposure data using the LitPop dataset.
    The function relies on climada built-in functions in the exposure module

    Input: prescribed region, which determines the latitude and longitude bounds used
    to extract the exposure Polygon, and to determine the country for which exposure data is loaded

    2022 is used as reference year for population count (most recent)

    Returns: exposure data in dataframes
    """

    # initiate litpop for a bounding box corresponding to region
    bounds = (lonmin,latmin,lonmax,latmax) 

    shape = Polygon([
                (bounds[0], bounds[3]),
                (bounds[2], bounds[3]),
                (bounds[2], bounds[1]),
                (bounds[0], bounds[1])
                ])


    # get exposure data from litpop corresponding to country and shape of interest

    # the function finds the nation and then crops to shape
    if reg =='Panaro_bom':
        exp_gdp = LitPop.from_shape_and_countries(shape, 'Italy',fin_mode='gdp',reference_year=2022) # specify reference year
        exp_pop = LitPop.from_shape_and_countries(shape, 'Italy',fin_mode='pop',reference_year=2022) 

    if reg =='Turia':
        exp_gdp = LitPop.from_shape_and_countries(shape, 'Spain',fin_mode='gdp',reference_year=2022)
        exp_pop = LitPop.from_shape_and_countries(shape, 'Spain',fin_mode='pop',reference_year=2022) 

    return exp_gdp, exp_pop



#3. Impact calculation


def compute_impact(reg,fun_id=3,exposure,H,haz):

    """
    Compute the impact of hazard events using climada's Impact Calculation module
    Input: 
    - region of interest
    -a function id defining the continental area of interest. This extracts the correct vulnerability curve.
    -the exposure dataframe, eg exp_gdp
    - optional = a step function H providing a threshold flood height below which impact is considered null

    """

    impf_set = flood_imp_func_set()
    impf_EUR = impf_set.get_func(fun_id=fun_id) # this line should be set according to the continental area of hazard
    info = pd.read_csv(RIVER_FLOOD_REGIONS_CSV)

    if reg=='Panaro_bom':
        exposure.gdf['impf_RF']=info.loc[info['ISO']=='ITA','impf_RF'].values[0] # or 3 which corresponds to europe
    if reg =='Turia':
        exposure.gdf['impf_RF']=info.loc[info['ISO']=='ESP','impf_RF'].values[0] # or 3 which corresponds to europe

    # Create a impact function for being affected by flooding
    impf_affected = ImpactFunc.from_step_impf(intensity=(0.0, H, 100.0), impf_id=3, haz_type="RF") #the threshold H can be changes or omitted
    impf_set_affected = ImpactFuncSet([impf_affected])
    impact = ImpactCalc(exposure,impf_set_affected,haz).impact(save_mat=True)

    return impact


# 4. Impact-based event ranking

def imp_ranking(event_names, impact):

    """
    Rank UNSEEn events by estimated impact.
    Input: event_names and impact computed from "compute_impact"

    """

    impact_df=pd.DataFrame(columns=['event','imp_gdp']

    impact_df['event']=event_names
    impact_df['imp_gdp']=impact.at_event)

    impact_ranked = impact_df.sort_values(by='imp_gdp', ascending=False)
   
    return impact_ranked






# Example usage

reg = reg

haz=generate_climada_hazard(heights)

exp_gdp,exp_pop = load_exposure(reg)

H = 0.5

impact = compute_impact(reg,fun_id=3,exp_gdp,H,haz)

event_names = event_names=np.arange(0,input.fheight.shape[0],1)

impact_ranked = imp_ranking(event_names,impact)



