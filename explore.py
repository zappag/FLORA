import xarray as xr
import cartopy.crs as ccrs
#import matplotlib
#matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt

EFASDIR_HIST="/home/zappa/work_big/EFAS/output"

# Open a NetCDF file
fname="{EFASDIR_HIST}/efas_italy_2016.nc"

dataset = xr.open_dataset(f"{EFASDIR_HIST}/efas_italy_2016.nc")

# Print the dataset to see its structure
print(dataset)

# Access variables within the dataset
variable = dataset['dis06']


# Select the data for the specific time step you want to plot
time_step = '2016-11-01T06:00:00'  # Example time step
data_for_time_step = dataset.sel(time=time_step)

# Extract longitude, latitude, and variable data
lon = data_for_time_step['lon']
lat = data_for_time_step['lat']
variable = data_for_time_step['dis06']  # Replace 'your_variable' with the name of your variable

# Plot the map
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Plot the variable on the map
plt.pcolormesh(lon, lat, variable, transform=ccrs.PlateCarree(), cmap='viridis',vmin=0,vmax=25)
plt.xlim(10.5, 12)  # Set longitude range
plt.ylim(44, 45.5)  # Set latitude range

# Add coastlines, gridlines, and title
ax.coastlines()
ax.gridlines()
plt.title(f'Map for {time_step}')

plt.show()


# Close the dataset when you're done
# dataset.close()