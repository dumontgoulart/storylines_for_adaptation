import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs # for map projections
import cartopy.feature as cfeature
import os
os.chdir('D:/paper_4/code')
import xarray as xr
import numpy as np

# Define a function that takes the 'ds' variable as an argument, and optionally other parameters; animation is generated
def create_animation(ds, variable = 'Ptot', threshold=5, file_name='precip_animation.gif'):

    # Create a masked array to mask values below the threshold
    masked_da = ds[variable].where(ds[variable] > threshold)

    # Define a function that creates the base map with cartopy features and returns a figure and an axis object
    def make_figure():
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())    # 
        ax.coastlines(resolution='10m') 
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        return fig, ax

    # Define the vmin, vmax and cmap values for the different variables
    if variable == 'Ptot':
        vmin, vmax, cmap = 0, 10, 'Blues'
    elif variable == 'U10m':
        vmin, vmax, cmap = 10, 30, 'Blues'
    elif variable == 'mslp':
        vmin, vmax, cmap = 970, 1030, 'RdBu'

    # Call the make_figure function and assign the returned values to fig and ax variables
    fig, ax = make_figure()

    # Define a function that draws the data on the map for a given frame (time index) and optionally adds a colorbar
    def draw(frame, add_colorbar):
        # Clear the previous plot
        ax.clear()
        # Re-add the cartopy features
        ax.coastlines(resolution='10m') 
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        # Plot the grid for the current frame
        grid = masked_da.isel(time=frame)
        contour = grid.plot(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', cmap = cmap, add_colorbar=add_colorbar, vmin=vmin, vmax=vmax)
        title = f"Ptot - {str(masked_da.time[frame].values)[:19]}"
        ax.set_title(title)
        return contour

    # Define an init function that calls the draw function for the first frame and adds a colorbar
    def init():
        return draw(0, add_colorbar=True)

    # Define an animate function that calls the draw function for subsequent frames without adding a colorbar
    def animate(frame):
        return draw(frame, add_colorbar=False)

    # Create an animation object using FuncAnimation with the figure, animate, init_func, frames, interval and repeat arguments
    ani = animation.FuncAnimation(fig=fig, func=animate, init_func=init, frames=len(masked_da.time), interval=200, repeat=False)

    # Save the animation as a video file using the writer and fps arguments
    ani.save(file_name)

    plt.close()

    # Return the animation object
    return ani

# Define a function that takes the 'ds' variable as an argument, and optionally other parameters; animation is generated
def create_animation_surge(ds, file_name='surge_animation.gif'):

    # Create a masked array to mask values below the threshold
    if 'x' in ds.coords:
        ds = ds.rename({'x':'station_x_coordinate', 'y':'station_y_coordinate'})
    if 'lat' in ds.coords:
        ds = ds.rename({'lon':'station_x_coordinate', 'lat':'station_y_coordinate'})
    masked_da = ds

    # Define a function that creates the base map with cartopy features and returns a figure and an axis object
    def make_figure():
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())    # 
        ax.coastlines(resolution='10m') 
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        return fig, ax


    # Call the make_figure function and assign the returned values to fig and ax variables
    fig, ax = make_figure()

    # Define a function that draws the data on the map for a given frame (time index) and optionally adds a colorbar
    def draw(frame, add_colorbar):
        # Clear the previous plot
        ax.clear()
        # Re-add the cartopy features
        ax.coastlines(resolution='10m') 
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        # Plot the grid for the current frame
        grid = masked_da.isel(time=frame)
        # contour = grid.plot(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', cmap = 'Blues', vmin=2, vmax=30, add_colorbar=add_colorbar)
        contour = grid.plot.scatter(
                        x='station_x_coordinate', 
                        y='station_y_coordinate', 
                        hue = 'waterlevel',
                        s=100,
                        edgecolor='none',
                        vmin=-2,
                        vmax=2,
                        transform=ccrs.PlateCarree(),
                        ax=ax,
                        cmap = 'RdBu',
                        cbar_kwargs={'shrink':0.6},
                        add_colorbar=add_colorbar);
        
        title = f"Surge - {str(masked_da.time[frame].values)[:19]}"
        ax.set_title(title)
        return contour

    # Define an init function that calls the draw function for the first frame and adds a colorbar
    def init():
        return draw(0, add_colorbar=True)

    # Define an animate function that calls the draw function for subsequent frames without adding a colorbar
    def animate(frame):
        return draw(frame, add_colorbar=False)

    # Create an animation object using FuncAnimation with the figure, animate, init_func, frames, interval and repeat arguments
    ani = animation.FuncAnimation(fig=fig, func=animate, init_func=init, frames=len(masked_da.time), interval=100, repeat=False)

    # Save the animation as a video file using the writer and fps arguments
    ani.save(file_name)

    plt.close()

    # Return the animation object
    return ani

################################################################################
from scipy.ndimage import laplace

storm = 'idai_seas5'
variable = 'tp'
scenario = 'forecast'
# Dict of storms
dict_storms = {
        'xynthia':{'lat':slice(56,38), 'lon':slice(-15,8),'tstart':'2010-02-26','tend':'2010-03-01'}, 
        'sandy':{'lat':slice(52,20), 'lon':slice(-85,-55),'tstart':'2012-10-26','tend':'2012-10-30'},
        'sandy_shifted':{'lat':slice(52,20), 'lon':slice(-85,-55),'tstart':'2012-10-26','tend':'2012-10-30'},
        'sandy_historical':{'lat':slice(52,20), 'lon':slice(-85,-55),'tstart':'2012-10-26','tend':'2012-10-30'},
        'sandy_era5':{'lat':slice(52,20), 'lon':slice(-85,-55),'tstart':'2012-10-26','tend':'2012-10-30'},
        'xaver':{'lat':slice(79.99,40), 'lon':slice(-26,23),'tstart':'2013-12-04','tend':'2013-12-06'},
        'idai':{'lat':slice(-10,-30), 'lon':slice(20,55),'tstart':'2019-03-12','tend':'2019-03-15'},
        'irma':{'lat':slice(30,10), 'lon':slice(-90,-60),'tstart':'2017-09-07','tend':'2017-09-14'},
        'andrew':{'lat':slice(30,10), 'lon':slice(-90,-60),'tstart':'1992-08-21','tend':'1992-08-29'},
        }

file_path = 'D:/paper_4/data/seas5/ecmwf/ecmwf_eps_pf_vars_n15_s90_20190313.nc'
ds = xr.open_dataset(file_path)
ds['tp'] = ds['tp']*1000
# change name tp to Ptot
ds = ds.rename({'tp':'Ptot'})
if 'latitude' in ds.coords:
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    # convert lon from 0 to 360 to -180 to 180 
    ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180

# ds['Ptot'] is cumulative for every time step, how can I use dif to get the precipitation for every time step?
ds['Ptot'] = ds['Ptot'].diff(dim='time', n=1, label='lower')


# plot a map with cartopy using the first time step of the dataset
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())    #
ax.coastlines(resolution='10m')
ax.add_feature(cfeature.BORDERS, linestyle=':')
# plot the grid
grid = ds['Ptot'].sel(number=2).isel(time=-2)
contour = grid.plot(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', cmap = 'Blues',robust=True, add_colorbar=True)
plt.show()

for numbers in ds['number']:
    create_animation(ds.sel(number=numbers.item()), variable = 'Ptot', threshold=2, file_name=rf'D:\paper_4\data\Figures\animations\{storm}\{storm}_{scenario}_{variable}_ens{numbers.item()}_animation.gif')

# plot a timeseries of ds['Ptot'] average across lat and lon
ds['Ptot'].mean(dim=['lat','lon']).sel(number=1).plot()
plt.show()