# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:04:50 2023

@author: H. Goulart
"""
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np

import metpy.calc as mpcalc
import metpy.xarray as mpxarray
from scipy.signal import savgol_filter
from scipy.interpolate import BSpline
from scipy.ndimage import laplace
from haversine import haversine

import clim_functions as climfun 
################################################################################

def shift_on_closest_time(ds_storm_subset, storm_track_control, ds_control, target_time):
    """
    Find the closest datetime in storm_track_control to the target_time in ds_storm_subset and 
    shift ds_control by the time difference.

    :param ds_storm_subset: xarray Dataset with lat and lon coordinates for the subset
    :param storm_track_control: DataFrame with lat and lon columns for the control track
    :param ds_control: xarray Dataset to be shifted
    :param target_time: Target time as a string in format 'YYYY-MM-DD HH:MM:SS'
    :return: Shifted xarray Dataset
    """
    # Extract target lat, lon from ds_storm_subset at the specified time
    target_time_stamp = pd.Timestamp(target_time)
    target_lat = ds_storm_subset['lat'].sel(time=target_time_stamp).item()
    target_lon = ds_storm_subset['lon'].sel(time=target_time_stamp).item()

    # Calculate Haversine distances
    distances = [haversine((target_lat, target_lon), (lat, lon)) 
                 for lat, lon in zip(storm_track_control['lat'].values, storm_track_control['lon'].values)]

    # Create a new DataFrame for distances
    df_distance = storm_track_control.copy()
    df_distance['distances'] = distances

    # Find the datetime of the minimum distance
    time_min = df_distance['distances'].idxmin()

    # Calculate time difference and shift points
    time_difference = target_time_stamp - time_min
    print(f'Time difference: {time_difference}')
    time_resolution = pd.Timedelta('1H')
    shift_points = time_difference // time_resolution

    # Shift ds_control
    ds_shifted = ds_control.shift(time=int(shift_points))

    return ds_shifted

def bias_correct_dataset(ds, ds_composite_gpm, ds_composite_ens, ds_storm_subset_1hour, df_track,
                         bc_start_date, bc_end_date, var = 'tprate', ensemble = False, output_path=None):
    """
    Applies bias correction to a dataset and optionally saves the result as a NetCDF file.

    Parameters:
    ds (xarray.Dataset): Dataset to be bias-corrected.
    ds_composite_gpm (xarray.Dataset): GPM dataset used for bias correction.
    ds_composite_ens (xarray.Dataset): Ensemble dataset used for bias correction.
    ds_storm_subset_1hour (xarray.Dataset): Hourly storm dataset.
    bc_start_date (str): Start date for the bias correction period.
    bc_end_date (str): End date for the bias correction period.
    output_path (str, optional): File path to save the bias-corrected dataset as a NetCDF file.

    Returns:
    xarray.Dataset: Bias-corrected dataset.
    """
    # Calculate bias correction ratios
    if ensemble == True:
        bc_ratio_tp = (ds_composite_gpm['tp'].mean(dim=['lat','lon'])
                    .sel(time=slice(bc_start_date, bc_end_date)).mean() / 
                    ds_composite_ens[var].mean(dim=['lat','lon']).mean(dim=['number'])
                    .sel(time=slice(bc_start_date, bc_end_date)).mean())

        bc_ratio_msl = (ds_storm_subset_1hour['msl'].sel(time=slice(bc_start_date, bc_end_date)).mean() - 
                        ds_composite_ens['msl'].min(dim=['lat','lon']).mean(dim=['number'])
                        .sel(time=slice(bc_start_date, bc_end_date)).mean())

        bc_ratio_U10 = (ds_storm_subset_1hour['U10'].sel(time=slice(bc_start_date, bc_end_date)).mean() - 
                        ds_composite_ens['U10'].max(dim=['lat','lon']).mean(dim=['number'])
                        .sel(time=slice(bc_start_date, bc_end_date)).mean())
    else:
        bc_ratio_tp = (ds_composite_gpm['tp'].mean(dim=['lat','lon'])
                    .sel(time=slice(bc_start_date, bc_end_date)).mean() / 
                    ds_composite_ens[var].mean(dim=['lat','lon'])
                    .sel(time=slice(bc_start_date, bc_end_date)).mean())

        bc_ratio_msl = (ds_storm_subset_1hour['msl'].sel(time=slice(bc_start_date, bc_end_date)).mean() - 
                        ds_composite_ens['msl'].min(dim=['lat','lon'])
                        .sel(time=slice(bc_start_date, bc_end_date)).mean())

        bc_ratio_U10 = (ds_storm_subset_1hour['U10'].sel(time=slice(bc_start_date, bc_end_date)).mean() - 
                        ds_composite_ens['U10'].max(dim=['lat','lon'])
                        .sel(time=slice(bc_start_date, bc_end_date)).mean())

    # Apply bias correction only for where the storm is in space and time
    ds_base = ds.copy()
    radius = 2
    ds_ens_bc = []

    for time_step in df_track.index.values: # .time.values:
        
        center_lat = df_track.loc[time_step, 'lat']
        center_lon = df_track.loc[time_step, 'lon']

        # create mask of where the storm (subset) is
        mask = create_mask(ds_base, center_lat, center_lon, radius)
        # mask.plot() # test to see if it works

        ds_ens_bc_subset = ds_base.sel(time=time_step).where(mask)
        # apply bias correction        
        ds_ens_bc_subset[var] = ds_ens_bc_subset[var] * bc_ratio_tp
        ds_ens_bc_subset['msl'] = ds_ens_bc_subset['msl'] + bc_ratio_msl
        ds_ens_bc_subset['U10'] = ds_ens_bc_subset['U10'] + bc_ratio_U10

        ds_ens_bc_test = ds_base.sel(time=time_step).where(~mask, ds_ens_bc_subset)
        
        ds_ens_bc.append(ds_ens_bc_test)

    ds_ens_bc = xr.concat(ds_ens_bc, dim = 'time')

    # Adjust wind components based on the original wind direction
    ds_ens_bc['wind_direction'] = climfun.calculate_wind_direction_xarray(ds, u10_var='u10', v10_var='v10')
    ds_ens_bc['u10'], ds_ens_bc['v10'] = climfun.decompose_wind(ds_ens_bc, U10_var='U10', wind_dir_var='wind_direction')

    # Update metadata
    ds_ens_bc[var].attrs.update({'units': 'mm/h', 'long_name': 'Total precipitation rate'})
    ds_ens_bc['msl'].attrs.update({'units': 'hPa', 'long_name': 'Mean sea level pressure'})
    ds_ens_bc['U10'].attrs.update({'units': 'm/s', 'long_name': '10 metre total wind'})
    ds_ens_bc['u10'].attrs.update({'units': 'm/s', 'long_name': '10 metre U wind component'})
    ds_ens_bc['v10'].attrs.update({'units': 'm/s', 'long_name': '10 metre V wind component'})

    # Save as NetCDF if output path is provided
    if output_path:
        ds_ens_bc.to_netcdf(output_path)

    return ds_ens_bc

def create_mask(ds, center_lat, center_lon, radius):
        """Create a mask for points within the specified radius of the center."""
        lat_condition = (ds.lat >= center_lat - radius) & (ds.lat <= center_lat + radius)
        lon_condition = (ds.lon >= center_lon - radius) & (ds.lon <= center_lon + radius)
        return lat_condition & lon_condition

def generate_boundaries(start_time, end_time, freq='12H'):
    # Generate a date range at 6-hour intervals
    boundaries = pd.date_range(start=start_time, end=end_time, freq=freq)
    return boundaries

def adjust_time(ds):
   
    ds['tp'] = ds['tp'].diff(dim='time', n=1, label='lower')
    # Select the first 5 timesteps
    ds = ds.isel(time=slice(0, 12))

    return ds


# Load data and pararmeters
file_path = 'D:/paper_4/data/seas5/ecmwf/ecmwf_eps_pf_010_vars_n50_s48_20190314.nc' # ecmwf_eps_pf_vars_n15_s90_20190313 / ecmwf_eps_pf_vars_n15_s90_20170907
start_time = '2019-03-13T12:00:00.000000000'
end_time = '2019-03-16T00:00:00.000000000'
target_time = '2019-03-15 00:00:00' #landfall in Beira

# load IFS ensemble
ds = xr.open_dataset(file_path)
ds = climfun.preprocess_era5_dataset(ds, forecast=True, tprate_convert=True)
ds_ens=ds.copy().sel(time=slice(start_time,end_time))

# load control run
ds_control_orig = xr.open_dataset(r'D:\paper_4\data\seas5\ecmwf\ecmwf_oper_fc_vars_s72_20190313.nc')
ds_control_orig = climfun.preprocess_era5_dataset(ds_control_orig, forecast=True, tprate_convert=True)
ds_control = ds_control_orig.copy().sel(time=slice(start_time,end_time))

# load ERA5 dataset
ds_era5 = xr.open_dataset(r'D:\paper_4\data\era5\era5_hourly_vars_idai_single_2019_03.nc').sel(time=slice(start_time,end_time))
ds_era5 = climfun.preprocess_era5_dataset(ds_era5)

# try the 7 lead time rebuild forecast
ds_ips_rebuild_orig = xr.open_mfdataset('D:\paper_4\data\seas5\ecmwf\short_forecasts\ecmwf_oper_fc_010_vars_s15_201903*.nc', #ecmwf_eps_cf_010_vars_s10_201903*.nc
                                combine='nested', concat_dim='time', preprocess=adjust_time).load()
ds_ips_rebuild_orig = climfun.preprocess_era5_dataset(ds_ips_rebuild_orig, tprate_convert=True)
# ds_ips_rebuild_orig = ds_ips_rebuild_orig.rolling(time=6, center=False).mean()
# ds_ips_rebuild_orig = savgol_filter(ds_ips_rebuild_orig['U10'], 4,2, axis=0)
ds_ips_rebuild = ds_ips_rebuild_orig.sel(time=slice(start_time,end_time))

boundary_timestamps = generate_boundaries(ds_ips_rebuild_orig.time[0].values, ds_ips_rebuild_orig.time[-1].values)

# Load GPM dataset
ds_gpm = xr.open_dataset(r'D:\paper_4\data\nasa_data\gpm_imerg_201903.nc').sel(time=slice(start_time,end_time))
ds_gpm.coords['lon'] = (ds_gpm.coords['lon'] + 180) % 360 - 180
ds_gpm = ds_gpm.sortby('lat', ascending=False)
ds_gpm = ds_gpm.rename({'precipitation': 'tp'})
ds_gpm = ds_gpm.resample(time='1H').mean() # have to do this to have the same time resolution as the other datasets
# save ds_gpm_hour to netcdf
# ds_gpm_hour.to_netcdf(r'D:\paper_4\data\nasa_data\gpm_imerg_201903_hourly.nc')

# Load IBTrACS dataset
df_storm, df_storm_subset = climfun.process_ibtracs_storm_data(ibtracs_path = r'D:\paper_4\data\ibtracs\IBTrACS.since1980.v04r00.nc',
                                                  storm_id = '2019063S18038', 
                                                  ds_time_range = ds_control_orig)
ds_storm_subset_1hour = climfun.interpolate_dataframe(df_storm_subset, ['U10', 'msl', 'time', 'lat', 'lon'], 'time', '1H')

# track all the storms using the storm tracker
# track storm ensemble
storm_track_ens = climfun.storm_tracker_mslp_ens(ds_ens, df_storm, smooth_step='savgol', large_box=3, small_box=1.5)
# track ds_era5 storm
storm_track_era5 = climfun.storm_tracker_mslp_updated(ds_era5, df_storm, smooth_step='savgol', large_box=3, small_box=1.5)
# track control run
storm_track_control = climfun.storm_tracker_mslp_updated(ds_control, df_storm, smooth_step='savgol', large_box=3, small_box=1.5)
# track ips rebuild
storm_track_ips_rebuild = climfun.storm_tracker_mslp_updated(ds_ips_rebuild, df_storm, smooth_step='savgol', large_box=3, small_box=1.5)

# plot storm tracks 
climfun.plot_minimum_track(storm_track_ens)
climfun.plot_minimum_track(storm_track_era5, df_storm_subset)
climfun.plot_minimum_track(storm_track_control, df_storm_subset)
climfun.plot_minimum_track(storm_track_ips_rebuild, df_storm_subset)


# Shift control and ips_rebuild datasets to match the storm track (important for bias correction)

ds_control_shifted = shift_on_closest_time(ds_storm_subset_1hour, storm_track_control, ds_control_orig, target_time).sel(time=slice(start_time,end_time))
storm_track_control_shifted = climfun.storm_tracker_mslp_updated(ds_control_shifted, df_storm, smooth_step='savgol', large_box=3, small_box=1.5)

ds_ips_rebuild_orig_shifted = shift_on_closest_time(ds_storm_subset_1hour, storm_track_ips_rebuild, ds_ips_rebuild_orig, target_time)
ds_ips_rebuild_shifted = ds_ips_rebuild_orig_shifted.sel(time=slice(start_time,end_time))
storm_track_ips_rebuild_shifted = climfun.storm_tracker_mslp_updated(ds_ips_rebuild_shifted, df_storm, smooth_step='savgol', large_box=3, small_box=1.5)

# Calculate composite
ds_composite_ens = climfun.storm_composite_ens(ds_ens, storm_track_ens, radius = 2)
ds_composite_era5 = climfun.storm_composite(ds_era5, storm_track_era5, radius = 2)
ds_composite_control = climfun.storm_composite(ds_control, storm_track_control, radius = 2)
ds_composite_control_shifted = climfun.storm_composite(ds_control_shifted, storm_track_control_shifted, radius = 2)
ds_composite_ips_rebuild = climfun.storm_composite(ds_ips_rebuild, storm_track_ips_rebuild, radius = 2)
ds_composite_ips_rebuild_shifted = climfun.storm_composite(ds_ips_rebuild_shifted, storm_track_ips_rebuild_shifted, radius = 2)

# use the observed track to calculate the composite for gpm
ds_composite_gpm = climfun.storm_composite(ds_gpm, ds_storm_subset_1hour.to_dataframe(), radius = 2)

# PLOTS
plot_figures = False
if plot_figures == True:
    fig = plt.figure(figsize=(10, 10))
    ds_composite_gpm['tp'].mean(dim=['lat','lon']).plot(label='GPM imerg', linestyle='-.', linewidth=3)
    ds_composite_era5['tp'].mean(dim=['lat','lon']).plot(label='era5', linestyle='--' )
    ds_composite_ens['tp'].mean(dim=['lat','lon', 'number']).plot(label='IFS ensemble mean')
    ds_composite_control['tp'].mean(dim=['lat','lon']).plot(label='IFS control run')
    ds_composite_control_shifted['tp'].mean(dim=['lat','lon']).plot(label='IFS control run shifted', linestyle='--')
    ds_composite_ips_rebuild['tp'].mean(dim=['lat','lon']).plot(label='IFS rebuild', color = 'black')
    ds_composite_ips_rebuild_shifted['tp'].mean(dim=['lat','lon']).plot(label='IFS rebuild shifted', color = 'red')
    plt.legend()
    plt.show()

    # crea figure and plot mslp for the storm
    fig = plt.figure(figsize=(10, 10))
    ds_composite_control['msl'].min(dim=['lat','lon']).plot(label='IFS control run')
    ds_composite_control_shifted['msl'].min(dim=['lat','lon']).plot(label='IFS control run shifted', linestyle='--')
    ds_composite_era5['msl'].min(dim=['lat','lon']).plot(label='era5', linestyle='--' )
    ds_composite_ips_rebuild['msl'].min(dim=['lat','lon']).plot(label='IFS rebuild', color = 'black')
    ds_composite_ips_rebuild_shifted['msl'].min(dim=['lat','lon']).plot(label='IFS rebuild shifted', color = 'red')
    ds_storm_subset_1hour['msl'].plot(label='Ibtracs (ref)', linestyle=':' )
        
    # Adding vertical lines for each boundary timestamp
    for b_time in boundary_timestamps:
        plt.axvline(x=b_time, color='r', linestyle='--')

    plt.legend()
    plt.show()


    # crea figure and plot mslp for the storm
    fig = plt.figure(figsize=(10, 10))
    ds_composite_control['U10'].max(dim=['lat','lon']).plot(label='IFS control run')
    ds_composite_control_shifted['U10'].max(dim=['lat','lon']).plot(label='IFS control run shifted', linestyle='--')
    ds_composite_era5['U10'].max(dim=['lat','lon']).plot(label='era5', linestyle='--' )
    ds_composite_ips_rebuild['U10'].max(dim=['lat','lon']).plot(label='IFS rebuild', color = 'black')
    ds_composite_ips_rebuild_shifted['U10'].max(dim=['lat','lon']).plot(label='IFS rebuild shifted', color = 'red')
    ds_storm_subset_1hour['U10'].plot(label='Ibtracs (ref)', linestyle=':' )
         
    # Adding vertical lines for each boundary timestamp
    for b_time in boundary_timestamps:
        plt.axvline(x=b_time, color='r', linestyle='--')

    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10, 10))
    ds_composite_control['msl'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='IFS control run')
    ds_composite_era5['msl'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='era5', linestyle='--' )
    ds_composite_ips_rebuild_shifted['msl'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='IFS rebuild', color = 'black')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10, 10))
    ds_composite_control_shifted['U10'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='IFS control run')
    ds_composite_era5['U10'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='era5', linestyle='--' )
    ds_composite_ips_rebuild['U10'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='IFS rebuild', color = 'black')
    plt.legend()
    plt.show()
    
    fig = plt.figure(figsize=(10, 10))
    ds_composite_control['tp'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='IFS control run')
    ds_composite_control_shifted['tp'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='IFS control run shifted', linestyle='--')
    ds_composite_era5['tp'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='era5')
    ds_composite_ips_rebuild['tp'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='IFS rebuild', color = 'black')
    ds_composite_ips_rebuild_shifted['tp'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='IFS rebuild shifted', color = 'red', linestyle='--' )
    ds_composite_gpm['tp'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='GPM imerg', linestyle='-.', linewidth=3)
    plt.legend()
    plt.show()
    


# BIAS CORRECTION ##############################################################
bc_start_date = '2019-03-14T18:00:00.000000000'
bc_end_date = '2019-03-15T03:00:00.000000000'
path_bc_file = r'D:\paper_4\data\seas5\bias_corrected\ecmwf_eps_pf_010_vars_n50_s48_20190314_bc.nc'
# Execute bias correction
# ds_ens_bc = bias_correct_dataset(ds, ds_composite_gpm, ds_composite_ens, ds_storm_subset_1hour,storm_track_ens
#                                  bc_start_date, bc_end_date, ensemble=True, output_path = path_bc_file, var = 'tp')

ds_control_bc = bias_correct_dataset(ds_control_orig, ds_composite_gpm, ds_composite_control, ds_storm_subset_1hour, storm_track_control,
                                 bc_start_date, bc_end_date, ensemble= False, output_path = r'D:\paper_4\data\seas5\bias_corrected\ecmwf_eps_cf_010_vars_s72_20190313_bc.nc', var = 'tp')
ds_control_shifted_bc = bias_correct_dataset(ds_control_shifted, ds_composite_gpm, ds_composite_control_shifted, ds_storm_subset_1hour, storm_track_control_shifted,
                                    bc_start_date, bc_end_date, ensemble= False, output_path = r'D:\paper_4\data\seas5\bias_corrected\ecmwf_eps_cf_010_vars_s72_20190313_shifted_bc.nc', var = 'tp')
ds_era5_bc = bias_correct_dataset(ds_era5, ds_composite_gpm, ds_composite_era5, ds_storm_subset_1hour, storm_track_era5,
                                    bc_start_date, bc_end_date, ensemble= False, output_path = r'D:\paper_4\data\seas5\bias_corrected\era5_hourly_vars_idai_single_2019_03_bc.nc', var = 'tp')
ds_ips_rebuild_bc = bias_correct_dataset(ds_ips_rebuild_orig_shifted, ds_composite_gpm, ds_composite_ips_rebuild_shifted, ds_storm_subset_1hour, storm_track_ips_rebuild_shifted,
                                    bc_start_date, bc_end_date, ensemble= False, output_path = r'D:\paper_4\data\seas5\bias_corrected\ecmwf_oper_fc_rebuild_bc.nc', var = 'tp')
# CHECK RESULTS ################################################################
# ds_composite_ens_bc = climfun.storm_composite_ens(ds_ens_bc, storm_track_ens, radius = 2)
ds_composite_control_bc = climfun.storm_composite(ds_control_bc, storm_track_control, radius = 2)
ds_composite_control_shifted_bc = climfun.storm_composite(ds_control_shifted_bc, storm_track_control_shifted, radius = 2)
ds_composite_era5_bc = climfun.storm_composite(ds_era5_bc, storm_track_era5, radius = 2)
ds_composite_ips_rebuild_bc = climfun.storm_composite(ds_ips_rebuild_bc, storm_track_ips_rebuild_shifted, radius = 2)

# NON BIAS corrected
climfun.plot_comp_variable_timeseries(ds_composite_ens, 'U10', ds_era5_composite=ds_composite_era5, agg_method='max', obs_data=df_storm_subset)
climfun.plot_comp_variable_timeseries(ds_composite_ens, 'msl', ds_era5_composite=ds_composite_era5, agg_method='min', obs_data=df_storm_subset)
climfun.plot_comp_variable_timeseries(ds_composite_ens, 'tp', ds_era5_composite=ds_composite_gpm, agg_method='mean')

# plot timeseries average for bc u10 and msl
climfun.plot_comp_variable_timeseries(ds_composite_ens_bc, 'U10', ds_era5_composite=ds_composite_era5, agg_method='max', obs_data=df_storm_subset)
climfun.plot_comp_variable_timeseries(ds_composite_ens_bc, 'msl', ds_era5_composite=ds_composite_era5, agg_method='min', obs_data=df_storm_subset)
climfun.plot_comp_variable_timeseries(ds_composite_ens_bc, 'tp', ds_era5_composite=ds_composite_gpm, agg_method='mean')

# plot ds_composite_ens_bc and ds_composite_gpm and ds_composite_ens
ds_composite_control['tp'].mean(dim=['lat','lon']).plot(label='control')
ds_composite_control_shifted_bc['tp'].mean(dim=['lat','lon']).plot(label='shifted bias corrected')
ds_composite_ips_rebuild_bc['tp'].mean(dim=['lat','lon']).plot(label='rebuild bias corrected')
ds_composite_gpm['tp'].mean(dim=['lat','lon']).plot(label='gpm (ref)', linestyle='-.', color = 'black', linewidth=3)
plt.axvspan(bc_start_date, bc_end_date, color='grey', alpha=0.2)
plt.legend()
plt.title(f'Mean precipitation rate (bc ({np.round(ds_composite_control_bc["tp"].mean().item(),2)}) vs ref({np.round(ds_composite_gpm["tp"].mean().item(),2)}))')
plt.show()


# plot ds_composite_ens_bc and ds_composite_gpm and ds_composite_ens
ds_composite_control['U10'].max(dim=['lat','lon']).plot(label='ensemble raw')
ds_composite_control_shifted_bc['U10'].max(dim=['lat','lon']).plot(label='ensemble bias corrected')
ds_composite_ips_rebuild_bc['U10'].max(dim=['lat','lon']).plot(label='rebuild bias corrected')
ds_storm_subset_1hour['U10'].plot(label='Ibtracs (ref)')
plt.axvspan(bc_start_date, bc_end_date, color='grey', alpha=0.2)
plt.legend()
plt.title(f'Max wind speed (bc ({np.round(ds_composite_control_bc["U10"].max(dim=["lat","lon"]).mean().item(),2)}) vs ref({np.round(ds_storm_subset_1hour["U10"].mean().item(),2)}))')
plt.show()


# plot ds_composite_ens_bc and ds_composite_gpm and ds_composite_ens
ds_composite_control['msl'].min(dim=['lat','lon']).plot(label='ensemble raw')
ds_composite_control_shifted_bc['msl'].min(dim=['lat','lon']).plot(label='ensemble bias corrected')
ds_composite_ips_rebuild_bc['msl'].min(dim=['lat','lon']).plot(label='rebuild bias corrected')
ds_storm_subset_1hour['msl'].plot(label='Ibtracs (ref)')
plt.axvspan(bc_start_date, bc_end_date, color='grey', alpha=0.2)
plt.legend()
plt.title(f'Min mslp (bc ({np.round(ds_composite_control_bc["msl"].min(dim=["lat","lon"]).mean().item(),2)}) vs ref({np.round(ds_storm_subset_1hour["msl"].mean().item(),2)}))')
plt.show()


# now plot tp for the Beira location
ds_gpm['tp'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='gpm-imerg', linestyle='-.', color = 'black', linewidth=3)
ds_control_shifted_bc['tp'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='IFS control bc')
ds_control['tp'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='IFS control')
ds_ips_rebuild['tp'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='IFS rebuild', color = 'red')
ds_ips_rebuild_bc['tp'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='IFS rebuild bc', linestyle='--', color = 'red', linewidth=3)
ds_era5['tp'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='era5', linestyle='--')
plt.axvspan(bc_start_date, bc_end_date, color='grey', alpha=0.2)
plt.legend()
plt.show()

# now plot msl for the Beira location
ds_control_bc['U10'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='IFS control bc')
ds_control['U10'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='IFS control')
ds_ips_rebuild['U10'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='IFS rebuild', color = 'red')
ds_ips_rebuild_bc['U10'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='IFS rebuild bc', linestyle='--', color = 'red', linewidth=3)
ds_era5['U10'].sel(lat=-19.8, lon=34.9, method='nearest').plot(label='era5', linestyle='--')
plt.axvspan(bc_start_date, bc_end_date, color='grey', alpha=0.2)
plt.legend()
plt.show()

#TODO: add region figure to help understand how it is being corrected.

# use xarray to open this file: D:\paper_4\data\quadtree_ifs_rebuild_idai_bc\press_2d.nc
test = xr.open_dataset(r'D:\paper_4\data\quadtree_ifs_rebuild_idai_bc\press_2d.nc').close()