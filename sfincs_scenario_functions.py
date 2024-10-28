# -*- coding: utf-8 -*-
"""
Prepare files for HydroMT
Created on Tue Oct  9 16:39:47 2022
@author: morenodu
"""
import sys
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.pyplot as plt
import configparser
import yaml
from datetime import datetime
from os.path import join, isfile, isdir, dirname
import subprocess
import shutil
from datetime import datetime, timedelta

import hydromt
from hydromt_sfincs import SfincsModel, utils

########################################################################################
# Clipping Functions
# This section includes functions for clipping geospatial data to specific regions or
# boundaries. These functions are used to preprocess data for hydrological modeling.
########################################################################################

def clip_data_to_region(bbox, export_dir, data = 'deltares_data', list_indices = None):
    '''
    Choose a location to clip, a exportind directory and the indices.
    '''
    #Location
    if type(bbox[0]) == float:
        bbox = bbox
    elif type(bbox[0]) == dict:
        bbox = [bbox['lon_min'],bbox['lat_min'],bbox['lon_max'],bbox['lat_max']]

    # Open catalog:
    data_cat = hydromt.DataCatalog(data_libs=data)
  
    # Select indices:
    if list_indices == None:
        list_indices = ['merit_hydro','gebco', 'osm_coastlines', 'osm_landareas', 'gswo', 'fabdem', 'dtu10mdt_egm96', 'dtu10mdt', 'gcn250', 'vito', 'vito_2019_v3.0.1']

    # clip data:
    os.makedirs(export_dir, exist_ok=True) 
    data_cat.export_data(fr'{export_dir}', 
                        bbox=bbox, #xmin, ymin, xmax, ymax
                        time_tuple=None,
                        source_names=list_indices)
 
class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True   

def data_catalog_edit(data_catalog, key_to_edit, key_name, key_path):
    
    if not data_catalog.endswith(".yml"):
        raise ValueError('file needs to be a .yml format')
    
    # Open the YAML file for reading
    with open(data_catalog) as file:
        data = yaml.full_load(file)

    # Duplicate the entry
    duplicated_entry = data[key_to_edit].copy()

    # Modify the value of the duplicated entry
    duplicated_entry['path'] = key_path

    # Add the duplicated entry to the YAML data
    new_key = key_name
    data[new_key] = duplicated_entry

    # Write the modified data back to the YAML file
    with open(data_catalog, 'w') as file:
        yaml.dump(data, file, Dumper=NoAliasDumper)
        
# SFINCS functions
def add_waterlevel_offshore_sfincs(sf_model, his_file_path, slr = None):
    # function for obtaining data and index from ds_his
    ds_his = utils.read_sfincs_his_results(his_file_path, crs=sf_model.crs.to_epsg())

    wl_df = pd.DataFrame(
            data=ds_his.point_zs.to_numpy(),
            index=ds_his.time.to_numpy(),
            columns=np.arange(1, ds_his.point_zs.to_numpy().shape[1] + 1, 1),
        )
    
    # Add sea level rise if provided
    if slr is not None:
        wl_df += slr

    station_x = ds_his.station_id.station_x.values
    station_y = ds_his.station_id.station_y.values
    # Create a GeoDataFrame
    bnd = gpd.GeoDataFrame(index=ds_his.station_id.values.astype(int),
                        geometry=gpd.points_from_xy(station_x, station_y), crs=sf_model.crs)

    sf_model.setup_waterlevel_forcing(timeseries=wl_df, locations=bnd)
    
def generate_sfincs_model(root_folder, scenario, storm, data_libs, topo_map, writing_mode, tref, tstart, tstop, 
                          bbox = None, grid_setup=None, res = 100, mode = None, area_mask = None, waterlevel_path = None, slr = None):
    # Create sfincs model
    if mode != None:
        root=f'{root_folder}{storm}_{scenario}_{mode}'
    else:
        root=f'{root_folder}{storm}_{scenario}'

    if writing_mode == 'write':
        sf = SfincsModel(data_libs = data_libs, root = root, mode="w+")
        # Set up the grid
        if grid_setup is not None:
            sf.setup_grid(**grid_setup)
        else:
            sf.setup_grid_from_region(region={'bbox':bbox}, res = res)
        # Load in wanted elevation datasets:
        # the 2nd elevation dataset (gebco) is used where the 1st dataset returns nodata values
        if topo_map in ['fabdem','merit']:
            datasets_dep = [{"elevtn": topo_map, "zmin": 0.001}, {"elevtn": "gebco"}]
            dep = sf.setup_dep(datasets_dep=datasets_dep, buffer_cells = 2)
        elif topo_map == 'cudem':
            datasets_dep = [{"elevtn": topo_map}, {"elevtn": 'fabdem', "zmin": 0.0001}, {"elevtn": "gebco"}]
            dep = sf.setup_dep(datasets_dep=datasets_dep)
        elif topo_map == 'beira_dem':
            datasets_dep = [{"elevtn": topo_map}]
            dep = sf.setup_dep(datasets_dep=datasets_dep)
        else:
            raise ValueError('topo_map must be either fabdem, merit or cudem for now')

        # Choosing how to choose you active cells can be based on multiple criteria, here we only specify a minimum elevation of -5 meters
        if (area_mask != None) and (storm.split('_')[0] == 'sandy'):
            sf.setup_mask_active(mask = area_mask, include_mask=r"d:\paper_3\data\us_dem\osm_landareas_mask.gpkg", zmin=-10, reset_mask=True)
        else:
            sf.setup_mask_active( mask= 'osm_landareas', mask_buffer = 100, reset_mask=True) # mask= 'osm_landareas', mask_buffer = 100

        # Here we add water level cells along the coastal boundary, for cells up to an elevation of -5 meters
        sf.setup_mask_bounds(btype="waterlevel",all_touched=True,include_mask='osm_coastlines', reset_bounds=True) # include_mask='osm_coastlines',, zmax=0, 

        # Add spatially varying roughness data:
        # read river shapefile and add manning value to the attributes, enforce low roughness data
        # gdf = sf.data_catalog.get_rasterdataset("rivers_lin2019_v1", geom=sf.region).to_crs(
        #     sf.crs
        # )
        # gdf["geometry"] = gdf.buffer(50)
        # gdf["manning"] = 0.03

        # # rasterize the manning value of gdf to the  model grid
        # da_manning = sf.grid.raster.rasterize(gdf, "manning", nodata=np.nan)
        # datasets_rgh = [{"manning": da_manning}, {"lulc": "vito"}]
        datasets_rgh = [{"lulc": "vito"}]

        sf.setup_manning_roughness(
            datasets_rgh=datasets_rgh,
            manning_land=0.04,
            manning_sea=0.02,
            rgh_lev_land=0,  # the minimum elevation of the land
        )

        # ADD SUBGRID - Make subgrid derived tables TODO: Still not implemented because code does not work, and I wonder if the mix of my data with gebco makes subgrid irrelevant.
        # sf.setup_subgrid(
        #     datasets_dep=datasets_dep,
        #     datasets_rgh=datasets_rgh,
        #     nr_subgrid_pixels=5,
        #     write_dep_tif=True,
        #     write_man_tif=False,
        # )

        # Add spatially varying infiltration data:
        sf.setup_cn_infiltration("gcn250", antecedent_moisture="avg")

        # Add time-series:
        start_datetime = datetime.strptime(tstart, '%Y%m%d %H%M%S')
        stop_datetime = datetime.strptime(tstop, '%Y%m%d %H%M%S')
        sf.setup_config(
            **{
                "tref": tref,
                "tstart": tstart,
                "tstop": tstop,
                "dtout":10800,
                "dtmaxout": (stop_datetime - start_datetime).total_seconds(),
                "advection": 0,
            }
        )

    elif writing_mode == 'update':
        ## Now update model with forcings
        sf = SfincsModel(data_libs = data_libs, root = f'{root_folder}{storm}_base', mode="r")
        sf.read()
        sf.set_root(root = root, mode='w+')
        # update paths to static files in base root
        
        modes = ['rain', 'surge', 'rain_surge', 'era5', 'surge']
        if mode in modes:
            if mode in ['rain', 'rain_surge']:
                # rainfall
                sf.setup_precip_forcing_from_grid(precip=f"D:\paper_4\data\quadtree_era5\precip_2d.nc", aggregate=False)
            if mode in ['surge', 'rain_surge']:
                # SURGE - WATERLEVEL
                sf.setup_waterlevel_forcing(geodataset=f'gtsm_echam_sn_{scenario}_{storm}_waterlevel', offset = 'dtu10mdt', buffer = 1e4)
                # gdf_locations = sf.forcing['bzs'].vector.to_gdf()
                # sf.setup_mask_active(mask = area_mask, zmin=-4, include_mask = gdf_locations, reset_mask=True)  
            if mode == 'era5':
                # ERA5
                sf.setup_precip_forcing_from_grid(precip=f'era5_hourly_test', aggregate=False)
                sf.setup_wind_forcing_from_grid(wind=f'era5_hourly_test')
                sf.setup_pressure_forcing_from_grid(press = f'era5_hourly_test')   
            if (mode == 'surge') & (waterlevel_path != None):
                add_waterlevel_offshore_sfincs(sf, waterlevel_path, slr = slr)

    # save model
    sf.write()  # write all because all folders are run in parallel on snellius

def create_sfincs_base_model(root_folder, scenario, storm, data_libs, topo_map,
                             tref, tstart, tstop, bbox = None, grid_setup=None, res=100):
    """
    Creates a base Sfincs model with the specified settings.

    Args:
        root_folder: The root folder for the model.
        scenario: The scenario name.
        storm: The storm name.
        data_libs: A dictionary of data libraries used by the model.
        topo_map: The topography map to use (fabdem, merit, cudem, or beira_dem).
        bbox: The bounding box for the model (optional).
        grid_setup: A dictionary of parameters for setting up the grid (optional).
        res: The grid resolution in meters (default: 100).
        area_mask: A mask for the active cells (optional).

    Returns:
        A SfincsModel object representing the base model.
    """
    
    # Create the root directory based on the mode
    root = f"{root_folder}{storm}_{scenario}"
    print(f'{root}')

    # Initialize the SfincsModel object
    sf = SfincsModel(data_libs=data_libs, root=root, mode="w+")

    # Set up the grid
    if grid_setup is not None:
        sf.setup_grid(**grid_setup)
    elif bbox is not None:
        sf.setup_grid_from_region(region={"bbox": bbox}, res=res)
    elif (grid_setup is None) and (bbox is None):
        raise ValueError("Either grid_setup or bbox must be specified")
    elif (grid_setup is not None) and (bbox is not None):
        raise ValueError("Only one of grid_setup or bbox can be specified")

    # Load elevation datasets
    if topo_map in ["fabdem", "merit"]:
        datasets_dep = [{"elevtn": topo_map, "zmin": 0.001}, {"elevtn": "gebco"}]
    else:
        datasets_dep = [{"elevtn": topo_map}]
    # create dep
    sf.setup_dep(datasets_dep=datasets_dep, buffer_cells=2)

    # Set up active cells - this is a mask of cells that are active in the model and based on trial and error, not really automated
    sf.setup_mask_active(mask="osm_landareas", mask_buffer=100, reset_mask=True)

    # Set up water level cells
    sf.setup_mask_bounds(btype="waterlevel", all_touched=True, include_mask="osm_coastlines", reset_bounds=True)

    # Set up roughness data
    datasets_rgh = [{"lulc": "vito"}]
    sf.setup_manning_roughness(
        datasets_rgh=datasets_rgh,
        manning_land=0.04,
        manning_sea=0.02,
        rgh_lev_land=0,
    )

    # Set up infiltration data
    sf.setup_cn_infiltration("gcn250", antecedent_moisture="dry")

    # Add time-series:
    start_datetime = datetime.strptime(tstart, '%Y%m%d %H%M%S')
    stop_datetime = datetime.strptime(tstop, '%Y%m%d %H%M%S')
    sf.setup_config(
        **{
            "tref": tref,
            "tstart": tstart,
            "tstop": tstop,
            "dtout":10800,
            "dtmaxout": (stop_datetime - start_datetime).total_seconds(),
            "advection": 0,
        }
    )

    sf.write()


def update_sfincs_model(base_root, new_root, data_libs, mode, precip_path=None, waterlevel_path=None, slr=None):
    """
    Updates an SFINCS model with new forcings based on the specified mode.

    Parameters:
    base_root (str): The root directory of the base SFINCS model.
    new_root (str): The root directory for the updated SFINCS model.
    data_libs (str): Path to the data libraries.
    mode (str): The mode of update ('rain', 'surge', 'rain_surge', 'era5').
    precip_path (str, optional): Path to the precipitation data.
    waterlevel_path (str, optional): Path to the water level data.
    slr (float, optional): Sea level rise parameter.
    """
    # Initialize the base model for reading
    sf_base = SfincsModel(data_libs=data_libs, root=base_root, mode="r")
    sf_base.read()

    # Set new root for writing
    sf_base.set_root(root=new_root, mode='w+')

    # Update the model based on the specified mode
    if mode in ['rain', 'rain_surge']:
        if not precip_path:
            raise ValueError("Precipitation path required for rain or rain_surge mode.")
        sf_base.setup_precip_forcing_from_grid(precip=precip_path, aggregate=False)
        # sf_base.setup_precip_forcing_from_grid(precip=ds['precip'] * increase_factor, aggregate=False)

    if mode in ['surge', 'rain_surge']:
        if not waterlevel_path:
            raise ValueError("Water level path required for surge or rain_surge mode.")
        add_waterlevel_offshore_sfincs(sf_base, waterlevel_path, slr=slr)

    if mode == 'era5':
        sf_base.setup_precip_forcing_from_grid(precip='era5_hourly_test', aggregate=False)
        sf_base.setup_wind_forcing_from_grid(wind='era5_hourly_test')
        sf_base.setup_pressure_forcing_from_grid(press='era5_hourly_test')

    # Write the updated model
    sf_base.write()



# Climate scenario functions

def generate_forcing(grid_setup, root_folder, scenario, data_libs, tref,
                     tstart, tstop, forcing_catalog, forcings=['wind', 'precip', 'pressure']):
    
    root=f'{root_folder}quadtree_{scenario}_forcing'
    sf = SfincsModel(data_libs = data_libs, root = root, mode="w+")
    # Set up the grid
    sf.setup_grid(**grid_setup)
    # Add time-series:
    start_datetime = datetime.strptime(tstart, '%Y%m%d %H%M%S')
    stop_datetime = datetime.strptime(tstop, '%Y%m%d %H%M%S')
    sf.setup_config(
        **{
            "tref": tref,
            "tstart": tstart,
            "tstop": tstop,
            "dtout":10800,
            "dtmaxout": (stop_datetime - start_datetime).total_seconds(),
            "advection": 0,
        }
    )
    if 'precip' in forcings:
        sf.setup_precip_forcing_from_grid(precip=forcing_catalog, aggregate=False)
    if 'wind' in forcings:
        sf.setup_wind_forcing_from_grid(wind=forcing_catalog)
    if 'pressure' in forcings:
        sf.setup_pressure_forcing_from_grid(press=forcing_catalog)
    
    # save model
    sf.write()  # write sf.set_root(root = new_root, mode='w+')

def shift_time_in_netcdf(file_path, output_file_path, days_to_shift):
    # Load the netcdf file
    ds = xr.open_dataset(file_path)
    
    # Shift the time dimension
    ds_shifted = ds.assign_coords(time=ds.time + pd.to_timedelta(days_to_shift, unit='h'))

    ds_shifted.to_netcdf(output_file_path)
    
    return ds_shifted

def scale_precip_in_netcdf(file_path, output_file_path, scale_factor):
    # Load the netcdf file
    ds = xr.open_dataset(file_path)
    
    # Check if 'tp' is in the dataset variables
    if 'tp' not in ds:
        raise ValueError("'tp' not found in the dataset variables.")
    
    # Scale the precipitation data
    ds_scaled = ds.copy()
    ds_scaled['tp'] = ds['tp'] * scale_factor
    ds.close()

    ds_scaled.to_netcdf(output_file_path)
    
    return ds_scaled

def copy_entire_folder(src_folder, dest_folder):
    """
    Copies an entire folder and its content to a new location with a new name.

    Parameters:
    src_folder (str): The path to the source folder.
    dest_folder (str): The path to the destination folder with the new name.
    """
    try:
        shutil.copytree(src_folder, dest_folder)
        print(f"Folder '{src_folder}' copied successfully to '{dest_folder}'")
    except Exception as e:
        print(f"Error copying folder: {e}")

def copy_two_files(src_folder, dest_folder, file1_name, file2_name):
    """
    Copies two files from the source folder to the destination folder.

    Parameters:
    src_folder (str): The path to the source folder.
    dest_folder (str): The path to the destination folder.
    file1_name (str): The name of the first file to copy.
    file2_name (str): The name of the second file to copy.
    """
    # Construct the full paths for the source and destination of each file
    src_file1 = os.path.join(src_folder, file1_name)
    src_file2 = os.path.join(src_folder, file2_name)
    
    dest_file1 = os.path.join(dest_folder, file1_name)
    dest_file2 = os.path.join(dest_folder, file2_name)
    
    # Check if the source files exist
    if not os.path.exists(src_file1):
        print(f"File not found: {src_file1}")
        return
    if not os.path.exists(src_file2):
        print(f"File not found: {src_file2}")
        return
    
    # Ensure the destination folder exists, create if it does not
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Copy each file
    shutil.copy2(src_file1, dest_file1)
    shutil.copy2(src_file2, dest_file2)
    print(f"Files copied successfully to {dest_folder}")

def add_adaptation_measures(data_libs, original_root, new_root, structures_params):
    """
    Parameters:
    - data_libs: The data libraries used by the SfincsModel.
    - original_root: The root directory of the original model.
    - new_root: The root directory where the modified model should be saved.
    - structures_params: A dictionary containing the parameters for setup_structures method.
    """
    # Initialize and read the original model
    sf = SfincsModel(data_libs=data_libs, root=original_root, mode="r")
    sf.read()

    # Set new root for writing the modified model
    sf.set_root(root=new_root, mode='w+')

    # Update the model with adaptation measures using the parameters from the dictionary
    sf.setup_structures(**structures_params)

    # Write the updated model to the new location
    sf.write()
    print(f"Adaptation measures added and model saved to {new_root}")

def add_hours_to_time(time_string, hours_to_add):
    """
    Adds a specified number of hours to a given time string.

    Parameters:
    time_string (str): The original time string in the format 'YYYYMMDD HHMMSS'.
    hours_to_add (int): The number of hours to add to the time string.

    Returns:
    str: A new time string with the added hours, in the same format as the input.
    """
    # Define the time format
    time_format = '%Y%m%d %H%M%S'
    
    # Convert the string to a datetime object
    original_time = datetime.strptime(time_string, time_format)
    
    # Add the hours to the datetime object
    new_time = original_time + timedelta(hours=hours_to_add)
    
    # Convert the new datetime object back to a string
    return new_time.strftime(time_format)


def setup_ofshore_sfincs_model(root_folder, sim_name, storm, base_name, data_libs, grid_setup, tref_off, tstart_off, tstop_off, 
                               forcing_catalog):
    """
    Sets up and runs the SFINCS model for a given scenario.

    Parameters:
    - root_folder: The root directory for the model and data.
    - sim_name: Name of the scenario to run.
    - storm: Storm scenario to simulate.
    - data_libs: Libraries of data used in the model.
    - grid_setup: Configuration for the model grid.
    - tref_off: Reference time for the offshore model.
    - tstart_off: Start time for the model simulation.
    - tstop_off: Stop time for the model simulation.
    """
    # Create the base model for offshore simulation
    print("1 - create base model")
    create_sfincs_base_model(root_folder=root_folder, scenario='offshore_base'+base_name, storm=storm, data_libs=data_libs,
                             bbox=None, grid_setup=grid_setup, topo_map='beira_dem', res=None,
                             tref=tref_off, tstart=tstart_off, tstop=tstop_off)
    print("2 - generate forcing")
    # Generate forcings for the offshore model matching the grid
    generate_forcing(grid_setup, root_folder=root_folder, scenario=sim_name+base_name,
                     data_libs=data_libs, tref=tref_off, tstart=tstart_off, tstop=tstop_off, 
                     forcing_catalog=forcing_catalog, forcings=['wind', 'pressure'])
    
    print("3 - create folder for scenario")
    copy_entire_folder(rf'{root_folder}quadtree_beira_base{base_name}', rf'{root_folder}quadtree_{sim_name}{base_name}')
    
    print("4 - copy forcings")
    # Add the forcings from forcing into new folder
    copy_two_files(rf'{root_folder}quadtree_{sim_name}{base_name}_forcing', 
                   rf'{root_folder}quadtree_{sim_name}{base_name}', 'press_2d.nc', 'wind_2d.nc')

#################
# Running SFINCS / RUN MODEL
#################
def check_finished(root):
    finished = False
    if isfile(join(root, 'sfincs.log')):
        with open(join(root, 'sfincs.log'), 'r') as f:
            finished = np.any(['Simulation is finished' in l for l in f.readlines()])
    return finished

def run_sfincs(base_root, fn_exe):
    runs = [dirname(fn) for fn in glob.glob(join(base_root, 'sfincs.inp')) if not check_finished(dirname(fn))]
    n = len(runs)
    print(n)
    if n == 0:
        print('No simulations run because simulation is finished')
    for i, root in enumerate(runs):
        print(f'{i+1:d}/{n:d}: {base_root}')
        with open(join(base_root, "sfincs.log"), 'w') as f:
            p = subprocess.Popen([fn_exe], stdout=f, cwd=root)
            p.wait()
            print('Check sfincs.log inside folder for progress.')

##########################################################################################
if __name__ == "__main__":            

    # GENERATE HYDROMT CONFIG FILE
    fn_exe = r"D:\paper_4\data\sfincs_model\SFINCS_v2.0.3_Cauberg_release_exe\sfincs.exe"
    root_folder = 'D:/paper_4/data/sfincs_input/'  #'D:/paper_4/data/version006_quadtree_8km_500m_60s/'
    os.chdir(root_folder)

    # grid setup    
    bbox_beira = [34.8180412,-19.8658097,34.939334,-19.76172]
    grid_setup = {
        "x0": 794069.0,
        "y0": 7416897.0,
        "dx": 8000.0,
        "dy": 8000.0,
        "nmax": 77,
        "mmax": 112,
        "rotation": 55,
        "epsg": 32736,}

    # general parameters
    res = 5
    storm = 'idai'
    sim_name = 'ifs_rebuild_bc'
    
    # offshore simulation dates
    tref_off = '20190301 000000' # Keep 20190301 000000 for offshore model - it doesn't work with other dates
    tstart_off = '20190313 000000' # Keep 20190301 000000 for offshore model - it doesn't work with other dates
    tstop_off = '20190317 000000' # Keep 20190317 120000 for offshore model - it doesn't work with other dates
    hours_shifted = -137 # difference in hours for peak tide in Beira

    # offshor simulation dates for the high tide
    tref_off_hightide = tref_off #'20190301 000000' # Keep 20190301 000000 for offshore model - it doesn't work with other dates
    tstart_off_hightide = add_hours_to_time(tstart_off, hours_shifted)#'20190307 070000'
    tstop_off_hightide = add_hours_to_time(tstop_off, hours_shifted) #'20190311 070000'

    # onshore simulation dates
    tref = '20190314 000000' # Keep 20190301 000000 for offshore model - it doesn't work with other dates
    tstart = '20190314 000000' # Keep 20190301 000000 for offshore model - it doesn't work with other dates
    tstop = '20190316 000000' # Keep 20190317 120000 for offshore model - it doesn't work with other dates

    tref_hightide = add_hours_to_time(tref, hours_shifted) #'20190308 070000' # Keep 20190301 000000 for offshore model - it
    tstart_hightide = add_hours_to_time(tstart, hours_shifted) # '20190308 070000' # Keep 20190301 000000 for offshore model - it
    tstop_hightide = add_hours_to_time(tstop, hours_shifted) #'20190310 070000' # Keep 20190317 120000 for offshore model - it

    hold_params = {
        "structures": r"D:\paper_4\data\qgis\beira_seawall.geojson",
        "stype": "weir",
        "dz": 2,
    }

    retreat_params = {
        "structures": r"D:\paper_4\data\qgis\beira_internal_seawall.geojson",
        "stype": "weir",
        "dz": 2,
    }

    # libraries
    data_libs = ['d:/paper_4/data/data_catalogs/data_catalog_converter.yml', root_folder+f'data_deltares_{storm}/data_catalog.yml']
    list_indices_storm = ['merit_hydro','gebco', 'osm_coastlines', 'osm_landareas', 'gswo', 'fabdem', 'dtu10mdt', 'gcn250', 'vito', "rivers_lin2019_v1"]

    # Preparing data for SFINCS and creation of a catalog
    # #### this takes a lot of time so do it once
    # clip_data_to_region(bbox_beira, export_dir = f'data_deltares_test', data = ['deltares_data', data_libs[0]], list_indices = list_indices_storm)
    # correct_fabdem_nas(path_folder = f'data_deltares_test', path_catalogue = f'data_deltares_test/data_catalog.yml')
    # ####

    # ### Add to catalogue the new data ##############################################
    # data_catalog_edit(rf'd:\paper_4\data\data_catalogs\data_catalog_converter.yml', 'era5_hourly_sample', 
    #                     f'era5_hourly_test', f'd:/paper_4/data/era5/era5_hourly_vars_idai_single_2019_03.nc')

    # data_catalog_edit(rf'd:\paper_4\data\data_catalogs\data_catalog_converter.yml', 'era5_hourly_sample', 
    #                     f'era5_hourly_bc', f'd:/paper_4/data/seas5/bias_corrected/era5_hourly_vars_idai_single_2019_03_bc.nc')

    # # add the 5d shifted data to the catalog
    # data_catalog_edit(rf'd:\paper_4\data\data_catalogs\data_catalog_converter.yml', 'era5_hourly_sample', 
    #                     f'era5_hourly_5d_shifted', f'd:/paper_4/data/era5/era5_hourly_vars_idai_single_2019_03_5d_shifted.nc')
    # # add GPM data
    # data_catalog_edit(rf'd:\paper_4\data\data_catalogs\data_catalog_converter.yml', 'gpm_30min_sample',
    #                         f'gpm_30min_idai_hourly', f'd:/paper_4/data/nasa_data/gpm_imerg_201903_hourly.nc')
    # # add IFS data
    # data_catalog_edit(rf'd:\paper_4\data\data_catalogs\data_catalog_converter.yml', 'ifs_ens_hourly_test',
    #                             f'ifs_ens_idai_hourly', f'd:/paper_4/data/seas5/bias_corrected/ecmwf_eps_pf_010_vars_n50_s60_20190313_bc.nc')

    # Scenario creation
    # shift data to make the surge occurs at the same time as the max tide
    shifted_data = shift_time_in_netcdf("d:/paper_4/data/seas5/bias_corrected/ecmwf_oper_fc_rebuild_bc.nc",
                                        "d:/paper_4/data/seas5/bias_corrected/ecmwf_oper_fc_rebuild_bc_hightide.nc", 
                                        hours_shifted)  
    
    # Increase in precpitation due to clausius claperyon relation of 7% per degree
    degrees_projection = 3 - 1.1 #3c - t at 2019
    scaled_precip_data = scale_precip_in_netcdf("d:/paper_4/data/seas5/bias_corrected/ecmwf_oper_fc_rebuild_bc.nc",
                                                "d:/paper_4/data/seas5/bias_corrected/ecmwf_oper_fc_rebuild_bc_tp_3c.nc",
                                                1 + (0.07 * degrees_projection))
    
    scaled_precip_data_hightide = scale_precip_in_netcdf("d:/paper_4/data/seas5/bias_corrected/ecmwf_oper_fc_rebuild_bc_hightide.nc",
                                                "d:/paper_4/data/seas5/bias_corrected/ecmwf_oper_fc_rebuild_bc_tp_3c_hightide.nc",
                                                1 + (0.07 * degrees_projection))
    
    # Add cata do catalogs
    data_catalog_edit(rf'd:\paper_4\data\data_catalogs\data_catalog_converter.yml', 'ifs_ens_hourly_test',
                                f'{sim_name}_hourly', f'd:/paper_4/data/seas5/bias_corrected/ecmwf_oper_fc_rebuild_bc.nc')
    
    data_catalog_edit(rf'd:\paper_4\data\data_catalogs\data_catalog_converter.yml', 'ifs_ens_hourly_test',
                                f'{sim_name}_hightide_hourly', f'd:/paper_4/data/seas5/bias_corrected/ecmwf_oper_fc_rebuild_bc_hightide.nc')  
    
    data_catalog_edit(rf'd:\paper_4\data\data_catalogs\data_catalog_converter.yml', 'ifs_ens_hourly_test',
                                f'{sim_name}_tp_3c_hourly', f'd:/paper_4/data/seas5/bias_corrected/ecmwf_oper_fc_rebuild_bc_tp_3c.nc')
    
    data_catalog_edit(rf'd:\paper_4\data\data_catalogs\data_catalog_converter.yml', 'ifs_ens_hourly_test',
                                f'{sim_name}_tp_3c-hightide_hourly', f'd:/paper_4/data/seas5/bias_corrected/ecmwf_oper_fc_rebuild_bc_tp_3c_hightide.nc')
    

    ################################################################################

    # 2) GENERATE offshore SFINCS MODEL base
    # NEW updated function
    # historical
    setup_ofshore_sfincs_model(root_folder=root_folder, sim_name=sim_name, base_name='', storm=storm, 
                            data_libs=data_libs, grid_setup=grid_setup, tref_off=tref_off, 
                            tstart_off=tstart_off, tstop_off=tstop_off, forcing_catalog=f'{sim_name}_hourly')

    # high tide
    setup_ofshore_sfincs_model(root_folder=root_folder, sim_name=sim_name, base_name='_hightide', storm=storm, 
                            data_libs=data_libs, grid_setup=grid_setup, tref_off=tref_off_hightide, 
                            tstart_off=tstart_off_hightide, tstop_off=tstop_off_hightide, forcing_catalog=f'{sim_name}_hightide_hourly')

    # from here it's outdated VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    # Convert the ERA5 data to be compatible with SFINCS and clip it to the offshore model grid
    create_sfincs_base_model(root_folder = root_folder, scenario = 'offshore_base', storm = storm, data_libs = data_libs,
                            bbox = None, grid_setup = grid_setup, topo_map = 'beira_dem', res = None,
                            tref = tref_off, tstart = tstart_off, tstop = tstop_off)

    # #TODO: choose either the update sfincs model function or the generate_forcing function as they seem to be doing the same thing.
    # update_sfincs_model(base_root = r'D:\paper_4\data\sfincs_input\test_era5', new_root = rf'D:\paper_4\data\quadtree_era5_forcing', 
    #                     data_libs = data_libs, mode = 'era5', waterlevel_path=None, slr=None)

    # ####### Create forcings for offshore model matching the grid
    generate_forcing(grid_setup, root_folder = root_folder, scenario = sim_name,
                    data_libs = data_libs, tref = tref_off, tstart = tstart_off, tstop = tstop_off, 
                    forcing_catalog = f'{sim_name}_hourly', forcings=['wind', 'pressure'])
    
    # copy the folders and external forcings
    copy_entire_folder(rf'{root_folder}quadtree_beira_base', rf'{root_folder}quadtree_{sim_name}')
    # add the forcings from forcing into new folder
    copy_two_files(rf'{root_folder}quadtree_{sim_name}_forcing', 
                   rf'{root_folder}quadtree_{sim_name}', 'press_2d.nc', 'wind_2d.nc')

    # max tide scenario: change the external wind and pressure values from the 14th to the 9th, so in terms of total days: 5 days
    #TODO: not working; the inp file is being generated but we want the old inp file + the added forcings, so change that.
    create_sfincs_base_model(root_folder = root_folder, scenario = 'offshore_base_hightide', storm = storm, data_libs = data_libs,
                            bbox = None, grid_setup = grid_setup, topo_map = 'beira_dem', res = None,
                            tref = tref_off_hightide, tstart = tstart_off_hightide, tstop = tstop_off_hightide)
    
    generate_forcing(grid_setup, root_folder= root_folder,scenario = sim_name,data_libs = data_libs,
                     tref = tref_off_hightide, tstart = tstart_off_hightide, tstop = tstop_off_hightide,
                      forcing_catalog = f'{sim_name}_hightide_hourly',forcings=['wind', 'pressure'])

    copy_entire_folder(rf'D:\paper_4\data\sfincs_input\quadtree_beira_base_hightide', rf'D:\paper_4\data\sfincs_input\quadtree_{sim_name}_hightide')
    copy_two_files(rf'D:\paper_4\data\sfincs_input\quadtree_{sim_name}_forcing_hightide', 
                   rf'D:\paper_4\data\sfincs_input\quadtree_{sim_name}_hightide', 'press_2d.nc', 'wind_2d.nc')
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # #######
    # # 2) RUN OFFSHORE SFINCS MODEL 
    run_sfincs(base_root = rf'D:\paper_4\data\sfincs_input\quadtree_{sim_name}', fn_exe = fn_exe)

    # Run offshore sfincs model with max tide
    run_sfincs(base_root = rf'D:\paper_4\data\sfincs_input\quadtree_{sim_name}_hightide', fn_exe = fn_exe)

    # #######
    # 3a) generate base SFINCS model for onshore
    create_sfincs_base_model(root_folder = root_folder, scenario = 'base', storm = storm, data_libs = data_libs,
                            bbox = bbox_beira, topo_map = 'beira_dem', res = res,
                            tref = tref, tstart = tstart, tstop = tstop)
    
    # create base for hightide
    create_sfincs_base_model(root_folder = root_folder, scenario = 'base_hightide', storm = storm, data_libs = data_libs,
                            bbox = bbox_beira, topo_map = 'beira_dem', res = res,
                            tref = tref_hightide, tstart = tstart_hightide, tstop = tstop_hightide)    

    # 3b) Use storm surge on ONSHORE SFINCS models
    # if we want hazard analysis, use single hazard:
    # # surge bc
    # update_sfincs_model(base_root = f'{root_folder}{storm}_base', new_root = f'{root_folder}{storm}_{sim_name}_surge', 
    #                     data_libs = data_libs, mode = 'surge', 
    #                     waterlevel_path = rf"D:/paper_4/data/sfincs_input/quadtree_{sim_name}/sfincs_his.nc")
    # # rain bc
    # update_sfincs_model(base_root = f'{root_folder}{storm}_base', new_root = f'{root_folder}{storm}_{sim_name}_rain',
    #                     data_libs = data_libs, mode = 'rain', 
    #                     precip_path = f'{sim_name}_hourly')
    
    
    # compound simulations
    # compound: surge + rain
    slr_level = 0.64 - 3.6*0.1*(2019-2005) # 3.6mm/year IPCC *0.1cm/mm * time difference
    update_sfincs_model(base_root = f'{root_folder}{storm}_base', new_root = f'{root_folder}{storm}_{sim_name}_hist_rain_surge_noadpt',
                        data_libs = data_libs, mode = 'rain_surge', precip_path = f'{sim_name}_hourly',
                        waterlevel_path =  rf"D:/paper_4/data/sfincs_input/quadtree_{sim_name}/sfincs_his.nc")
    
    # high tide scenario
    update_sfincs_model(base_root = f'{root_folder}{storm}_base_hightide', new_root = f'{root_folder}{storm}_{sim_name}_hightide_rain_surge_noadpt',
                        data_libs = data_libs, mode = 'rain_surge', precip_path = f'{sim_name}_hightide_hourly',
                        waterlevel_path =  rf"D:/paper_4/data/sfincs_input/quadtree_{sim_name}_hightide/sfincs_his.nc")

    # now with slr = 1 + storm surge
    update_sfincs_model(base_root = f'{root_folder}{storm}_base', new_root = f'{root_folder}{storm}_{sim_name}_3c_rain_surge_noadpt',
                        data_libs = data_libs, mode = 'rain_surge', precip_path = f'{sim_name}_tp_3c_hourly',
                        waterlevel_path =  rf"D:/paper_4/data/sfincs_input/quadtree_{sim_name}/sfincs_his.nc", slr=slr_level)
    
    # high tide scenario + slr100 + 3c
    update_sfincs_model(base_root = f'{root_folder}{storm}_base_hightide', new_root = f'{root_folder}{storm}_{sim_name}_3c-hightide_rain_surge_noadpt',
                        data_libs = data_libs, mode = 'rain_surge', precip_path = f'{sim_name}_tp_3c-hightide_hourly',
                        waterlevel_path =  rf"D:/paper_4/data/sfincs_input/quadtree_{sim_name}_hightide/sfincs_his.nc", slr=slr_level)
    
    # add adaptation options
    for measure in ['hold', 'retreat']:
        adapt_params = hold_params if measure == 'hold' else retreat_params

        add_adaptation_measures(data_libs=data_libs, original_root=rf"{root_folder}{storm}_{sim_name}_hist_rain_surge_noadpt",
                                new_root=rf"{root_folder}{storm}_{sim_name}_rain_surge_{measure}", structures_params=adapt_params)
        
        add_adaptation_measures(data_libs=data_libs, original_root=rf"{root_folder}{storm}_{sim_name}_hightide_rain_surge_noadpt",
                                new_root=rf"{root_folder}{storm}_{sim_name}_hightide_rain_surge_{measure}", structures_params=adapt_params)
        
        add_adaptation_measures(data_libs=data_libs, original_root=rf"{root_folder}{storm}_{sim_name}_3c_rain_surge_noadpt",
                                new_root=rf"{root_folder}{storm}_{sim_name}_3c_rain_surge_{measure}", structures_params=adapt_params)
        
        add_adaptation_measures(data_libs=data_libs, original_root=rf"{root_folder}{storm}_{sim_name}_3c-hightide_rain_surge_noadpt",
                                new_root=rf"{root_folder}{storm}_{sim_name}_3c-hightide_rain_surge_{measure}", structures_params=adapt_params)
                            

                            
    

    # 4) RUN ONSHORE SFINCS MODEL
    # Baseline simulations
    run_sfincs(base_root = rf'D:\paper_4\data\sfincs_input\{storm}_{sim_name}_surge', fn_exe = fn_exe) # test_slr100_surge # test_rain_gpm
    run_sfincs(base_root = rf'D:\paper_4\data\sfincs_input\{storm}_{sim_name}_rain', fn_exe = fn_exe) # test_slr100_surge # test_rain_gpm
    run_sfincs(base_root = rf'D:\paper_4\data\sfincs_input\{storm}_{sim_name}_hist_rain_surge', fn_exe = fn_exe) # test_slr100_surge # test_rain_gpm
    
    # high tide scenario
    run_sfincs(base_root = rf'D:\paper_4\data\sfincs_input\{storm}_{sim_name}_hightide_rain_surge', fn_exe = fn_exe) # test_slr100_surge # test_rain_gpm

    # SLR scenario (check if we want to model each hazard separately)
    run_sfincs(base_root = rf'D:\paper_4\data\sfincs_input\{storm}_{sim_name}_3c_rain_surge', fn_exe = fn_exe) # test_slr100_surge # test_rain_gpm

    # high tide scenario
    run_sfincs(base_root = rf'D:\paper_4\data\sfincs_input\{storm}_{sim_name}_3c_hightide_rain_surge', fn_exe = fn_exe) # test_slr100_surge # test_rain_gpm

    # CC + high tide scenario
    


    # scenario base + adaptation
    run_sfincs(base_root = rf'{root_folder}{storm}_{sim_name}_rain_surge_floodwall', fn_exe = fn_exe) # test_slr100_surge # test_rain_gpm
    # scenario SLR + adaptation
    run_sfincs(base_root = rf'{root_folder}{storm}_{sim_name}_hightide_rain_surge_floodwall', fn_exe = fn_exe) # test_slr100_surge # test_rain_gpm
    # scenario SLR + CC + adaptation
    run_sfincs(base_root = rf'{root_folder}{storm}_{sim_name}_3c_rain_surge_floodwall', fn_exe = fn_exe) # test_slr100_surge # test_rain_gpm
    # scenario SLR + CC + high tide + adaptation
    run_sfincs(base_root = rf'{root_folder}{storm}_{sim_name}_3c_hightide_rain_surge_floodwall', fn_exe = fn_exe) # test_slr100_surge # test_rain_gpm



    # mod_nr = generate_maps(sfincs_root = r'D:\paper_4\data\sfincs_input\test_base_surge', catalog_path = f'D:\paper_4\data\sfincs_input\test_base\hydromt_data.yml', storm = storm, scenario = 'surge', mode = 'surge')
##########################################################################################
    
    mod_nr = SfincsModel(r'D:\paper_4\data\sfincs_input\test_rain_gpm', mode="r")
    mod_nr.read_results()

    mod_era5 = SfincsModel(r'D:\paper_4\data\sfincs_input\test_rain', mode="r")
    mod_era5.read_results()
    # mod_nr.write_raster(f"results.hmax", compress="LZW")
    # _ = mod_nr.plot_forcing()

    da_hmax = mod_nr.results["hmax"].max(['timemax'])
    da_hmax = da_hmax.where(da_hmax > 0.1)

    da_hmax_era5 = mod_era5.results["hmax"].max(['timemax'])
    da_hmax_era5 = da_hmax_era5.where(da_hmax_era5 > 0.1)

    # calculate difference between da_hmax_era5 and da_hmax
    da_hmax_diff = da_hmax_era5 - da_hmax



    fig, ax = mod_nr.plot_basemap(
        fn_out=None,
        figsize=(16, 12),
        variable=da_hmax,
        plot_bounds=False,
        plot_geoms=False,
        bmap="sat",
        zoomlevel=14,
        vmin=0.0,
        vmax=1.0,
        alpha=0.8,
        cmap=plt.cm.Blues,
        cbar_kwargs = {"shrink": 0.6, "anchor": (0, 0)}
    )
    # ax.set_title(f"SFINCS maximum water depth")
    # plt.savefig(join(mod.root, 'figs', 'hmax.png'), dpi=225, bbox_inches="tight")

    plt.show()



    # open the sfincs_map.nc file and plot the map
    ds = xr.open_dataset(f'D:\paper_4\data\sfincs_input\quadtree_ifs_rebuild_bc\sfincs_map.nc')
    ds_his = xr.open_dataset(f'D:\paper_4\data\sfincs_input\quadtree_ifs_rebuild_bc\sfincs_his.nc')
    ds_his_maxtide = xr.open_dataset(f'D:\paper_4\data\sfincs_input\quadtree_ifs_rebuild_bc_hightide\sfincs_his.nc')
    ds_his_tides = xr.open_dataset(f'D:\paper_4\data\quadtree_era5\sfincs_his_tides.nc')
    ds_his_spider = xr.open_dataset(f'D:\paper_4\data\quadtree_era5\sfincs_his_spider.nc')
    ds_his_era5 = xr.open_dataset(f'D:\paper_4\data\quadtree_era5\sfincs_his_era5.nc')
    # ds_truth = xr.open_dataset(rf'P:\11209202-tc-risk-analysis-2023\01_Beira_forecasting_TCFF\04_model_runs\20230509_variations_Python_based_n10000\days_before_landfall_1\ensemble09430\sfincs_his.nc')
    # ds_truth2 = xr.open_dataset(rf'P:\11209202-tc-risk-analysis-2023\01_Beira_forecasting_TCFF\04_model_runs\20230310_variations_Python_based\tide_only\sfincs_his.nc')
    # get maximum value of point_zs for each ds_his and ds_truth
    ds_his_spider['point_zs'].max()
    # ds_truth['point_zs'].max()

    # plot the timeseries for point_zs for station 'station_name' == 10
    ds_his['point_zs'].isel(stations = 4).plot(label = 'ifs bc', linewidth = 2)
    # ds_his_era5['point_zs'].isel(stations = 4).plot(label = 'era5')
    ds_his_tides['point_zs'].isel(stations = 4).plot(label = 'kees_tides_only')
    ds_his_spider['point_zs'].isel(stations = 4).plot(label = 'kees_spiderweb')
    ds_his_maxtide['point_zs'].isel(stations = 4).plot(label = 'high tide', linewidth = 2)
    # ds_truth2['point_zs'].isel(stations = 4).plot(label = 'kees2_tides_only')
    plt.legend()
    plt.show()

    print(ds_his['point_zs'].isel(stations = 4).max().item(), "which correspond to ")
    print(ds_his_maxtide['point_zs'].isel(stations = 4).max().item())
    print(ds_his_spider['point_zs'].isel(stations = 4).max().item())

    # find the time value where the maximum value of ds_his['point_zs'].isel(stations = 4) occurs
    ds_his['point_zs'].isel(stations = 4).argmax()
    time_peak_tide = ds_his['point_zs'].isel(stations = 4, time = ds_his['point_zs'].isel(stations = 4).argmax()).time.values
    # now find for ds_his_tides['point_zs'].isel(stations = 4) the same time from the previous step
    ds_his['point_zs'].isel(stations = 4).sel(time =time_peak_tide).item() - ds_his_tides['point_zs'].isel(stations = 4).sel(time =time_peak_tide).item()


    # find time for maximu mvalue of ds_his_tides['point_zs'].isel(stations = 4)
    ds_his_spider['point_zs'].isel(stations = 4).argmax()
    max_all = ds_his_spider['point_zs'].isel(stations = 4)[990]

    ds_his_tides['point_zs'].isel(stations = 4).argmax()
    max_tides = ds_his_tides['point_zs'].isel(stations = 4)[168]

    # find the time difference between the max_all and max_tides and convert into hours
    time_diff = max_all['time'] - max_tides['time']
    time_diff_hours = time_diff.values.astype('timedelta64[h]')


    # open precip.nc in D:\paper_3\data\sfincs_ini\spectral_nudging\sandy_counter_2_rain_surge
    ds2 = xr.open_dataset(r'D:\paper_3\data\sfincs_ini\spectral_nudging\sandy_counter_2_rain_surge\precip.nc')

    ds3 = xr.open_dataset(r'D:\paper_4\data\era5\era5_hourly_vars_idai_single_2019_03.nc')

    ds4 = xr.open_dataset(r'D:\paper_4\data\sfincs_input\test_rain\precip_2d.nc')
    ds_wind = xr.open_dataset(r'D:\paper_4\data\sfincs_input\test_forcing\wind_2d.nc')
    # calculate total wind speed based on  eastward_wind   (time, y, x) float32 northward_wind  (time, y, x) 
    ds_wind['wind_speed'] = np.sqrt(ds_wind['eastward_wind']**2 + ds_wind['northward_wind']**2)


    # Select a specific time slice
    precip = ds_wind['wind_speed'].isel(time=160)
    precip = precip.where(precip > 0)
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot the data
    precip.plot(ax=ax)
    # Set a title with the time of the data
    ax.set_title(precip.time.values)

    # Show the plot
    plt.show()

    # plot ds4['Precipitation'] timeseries for x and y == 742438.71 7748921.02 
    ds4['Precipitation'].sel(x=742438.71, y=7748921.02, method='nearest').plot()
    plt.show()

    ds_wind['wind_speed'].sel(x=742438.71, y=7748921.02, method='nearest').plot()
    plt.show()



import rasterio
import geopandas as gpd

# load Downloads/Clipped_elevation.tiff
elev = rasterio.open(r'C:\Users\morenodu\Downloads\Clipped_elevation (2).tif')
# get maximum value of elev
elev.read().min()

# plot elev with colormap from min to max
plt.imshow(elev.read(1), cmap='terrain')
plt.colorbar()
plt.show()

# load the vector mask
mask = gpd.read_file(r'C:\Users\morenodu\Downloads\mask_layer.shp')
# mask the raster with the vector mask
out_image, out_transform = rasterio.mask.mask(elev, mask.geometry, crop=True)
# plot the masked raster
plt.imshow(out_image[0], cmap='terrain')
plt.colorbar()







