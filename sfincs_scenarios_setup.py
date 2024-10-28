# -*- coding: utf-8 -*-
"""
Prepare files for HydroMT
Created on Tue Oct  9 16:39:47 2022
@author: morenodu
"""
import sys
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import configparser
import yaml

import hydromt
from hydromt_sfincs import SfincsModel, utils

import sfincs_scenario_functions as sfincs_scen


##########################################################################################
# GENERATE HYDROMT CONFIG FILE
if __name__ == "__main__":
    # GENERATE HYDROMT CONFIG FILE
    fn_exe = (
        r"D:\paper_4\data\sfincs_model\SFINCS_v2.0.3_Cauberg_release_exe\sfincs.exe"
    )
    root_folder = "D:/paper_4/data/sfincs_input/"  #'D:/paper_4/data/version006_quadtree_8km_500m_60s/'
    os.chdir(root_folder)

    # grid setup
    bbox_beira = [34.8180412, -19.8658097, 34.939334, -19.76172]
    grid_setup = {
        "x0": 794069.0,
        "y0": 7416897.0,
        "dx": 8000.0,
        "dy": 8000.0,
        "nmax": 77,
        "mmax": 112,
        "rotation": 55,
        "epsg": 32736,
    }

    # general parameters ###
    res = 5
    storm = "idai"
    sim_name = "ifs_rebuild_bc"

    # climate parameters ###
    slr_level = (
        0.64 - 3.6 * (2019 - 2005) * 0.001
    )  # 3.6mm/year IPCC *0.1cm/mm * time difference# meters
    slr15_level = 0.46 - 3.6 * (2019 - 2005) * 0.001
    degrees_projection = 3 - 1.2  # check # celsius degrees
    degrees_projection_15 = 1.5 - 1.2  # check # celsius degrees
    hours_shifted = (
        -137
    )  # difference in hours for peak tide in Beira - hightide scenario

    # offshore simulation dates
    tref_off = "20190301 000000"  # Keep 20190301 000000 for offshore model - it doesn't work with other dates
    tstart_off = "20190313 000000"  # Keep 20190301 000000 for offshore model - it doesn't work with other dates
    tstop_off = "20190317 000000"  # Keep 20190317 120000 for offshore model - it doesn't work with other dates

    # offshor simulation dates for the high tide
    tref_off_hightide = tref_off  #'20190301 000000' # Keep 20190301 000000 for offshore model - it doesn't work with other dates
    tstart_off_hightide = sfincs_scen.add_hours_to_time(
        tstart_off, hours_shifted
    )  #'20190307 070000'
    tstop_off_hightide = sfincs_scen.add_hours_to_time(
        tstop_off, hours_shifted
    )  #'20190311 070000'

    # onshore simulation dates
    tref = "20190314 000000"  # Keep 20190301 000000 for offshore model - it doesn't work with other dates
    tstart = "20190314 000000"  # Keep 20190301 000000 for offshore model - it doesn't work with other dates
    tstop = "20190316 000000"  # Keep 20190317 120000 for offshore model - it doesn't work with other dates

    tref_hightide = sfincs_scen.add_hours_to_time(
        tref, hours_shifted
    )  #'20190308 070000' # Keep 20190301 000000 for offshore model - it
    tstart_hightide = sfincs_scen.add_hours_to_time(
        tstart, hours_shifted
    )  # '20190308 070000' # Keep 20190301 000000 for offshore model - it
    tstop_hightide = sfincs_scen.add_hours_to_time(
        tstop, hours_shifted
    )  #'20190310 070000' # Keep 20190317 120000 for offshore model - it

    # adaptation scenarios
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

    # data and libraries
    data_libs = [
        "d:/paper_4/data/data_catalogs/data_catalog_converter.yml",
        root_folder + f"data_deltares_{storm}/data_catalog.yml",
    ]

    list_indices_storm = [
        "merit_hydro",
        "gebco",
        "osm_coastlines",
        "osm_landareas",
        "gswo",
        "fabdem",
        "dtu10mdt",
        "gcn250",
        "vito",
        "rivers_lin2019_v1",
    ]

    ##########################################################################################
    # optional step: create data catalog
    # sfincs_scen.clip_data_to_region(bbox_beira, export_dir = f'data_deltares_idai', data = ['deltares_data', data_libs[0]], list_indices = list_indices_storm)

    # 1) Offshore model
    # historical
    sfincs_scen.setup_ofshore_sfincs_model(
        root_folder=root_folder,
        sim_name=sim_name,
        base_name="",
        storm=storm,
        data_libs=data_libs,
        grid_setup=grid_setup,
        tref_off=tref_off,
        tstart_off=tstart_off,
        tstop_off=tstop_off,
        forcing_catalog=f"{sim_name}_hourly",
    )

    # high tide
    sfincs_scen.setup_ofshore_sfincs_model(
        root_folder=root_folder,
        sim_name=sim_name,
        base_name="_hightide",
        storm=storm,
        data_libs=data_libs,
        grid_setup=grid_setup,
        tref_off=tref_off_hightide,
        tstart_off=tstart_off_hightide,
        tstop_off=tstop_off_hightide,
        forcing_catalog=f"{sim_name}_hightide_hourly",
    )

    # # # 2) RUN OFFSHORE SFINCS MODEL
    # sfincs_scen.run_sfincs(base_root = f'D:\paper_4\data\sfincs_input\quadtree_{sim_name}', fn_exe = fn_exe)

    # # Run offshore sfincs model with max tide
    # sfincs_scen.run_sfincs(base_root = rf'D:\paper_4\data\sfincs_input\quadtree_{sim_name}_hightide', fn_exe = fn_exe)

    ##########################################################################################
    # 2) Onshore model
    sfincs_scen.create_sfincs_base_model(
        root_folder=root_folder,
        scenario="base",
        storm=storm,
        data_libs=data_libs,
        bbox=bbox_beira,
        topo_map="beira_dem",
        res=res,
        tref=tref,
        tstart=tstart,
        tstop=tstop,
    )

    # create base for hightide
    sfincs_scen.create_sfincs_base_model(
        root_folder=root_folder,
        scenario="base_hightide",
        storm=storm,
        data_libs=data_libs,
        bbox=bbox_beira,
        topo_map="beira_dem",
        res=res,
        tref=tref_hightide,
        tstart=tstart_hightide,
        tstop=tstop_hightide,
    )

    # Define physical and adaptation scenarios
    physical_scenarios = {
        "hist_rain_surge": {
            "base_root_suffix": "base",
            "scenario_root_suffix": f"{sim_name}_hist_rain_surge_noadapt",
            "precip_path_suffix": "hourly",
            "waterlevel_path_folder": "",
        },
        "3c_rain_surge": {
            "base_root_suffix": "base",
            "scenario_root_suffix": f"{sim_name}_3c_rain_surge_noadapt",
            "precip_path_suffix": "tp_3c_hourly",
            "waterlevel_path_folder": "",
            "slr": slr_level,
        },
        "hightide_rain_surge": {
            "base_root_suffix": "base_hightide",
            "scenario_root_suffix": f"{sim_name}_hightide_rain_surge_noadapt",
            "precip_path_suffix": "hightide_hourly",
            "waterlevel_path_folder": "_hightide",
        },
        "3c-hightide_rain_surge": {
            "base_root_suffix": "base_hightide",
            "scenario_root_suffix": f"{sim_name}_3c-hightide_rain_surge_noadapt",
            "precip_path_suffix": "tp_3c-hightide_hourly",
            "waterlevel_path_folder": "_hightide",
            "slr": slr_level,
        },
    }

    for scenario_name, scenario_config in physical_scenarios.items():
        scenario_root = (
            f'{root_folder}{storm}_{scenario_config["scenario_root_suffix"]}'
        )
        print(f"Creating scenario: {scenario_name}")
        sfincs_scen.update_sfincs_model(
            base_root=f'{root_folder}{storm}_{scenario_config["base_root_suffix"]}',
            new_root=scenario_root,
            data_libs=data_libs,
            mode="rain_surge",
            precip_path=f'{sim_name}_{scenario_config["precip_path_suffix"]}',
            waterlevel_path=rf'{root_folder}quadtree_{sim_name}{scenario_config["waterlevel_path_folder"]}/sfincs_his.nc',
            slr=scenario_config.get("slr"),
        )

        for measure in ["hold", "retreat"]:
            adapt_root = scenario_root.replace("_noadapt", f"_{measure}")
            adapt_params = hold_params if measure == "hold" else retreat_params
            # add adaptation measures
            sfincs_scen.add_adaptation_measures(
                data_libs=data_libs,
                original_root=scenario_root,
                new_root=adapt_root,
                structures_params=adapt_params,
            )

    # optional: run locally sfincs model
    sfincs_scen.run_sfincs(
        base_root=rf"D:\paper_4\data\sfincs_input\{storm}_{sim_name}_3c-hightide_rain_surge_noadapt",
        fn_exe=fn_exe,
    )
