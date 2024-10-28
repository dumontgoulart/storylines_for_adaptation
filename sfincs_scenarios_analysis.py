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

import hydromt
from hydromt_sfincs import SfincsModel, utils

#general params   
base_folder = 'D:/paper_4/data/sfincs_input/'  #'D:/paper_4/data/version006_quadtree_8km_500m_60s/'
storm = 'idai'
data_libs = ['d:/paper_4/data/data_catalogs/data_catalog_converter.yml', base_folder+rf'/data_deltares_{storm}/data_catalog.yml']

# choose scenario
scenario = 'idai_ifs_rebuild_bc_hist_rain_surge_noadapt' #test_surge_ifs_rebuild_idai_bc test_rain_ifs_cf_bc
tif_file = rf'D:\paper_4\data\sfincs_output\test\{scenario}.tiff'

mod_nr = SfincsModel(base_folder+scenario, data_libs = data_libs, mode="r") #test_rain_gpm
    # we can simply read the model results (sfincs_map.nc and sfincs_his.nc) using the read_results method
mod_nr.read_results()
# mod_nr.write_raster(f"results.hmax", compress="LZW")
# _ = mod_nr.plot_forcing()
landmask = mod_nr.data_catalog.get_geodataframe(f"D:\paper_4\data\sfincs_input\data_deltares_idai\osm_landareas.gpkg")

da_hmax = mod_nr.results["hmax"].max(['timemax'])
mask = da_hmax.raster.geometry_mask(landmask)

da_h = mod_nr.results["h"].isel(time=-1)
da_h = da_h.where(da_h > 0.05).where(mask)

da_hmax = da_hmax.where(da_hmax > 0.05).where(mask)

# update attributes for colorbar label later
da_h.attrs.update(long_name="flood depth", unit="m")
# check it's in north-up order
if da_h.y.values[0] < da_h.y.values[-1]:
    # Flip vertically
    da_h = da_h[::-1, :]
    print("Flipped the raster as it was not in north-up order.")
else:
    print("Raster already in north-up order, no flip needed.")

fig, ax = mod_nr.plot_basemap(
    fn_out=None,
    figsize=(16, 12),
    variable=da_h,
    plot_bounds=False,
    plot_geoms=False,
    bmap="sat",
    zoomlevel=14,
    vmin=0.0,
    vmax=5.0,
    alpha=0.8,
    cmap=plt.cm.Blues,
    cbar_kwargs = {"shrink": 0.6, "anchor": (0, 0)}
)
# ax.set_title(f"SFINCS maximum water depth")
# plt.savefig(join(mod.root, 'figs', 'hmax.png'), dpi=225, bbox_inches="tight")
plt.show()

da_hmax.rio.to_raster(tif_file, tiled=True, compress='LZW')


# load this file D:\paper_4\data\FloodAdapt-GUI\Database\beira\output\Scenarios\idai_ifs_rebuild_bc_hist_rain_surge_noadapt\Flooding\simulations\overland\sfincs_map.nc

sfincs_map = xr.open_dataset(r'D:\paper_4\data\version006_quadtree_8km_500m_60s_copy\sfincs_map.nc')

# now find the "h" for the last time step
sfincs_map_1 = sfincs_map['h'].isel(time=-1)

# plot results
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
sfincs_map_1.plot(ax=ax, cmap='Blues', add_colorbar=True)
ax.set_title('Flood depth [m]')
plt.show()


# plot sfincs_map.zsmax using lat and lon 
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
sfincs_map.zsmax.isel(timemax=0).plot()
ax.set_title('Maximum water level [m]')
plt.show()

# load tiff file from D:\paper_4\data\floodadapt_results\hazard_tiff\hmax_idai_ifs_rebuild_bc_hist_rain_surge_noadapt.tif using rasterio and plot a map
# load tiff file from D:\paper_4\data\mester_TC_Idai_data_collection\shp_template\cfwindzos065_no_dif.tif using rasterio and plot a map
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

mester_flood = r'D:\paper_4\data\mester_TC_Idai_data_collection\shp_template\cfwindzos065_no_dif.tif'
sfincs_flood = r'D:\paper_4\data\floodadapt_results\hazard_tiff\hmax_idai_ifs_rebuild_bc_hist_rain_surge_noadapt.tiff'
reprojected_tiff2_path = r'D:\paper_4\data\mester_TC_Idai_data_collection\shp_template\hmax_idai_ifs_rebuild_bc_hist_rain_surge_noadapt.tif'

# Load the first GeoTIFF file
with rasterio.open(mester_flood) as src1:
    tiff1 = src1.read(1)
    tiff1_meta = src1.meta
    tiff1_transform = src1.transform
    tiff1_crs = src1.crs

# Load the second GeoTIFF file
with rasterio.open(sfincs_flood) as src2:
    tiff2 = src2.read(1)
    tiff2_meta = src2.meta
    tiff2_transform = src2.transform
    tiff2_crs = src2.crs


from rasterio.warp import calculate_default_transform, reproject, Resampling
with rasterio.open(sfincs_flood) as src1:
    # Calculate the transform, width, and height for the new raster
    transform, width, height = calculate_default_transform(
        src1.crs, 
        tiff2_crs, 
        src1.width, 
        src1.height, 
        *src1.bounds)

    # Update metadata for the new file
    kwargs = src1.meta.copy()
    kwargs.update({
        'crs': tiff1_crs,  # set to match tiff2's CRS
        'transform': transform,
        'width': width,
        'height': height
    })

    # Reproject the raster
    with rasterio.open(reprojected_tiff2_path, 'w', **kwargs) as dst:
        for i in range(1, src1.count + 1):
            reproject(
                source=rasterio.band(src1, i),
                destination=rasterio.band(dst, i),
                src_transform=src1.transform,
                src_crs=src1.crs,
                dst_transform=transform,
                dst_crs=tiff2_crs,
                resampling=Resampling.nearest)

# Load the second GeoTIFF file
with rasterio.open(reprojected_tiff2_path) as src2:
    tiff2 = src2.read(1)
    tiff2_meta = src2.meta
    tiff2_transform = src2.transform
    tiff2_crs = src2.crs

# Ensure both rasters have the same CRS
if tiff1_crs != tiff2_crs:
    raise ValueError("CRS do not match. Please reproject one of the rasters to match the other.")


tiff1_masked = np.ma.masked_where(tiff1 == 0, tiff1)

# Create a plot with subplots
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the first tiff image
show(tiff1_masked, ax=ax, cmap='gray', title="Overlay of Two TIFF Files")

# Overlay the second tiff image with some transparency (alpha)
show(tiff2, ax=ax, cmap='jet', alpha=0.5)

# Display the plot
plt.show()