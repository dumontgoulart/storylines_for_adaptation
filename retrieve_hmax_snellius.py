import xarray as xr
import numpy as np
import os
from os.path import join, isfile, isdir, dirname

import hydromt
from hydromt_sfincs import SfincsModel

storm = 'idai'

adapt_scenarios = ["noadapt","hold","retreat"]

physical_scenarios = ['hist','hightide','3c','3c-hightide']

sim = 'idai_ifs_rebuild_bc'

combinations = [f'{sim}_{physical}_rain_surge_{adapt}' for physical in physical_scenarios for adapt in adapt_scenarios]

def generate_maps(sfincs_root, catalog_path, storm, scenario):
  mod = SfincsModel(sfincs_root, mode="r")
  mod.read_results()
  #mod.write_raster("results.hmax", compress="LZW")
  
  mod.data_catalog.from_yml(catalog_path)
  gswo = mod.data_catalog.get_rasterdataset(f"/gpfs/work3/0/einf4318/paper_4/sfincs/data_deltares_{storm}/gswo.tif", geom=mod.region, buffer=10)
  # permanent water where water occurence > 5%
  gswo_mask = gswo.raster.reproject_like(mod.grid, method="max") <= 5

  da_hmax = mod.results["hmax"].max(['timemax'])
  da_hmax = da_hmax.where(gswo_mask).where(da_hmax > 0.05)
  # update attributes for colorbar label later
  da_hmax.attrs.update(long_name="flood depth", unit="m")
  # check it's in north-up order
  if da_hmax.y.values[0] < da_hmax.y.values[-1]:
      # Flip vertically
      da_hmax = da_hmax[::-1, :]
      print("Flipped the raster as it was not in north-up order.")
  else:
      print("Raster already in north-up order, no flip needed.")
  
  # create folder if not exists:
  path_to_folder = rf'/gpfs/work3/0/einf4318/paper_4/sfincs/output/{storm}'
  if not os.path.exists(path_to_folder):
      os.makedirs(path_to_folder)
      os.makedirs(path_to_folder+r"/raster")
  
  da_hmax.to_netcdf(rf'{path_to_folder}/hmax_{scenario}.nc') 
  da_hmax.rio.to_raster(rf'{path_to_folder}/raster/hmax_{scenario}.tiff', tiled=True, compress='LZW')
  
for scenario in combinations:
  mod = generate_maps(sfincs_root = f"/gpfs/work3/0/einf4318/paper_4/sfincs/{scenario}", catalog_path = f'/gpfs/work3/0/einf4318/paper_4/sfincs/data_deltares_{storm}/data_catalog.yml', storm = storm, scenario = scenario)