import os
import shutil

def copy_and_rename_gpkg(source_dir, target_dir, sim, climate_scenarios, adapt_scenarios):
    """
    Copy spatial.gpkg files from source directory to target directory,
    renaming them based on predefined scenario names.

    :param source_dir: The source directory containing scenario subdirectories.
    :param target_dir: The target directory where .gpkg files will be copied to.
    :param sim: The simulation prefix.
    :param climate_scenarios: List of climate scenarios.
    :param adapt_scenarios: List of adaptation scenarios.
    """
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Generate all combinations of climate and adaptation scenarios
    all_adapt_scenarios = [f"{sim}{climate}_rain_surge_{adapt}" for climate in climate_scenarios for adapt in adapt_scenarios]

    # Walk through the source directory to find matching scenario directories
    for scenario in all_adapt_scenarios:
        scenario_path = os.path.join(source_dir, scenario, 'Impacts', 'fiat_model', 'output', 'spatial.gpkg')
        if os.path.exists(scenario_path):
            new_filename = f"spatial_{scenario}.gpkg"
            target_path = os.path.join(target_dir, new_filename)
            
            # Copy the file to the new location with the new name
            shutil.copy2(scenario_path, target_path)
            print(f"Copied and renamed 'spatial.gpkg' to '{target_path}'")
        else:
            print(f"Scenario not found: {scenario_path}")

# Example usage
source_dir = r"D:\paper_4\data\FloodAdapt-GUI\Database\beira\output\Scenarios"
target_dir = r"D:\paper_4\data\floodadapt_results"
sim = 'idai_ifs_rebuild_bc_'
climate_scenarios = ['hist', '3c', 'hightide', '3c-hightide']
adapt_scenarios = ['noadapt', 'hold', 'retreat']

copy_and_rename_gpkg(source_dir, target_dir, sim, climate_scenarios, adapt_scenarios)


from hydromt_sfincs import SfincsModel
import os
import shutil
import xarray as xr
import geopandas as gpd

def process_hazard_files(source_dir, target_dir, sim, climate_scenarios, adapt_scenarios, base_folder, storm, data_libs):
    """
    Process 'hazard_map.nc' files for each scenario, applying specified operations and saving as TIFF.

    :param source_dir: Directory containing scenario subdirectories.
    :param target_dir: Target directory for processed TIFF files.
    :param sim: Simulation prefix.
    :param climate_scenarios: List of climate scenarios.
    :param adapt_scenarios: List of adaptation scenarios.
    :param base_folder: Base folder for SfincsModel.
    :param storm: Storm identifier.
    :param data_libs: List of data library paths.
    """
    os.makedirs(target_dir, exist_ok=True)

    all_adapt_scenarios = [f"{sim}{climate}_rain_surge_{adapt}" for climate in climate_scenarios for adapt in adapt_scenarios]

    for scenario in all_adapt_scenarios:
        hazard_path = os.path.join(source_dir, scenario, 'Impacts', 'fiat_model', 'hazard', 'hazard_map.nc')
        if os.path.exists(hazard_path):
            tif_file = os.path.join(target_dir, f'hmax_{scenario}.tiff')
            
            mod_nr = SfincsModel(base_folder+scenario, data_libs=data_libs, mode="r")
            mod_nr.read_results()
            landmask = mod_nr.data_catalog.get_geodataframe(f"{base_folder}data_deltares_{storm}/osm_landareas.gpkg")

            da_hmax = mod_nr.results["hmax"].max(['timemax'])
            mask = da_hmax.raster.geometry_mask(landmask)
            da_hmax = da_hmax.where(da_hmax > 0.05).where(mask)

            # Update attributes and check orientation
            da_hmax.attrs.update(long_name="flood depth", unit="m")
            if da_hmax.y.values[0] < da_hmax.y.values[-1]:
                da_hmax = da_hmax[::-1, :]  # Flip vertically if not north-up
                print(f"Flipped '{scenario}' raster as it was not in north-up order.")
            else:
                print(f"Raster for '{scenario}' already in north-up order, no flip needed.")
            
            da_hmax.rio.to_raster(tif_file, tiled=True, compress='LZW')
            print(f"Processed and saved '{tif_file}'")

        else:
            print(f"'hazard_map.nc' not found for scenario: {scenario}")

# Example usage
source_dir = r"D:\paper_4\data\FloodAdapt-GUI\Database\beira\output\Scenarios"
target_dir = r"D:\paper_4\data\floodadapt_results\hazard_tiff"
sim = 'idai_ifs_rebuild_bc_'
climate_scenarios = ['hist', '3c', 'hightide', '3c-hightide']
adapt_scenarios = ['noadapt', 'hold', 'retreat']
base_folder = 'D:/paper_4/data/sfincs_input/'  # Adjust as needed
storm = 'idai'
data_libs = ['d:/paper_4/data/data_catalogs/data_catalog_converter.yml', base_folder+rf'/data_deltares_{storm}/data_catalog.yml']

process_hazard_files(source_dir, target_dir, sim, climate_scenarios, adapt_scenarios, base_folder, storm, data_libs)