import rasterio
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box
from pyproj import Transformer
import pandas as pd

# load the buildings
buildings_centroids_path = rf'D:\paper_4\data\vanPanos\FIAT_model_new\Exposure\exposure_clip4.gpkg'
buildings_centroids = gpd.read_file(buildings_centroids_path)

buildings_fp = gpd.read_file(rf'D:\paper_4\data\vanPanos\FIAT_model_new\footprints\buildings.shp')
buildings_fp ['Shape_Area'] = buildings_fp.geometry.area

#add the area to the centroids
buildings_centroids = buildings_centroids.merge(buildings_fp[['Object ID', 'Shape_Area']], on='Object ID', how='left')
# Set the CRS for the buildings_centroids GeoDataFrame
transformer = Transformer.from_crs('EPSG:32736', 'EPSG:4326', always_xy=True)
# Transform the building centroids to lat/lon
buildings_centroids['lon_lat'] = buildings_centroids['geometry'].apply(
    lambda geom: transformer.transform(geom.centroid.x, geom.centroid.y)
)
# Now, creating new points based on the transformed coordinates for intersection checks
buildings_centroids['geometry_latlon'] = buildings_centroids['lon_lat'].apply(
    lambda coords: Point(coords[0], coords[1])
)
# Ensure buildings_centroids is a GeoDataFrame with the appropriate geometry set
buildings_centroids['geometry_latlon'] = gpd.GeoSeries(buildings_centroids['geometry_latlon'], crs='EPSG:4326')

# Now set 'geometry_latlon' as the active geometry for spatial operations
buildings_centroids.set_geometry('geometry_latlon', inplace=True)

# create the target dataframe and populate with 0s
result_df = buildings_centroids.copy()
result_df['population'] = 0

# Load the population raster
raster_path = 'D:\paper_4\data\qgis\moz_ppp_2020_UNadj_constrained_clip.tif'

# Open the population raster
with rasterio.open(raster_path) as src:
    pop_raster = src.read(1)  # Assuming population data is in the first band

    for j in range(src.width):
        for i in range(src.height):
            cell_population = pop_raster[i, j]
            if cell_population > 0:  # Process only cells with population
                # Convert cell coordinates to a bounding box (polygon)
                cell_polygon = box(*src.window_bounds(window=((i, i+1), (j, j+1))))

                # Identify centroids within this cell
                within_cell = result_df[result_df.intersects(cell_polygon)]

                if not within_cell.empty:
                    # Calculate total area within cell for buildings identified by 'Object ID'
                    total_area_within_cell = buildings_fp[buildings_fp['Object ID'].isin(within_cell['Object ID'])]['Shape_Area'].sum()

                    for _, row in within_cell.iterrows():
                        object_id = row['Object ID']
                        building_area = buildings_fp[buildings_fp['Object ID'] == object_id]['Shape_Area'].iloc[0]

                        # Calculate population proportion for each building
                        proportion_of_population = (building_area / total_area_within_cell) * cell_population if total_area_within_cell > 0 else 0

                        # Update the population in result_df for the current building
                        result_df.loc[result_df['Object ID'] == object_id, 'population'] = proportion_of_population

# save the result
# add the population to the original file
buildings_centroids = buildings_centroids.merge(result_df[['Object ID','population']], on='Object ID', how='left')

buildings_centroids.to_file(rf'D:\paper_4\data\vanPanos\qgis_data\exposure_clip4_pop.gpkg', driver='GPKG')


exposure_csv = pd.read_csv(r'D:\paper_4\data\vanPanos\qgis_data\exposure_clip4.csv')

merged_gdf = buildings_centroids.merge(exposure_csv, on='Object ID')


merged_df = merged_gdf.drop(columns='geometry')
merged_df.to_csv(r'D:\paper_4\data\vanPanos\qgis_data\exposure_clip4_pop.csv', index=False)

# test sanity
test_sanity = False
if test_sanity == True:
    # Load the population raster
    raster_path = 'D:\paper_4\data\qgis\moz_ppp_2020_UNadj_constrained_clip.tif'
    # Open the population raster
    with rasterio.open(raster_path) as src:
        pop_raster = src.read(1)
    # Calculate the total population from pop_raster
    total_population_values = pop_raster[pop_raster > 0].sum()
    # load merged_df
    merged_df = pd.read_csv(r'D:\paper_4\data\vanPanos\qgis_data\exposure_clip4_pop.csv')
    # Calculate the total population from result_df
    total_population_result = merged_df['population'].sum()

    # Check if the total population from the raster and result_df are the same
    print(f"Total population from raster: {total_population_values}")
    print(f"Total population from result_df: {total_population_result}")
    # rough test to check how close they can be
    if total_population_result > total_population_values *1.1 or total_population_result < total_population_values *0.9:
        print("Sanity check failed: Total population from raster and result_df are not the same")
    else:
        print("Sanity check passed: Total population from raster and result_df are compatible")
        print("Ratio between the two: ", total_population_result/total_population_values)