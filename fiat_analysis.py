from hydromt.log import setuplog
from pathlib import Path
import geopandas as gpd
import pandas as pd
import os
import json
import yaml
from hydromt.config import configread
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import rasterio
from rasterio.enums import Resampling
import toml
import shutil
import numpy as np
import fiat
from fiat.io import *
from fiat.main import *
from matplotlib import font_manager
import xarray as xr
import matplotlib.patches as mpatches

from rasterio.plot import show
from matplotlib.colors import Normalize, Colormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable
import contextily as ctx

# add Roboto font to the font_manager
font_files = font_manager.findSystemFonts(fontpaths=rf"C:\Users\morenodu\Downloads", fontext='ttf')
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Roboto'
# general setup for plots
singlecol = 8.3 * 0.393701
doublecol = 14 * 0.393701
##########################################################################################

# Assuming the alignment function needs to handle missing geometries appropriately
def align_gdf(gdf, all_ids, master_geometry_dict, metric = 'Total Damage'):
    # Ensure the GDF includes all 'Object IDs', filling missing ones with default values
    aligned_df = pd.DataFrame(all_ids, columns=['Object ID']).merge(
        gdf.drop(columns='geometry'),
        on='Object ID',
        how='left'
    ).fillna({metric: 0})

    # Assign geometries from the master_geometry_dict based on 'Object ID'
    aligned_df['geometry'] = aligned_df['Object ID'].map(master_geometry_dict)

    # Convert back to GeoDataFrame
    aligned_gdf = gpd.GeoDataFrame(aligned_df, geometry='geometry', crs=gdf.crs)
    
    return aligned_gdf

##########################################################################################
# plot the water level line graphs

# open the sfincs_map.nc file and plot the map
ds_his = xr.open_dataset(f'D:\paper_4\data\sfincs_input\quadtree_ifs_rebuild_bc\sfincs_his.nc')
ds_his_maxtide = xr.open_dataset(f'D:\paper_4\data\sfincs_input\quadtree_ifs_rebuild_bc_hightide\sfincs_his.nc')
ds_his_slr3c = ds_his + (0.64 - 3.6*(2019-2005)*0.001)

print("The max tide for the baseline is: ", ds_his['point_zs'].isel(stations = 4).max().item())
print("The max tide for the hightide is: ", ds_his_maxtide['point_zs'].isel(stations = 4).max().item())
print("The difference is ", ds_his_maxtide['point_zs'].isel(stations = 4).max().item() - ds_his['point_zs'].isel(stations = 4).max().item())
# plot the timeseries for point_zs for station 'station_name' == 10
ds_his['point_zs'].isel(stations = 4).plot(label = 'Baseline', linewidth = 2)
# ds_his_maxtide['point_zs'].isel(stations = 4).plot(label = 'Springtide', linewidth = 2)
ds_his_slr3c['point_zs'].isel(stations = 4).plot(label = '3C', linewidth = 2)
# ds_truth2['point_zs'].isel(stations = 4).plot(label = 'kees2_tides_only')
plt.legend()
plt.show()

##########################################################################################
hazard_folder_path = r'D:\paper_4\data\floodadapt_results\hazard_tiff'
fiat_output_path = r'D:\paper_4\data\floodadapt_results\spatial_gpkg'
sim = 'idai_ifs_rebuild_bc_'
# scenarios
climate_scenarios = ['hist', 'hightide', '3c', '3c-hightide']
adapt_scenarios = ['noadapt', 'hold','retreat']

scen_dict = {'hist':'Baseline', '3c':'3C', '3c-hightide':'3C-springtide', 'hightide':'Springtide'}
adapt_dict = {'noadapt':'No Adaptation', 'hold':'Hold the line', 'retreat':'Integrated'}

# Generate all combinations of climate and adaptation scenarios
all_adapt_scenarios = [f"{sim}{climate}_rain_surge_{adapt}" for climate in climate_scenarios for adapt in adapt_scenarios]

# Initialize an empty list to store each GeoDataFrame
gdf_list = []

# Loop through each combination of scenarios
for climate in climate_scenarios:
    for adapt in adapt_scenarios:
        # Construct the file name based on the scenario
        file_name = rf'{fiat_output_path}\spatial_{sim}{climate}_rain_surge_{adapt}.gpkg'
        
        # Load the .gpkg file
        gdf = gpd.read_file(file_name)
        
        # Add climate and adaptation scenario as columns
        gdf['climate_scenario'] = climate
        gdf['adapt_scenario'] = adapt
        
        # Append to the list
        gdf_list.append(gdf)

# Concatenate all GeoDataFrames into one
all_gdf = pd.concat(gdf_list, ignore_index=True)

# Step 1: Aggregate the data
agg_data = all_gdf.groupby(['climate_scenario', 'adapt_scenario'])['Total Damage'].sum().reset_index()
#apply the scen_dict and adapt_dict to the columns
agg_data['climate_scenario'] = agg_data['climate_scenario'].map(scen_dict)
agg_data['adapt_scenario'] = agg_data['adapt_scenario'].map(adapt_dict)

# Step 2: Create the bar charts
# Define the order of adaptation scenarios and climate scenarios for consistency in plotting
adapt_order = ['No Adaptation', 'Hold the line', 'Integrated']
climate_order = ['Baseline', 'Springtide', '3C', '3C-springtide']

# plot 2 - all on the same plot
colors = ['salmon', '#9A607F', '#B4BA39']  # Colors for each adaptation scenario

# Number of climate scenarios
n_climate = len(climate_order)
# Width of the bars
bar_width = 0.15

# Set positions of the bars for each adaptation scenario
positions = np.arange(len(climate_order))  # Base positions

noadapt_data = agg_data[agg_data['adapt_scenario'] == 'No Adaptation']
# set the order of the climate scenarios
noadapt_data = noadapt_data.set_index('climate_scenario').reindex(climate_order).reset_index()

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
tick_increment = 4 *10e6
ax.set_yticks(np.arange(0, noadapt_data['Total Damage'].max()*1.1 + tick_increment, tick_increment))
ax.yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.bar(positions, noadapt_data['Total Damage'], width=bar_width, color=colors[0], zorder=2)

# Formatting the plot
ax.set_xlabel('Climate Scenario')
ax.set_ylabel('Total Damage [$]')
ax.set_title('Total Damage of Idai in Beira for different climate scenarios', fontsize=17, fontweight='medium', loc='left')
ax.set_xticks(positions)
ax.set_xticklabels(climate_order)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.8, 1.1), ncol=3, fontsize=12)

plt.show()

##########################################################################################
# Create the plot
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
ax.yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)

for i, adapt in enumerate(adapt_order):
    # Calculate offset for each group based on adaptation scenario index
    offset = (i - np.floor(len(adapt_scenarios) / 2)) * bar_width
    # Filter data for the current adaptation scenario
    adapt_data = agg_data[agg_data['adapt_scenario'] == adapt]
    # Ensure the data is in the order of climate scenarios
    adapt_data = adapt_data.set_index('climate_scenario').reindex(climate_order).reset_index()
    # Plot
    ax.bar(positions + offset, adapt_data['Total Damage'], width=bar_width, label=adapt, color=colors[i], zorder=2)

# Formatting the plot
ax.set_xlabel('Climate Scenario')
ax.set_ylabel('Total Damage [$]')
ax.set_title('Total Damage of Idai storylines in Beira', fontsize=17, fontweight='medium', loc='left')
ax.set_xticks(positions)
ax.set_xticklabels(climate_order)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.8, 1.1), ncol=3, fontsize=12)

plt.show()

# figure 3 - percentage changes
pivot_data = agg_data.pivot(index='climate_scenario', columns='adapt_scenario', values='Total Damage')

# Calculate percentages relative to 'noadapt'
pivot_data['hold_pct'] = (pivot_data['Hold the line'] / pivot_data['No Adaptation']) * 100 - 100
pivot_data['retreat_pct'] = (pivot_data['Integrated'] / pivot_data['No Adaptation']) * 100 - 100
pivot_data['noadapt_pct'] = 100 - 100 # Baseline

# Prepare data for plotting
percent_data = pd.DataFrame({
    'climate_scenario': pivot_data.index,
    'noadapt': pivot_data['noadapt_pct'],
    'Hold the line': pivot_data['hold_pct'],
    'Integrated': pivot_data['retreat_pct']
}).melt(id_vars='climate_scenario', var_name='adapt_scenario', value_name='Damage Percent')
# drop rows where adapt_scenario is noadapt
percent_data = percent_data[percent_data['adapt_scenario'] != 'noadapt']


# Plotting
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
colors_diff = ['#9A607F', '#B4BA39']  # Colors for each adaptation scenario
# Add thin gray horizontal lines at intervals of 20
ax.yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)

# Set positions of the bars for each adaptation scenario
positions = np.arange(len(climate_order))  # Base positions

for i, adapt in enumerate(adapt_order[1:3]):
    # Calculate offset for each group based on adaptation scenario index
    offset = (i - np.floor(len(adapt_order[1:3]) / 2)) * bar_width
    # Filter data for the current adaptation scenario
    adapt_data = percent_data[percent_data['adapt_scenario'] == adapt]
    # Ensure the data is in the order of climate scenarios
    adapt_data = adapt_data.set_index('climate_scenario').reindex(climate_order).reset_index()
    # Plot
    ax.bar(positions + offset, adapt_data['Damage Percent'], width=bar_width, label=adapt, color=colors_diff[i], zorder=2)
# add a dahsed horizontal line at 0
# Formatting the plot
ax.set_xlabel('Climate Scenario')
ax.set_ylabel('Damage change (%)')
ax.set_title('Damage wrt No Adaptation', fontsize=17, fontweight='medium', loc='left')
ax.set_xticks(positions)
ax.set_xticklabels(climate_order)
ax.set_ylim(-100, 10)  # Adjust as needed based on your data
ax.legend(frameon=False,  bbox_to_anchor=(0.85, 1.1), ncol=2, loc='upper center', fontsize=12)

plt.show()

##########################################################################################
# Initialize a dictionary to store the loaded data
raster_data = {}

# Loop through each combination of scenarios
for climate in climate_scenarios:
    for adapt in adapt_scenarios:
        # Construct the file name based on the scenario
        file_name = f'{hazard_folder_path}/hmax_{sim}{climate}_rain_surge_{adapt}.tiff'
        
        # Load the .tiff file using rasterio
        with rasterio.open(file_name) as src:
            # Read the raster data
            data = src.read(1)  # Reads the first band; adjust as needed
            
            # Optional: mask out no data values if necessary
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            
            # Store the data in the dictionary
            key = f'{climate}_{adapt}'
            raster_data[key] = data            

# Step 1: Calculate the average value for each raster dataset
avg_values = {}
for key, data in raster_data.items():
    # Calculate the average, excluding NaN values
    avg_value = np.nansum(data)
    avg_values[key] = avg_value

# Step 2: Prepare the data for plotting
# Convert the avg_values dictionary into a DataFrame for easier plotting
df = pd.DataFrame(list(avg_values.items()), columns=['Scenario', 'SumValue'])
df['ClimateScenario'], df['AdaptScenario'] = zip(*df['Scenario'].str.split('_'))

# Now, sort the DataFrame by these columns to ensure the order is respected in the pivot
df_sorted = df.sort_values(['ClimateScenario', 'AdaptScenario'])

# Step 2: Pivot the sorted DataFrame
pivot_df = df_sorted.pivot(index="ClimateScenario", columns="AdaptScenario", values="SumValue")

# apply the scen_dict and adapt_dict to the columns
pivot_df.columns = pivot_df.columns.map(adapt_dict)
pivot_df.index = pivot_df.index.map(scen_dict)
# Optional: Reindex the pivot table to ensure the order (this step may be redundant if sorting was effective)
pivot_df = pivot_df.reindex(index=climate_order, columns=adapt_order)

# # Step 3: Plot the data
# pivot_df.plot(kind='bar', figsize=(10, 6), width=0.4, color = colors, zorder=2)
# ax = plt.gca()
# ax.yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.title('Total inundation volume per scenario', fontsize=14, fontweight='medium', loc='left')
# plt.ylabel('Total volume (m3)')
# plt.xticks(rotation=0)
# plt.legend(bbox_to_anchor=(0.75, 1.1), loc='upper center', frameon=False, ncol=3)
# plt.tight_layout()
# plt.show()


##########################################################################################
# load populations for each scenario
sim = 'idai_ifs_rebuild_bc_'
climate_scenarios = ['hist', '3c', 'hightide', '3c-hightide'] #
adapt_scenarios = ['noadapt', 'retreat', 'hold'] #

# Generate all combinations of climate and adaptation scenarios
all_adapt_scenarios = [f"{sim}{climate}_rain_surge_{adapt}" for climate in climate_scenarios for adapt in adapt_scenarios]

root_fa_output = rf'D:\paper_4\data\FloodAdapt-GUI\Database\beira\output\Scenarios'

selected_metrics = ['TotalDamageEvent', 'PeopleAffected10cm', 'PeopleAffected30cm',
                    'PeopleAffected50cm', 'PeopleAffected1m']

# Initialize an empty DataFrame to concatenate all CSV files
all_infometrics_df = pd.DataFrame()

for scenario in all_adapt_scenarios:
    # Define the path pattern for the CSV files
    infometrics_csv_path = rf'{root_fa_output}\{scenario}\infometrics_{scenario}.csv'    
    # Use glob to find all files matching the pattern
    infometrics = pd.read_csv(infometrics_csv_path)
    # pivot the table

    filtered_df = infometrics[infometrics['Unnamed: 0'].isin(selected_metrics)].drop(columns=['Unnamed: 0','Description','Show In Metrics Table'])

    filtered_df = filtered_df.pivot_table(columns='Long Name', values='Value').reset_index().drop(columns=['index'])
            
    filtered_df['clim_scen'] = scenario.split('_')[-4]
    filtered_df['adapt_scen'] = scenario.split('_')[-1] 
        
        # Concatenate the current dataframe to the main dataframe
    all_infometrics_df = pd.concat([all_infometrics_df, filtered_df], ignore_index=True)

# create a dictionary to change: "People Affected between 15 to 50 cm": "15 to 50 cm", "People Affected between 50 and 150 cm": "50 to 150 cm", "People Affected above 150 cm": "above 150 cm"
pop_dict = {'People Affected between 15 to 50 cm': '15 to 50 cm', 'People Affected between 50 and 150 cm': '50 and 150 cm', 'People Affected above 150 cm': 'above 150 cm'}
all_infometrics_df = all_infometrics_df.rename(columns=pop_dict)
    
# Step 1: Aggregate the data for 'People Affected above 150 cm'
agg_info = all_infometrics_df.groupby(['clim_scen', 'adapt_scen'])['above 150 cm'].sum().reset_index()

# use scen_dict to change values in clim_scen column in all_infometrics_df
all_infometrics_df['clim_scen'] = all_infometrics_df['clim_scen'].map(scen_dict)
# now use adapt_dict to change values in adapt_scen column in all_infometrics_df
all_infometrics_df['adapt_scen'] = all_infometrics_df['adapt_scen'].map(adapt_dict)

# Step 2: Create the bar charts
adapt_order = ['No Adaptation', 'Hold the line', 'Integrated']
climate_order = ['Baseline', 'Springtide', '3C', '3C-springtide']
colors = ['salmon', 'lightblue', 'lightgreen']  # Colors for each adaptation scenario

# Number of climate scenarios
n_climate = len(climate_order)
# Width of the bars
bar_width = 0.15

# Step 2: Create the stacked bar chart
population_metrics = [
    '15 to 50 cm',
    '50 and 150 cm',
    'above 150 cm'
]

# bar plot only for noadapt
noadapt_pop_data = all_infometrics_df[all_infometrics_df['adapt_scen'] == 'No Adaptation']
# drop column 'People Affected up 15 cm'
noadapt_pop_data = noadapt_pop_data.drop(columns=['People Affected up 15 cm'])

# Sort noadapt_pop_data based on the order of climate scenarios
noadapt_pop_data = noadapt_pop_data.set_index('clim_scen').reindex(climate_order).reset_index()

#sum all the population metrics for noadapt_pop_data
noadapt_pop_data['Total significant exposed people'] = noadapt_pop_data['15 to 50 cm'] + noadapt_pop_data['50 and 150 cm'] + noadapt_pop_data['above 150 cm']
# divide all rows by the Baseline
percentage_pop_increase = noadapt_pop_data['Total significant exposed people'] / noadapt_pop_data['Total significant exposed people'].iloc[0]
percentage_damage_increase = noadapt_pop_data['Total building damage'] / noadapt_pop_data['Total building damage'].iloc[0]

print((noadapt_pop_data['above 150 cm'] / noadapt_pop_data['Total significant exposed people'])*100)
# print noadapt_pop_data without adapt_scen column and without the index
noadapt_pop_data_values = noadapt_pop_data.drop(columns=['adapt_scen']).set_index('clim_scen')
# make now the table from noadapt_pop_data_values with relative values based on the Baseline
noadapt_pop_data_values_relative = noadapt_pop_data_values.div(noadapt_pop_data_values.iloc[0], axis=1).round(2)
# rename the columns Total significant exposed people to Total exposed people and change the order of the columns to be : Total exposed people above 150 cm  15 to 50 cm  50 and 150 cm  Total building damage
noadapt_pop_data_values_relative = noadapt_pop_data_values_relative.rename(columns={'Total significant exposed people':'Total exposed people (%)', 'above 150 cm':'above 150 cm (%)', '15 to 50 cm':'15 to 50 cm (%)', '50 and 150 cm':'50 and 150 cm (%)', 'Total building damage':'Total building damage (%)'})
noadapt_pop_data_values_relative = noadapt_pop_data_values_relative[['Total exposed people (%)', 'Total building damage (%)']]
# now combine the relative values with the original noadapt_pop_data
noadapt_pop_data_comb = noadapt_pop_data.merge(noadapt_pop_data_values_relative, left_on='clim_scen', right_index=True).round(2)
noadapt_pop_data_comb[['clim_scen','Total significant exposed people', 'Total exposed people (%)', 'Total building damage', 'Total building damage (%)']].set_index('clim_scen')


blue_colors = ['#a9cce3', '#5dade2', '#2e86c1']

# # Set up the figure
# fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
# tick_increment = 50000
# ax.set_yticks(np.arange(0, noadapt_pop_data['50 and 150 cm'].max()*1.1 + tick_increment, tick_increment))

# ax.yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # Define the bottom of the stack
# bottom = np.zeros(len(climate_order))

# # Plot the bars with the blue color gradient
# for i, metric in enumerate(population_metrics):
#     ax.bar(
#         noadapt_pop_data['clim_scen'],
#         noadapt_pop_data[metric],
#         bottom=bottom,
#         color=blue_colors[i],
#         width=0.2, 
#         label=metric,
#         edgecolor='white',
#         zorder=2,
#     )
#     # Update the bottom position for the next segment of the stack
#     bottom += noadapt_pop_data[metric].values

# # Formatting the plot
# # ax.set_xlabel('Climate Scenario')
# ax.set_ylabel('Population exposure')
# ax.set_title('Population exposure for climate scenarios', fontsize=17, fontweight= 'medium', loc='left')
# ax.legend(frameon=False,  bbox_to_anchor=(0.85, 1.16), ncol=1, loc='upper center', fontsize=11)

# plt.xticks(rotation=0, ha='center')
# # save figure in D:\paper_4\data\Figures\paper_figures
# plt.savefig(r'D:\paper_4\data\Figures\paper_figures\population_exposure_bar.png', dpi=300, bbox_inches='tight')
# plt.show()


#########
# PAPER FIGURE 2
fig, axs = plt.subplots(1, 2, figsize=(doublecol*1.3, singlecol), dpi=150)  # 2 rows, 1 column, sharing the x-axis

# Plot 2: Population exposure for climate scenarios
tick_increment = 50000
axs[0].set_yticks(np.arange(0, noadapt_pop_data['50 and 150 cm'].max() + tick_increment, tick_increment))
axs[0].yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
axs[0].spines['right'].set_visible(False)
axs[0].spines['left'].set_visible(False)
axs[0].spines['top'].set_visible(False)

# Plot the bars with the blue color gradient for the second plot
bottom = np.zeros(len(climate_order))
for i, metric in enumerate(population_metrics):
    axs[0].bar(
        noadapt_pop_data['clim_scen'],
        noadapt_pop_data[metric],
        bottom=bottom,
        color=blue_colors[i],
        width=0.35, 
        label=metric,
        edgecolor='white',
        zorder=2,
    )
    # Update the bottom position for the next segment of the stack
    bottom += noadapt_pop_data[metric].values

# Formatting the second plot
axs[0].set_ylabel('Population')
title = axs[0].set_title('a) Population exposure in Beira', fontsize=14, fontweight='medium', loc='left', pad = 15)
title.set_position([-0.25, 1.3])  # Adjust the first value to move left/right, second value to move up/down
axs[0].legend(frameon=False, bbox_to_anchor=(-0.28, 1.05), ncol=1, loc='upper left', fontsize=8, handletextpad=0.1, columnspacing=0.5)

# Plot 1: Total Damage for different climate scenarios
tick_increment = 50
axs[1].set_yticks(np.arange(0, noadapt_data['Total Damage'].max() / 10e5 + tick_increment, tick_increment))
axs[1].yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
axs[1].spines['right'].set_visible(False)
axs[1].spines['left'].set_visible(False)
axs[1].spines['top'].set_visible(False)

# Width of the bars and positions
bar_width = 0.35
positions = np.arange(len(climate_order))  # Base positions
axs[1].bar(positions, noadapt_data['Total Damage'] / 10e5, width=bar_width, color='salmon', zorder=2)

# Formatting the first plot
axs[1].set_ylabel('Building damage [$10^6$ USD]')
title = axs[1].set_title('b) Total damage in Beira', fontsize=14, fontweight='medium', loc='left', pad = 15)
title.set_position([-0.1, 1.3])  # Adjust these values as needed
axs[1].set_xticks(positions)
axs[1].set_xticklabels(climate_order)
# axs[1].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.8, 1.1), ncol=1, fontsize=12)
# add some space between the plots
plt.subplots_adjust(wspace=0.3)

plt.xticks(rotation=0, ha='center')
# plt.xlabel('Hydrometeorological scenarios')  # Since x-axis is shared, set xlabel on the whole figure

# Save the combined figure
plt.savefig(r'D:\paper_4\data\Figures\paper_figures\pop_damage_fig1.png', dpi=300, bbox_inches='tight')

plt.show()



##########################################################################################
# Define the metrics and scenarios
population_metrics = [
    '15 to 50 cm',
    '50 and 150 cm',
    'above 150 cm',
]

adapt_order = ['Hold the line', 'Integrated']  # Specify the desired order
# Initialize an empty list to store the data for differences
diff_data = []

# Calculate differences for 'noadapt' against other adaptation scenarios
for clim_scen in climate_order:
    noadapt_values = all_infometrics_df[
        (all_infometrics_df['clim_scen'] == clim_scen) & 
        (all_infometrics_df['adapt_scen'] == 'No Adaptation')
    ][population_metrics].values.flatten()

    for adapt_scen in all_infometrics_df['adapt_scen'].unique():
        if adapt_scen != 'No Adaptation':
            adapt_values = all_infometrics_df[
                (all_infometrics_df['clim_scen'] == clim_scen) & 
                (all_infometrics_df['adapt_scen'] == adapt_scen)
            ][population_metrics].values.flatten()
            
            # Calculate the difference for each metric
            diff_values = (adapt_values - noadapt_values)/noadapt_values * 100
            diff_data.append(pd.Series([clim_scen, adapt_scen] + diff_values.tolist(), 
                                       index=['clim_scen', 'adapt_scen'] + population_metrics))

# Convert the list of Series into a DataFrame
diff_df = pd.concat(diff_data, axis=1).T


# Convert the metric columns to numeric
for metric in population_metrics:
    diff_df[metric] = pd.to_numeric(diff_df[metric])

from matplotlib.patches import Patch
# Define colors for the bars, from lightest to darkest blue
# blue_colors = ['#d4e6f1', '#a9cce3', '#5dade2', '#2e86c1']
hatch_patterns = {'Hold the line': '','Integrated': '///' }  # Adjust patterns as needed

bar_width = 0.15
positions = np.arange(len(climate_order))

# remove noadapt from all_infometrics_df
adapt_df = all_infometrics_df[all_infometrics_df['adapt_scen'] != 'No Adaptation']
# drop column 'People Affected up 15 cm'
adapt_df = adapt_df.drop(columns=['People Affected up 15 cm'])
noadapt_df2 = all_infometrics_df[all_infometrics_df['adapt_scen'] == 'No Adaptation']
# drop column 'People Affected up 15 cm'
noadapt_df2 = noadapt_df2.drop(columns=['People Affected up 15 cm'])
# Prepare the data for both scenarios
hold_pop_data = all_infometrics_df[all_infometrics_df['adapt_scen'] == 'Hold the line'].drop(columns=['People Affected up 15 cm'])
int_pop_data = all_infometrics_df[all_infometrics_df['adapt_scen'] == 'Integrated'].drop(columns=['People Affected up 15 cm'])

# Ensure data is sorted according to climate_order
hold_pop_data = hold_pop_data.set_index('clim_scen').reindex(climate_order).reset_index()
int_pop_data = int_pop_data.set_index('clim_scen').reindex(climate_order).reset_index()

# Create a new column for 'Total significant exposed people' and calculate the difference
hold_pop_data['Total significant exposed people'] = hold_pop_data['15 to 50 cm'] + hold_pop_data['50 and 150 cm'] + hold_pop_data['above 150 cm']
int_pop_data['Total significant exposed people'] = int_pop_data['15 to 50 cm'] + int_pop_data['50 and 150 cm'] + int_pop_data['above 150 cm']
noadapt_df2['Total significant exposed people'] = noadapt_df2['15 to 50 cm'] + noadapt_df2['50 and 150 cm'] + noadapt_df2['above 150 cm']
noadapt_df2= noadapt_df2[['clim_scen','Total significant exposed people','Total building damage']]

hold_diff = hold_pop_data[['clim_scen','Total significant exposed people','Total building damage']].copy()
int_diff = int_pop_data[['clim_scen','Total significant exposed people','Total building damage']].copy()
hold_diff['difference_pop'] = hold_pop_data['Total significant exposed people'] - noadapt_pop_data['Total significant exposed people']
int_diff['difference_pop'] = int_pop_data['Total significant exposed people'] - noadapt_pop_data['Total significant exposed people']
hold_diff['difference_damage'] = hold_pop_data['Total building damage'] - noadapt_pop_data['Total building damage']
int_diff['difference_damage'] = int_pop_data['Total building damage'] - noadapt_pop_data['Total building damage']
hold_diff['difference_pop_prc'] = 100*(hold_pop_data['Total significant exposed people'] - noadapt_pop_data['Total significant exposed people'])/noadapt_pop_data['Total significant exposed people']
int_diff['difference_pop_prc'] = 100*(int_pop_data['Total significant exposed people'] - noadapt_pop_data['Total significant exposed people'])/noadapt_pop_data['Total significant exposed people']
hold_diff['difference_damage_prc'] = 100*(hold_pop_data['Total building damage'] - noadapt_pop_data['Total building damage'])/noadapt_pop_data['Total building damage']
int_diff['difference_damage_prc'] = 100*(int_pop_data['Total building damage'] - noadapt_pop_data['Total building damage'])/noadapt_pop_data['Total building damage']

int_diff['adapt_scen'] = 'integrated'
hold_diff['adapt_scen'] = 'hold the line'

# Concatenate DataFrames
combined_df = pd.concat([int_diff, hold_diff], ignore_index=True)
# print combined_df with columns Total significant exposed people  Total building damage  difference_pop_prc  difference_damage_prc     adapt_scen, having index as clim_scen and adapt_scen
print("main table of results: ",combined_df[['clim_scen','Total significant exposed people','Total building damage','difference_pop_prc','difference_damage_prc','adapt_scen']].set_index(['clim_scen','adapt_scen']))

# Common settings for both plots
tick_increment_population = 20  # Adjust based on your actual data
tick_increment_damage = 20  # Adjust based on your actual data

# Define y-ticks for population and damage plots
y_max_population = -80
yticks_population = np.arange(y_max_population, 0.01, tick_increment_population)

y_max_damage = -80
yticks_damage = np.arange(y_max_damage, 0.01, tick_increment_damage)

# Set up the colors and hatches for visual distinction
hatches = ['', '///']  # No hatching for 'hold the line', hatching for 'integrated'
metric_colors = {'difference_pop_prc': '#5dade2', 'difference_damage_prc': 'salmon'}

# Get unique 'clim_scen' and 'adapt_scen' for consistent ordering
clim_scens = combined_df['clim_scen'].unique()
adapt_scens= ['hold the line', 'integrated']

# Bar width and positions
bar_width = 0.35
clim_scen_positions = np.arange(len(climate_order))

# FIGURE
fig, axs = plt.subplots(1, 2, figsize=(doublecol*1.4, singlecol), dpi=150)
for ax, metric, ylabel, yticks in zip(axs, ['difference_pop_prc', 'difference_damage_prc'], ['Population reduction (%)', 'Damage reduction (%)'], [yticks_population, yticks_damage]):
    for i, adapt_scen in enumerate(adapt_scens):
        # Filter and sort DataFrame for the current adaptation scenario
        adapt_scen_df = combined_df[combined_df['adapt_scen'] == adapt_scen]
        adapt_scen_df = adapt_scen_df.set_index('clim_scen').reindex(clim_scens).reset_index()
        # Calculate offsets for side-by-side bars
        offsets = clim_scen_positions + (i - np.mean(range(len(adapt_scens)))) * bar_width
        # Plotting
        ax.bar(offsets, adapt_scen_df[metric], width=bar_width, color=metric_colors[metric], label=adapt_scen, hatch=hatches[i], edgecolor='white', zorder=2)
        # Apply common styles
        ax.yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticks(yticks)

    ax.set_ylabel(ylabel)

# Additional formatting
title = axs[0].set_title('a) Changes in exposed population', fontsize=13, fontweight='medium', loc='left', pad=20)
title_2 = axs[1].set_title('b) Changes in building damage', fontsize=13, fontweight='medium', loc='left', pad=20)
title.set_position([-0.2, 2.1])  # Adjust the first value to move left/right, second value to move up/down
title_2.set_position([-0.2, 2.1])  # Adjust the first value to move left/right, second value to move up/down
# axs[1].set_title('Damage Changes', fontsize=15, fontweight='medium', loc='left')
axs[0].set_xticks(clim_scen_positions)
axs[1].set_xticks(clim_scen_positions)
axs[0].set_xticklabels(clim_scens)
axs[1].set_xticklabels(clim_scens)
# axs[1].set_xlabel('Hydrometeorological scenarios')
# add some vertical space between the plots
plt.subplots_adjust(wspace=0.3)
# Create a single legend for the first plot for clarity
handles, labels = axs[0].get_legend_handles_labels()

# Creating custom legend handles for hatching, ignoring color
legend_handles = [mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch, label=label)
                  for hatch, label in zip(hatches, adapt_scens)]
fig.legend(handles=legend_handles, frameon=False, bbox_to_anchor=(0.35, 0.04), ncol=2, loc='upper left', fontsize=10)

plt.savefig(r'D:\paper_4\data\Figures\paper_figures\population_exposure_and_damage_prc_bar.png', dpi=250, bbox_inches='tight')
plt.show()



# Set up the figure with a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(doublecol*1.6, doublecol*1.3), dpi=150, sharex='col', gridspec_kw={'height_ratios': [1, 1]})
# Flatten the axes array for easy indexing
axes = axes.flatten()

# Define common settings for the top plots
tick_increment_population = 20000  # Adjust this value as needed for population exposure plots
tick_increment_damage = 20000000  # Adjust this value as needed for building damage plots
yticks_population = np.arange(0, hold_pop_data['Total significant exposed people'].max() + tick_increment_population, tick_increment_population)
y_max_damage = max([pop_data['Total building damage'].max() for pop_data in [hold_pop_data, int_pop_data]]) * 1.1
yticks_damage = np.arange(0, y_max_damage + tick_increment_damage, tick_increment_damage)

dict_fig1 = {0: "a)", 1: "b)"}
dict_fig2 = {0: "c)", 1: "d)"}
# Top Row Plots for Population Exposure
for i, (ax, pop_data, title) in enumerate(zip(axes[:2], [hold_pop_data, int_pop_data], ['Hold the line', 'Integrated'])):
    # ax.set_yticks(yticks_population)
    ax.yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    bottom = np.zeros(len(climate_order))

    for j, metric in enumerate(population_metrics):
        ax.bar(
            pop_data['clim_scen'],
            pop_data[metric],
            bottom=bottom,
            color=blue_colors[j],
            width=0.25,
            label=metric if i == 0 else "",  # Avoid repeating labels in the legend
            edgecolor='white',
            zorder=2,
        )
        bottom += pop_data[metric].values

    ax.set_ylabel('Population exposure' if i == 0 else "")  # Label only for the first column
    title = ax.set_title(f'{dict_fig1[i]} Population exposure for {title}', fontsize=13, fontweight='medium', loc='left', pad=10)
    title.set_position([-0.1, 1])  # Adjust the first value to move left/right, second value to move up/down
    ax.tick_params(axis='x', which='both', length=0)  # Hide x-axis ticks

# Bottom Row Plots for Total Building Damage
for i, (ax, pop_data, title) in enumerate(zip(axes[2:], [hold_pop_data, int_pop_data], ['Hold the line', 'Integrated'])):
    damage_data = pop_data['Total building damage'] / 1e6  # Divide the damage data by 10^7 for visualization
    max_damage = damage_data.max()

    # ax.set_yticks(yticks_damage)
    ax.yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.bar(
        pop_data['clim_scen'],
        damage_data,  # Assuming 'Total building damage' column exists
        color=colors[0],
        width=0.25,
        edgecolor='white',
        zorder=2,
    )

    ax.set_ylabel('Damage [10e6 USD]' if i == 0 else "")
    title = ax.set_title(f'{dict_fig2[i]} Economic damage for {title}', fontsize=14, fontweight='medium', loc='left', pad=10)
    title.set_position([-0.1, 1])  # Adjust the first value to move left/right, second value to move up/down
    ax.set_xticklabels(pop_data['clim_scen'], rotation=0, ha='center')

# add a vertical space between the two rows
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Create a single legend for the population metrics
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, bbox_to_anchor=(0.5, 0.54), ncol=3, loc='upper center', fontsize=11)

# Save the figure
plt.savefig(r'D:\paper_4\data\Figures\paper_figures\population_exposure_and_damage_bar.png', dpi=250, bbox_inches='tight')
plt.show()

##########################################################################################
# plot maps
tiff_paths = [
    rf'{hazard_folder_path}\hmax_idai_ifs_rebuild_bc_hist_rain_surge_noadapt.tiff',
    rf'{hazard_folder_path}\hmax_idai_ifs_rebuild_bc_hightide_rain_surge_noadapt.tiff',
    rf'{hazard_folder_path}\hmax_idai_ifs_rebuild_bc_3c_rain_surge_noadapt.tiff',
    rf'{hazard_folder_path}\hmax_idai_ifs_rebuild_bc_3c-hightide_rain_surge_noadapt.tiff'
]
        # add a title where the tiff_path.split('_')[-4] is converted using scen_dict
dict_letters = {'Baseline':"a)",
                "Springtide":"b)",
                "3C":"c)",
                "3C-springtide":"d)"}
landareas_path = rf'D:\paper_4\data\sfincs_input\data_deltares_idai\osm_landareas.gpkg'
gdf_landareas = gpd.read_file(landareas_path)
# change crs to match the tiff
gdf_landareas = gdf_landareas.to_crs('EPSG:32736')

# Define colorbar boundaries
boundaries = np.linspace(0, 3, 11)  # 10 intervals from 0 to 3
norm = BoundaryNorm(boundaries, ncolors=256)

gdf_to_plot = gdf_landareas

# Create a figure with 3 subplots
fig, axes = plt.subplots(2, 2, figsize=(doublecol*1.6, doublecol*1.4), sharex='col', sharey='row', dpi=300)
axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it
# Loop through each TIFF path and plot it
for ax, tiff_path in zip(axes, tiff_paths):
    with rasterio.open(tiff_path) as src:        
        # Plot the GeoPackage data as basemap
        gdf_to_plot.plot(ax=ax, color='darkgray')  # Adjust color as needed

        # Plot the raster data
        show(src, ax=ax, cmap='Blues', norm=norm, zorder=2)

        # Set the axis limits to the raster bounds
        ax.set_xlim([src.bounds.left+100, src.bounds.right-2000])
        ax.set_ylim([src.bounds.bottom+1300, src.bounds.top-2000])        
        ax.set_title(f"{dict_letters[scen_dict[tiff_path.split('_')[-4]]]} {scen_dict[tiff_path.split('_')[-4]]}", fontsize=12, fontweight='medium', loc='left')
        
# Adjust the position of the color bar to not overlap with the subplots
plt.subplots_adjust(wspace=0.091, hspace=0.05, right=0.95)
cbar_ax = fig.add_axes([0.96, 0.15, 0.015, 0.7])
# Create a common color bar for all subplots
cb = ColorbarBase(cbar_ax, cmap='Blues', norm=norm, boundaries=boundaries, ticks=boundaries, spacing='proportional', orientation='vertical')
cb.set_label('Inundation depth (m)')
for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
# save figure in D:\paper_4\data\Figures\paper_figures
plt.savefig(r'D:\paper_4\data\Figures\paper_figures\inundation_depth.png', dpi=300, bbox_inches='tight')
plt.show()


##########################################################################################
# Function to calculate raster difference
def calculate_raster_difference(path_a, path_b):
    with rasterio.open(path_a) as src_a:
        data_a = src_a.read(1).astype(float)  # Ensure the data is in float for NaN handling
        profile = src_a.profile
        # Replace NA values with 0 only where data_b is not NA
        data_a_masked = np.where(~np.isnan(data_a), data_a, 0)
        
    with rasterio.open(path_b) as src_b:
        data_b = src_b.read(1).astype(float)  # Ensure the data is in float for NaN handling
        # Replace NA values with 0 only where data_a is not NA
        data_b_masked = np.where(~np.isnan(data_b), data_b, 0)
        
    # Calculate the difference where both have data
    difference = np.where(~np.isnan(data_a) & ~np.isnan(data_b), data_a - data_b, 
                          # Where data_a is NA and data_b is not
                          np.where(np.isnan(data_a) & ~np.isnan(data_b), -data_b, 
                                   # Where data_b is NA and data_a is not
                                   np.where(~np.isnan(data_a) & np.isnan(data_b), data_a, 
                                            np.nan)))  # Keep as NA where both are NA
    
    return difference, profile

# Calculate the differences
diff_2_minus_0, profile = calculate_raster_difference(tiff_paths[2], tiff_paths[0])
diff_3_minus_1, _ = calculate_raster_difference(tiff_paths[3], tiff_paths[1])
# Load original rasters to create masks
with rasterio.open(tiff_paths[0]) as src_0:
    data_0 = src_0.read(1).astype(float)  # Ensure float for NaN handling

with rasterio.open(tiff_paths[1]) as src_1:
    data_1 = src_1.read(1).astype(float)

# Create masks
mask_2_minus_0 = np.isnan(data_0) & ~np.isnan(diff_2_minus_0)
mask_3_minus_1 = np.isnan(data_1) & ~np.isnan(diff_3_minus_1)

# Original Rasters Color Range
vmin_orig, vmax_orig = 0, 3
# Difference Rasters Color Range
vmin_diff, vmax_diff = -2, 2

fig, axes = plt.subplots(2, 2, figsize=(17, 15), sharex='col', sharey='row')
axes = axes.flatten()
# Plot the original raster data for the top two subplots
for i, tiff_path in enumerate(tiff_paths[:2]):
    with rasterio.open(tiff_path) as src:
        gdf_landareas.plot(ax=axes[i], color='darkgray')
        show(src, ax=axes[i], cmap='Blues', norm=Normalize(vmin=vmin_orig, vmax=vmax_orig), zorder=2)
        axes[i].set_title(tiff_path.split('_')[-4])

                # Set the axis limits to the raster bounds
        axes[i].set_xlim([src.bounds.left+100, src.bounds.right-2000])
        axes[i].set_ylim([src.bounds.bottom+1300, src.bounds.top-2000])

# Plot the differences for the bottom two subplots
diff_arrays = [diff_2_minus_0, diff_3_minus_1]
for i, (diff_array, mask) in enumerate(zip(diff_arrays, [mask_2_minus_0, mask_3_minus_1]), start=2):
    title_appendix = '3C - Hist' if i == 2 else '3C-Hightide - Hightide'
    gdf_landareas.plot(ax=axes[i], color='darkgray')
    im = axes[i].imshow(diff_array, cmap='BrBG_r', norm=Normalize(vmin=vmin_diff, vmax=vmax_diff), zorder=2,
                        extent=[profile['transform'][2], profile['transform'][2] + profile['transform'][0] * profile['width'],
                                profile['transform'][5] + profile['transform'][4] * profile['height'], profile['transform'][5]])
    
    # overlay_array = np.ma.masked_where(~mask, mask)
    # axes[i].imshow(overlay_array, cmap='gray', hatch='///', alpha=0)
    axes[i].set_title(f'{title_appendix}')

    # Set the axis limits to the raster bounds
    axes[i].set_xlim([src.bounds.left+100, src.bounds.right-2000])
    axes[i].set_ylim([src.bounds.bottom+1300, src.bounds.top-2000])

# Adjust layout for colorbars
plt.subplots_adjust(wspace=-0.1, hspace=0.1, right=0.95)

# Add colorbars
# Original rasters colorbar
cbar_ax_orig = fig.add_axes([0.93, 0.53, 0.014, 0.35])  # Adjust these values as needed
cb_orig = ColorbarBase(cbar_ax_orig, cmap='Blues', norm=Normalize(vmin=vmin_orig, vmax=vmax_orig), orientation='vertical')
cb_orig.set_label('Inundation Depth (m)')

# Difference rasters colorbar
cbar_ax_diff = fig.add_axes([0.93, 0.12, 0.014, 0.35])  # Adjust these values as needed
cb_diff = ColorbarBase(cbar_ax_diff, cmap='BrBG_r', norm=Normalize(vmin=vmin_diff, vmax=vmax_diff), orientation='vertical')
cb_diff.set_label('Inundation Depth Difference (m)')

plt.show()



from shapely.ops import unary_union
from matplotlib.patches import Patch
##########################################################################################
# NOW FOR THE IMPACT VECTORS
gpkg_paths = [
    rf'{fiat_output_path}\spatial_idai_ifs_rebuild_bc_hist_rain_surge_noadapt.gpkg',
    rf'{fiat_output_path}\spatial_idai_ifs_rebuild_bc_hightide_rain_surge_noadapt.gpkg',
    rf'{fiat_output_path}\spatial_idai_ifs_rebuild_bc_3c_rain_surge_noadapt.gpkg',
    rf'{fiat_output_path}\spatial_idai_ifs_rebuild_bc_3c-hightide_rain_surge_noadapt.gpkg'
]

def process_gdf(gdf, merged_gdf_exp):
    # Merge the input GeoDataFrame with 'merged_gdf_exp' on 'Object ID'
    merged_gdf = gdf.merge(merged_gdf_exp[['Object ID', "Primary Object Type", 'Max Potential Damage: Structure']], on='Object ID')
    # Drop the rows where 'Total Damage' is 0
    merged_gdf = merged_gdf[merged_gdf['Total Damage'] > 0]
    # Calculate 'damage prc'
    merged_gdf['damage prc'] = (merged_gdf['Total Damage'] / merged_gdf['Max Potential Damage: Structure']) * 100
    
    return merged_gdf

gpkg_exp_path = rf'D:\paper_4\data\vanPanos\FIAT_model_new\Exposure\exposure_clip5.gpkg'
gdf_exp = gpd.read_file(gpkg_exp_path)
# load csv exposure
csv_exp_path = rf'D:\paper_4\data\vanPanos\FIAT_model_new\Exposure\exposure_clip5.csv'
gdf_exp_csv = pd.read_csv(csv_exp_path)
# add a column named maximum damage by joining over Object ID columns of both gdf_exp and gdf_exp_csv
merged_gdf_exp = gdf_exp.merge(gdf_exp_csv, on='Object ID')

# Pre-load and merge GPKG files with the exposure data
merged_gdfs = []
for gpkg_path in gpkg_paths:
    gdf = gpd.read_file(gpkg_path)
    merged_gdf = process_gdf(gdf, merged_gdf_exp)
    merged_gdfs.append(merged_gdf)

# Create a unary union of the 'Residential4' polygons to plot informal settlements
merged_gdf_exp = merged_gdf_exp.to_crs(merged_gdf.crs)
# filter the 'Primary Object Type' column to only include 'Residential4' in merged_gdf_exp
df_exp_residential4_df = merged_gdf_exp[merged_gdf_exp['Primary Object Type'] == 'Residential4']
# buffer the 'Residential4' by 60 meters
df_exp_buffered = df_exp_residential4_df.buffer(50)
# create a unary union of the buffered 'Residential4'
df_exp_residential4_df_bf = unary_union(df_exp_buffered)
# create a GeoDataFrame of the buffered 'Residential4'
gdf_exp_residential4 = gpd.GeoDataFrame(geometry=[df_exp_residential4_df_bf], crs=merged_gdf_exp.crs)

# Adjust the linspace boundaries for your specific range and number of intervals
boundaries = np.linspace(0, 100, 11)  # Creates 10 intervals from 0 to 100
norm = BoundaryNorm(boundaries, ncolors=256, clip=True)
cmap = plt.get_cmap('Reds')  # Adjust 'Reds' to your preferred colormap

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(doublecol*1.6, doublecol*1.4), sharex=True, sharey=True, dpi=300)
axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it
# Loop through each GPKG path and plot it
for ax, merged_gdf, gpkg_path in zip(axes, merged_gdfs, gpkg_paths):    # Load the current GeoPackage
    # Set the axis limits to the raster bounds
    ax.set_xlim([src.bounds.left+1000, src.bounds.right-2500])
    ax.set_ylim([src.bounds.bottom+1300, src.bounds.top-2500])
    gdf_to_plot.plot(ax=ax, color='lightgray')  # Adjust color as needed
    merged_gdf.plot(ax=ax, column = 'damage prc', cmap=cmap, norm=norm, markersize=1, zorder=2)  # Adjust color and edgecolor as needed #edgecolor='black'
    gdf_exp_residential4.plot(ax=ax, facecolor='none', edgecolor = 'black', zorder=3, alpha=1, label = 'Informal setllements' )  # Adjust color, markersize, and alpha as needed
    ax.set_title(f"{dict_letters[scen_dict[gpkg_path.split('_')[-4]]]} {scen_dict[gpkg_path.split('_')[-4]]}", fontsize=12, fontweight='medium', loc='left')
# Adjust layout
plt.subplots_adjust(wspace= 0.09, hspace=0.1)
# Colorbar setup
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])  # Adjust these values as needed
fig.colorbar(sm, cax=cbar_ax).set_label('Relative damage (%)', pad=-3)
# add legend
informal_settlements_legend = Patch(facecolor='none', edgecolor='black', label='Informal settlements')
plt.legend(handles=[informal_settlements_legend], fontsize=10,frameon=False, bbox_to_anchor=(-1.40, -0.045))
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

# save figure in D:\paper_4\data\Figures\paper_figures
plt.savefig(r'D:\paper_4\data\Figures\paper_figures\impact_vectors.png', dpi=300, bbox_inches='tight')
plt.show()

# Adjust the linspace boundaries for your specific range and number of intervals
boundaries = np.linspace(0, 0.9*merged_gdfs[3]['Total Damage'].mean(), 11)  # Creates 10 intervals from 0 to 100
norm = BoundaryNorm(boundaries, ncolors=256, clip=True)
cmap = plt.get_cmap('Reds')  # Adjust 'Reds' to your preferred colormap

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(doublecol*1.6, doublecol*1.4), sharex=True, sharey=True, dpi=300)
axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it
# Loop through each GPKG path and plot it
for ax, merged_gdf, gpkg_path in zip(axes, merged_gdfs, gpkg_paths):    # Load the current GeoPackage
    # Set the axis limits to the raster bounds
    ax.set_xlim([src.bounds.left+1000, src.bounds.right-2500])
    ax.set_ylim([src.bounds.bottom+1300, src.bounds.top-2500])
    gdf_to_plot.plot(ax=ax, color='lightgray')  # Adjust color as needed
    merged_gdf.plot(ax=ax, column = 'Total Damage', cmap=cmap, norm=norm, markersize=1, zorder=2)  # Adjust color and edgecolor as needed #edgecolor='black'
    gdf_exp_residential4.plot(ax=ax, facecolor='none', edgecolor = 'black', zorder=3, alpha=1, label = 'Informal setllements' )  # Adjust color, markersize, and alpha as needed
    ax.set_title(f"{dict_letters[scen_dict[gpkg_path.split('_')[-4]]]} {scen_dict[gpkg_path.split('_')[-4]]}", fontsize=12, fontweight='medium', loc='left')
# Adjust layout
plt.subplots_adjust(wspace= 0.09, hspace=0.1)
# Colorbar setup
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])  # Adjust these values as needed
fig.colorbar(sm, cax=cbar_ax).set_label('Physical Damage (USD)', pad=-3)
# add legend
informal_settlements_legend = Patch(facecolor='none', edgecolor='black', label='Informal settlements')
plt.legend(handles=[informal_settlements_legend], fontsize=10,frameon=False, bbox_to_anchor=(-1.40, -0.045))
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

# save figure in D:\paper_4\data\Figures\paper_figures
plt.savefig(r'D:\paper_4\data\Figures\paper_figures\impact_vectors_total.png', dpi=300, bbox_inches='tight')
plt.show()

##########################################################################################
#2nd attempt: differences plot maps
run_name = '3c-hightide'

diff_paths = [
    rf'{fiat_output_path}\spatial_idai_ifs_rebuild_bc_{run_name}_rain_surge_noadapt.gpkg',
    rf'{fiat_output_path}\spatial_idai_ifs_rebuild_bc_{run_name}_rain_surge_hold.gpkg',
    rf'{fiat_output_path}\spatial_idai_ifs_rebuild_bc_{run_name}_rain_surge_retreat.gpkg'
]

# Pre-load and merge GPKG files with the exposure data
merged_diff_gdfs = []
for gpkg_path in diff_paths:
    gdf = gpd.read_file(gpkg_path)
    merged_gdf = process_gdf(gdf, merged_gdf_exp)
    merged_diff_gdfs.append(merged_gdf)

# remove rows with 0 total damage
merged_diff_gdfs = [gdf[gdf['Total Damage'] > 0] for gdf in merged_diff_gdfs]
gdf_no_adapt = merged_diff_gdfs[0]
gdf_hold = merged_diff_gdfs[1]
gdf_retreat = merged_diff_gdfs[2]

# Combine 'Object ID' from all GeoDataFrames to find the unique set
all_ids = pd.Series(pd.concat([gdf_no_adapt['Object ID'], gdf_hold['Object ID'], gdf_retreat['Object ID']]).unique(), name='Object ID')

# First, create a master dictionary of 'Object IDs' to geometries
master_geometry_dict = {}
for gdf in [gdf_no_adapt, gdf_hold, gdf_retreat]:
    for object_id, geometry in zip(gdf['Object ID'], gdf.geometry):
        master_geometry_dict[object_id] = geometry

# Align GDFs
gdf_no_adapt_aligned = align_gdf(gdf_no_adapt, all_ids, master_geometry_dict, "damage prc")
gdf_hold_aligned = align_gdf(gdf_hold, all_ids, master_geometry_dict, "damage prc")
gdf_retreat_aligned = align_gdf(gdf_retreat, all_ids, master_geometry_dict, "damage prc")

diff_gdf_1 = gdf_no_adapt_aligned.copy()
diff_gdf_1['damage prc'] = gdf_hold_aligned['damage prc'] - gdf_no_adapt_aligned['damage prc']

diff_gdf_2 = gdf_no_adapt_aligned.copy()
diff_gdf_2['damage prc'] = gdf_retreat_aligned['damage prc'] - gdf_no_adapt_aligned['damage prc']

diff_gdf_3 = gdf_hold_aligned.copy()
diff_gdf_3['damage prc'] = gdf_retreat_aligned['damage prc'] - gdf_hold_aligned['damage prc']


# Determine global min and max for consistent color scaling across subplots
vmin = -100
vmax = 100
# Define the original colormap
original_cmap = plt.cm.BrBG_r
boundaries = np.linspace(vmin, vmax, 11)  # Creates 10 intervals between 0 and 200
# Create a normalization based on these boundaries
norm = BoundaryNorm(boundaries, ncolors=256, clip=True)

# Figure comparing the adaptations to the no adapatation scenario
fig, axes = plt.subplots(1, 2, figsize=(doublecol*1.6, doublecol*1), sharex=True, sharey=True)
# Base plotting for all points
gdf_to_plot.plot(ax=axes[0], color='lightgray', zorder=1)  # Adjust color as needed
gdf_to_plot.plot(ax=axes[1], color='lightgray', zorder=1)  # Adjust color as needed

gdf_exp_residential4.plot(ax=axes[0], facecolor='none', edgecolor = 'black', zorder=3, alpha=1, label = 'Informal setllements' )  # Adjust color, markersize, and alpha as needed
gdf_exp_residential4.plot(ax=axes[1], facecolor='none', edgecolor = 'black', zorder=3, alpha=1, label = 'Informal setllements' )  # Adjust color, markersize, and alpha as needed

diff_gdf_1.plot(ax=axes[0], column='damage prc', cmap=original_cmap, norm=norm, 
                markersize=1, linewidth=0.0, zorder=2)
axes[0].set_title('a) Damage change for Hold the Line', fontsize=12, fontweight='medium', loc='left')

diff_gdf_2.plot(ax=axes[1], column='damage prc', cmap=original_cmap, norm=norm,
                markersize=1, linewidth=0.0, zorder=2)
axes[1].set_title('b) Damage change for Integrated', fontsize=12, fontweight='medium', loc='left')
# axes[1].axis('off')
#remove horizontal space between subplots
plt.subplots_adjust(wspace=0.3)

for ax in axes:
    # Set the axis limits to the raster bounds
    ax.set_xlim([src.bounds.left+1000, src.bounds.right-2500])
    ax.set_ylim([src.bounds.bottom+1300, src.bounds.top-2500])
    ax.set_xticks([])
    ax.set_yticks([])
# Create and position the color bar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.454, 0.29, 0.015, 0.44])  # Adjust these values as needed
sm = ScalarMappable(norm=norm, cmap=original_cmap)
fig.colorbar(sm, cax=cbar_ax, boundaries=boundaries[::2], ticks=boundaries[::2], orientation='vertical').set_label('Change in damage (%)', labelpad=-4)
informal_settlements_legend = Patch(facecolor='none', edgecolor='black', label='Informal settlements')
plt.legend(handles=[informal_settlements_legend], fontsize=10,frameon=False, bbox_to_anchor=(-1.0, 0.0))

# save figure in D:\paper_4\data\Figures\paper_figures
plt.savefig(r'D:\paper_4\data\Figures\paper_figures\adaptation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# Process the GeoDataFrames
gdf_hold_merged = process_gdf(gdf_hold, merged_gdf_exp)
gdf_retreat_merged = process_gdf(gdf_retreat, merged_gdf_exp)

# Create a figure with 2x2 subplots

# Adjust the linspace boundaries for your specific range and number of intervals
boundaries = np.linspace(0, 100, 11)  # Creates 10 intervals from 0 to 100
norm = BoundaryNorm(boundaries, ncolors=256, clip=True)
cmap = plt.get_cmap('Reds')  # Adjust 'Reds' to your preferred colormap

fig, axs = plt.subplots(2, 2, figsize=(doublecol*1.6, doublecol*1.3), sharex=False, sharey=False, dpi=300, gridspec_kw={'height_ratios': [1, 2]})
axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration

for ax, metric, ylabel, yticks in zip(axs[:2], ['difference_pop_prc', 'difference_damage_prc'], ['Population reduction (%)', 'Damage reduction (%)'], [yticks_population, yticks_damage]):
    for i, adapt_scen in enumerate(adapt_scens):
        # Filter and sort DataFrame for the current adaptation scenario
        adapt_scen_df = combined_df[combined_df['adapt_scen'] == adapt_scen]
        adapt_scen_df = adapt_scen_df.set_index('clim_scen').reindex(clim_scens).reset_index()
        # Calculate offsets for side-by-side bars
        offsets = clim_scen_positions + (i - np.mean(range(len(adapt_scens)))) * bar_width
        # Plotting
        ax.bar(offsets, adapt_scen_df[metric], width=bar_width*0.9, color=metric_colors[metric], label=adapt_scen, hatch=hatches[i], edgecolor='white', zorder=2)
        # Apply common styles
        ax.yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticks(yticks)

    ax.set_ylabel(ylabel)

# Additional formatting
title = axs[0].set_title('a) Changes in exposed population', fontsize=13, fontweight='medium', loc='left', pad=20)
title_2 = axs[1].set_title('b) Changes in building damage', fontsize=13, fontweight='medium', loc='left', pad=20)
title.set_position([-0.1, 2])  # Adjust the first value to move left/right, second value to move up/down
title_2.set_position([-0.11, 2])  # Adjust the first value to move left/right, second value to move up/down
axs[0].set_xticks(clim_scen_positions)
axs[1].set_xticks(clim_scen_positions)
axs[0].set_xticklabels(climate_order)
axs[1].set_xticklabels(climate_order)
# axs[1].set_xlabel('Hydrometeorological scenarios')
# add some vertical space between the plots
plt.subplots_adjust(wspace=0.3)
# Create a single legend for the first plot for clarity
handles, labels = axs[0].get_legend_handles_labels()

# Creating custom legend handles for hatching, ignoring color
legend_handles = [mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch, label=label)
                  for hatch, label in zip(hatches, adapt_scens)]
fig.legend(handles=legend_handles, frameon=False, bbox_to_anchor=(0.80, 0.95), ncol=1, loc='upper left', fontsize=10)

for ax, gdf, title in zip(axs[2:], [gdf_hold_merged, gdf_retreat_merged], ['c) Hold the Line under 3C-springide', 'd) Integrated under 3C-springide']):
    # Set the axis limits to the raster bounds
    ax.set_xlim([src.bounds.left+1000, src.bounds.right-2500])
    ax.set_ylim([src.bounds.bottom+1300, src.bounds.top-2500])
    gdf_to_plot.plot(ax=ax, color='lightgray')  # Adjust color as needed
    gdf.plot(ax=ax, column='damage prc', cmap=cmap, norm=norm, markersize=1, zorder=2)  # Adjust color and edgecolor as needed
    gdf_exp_residential4.plot(ax=ax, facecolor='none', edgecolor='black', zorder=3, alpha=1, label='Informal settlements')  # Adjust color, markersize, and alpha as needed
    title_3 = ax.set_title(title, fontsize=13, fontweight='medium', loc='left', pad=20)
    title_3.set_position([-0.07, 0.0])  # Adjust the first value to move left/right, second value to move up/down
    ax.set_xticks([])
    ax.set_yticks([])
# # Adjust layout
# plt.subplots_adjust(wspace=0.3, hspace=0.4)

# Colorbar setup
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar_ax = fig.add_axes([0.48, 0.18, 0.015, 0.34])  # Adjust these values as needed
fig.colorbar(sm, cax=cbar_ax).set_label('Relative damage (%)', labelpad=-5)

# # Assuming your legend setup
informal_settlements_legend = Patch(facecolor='none', edgecolor='black', label='Informal settlements')
fig.legend(handles=[informal_settlements_legend], fontsize=10, frameon=False, bbox_to_anchor=(0.22, 0.175), loc='upper center', ncol=1)


plt.savefig(r'D:\paper_4\data\Figures\paper_figures\combined_impact_bars_maps.png', dpi=300, bbox_inches='tight')
plt.show()


##########################################################################################
# FIGURE FOR INFORMAL SETTLEMENTS
##########################################################################################

import glob
# Specify the base directory where the search will start
base_dir = r"D:\paper_4\data\FloodAdapt-GUI\Database\beira\output\Scenarios"

# Create a pattern to match all the relevant CSV files
pattern = os.path.join(base_dir, '**', 'Impacts_detailed_idai_ifs_*.csv')

# Use glob to find all files matching the pattern
csv_files = glob.glob(pattern, recursive=True)

# Initialize an empty list to store DataFrames
df_list = []

# Loop through the found CSV files and process each one
for file in csv_files:
    # Extract the scenario name from the filename
    scenario = os.path.splitext(os.path.basename(file))[0].replace('Impacts_detailed_idai_ifs_rebuild_bc_', '')
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file)
    
    # Add a new column for the scenario
    df['climate_scen'] = scenario
    
    # Append the DataFrame to the list
    df_list.append(df)

combined_df_damages = pd.concat(df_list, ignore_index=True)

# subset the data to only include the 'Residential4' type
combined_df_residential4 = combined_df_damages[combined_df_damages['Primary Object Type'] == 'Residential4']
combined_df_residential4 = combined_df_residential4[combined_df_residential4['Total Damage'] > 0]

informal_impact_per_scenario = combined_df_residential4.groupby('climate_scen').agg(
    total_damage=pd.NamedAgg(column='Total Damage', aggfunc='sum'),
    total_population=pd.NamedAgg(column='population', aggfunc='sum')).reset_index()

# Split 'climate_scen' into 'climate_scen' and 'adapt_scen'
informal_impact_per_scenario['adapt_scen'] = informal_impact_per_scenario['climate_scen'].apply(lambda x: x.split('_', -1)[-1])
informal_impact_per_scenario['climate_scen'] = informal_impact_per_scenario['climate_scen'].apply(lambda x: x.split('_', 1)[0])
informal_impact_per_scenario = informal_impact_per_scenario[['climate_scen', 'adapt_scen', 'total_damage', 'total_population']]


# Extract 'noadapt' data
noadapt_df = informal_impact_per_scenario[informal_impact_per_scenario['adapt_scen'] == 'noadapt'].set_index('climate_scen')

# Merge with the original DataFrame to calculate differences
merged_df = informal_impact_per_scenario.merge(noadapt_df, on='climate_scen', suffixes=('', '_noadapt'))

# Calculate percentage differences
merged_df['difference_damage_prc'] = 100 * (merged_df['total_damage'] - merged_df['total_damage_noadapt']) / merged_df['total_damage_noadapt']
merged_df['difference_pop_prc'] = 100 * (merged_df['total_population'] - merged_df['total_population_noadapt']) / merged_df['total_population_noadapt']

# Drop unnecessary columns
informal_impact_per_scenario_diff = merged_df[['climate_scen', 'adapt_scen', 'total_damage','total_population', 'difference_damage_prc', 'difference_pop_prc']]
informal_impact_per_scenario_diff = informal_impact_per_scenario_diff.reset_index(drop=True)

informal_impact_diff = informal_impact_per_scenario_diff[['climate_scen', 'adapt_scen', 'difference_damage_prc', 'difference_pop_prc']]
# drop rows where adapt_scen == noadapt
informal_impact_diff = informal_impact_diff[informal_impact_diff['adapt_scen'] != 'noadapt']
# now  convert adapt_scen using adapt_dict
informal_impact_diff['adapt_scen'] = informal_impact_diff['adapt_scen'].map(adapt_dict)
# convert climate_scen using scen_dict
informal_impact_diff['climate_scen'] = informal_impact_diff['climate_scen'].map(scen_dict)


# Prepare the figure and axes climate_order
fig, axs = plt.subplots(1, 2, figsize=(doublecol*1.6, singlecol*1.1), dpi=300, sharex=True, sharey=False)
adapt_scens2 = ['Hold the line', 'Integrated']
# Define y-ticks for population and damage plots
# Common settings for both plots
tick_increment_population = 20  # Adjust based on your actual data
tick_increment_damage = 20  # Adjust based on your actual data
# Define y-ticks for population and damage plots
y_max_population = -80
yticks_population = np.arange(y_max_population, 0.01, tick_increment_population)
y_max_damage = -80
yticks_damage = np.arange(y_max_damage, 0.01, tick_increment_damage)
# Plot the data
for ax, metric, ylabel, yticks in zip(axs, ['difference_pop_prc', 'difference_damage_prc'], ['Reduction (%)', ''], [yticks_population, yticks_damage]):
    for i, adapt_scen in enumerate(adapt_scens2):
        # Filter and sort DataFrame for the current adaptation scenario
        adapt_scen_df = informal_impact_diff[informal_impact_diff['adapt_scen'] == adapt_scen]
        adapt_scen_df = adapt_scen_df.set_index('climate_scen').reindex(climate_order).reset_index()
        
        # Calculate offsets for side-by-side bars
        offsets = clim_scen_positions + (i - np.mean(range(len(adapt_scens)))) * bar_width
        
        # Plotting
        ax.bar(offsets, adapt_scen_df[metric], width=bar_width, color=metric_colors[metric], label=adapt_scen, hatch=hatches[i], edgecolor='white', zorder=2)
        
        # Apply common styles
        ax.yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticks(yticks)
        ax.set_ylabel(ylabel)

    ax.set_xticks(clim_scen_positions)
    ax.set_xticklabels(climate_order, rotation=45, ha='right')

# Additional formatting
axs[0].set_title('a) Exposed population in informal settlements', fontsize=12, fontweight='medium', loc='left', pad=20)
axs[1].set_title('b) Damage in informal settlements', fontsize=12, fontweight='medium', loc='left', pad=20)
# Adjust layout to prevent overlap
plt.tight_layout()

# Create a legend for hatching patterns
handles, labels = axs[0].get_legend_handles_labels()
# legend_handles = [mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch, label=label) for hatch, label in hatches.items()]
# fig.legend(handles=legend_handles, frameon=False, bbox_to_anchor=(0.5, -0.05), ncol=2, loc='upper center', fontsize=10)

# Creating custom legend handles for hatching, ignoring color
legend_handles = [mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch, label=label)
                  for hatch, label in zip(hatches, adapt_scens2)]
fig.legend(handles=legend_handles, frameon=False, bbox_to_anchor=(0.84, 0.99), ncol=1, loc='upper left', fontsize=10)

plt.savefig(r'D:\paper_4\data\Figures\paper_figures\informal_settlements_impact_bars.png', dpi=300, bbox_inches='tight')
plt.show()
# ##########################################################################################
# now plot map only for the validation part
# import numpy.ma as ma
# import rioxarray as rxr
# # Load the raster file
# src_hmdg = rxr.open_rasterio(r'D:\paper_4\data\mester_TC_Idai_data_collection\factual\2019063S18038_cf-zos_aviso-fes_no.tif') #D:\paper_4\data\floodadapt_results\hazard_tiff\hmax_idai_ifs_rebuild_bc_hist_rain_surge_noadapt.tiff
# src_hmdg_newcrs = src_hmdg.rio.reproject(src.crs)
# src_hmdg_newcrs = src_hmdg_newcrs.where(src_hmdg_newcrs > 0.1)
# src_hmdg_newcrs = src_hmdg_newcrs.isel(band=0)
# # Specify the output file path
# output_file = r'D:\paper_4\data\mester_TC_Idai_data_collection\factual\2019063S18038_cf-zos_aviso-fes_no_convert.tif'
# # Save the reprojected raster to a new .tif file
# src_hmdg_newcrs.rio.to_raster(output_file)
# # Define colorbar boundaries
# boundaries = np.linspace(0, 4, 11)  # 10 intervals from 0 to 3
# norm = BoundaryNorm(boundaries, ncolors=256)
# # make plot like the previous one
# fig, ax = plt.subplots(1, 1, figsize=(doublecol*1.6, singlecol*1.1), dpi=300)
# # Plot the raster
# gdf_to_plot.plot(ax=ax, color='darkgray' )  # Adjust color as needed
# src_hmdg_newcrs.plot.imshow(ax=ax, cmap='Blues', zorder=2, norm=norm, add_colorbar=False, add_labels=False)

# ax.set_xlim([src.bounds.left+1000, src.bounds.right-2500])
# ax.set_ylim([src.bounds.bottom+1300, src.bounds.top-2500])
# ax.set_title(f"Simulated flood", fontsize=12, fontweight='medium', loc='left')
# # Adjust the position of the color bar to not overlap with the subplots
# plt.subplots_adjust(wspace=0.091, hspace=0.05, right=0.95)
# cbar_ax = fig.add_axes([0.96, 0.15, 0.015, 0.7])
# # Create a common color bar for all subplots
# cb = ColorbarBase(cbar_ax, cmap='Blues', norm=norm, boundaries=boundaries, ticks=boundaries, spacing='proportional', orientation='vertical')
# cb.set_label('Inundation depth (m)')
# for ax in axes.flatten():
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.show()


tiff_paths_2 = [r'D:\paper_4\data\floodadapt_results\hazard_tiff\hmax_idai_ifs_rebuild_bc_hist_rain_surge_noadapt.tiff', r'D:\paper_4\data\mester_TC_Idai_data_collection\factual\2019063S18038_cf-zos_aviso-fes_no_convert.tif']
vector_paths = [
    r'D:\paper_4\data\beira_vector\EMSR348_03BEIRA_DEL_MONIT01_v2_observed_event_a.shp',
    r'D:\paper_4\data\beira_vector\v2\EMSR348_03BEIRA_DEL_MONIT02_v2_observed_event_a.shp']
titles = ["a) Satellite imagery", "b) Baseline scenario", "c) Mester et al., 2023"]
boundaries = np.linspace(0, 4, 11)  # 10 intervals from 0 to 3
norm = BoundaryNorm(boundaries, ncolors=256)

# Load the vector data
gdf_list = [gpd.read_file(path) for path in vector_paths]

# Merge the vector GeoDataFrames
gdf_merged = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
# Reproject the vector data to the same CRS as the raster
gdf_vector = gdf_merged.to_crs(src.crs)

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(doublecol*1.8, doublecol*1), sharex='col', sharey='row', dpi=300)
axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it
# Load and plot the vector data on the first subplot
ax = axes[0]
gdf_to_plot.plot(ax=ax, color='darkgray')  # Plot background
gdf_vector.plot(ax=ax, color='black', edgecolor='black')  # Plot vector data
ax.set_title(titles[0], fontsize=12, fontweight='medium', loc='left')
ax.set_xlim([src.bounds.left+1000, src.bounds.right-1500])
ax.set_ylim([src.bounds.bottom+1300, src.bounds.top-1000])
# Loop through each TIFF path and plot it
for ax, tiff_path, title in zip(axes[1:], tiff_paths_2, titles[1:]):
    with rasterio.open(tiff_path) as src2:        
        # Plot the GeoPackage data as basemap
        gdf_to_plot.plot(ax=ax, color='darkgray')  # Adjust color as needed

        # Plot the raster data
        show(src2, ax=ax, cmap='Blues', norm=norm, zorder=2)
        
        ax.set_xlim([src.bounds.left+1000, src.bounds.right-1500])
        ax.set_ylim([src.bounds.bottom+1300, src.bounds.top-1000])
        ax.set_title(title, fontsize=12, fontweight='medium', loc='left')
plt.subplots_adjust(wspace=0.091, hspace=0.05, right=0.95)
cbar_ax = fig.add_axes([0.96, 0.3, 0.015, 0.4])
# Create a common color bar for all subplots
cb = ColorbarBase(cbar_ax, cmap='Blues', norm=norm, boundaries=boundaries, ticks=boundaries, spacing='proportional', orientation='vertical')
cb.set_label('Inundation depth (m)')
for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig(r'D:\paper_4\data\Figures\paper_figures\inundation_validation.png', dpi=300, bbox_inches='tight')
plt.show()



# plot for strategies
gdf_elevate_port_2m = gpd.read_file(r'D:\paper_4\data\FloodAdapt-GUI\Database\beira\input\measures\elevate_port_2m\elevate_port_2m.geojson')
gdf_buyout_informal = gpd.read_file(r'D:\paper_4\data\FloodAdapt-GUI\Database\beira\input\measures\buyout_informal\buyout_informal.geojson')
gdf_outer_wall = gpd.read_file(r'D:\paper_4\data\FloodAdapt-GUI\Database\beira\input\measures\outer_wall\outer_wall.geojson')
gdf_inner_wall = gpd.read_file(r'D:\paper_4\data\FloodAdapt-GUI\Database\beira\input\measures\inner_wall\inner_wall.geojson')
gdf_buildings = gpd.read_file(r'D:\paper_4\data\FloodAdapt-GUI\Database\beira\static\templates\fiat\footprints\buildings.shp')
gdf_outer_wall = gdf_outer_wall.to_crs('EPSG:32736')
gdf_inner_wall = gdf_inner_wall.to_crs('EPSG:32736')
gdf_buyout_informal = gdf_buyout_informal.to_crs('EPSG:32736')
gdf_elevate_port_2m = gdf_elevate_port_2m.to_crs('EPSG:32736')
gdf_buildings = gdf_buildings.to_crs('EPSG:32736')
gdf_buildings = gdf_buildings.cx[src.bounds.left:src.bounds.right, src.bounds.bottom:src.bounds.top]


# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(doublecol*1.6, singlecol*1.1), sharex='col', sharey='row', dpi=300)
axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it
# make the background color fo this figure #F2CF9A
fig.patch.set_facecolor('#F2F2F2')

gdf_to_plot.plot(ax=axes[0], color='lightgray')
gdf_buildings.plot(ax=axes[0], color='black', zorder=1)
axes[0].set_title('No adaptation', fontsize=12, fontweight='medium', loc='left')

gdf_to_plot.plot(ax=axes[1], color='lightgray')
gdf_buildings.plot(ax=axes[1], color='black', zorder=1)
gdf_outer_wall.plot(ax=axes[1], color='darkred', zorder=10, linewidth=2)
axes[1].set_title('Hold the line', fontsize=12, fontweight='medium', loc='left')

gdf_to_plot.plot(ax=axes[2], color='lightgray')
gdf_buildings.plot(ax=axes[2], color='black', zorder=1)
gdf_inner_wall.plot(ax=axes[2], color='darkred', zorder=2, linewidth=2)
gdf_buyout_informal.plot(ax=axes[2], color='#FBD051', zorder=2, hatch='', alpha=0.6, edgecolor='black')
gdf_elevate_port_2m.plot(ax=axes[2], color='#F97A1F', zorder=2, hatch='', alpha=0.6, edgecolor='black')
axes[2].set_title('Integrated', fontsize=12, fontweight='medium', loc='left')

for ax in axes:
    ax.set_xlim([src.bounds.left+100, src.bounds.right-2000])
    ax.set_ylim([src.bounds.bottom+1300, src.bounds.top-2000])

# Adjust the position of the color bar to not overlap with the subplots
plt.subplots_adjust(wspace=0.091, hspace=0.05, right=0.95)
# Create a common color bar for all subplots
for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
# add a vertical space between the two plots
plt.subplots_adjust(wspace=0.5)
# save figure in D:\paper_4\data\Figures\paper_figures
plt.savefig(r'D:\paper_4\data\Figures\paper_figures\adapt_measures.png', dpi=300, bbox_inches='tight')
plt.show()

import cartopy.crs as ccrs
import cartopy.feature as cfeature
# make plot of Beira region with lat,lon information and satellite image

# Load a GeoDataFrame with the boundaries of the countries
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# Filter the GeoDataFrame to only include Mozambique
mozambique = world[world['name'] == 'Mozambique']
# filter the geodataframe to include the entire Africa
africa = world[world['continent'] == 'Africa']

fig = plt.figure(figsize=(doublecol*1.6, singlecol*1.1), dpi=300)
# Create the first subplot with the PlateCarree projection
ax1 = plt.subplot(1, 2, 1, projection=ccrs.Orthographic(central_longitude=35, central_latitude=-18))
ax1.set_extent([5, 55, -35, 0])
# ax1.add_feature(cfeature.BORDERS)
# ax1.add_feature(cfeature.COASTLINE)
# ax1.add_feature(cfeature.LAND, color='lightgray')

# Plot Mozambique with a different color
africa.plot(ax=ax1, color='lightgray', transform=ccrs.PlateCarree())
mozambique.plot(ax=ax1, color='darkgray', transform=ccrs.PlateCarree(), edgecolor='black', linewidth=0.5)
# add a red dot for Beira
ax1.plot(34.84, -19.84, 'rs', markersize=3, transform=ccrs.PlateCarree())
# now write title for the plot
ax1.set_title('a) View of Mozambique and Beira', fontsize=10, fontweight='medium', loc='left')
# add lat lon on the axes
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False

# Create the second subplot
ax2 = plt.subplot(1, 2, 2)
gdf_to_plot.plot(ax=ax2, color='lightgray')
gdf_buildings.plot(ax=ax2, color='black', zorder=1)
ax2.set_xlim([src.bounds.left+100, src.bounds.right-2000])
ax2.set_ylim([src.bounds.bottom+1300, src.bounds.top-2000])
# remove axis values
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title('b) City of Beira', fontsize=10, fontweight='medium', loc='left')
# save figure in D:\paper_4\data\Figures\paper_figures
plt.savefig(r'D:\paper_4\data\Figures\paper_figures\beira_region.png', dpi=300, bbox_inches='tight')
plt.show()



