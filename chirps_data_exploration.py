
import rioxarray
import glob
from datetime import datetime

# Initialize a list to store the data arrays
data_arrays = []

# Go through each TIFF file
for file in glob.glob(r'D:\paper_4\data\chirps\*.tif'):
    # Extract the date from the file name
    time_str = file[-14:-4]
    time = datetime.strptime(time_str, '%Y.%m.%d')

    # Open the raster file
    raster = rioxarray.open_rasterio(file)

    # Since each file represents a single time point, we use 'expand_dims' to add a time dimension
    raster = raster.expand_dims(time=[time])

    # Append this raster layer to our list
    data_arrays.append(raster)

# Combine all the data arrays into a single dataset
combined = xr.concat(data_arrays, dim='time')
# drop the band dimension and coordinate
combined = combined.squeeze('band')
combined = combined.drop('band')
# correct -9999 values to nan
combined = combined.where(combined != -9999)
# rename x,y to lon,lat
combined = combined.rename({'x': 'lon', 'y': 'lat'})
# Now you can export this combined dataset to NetCDF
# set name of variable as 'tp'
combined = combined.rename('tp')
combined.to_netcdf(r'D:\paper_4\data\chirps\chirps_2019_03_daily.nc')

# now load all .nc files in D:\paper_4\data\chirps
ds_chirps = xr.open_mfdataset(r'D:\paper_4\data\chirps\chirps_2019_03_daily.nc').sel(time=slice(start_time,end_time))
