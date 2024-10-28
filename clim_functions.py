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

### FUNCTIONS


def U10m_totalwind(ds):
    if all(elem in ds.keys() for elem in ["u10", "v10"]):
        ds["u10"].attrs["units"] = "m/s"
        ds["v10"].attrs["units"] = "m/s"

        ds["U10"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
        ds["U10"].attrs["standard_name"] = "total_wind_speed"
        ds["U10"].attrs["units"] = "m/s"
    else:
        raise ValueError(
            "To calculate the absolute wind speed, it is required the u and v components"
        )


def calculate_wind_direction_xarray(ds, u10_var="u10", v10_var="v10"):
    """
    Calculate the wind direction from zonal and meridional wind components.
    """
    u10 = ds[u10_var]
    v10 = ds[v10_var]

    wind_dir_rad = np.arctan2(-u10, -v10)  # Negative to convert from "to" to "from"
    wind_dir_deg = np.rad2deg(wind_dir_rad)
    wind_dir_deg = wind_dir_deg.where(
        wind_dir_deg >= 0, wind_dir_deg + 360
    )  # Adjust to 0-360 degrees

    return wind_dir_deg


def decompose_wind(ds, U10_var="U10", wind_dir_var="wind_direction"):
    """
    Decomposes the total wind speed U10 into its zonal and meridional components (u10 and v10).
    """
    U10 = ds[U10_var]
    wind_dir_deg = ds[wind_dir_var]

    # Convert wind direction from degrees to radians
    wind_dir_rad = np.deg2rad(wind_dir_deg)

    # Decompose U10 into u10 and v10
    u10 = -U10 * np.sin(wind_dir_rad)  # Negative because wind blows from this direction
    v10 = -U10 * np.cos(wind_dir_rad)  # Negative because wind blows from this direction

    return u10, v10


def interpolate_dataframe(df, columns_to_keep, time_col, new_freq):
    """
    Interpolates the specified columns of a DataFrame from its current time step to a desired frequency.
    """
    # Subset the DataFrame
    df_subset = df[columns_to_keep]
    # Set the time column as the index
    df_subset.set_index(time_col, inplace=True)
    # Resample to the new frequency and interpolate
    df_interpolated = df_subset.resample(new_freq).asfreq().interpolate(method="time")
    return df_interpolated.to_xarray()


def preprocess_era5_dataset(
    ds, forecast=False, calculate_wind=True, tprate_convert=False
):
    # Rename coordinates
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    # Rename variables
    ds = ds.rename({"tp": "tp", "msl": "msl", "u10": "u10", "v10": "v10"})
    # Convert precipitation from m to mm
    ds["tp"] = ds["tp"] * 1000
    if forecast == True:
        # Calculate the difference in precipitation between time steps
        ds["tp"] = ds["tp"].diff(dim="time", n=1, label="lower")
    # Convert mean sea level pressure from Pa to hPa
    ds["msl"] = ds["msl"] / 100
    # convert tprate to tp
    if tprate_convert == True:
        ds["tprate"] = ds["tprate"] * 3600
        ds["tp"] = ds["tprate"]
    # Adjust longitude coordinates to range from -180 to 180
    ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180
    # Calculate the total wind speed
    if calculate_wind == True:
        U10m_totalwind(ds)

    return ds


def process_ibtracs_storm_data(ibtracs_path, storm_id, ds_time_range):
    """
    Loads IBTrACS data, locates a specific storm, and processes the data for wind speed (U10m) and MSLP.

    Parameters:
    ibtracs_path (str): Path to the IBTrACS dataset.
    storm_id (str): Unique identifier for the storm.
    ds_time_range (xarray.DataArray): Time range from an existing dataset to filter the storm data.

    Returns:
    pandas.DataFrame: DataFrame containing processed storm data with U10m and MSLP.
    """
    # Load IBTRACS data
    ds_ibtracs = xr.open_mfdataset(ibtracs_path)

    # Locate the storm using a provided function `locate_storm`
    ds_storm, df_storm = locate_storm(ds_ibtracs, storm_id)

    # Process wind and pressure data
    df_storm["U10"] = df_storm["usa_wind"] * 0.5144444444444444  # Convert knots to m/s
    df_storm["msl"] = df_storm["usa_pres"]

    # Drop the original columns
    df_storm = df_storm.drop(columns=["usa_wind", "usa_pres"])

    # Filter based on the time range of the dataset 'ds'
    df_storm_subset = df_storm[
        (df_storm["time"] >= ds_time_range.time[0].values)
        & (df_storm["time"] <= ds_time_range.time[-1].values)
    ]

    return df_storm, df_storm_subset


def storm_tracker_mslp_updated(
    ds, df_ws, smooth_step=None, grid_conformity=False, large_box=6, small_box=3
):
    # Calculate the vorticity for all timesteps
    ds_vorticity = mpcalc.vorticity(u=ds["u10"], v=ds["v10"])
    ds["vorticity"] = ds_vorticity

    # Ensure ds and df_ws have the same starting and end date
    start_date = max(ds.time.values[0], df_ws["time"].iloc[0])
    end_date = min(ds.time.values[-1], df_ws["time"].iloc[-1])
    ds = ds.sel(time=slice(start_date, end_date))

    lat_ini = df_ws.loc[df_ws["time"] == start_date, "lat"].values[0]
    lon_ini = df_ws.loc[df_ws["time"] == start_date, "lon"].values[0]

    # 1) Determine eye position with respect to IBTracs historical records
    da_eye_track_ini = ds.sel(
        time=start_date, lat=lat_ini, lon=lon_ini, method="nearest"
    )

    ds_relative_track = []
    for time_step in ds.time.values:
        # Initial dataset at time T
        ds_box = ds.sel(time=time_step)
        # 1) Determine eye position with respect to IBTracs historical records at forecsat time
        if time_step == start_date:
            da_eye_track = da_eye_track_ini
        else:  # use the previous iteartion of the center as a reference
            da_eye_track = ds_box.sel(
                lat=ds_relative_track[-1].index.get_level_values("lat")[0],
                lon=ds_relative_track[-1].index.get_level_values("lon")[0],
                method="nearest",
            )

        # 2) Draw a X degree box around 1) and find maximum vorticity position:
        ds_box5 = ds_box.sel(
            lat=slice(da_eye_track["lat"] + large_box, da_eye_track["lat"] - large_box),
            lon=slice(da_eye_track["lon"] - large_box, da_eye_track["lon"] + large_box),
        )
        # if latitude is positive, we look for maximum vorticity, but if it's negative, we look for minimum vorticity
        if da_eye_track["lat"] > 0:
            coords_max_vort = (
                ds_box5.where(
                    ds_box5["vorticity"] == ds_box5["vorticity"].max(dim=["lat", "lon"])
                )
                .to_dataframe()["vorticity"]
                .dropna()
                .reset_index(["lat", "lon"])
            )
        else:
            coords_max_vort = (
                ds_box5.where(
                    ds_box5["vorticity"] == ds_box5["vorticity"].min(dim=["lat", "lon"])
                )
                .to_dataframe()["vorticity"]
                .dropna()
                .reset_index(["lat", "lon"])
            )

        # 2-a) update values if vorticity position indicates lower pressure:
        if (
            ds_box5["msl"]
            .sel(
                lat=coords_max_vort["lat"].values,
                lon=coords_max_vort["lon"].values,
                method="nearest",
            )
            .values
            < da_eye_track["msl"].values
        ):
            print("Step 2: update location minimum pressure for vorticity maximum")
            da_eye_track = ds_box5.sel(
                lat=coords_max_vort["lat"].values,
                lon=coords_max_vort["lon"].values,
                method="nearest",
            )

        # 3) Establish minimum pressure within X degrees from eye of the storm:
        ds_box_small = ds_box.sel(
            lat=slice(
                da_eye_track["lat"].values.item() + small_box,
                da_eye_track["lat"].values.item() - small_box,
            ),
            lon=slice(
                da_eye_track["lon"].values.item() - small_box,
                da_eye_track["lon"].values.item() + small_box,
            ),
        )
        coords_minimum_small = (
            ds_box_small.where(
                ds_box_small["msl"] == ds_box_small["msl"].min(dim=["lat", "lon"])
            )
            .to_dataframe()["msl"]
            .dropna()
            .reset_index(["lat", "lon"])
        )
        coords_minimum_small = coords_minimum_small.mean(numeric_only=True).to_frame().T
        # 3-a) update values if new position indicates lower pressure:
        if (
            ds_box_small["msl"]
            .sel(
                lat=coords_minimum_small["lat"].values,
                lon=coords_minimum_small["lon"].values,
                method="nearest",
            )
            .values
            <= da_eye_track["msl"].values
        ):
            # print('Step 3: update location minimum pressure for minimum MSLP')
            da_eye_track = ds_box_small.sel(
                lat=coords_minimum_small["lat"].values,
                lon=coords_minimum_small["lon"].values,
                method="nearest",
            )

        ds_relative_track.append(
            da_eye_track.to_dataframe()
        )  # check out the to_dataframe. Keep it as dataset. Concatenate
    df_relative_track = pd.concat(ds_relative_track).reset_index(["lat", "lon"])
    df_relative_track = df_relative_track.set_index("time")

    # FILTERING THE TRACKS TO MAKE THEM SMOOTHER
    if smooth_step == "savgol":
        print(f"smooth: {smooth_step}")
        # define window length and polynomial order for Savitzky-Golay filter, or use window_length to 9 because that's the minimum for one day
        window_length = max(int(np.ceil(len(df_relative_track) / 10) // 2 * 2 + 1), 9)
        poly_order = 2
        # smooth lat and lon coordinates separately using Savitzky-Golay filter
        df_relative_track["lat"] = savgol_filter(
            df_relative_track["lat"], window_length, poly_order
        )
        df_relative_track["lon"] = savgol_filter(
            df_relative_track["lon"], window_length, poly_order
        )

    elif smooth_step == "bspline":
        print(f"smooth: {smooth_step}")  # define degree and knots for B-spline
        k = 3
        t = np.linspace(
            0, 1, len(df_relative_track)
        )  # construct B-spline for lat and lon coordinates separately
        spl_lat = BSpline(t, df_relative_track["lat"], k)
        spl_lon = BSpline(
            t, df_relative_track["lon"], k
        )  # evaluate B-spline at original points
        df_relative_track["lat"] = spl_lat(t)
        df_relative_track["lon"] = spl_lon(t)

    elif (smooth_step != None) and (smooth_step != 0):
        print(f"smooth: {smooth_step}")
        df_relative_track = df_relative_track.rolling(
            window=smooth_step, closed="both", min_periods=1
        ).mean()

    if grid_conformity == True:
        for i, row in df_relative_track.iterrows():
            # select the nearest grid cell in the dataset to the (lat, lon) pair
            nearest_latlon = ds.sel(lat=row["lat"], lon=row["lon"], method="nearest")
            # store the closest lat, lon values in the dataframe
            df_relative_track.at[i, "lat"] = nearest_latlon["lat"].values
            df_relative_track.at[i, "lon"] = nearest_latlon["lon"].values

    return df_relative_track


def storm_tracker_mslp_ens(ds, df_ws, large_box, small_box, smooth_step=None):
    storm_tracker_list = []
    for numbers in ds.number:
        print(f"member: {numbers.item()}")
        storm_tracker_list.append(
            storm_tracker_mslp_updated(
                ds.sel(number=numbers),
                df_ws,
                smooth_step=smooth_step,
                large_box=large_box,
                small_box=small_box,
            )
        )
    return pd.concat(storm_tracker_list)


def locate_storm(ds, storm_id):
    # use IBTracs to locate storms based on their SID code.
    storm_ids = ["".join(name.astype(str)) for name in ds.variables["sid"].values]
    # create a dictionary with the saffit-simpson hurricane scale
    wind_speed_scale = {
        -5: "Unknown",
        -4: "Post-tropical",
        -3: "Miscellaneous disturbances",
        -2: "Subtropical",
        -1: "Tropical depression",
        0: "Tropical storm",
        1: "Category 1",
        2: "Category 2",
        3: "Category 3",
        4: "Category 4",
        5: "Category 5",
    }

    sel_tracks = []
    # filter name
    if storm_id:
        if not isinstance(storm_id, list):
            storm_id = [storm_id]
        for storm in storm_id:
            sel_tracks.append(storm_ids.index(storm))
        sel_tracks = np.array(sel_tracks)
        ds_storm = ds.sel(storm=sel_tracks)
        print(ds_storm.name.values)
        df_wind_speed = (
            ds_storm[["usa_wind", "usa_pres", "storm_speed", "usa_status", "usa_sshs"]]
            .to_dataframe()
            .dropna()
        )
        df_wind_speed["usa_status"] = df_wind_speed["usa_status"].apply(
            lambda x: x.decode()
        )
        df_wind_speed["usa_sshs"] = df_wind_speed["usa_sshs"].map(wind_speed_scale)
        df_wind_speed["time"] = pd.to_datetime(df_wind_speed["time"]).round("1s")
    return ds_storm, df_wind_speed


def storm_composite(ds, df_center, radius=3):
    """This function has as objective to create a storm composite.
    Composites are a reference frame centred on the tropical cyclone eye.
    at each time step, select data around the eye center and buffer around X degrees, translating all corrdinates to [0,0] at the eye.
    """
    ds = ds.sel(time=slice(df_center.index[0], df_center.index[-1]))
    ds_relative_track = []
    for time_step in ds.time.values:
        # initial time step
        ds_ini = ds.sel(
            time=time_step,
            lat=df_center["lat"].loc[time_step],
            lon=df_center["lon"].loc[time_step],
            method="nearest",
        )
        # Find radius around the centre
        ds_test = ds.sel(
            time=time_step,
            lat=slice(ds_ini["lat"] + radius, ds_ini["lat"] - radius),
            lon=slice(ds_ini["lon"] - radius, ds_ini["lon"] + radius),
        )
        # Normalise it so centre becomes (0,0)
        ds_test["lat"] = (ds_test["lat"] - ds_ini["lat"]).round(3)
        ds_test["lon"] = (ds_test["lon"] - ds_ini["lon"]).round(3)
        # organise data
        ds_test = ds_test.sortby(ds_test.lat)
        ds_test = ds_test.sortby(ds_test.lon)
        ds_relative_track.append(ds_test)

    return xr.concat(ds_relative_track, dim="time")


def storm_composite_ens(ds, df_center=None, radius=3):
    # Same as storm_composite but for data with multiple ensemble members (check if numbers are in string format)
    if not isinstance(df_center["number"][0], int):
        df_center["number"] = df_center["number"].astype(int)
    list_storm_composite = []
    for numbers in ds.number.values:  # convert numbers to a numpy array
        list_storm_composite.append(
            storm_composite(
                ds.sel(number=numbers),
                df_center=df_center[["lat", "lon", "number"]].loc[
                    df_center["number"] == numbers.item()
                ],
                radius=radius,
            )
        )  # convert numbers to a native Python int
    return xr.concat(list_storm_composite, dim="number")


def plot_minimum_track(df, df2=None):
    fig = plt.figure(figsize=(6, 8))
    central_lon = df.lon.min() + (df.lon.max() - df.lon.min()) / 2
    central_lat = df.lat.min() + (df.lat.max() - df.lat.min()) / 2
    ax = plt.axes(
        projection=ccrs.LambertAzimuthalEqualArea(
            central_longitude=central_lon, central_latitude=central_lat
        )
    )
    ax.coastlines(resolution="10m")
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    # Plot the first DataFrame
    if "number" in df.columns:
        sns.lineplot(
            data=df,
            x="lon",
            y="lat",
            hue="number",
            palette="tab10",
            linewidth=1.5,
            estimator=None,
            ax=ax,
            transform=ccrs.PlateCarree(),
        )
        # hide legend
        ax.get_legend().remove()
    else:
        sns.lineplot(
            data=df,
            x="lon",
            y="lat",
            linewidth=1.5,
            estimator=None,
            ax=ax,
            transform=ccrs.PlateCarree(),
        )

    # Plot the second DataFrame if provided
    if df2 is not None:
        if "number" in df2.columns:
            sns.lineplot(
                data=df2,
                x="lon",
                y="lat",
                hue="number",
                palette="tab10",
                linewidth=1.5,
                estimator=None,
                ax=ax,
                transform=ccrs.PlateCarree(),
                linestyle="--",
            )  # Using dashed line for the second track
        else:
            sns.lineplot(
                data=df2,
                x="lon",
                y="lat",
                linewidth=1.5,
                estimator=None,
                ax=ax,
                transform=ccrs.PlateCarree(),
                linestyle="--",
            )

    plt.tight_layout()
    plt.show()
    return fig


def plot_cumulative_precipitation_timeseries(ds, lat, lon, ds_era5=None):
    # Select data for the given latitude and longitude
    ds_selected = ds.sel(lat=lat, lon=lon, method="nearest")
    # Convert the selected data to a DataFrame
    df = ds_selected.to_dataframe().reset_index()
    # Calculate the cumulative sum of precipitation
    df["Ptot_cumulative"] = df.groupby("number")["tp"].cumsum()
    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot the cumulative time series of precipitation for each number
    sns.lineplot(
        data=df,
        x="time",
        y="Ptot_cumulative",
        hue="number",
        palette="tab10",
        linewidth=1.5,
        ax=ax,
        legend=False,
    )

    # If ds_era5 is provided, plot its data
    if ds_era5 is not None:
        # Select data for the given latitude and longitude from ds_era5
        ds_era5_selected = ds_era5.sel(lat=lat, lon=lon, method="nearest")
        # Convert the selected data to a DataFrame
        df_era5 = ds_era5_selected.to_dataframe().reset_index()
        # Calculate the cumulative sum of precipitation
        df_era5["Ptot_cumulative"] = df_era5["tp"].cumsum()
        # Plot the cumulative time series of precipitation for ds_era5
        ax.plot(
            df_era5["time"],
            df_era5["Ptot_cumulative"],
            color="black",
            linestyle="dashed",
            linewidth=2.5,
            label="ERA5",
        )
        # add label for era5
        ax.legend()

    # Set the title and labels
    ax.set_title(
        "Cumulative Precipitation Time Series at location ({}, {})".format(lat, lon)
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Precipitation")
    # Show the plot
    plt.show()


def plot_var_timeseries(ds, lat, lon, variable, ds_era5=None):
    # Select data for the given latitude and longitude
    ds_selected = ds.sel(lat=lat, lon=lon, method="nearest")

    # Calculate the wind speed
    if variable == "u10":
        ds_selected["u10"] = np.sqrt(ds_selected["u10"] ** 2 + ds_selected["v10"] ** 2)

    # Convert the selected data to a DataFrame
    df = ds_selected.to_dataframe().reset_index()

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the time series of maximum wind speed for each number
    sns.lineplot(
        data=df,
        x="time",
        y=variable,
        hue="number",
        palette="tab10",
        linewidth=1.5,
        ax=ax,
    )

    # If ds_era5 is provided, plot its data
    if ds_era5 is not None:
        # Select data for the given latitude and longitude from ds_era5
        ds_era5_selected = ds_era5.sel(lat=lat, lon=lon, method="nearest")

        # Calculate the wind speed
        if variable == "u10":
            ds_era5_selected["u10"] = np.sqrt(
                ds_era5_selected["u10"] ** 2 + ds_era5_selected["v10"] ** 2
            )

        # Convert the selected data to a DataFrame
        df_era5 = ds_era5_selected.to_dataframe().reset_index()

        # Plot the time series of maximum wind speed for ds_era5
        ax.plot(
            df_era5["time"],
            df_era5[variable],
            color="black",
            linestyle="dashed",
            linewidth=2.5,
        )

    # Set the title and labels
    ax.set_title(f"{variable} at location ({lat}, {lon})")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{variable}")

    # Show the plot
    plt.show()


def adjust_time(ds):

    ds["tp"] = ds["tp"].diff(dim="time", n=1, label="lower")

    # Select the first 5 timesteps
    ds = ds.isel(time=slice(0, 6))

    return ds


def plot_comp_variable_timeseries(
    ds_composite, variable, ds_era5_composite=None, agg_method="mean", obs_data=None
):
    # Aggregate the composite data
    if agg_method == "mean":
        ds_agg = ds_composite.mean(dim=["lat", "lon"])
    elif agg_method == "max":
        ds_agg = ds_composite.max(dim=["lat", "lon"])
    elif agg_method == "min":
        ds_agg = ds_composite.min(dim=["lat", "lon"])
    else:
        raise ValueError("Invalid aggregation method")

    # Convert the aggregated data to a DataFrame
    df = ds_agg.to_dataframe().reset_index()

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the time series of the variable for each number
    sns.lineplot(data=df, x="time", y=variable, hue="number", linewidth=1.5, ax=ax)

    # If ds_era5_composite is provided, aggregate and plot its data
    if ds_era5_composite is not None:
        # Aggregate the composite data
        if agg_method == "mean":
            ds_era5_agg = ds_era5_composite.mean(dim=["lat", "lon"])
        elif agg_method == "max":
            ds_era5_agg = ds_era5_composite.max(dim=["lat", "lon"])
        elif agg_method == "min":
            ds_era5_agg = ds_era5_composite.min(dim=["lat", "lon"])

        # Convert the aggregated data to a DataFrame
        df_era5 = ds_era5_agg.to_dataframe().reset_index()

        # Plot the time series of the variable for ds_era5
        ax.plot(
            df_era5["time"],
            df_era5[variable],
            color="black",
            linestyle="dashed",
            linewidth=2.5,
            label="ERA5",
        )

    # if obs_data is provided, plot it (but make sure it is compatible with the other data)
    if obs_data is not None:
        # select time for obs_data as the same as the ds_composite
        obs_data = obs_data.loc[obs_data["time"].isin(df["time"])]
        ax.plot(
            obs_data["time"],
            obs_data[variable],
            color="red",
            linestyle="dotted",
            linewidth=4,
            label="Observed",
            zorder=10,
        )

    # Set the title and labels
    ax.set_title("{} Time Series at Storm Center ({})".format(variable, agg_method))
    ax.set_xlabel("Time")
    ax.set_ylabel("{} ({})".format(variable, agg_method))
    # set the legend
    ax.legend()
    # Show the plot
    plt.show()


####
# make the rest of the script if main
if __name__ == "__main__":

    # load data
    file_path = "D:/paper_4/data/seas5/ecmwf/ecmwf_eps_pf_010_vars_n50_s90_20190313.nc"  # ecmwf_eps_pf_vars_n15_s90_20190313 / ecmwf_eps_pf_vars_n15_s90_20170907
    start_time = "2019-03-14T00:00:00.000000000"
    end_time = "2019-03-15T12:00:00.000000000"
    ds = xr.open_dataset(file_path).sel(time=slice(start_time, end_time))
    ds = preprocess_era5_dataset(ds, forecast=True)
    # load ERA5 data
    ds_era5 = xr.open_dataset(
        r"D:\paper_4\data\era5\era5_hourly_vars_idai_single_2019_03.nc"
    ).sel(time=slice(start_time, end_time))
    ds_era5 = preprocess_era5_dataset(ds_era5)
    # load control run
    ds_control = xr.open_dataset(
        r"D:\paper_4\data\seas5\ecmwf\ecmwf_eps_cf_010_vars_s90_20190313_00.nc"
    ).sel(time=slice(start_time, end_time))
    ds_control = preprocess_era5_dataset(ds_control, forecast=True)
    # Load the NetCDF files into a combined dataset
    ds_ips_rebuild = xr.open_mfdataset(
        "D:\paper_4\data\seas5\ecmwf\ecmwf_eps_cf_010_vars_s7_201903*.nc",
        combine="nested",
        concat_dim="time",
        preprocess=adjust_time,
    ).sel(time=slice(start_time, end_time))
    ds_ips_rebuild = preprocess_era5_dataset(ds_ips_rebuild)

    ds_gpm = xr.open_dataset(r"D:\paper_4\data\nasa_data\gpm_imerg_201903.nc").sel(
        time=slice(start_time, end_time)
    )
    ds_gpm = ds_gpm.rename({"precipitation": "tp"})

    ds_gfs = xr.open_mfdataset(r"D:\paper_4\data\gfs\v2\gfs.0p25.201903*.nc").sel(
        time=slice(start_time, end_time)
    )
    # rename PRATE_l1_avg_1 to Ptot  U_GRD_L103   V_GRD_L103   PRMSL_L101
    ds_gfs = ds_gfs.rename(
        {"U_GRD_L103": "u10", "V_GRD_L103": "v10", "PRMSL_L101": "msl"}
    )
    U10m_totalwind(ds_gfs)

    # load IBTRACS data
    ds_ibtracs = xr.open_mfdataset(
        r"D:\paper_4\data\ibtracs\IBTrACS.since1980.v04r00.nc"
    )
    ds_storm, df_storm = locate_storm(
        ds_ibtracs, "2019063S18038"
    )  # idai: '2019063S18038'
    # convert and rename usa_wind to U10m and usa_pres to mslp
    df_storm["U10"] = df_storm["usa_wind"]
    df_storm["msl"] = df_storm["usa_pres"]
    df_storm = df_storm.drop(columns=["usa_wind", "usa_pres"])
    df_storm["U10"] = df_storm["U10"] * 0.5144444444444444
    df_storm_subset = df_storm[
        (df_storm["time"] >= ds.time[0].values)
        & (df_storm["time"] <= ds.time[-1].values)
    ]

    df_storm2, df_storm_subset2 = process_ibtracs_storm_data(
        ibtracs_path=r"D:\paper_4\data\ibtracs\IBTrACS.since1980.v04r00.nc",
        storm_id="2019063S18038",
        ds_time_range=ds,
    )

    # plot ds_era5 precipitation at time step 0 on a map and add coastlines
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())  #
    ax.coastlines(resolution="10m")
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ds_ips_rebuild["tp"].isel(time=35).plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        x="lon",
        y="lat",
        cmap="Blues",
        robust=True,
        add_colorbar=True,
    )
    # add lat lon coordinates on axis
    gl = ax.gridlines(draw_labels=True, linestyle=":", color="black", alpha=0.5)
    plt.show()

    # track storm
    storm_track_ens = storm_tracker_mslp_ens(
        ds, df_storm, smooth_step="savgol", large_box=3, small_box=1.5
    )
    # track ds_era5 storm
    storm_track_era5 = storm_tracker_mslp_updated(
        ds_era5, df_storm, smooth_step="savgol", large_box=3, small_box=1.5
    )
    # track control run
    storm_track_control = storm_tracker_mslp_updated(
        ds_control, df_storm, smooth_step="savgol", large_box=3, small_box=1.5
    )
    # track test
    storm_track_rebuild = storm_tracker_mslp_updated(
        ds_ips_rebuild, df_storm, smooth_step="savgol", large_box=3, small_box=1.5
    )
    # track gfs
    storm_track_gfs = storm_tracker_mslp_updated(
        ds_gfs, df_storm, smooth_step="savgol", large_box=3, small_box=1.5
    )

    plot_minimum_track(storm_track_ens)
    plot_minimum_track(storm_track_era5)
    plot_minimum_track(storm_track_control)
    plot_minimum_track(storm_track_control, df_storm_subset)
    plot_minimum_track(storm_track_rebuild, df_storm_subset)
    plot_minimum_track(storm_track_gfs, df_storm_subset)

    # Create a figure and a map projection
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # Plot the data
    ax.scatter(df_storm["lon"], df_storm["lat"], color="blue")
    ax.scatter(storm_track_era5["lon"], storm_track_era5["lat"], color="red")
    ax.scatter(storm_track_rebuild["lon"], storm_track_rebuild["lat"], color="green")
    # Add coastlines
    ax.coastlines()
    # Show the plot
    plt.show()

    # latitude and longitude of Beira, Mozambique
    plot_cumulative_precipitation_timeseries(ds, -19.8436, 34.8389, ds_ips_rebuild)
    plot_var_timeseries(ds, -19.8436, 34.8389, "msl", ds_ips_rebuild)
    plot_var_timeseries(ds, -19.8436, 34.8389, "u10", ds_ips_rebuild)
    plot_var_timeseries(ds, -19.8436, 34.8389, "tp", ds_era5)

    ds_gpm["tp"].sel(lat=-19.8436, lon=34.8389, method="nearest").plot(
        label="GPM", linestyle="dashed", linewidth=3.5
    )
    ds_era5["tp"].sel(lat=-19.8436, lon=34.8389, method="nearest").plot(label="era5")
    ds_ips_rebuild["tp"].sel(lat=-19.8436, lon=34.8389, method="nearest").plot(
        label="ips_rebuild"
    )
    ds_control["tp"].sel(lat=-19.8436, lon=34.8389, method="nearest").plot(
        label="control"
    )
    plt.legend()
    plt.show()

    # Calculate composite.
    ds_composite_ens = storm_composite_ens(ds, storm_track_ens, radius=2)
    ds_era5_composite = storm_composite(ds_era5, storm_track_era5, radius=2)
    ds_control_composite = storm_composite(ds_control, storm_track_control, radius=2)
    ds_ips_rebuild_composite = storm_composite(
        ds_ips_rebuild, storm_track_rebuild, radius=2
    )
    ds_gfs_composite = storm_composite(ds_gfs, storm_track_gfs, radius=2)

    plot_comp_variable_timeseries(
        ds_composite_ens,
        "tp",
        ds_era5_composite=ds_control_composite,
        agg_method="mean",
    )
    plot_comp_variable_timeseries(
        ds_composite_ens,
        "msl",
        ds_era5_composite=ds_control_composite,
        agg_method="min",
        obs_data=df_storm,
    )
    plot_comp_variable_timeseries(
        ds_composite_ens,
        "u10",
        ds_era5_composite=ds_ips_rebuild_composite,
        agg_method="max",
        obs_data=df_storm,
    )

    orig_u10m = ds_composite_ens["u10"].max(["lat", "lon"]).mean()
    orig_u10m_era5 = ds_era5_composite["u10"].max(["lat", "lon"]).mean()
    ref_u10m = df_storm[
        df_storm["time"].between(
            ds_composite_ens.time.min().values, ds_composite_ens.time.max().values
        )
    ]["u10"].mean()

    bc_u10m = ref_u10m / orig_u10m
    bc_u10m_era5 = ref_u10m / orig_u10m_era5
    # caclulate the bias correction factor for wind speed at every time step
    orig_u10m_time = ds_composite_ens["u10"].max(["lat", "lon"])
    orig_u10m_era5_time = ds_era5_composite["u10"].max(["lat", "lon"])
    ref_u10m_time = df_storm[
        df_storm["time"].between(
            ds_composite_ens.time.min().values, ds_composite_ens.time.max().values
        )
    ]["u10"]
    bc_u10m_time = ref_u10m_time / orig_u10m_time

    # compare the bias between ds_gpm and ds_era5['tp']
    orig_precip = ds_gpm["tp"].sel(lat=-19.8436, lon=34.8389, method="nearest").mean()
    orig_precip_era5 = (
        ds_era5["tp"].sel(lat=-19.8436, lon=34.8389, method="nearest").mean()
    )
    bc_precip = orig_precip / orig_precip_era5

    ds_composite_ens_adj = ds_composite_ens.copy()
    ds_composite_ens_adj["u10"] = ds_composite_ens["u10"] * bc_u10m

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot the time series of the variable for each number
    sns.lineplot(
        data=ds_composite_ens.max(dim=["lat", "lon"]).to_dataframe().reset_index(),
        x="time",
        y="u10",
        hue="number",
        linewidth=1.5,
        ax=ax,
    )
    sns.lineplot(
        data=ds_ips_rebuild_composite.max(dim=["lat", "lon"])
        .to_dataframe()
        .reset_index(),
        x="time",
        y="u10",
        linewidth=3.5,
        ax=ax,
        label="control",
        color="red",
    )
    sns.lineplot(
        data=df_storm_subset,
        x="time",
        y="u10",
        linewidth=3.5,
        ax=ax,
        label="observed",
        color="black",
        linestyle="dashed",
    )

    plt.show()
