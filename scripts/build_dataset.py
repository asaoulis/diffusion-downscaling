"""Script for building training and validation datasets from reanalysis data.

This script implements all the key operations: remapcon, mergetime, etc.
for resampling reanalysis data onto the desired coarse grid, merging
all historical data, and then combining all data sources into a single netCDF file.

We support MSWEP, MSWX, and ERA5 resampling (to CESM2 grid) and merging, as well as 
resampling static high res elevation and land sea mask onto the MSWEP grid. 
"""

import xarray as xr
from pathlib import Path
from datetime import datetime

from diffusion_downscaling.data.constants import COLORADO_MSWEP_COORDS, COLORADO_RIVER_BASIN_COORDS_WIDE
from diffusion_downscaling.data.processing import (
    GridCoordinateManager,
    CDOWrapper,
)
from diffusion_downscaling.data.dataset_building import (
    DatasetBuilder,
    MSWEP_DataType,
    MSWX_DataType,
    ERA5_DataType,
    GeographicData,
)

# First some various paths
# pointing to some prerequisite data files
DATASET_BUILDING_RESOURCES_PATH = Path("/data/climate/Downscaling/diffusion/data/dataset_building")

ELEVATION_MAP_PATH = DATASET_BUILDING_RESOURCES_PATH / "full_world_elevation.nc"
LANDSEAMASK_PATH = DATASET_BUILDING_RESOURCES_PATH / "IMERG_land_sea_mask.nc"

# files containing lat lon grids for each dataset
CESM_lat_lon_grid_path = DATASET_BUILDING_RESOURCES_PATH / "cesm_lat_lon.pkl"
mswep_lat_lon_grid_path = DATASET_BUILDING_RESOURCES_PATH / "mswep_lat_lon.pkl"

# data paths
MSWEP_NETCDF_FILEPATH = "/data/climate/mswep/mswep_v280/daily/"
MSWX_NETCDF_FILEPATH = Path("/data/climate/mswx/mswx_v100/")
ERA5_NETCDF_FILEPATH = Path("/data/climate/reanalysis/era5_daily")



### EDITABLE VARIABLES - CHANGE THESE TO MODIFY THE DATASET

# where to save outputs
target_path = Path("outputs/colorado_wind")

# Parallelising CDO. You will need to tweak this to avoid 
# "Killed" errors due to OOM depending on your system memory.
# my rule of thumb: batch_size * num_workers / 10 = max memory in G
# i.e. 50 * 10 /10 requires > 50 G of memory. 
batch_size = 50
NUM_WORKERS = 10

dataset_datetime_range = (datetime(1985, 1, 1), datetime(2015, 1, 1))

mswep_config = {"precipitation": MSWEP_NETCDF_FILEPATH}

# dict of MSWX variables to include and their paths
mswx_config = {
    "temperature_psl": MSWX_NETCDF_FILEPATH / "T/daily",
    "relative_humidity_psl": MSWX_NETCDF_FILEPATH / "RelHum/daily",
    "pressure_psl": MSWX_NETCDF_FILEPATH / "pres/daily",
}

# dict of ERA5 variables to include and their variable name
era_5_config = (
    ERA5_NETCDF_FILEPATH,
    {
        "temperature_200": "t",
        "temperature_500": "t",
        "temperature_700": "t",
        "temperature_850": "t",
        "specific_humidity_200": "q",
        "specific_humidity_500": "q",
        "specific_humidity_700": "q",
        "specific_humidity_850": "q",
        "vorticity_500": "z",
        "vorticity_700": "z",
        "vorticity_850": "z",
    },
)

### Script proper begins here 

dataset_builder_configs = [
    (MSWEP_DataType, mswep_config),
    (MSWX_DataType, mswx_config),
    (ERA5_DataType, era_5_config),
]


grid_coord_manager = GridCoordinateManager(CESM_lat_lon_grid_path)
latlon_grid_data = grid_coord_manager.compute_latlon_data(COLORADO_MSWEP_COORDS)

mswep_grid_coord_manager = GridCoordinateManager(mswep_lat_lon_grid_path)
mswep_latlon_grid_data = mswep_grid_coord_manager.compute_latlon_data(
    COLORADO_MSWEP_COORDS, grid_overshoot=False
)

geographic_data = GeographicData(
    latlon_grid_data, COLORADO_RIVER_BASIN_COORDS_WIDE, COLORADO_MSWEP_COORDS
)

cdo = CDOWrapper(NUM_WORKERS, batch_size)

dataset_builder = DatasetBuilder(dataset_datetime_range, geographic_data, cdo)

dataset_builder.build_coarsened_files(dataset_builder_configs, target_path)
dataset_builder.cleanup_non_coarse_files(target_path)

coarse_files = list(target_path.rglob("*_coarse.nc"))
filepath_strings = [file.as_posix() for file in coarse_files]

res = cdo.merge_with_wildcard(filepath_strings,  target_path / 'all_vars_coarse_1985_2015_dataset.nc', 'coarse')


final_merge = [
    str(target_path / "mswep_precipitation_coarse.nc"),
    str(target_path / "mswx_temperature_psl_coarse.nc"),
    str(target_path / "all_vars_coarse_1985_2015_dataset.nc"),
]
res = cdo.merge(final_merge, str(target_path /'pT_1985_2015_dataset.nc'))


xs = xr.open_dataset(target_path / "pT_1985_2015_dataset.nc")

cdo.resample_and_combine(
    ELEVATION_MAP_PATH,
    str(target_path / "elevation.nc"),
    mswep_latlon_grid_data,
    COLORADO_MSWEP_COORDS,
)
elevation_xs = xr.open_dataset(str(target_path / "elevation.nc"))

landseamask = xr.open_dataset(LANDSEAMASK_PATH)
interped_landseamask = landseamask.interp(
    lat=xs.lat.values, lon=xs.lon.values + 360, method="nearest"
)  # will only work for colorado...
reindexed_landseamask = xr.Dataset(
    data_vars=dict(
        landseamask=(["lat", "lon"], interped_landseamask.landseamask.values)
    ),
    coords=dict(lon=xs.lon.values, lat=xs.lat.values),
)
reindexed_landseamask.to_netcdf(str(target_path / "reindexed_landseamask.nc"))


reindexed_elevation = xr.Dataset(
    data_vars=dict(
        coarse_elevation=(["lat", "lon"], elevation_xs.coarse_elevation.values[::-1])
    ),
    coords=dict(lon=xs.lon.values, lat=xs.lat.values),
)
reindexed_elevation.to_netcdf(str(target_path / "reindexed_elevation.nc"))
add_elevation = [
    str(target_path / "pT_1985_2015_dataset.nc"),
    str(target_path / "reindexed_elevation.nc"),
    str(target_path / "reindexed_landseamask.nc"),
]
res = cdo.merge(add_elevation, str(target_path / "pT_elevation_1985_2015_dataset.nc"))
