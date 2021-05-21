"""
Author: Riley Hales
Description: Resamples GLDAS data
Dependencies:
    numpy
    netcdf4
    rch (pip)
"""
import datetime
import glob
import logging
import os
import sys

import netCDF4 as nc
import numpy as np
import xarray as xr
import rch


def resample(open_path: str, save_dir: str, factor: int, prefix: str):
    start_time = datetime.datetime.utcnow()
    logging.info(f'Working on file: {open_path}')
    logging.info(f'-- started at: {start_time.strftime("%Y-%m-%d %X")}')

    save_path = os.path.join(save_dir, f'{prefix}_{os.path.basename(open_path)}')

    # open the file to be copied
    original = nc.Dataset(open_path, 'r', clobber=False, diskless=True)
    duplicate = nc.Dataset(save_path, 'w', clobber=True, format='NETCDF4', diskless=False)

    # copy the global netcdf attributes
    duplicate.setncatts(original.__dict__)

    # specify dimensions from what we copied before
    for dimension in original.dimensions:
        size = original.dimensions[dimension].size
        if dimension in ('lat', 'lon',):
            size = size / factor
        duplicate.createDimension(dimension, size)

    # handle lat, lon, and time separately because they have 1d arrays
    duplicate.createVariable(varname='lat', datatype='f4', dimensions='lat', zlib=True, shuffle=True)
    duplicate.createVariable(varname='lon', datatype='f4', dimensions='lon', zlib=True, shuffle=True)
    duplicate.createVariable(varname='time', datatype='i', dimensions='time', zlib=True, shuffle=True)
    duplicate['lat'][:] = rch.arrays.resample_1d(np.array(original['lat'][:]), factor, 'mean')
    duplicate['lon'][:] = rch.arrays.resample_1d(np.array(original['lon'][:]), factor, 'mean')
    duplicate['time'][:] = original['time'][:]
    for variable in ('time', 'lat', 'lon',):
        for attr in original[variable].__dict__:
            if attr != "_FillValue":
                duplicate[variable].setncattr(attr, original[variable].__dict__[attr])

    # copy the rest of the variables
    for variable in original.variables:
        if variable in ('time', 'lat', 'lon', 'time_bnds'):
            continue
        # create the variable
        duplicate.createVariable(varname=variable,
                                 datatype='f4',
                                 dimensions=original[variable].dimensions,
                                 zlib=True,
                                 shuffle=True,
                                 fill_value=original[variable].__dict__.get('_FillValue', None))
        # set the attributes for lat and lon (except fill value, you just can't copy it)
        for attr in original[variable].__dict__:
            if attr != "_FillValue":
                duplicate[variable].setncattr(attr, original[variable].__dict__[attr])
        # copy the array and resample
        arr = np.array(original[variable][:])
        arr[arr < -9000] = np.nan
        arr = np.squeeze(arr)
        duplicate[variable][:] = rch.arrays.resample_2d(arr, factor, factor, 'median')

    # close the netcdfs
    original.close()
    duplicate.close()

    end_time = datetime.datetime.utcnow()
    logging.info(f'-- finished at: {end_time.strftime("%Y-%m-%d %X")}')
    logging.info(f'-- elapsed time: {(end_time - start_time).total_seconds() / 60} minutes')
    return


def resample_and_clip(open_path: str, save_dir: str, factor: int, prefix: str, clip: bool):
    start_time = datetime.datetime.utcnow()
    logging.info(f'Working on file: {open_path}')
    logging.info(f'-- started at: {start_time.strftime("%Y-%m-%d %X")}')

    save_path = os.path.join(save_dir, f'{prefix}_{os.path.basename(open_path)}')

    # open the file to be copied
    original = nc.Dataset(open_path, 'r', clobber=False, diskless=True)
    duplicate = nc.Dataset(save_path, 'w', clobber=True, format='NETCDF4', diskless=False)

    # copy the global netcdf attributes
    duplicate.setncatts(original.__dict__)

    # specify dimensions from what we copied before
    for dimension in original.dimensions:
        if dimension == 'lat':
            size = 55
        elif dimension == 'lon':
            size = 144
        else:
            size = original.dimensions[dimension].size
        duplicate.createDimension(dimension, size)

    # handle lat, lon, and time separately because they have 1d arrays
    duplicate.createVariable(varname='lat', datatype='f4', dimensions='lat', zlib=True, shuffle=True)
    duplicate.createVariable(varname='lon', datatype='f4', dimensions='lon', zlib=True, shuffle=True)
    duplicate.createVariable(varname='time', datatype='i', dimensions='time', zlib=True, shuffle=True)
    duplicate['lat'][:] = rch.arrays.resample_1d(np.array(original['lat'][:]), factor, 'mean')[0:55]
    duplicate['lon'][:] = rch.arrays.resample_1d(np.array(original['lon'][:]), factor, 'mean')
    duplicate['time'][:] = original['time'][:]
    for variable in ('time', 'lat', 'lon',):
        for attr in original[variable].__dict__:
            if attr != "_FillValue":
                duplicate[variable].setncattr(attr, original[variable].__dict__[attr])

    # copy the rest of the variables
    for variable in original.variables:
        if variable in ('time', 'lat', 'lon', 'time_bnds'):
            continue
        # create the variable
        duplicate.createVariable(varname=variable,
                                 datatype='f4',
                                 dimensions=original[variable].dimensions,
                                 zlib=True,
                                 shuffle=True,
                                 fill_value=original[variable].__dict__.get('_FillValue', None))
        # set the attributes for lat and lon (except fill value, you just can't copy it)
        for attr in original[variable].__dict__:
            if attr != "_FillValue":
                duplicate[variable].setncattr(attr, original[variable].__dict__[attr])
        # copy the array and resample
        arr = np.array(original[variable][:])
        arr[arr < -9000] = np.nan
        arr = np.squeeze(arr)
        arr = rch.arrays.resample_2d(arr, factor, factor, 'median')
        if clip:
            arr = arr[0:55, :]
        duplicate[variable][:] = arr

    # close the netcdfs
    original.close()
    duplicate.close()

    end_time = datetime.datetime.utcnow()
    logging.info(f'-- finished at: {end_time.strftime("%Y-%m-%d %X")}')
    logging.info(f'-- elapsed time: {(end_time - start_time).total_seconds() / 60} minutes')
    return


def clip_latlon(open_path: str, save_dir: str, prefix: str,
                min_lat: float, max_lat: float, min_lon: float, max_lon: float):
    start_time = datetime.datetime.utcnow()
    logging.info(f'Working on file: {open_path}')
    logging.info(f'-- started at: {start_time.strftime("%Y-%m-%d %X")}')

    save_path = os.path.join(save_dir, f'{prefix}_{os.path.basename(open_path)}')

    # open the file to be copied
    original = nc.Dataset(open_path, 'r', clobber=False, diskless=True)
    duplicate = nc.Dataset(save_path, 'w', clobber=True, format='NETCDF4', diskless=False)

    lats = np.array(original['lat'][:])
    below_mins = lats[lats < min_lat]
    if below_mins.shape[0] == 0:
        minlatidx = 0
    else:
        minlatidx = lats.tolist().index(np.max(below_mins))
    above_maxs = lats[lats > max_lat]
    if above_maxs.shape[0] == 0:
        maxlatidx = lats.shape[0]
    else:
        maxlatidx = lats.tolist().index(np.min(above_maxs))

    lons = np.array(original['lon'][:])
    below_mins = lons[lons < min_lon]
    if below_mins.shape[0] == 0:
        minlonidx = 0
    else:
        minlonidx = lats.tolist().index(np.max(below_mins))
    above_maxs = lons[lons > max_lon]
    if above_maxs.shape[0] == 0:
        maxlonidx = lats.shape[0]
    else:
        maxlonidx = lats.tolist().index(np.min(above_maxs))
    lats = lats[lats < max_lat]
    lats = lats[lats > min_lat]
    new_lats_size = lats.shape[0]
    lons = lons[lons < max_lon]
    lons = lons[lons > min_lon]
    new_lons_size = lons.shape[0]

    # copy the global netcdf attributes
    duplicate.setncatts(original.__dict__)

    # specify dimensions from what we copied before
    for dimension in original.dimensions:
        if dimension == 'lat':
            size = new_lats_size
        elif dimension == 'lon':
            size = new_lons_size
        else:
            size = original.dimensions[dimension].size
        duplicate.createDimension(dimension, size)

    # handle lat, lon, and time separately because they have 1d arrays
    duplicate.createVariable(varname='lat', datatype='f4', dimensions='lat', zlib=True, shuffle=True)
    duplicate.createVariable(varname='lon', datatype='f4', dimensions='lon', zlib=True, shuffle=True)
    duplicate.createVariable(varname='time', datatype='i', dimensions='time', zlib=True, shuffle=True)
    duplicate['lat'][:] = lons
    duplicate['lon'][:] = lats
    duplicate['time'][:] = original['time'][:]
    for variable in ('time', 'lat', 'lon',):
        for attr in original[variable].__dict__:
            if attr != "_FillValue":
                duplicate[variable].setncattr(attr, original[variable].__dict__[attr])

    # copy the rest of the variables
    for variable in original.variables:
        if variable in ('time', 'lat', 'lon', 'time_bnds'):
            continue
        # create the variable
        duplicate.createVariable(varname=variable,
                                 datatype='f4',
                                 dimensions=original[variable].dimensions,
                                 zlib=True,
                                 shuffle=True,
                                 fill_value=original[variable].__dict__.get('_FillValue', None))
        # set the attributes for lat and lon (except fill value, you just can't copy it)
        for attr in original[variable].__dict__:
            if attr != "_FillValue":
                duplicate[variable].setncattr(attr, original[variable].__dict__[attr])
        # copy the array and resample
        arr = np.array(original[variable][:])
        arr[arr < -9000] = np.nan
        arr = np.squeeze(arr)
        duplicate[variable][:] = arr[minlatidx:maxlatidx, minlonidx:maxlonidx]

    # close the netcdfs
    original.close()
    duplicate.close()

    end_time = datetime.datetime.utcnow()
    logging.info(f'-- finished at: {end_time.strftime("%Y-%m-%d %X")}')
    logging.info(f'-- elapsed time: {(end_time - start_time).total_seconds() / 60} minutes')
    return


def split_variables(files: list or str, save_dir: str):
    """
    split_variables('/Users/rchales/data/spatialdata/GLDAS_TwoHalfClip/TwoHalfClip*.nc4',
                '/Users/rchales/data/spatialdata/GLDAS_TwoHalfClip')
    """
    gldas_variables = [
        'Tair_f_inst', 'CanopInt_inst', 'Qg_tavg', 'ECanop_tavg', 'ESoil_tavg', 'PotEvap_tavg', 'Rainf_f_tavg',
        'Rainf_tavg', 'RootMoist_inst', 'Snowf_tavg', 'SoilTMP0_10cm_inst', 'Qair_f_inst', 'Qsb_acc', 'Psurf_f_inst',
        'Albedo_inst', 'LWdown_f_tavg', 'SWdown_f_tavg', 'Lwnet_tavg', 'Swnet_tavg', 'Qs_acc', 'SWE_inst', 'Qsm_acc',
        'SnowDepth_inst', 'AvgSurfT_inst', 'Qle_tavg', 'Qh_tavg', 'Tveg_tavg', 'Evap_tavg', 'Wind_f_inst']
    print('opening all files')
    combined_dataset = xr.open_mfdataset(files)
    print(combined_dataset)
    # logging.info('writing combined dataset to file')
    # combined_dataset.to_netcdf(os.path.join(save_dir, 'GLDAS_TwoHalf_TimeSeries.nc'))
    for num, variable in enumerate(gldas_variables):
        print(f'writing {variable} dataset to file ({num}/{len(gldas_variables)})')
        combined_dataset[variable].to_netcdf(os.path.join(save_dir, f'GLDAS_TwoHalfClip_TimeSeries_{variable}.nc'))
    return


# if __name__ == '__main__':
#     """
#     Expects 2 arguments
#     1- the relative path to the directory of data to be converted
#     2- the relative path to the directory to save converted data to
#     2- the relative path to the log file
#
#     Recommended usage:
#         python resample_gldas.py original-data/ half-degree-data/ log.log &
#     """
#     home_path = os.path.dirname(__file__)
#     read_path = sys.argv[1]
#     save_dir = sys.argv[2]
#     log_file = sys.argv[3]
#     files_to_convert = sorted(glob.glob(os.path.join(read_path, '*.nc4')))
#     logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
#     try:
#         for file in files_to_convert:
#             resample_and_clip(file, save_dir, 10, 'TwoHalf_Clipped', True)
#     except Exception as e:
#         logging.info('\n\n\n')
#         logging.info(f'FAILED at {datetime.datetime.utcnow()}')
#         logging.info(e)
