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


if __name__ == '__main__':
    """
    Expects 2 arguments
    1- the relative path to the directory of data to be converted
    2- the relative path to the directory to save converted data to
    2- the relative path to the log file

    Recommended usage:
        python resample_gldas.py original-data/ half-degree-data/ log.log &
    """
    home_path = os.path.dirname(__file__)
    read_path = sys.argv[1]
    save_dir = sys.argv[2]
    log_file = sys.argv[3]
    files_to_convert = sorted(glob.glob(os.path.join(read_path, '*.nc4')))
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
    try:
        for file in files_to_convert:
            resample(file, save_dir, 5, 'TwoHalfDeg')
    except Exception as e:
        logging.info('\n\n\n')
        logging.info(f'FAILED at {datetime.datetime.utcnow()}')
        logging.info(e)
