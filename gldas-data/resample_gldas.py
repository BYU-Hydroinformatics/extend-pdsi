"""
array[::2] "resamples" an array taking every other point

how this works:
1. get a list of all the indices in an array on both axes
2. resample both to every other point
3. iterate over both at the same time
4. for each combo, average the point, plus the 3 cells 1 step forward in each axis
5. put that in a new "row"
6. After finishing iterating across one of the directions, put that row into the new array
7. After finishing all iterations, create an np.array with the collection

Example:
    _1darray = np.array(range(1, 51))
    print(_1darray)
    print(resample_1d_array(_1darray))

    _2darray = np.array(range(1, 201)).reshape((20, 10))
    print(_2darray)
    print(resample_2d_array(_2darray))

"""

import glob
import os
import netCDF4 as nc
import numpy as np
import sys
import datetime


def resample_1d_array(a: np.array):
    new_array = []
    for i in np.array(range(a.shape[0]))[::2]:
        avg = np.nanmean([a[i], a[i + 1]])
        new_array.append(avg)
    return np.array(new_array)


def resample_2d_array(a: np.array):
    new_array = []
    for i in np.array(range(a.shape[0]))[::2]:
        new_row = []
        for j in np.array(range(a.shape[1]))[::2]:
            avg = np.nanmean([a[i, j], a[i + 1, j], a[i + 1, j + 1], a[i, j + 1]])
            new_row.append(avg)
        new_array.append(new_row)
    return np.array(new_array)


def resample(open_path: str, save_dir: str):
    start_time = datetime.datetime.utcnow()
    print(f'Working on file: {open_path}')
    print(f'-- started at: {start_time.strftime("%Y-%m-%d %X")}')

    save_path = os.path.join(save_dir, f'HalfDegree_{os.path.basename(open_path)}')

    # open the file to be copied
    original = nc.Dataset(open_path, 'r', clobber=False, diskless=True)
    duplicate = nc.Dataset(save_path, 'w', clobber=True, format='NETCDF4', diskless=False)

    # copy the global netcdf attributes
    duplicate.setncatts(original.__dict__)

    # specify dimensions from what we copied before
    for dimension in original.dimensions:
        size = original.dimensions[dimension].size
        if dimension in ('lat', 'lon', ):
            size = size / 2
        duplicate.createDimension(dimension, size)

    # handle lat, lon, and time separately because they have 1d arrays
    duplicate.createVariable(varname='lat', datatype='f4', dimensions='lat', zlib=True, shuffle=True)
    duplicate.createVariable(varname='lon', datatype='f4', dimensions='lon', zlib=True, shuffle=True)
    duplicate.createVariable(varname='time', datatype='i', dimensions='time', zlib=True, shuffle=True)
    duplicate['lat'][:] = resample_1d_array(original['lat'][:])
    duplicate['lon'][:] = resample_1d_array(original['lon'][:])
    duplicate['time'][:] = original['time'][:]
    for variable in ('time', 'lat', 'lon', ):
        for attr in original[variable].__dict__:
            if attr != "_FillValue":
                duplicate[variable].setncattr(attr, original[variable].__dict__[attr])

    # copy the rest of the variables
    for variable in original.variables:
        if variable in ('time', 'lat', 'lon', 'time_bnds'):
            continue
        # create the variable
        duplicate.createVariable(varname=variable, datatype='f4', dimensions=original[variable].dimensions,
                                 zlib=True, shuffle=True, fill_value=original[variable].__dict__.get('_FillValue', None))
        # set the attributes for lat and lon (except fill value, you just can't copy it)
        for attr in original[variable].__dict__:
            if attr != "_FillValue":
                duplicate[variable].setncattr(attr, original[variable].__dict__[attr])
        # copy the array and resample
        arr = np.array(original[variable][:])
        arr[arr < -9000] = np.nan
        arr = np.squeeze(arr)
        duplicate[variable][:] = resample_2d_array(arr)

    # close the netcdfs
    original.close()
    duplicate.close()

    end_time = datetime.datetime.utcnow()
    print(f'-- finished at: {end_time.strftime("%Y-%m-%d %X")}')
    print(f'-- elapsed time: {(end_time - start_time).total_seconds() / 60} minutes')
    return


if __name__ == '__main__':
    """
    Expects 2 arguments
    1- the relative path to the directory of data to be converted
    2- the relative path to the directory to save converted data to
    
    Recommended usage:
        python resample_gldas.py original-data/ half-degree-data/ >> log.log &
    """
    home_path = os.path.dirname(__file__)
    read_path = sys.argv[1]
    save_dir = sys.argv[2]
    files_to_convert = sorted(glob.glob(os.path.join(read_path, '*.nc4')))
    try:
        for file in files_to_convert:
            resample(file, save_dir)
    except Exception as e:
        print('\n\n\n')
        print(f'FAILED at {datetime.datetime.utcnow()}')
        print(e)
