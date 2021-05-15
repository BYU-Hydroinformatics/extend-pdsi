import glob
import os
import netCDF4 as nc
import numpy as np
import xarray as xr


home_path = os.path.dirname(__file__)
raw_files = sorted(glob.glob(os.path.join(home_path, 'original-data', 'GLDAS*.nc4')))
save_path = os.path.join(home_path, 'half-degree-data')

filepaths = ['original-data/GLDAS_NOAH025_M.A194801.020.nc4', ]
# a = nc.Dataset('GLDAS_NOAH025_M.A194801.020.nc4')
# b = nc.Dataset('HALFDEGREE_GLDAS_NOAH025_M.A194801.020.nc4')

# test = np.array([i for i in range(1, 51)])
# print(test)
# test1 = test[::2]
# print(test1)
# exit()

raw_files = raw_files[:20]
a = xr.open_mfdataset(raw_files, combine='nested', concat_dim='time')
print(a)
a.close()

exit()


# this is where the files start getting copied
for filepath in filepaths:
    filename = os.path.basename(filepath)
    print('Working on file ' + str(filepath))
    openpath = os.path.join(home_path, filepath)
    savepath = os.path.join(save_path, 'processed_' + filepath)
    # open the file to be copied
    original = nc.Dataset(openpath, 'r', clobber=False, diskless=True)
    duplicate = nc.Dataset(savepath, 'w', clobber=True, format='NETCDF4', diskless=False)
    # set the global netcdf attributes - important for georeferencing
    duplicate.setncatts(original.__dict__)

    # specify dimensions from what we copied before
    for dimension in dimensions:
        duplicate.createDimension(dimension, dimensions[dimension])

    duplicate.createVariable(varname='lat', datatype='f4', dimensions='lat', zlib=True, shuffle=True)
    duplicate.createVariable(varname='lon', datatype='f4', dimensions='lon', zlib=True, shuffle=True)
    # duplicate.createVariable(varname='lat', datatype='f4', dimensions='lat')
    # duplicate.createVariable(varname='lon', datatype='f4', dimensions='lon')

    # create the lat and lon values as a 1D array
    lat_list = [lat_min + i * lat_step for i in range(dimensions['lat'])]
    lon_list = [lon_min + i * lon_step for i in range(dimensions['lon'])]
    duplicate['lat'][:] = lat_list
    duplicate['lon'][:] = lon_list

    # set the attributes for lat and lon (except fill value, you just can't copy it)
    for attr in original['lat'].__dict__:
        if attr != "_FillValue":
            duplicate['lat'].setncattr(attr, original['lat'].__dict__[attr])
    for attr in original['lon'].__dict__:
        if attr != "_FillValue":
            duplicate['lon'].setncattr(attr, original['lon'].__dict__[attr])

    # copy the rest of the variables
    date = '201906'
    timestep = 0
    timedelta = 1
    for variable in variables:
        # check to use the lat/lon dimension names
        dimension = original[variable].dimensions
        if 'latitude' in dimension:
            dimension = list(dimension)
            dimension.remove('latitude')
            dimension.append('lat')
            dimension = tuple(dimension)
        if 'north_south' in dimension:
            dimension = list(dimension)
            dimension.remove('north_south')
            dimension.append('lat')
            dimension = tuple(dimension)
        if 'longitude' in dimension:
            dimension = list(dimension)
            dimension.remove('longitude')
            dimension.append('lon')
            dimension = tuple(dimension)
        if 'east_west' in dimension:
            dimension = list(dimension)
            dimension.remove('east_west')
            dimension.append('lon')
            dimension = tuple(dimension)
        if 'time' not in dimension:
            dimension = list(dimension)
            dimension = ['time'] + dimension
            dimension = tuple(dimension)
        if len(dimension) == 2:
            dimension = ('time', 'lat', 'lon')
        if variable == 'time':
            dimension = ('time',)

        print(variable)
        print(dimension)

        # create the variable
        if compress:
            duplicate.createVariable(varname=variable, datatype='f4', dimensions=dimension, zlib=True, shuffle=True)
        else:
            duplicate.createVariable(varname=variable, datatype='f4', dimensions=dimension)

        # copy the arrays of data and set the metadata/properties
        if variable == 'time':
            duplicate[variable][:] = [timestep]
            timestep += timedelta
            duplicate[variable].long_name = original[variable].long_name
            duplicate[variable].units = "hours since " + date
            duplicate[variable].axis = "T"  # or time
            # also set the begin date of this data
            duplicate[variable].begin_date = date
        if variable == 'lat':
            duplicate[variable][:] = original[variable][:]
            duplicate[variable].axis = "Y"  # or lat
        if variable == 'lon':
            duplicate[variable][:] = original[variable][:]
            duplicate[variable].axis = "X"  # or lon
        else:
            duplicate[variable][:] = original[variable][:]
            duplicate[variable].axis = "lat lon"

        duplicate[variable].begin_date = date
        try:
            duplicate[variable].long_name = original[variable].long_name
        except AttributeError:
            duplicate[variable].long_name = variable
        try:
            duplicate[variable].units = original[variable].units
        except AttributeError:
            duplicate[variable].units = 'unknown units'

    # close the file then start again
    original.close()
    duplicate.sync()
    duplicate.close()

