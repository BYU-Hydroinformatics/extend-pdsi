import glob

import pandas as pd
import numpy as np
import netCDF4 as nc
import rch
import os
import json
import datetime
import dateutil.relativedelta


def gen_cell_list(sample_file: str, csv_name: str, sample_var: str, bool_mask: bool = True) -> None:
    a = nc.Dataset(sample_file, 'r')
    y = a['lat'][:]
    x = a['lon'][:]
    if bool_mask:
        mask = np.squeeze(a[sample_var][:].mask)
    else:
        mask = np.array(np.squeeze(a[sample_var][:]))
        mask = np.where(mask > -1000, 1, 0)
        # mask[mask > -1000] = 1
        # mask[mask < -1000] = 0
        # mask = np.nan_to_num(mask)
        mask = mask.astype(bool)
        mask = mask[-1]
    a.close()
    y_shp = y.shape[0]
    x_shp = x.shape[0]

    # create a 2D array where the values are the x index of each cell (bottom left is 0,0)
    x_idxs = list(range(x_shp)) * y_shp
    x_idxs = np.array(x_idxs).reshape((y_shp, x_shp))

    # create a 2D array where the values are the y index of each cell (bottom left is 0,0)
    y_idxs = list(range(y_shp)) * x_shp
    y_idxs = np.transpose(np.array(y_idxs).reshape((x_shp, y_shp)))

    # create a 2D array where the values are the x value of each cell (bottom left is 0,0)
    x_vals = np.array(list(x) * y_shp).reshape(y_shp, x_shp)
    # create a 2D array where the values are the y value of each cell (bottom left is 0,0)
    y_vals = np.transpose(np.array(list(y) * x_shp).reshape(x_shp, y_shp))

    # create a 2D array where the values the number of the cell starting with 0, numbered like reading a book
    c_nums = np.array(range(0, x_shp * y_shp)).reshape(y_shp, x_shp)

    pd.DataFrame({
        "c_num": c_nums[mask],
        "x_idx": x_idxs[mask],
        "y_idx": y_idxs[mask],
        "x_val": x_vals[mask],
        "y_val": y_vals[mask]
    }).to_csv(csv_name)
    return


def gen_grid_view(file_to_duplicate: str, save_path: str, sample_var_name: str, bool_mask: bool):
    # open the file to be copied
    original = nc.Dataset(file_to_duplicate, 'r', clobber=False, diskless=True)
    duplicate = nc.Dataset(save_path, 'w', clobber=True, format='NETCDF4', diskless=False)

    # copy the global netcdf attributes
    duplicate.setncatts(original.__dict__)

    # specify dimensions from what we copied before
    for dimension in original.dimensions:
        duplicate.createDimension(dimension, original.dimensions[dimension].size)

    # handle lat, lon, and time separately because they have 1d arrays
    duplicate.createVariable(varname='lat', datatype='f4', dimensions='lat', zlib=True, shuffle=True)
    duplicate['lat'][:] = np.array(original['lat'][:])

    duplicate.createVariable(varname='lon', datatype='f4', dimensions='lon', zlib=True, shuffle=True)
    duplicate['lon'][:] = np.array(original['lon'][:])

    duplicate.createVariable(varname='gridview', datatype='f4', dimensions=('lat', 'lon'), zlib=True, shuffle=True,
                             fill_value=np.nan)
    alternating_array = rch.arrays.gen_checkerboard(original['lat'][:].shape[0], original['lon'][:].shape[0])
    if bool_mask:
        mask = np.squeeze(original[sample_var_name][:].mask)
    else:
        mask = np.array(np.squeeze(original[sample_var_name][:]))
        mask[mask > 0] = 1
        mask = np.nan_to_num(mask)
        mask = mask.astype(bool)
    alternating_array = alternating_array.astype('float')
    if mask.ndim == 3:
        mask = mask[-1]
        mask = ~mask
    alternating_array[~mask] = np.nan
    duplicate['gridview'][:] = alternating_array

    for variable in ('lat', 'lon'):
        for attr in original[variable].__dict__:
            if attr != "_FillValue":
                duplicate[variable].setncattr(attr, original[variable].__dict__[attr])

    # close the netcdfs
    original.close()
    duplicate.close()

    return


def cell_time_series_table(file_path: str, table_path: str, dataset: str):
    a = nc.Dataset(file_path, 'r')
    table = pd.read_csv(table_path)
    data = {}
    if 'gldas' in dataset:
        variable = os.path.basename(file_path).replace('GLDAS_TwoHalfClip_TimeSeries_', '').replace('.nc', '')
        with open('lookup_tables/variables_lookup.json') as f:
            gldas_var_lookup = json.loads(f.read())
        var_code = gldas_var_lookup[variable]
        origin = datetime.date(year=1948, month=1, day=1)
        data['datetime'] = [origin + datetime.timedelta(days=int(i)) for i in a['time'][:]]
        time_idx = slice(None)
    else:
        variable = 'sc_PDSI_pm'
        var_code = 'v00'
        origin = datetime.date(year=1948, month=1, day=1)
        data['datetime'] = [origin + dateutil.relativedelta.relativedelta(months=int(i)) for i in range(853)]
        time_idx = slice(1175, 2028)

    array = np.array(a[variable][:])
    array[array == -9999] = np.nan
    array[array == -99999] = np.nan
    array = np.squeeze(array)
    for c_num, y_idx, x_idx in table[['c_num', 'y_idx', 'x_idx']].values:
        data[f'{var_code}_c{int(c_num):04}'] = array[time_idx, int(y_idx), int(x_idx)]

    a = pd.DataFrame(data)
    a.index = a['datetime']
    del a['datetime']
    a.index.name = 'datetime'
    # delete the empty columns
    a = a.dropna(axis=1)
    a.to_csv(f'timeseries_tables_csv/{var_code}_cell_timeseries.csv')
    a.to_pickle(f'timeseries_tables_pickle/{var_code}_cell_timeseries.pickle')
    return


# todo figure out why there is a difference between the masked cells placed on the list and cells which have values

gen_cell_list('/Users/rchales/data/spatialdata/GLDAS_TwoHalfClip/GLDAS_TwoHalfClip_TimeSeries_Tair_f_inst.nc',
              'lookup_tables/cell_assign_gldas_250_clipped.csv', 'Tair_f_inst', False)
for file in glob.glob('/Users/rchales/data/spatialdata/GLDAS_TwoHalfClip/GLDAS_TwoHalfClip*'):
    cell_time_series_table(file, 'lookup_tables/cell_assign_gldas_250_clipped.csv', 'gldas_twohalfclip')

gen_cell_list('/Users/rchales/data/spatialdata/pdsisc.monthly.maps.1850-2018.fawc-1.r2.5x2.5.ipe-2.nc',
              'lookup_tables/cell_assign_pdsi.csv', 'sc_PDSI_pm', False)
cell_time_series_table('/Users/rchales/data/spatialdata/pdsisc.monthly.maps.1850-2018.fawc-1.r2.5x2.5.ipe-2.nc',
                       'lookup_tables/cell_assign_pdsi.csv',
                       'pdsi')


# gen_grid_view('/Users/rchales/data/spatialdata/GLDAS_Half/HalfDegree_GLDAS_NOAH025_M.A194801.020.nc4', 'gldas05grid.nc', 'Tair_f_inst', False)
# gen_grid_view('/Users/rchales/data/spatialdata/GLDAS_OneTwoFive/OneTwoFive_GLDAS_NOAH025_M.A194801.020.nc4', 'gldas125grid.nc', 'Tair_f_inst', False)
# gen_grid_view('/Users/rchales/data/spatialdata/GLDAS_TwoHalf/TwoHalfDeg_GLDAS_NOAH025_M.A194801.020.nc4', 'gldas25grid.nc', 'Tair_f_inst', False)
# gen_grid_view('/Users/rchales/data/spatialdata/pdsisc.monthly.maps.1850-2018.fawc-1.r2.5x2.5.ipe-2.nc', 'pdsigrid.nc', 'sc_PDSI_pm', True)

# gen_cell_list('/Users/rchales/data/spatialdata/GLDAS_Original/GLDAS_NOAH025_M.A194801.020.nc4', 'cell_assign_gldas_025.csv', 'Tair_f_inst', True)
# gen_cell_list('/Users/rchales/data/spatialdata/GLDAS_Half/HalfDegree_GLDAS_NOAH025_M.A194801.020.nc4', 'cell_assign_gldas_050.csv', 'Tair_f_inst', False)
# gen_cell_list('/Users/rchales/data/spatialdata/GLDAS_OneTwoFive/OneTwoFive_GLDAS_NOAH025_M.A194801.020.nc4', 'cell_assign_gldas_125.csv', 'Tair_f_inst', False)
# gen_cell_list('/Users/rchales/data/spatialdata/GLDAS_TwoHalf/TwoHalfDeg_GLDAS_NOAH025_M.A194801.020.nc4', 'cell_assign_gldas_250_clipped.csv.csv', 'Tair_f_inst', False)
# gen_cell_list('/Users/rchales/data/spatialdata/pdsisc.monthly.maps.1850-2018.fawc-1.r2.5x2.5.ipe-2.nc', 'cell_assign_pdsi.csv', 'sc_PDSI_pm', True)
