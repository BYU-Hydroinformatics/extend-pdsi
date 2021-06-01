import datetime
import json
import os
import random

import dateutil.relativedelta
import netCDF4 as nc
import numpy as np
import pandas as pd
import rch


def gen_pdsi_grid_table(pdsi_nc: str, res_json: str, save_path: str) -> None:
    a = nc.Dataset(pdsi_nc)
    mask = gen_array_mask(a['sc_PDSI_pm'][:])
    a.close()

    with open(res_json) as f:
        table = pd.DataFrame(json.loads(f.read()))

    # select the PDSI values
    prod, res, x0, y0, x_shp, y_shp = table[table['prod'] == 'pdsi'].values[0]
    x_idxs, y_idxs, x_vals, y_vals, c_nums = gen_reference_arrays(x_shp, y_shp, x0, y0, res)

    pd.DataFrame({
        "c_num": c_nums[~mask],
        "x_idx": x_idxs[~mask],
        "y_idx": y_idxs[~mask],
        "x_val": x_vals[~mask],
        "y_val": y_vals[~mask]
    }).to_csv(save_path, index=False)
    return


def gen_gldas_grid_tables(master_table: str, res_json: str):
    with open(res_json) as f:
        table = pd.DataFrame(json.loads(f.read()))

    for row in table[table['prod'] == 'gldas'].iterrows():
        row = row[1]

        if row.res != .5 and row.res != 2.5:
            continue

        new_x_vals = [row.x0 + (i * row.res) for i in range(row.x_shp)]
        new_y_vals = [row.y0 + (i * row.res) for i in range(row.y_shp)]

        c_nums = []
        x_idxs = []
        y_idxs = []
        x_vals = []
        y_vals = []

        for pdsi_row in pd.read_csv(master_table).iterrows():
            pdsi_row = pdsi_row[1]
            c_nums.append(pdsi_row.c_num)
            x_idxs.append(new_x_vals.index(pdsi_row.x_val))
            y_idxs.append(new_y_vals.index(pdsi_row.y_val))
            x_vals.append(pdsi_row.x_val)
            y_vals.append(pdsi_row.y_val)

        pd.DataFrame({
            "c_num": c_nums,
            "x_idx": x_idxs,
            "y_idx": y_idxs,
            "x_val": x_vals,
            "y_val": y_vals
        }).to_csv(f'lookup_tables/cell_table_gldas_{row.res}.csv', index=False)

    return


def gen_gldas_random_tables(master_table: str, n: int):
    """
    randomly picks n 1/4 gldas cells within each 2.5 degree pdsi cell to train on
    """
    x_shp = 1440
    y_shp = 600
    x0 = -179.975
    y0 = -59.875
    res = .25
    x_idxs, y_idxs, x_vals, y_vals, c_nums = gen_reference_arrays(x_shp, y_shp, x0, y0, res)
    subarrays = []
    for i in np.split(c_nums, y_shp / 10, axis=0):
        for j in np.split(i, x_shp / 10, axis=1):
            subarrays.append(j)

    target_pdsi_cells = pd.read_csv(master_table)['c_num'].tolist()

    all_cells = {}
    random_choices = {}
    # the first 720 subarrays are in the top 12.5 degrees of the 2.5 deg resolution grid
    for pdsi_cell_number, subarray in enumerate(subarrays[720:]):
        if len(target_pdsi_cells) == 0:
            break
        if pdsi_cell_number != target_pdsi_cells[0]:
            continue
        # remove the current cell from the list to iterate over
        target_pdsi_cells.pop(0)
        # create a dublicate of subarray
        choices = np.array(subarray).flatten()
        # shuffle the contents
        random.shuffle(choices)
        # select the first n elements
        choices = choices[:n].tolist()

        all_cells[f'pdsi_{pdsi_cell_number:04}'] = subarray.flatten().tolist()
        random_choices[f'pdsi_{pdsi_cell_number:04}'] = choices

    with open('lookup_tables/gldas_cells_random_assign.json', 'w') as f:
        f.write(json.dumps(random_choices))
    with open('lookup_tables/gldas_cells_pdsi_map.json', 'w') as f:
        f.write(json.dumps(all_cells))

    return


def gen_grid_view(source: str, save_path: str, sample_var: str):
    # open the file to be copied
    original = nc.Dataset(source, 'r', clobber=False, diskless=True)
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

    mask = gen_array_mask(original[sample_var][:])
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


def gen_time_series_table(file_path: str, table_path: str, var_table: str, dataset: str):
    a = nc.Dataset(file_path, 'r')
    table = pd.read_csv(table_path)
    data = {}
    origin = datetime.date(year=1948, month=1, day=1)
    if 'gldas' in dataset:
        variable = os.path.basename(file_path).replace('.nc', '').split('_')[-1]
        with open(var_table) as f:
            gldas_var_lookup = json.loads(f.read())
        var_code = gldas_var_lookup[variable]
        data['datetime'] = [origin + datetime.timedelta(days=int(i)) for i in a['time'][:]]
        time_idx = slice(None)
    else:
        variable = 'sc_PDSI_pm'
        var_code = 'v00'
        data['datetime'] = [origin + dateutil.relativedelta.relativedelta(months=int(i)) for i in range(852)]
        time_idx = slice(1176, 2028)

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
    a.to_csv(f'timeseries_tables_csv/{var_code}_cell_timeseries.csv')
    a.to_pickle(f'timeseries_tables_pickle/{var_code}_cell_timeseries.pickle')
    return


def gen_reference_arrays(x_shp, y_shp, x0, y0, res) -> tuple:
    # create a 2D array where the values are the x index of each cell
    x_idxs = np.array(list(range(x_shp)) * y_shp).reshape((y_shp, x_shp))
    # create a 2D array where the values are the y index of each cell
    y_idxs = np.transpose(np.array(list(range(y_shp)) * x_shp).reshape((x_shp, y_shp)))
    # create a 2D array where the values are the x value of each cell
    x_vals = np.array([x0 + (i * res) for i in range(x_shp)] * y_shp).reshape(y_shp, x_shp)
    # create a 2D array where the values are the y value of each cell
    y_vals = np.transpose(np.array([y0 + (i * res) for i in range(y_shp)] * x_shp).reshape(x_shp, y_shp))
    # create a 2D array where the values the number of the cell starting with 0, numbered like reading a book
    c_nums = np.array(range(x_shp * y_shp)).reshape(y_shp, x_shp)
    return x_idxs, y_idxs, x_vals, y_vals, c_nums


def gen_array_mask(a: np.array) -> np.array:
    if isinstance(a, np.ma.core.MaskedArray):
        if a.mask.dtype == bool:
            mask = a.mask
    else:
        mask = np.array(a)
        mask[mask > 0] = 1
        mask = np.nan_to_num(mask)
        mask = mask.astype(bool)
        mask = ~mask
    if mask.ndim == 3:
        mask = mask[-1]
    return mask


gen_pdsi_grid_table(
    '/Users/rchales/data/spatialdata/pdsisc.monthly.maps.1850-2018.fawc-1.r2.5x2.5.ipe-2.nc',
    'lookup_tables/dataset_resolution_map.json',
    'lookup_tables/cell_table_pdsi.csv'
)
gen_gldas_grid_tables(
    'lookup_tables/cell_table_pdsi.csv',
    'lookup_tables/dataset_resolution_map.json'
)
gen_gldas_random_tables('lookup_tables/cell_table_pdsi.csv', 15)
gen_time_series_table()
