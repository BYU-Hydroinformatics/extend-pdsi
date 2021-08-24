import os

import netCDF4 as nc
import numpy as np
import pandas as pd


def produce_netcdf(save_dir: str, cell_table: str, error_table: str, pdsi_ext_table: str, pdsi_mod_table: str):
    new_nc = nc.Dataset(os.path.join(save_dir, 'pdsi_extended.nc4'), 'w', format="NETCDF4")

    cell_df = pd.read_csv(cell_table, index_col=0)
    error_df = pd.read_pickle(error_table)
    error_df.index = error_df.index.str.replace('c', '').astype(int)
    error_df = error_df.join(cell_df)
    pdsi_ext_df = pd.read_pickle(pdsi_ext_table)
    new_values = np.where(pdsi_ext_df.values > 10, 10, pdsi_ext_df.values)
    new_values = np.where(new_values < -10, -10, new_values)
    pdsi_ext_df = pd.DataFrame(new_values, columns=pdsi_ext_df.columns, index=pdsi_ext_df.index)
    pdsi_mod_df = pd.read_pickle(pdsi_mod_table)

    x_origin = -178.75
    y_origin = -58.75
    x_size = 144
    y_size = 55
    t_size_ext = pdsi_ext_df.index.shape[0]
    t_size_mod = pdsi_mod_df.index.shape[0]
    resolution = 2.5
    pdsi_lat = [y_origin + (i * resolution) for i in range(y_size)]
    pdsi_lon = [x_origin + (i * resolution) for i in range(x_size)]

    new_nc.createDimension('time', None)
    new_nc.createDimension('lat', len(pdsi_lat))
    new_nc.createDimension('lon', len(pdsi_lon))

    time = new_nc.createVariable('time', 'i', 'time', zlib=True, shuffle=True)
    lat = new_nc.createVariable('lat', 'f4', 'lat', zlib=True, shuffle=True)
    lon = new_nc.createVariable('lon', 'f4', 'lon', zlib=True, shuffle=True)

    new_nc.createVariable("pdsi_extended", 'f4', ('time', 'lat', 'lon'), zlib=True, shuffle=True, fill_value=np.nan)
    new_nc.createVariable("pdsi_modeled", 'f4', ('time', 'lat', 'lon'), zlib=True, shuffle=True, fill_value=np.nan)
    new_nc.createVariable('me', 'f4', ('lat', 'lon'), zlib=True, shuffle=True, fill_value=np.nan)
    new_nc.createVariable('mse', 'f4', ('lat', 'lon'), zlib=True, shuffle=True, fill_value=np.nan)
    new_nc.createVariable('rmse', 'f4', ('lat', 'lon'), zlib=True, shuffle=True, fill_value=np.nan)

    new_nc['pdsi_extended'].setncattr('long_name', "PDSI Extended to Present")
    new_nc['pdsi_modeled'].setncattr('long_name', "PDSI Modeled by Regression")
    new_nc['me'].setncattr('long_name', "Mean Error")
    new_nc['mse'].setncattr('long_name', "Mean Square Error")
    new_nc['rmse'].setncattr('long_name', "Root Mean Square Error")

    # assign the time variable
    time[:] = [(i - pdsi_ext_df.index[0]).days for i in pdsi_ext_df.index]
    time.axis = "T"
    time.calendar = 'standard'
    time.units = 'days since since 1948-01-01 00:00:00'

    # assign the lat/lon arrays
    new_nc['lat'][:] = pdsi_lat
    lat.long_name = "Latitude"
    lat.units = "degrees_north"
    lat.axis = "Y"
    new_nc['lon'][:] = pdsi_lon
    lon.long_name = "Longitude"
    lon.units = "degrees_east"
    lon.axis = "X"

    # assign the error variables
    me_array = np.array([np.nan] * x_size * y_size).reshape((y_size, x_size))
    rmse_array = np.array([np.nan] * x_size * y_size).reshape((y_size, x_size))
    mse_array = np.array([np.nan] * x_size * y_size).reshape((y_size, x_size))
    for row in error_df.iterrows():
        me_array[row[1].y_idx, row[1].x_idx] = row[1].ME
        mse_array[row[1].y_idx, row[1].x_idx] = row[1].MSE
        rmse_array[row[1].y_idx, row[1].x_idx] = row[1].RMSE
    new_nc['me'][:] = me_array
    new_nc['mse'][:] = me_array
    new_nc['rmse'][:] = rmse_array

    # write the pdsi variables
    pdsi_ext_array = np.array([np.nan] * t_size_ext * y_size * x_size).reshape((t_size_ext, y_size, x_size))
    pdsi_mod_array = np.array([np.nan] * t_size_mod * y_size * x_size).reshape((t_size_mod, y_size, x_size))
    for i, cell in enumerate(pdsi_ext_df.columns):
        indices = cell_df.loc[[int(cell.replace('c', ''))]]
        pdsi_ext_array[:, indices.y_idx, indices.x_idx] = pdsi_ext_df[cell].values.reshape(t_size_ext, 1)
        pdsi_mod_array[:, indices.y_idx, indices.x_idx] = pdsi_mod_df[cell].values.reshape(t_size_mod, 1)
    pdsi_mod_array = np.pad(pdsi_mod_array, ((t_size_ext - t_size_mod, 0), (0, 0), (0, 0)), constant_values=np.nan)
    new_nc['pdsi_extended'][:] = pdsi_ext_array
    new_nc['pdsi_modeled'][:] = pdsi_mod_array

    new_nc.sync()
    new_nc.close()
    return


save_dir = '/Users/rchales/data/pdsi_data/results'
cell_table = '/Users/rchales/code/extend-pdsi/lookup_tables/cell_table_pdsi.csv'
error_table = '/Users/rchales/data/pdsi_data/PDSI_results_8_24/Model_Error.pickle'
pdsi_ext_table = '/Users/rchales/data/pdsi_data/PDSI_results_8_24/PDSI_Extended1948_2021.pickle'
pdsi_mod_table = '/Users/rchales/data/pdsi_data/PDSI_results_8_24/PDSI_Predictions1958_2021.pickle'

produce_netcdf(save_dir, cell_table, error_table, pdsi_ext_table, pdsi_mod_table)
