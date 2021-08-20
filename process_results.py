import netCDF4 as nc
import pandas as pd
import numpy as np


def gen_error_map(error_table: str, cell_table: str, res: str, model: str):
    err = pd.read_pickle(error_table)
    err.index = err.index.str.replace('c', '').astype(int)
    ltab = pd.read_csv(cell_table, index_col=0)
    assign_table = err.join(ltab)

    x_origin = -178.75
    y_origin = -58.75
    x_size = 144
    y_size = 55
    resolution = 2.5

    map = nc.Dataset(f'error_maps_{res}_gldas_{model}.nc', 'w')
    # create dimensions
    map.createDimension('lat', y_size)
    map.createDimension('lon', x_size)
    # create variables
    map.createVariable(varname='lat', datatype='f4', dimensions='lat', zlib=True, shuffle=True)
    map.createVariable(varname='lon', datatype='f4', dimensions='lon', zlib=True, shuffle=True)
    map.createVariable(varname='ME', datatype='f4', dimensions=('lat', 'lon'), zlib=True, shuffle=True, fill_value=np.nan)
    map.createVariable(varname='RMSE', datatype='f4', dimensions=('lat', 'lon'), zlib=True, shuffle=True, fill_value=np.nan)
    map.createVariable(varname='MSE', datatype='f4', dimensions=('lat', 'lon'), zlib=True, shuffle=True, fill_value=np.nan)
    # fill variables
    map['lat'][:] = np.array([y_origin + (i * resolution) for i in range(y_size)])
    map['lon'][:] = np.array([x_origin + (i * resolution) for i in range(x_size)])

    # create arrays for the errors
    me_array = np.array([np.nan] * x_size * y_size).reshape(y_size, x_size)
    rmse_array = np.array([np.nan] * x_size * y_size).reshape(y_size, x_size)
    mse_array = np.array([np.nan] * x_size * y_size).reshape(y_size, x_size)
    for row in assign_table.iterrows():
        me_array[row[1].y_idx, row[1].x_idx] = row[1].ME
        rmse_array[row[1].y_idx, row[1].x_idx] = row[1].RMSE
        mse_array[row[1].y_idx, row[1].x_idx] = row[1].MSE

    # put error arrays in the netcdf variables
    map['ME'][:] = me_array
    map['RMSE'][:] = rmse_array
    map['MSE'][:] = mse_array

    map.close()
    return


# gen_error_map('error_250_reg.pickle', 'lookup_tables/cell_assign_gldas_250_clipped.csv', '250', 'reg')
# gen_error_map('error_250_unreg.pickle', 'lookup_tables/cell_assign_gldas_250_clipped.csv', '250', 'ureg')
gen_error_map('/Users/rchales/Downloads/PDSI_Error 7_16/regularized_error.pickle', '/Users/rchales/code/extend-pdsi/lookup_tables/cell_table_gldas_2.5.csv', '250', 'reg')

# pd.read_pickle('/Users/rchales/Downloads/PDSI_Error 7_16/regularized_error.pickle').to_csv('regerror.csv')
