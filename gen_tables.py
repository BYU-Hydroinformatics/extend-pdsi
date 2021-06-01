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





gen_cell_list('/Users/rchales/data/spatialdata/GLDAS_TwoHalfClip/GLDAS_TwoHalfClip_TimeSeries_Tair_f_inst.nc',
              'lookup_tables/cell_assign_gldas_250_clipped.csv', 'Tair_f_inst', False)
for file in glob.glob('/Users/rchales/data/spatialdata/GLDAS_TwoHalfClip/GLDAS_TwoHalfClip*'):
    cell_time_series_table(file, 'lookup_tables/cell_assign_gldas_250_clipped.csv', 'gldas_twohalfclip')

gen_cell_list('/Users/rchales/data/spatialdata/pdsisc.monthly.maps.1850-2018.fawc-1.r2.5x2.5.ipe-2.nc',
              'lookup_tables/cell_assign_pdsi.csv', 'sc_PDSI_pm', False)
cell_time_series_table('/Users/rchales/data/spatialdata/pdsisc.monthly.maps.1850-2018.fawc-1.r2.5x2.5.ipe-2.nc',
                       'lookup_tables/cell_assign_pdsi.csv',
                       'pdsi')

