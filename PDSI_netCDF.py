# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:50:59 2021

@author: saulg
"""

import numpy as np
import netCDF4 as nc
import pickle
import pandas as pd
from tqdm import tqdm

def PDSI_Grid_Creation():
        x_origin = -178.75
        y_origin = -58.75
        x_size = 144
        y_size = 55
        resolution = 2.5
        
        lat_range = np.array([y_origin + (i * resolution) for i in range(y_size)])
        lon_range = np.array([x_origin + (i * resolution) for i in range(x_size)]) 

        return lat_range, lon_range

# Opens generic pickle file based on file path and loads data.
def read_pickle(file, root):
    file = root + '/' + file + '.pickle'
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    return data

def netcdf_setup(grid_long, grid_lat, timestamp, filename):
        # setup a netcdf file to store the time series of rasters
        # copied from other lab code - you probably don't need everything here
        
        file = nc.Dataset(filename, 'w', format="NETCDF4")
        
        lon_len  = len(grid_long)  # size of grid in x_dir
        lat_len  = len(grid_lat)  # size of grid in y_dir
        
        time = file.createDimension("time", None) # time dimension - can extend e.g., size=0
        lat  = file.createDimension("lat", lat_len)  # create lat dimension in netcdf file of len lat_len
        lon  = file.createDimension("lon", lon_len)  # create lon dimension in netcdf file of len lon_len
        
        time      = file.createVariable("time", np.float64, ("time"))
        latitude  = file.createVariable("lat", np.float64, ("lat"))  # create latitude varilable
        longitude = file.createVariable("lon", np.float64, ("lon")) # create longitude varilbe
        
        sc_PDSI_pm   = file.createVariable("sc_PDSI_pm", np.float64, ('time', 'lat', 'lon'), fill_value=np.nan)
        sc_PDSI_pm.long_name = "Extended Monthly Self-calibrated Palmer Drought Severity Index using Penman-Monteith PE"
        
        PDSI_Model_Output   = file.createVariable("PDSI_Model_Output", np.float64, ('time', 'lat', 'lon'), fill_value=np.nan)
        PDSI_Model_Output.long_name = "Monthly Self-calibrated Palmer Drought Severity Index using Penman-Monteith PE"        
        
        Mean_Error   = file.createVariable("Mean_Error", np.float64, ('time', 'lat', 'lon'), fill_value=np.nan)
        Mean_Error.long_name = "Mean Error showing model bias"  

        
        RMSE   = file.createVariable("RMSE", np.float64, ('time', 'lat', 'lon'), fill_value=np.nan)
        RMSE.long_name = "Root Mean Squared Error of cells"          
        # Netcdf seems to flip lat/long for building grid
        latitude[:] = grid_lat[:] 
        longitude[:] = grid_long[:]

    
        latitude.long_name = "latitute in degrees, negative for the Southern Hemisphere"
        latitude.units = "degrees_north"
        latitude.axis = "Y"
        
        longitude.long_name = "longitude in degrees, negative for the Western Hemisphere"
        longitude.units = "degrees_east"
        longitude.axis = "X"
        
        timestamp = list(pd.to_datetime(timestamp))
        units = 'days since 0001-01-01 00:00:00'
        calendar = 'standard'
        time[:] = nc.date2num(dates = timestamp, units = units, calendar= calendar)
        time.axis = "T"
        time.units = units
        
        return file, sc_PDSI_pm, PDSI_Model_Output, Mean_Error, RMSE

def Error_Maps(lat_range, long_range, cell_names, cell_location, model_error):
    ME_Array = np.zeros((len(lat_range), len(lon_range)))
    RMSE_Array = np.zeros((len(lat_range), len(lon_range)))
    for i, cell in enumerate(tqdm(cell_names)):
        try:
            data_index = int(cell.replace('c', ''))
            indices = cell_location.loc[[data_index]]
            cell_data = model_error.loc[[cell]]
            ME_Array[indices['y_idx'], indices['x_idx']] = cell_data['ME']
            RMSE_Array[indices['y_idx'], indices['x_idx']] = cell_data['RMSE']
        except:
            print(cell)
    ME_Array = np.where(ME_Array == 0, np.nan, ME_Array)
    RMSE_Array = np.where(RMSE_Array == 0, np.nan, RMSE_Array)
    return ME_Array, RMSE_Array

#Load Dataframe
data_1948_2021 = read_pickle('PDSI_Extended1948_2021', './PDSI Extension Results')
data_1958_2021 = read_pickle('PDSI_Predictions1958_2021', './PDSI Extension Results')
data_2019_2021 = read_pickle('PDSI_Predictions2019_2021', './PDSI Extension Results')
model_error = read_pickle('Model_Error', './PDSI Extension Results')
cell_location = pd.read_csv('./cell_location.csv', index_col=0)


# Create grid based on netcdf metadata. Inputs are NE_lat, SW_lat, NE_lon, SW_lon
# x resolution, and y resolution. Calculates the centroids.
lat_range, lon_range = PDSI_Grid_Creation()

# Create Error Array so it can be added to netcdf in 1 pass without timesteps
ME_Array, RMSE_Array = Error_Maps(lat_range, lon_range, data_1948_2021.columns, cell_location, model_error)

# Create NetCDF file     
file_nc, sc_PDSI_pm, PDSI_Model_Output, Mean_Error, RMSE = netcdf_setup(lon_range, lat_range,
     data_1948_2021.index, filename='PDSI Extended 1948-2021.nc')

# Append Error Arrays
Mean_Error[0,:,:] = ME_Array
RMSE[0,:,:] = RMSE_Array

# Loop through each date, create variogram for time step create krig map.
# Inserts the grid at the timestep within the netCDF.
for i, cell in enumerate(tqdm(data_1948_2021.columns)):
    try:
        data_index = int(cell.replace('c', ''))
        indices = cell_location.loc[[data_index]]
        # write data to netcdf file
        sc_PDSI_pm[:, indices['y_idx'], indices['x_idx']] = data_1948_2021[cell].clip(-10,10).values
        PDSI_Model_Output[:, indices['y_idx'], indices['x_idx']] = data_1958_2021[cell].values
    except:
        print(cell)

file_nc.close()