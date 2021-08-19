# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:12:02 2021

@author: saulg
"""

import pandas as pd
import pickle


##############################
def read_well_pickle(data_file):
    # Load pickle file that has been concatenated as dictionary containing
    # all data
    with open(data_file, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary

def feature_stat_calc(data):
    data_list = list()
    data_list.append(data.mean())
    data_list.append(data.std())        
    data_list.append(data.max())
    data_list.append(data.min())
    return data_list

##############################
##############################

test_error = read_well_pickle('./PDSI Extension Results/Test_Metrics.pickle')
cell_error = read_well_pickle('./PDSI Extension Results/Model_Error.pickle')


Predictions = './PDSI Extension Results/PDSI_Predictions1958_2021.pickle' #Combined Dataset Location
Pred_Data = read_well_pickle(Predictions) #Data dictionary
Pred_Data_1958_2021 = pd.DataFrame(columns = (['Mean', 'STD', 'Max', 'Min']), index=Pred_Data.columns)
for i, cell in enumerate(Pred_Data):
    Pred_Data_1958_2021.loc[cell] = feature_stat_calc(Pred_Data[cell])


Extrapolation = './PDSI Extension Results/PDSI_Predictions2019_2021.pickle'
Extrap_Data = read_well_pickle(Extrapolation)
Pred_Data_2019_2021 = pd.DataFrame(columns = (['Mean', 'STD', 'Max', 'Min']), index=Pred_Data.columns)
for i, cell in enumerate(Extrap_Data):
    Pred_Data_2019_2021.loc[cell] = feature_stat_calc(Extrap_Data[cell])

