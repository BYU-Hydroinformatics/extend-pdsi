# -*- coding: utf-8 -*-
"""
Created on Mon May 24 23:18:42 2021

@author: saulg
"""

import numpy as np
import pandas as pd
import os
import pickle


# Reorganizing Images and renaming based on index
#os.getcwd()
collection = './timeseries_tables_pickle/' #Location of Raw Images

dirs = os.walk(collection) #Get all files in a directory
Data_Dictionary = dict()
for i, filename in enumerate(os.listdir(collection)):
    table = pd.read_pickle(collection + filename)
    var_name = filename.split('.')[0]
    Data_Dictionary[var_name] = table
pickle_name = 'Complete_Timeseries' + '.pickle'
with open(pickle_name, 'wb') as handle:
    pickle.dump(Data_Dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


