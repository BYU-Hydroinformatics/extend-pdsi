# -*- coding: utf-8 -*-
"""
Created on Mon May 24 23:18:42 2021

@author: saulg
"""


import pandas as pd
import os
import pickle


root = './Data_Processed/'

# Import variable names
names = pd.read_csv(root + 'variable_names.csv', delimiter=',')
names = names['Variables'].tolist()

# Reorganizing netCDF
collection = root + 'Timeseries/' #Location

dirs = os.walk(collection) #Get all files in a directory
Data_Dictionary = dict()
for i, filename in enumerate(os.listdir(collection)):
    table = pd.read_pickle(collection + filename)
    var_name = names[i]
    Data_Dictionary[var_name] = table
    
    
var_name = list(Data_Dictionary.keys()) #Data dictionary Keys
cell_name = Data_Dictionary[var_name[0]].columns.tolist() #Global cell name taken from PDSI column names
cell_name = [cell_name[i][cell_name[i].find('c'):] for i in range(len(cell_name))]
for i, key in enumerate(Data_Dictionary): #Renaming columns with clean cell name
    Data_Dictionary[key].columns = cell_name     
    
    
pickle_name = root + 'Complete_Timeseries' + '.pickle'
#with open(pickle_name, 'wb') as handle:
    #pickle.dump(Data_Dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


