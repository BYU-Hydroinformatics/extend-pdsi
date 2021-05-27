# -*- coding: utf-8 -*-
"""
Created on Mon May 24 23:46:27 2021

@author: saulg
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import traceback
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import keras.callbacks
import HydroErr as he
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from tqdm import tqdm
from copy import deepcopy
import tensorflow as tf

##############################
def read_well_pickle(data_file):
    with open(data_file, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary

def create_data_table(cell_name, var_name, data_dictionary):
    id_cell = cell_name.split('_')[1]
    model_data = pd.DataFrame()
    for i, var in enumerate(var_name):
        id_var = var.split('_')[0]
        id_var = id_var + '_' + id_cell
        var_dataframe = pd.DataFrame(data_dictionary[var][id_var])
        model_data = Data_Join(model_data, var_dataframe)
    return model_data

def Data_Split(Data, target, Shuffle=False):
    if Shuffle:
         Data = Data.sample(frac=1) #The frac keyword argument specifies the fraction of rows to return in the random sample
    Y = Data[target].to_frame()
    X = Data.drop(target, axis=1)
    return Y, X

def Data_Join(pd1, pd2, method='outer', axis=1):
    return pd.concat([pd1, pd2], join='outer', axis=1)

def Rolling_Window(Feature_Data, Names, years=[1, 3, 5, 10]):
        newnames = deepcopy(Names)  # names is a list of the varibiles in the data frame, need to unlink for loop
        # This loop adds the yearly, 3-year, 5-year, and 10-year rolling averages of each variable to the dataframe
        # rolling averages uses the preceding time to compute the last value, e.g., using the preceding 5 years of data to get todays
        for name in Names:
            for year in years:
                new = name + '_rw_yr_' + str(year).zfill(2)
                Feature_Data[new] = Feature_Data[name].rolling(year * 12).mean()
        return Feature_Data.dropna()
        
def Mean_Error_Metric(y_true, y_pred):
    mean_error = tf.reduce_mean(y_true - y_pred, axis=-1)
    return mean_error
##############################


file_name = 'Timeseries_Complete_Upsampled.pickle'
Data = read_well_pickle('./' + file_name)
var_name = sorted(list(Data.keys()))
cell_name = Data[var_name[0]].columns

#rand_int = np.random.randint(0, len(cell_name))
#rand_int = 2036
rand_int = 860
norm_method = 1
hidden_nodes = 1000
debug_cells = 5
cell_name_debug = cell_name[rand_int: rand_int + debug_cells]

np.random.seed(42)
Error_DataFrame = pd.DataFrame(index= cell_name, columns=['MSE','RMSE','ME'])
#for i, cell in enumerate(tqdm(cell_name)):
for i, cell in enumerate(tqdm(cell_name_debug)):
    try:
        ###### Matrix Creation
        data_temp = create_data_table(cell, var_name, Data)
        target_name = data_temp.columns[0]
        
        ###### Splitting and Augmentation Data
        Y_truth, X = Data_Split(data_temp, target_name)
        X = Rolling_Window(X, X.columns, years=[1, 3, 5, 10])
        data_temp = Data_Join(Y_truth, X)
        data_temp = data_temp[data_temp[data_temp.columns[1]].notnull()]
        
        ###### Data Scaling
        target_scaler = MinMaxScaler()
        data_scaler = StandardScaler()
        X_predict = pd.DataFrame(data_scaler.fit_transform(X), 
                                    index=X.index, columns = X.columns)
        #Y_truth = pd.DataFrame(target_scaler.fit_transform(Y_truth), 
                                    #index=Y_truth.index, columns = Y_truth.columns)
        Data_Scaled = Data_Join(Y_truth, X_predict)
        Data_Prep = Data_Scaled.dropna()
        Y, X = Data_Split(Data_Prep, target_name)
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)

        
        model = Sequential()
        model.add(Dense(hidden_nodes, input_dim = Data_Scaled.shape[1]-1, activation = 'relu', use_bias=True,
                        kernel_initializer='glorot_uniform', kernel_regularizer=keras.regularizers.L2(l2=0.01))) #he_normal
        model.add(keras.layers.Dropout(rate=0.2))
        model.add(Dense(hidden_nodes, input_dim = Data_Scaled.shape[1]-1, activation = 'relu', use_bias=True,
                kernel_initializer='glorot_uniform', kernel_regularizer=keras.regularizers.L2(l2=0.01))) #he_normal glorot_uniform
        model.add(keras.layers.Dropout(rate=0.2))
        model.add(Dense(1))
        model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=[keras.metrics.RootMeanSquaredError(), Mean_Error_Metric])
        
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0, restore_best_weights=True)
        history = model.fit(x_train, y_train, epochs=300, validation_data = (x_val, y_val), verbose= 3, callbacks=[early_stopping])
        y_test_hat = model.predict(x_test)
        score = model.evaluate(x_test, y_test)
        
        
        Prediction = pd.DataFrame(model.predict(X_predict), 
                                 index=X_predict.index, columns = ['Prediction'])
        #Prediction = pd.DataFrame(target_scaler.inverse_transform(model.predict(X_predict)), 
                                  #index=X_predict.index, columns = ['Prediction'])

        Error_DataFrame.loc[cell] = score
        
        #if i % 50 == 0:
        if i % 1 == 0:
            pd.DataFrame(history.history).plot(figsize=(8,5))
            plt.grid(True)
            plt.gca().set_ylim(-0.25,25)
            plt.savefig('./Figures/' + str(cell) + '_Training')
            plt.show()
            print(model.summary())
            
            #Plotting Prediction Correlation
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            plt.scatter(y_test_hat, y_test)
            plt.ylabel('Observation')
            plt.xlabel('Prediction')
            plt.legend(['Prediction', 'Observation'])
            plt.title('Prediction Correlation: Cell ' + str(cell.split('_')[1]))
            limit_low = -10
            limit_high = 10
            cor_line_x = np.linspace(limit_low, limit_high, 9)
            cor_line_y = cor_line_x
            plt.xlim(limit_low, limit_high)
            plt.ylim(limit_low, limit_high)
            plt.plot(cor_line_x, cor_line_y, color='r')
            ax1.set_aspect('equal', adjustable='box')
            plt.savefig('./Figures/' + str(cell) + '_Correlation')
            plt.show()
            
            #Plot Results
            plt.figure(figsize=(12, 8))
            plt.plot(Prediction.index, Prediction['Prediction'], "r")
            plt.plot(data_temp.index, data_temp[cell], label= 'Observations', color='b')
            #plt.scatter(data_temp.index, data_temp[target_name], label= 'Observations', color='b')
            plt.ylabel('PDSI Index')
            plt.xlabel('Date')
            plt.legend(['Prediction', 'Observation'])
            plt.title('PDSI Extension: Cell ' + str(cell.split('_')[1]))
            plt.savefig('./Figures/' + str(cell) + '_Timeseries')
            plt.show()
            print('PDSI Extension Error is: ' + str(score))
            
    except Exception as e:
        print(traceback.format_exc())
    
Error_DataFrame.to_pickle('./Unregularized_Error.pickle')