# -*- coding: utf-8 -*-
"""
Created on Mon May 24 23:46:27 2021

@author: saulg
"""
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import traceback

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import datetime as dt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from tensorflow import reduce_mean
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.metrics import RootMeanSquaredError




##############################
def read_well_pickle(data_file):
    # Load pickle file that has been concatenated as dictionary containing
    # all data
    with open(data_file, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary

def cell_name_clean(c_name, vc=1):
    
    clean_names = []
    for i, var in enumerate(c_name):
        id_var = var.split('_')[vc] # Split Data dictionary column name to get variable name
        clean_names.append(id_var)
    return clean_names

def replace_cell_name(var_names, cell_name, clean_name=True):
    if clean_name:
        clean = cell_name_clean([cell_name], vc=1)
        cell_name = clean[0]
    var_names = var_names.to_list()
    var_names = [v.replace('_' + cell_name, '') for v in var_names]
    return var_names
        
def create_data_table(cell_name, var_name, data_dictionary):
    model_data = pd.DataFrame() #Create Dataframe
    # Iterate through Data dictionary and identify all variable with cell name
    # Mask all time series
    # Concatenate timeseries
    for i, var in enumerate(var_name):
        var_dataframe = pd.DataFrame(data_dictionary[var][cell_name].values, columns=[var], 
                                     index = data_dictionary[var][cell_name].index) # Mask time series
        model_data = Data_Join(model_data, var_dataframe) # Update Datafame
    return model_data

def Data_Split(Data, target, Shuffle=False):
    # Split an array into a dataset and target
    # Assumption is that target is first column in a pandas dataframe
    # Will return data as pandas dataframe
    if Shuffle:
         # The frac keyword argument specifies the fraction of rows 
         # to return in the random sample
         Data = Data.sample(frac=1) 
    Y = Data[target].to_frame()
    X = Data.drop(target, axis=1)
    return Y, X

def Data_Join(pd1, pd2, method='outer', axis=1):
    #Joins two pandas dataframes in an outter join
    return pd.concat([pd1, pd2], join='outer', axis=1)

def Rolling_Window(Feature_Data, Names, years=[1, 3, 5, 10]):
        # This loop adds the yearly, 3-year, 5-year, and 10-year rolling averages of each variable to the dataframe
        # rolling averages uses the preceding time to compute the last value, e.g., using the preceding 5 years of data to get todays
        # Causes the loss of X number of data points
        for name in Names:
            for year in years:
                new = name + '_rw_yr_' + str(year).zfill(2)
                Feature_Data[new] = Feature_Data[name].rolling(year * 12).mean()
        return Feature_Data.dropna()
        
def Mean_Error_Metric(y_true, y_pred):
    mean_error = reduce_mean(y_true - y_pred, axis=-1)
    return mean_error

def Model_Training_Metrics_plot(Data):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(-0.5, 10)
    plt.savefig('./Figures/' + str(cell) + '_Training')
    plt.show()
    print(model.summary())
    return

def Q_Q_plot(y_test_hat, y_test):
    #Plotting Prediction Correlation
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    plt.scatter(y_test_hat, y_test)
    plt.ylabel('Observation')
    plt.xlabel('Prediction')
    plt.legend(['Prediction', 'Observation'])
    plt.title('Prediction Correlation: Cell ' + str(cell))
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
    return

def Prediction_Vs_Truth_plot(Prediction_X, Prediction_Y, Truth_X, Truth_Y):
    #Plot Results
    plt.figure(figsize=(12, 8))
    plt.plot(Prediction_X, Prediction_Y, "r")
    plt.plot(Truth_X, Truth_Y, label= 'Observations', color='b')
    plt.ylabel('PDSI Index')
    plt.xlabel('Date')
    plt.legend(['Prediction', 'Observation'])
    plt.title('PDSI Extension: Cell ' + str(cell))
    plt.savefig('./Figures/' + str(cell) + '_Timeseries')
    plt.show()
    return

def Feature_Importance_plot(importance_df, cell):
    importance_df = importance_df.transpose()
    importance_df = importance_df.sort_values(by=[cell], ascending=True)
    importance_df.plot.barh(align ='center',width = 0.75, legend = None, figsize=(10,20))
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.title('Most Prevalent Features:')
    plt.savefig('./Figures/' + str(cell) + '_Feature_Importance')
    plt.show()
    return
        
def Feature_Importance_box_plot(importance_df):
    #All Data       
    importance_df.boxplot(figsize=(20,10))
    plt.xlabel('Feature')
    plt.ylabel('Relative Importance')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('./Figures/'+ str(cell) + '_Feature_Importance_Complete')
    plt.show()

    #Calc Mean and sort
    importance_mean_df = importance_df.mean()
    importance_mean_df = pd.DataFrame(importance_mean_df.sort_values(axis=0, ascending=False)).T
    importance_mean = importance_df.transpose().reindex(list(importance_mean_df.columns)).transpose()
    importance_mean.iloc[:,:10].boxplot(figsize=(5,5))
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Relative Importance')
    plt.title('Most Prevalent Features:')
    plt.tight_layout()
    plt.savefig('./Figures/' + 'Feature_Importance_Uppper')
    plt.show()
    
    #Lower
    importance_mean.iloc[:,importance_mean.shape[1]-10:].boxplot(figsize=(5,5))
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Relative Importance')
    plt.title('Least Prevalent Features:')
    plt.tight_layout()
    plt.savefig('./Figures/' + 'Feature_Importance_Lower')
    plt.show()
    return

##############################
##############################

file_name = './Data_Processed/Complete_Timeseries.pickle' #Combined Dataset Location
Data = read_well_pickle(file_name) #Data dictionary

var_name = list(Data.keys()) #Data dictionary Keys
cell_name = Data[var_name[0]].columns.tolist() #Global cell name taken from PDSI column names


#rand_int = 1 #865 # Hardcoded Random cell used in paper #865
#rand_int = np.random.randint(0, len(cell_name)) #Random cell to view
#debug_cells = 1 # Number of cells to look at for debuging
#cell_name_debug = cell_name[rand_int: rand_int + debug_cells] #Used to create a range of cells
hidden_nodes = 1000 # Hidden Layer size

np.random.seed(42) #Setting all random seeds to 42
Test_DataFrame = pd.DataFrame(index= cell_name, columns=['MSE','RMSE','ME']) #Creating Error Tracker
Error_Model_DataFrame = pd.DataFrame(index= cell_name, columns=['MSE','RMSE','ME']) #Creating Error Tracker

#Feature_Importance = pd.DataFrame() #Feature importance Tracker
PDSI_Extended = pd.DataFrame(columns=(Data['sc_PDSI_pm'].columns)) #Data Frame that will contain extrapolated values
PDSI_Prediction = pd.DataFrame(columns=(Data['sc_PDSI_pm'].columns))
PDSI_Extrapolation = pd.DataFrame(columns=(Data['sc_PDSI_pm'].columns))

for i, cell in enumerate(tqdm(cell_name)): #Uncommented when running all cells
#for i, cell in enumerate(tqdm(cell_name_debug)): #Comment when not dubugging
    try:
        ###### Matrix Creation
        # Create temporary data matrix by masking all variables on 
        data_temp = create_data_table(cell, var_name, Data)
        target_name = 'sc_PDSI_pm'
        
        ###### Splitting and Augmentation Data
        Y_truth, X = Data_Split(data_temp, target_name) # Sepearate for Augment
        X = Rolling_Window(X, X.columns, years=[1, 3, 5, 10]) # Augment
        data_temp = Data_Join(Y_truth, X) # Join for null removals
        data_temp = data_temp[data_temp[data_temp.columns[1]].notnull()] #Remove nulls only within training data
        
        ###### Data Scaling
        data_scaler = StandardScaler()
        X_predict = pd.DataFrame(data_scaler.fit_transform(X), index=X.index, columns = X.columns)
        Data_Scaled = Data_Join(Y_truth, X_predict)
        Data_Prep = Data_Scaled.dropna()
        
        ###### Data Split
        Y, X = Data_Split(Data_Prep, target_name)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42)

        ###### Model Initialization
        model = Sequential()
        model.add(Dense(hidden_nodes, input_dim = Data_Scaled.shape[1]-1, activation = 'relu', use_bias=True,
            kernel_initializer='glorot_uniform', kernel_regularizer=L2(l2=0.01))) #he_normal
        model.add(Dropout(rate=0.2))
        model.add(Dense(hidden_nodes, input_dim = Data_Scaled.shape[1]-1, activation = 'relu', use_bias=True,
            kernel_initializer='glorot_uniform', kernel_regularizer=L2(l2=0.01))) #he_normal
        model.add(Dropout(rate=0.2))
        model.add(Dense(1))
        model.compile(optimizer = Adam(learning_rate=0.001), loss='mse', metrics=[RootMeanSquaredError(), Mean_Error_Metric])
        
        ###### Hyper Paramter Adjustments
        early_stopping = EarlyStopping(monitor='val_loss', patience=7, min_delta=0.0, restore_best_weights=True)
        adaptive_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_lr=0)
        history = model.fit(x_train, y_train, epochs=500, validation_data = (x_val, y_val), verbose= 0, callbacks=[early_stopping, adaptive_lr])
        y_test_hat = model.predict(x_test)
        score_test = model.evaluate(x_test, y_test)
        score_model_cell = model.evaluate(X,Y)
        
        ###### Score and Tracking Metrics
        Test_DataFrame.loc[cell] = score_test
        Error_Model_DataFrame.loc[cell] = score_model_cell
        
        ###### Model Prediction
        Prediction = pd.DataFrame(model.predict(X_predict), index=X_predict.index, columns = ['Prediction'])
        
        # Include PDSI Value Update
        Gap_time_series = pd.DataFrame(Data['sc_PDSI_pm'][cell], index = Data['sc_PDSI_pm'][cell].index)
        Gap_time_series = Data_Join(Gap_time_series, Prediction['Prediction'])
        PDSI_Extended[cell] = Gap_time_series[cell].fillna(Prediction['Prediction'])
        PDSI_Prediction[cell] = Prediction['Prediction']
        PDSI_Extrapolation[cell] = Prediction.loc[dt.date(2019,1,1):dt.date(2020,12,1), 'Prediction']
        
        ###### Permutation Feature Importance
        #results = permutation_importance(model, x_test, y_test, n_repeats=5, random_state=42, scoring='neg_mean_squared_error')
        #importance_df = pd.DataFrame(results.importances_mean, index = x_test.columns, columns=([cell])).sort_index(ascending=True).transpose()
        #Feature_Importance = Feature_Importance.append(importance_df)

        #Model Monitoring and Plotting
        if i % 50 == 0:
            #Model_Training_Metrics_plot(history.history)
            Q_Q_plot(y_test_hat, y_test)
            #Prediction_Vs_Truth_plot(Prediction.index, Prediction['Prediction'], data_temp.index, data_temp['sc_PDSI_pm'])
            #Feature_Importance_plot(importance_df, cell)
            #Feature_Importance_box_plot(Feature_Importance)
            
    except:
        print(traceback.format_exc())
#Feature_Importance_box_plot(Feature_Importance)
#Feature_Importance.to_pickle('./Feature_Importance.pickle')    
Test_DataFrame.to_pickle('./Test_Metrics.pickle')
Error_Model_DataFrame.to_pickle('./Model_Error.pickle')
PDSI_Extended.to_pickle('./PDSI_Extended1948_2018.pickle')
PDSI_Prediction.to_pickle('./PDSI_Predictions1958_2018.pickle')
PDSI_Extrapolation.to_pickle('./PDSI_Predictions2019_2021.pickle')