# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:14:46 2019

@author: David Bestue
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from model_functions import f, pos_channels
import time
import statsmodels.api as sm
from scipy.stats import zscore



def Weights_matrix_LM_i_zs( training_data, training_angles ):
    #####
    ## function to get Weight matrix
    ## trainiing data (trials, vx)
    ## training_angles (trials)
    start_train_weights = time.time()
    #####
    n_voxels = np.shape(training_data)[1]
    
    ### Expected activity from the model
    M_model=[] #matrix of the activity from the model
    for i in training_angles:
        channel_values=f(i)  #f #f_quadrant (function that generates the expectd reponse in each channel)
        M_model.append(channel_values)
        
    M_model=pd.DataFrame(np.array(M_model)) # (trials, channel_activity)
    channel_names = ['ch_' +str(i+1) for i in range(0, len(pos_channels))] #names of the channels 
    M_model.columns=channel_names
    
    ####   2. Train the model and get matrix of weights
    Matrix_weights=np.zeros(( n_voxels, len(pos_channels) )) # (voxels, channels) how each channels is represented in each voxel
    M_model_zscored = zscore(M_model, axis=1) ## Standarize
    for voxel_x in range(0, n_voxels): #train each voxel
        # set Y and X for the GLM
        Y = training_data[:, voxel_x] ## Y is the real activity
        X = M_model_zscored ## X is the hipothetycal activity 
        ###
        X = sm.add_constant(X)
        a = sm.OLS(Y, X )
        resul = a.fit()
        #print(resul.params[0])
        betas= resul.params[1:]
        Matrix_weights[voxel_x, :]=betas
        
        #### Lasso with penalization of 0.0001 (higher gives all zeros), fiting intercept (around 10 ) and forcing the weights to be positive
        #Y = training_data[:, voxel_x] ## Y is the real activity
        #X = M_model## X is the hipothetycal activity (no seed to standarize, normalize of lasso does it)
        #lin = Lasso(alpha=0.001, precompute=True,  fit_intercept=True,  positive=True, normalize=True, selection='random')   
        #lin.fit(X,Y) # fits the best combination of weights to explain the activity
        #betas = lin.coef_ #ignore the intercept and just get the weights of each channel
        #Matrix_weights[voxel_x, :]=betas #save the 36 weights for each voxel
    
    
    #####
    
    #Save the matrix of weights 
    Matrix_weights =pd.DataFrame(Matrix_weights) #convert the array to dataframe
    end_train_weights = time.time()
    process_train_weights = end_train_weights - start_train_weights
    print( 'Time train Weights: ' +str(process_train_weights))    
    return Matrix_weights







def Weights_matrix_LM( training_data, training_angles ):
    #####
    ## function to get Weight matrix
    ## trainiing data (trials, vx)
    ## training_angles (trials)
    start_train_weights = time.time()
    #####
    n_voxels = np.shape(training_data)[1]
    
    ### Expected activity from the model
    M_model=[] #matrix of the activity from the model
    for i in training_angles:
        channel_values=f(i)  #f #f_quadrant (function that generates the expectd reponse in each channel)
        M_model.append(channel_values)
        
    M_model=pd.DataFrame(np.array(M_model)) # (trials, channel_activity)
    channel_names = ['ch_' +str(i+1) for i in range(0, len(pos_channels))] #names of the channels 
    M_model.columns=channel_names
    
    ####   2. Train the model and get matrix of weights
    Matrix_weights=np.zeros(( n_voxels, len(pos_channels) )) # (voxels, channels) how each channels is represented in each voxel
    #M_model_zscored = zscore(M_model, axis=0) ## standarize if runing the liniar regression
    
    for voxel_x in range(0, n_voxels): #train each voxel
        # set Y and X for the GLM
        Y = training_data[:, voxel_x] ## Y is the real activity
        X = M_model #_zscored #M_model ## X is the hipothetycal activity 
        ##
        #X = sm.add_constant(X)
        a = sm.OLS(Y, X )
        resul = a.fit()
        #print(resul.params[0])
        betas= resul.params[1:]
        Matrix_weights[voxel_x, :]=betas
        
        #### Lasso with penalization of 0.0001 (higher gives all zeros), fiting intercept (around 10 ) and forcing the weights to be positive
#        Y = training_data[:, voxel_x] ## Y is the real activity
#        X = M_model ## X is the hipothetycal activit
#        lin = Lasso(alpha=0.001, precompute=True,  fit_intercept=True,  positive=True, normalize=True, selection='random')   
#        lin.fit(X,Y) # fits the best combination of weights to explain the activity
#        betas = lin.coef_ #ignore the intercept and just get the weights of each channel
#        Matrix_weights[voxel_x, :]=betas #save the 36 weights for each voxel
#    
    
    #####
    
    #Save the matrix of weights 
    Matrix_weights =pd.DataFrame(Matrix_weights) #convert the array to dataframe
    end_train_weights = time.time()
    process_train_weights = end_train_weights - start_train_weights
    print( 'Time train Weights: ' +str(process_train_weights))   
    
    return Matrix_weights






