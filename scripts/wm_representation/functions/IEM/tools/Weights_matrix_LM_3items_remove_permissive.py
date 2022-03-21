# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def Weights_matrix_LM_3items_remove_permissive( training_data, training_angles ):
    # no intercept
    # no regressors scaling
    # training_angles is a vector of [a1, a2, a3]
    #####
    start_train_weights = time.time()
    #####
    n_voxels = np.shape(training_data)[1]
    
    ### Expected activity from the model
    M_model0=[] #matrix of the activity from the model
    for i in range(len(training_angles)):
        channel_values1=f(training_angles[i][0])  #f #f_quadrant (function that generates the expectd reponse in each channel)
        channel_values2=f(training_angles[i][1])
        channel_values3=f(training_angles[i][2])
        channel_values = np.array(channel_values1) + np.array(channel_values2) + np.array(channel_values3)
        channel_values = list(channel_values)
        M_model0.append(channel_values)
        
    M_model1=pd.DataFrame(np.array(M_model0)) # (trials, channel_activity)
    channel_names = ['ch_' +str(i+1) for i in range(0, len(pos_channels))] #names of the channels 
    M_model1.columns=channel_names
    M_model = [['ch_2', 'ch_3', 'ch_4', 'ch_5', 'ch_6', 'ch_7', 'ch_8', 'ch_11','ch_12', 'ch_13', 'ch_14', 'ch_15', 'ch_16', 'ch_17', 
    'ch_20', 'ch_21', 'ch_22', 'ch_23', 'ch_24', 'ch_25', 'ch_26', 'ch_29','ch_30', 'ch_31', 'ch_32', 'ch_33', 'ch_34', 'ch_35']]
    ####   2. Train the model and get matrix of weights
    Matrix_weights=np.zeros(( n_voxels, len(pos_channels) )) # (voxels, channels) how each channels is represented in each voxel
    
    for voxel_x in range(0, n_voxels): #train each voxel
        # set Y and X for the GLM
        Y = training_data[:, voxel_x] ## Y is the real activity
        X = M_model #_zscored #M_model ## X is the hipothetycal activity 
        ##
        a = sm.OLS(Y, X )
        resul = a.fit()
        betas= resul.params
        Matrix_weights[voxel_x, :]=betas
    
    #Save the matrix of weights 
    Matrix_weights =pd.DataFrame(Matrix_weights) #convert the array to dataframe
    end_train_weights = time.time()
    process_train_weights = end_train_weights - start_train_weights
    ##print( 'Time train Weights: ' +str(process_train_weights))   
    Inter = False #intercept true or false
    
    return Matrix_weights, Inter


