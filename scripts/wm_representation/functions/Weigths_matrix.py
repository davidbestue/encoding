# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:14:46 2019

@author: David Bestue
"""


import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


#Generate the positions of the channels (there will be 14)
sep_channels=10
adjusted_size_contant = 48.519
pos_channels = np.arange(sep_channels/2,360,sep_channels)
pos_channels = [round(pos_channels[i],3) for i in range(0, len(pos_channels))]


def circ_dist(a1,a2):
    ## Returns the minimal distance in angles between to angles 
    op1=abs(a2-a1)
    angs=[a1,a2]
    op2=min(angs)+(360-max(angs))
    options=[op1,op2]
    return min(options)



def f(position_target):
    #I want to return a list of the activity of each channel in front of a stim at any location
    #
    #The f function imput is the distance from the position to the channel. That is why first we need to
    #get a distance from the locaion to each channel
    #
    #First i calculate the distance in degrees from the location to each channel
    #Once I have the distance value, I use the same formula as Sprague to extract a value of f for each channel.
    #colculate the r : the circular distance between target position and each channel
    list_r=[]
    for channel in pos_channels:
        R = round(circ_dist(position_target, channel), 3)
        list_r.append(R)
    
    #I need the adjusted because the r is not in visual angles, it is in degrees
    #I calculate the f for those inside the spread of the maximum, farther, it is 0
    f_list=[]
    for r in list_r:
        if r<adjusted_size_contant:
            #f = ( 0.5 + 0.5*cos(r*pi/adjusted_size_contant) )
            f = ( 0.5 + 0.5*np.cos(r*pi/adjusted_size_contant) )**7
            f=round(f,3)
            f_list.append(f)
        else:
            f = 0
            f_list.append(f)
    
    
    #Return the list   
    return f_list




#### function to get Weight matrix
## trainiing data (trials, vx)
## training_angles (trials)


def Weights_matrix( training_data, training_angles ):
    #####
    n_voxels = np.shape(training_data)[0]
    
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
    
    for voxel_x in range(0, n_voxels): #train each voxel
        # set Y and X for the GLM
        Y = training_data[:, voxel_x] ## Y is the real activity
        X = M_model ## X is the hipothetycal activity 
        
        #### Lasso with penalization of 0.0001 (higher gives all zeros), fiting intercept (around 10 ) and forcing the weights to be positive
        lin = Lasso(alpha=0.001, precompute=True,  fit_intercept=True,  positive=True, selection='random')   
        lin.fit(X,Y) # fits the best combination of weights to explain the activity
        betas = lin.coef_ #ignore the intercept and just get the weights of each channel
        Matrix_weights[voxel_x, :]=betas #save the 36 weights for each voxel
    
    
    #####
    
    #Save the matrix of weights 
    Matrix_weights =pd.DataFrame(Matrix_weights) #convert the array to dataframe
    
    return Matrix_weights