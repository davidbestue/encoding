# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:16:21 2019

@author: David Bestue
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model_functions import *
from joblib import Parallel, delayed
import multiprocessing
from scipy import stats




def trial_rep(Signal, angle_trial, Weights, Weights_t, ref, intercept_):
    ###
    channel_36 = np.dot( np.dot ( np.linalg.pinv( np.dot(Weights_t, Weights ) ),  Weights_t),  Signal) #Run the inverse model
    ###
    if intercept_==True:     
        channel= ch2vrep3_int(channel_36)
    else: #no intercept in the Weights matrix
        channel= ch2vrep3(channel_36) ###Convert 36 into 720 channels for the reconstruction
    ##
    to_roll = int( (ref - angle_trial)*(len(channel)/360) ) ## degrees to roll
    channel=np.roll(channel, to_roll) ## roll this degrees
    return channel



def Representation(testing_data, testing_angles, Weights, Weights_t, ref_angle=180, plot=False, intercept=False):
    ## Make the data parallelizable
    n_trials_test = len(testing_data) #number trials
    data_prall = []
    for i in range(n_trials_test):
        data_prall.append(testing_data[i, :])
        #data_prall.append(    np.array( stats.zscore(    testing_data[i, :] ))   ) ###what enters the formula is zscored!
        
        
    
    ###
    numcores = multiprocessing.cpu_count()
    Channel_all_trials_rolled = Parallel(n_jobs = numcores)(delayed(trial_rep)(Signal, angle_trial, Weights, Weights_t, ref=ref_angle, intercept_ = intercept)  for Signal, angle_trial in zip( data_prall, testing_angles))    ####
    Channel_all_trials_rolled = np.array(Channel_all_trials_rolled)
    
    df = pd.DataFrame()
    n = list(Channel_all_trials_rolled.mean(axis=0)) #mean of all the trials rolled
    df['TR'] = n #Name of the column
    if plot==True:
        #Plot heatmap
        plt.figure()
        plt.title('Heatmap decoding')
        ######midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
        ax = sns.heatmap(df, yticklabels=list(df.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
        ax.plot([0.25, np.shape(df)[1]-0.25], [posch1_to_posch2(18),posch1_to_posch2(18)], 'k--')
        plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
        plt.ylabel('Angle')
        plt.xlabel('time (s)')
        plt.show(block=False)
    
    return df


#Representation(testing_data, testing_angles, WM, WM_t, ref_angle=180, plot=True)
        
