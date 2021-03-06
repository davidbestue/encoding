# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


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
    #Channel_all_trials_rolled = Parallel(n_jobs = numcores)(delayed(trial_rep_decode_trial_by_trial)(Signal, angle_trial, Weights, Weights_t, ref=ref_angle, intercept_ = intercept)  for Signal, angle_trial in zip( data_prall, testing_angles))    ####
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

