# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def Representation_angle(testing_data, testing_angles, testing_distractors, Weights, Weights_t, ref_angle=180,  intercept=False):
    ## Make the data parallelizable
    n_trials_test = len(testing_data) #number trials
    data_prall = []
    for i in range(n_trials_test):
        data_prall.append(testing_data[i, :])
    ###
    ###
    numcores = multiprocessing.cpu_count()
    Channel_all_trials_rolled = Parallel(n_jobs = numcores)(delayed(trial_rep)(Signal, angle_trial, Weights, Weights_t, ref=ref_angle, intercept_ = intercept)  for Signal, angle_trial in zip( data_prall, testing_angles))    ####
    Channel_all_trials_rolled = np.array(Channel_all_trials_rolled)
    
    ##
    dist_to_ref =  y_test - ref_angle
    targ_180 = y_test - dist_to_ref
    dist_180 = 


    ##df = pd.DataFrame()
    #n = list(Channel_all_trials_rolled.mean(axis=0)) #mean of all the trials rolled
    #df['TR'] = n #Name of the column
    return Channel_all_trials_rolled

