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
    dist_to_ref =  testing_angles - ref_angle
    targ_180 = testing_angles - dist_to_ref
    dist_180_ = testing_distractors - dist_to_ref
    dist_180 = []
    for n_dist in dist_180_:
        if n_dist<0:
            n_ = n_dist+360
            dist_180.append(n_)
        elif n_dist>360:
            n_ = n_dist-360
            dist_180.append(n_)
        else:
            dist_180.append(n_dist)
    ##
    dist_180 = np.array(dist_180)
    ##
    ##
    for i in range(len(Channel_all_trials_rolled)):
        Trial_reconstruction = Channel_all_trials_rolled[i,:]
        _135_ = ref_angle*2 - 45*2 
        _225_ = ref_angle*2 + 45*2 
        Trial_135_225 = Trial_reconstruction[_135_:_225_]






    ##df = pd.DataFrame()
    #n = list(Channel_all_trials_rolled.mean(axis=0)) #mean of all the trials rolled
    #df['TR'] = n #Name of the column
    return Channel_all_trials_rolled




def decode(activity):
    N=len(activity)
    R = []
    angles = np.radians(np.linspace(135,224,180) ) 
    R=np.dot(activity,np.exp(1j*angles)) / N
    angle = np.angle(R)
    if angle < 0:
        angle +=2*np.pi 
    return np.degrees(angle)




decode(Trial_135_225)




Trial_reconstruction = Channel_all_trials_rolled[1,:]
_135_ = ref_angle*2 - 45*2 
_225_ = ref_angle*2 + 45*2 
Trial_135_225 = Trial_reconstruction[_135_:_225_]

activity = Trial_135_225
activity[80:100]=9999

N=len(activity)
R = []
angles = np.radians(np.linspace(135,224,180) ) 
R=np.dot(activity,np.exp(1j*angles)) / N
angle = np.angle(R)
if angle < 0:
    angle +=2*np.pi 

np.degrees(angle)

