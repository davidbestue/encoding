# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def trial_rep_decode_trial_by_trial(Signal, angle_trial, Weights, Weights_t, ref, intercept_):
    ###
    channel_36 = np.dot( np.dot ( np.linalg.pinv( np.dot(Weights_t, Weights ) ),  Weights_t),  Signal) #Run the inverse model
    ###
    if intercept_==True:     
        channel= ch2vrep3_int(channel_36)
    else: #no intercept in the Weights matrix
        channel= ch2vrep3(channel_36) ###Convert 36 into 720 channels for the reconstruction
    ##
    channel_dec =  channel * f2(angle_trial)
    #to_roll = int( (ref - angle_trial)*(len(channel)/360) ) ## degrees to roll
    #channel=np.roll(channel, to_roll) ## roll this degrees
    return channel_dec ### in noise you will also have a peak here, the point is that with signal will be higher (reconstruct by substracting noise here)

