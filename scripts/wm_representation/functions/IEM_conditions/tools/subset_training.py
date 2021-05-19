# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""
import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def subset_training(training_activity, training_behaviour, training_item, training_time, tr_st, tr_end):
    ####
    ####
    #### get the specific TRs tu pass to generate the matrix of weights and the specific behaviour
    ####
    if training_time=='stim_p':
        delay_TR_cond = training_activity[:, tr_st, :]
    if training_time=='delay':
        delay_TR_cond = np.mean(training_activity[:, tr_st:tr_end, :], axis=1) ## training_activity[:, 8, :]
    if training_time=='respo':
        delay_TR_cond = training_activity[:, tr_st, :]
    #
    #
    training_thing = training_behaviour[training_item]
    ####
    return delay_TR_cond, training_thing


