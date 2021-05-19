# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""
import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def IEM(testing_activity, testing_behaviour, decode_item, WM, WM_t, Inter, tr_st, tr_end):
    ####
    ####
    #### IEM: Inverted encoding model
    #### no cv: no cross validation. Different training and testing datasets
    ####  
    #### I use this function to run the IEM between condition
    #### Notice that here we need the WM and WM transposed (weights matrix) so we can not modify the training procedure
    #### 
    ##### decode_item (decide what you try to decode)
    ##### decode_item = 'T_alone', 'T'
    ##### decode_item = 'dist_alone' , 'Dist'  
    ####
    #### Get the Trs with shared information and the TRs without shared information
    list_wm_scans= range(nscans_wm)  
    ####
    #### Get the angles of the decoding item
    testing_angles = np.array(testing_behaviour[decode_item])   
    ####
    ### Respresentation
    signal_paralel =[ testing_activity[:, i, :] for i in list_wm_scans ]
    Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_paralel)    #### reconstruction standard (paralel)
    Reconstruction= pd.concat(Reconstructions, axis=1) #mean of the reconstructions (all trials)
    Reconstruction.columns =  [str(i * TR) for i in list_wm_scans ]    ##column names
    ####
    return Reconstruction


