# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:52:46 2019

@author: David Bestue
"""


import numpy as np
import pandas as pd
from nilearn.masking import apply_mask
from nitime.timeseries import TimeSeries
from nitime.analysis import FilterAnalyzer


def ub_wind_path(PATH, system):
    if system=='wind':
        A = PATH                                                                                                    
        B = A.replace('/', os.path.sep)                                                                                               
        C= B.replace('\\mnt\\c\\', 'C:\\') 
    
    if system=='unix':
        C=PATH
    
    ###
    return C


def mask_fmri(fmri_path, masks, beh_path, sys_use='unix'):
    ### Inputs: 
    ###### fmri_paths: list of paths
    ###### beh_paths: list of paths
    ###### masks: [rh_mask, lh_mask]
    ###### sys_use (unix or windows: to change the paths)
    ###### hd hemodynamic delay (seconds)
    ###### TR=2.335 (fixed)
    
    ## Processes: 
    ###### 1. Load and mask the data
    ###### 2. Process encoding data
    ##
    ### 1. Load and mask the data
    fmri_path = ub_wind_path(fmri_path, system=sys_use) #change the path format wind-unix
    
    mask_img_rh= masks[0] #right hemisphere mask
    mask_img_rh = ub_wind_path(mask_img_rh, system=sys_use)
    mask_img_lh= masks[1] #left hemisphere mask
    mask_img_lh = ub_wind_path(mask_img_lh, system=sys_use)
    
    #Apply the masks and concatenate   
    masked_data_rh = apply_mask(fmri_path, mask_img_rh)
    masked_data_lh = apply_mask(fmri_path, mask_img_lh)    
    masked_data=np.hstack([masked_data_rh, masked_data_lh])
    
    #append it and save the data
    return masked_data     
        