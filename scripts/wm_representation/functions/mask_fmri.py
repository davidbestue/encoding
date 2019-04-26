# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:52:46 2019

@author: David Bestue
"""


##### remove when process of encoding is complete!

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


def mask_fmri(fmri_path, masks, sys_use='unix'):
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



mri_n = '/home/david/Desktop/IEM_data/n001/encoding/s01/r01/nocfmri3_Encoding_Ax.nii'
masks_n = ['/home/david/Desktop/IEM_data/temp_masks/n001/visual_fsign_rh.nii.gz', '/home/david/Desktop/IEM_data/temp_masks/n001/visual_fsign_lh.nii.gz']
mask_fmri(mri_n, masks_n)



root= '/home/david/Desktop/IEM_data/'
files_n= [root +'n001/encoding/s01/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r04/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r05/nocfmri3_Encoding_Ax.nii',
          root +'n001/encoding/s02/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r04/nocfmri3_Encoding_Ax.nii',
          root +'n001/encoding/s03/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r04/nocfmri3_Encoding_Ax.nii']



#paralel
from joblib import Parallel, delayed
import multiprocessing

numcores = multiprocessing.cpu_count()
all_masked= Parallel(n_jobs = numcores)(delayed(mask_fmri)(fmri_path, masks_n, sys_use='unix')  for fmri_path in files_n)    ####
np.shape(all_masked)







