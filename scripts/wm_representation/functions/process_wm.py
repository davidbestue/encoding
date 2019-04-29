# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""


import numpy as np
import pandas as pd
from nilearn.masking import apply_mask
from nitime.timeseries import TimeSeries
from nitime.analysis import FilterAnalyzer
from joblib import Parallel, delayed
import multiprocessing
from scipy import stats



wm_fmri_paths = [root +'n001/WMtask/s01/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r04/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r05/nocfmri5_task_Ax.nii',
                 root +'n001/WMtask/s02/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r04/nocfmri5_task_Ax.nii',
                 root +'n001/WMtask/s03/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r04/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r05/nocfmri5_task_Ax.nii']


wm_beh_paths=[root +'n001/WMtask/s01/r01/wm_beh.txt', root +'n001/WMtask/s01/r02/wm_beh.txt', root +'n001/WMtask/s01/r03/wm_beh.txt', root +'n001/WMtask/s01/r04/wm_beh.txt', root +'n001/WMtask/s01/r05/wm_beh.txt',
              root +'n001/WMtask/s02/r01/wm_beh.txt', root +'n001/WMtask/s02/r02/wm_beh.txt', root +'n001/WMtask/s02/r03/wm_beh.txt', root +'n001/WMtask/s02/r04/wm_beh.txt',
              root +'n001/WMtask/s03/r01/wm_beh.txt', root +'n001/WMtask/s03/r02/wm_beh.txt', root +'n001/WMtask/s03/r03/wm_beh.txt', root +'n001/WMtask/s03/r04/wm_beh.txt', root +'n001/WMtask/s03/r05/wm_beh.txt']






def ub_wind_path(PATH, system):
    if system=='wind':
        A = PATH                                                                                                    
        B = A.replace('/', os.path.sep)                                                                                               
        C= B.replace('\\mnt\\c\\', 'C:\\') 
    
    if system=='unix':
        C=PATH
    
    ###
    return C




def mask_fmri_process(fmri_path, masks, sys_use='unix'):
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
    
    ### 2. Filter and zscore
    n_voxels = np.shape(masked_data)[1]
    for voxel in range(0, n_voxels):
        data_to_filter = masked_data[:,voxel]                        
        #apply the filter 
        data_to_filter = TimeSeries(data_to_filter, sampling_interval=2.335)
        F = FilterAnalyzer(data_to_filter, ub=0.15, lb=0.02)
        data_filtered=F.filtered_boxcar.data
        masked_data[:,voxel] = data_filtered                        
        #Z score
        masked_data[:,voxel] = np.array( stats.zscore( masked_data[:,voxel]  ) ) + 10 ; ## zscore + 5 just to get + values
    
    #append it and save the data
    return masked_data    




numcores = multiprocessing.cpu_count()
all_masked= Parallel(n_jobs = numcores)(delayed(mask_fmri_process)(fmri_path, masks, sys_use='unix')  for fmri_path in wm_fmri_paths)    ####
scans_enc_runs = [len(all_masked[r]) for r in range(len(all_masked)) ]




def get_enc_timestamps_targets(beh_path, n_scans, sys_use='unix', hd=6, TR=2.335):
    #### get the delay timestamps of the fmri data & the target location of the run
    beh_path = ub_wind_path(beh_path, system=sys_use) #change the path format wind-unix
    behaviour=np.genfromtxt(beh_path, skip_header=1) ## load the file
    p_target = np.array(behaviour[:-1,4]) ## Get the position (hypotetical channel coef)
    ref_time=behaviour[-1, 1] ## Reference time (start scanner - begiinging of recording)
    st_delay = behaviour[:-1, 11] -ref_time #start of the delay time & take off the reference from
    start_delay_hdf = st_delay + hd # add some seconds for the hemodynamic delay     
    start_delay_hdf_scans = start_delay_hdf/TR  #convert seconds to scans (number of scan to take)
    timestamps = [  int(round(  start_delay_hdf_scans[n] ) ) for n in range(0, len(start_delay_hdf_scans) )] #make it an integrer
    #In case  the last one has no space, exclude it (and do the same for the ones of step 1, lin step 3 you will combie and they must have the same length)
    #you short the timestamps and the matrix fro the hipotetical cannel coefici
    while timestamps[-1]>n_scans-2:
        timestamps=timestamps[:-1] ##  1st scan to take in each trial
        p_target = p_target[:-1] ## targets of the session (append to the genearl at the end)
    
    return p_target, timestamps

















