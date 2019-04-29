# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:24:52 2019

@author: David Bestue
"""

import numpy as np
import pandas as pd
from nilearn.masking import apply_mask
from nitime.timeseries import TimeSeries
from nitime.analysis import FilterAnalyzer
from joblib import Parallel, delayed
import multiprocessing


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



scans_enc_sess

def get_enc_info(beh_path, n_scans_sess, sys_use='unix', hd=6, TR=2.335):
    #### get the timestamps of the fmri data & the target location of the run
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
    while timestamps[-1]>n_scans_sess-2:
        timestamps=timestamps[:-1] ##  1st scan to take in each trial
        p_target = p_target[:-1] ## targets of the session (append to the genearl at the end)
    
    return p_target, timestamps

    





def process_encoding_files(fmri_paths, masks, beh_paths, sys_use='unix', hd=6, TR=2.335):
    ### Inputs: 
    ###### fmri_paths: list of paths
    ###### beh_paths: list of paths
    ###### masks: [rh_mask, lh_mask]
    ###### sys_use (unix or windows: to change the paths)
    ###### hd hemodynamic delay (seconds)
    ###### TR=2.335 (fixed)
    
    ### Outputs
    Training_dataset_activity=[]
    Training_dataset_targets=[]
    
    ## Processes: 
    ###### 1. Load and mask the data
    ###### 2. Process encoding data
    ##
    ### 1. Load and mask the data of all sessions     
    numcores = multiprocessing.cpu_count()
    all_data_masked= Parallel(n_jobs = numcores)(delayed(mask_fmri)(fmri_path, masks, sys_use='unix')  for fmri_path in fmri_paths)    ####
    scans_enc_sess = [len(all_data_masked[r]) for r in range(len(all_data_masked)) ]
    
    
    ### 2. timestamps and beh targets
    
    Channel_all_trials_rolled = Parallel(n_jobs = numcores)(delayed(get_enc_info)((beh_path, n_scans_sess, sys_use='unix', hd=6, TR=2.335))  for beh_path, n_scans_sess in zip( beh_paths, scans_enc_sess))    ####
    Channel_all_trials_rolled = np.array(Channel_all_trials_rolled)
    
    
    
    numcores = multiprocessing.cpu_count()
    all_data_masked= Parallel(n_jobs = numcores)(delayed(mask_fmri)(fmri_path, masks, sys_use='unix')  for fmri_path in fmri_paths)    ####
    scans_enc_sess = [len(all_data_masked[r]) for r in range(len(all_data_masked)) ]
    
    
    
    ####
    ### 2. Process encoding data
    ###### In each session I will:
        ####   1. Select the times corresponding to the delay (2TR), append the target of each trial 
        ####   2. Apply a filter for each voxel
        ####   3. Subset of data corresponding to the delay times (all voxels)
        ####   4. zscore + 10 in each voxel in the temporal dimension (with the other 2TR of the same session)
        ####   5. append activity and targets of the session
    ####
    ###### Concatenate all the sessions (targets and activity) to create the training dataset
    Training_dataset_activity=[] ##  activity for all the trials (all the sessions) (trials, voxels)
    Training_dataset_targets=[] ##  targets for all the trials (all the sessions) (trials)
    n_voxels = np.shape(encoding_datasets[0])[1] ## number of voxels
    
    for run in range(len(fmri_paths)):
        ### 1. Select the times corresponding to the delay (2TR), append the target of each trial
        Beh_enc_file = beh_paths[run] #file encoding behaviour
        Beh_enc_file = ub_wind_path(Beh_enc_file, system=sys_use) #change the path format wind-unix
        behaviour=np.genfromtxt(Beh_enc_file, skip_header=1) ## load the file
        p_target = np.array(behaviour[:-1,4]) ## Get the position (hypotetical channel coef)
        ref_time=behaviour[-1, 1] ## Reference time (start scanner - begiinging of recording)
        st_delay = behaviour[:-1, 11] -ref_time #start of the delay time & take off the reference from
        start_delay_hdf = st_delay + hd # add some seconds for the hemodynamic delay     
        start_delay_hdf_scans = start_delay_hdf/TR  #convert seconds to scans (number of scan to take)
        timestamps = [  int(round(  start_delay_hdf_scans[n] ) ) for n in range(0, len(start_delay_hdf_scans) )] #make it an integrer
        #In case  the last one has no space, exclude it (and do the same for the ones of step 1, lin step 3 you will combie and they must have the same length)
        #you short the timestamps and the matrix fro the hipotetical cannel coefici
        while timestamps[-1]>  len(encoding_datasets[run])-2:
            timestamps=timestamps[:-1] ##  1st scan to take in each trial
            p_target = p_target[:-1] ## targets of the session (append to the genearl at the end)
        
        
        ####   2. Apply a filter for each voxel               
        for voxel in range(0, n_voxels ):
            data_to_filter = encoding_datasets[run][:,voxel] #data of the voxel along the session
            #apply the filter 
            data_to_filter = TimeSeries(data_to_filter, sampling_interval=TR)
            F = FilterAnalyzer(data_to_filter, ub=0.15, lb=0.02) ##upper and lower boundaries
            data_filtered=F.filtered_boxcar.data
            encoding_datasets[run][:,voxel] = data_filtered ## replace old data with the filtered one.
        
        
        ####   3. Subset of data corresponding to the delay times (all voxels)
        encoding_delay_activity = np.zeros(( len(timestamps), n_voxels)) ## emply matrix (n_trials, n_voxels)
        for idx,t in enumerate(timestamps): #in each trial
            delay_TRs =  encoding_datasets[run][t:t+2, :] #take the first scan of the delay and the nex
            delay_TRs_mean =np.mean(delay_TRs, axis=0) #make the mean in each voxel of 2TR
            encoding_delay_activity[idx, :] =delay_TRs_mean #index the line in the matrix
        
        
        ####   4. zscore + 10 in each voxel in the temporal dimension (with the other 2TR of the same session)
        for vxl in range(0, n_voxels ): # by voxel
            vx_act = encoding_delay_activity[:, vxl]
            vx_act_zs = np.array( zscore(vx_act) ) +10 ; ## zscore + 10 just to get + values
            encoding_delay_activity[:, vxl] = vx_act_zs  ## replace previos activity
        
        
        ####   5. append activity and targets of the session
        p_target = list(p_target) ### make a list that will be added to another list
        Training_dataset_targets.extend(p_target) ## append the position of the target for the trial    
        Training_dataset_activity.append(encoding_delay_activity) ## append the activity used for the training
    
    
    ##### Concatenate sessions to create Trianing Dataset  ### ASSUMPTION: each voxel is the same across sessions!               
    Training_dataset_activity = np.vstack(Training_dataset_activity) #make an array (n_trials(all sessions together), voxels)
    Training_dataset_targets = np.array(Training_dataset_targets) ## make an array (trials, 1)
    
    return Training_dataset_activity, Training_dataset_targets
    
    