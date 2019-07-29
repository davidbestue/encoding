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
import time


nscans_wm=16


#root= '/home/david/Desktop/IEM_data/'
#
#masks = [ root+'temp_masks/n001/visual_fsign_rh.nii.gz', root+ 'temp_masks/n001/visual_fsign_lh.nii.gz']
#
#wm_fmri_paths = [root +'n001/WMtask/s01/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r04/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r05/nocfmri5_task_Ax.nii',
#                 root +'n001/WMtask/s02/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r04/nocfmri5_task_Ax.nii',
#                 root +'n001/WMtask/s03/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r04/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r05/nocfmri5_task_Ax.nii']
#
#
#wm_beh_paths=[root +'n001/WMtask/s01/r01/wm_beh.txt', root +'n001/WMtask/s01/r02/wm_beh.txt', root +'n001/WMtask/s01/r03/wm_beh.txt', root +'n001/WMtask/s01/r04/wm_beh.txt', root +'n001/WMtask/s01/r05/wm_beh.txt',
#              root +'n001/WMtask/s02/r01/wm_beh.txt', root +'n001/WMtask/s02/r02/wm_beh.txt', root +'n001/WMtask/s02/r03/wm_beh.txt', root +'n001/WMtask/s02/r04/wm_beh.txt',
#              root +'n001/WMtask/s03/r01/wm_beh.txt', root +'n001/WMtask/s03/r02/wm_beh.txt', root +'n001/WMtask/s03/r03/wm_beh.txt', root +'n001/WMtask/s03/r04/wm_beh.txt', root +'n001/WMtask/s03/r05/wm_beh.txt']
#



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
    
    ### 2. Filter ####and zscore
    n_voxels = np.shape(masked_data)[1]
    for voxel in range(0, n_voxels):
        data_to_filter = masked_data[:,voxel]                        
        #apply the filter 
        data_to_filter = TimeSeries(data_to_filter, sampling_interval=2.335)
        F = FilterAnalyzer(data_to_filter, ub=0.15, lb=0.02)
        data_filtered=F.filtered_boxcar.data
        masked_data[:,voxel] = data_filtered                        
        #Z score
        masked_data[:,voxel] = np.array( stats.zscore( masked_data[:,voxel]  ) ) ; ## zscore + 5 just to get + values
    
    #append it and save the data
    return masked_data    



### Example
#numcores = multiprocessing.cpu_count()
#wm_masked= Parallel(n_jobs = numcores)(delayed(mask_fmri_process)(fmri_path, masks, sys_use='unix')  for fmri_path in wm_fmri_paths)    ####
#scans_wm_runs = [len(wm_masked[r]) for r in range(len(wm_masked)) ]
    


#### zscore
#### What enters must be zscored!
### Each column 


def condition_wm( activity, behaviour, condition, distance, zscore_=True):
    if distance=='mix':
        if condition == '1_0.2': 
            Subset = activity[  np.array(behaviour['delay1']==0.2)  *  np.array(behaviour['order']==1) , :, :]
            beh_Subset = behaviour.loc[(behaviour['delay1']==0.2) & (behaviour['order']==1)  ] 
          
        elif condition == '1_7':
            Subset = activity[  np.array(behaviour['delay1']==7)  *  np.array(behaviour['order']==1) , :, :]
            beh_Subset = behaviour.loc[(behaviour['delay1']==7) & (behaviour['order']==1)  ] 
            
        elif condition == '2_0.2':
            Subset = activity[  np.array(behaviour['delay1']==0.2)  *  np.array(behaviour['order']==2)   , :, :]
            beh_Subset = behaviour.loc[(behaviour['delay1']==0.2) & (behaviour['order']==2) ] 
          
        elif condition == '2_7':
            Subset = activity[  np.array(behaviour['delay1']==7)  *  np.array(behaviour['order']==2)  , :, :]
            beh_Subset = behaviour.loc[(behaviour['delay1']==7) & (behaviour['order']==2)  ] 
    
    
    else: ### close or far
        if condition == '1_0.2':
            Subset = activity[  np.array(behaviour['delay1']==0.2)  *  np.array(behaviour['order']==1) *  np.array(behaviour['type']==distance)  , :, :]
            beh_Subset = behaviour.loc[(behaviour['delay1']==0.2) & (behaviour['order']==1) & (behaviour['type']==distance) ] 
          
        elif condition == '1_7':
            Subset = activity[  np.array(behaviour['delay1']==7)  *  np.array(behaviour['order']==1) * np.array(behaviour['type']==distance)  , :, :]
            beh_Subset = behaviour.loc[(behaviour['delay1']==7) & (behaviour['order']==1) & (behaviour['type']==distance)  ] 
            
        elif condition == '2_0.2':
            Subset = activity[  np.array(behaviour['delay1']==0.2)  *  np.array(behaviour['order']==2) * np.array(behaviour['type']==distance)  , :, :]
            beh_Subset = behaviour.loc[(behaviour['delay1']==0.2) & (behaviour['order']==2) & (behaviour['type']==distance)  ] 
          
        elif condition == '2_7':
            Subset = activity[  np.array(behaviour['delay1']==7)  *  np.array(behaviour['order']==2) *  np.array(behaviour['type']==distance) , :, :]
            beh_Subset = behaviour.loc[(behaviour['delay1']==7) & (behaviour['order']==2) & (behaviour['type']==distance)  ]
    
    
    #####zscore
    ### Per voxel (problem of mixing session, not in encoding)
    ### Get just the scans_wm that I will use
    if zscore_ == True:
        n_voxels = np.shape(Subset)[2]
        nscans_wm = np.shape(Subset)[1]
        for sc_time in range(nscans_wm):
            for voxel in range(0, n_voxels):
                Subset[:, sc_time, voxel] =  np.array( stats.zscore(Subset[:, sc_time, voxel]  ) ) ;
    
    
    ####
    return Subset, beh_Subset



### Example
#s_act, s_beh = condition_wm( sig, beh, condition='2_7', distance='mix')
    



def wm_condition(masked_data, beh_path, n_scans, condition,  distance, sys_use='unix', TR=2.335, nscans_wm=16):
    # Behaviour 
    beh_path = ub_wind_path(beh_path, system=sys_use) #change depending on windoxs/unix
    behaviour=np.genfromtxt(beh_path, skip_header=1) #open the file
    Beh = pd.DataFrame(behaviour)  #convert it to dataframe
    headers_col = ['type', 'delay1', 'delay2', 'T', 'NT1', 'NT2', 'Dist', 'Dist_NT1', 'Dist_NT2', 'distance_T_dist', 'cue', 'order',
                'orient', 'horiz_vertical', 'A_R', 'A_err', 'Abs_angle_error', 'Error_interference', 'A_DC', 'A_DC_dist', 'Q_DC', 
                'A_DF', 'A_DF_dist', 'Q_DF', 'A_DVF', 'Q_DVF', 'A_DVF_dist', 'Q_DVF_dist', 'presentation_att_cue_time', 'presentation_target_time',
                'presentation_dist_time', 'presentation_probe_time', 'R_T', 'trial_time', 'disp_time']
    Beh.columns=headers_col #add columns
    #take off the reference    
    ref_time = Beh.iloc[-1, 1] # get the reference(diff between tsat¡rt the display and start de recording)
    start_trial=Beh['presentation_att_cue_time'].iloc[0:-1]  - ref_time #take off the reference  
    Beh = Beh.iloc[0:-1, :] # behaviour is the same except the last line (reference time) 
    start_trial_hdf_scans = start_trial/TR#transform seconds to scans 
    timestamps = [  int(round(  start_trial_hdf_scans[n] ) ) for n in range(0, len(start_trial_hdf_scans) )]
    
    #adjust according to the number of scans you want (avoid having an incomplete trial)
    while timestamps[-1]> (n_scans-nscans_wm):
        timestamps=timestamps[:-1] #take off one trial form activity
        Beh = Beh.iloc[0:-1, :] #take off one trial from behaviour
    
    
    #append the timestands you want from this session
    n_trials = len(timestamps)
    n_voxels_wm = np.shape(masked_data)[1]
    ### Take the important TRs (from cue, the next 14 TRs)
    run_activity=np.zeros(( n_trials, nscans_wm,  n_voxels_wm   )) ## np.zeros matrix with the correct dimensions of the session
    for idx, t in enumerate(timestamps): #beginning of the trial
        for sc in range(0, nscans_wm): #each of the 14 TRs
            trial_activity = masked_data[t+sc, :]   
            run_activity[idx, sc, :] =trial_activity                    
    
    ### 
    
    Subset, beh_Subset = condition_wm( run_activity, Beh, condition, distance=distance, zscore_=False)
    
    return Subset, beh_Subset





def preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition, distance='mix', sys_use='unix', nscans_wm=16, TR=2.335):
    ### Mask and process the fmri data
    start_process_wm = time.time()
    numcores = multiprocessing.cpu_count()
    wm_masked= Parallel(n_jobs = numcores)(delayed(mask_fmri_process)(fmri_path, masks, sys_use='unix')  for fmri_path in wm_fmri_paths)    ####
    scans_wm_runs = [len(wm_masked[r]) for r in range(len(wm_masked)) ]
    
    ### TRs of interest
    activity_beh  = Parallel(n_jobs = numcores)(delayed(wm_condition)(masked_data, beh_path, n_scans, condition, distance=distance, sys_use='unix', TR=TR, nscans_wm=nscans_wm) for masked_data, beh_path, n_scans in zip( wm_masked, wm_beh_paths, scans_wm_runs))    ####
    runs_signal = [activity_beh[i][0] for i in range(len(activity_beh))]
    runs_beh = [activity_beh[i][1] for i in range(len(activity_beh))]
    
    ## concatenate the runs
    testing_activity = np.vstack(runs_signal)
    testing_behaviour = pd.concat(runs_beh)
    ##
    end_process_wm = time.time()
    process_wm = end_process_wm - start_process_wm
    print( 'Time process wm: ' +str(process_wm))
    return testing_activity, testing_behaviour

###


#def wm_timestamps_targets(masked_data, beh_path, n_scans, sys_use='unix', TR=2.335, nscans_wm=16):
#    # Behaviour 
#    beh_path = ub_wind_path(beh_path, system=sys_use) #change depending on windoxs/unix
#    behaviour=np.genfromtxt(beh_path, skip_header=1) #open the file
#    Beh = pd.DataFrame(behaviour)  #convert it to dataframe
#    headers_col = ['type', 'delay1', 'delay2', 'T', 'NT1', 'NT2', 'Dist', 'Dist_NT1', 'Dist_NT2', 'distance_T_dist', 'cue', 'order',
#                'orient', 'horiz_vertical', 'A_R', 'A_err', 'Abs_angle_error', 'Error_interference', 'A_DC', 'A_DC_dist', 'Q_DC', 
#                'A_DF', 'A_DF_dist', 'Q_DF', 'A_DVF', 'Q_DVF', 'A_DVF_dist', 'Q_DVF_dist', 'presentation_att_cue_time', 'presentation_target_time',
#                'presentation_dist_time', 'presentation_probe_time', 'R_T', 'trial_time', 'disp_time']
#    Beh.columns=headers_col #add columns
#    #take off the reference    
#    ref_time = Beh.iloc[-1, 1] # get the reference(diff between tsat¡rt the display and start de recording)
#    start_trial=Beh['presentation_att_cue_time'].iloc[0:-1]  - ref_time #take off the reference  
#    Beh = Beh.iloc[0:-1, :] # behaviour is the same except the last line (reference time) 
#    start_trial_hdf_scans = start_trial/TR#transform seconds to scans 
#    timestamps = [  int(round(  start_trial_hdf_scans[n] ) ) for n in range(0, len(start_trial_hdf_scans) )]
#    
#    #adjust according to the number of scans you want (avoid having an incomplete trial)
#    while timestamps[-1]> (n_scans-nscans_wm):
#        timestamps=timestamps[:-1] #take off one trial form activity
#        Beh = Beh.iloc[0:-1, :] #take off one trial from behaviour
#    
#    
#    #append the timestands you want from this session
#    n_trials = len(timestamps)
#    n_voxels_wm = np.shape(masked_data)[1]
#    ### Take the important TRs (from cue, the next 14 TRs)
#    run_activity=np.zeros(( n_trials, nscans_wm,  n_voxels_wm   )) ## np.zeros matrix with the correct dimensions of the session
#    for idx, t in enumerate(timestamps): #beginning of the trial
#        for sc in range(0, nscans_wm): #each of the 14 TRs
#            trial_activity = masked_data[t+sc, :]   
#            run_activity[idx, sc, :] =trial_activity                    
#    
#    ### 
#    return run_activity, Beh
#

### Example
#activity_beh  = Parallel(n_jobs = numcores)(delayed(wm_timestamps_targets)(masked_data, beh_path, n_scans, sys_use='unix', TR=2.335, nscans_wm=16) for masked_data, beh_path, n_scans in zip( wm_masked, wm_beh_paths, scans_wm_runs))    ####
#runs_signal = [activity_beh[i][0] for i in range(len(activity_beh))]
#runs_beh = [activity_beh[i][1] for i in range(len(activity_beh))]
    


#testing_activity, testing_behaviour = process_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition='2_7', distance='mix', sys_use='unix', nscans_wm=16, TR=2.335)
#
#def process_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition, distance='mix', sys_use='unix', nscans_wm=16, TR=2.335):
#    ### Mask and process the fmri data
#    start_process_wm = time.time()
#    numcores = multiprocessing.cpu_count()
#    wm_masked= Parallel(n_jobs = numcores)(delayed(mask_fmri_process)(fmri_path, masks, sys_use='unix')  for fmri_path in wm_fmri_paths)    ####
#    scans_wm_runs = [len(wm_masked[r]) for r in range(len(wm_masked)) ]
#    
#    ### TRs of interest
#    activity_beh  = Parallel(n_jobs = numcores)(delayed(wm_timestamps_targets)(masked_data, beh_path, n_scans, sys_use='unix', TR=TR, nscans_wm=nscans_wm) for masked_data, beh_path, n_scans in zip( wm_masked, wm_beh_paths, scans_wm_runs))    ####
#    runs_signal = [activity_beh[i][0] for i in range(len(activity_beh))]
#    runs_beh = [activity_beh[i][1] for i in range(len(activity_beh))]
#    
#    ## concatenate the runs
#    runs_signal = np.vstack(runs_signal)
#    runs_beh = pd.concat(runs_beh)
#    
#    ## get subset of activity
#    testing_activity, testing_behaviour = condition_wm( runs_signal, runs_beh, condition, distance='mix')
#    end_process_wm = time.time()
#    process_wm = end_process_wm - start_process_wm
#    print( 'Time process wm: ' +str(process_wm))
#    return testing_activity, testing_behaviour






