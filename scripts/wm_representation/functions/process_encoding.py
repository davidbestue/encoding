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
from scipy import stats


TR=2.335

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




def enc_timestamps_targets(beh_path, n_scans, sys_use='unix', hd=6, TR=2.335):
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

    


def process_enc_timestamps( masked_data, timestamp_run, TR=2.335):
    n_voxels = np.shape(masked_data)[1]
    ####   2. Apply a filter for each voxel
    for voxel in range(0, n_voxels ):
        data_to_filter = masked_data[:,voxel] #data of the voxel along the session
        #apply the filter 
        data_to_filter = TimeSeries(data_to_filter, sampling_interval=TR)
        F = FilterAnalyzer(data_to_filter, ub=0.15, lb=0.02) ##upper and lower boundaries
        data_filtered=F.filtered_boxcar.data
        masked_data[:,voxel] = data_filtered ## replace old data with the filtered one.
    
    ####   3. Subset of data corresponding to the delay times (all voxels)
    encoding_delay_activity = np.zeros(( len(timestamp_run), n_voxels)) ## emply matrix (n_trials, n_voxels)
    for idx,t in enumerate(timestamp_run): #in each trial
        delay_TRs =  masked_data[t:t+2, :] #take the first scan of the delay and the nex
        delay_TRs_mean =np.mean(delay_TRs, axis=0) #make the mean in each voxel of 2TR
        encoding_delay_activity[idx, :] =delay_TRs_mean #index the line in the matrix
    
    ###   4. zscore + 10 in each voxel in the temporal dimension (with the other 2TR of the same session)
    for voxel in range(0, n_voxels ): # by voxel
        vx_act = encoding_delay_activity[:, voxel]
        vx_act_zs = np.array( stats.zscore(vx_act) ) ; ## zscore + 10 just to get + values
        encoding_delay_activity[:, voxel] = vx_act_zs  ## replace previos activity
    
    
    return encoding_delay_activity



def process_encoding_files(fmri_paths, masks, beh_paths, sys_use='unix', hd=6, TR=2.335):
    ### Inputs: 
    ###### fmri_paths: list of paths
    ###### beh_paths: list of paths
    ###### masks: [rh_mask, lh_mask]
    ###### sys_use (unix or windows: to change the paths)
    ###### hd hemodynamic delay (seconds)
    ###### TR=2.335 (fixed)
    ## Processes:
    ### 1. Load and mask the data of all sessions     
    numcores = multiprocessing.cpu_count()
    all_masked= Parallel(n_jobs = numcores)(delayed(mask_fmri)(fmri_path, masks, sys_use='unix')  for fmri_path in fmri_paths)    ####
    scans_enc_runs = [len(all_masked[r]) for r in range(len(all_masked)) ]
    
    ### 2. delay timestamps and beh targets
    targets_timestamps  = Parallel(n_jobs = numcores)(delayed(enc_timestamps_targets)(beh_path, n_scans, sys_use='unix', hd=hd, TR=TR) for beh_path, n_scans in zip( beh_paths, scans_enc_runs))    ####
    targets= [targets_timestamps[i][0] for i in range(len(targets_timestamps))]
    timestamps = [targets_timestamps[i][1] for i in range(len(targets_timestamps))]
    
    ### 3. combine to get training data & process
    training_dataset  = Parallel(n_jobs = numcores)(delayed(process_enc_timestamps)( masked_data, timestamp_run, TR=TR) for masked_data, timestamp_run in zip( all_masked, timestamps))    ####
    training_dataset = np.vstack(training_dataset)
    training_targets = np.hstack(targets)
    
    return training_dataset, training_targets



###############################################
###############################################
###############################################
    

#root= '/home/david/Desktop/IEM_data/'
#
#masks = [ root+'temp_masks/n001/visual_fsign_rh.nii.gz', root+ 'temp_masks/n001/visual_fsign_lh.nii.gz']
#
#fmri_paths= [root +'n001/encoding/s01/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r04/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r05/nocfmri3_Encoding_Ax.nii',
#             root +'n001/encoding/s02/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r04/nocfmri3_Encoding_Ax.nii',
#             root +'n001/encoding/s03/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r04/nocfmri3_Encoding_Ax.nii']
#
#beh_paths =[root +'n001/encoding/s01/r01/enc_beh.txt', root +'n001/encoding/s01/r02/enc_beh.txt', root +'n001/encoding/s01/r03/enc_beh.txt', root +'n001/encoding/s01/r04/enc_beh.txt', root +'n001/encoding/s01/r05/enc_beh.txt',
#            root +'n001/encoding/s02/r01/enc_beh.txt', root +'n001/encoding/s02/r02/enc_beh.txt', root +'n001/encoding/s02/r03/enc_beh.txt', root +'n001/encoding/s02/r04/enc_beh.txt',
#            root +'n001/encoding/s03/r01/enc_beh.txt', root +'n001/encoding/s03/r02/enc_beh.txt', root +'n001/encoding/s03/r03/enc_beh.txt', root +'n001/encoding/s03/r04/enc_beh.txt']
##
##
##
#
#training_dataset, training_targets = process_encoding_files(fmri_paths, masks, beh_paths, sys_use='unix', hd=6, TR=2.335)







    
    