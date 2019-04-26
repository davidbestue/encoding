# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:24:52 2019

@author: David Bestue
"""

import numpy as np
import pandas as pd
from nilearn.masking import apply_mask


def ub_wind_path(PATH, system):
    if system=='wind':
        A = PATH                                                                                                    
        B = A.replace('/', os.path.sep)                                                                                               
        C= B.replace('\\mnt\\c\\', 'C:\\') 
    
    if system=='unix':
        C=PATH
    
    ###
    return C



def process_encoding_files(fmri_paths, masks, beh_paths, sys_use='unix'):
    ### Inputs: 
    ###### fmri_paths: list of paths
    ###### beh_paths: list of paths
    ###### masks: [rh_mask, lh_mask]
    
    for run in range(len(fmri_paths)):
        func_encoding = fmri_paths[run] #file encoding image
        func_encoding = ub_wind_path(func_encoding, system=sys_use) #change the path format wind-unix
        
        Beh_enc_files = beh_paths[run] #file encoding behaviour
        Beh_enc_files = ub_wind_path(Beh_enc_files, system=sys_use) #change the path format wind-unix
        
        mask_img_rh= masks[0] #right hemisphere mask
        mask_img_rh = ub_wind_path(mask_img_rh, system=sys_use)
        mask_img_lh= masks[1] #left hemisphere mask
        mask_img_lh = ub_wind_path(mask_img_lh, system=sys_use)
        
        #Apply the masks and concatenate   
        masked_data_rh = apply_mask(func_encoding, mask_img_rh)
        masked_data_lh = apply_mask(func_encoding, mask_img_lh)    
        masked_data=np.hstack([masked_data_rh, masked_data_lh])
        #append it and save the data
        encoding_datasets.append(masked_data)
        enc_lens_datas.append(len(masked_data))
        
        
        

    
    ### Imaging encoding
    ##### 1. Imaging
    enc_lens_datas=[]
    encoding_datasets=[]
    
    #####################
    ##################### STEP 1: GET DATA AND APPLY THE MASK
    #####################
    for i in range(0, len(func_encoding)):
        func_filename=func_encoding[i] #+ 'setfmri3_Encoding_Ax.nii' # 'regfmcpr.nii.gz'
        func_filename = ub_wind_path(func_filename, system=sys_use)
        #
        mask_img_rh= path_masks  + Maskrh #maskV1rh_2.nii.gz' #maskV1rh.nii.gz'  maskV1rh_2.nii.gz maskipsrh_2.nii.gz
        mask_img_rh = ub_wind_path(mask_img_rh, system=sys_use)
        #
        mask_img_lh= path_masks + Masklh #maskV1lh_2.nii.gz' #maskV1lh.nii.gz'   maskV1lh_2.nii.gz maskipslh_2.nii.gz
        mask_img_lh = ub_wind_path(mask_img_lh, system=sys_use)
        ##Apply the masks and concatenate   
        masked_data_rh = apply_mask(func_filename, mask_img_rh)
        masked_data_lh = apply_mask(func_filename, mask_img_lh)    
        masked_data=hstack([masked_data_rh, masked_data_lh])
        #append it and save the data
        encoding_datasets.append(masked_data)
        enc_lens_datas.append(len(masked_data))
    
    
    #####################
    ##################### STEP 2: PROCESS TRAINGN DATA
    #####################
    
    #### TRAING DATA: ALL SESSIONS TOGETHER
    
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
    n_voxels = shape(encoding_datasets[0])[1] ## number of voxels
    
    for session_enc_sess in range(0, len(enc_lens_datas)):                  
        
        ### 1. Select the times corresponding to the delay (2TR), append the target of each trial
        Enc_delay=[] ## Get the scans to take from the data (beggining of the delay)
        
        ## load the file
        Beh_enc_files_path = Beh_enc_files[session_enc_sess] ## name of the file
        Beh_enc_files_path = ub_wind_path(Beh_enc_files_path, system=sys_use) ##function to convert paths windows-linux
        behaviour=genfromtxt(Beh_enc_files_path, skip_header=1) ## load the file
        
        
        p_target = array(behaviour[:-1,4]) ## Get the position (hypotetical channel coef)
        
        ### shuffle trial labels
        #v= list( p_target)
        #import random
        #random.shuffle(v)
        #p_target = v
        
        ref_time=behaviour[-1, 1] ## Reference time (start scanner - begiinging of recording)
        st_delay = behaviour[:-1, 11] -ref_time #start of the delay time & take off the reference from
        
        
        hd = 6 # hemodynmic delay  SOURCE OF ERROR!!!!!!!
        start_delay_hdf = st_delay + hd # add some seconds for the hemodynamic delay
        
        #convert seconds to scans (number of scan to take)
        start_delay_hdf_scans = start_delay_hdf/2.335 
        timestamps = [  int(round(  start_delay_hdf_scans[n] ) ) for n in range(0, len(start_delay_hdf_scans) )] #make it an integrer
        #In case  the last one has no space, exclude it (and do the same for the ones of step 1, lin step 3 you will combie and they must have the same length)
        #you short the timestamps and the matrix fro the hipotetical cannel coefici
        while timestamps[-1]>len(encoding_datasets[session_enc_sess])-2:
            timestamps=timestamps[:-1] ##  1st scan to take in each trial
            p_target = p_target[:-1] ## targets of the session (append to the genearl at the end)
        
        
                           
        ####   2. Apply a filter for each voxel               
        for voxel in range(0, n_voxels ):
            data_to_filter = encoding_datasets[session_enc_sess][:,voxel] #data of the voxel along the session
            
            #apply the filter 
            data_to_filter = TimeSeries(data_to_filter, sampling_interval=2.335)
            F = FilterAnalyzer(data_to_filter, ub=0.15, lb=0.02)
            data_filtered=F.filtered_boxcar.data
            encoding_datasets[session_enc_sess][:,voxel] = data_filtered ## replace old data with the filtered one.
        
        
        ####   3. Subset of data corresponding to the delay times (all voxels)
        encoding_delay_activity = zeros(( len(timestamps), n_voxels)) ## emply matrix (n_trials, n_voxels)
        for idx,t in enumerate(timestamps): #in each trial
            delay_TRs =  encoding_datasets[session_enc_sess][t:t+2, :] #take the first scan of the delay and the nex
            delay_TRs_mean = mean(delay_TRs, axis=0) #make the mean in each voxel of 2TR
            encoding_delay_activity[idx, :] =delay_TRs_mean #index the line in the matrix
        
        
        ####   4. zscore + 10 in each voxel in the temporal dimension (with the other 2TR of the same session)
        for vxl in range(0, n_voxels ): # by voxel
            vx_act = encoding_delay_activity[:, vxl]
            vx_act_zs = np.array( zscore(vx_act) ) +10 ; ## zscore + 10 just to get + values
            encoding_delay_activity[:, vxl] = vx_act_zs  ## replace previos activity
        
        
        ####   5. append activity and targets of the session
        p_target = list(p_target) ### make a list that will be added to another list
        Training_dataset_targets.extend(p_target) ## append the position of the target for the trial
        #
        Training_dataset_activity.append(encoding_delay_activity) ## append the activity used for the training
    
    
    ##### Concatenate sessions to create Trianing Dataset  ### ASSUMPTION: each voxel is the same across sessions!               
    Training_dataset_activity = vstack(Training_dataset_activity) #make an array (n_trials(all sessions together), voxels)
    Training_dataset_targets = array(Training_dataset_targets) ## make an array (trials, 1)
    
    