# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:12:03 2018

@author: David Bestue
"""
###Decide brain region


import easygui
import os
import numpy as np

msg = "Decide computer"
choices = ["local", "cluster"]
platform = easygui.buttonbox(msg, choices=choices)

if platform == "local":
    root_use ='/mnt/c/Users/David/Desktop/KI_Desktop/IEM_data/'
    encoding_path = 'C:\\Users\\David\\Dropbox\\KAROLINSKA\\encoding_model\\'
    Conditions_enc_path = 'C:\\Users\\David\\Dropbox\\KAROLINSKA\\encoding_model\\Conditions\\'
    PLOTS_path = '\\plots'
    sys_use='wind'
    
elif platform == "cluster":
    root_use ='/home/david/Desktop/IEM_data/'
    encoding_path = '/home/david/Desktop/KAROLINSKA/encoding_model/'
    Conditions_enc_path = '/home/david/Desktop/KAROLINSKA/encoding_model/Conditions/'
    PLOTS_path = '/plots'
    sys_use='unix'
    
    
##Methods_analysis=[]
##
for SUBJECT_USE_ANALYSIS in ['n001']: #'d001', 'n001', 'r001', 'b001', 'l001', 's001'
    print(SUBJECT_USE_ANALYSIS)
    for brain_region in ["visual"]:  #"ips"
        for CONDITION in ['1_0.2']: 
            Method_analysis = 'together'
            #CONDITION = '1_0.2'
            #brain_region = "visual"
            distance_ch='mix'
            #distance='mix'
            Subject_analysis=SUBJECT_USE_ANALYSIS
            os.chdir(encoding_path)
            ############################################       
            from functions_encoding_loop import *
            Method_analysis, distance_ch, Subject_analysis, brain_region, distance, func_encoding_sess, Beh_enc_files_sess, func_wmtask_sess, Beh_WM_files_sess, path_masks, Maskrh, Masklh, writer_matrix = variables_encoding(Method_analysis, distance_ch, Subject_analysis, brain_region, root_use ) 
            #############################################
            df_responses=[]
            dfs = {}
            
            for session_enc in range(0,len(func_encoding_sess)):
                print(session_enc)
                func_encoding = func_encoding_sess[session_enc] 
                Beh_enc_files = Beh_enc_files_sess[session_enc]
                #
                func_wmtask =func_wmtask_sess[session_enc]
                Beh_WM_files = Beh_WM_files_sess[session_enc]
                
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
                
                ###############################
                ###############################  STEP 3: TRAIN THE MODEL
                ###############################
                #For each voxel, I want to extract weight of each channel of our model
                #Con qué peso de canales explico mejor la actividad de este voxel a a lo largo de los trials))
                #Right now I will combine the two previous steps
                #Que canal explica mejor la activid de este voxel a lo largo de todos los trials? --> Weight para cada canal
                #If I have a voxel that responds to 27, the weight of the first channel is going to be hight because it means that the activity I have fits really weel with the activity I 
                # expect from the first channel
                
                ####   1. Generate hypothetical activity per trial
                ####   2. Train the model and get matrix of weights
                ####   
                
                ####   1. Generate hypothetical activity per trial                
                M_model=[] #matrix of the activity from the model
                for i in Training_dataset_targets:
                    channel_values=f(i)  #f #f_quadrant (function that generates the expectd reponse in each channel)
                    M_model.append(channel_values)
                    
                M_model=pd.DataFrame(array(M_model)) # (trials, channel_activity)
                channel_names = ['ch_' +str(i+1) for i in range(0, len(pos_channels))] #names of the channels 
                M_model.columns=channel_names
                
                
                ####   2. Train the model and get matrix of weights
                Matrix_weights=zeros(( n_voxels, len(pos_channels) )) # (voxels, channels) how each channels is represented in each voxel
                
                for voxel_x in range(0, n_voxels): #train each voxel
                    # set Y and X for the GLM
                    Y = Training_dataset_activity[:, voxel_x] ## Y is the real activity
                    X = M_model ## X is the hipothetycal activity 
                    
                    #### Lasso with penalization of 0.0001 (higher gives all zeros), fiting intercept (around 10 ) and forcing the weights to be positive
                    lin = Lasso(alpha=0.001, precompute=True,  fit_intercept=True,
                                max_iter=1000,  positive=True, random_state=9999, 
                                selection='random')   
                    lin.fit(X,Y) # fits the best combination of weights to explain the activity
                    betas = lin.coef_ #ignore the intercept and just get the weights of each channel
                    Matrix_weights[voxel_x, :]=betas #save the 36 weights for each voxel
                
                
                #####


                
                
                
                